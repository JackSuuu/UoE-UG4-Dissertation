"""
RQ3 — systems contribution (plan §3.5, Figs C/D/E).

  --part mem    Peak GPU memory & wall-clock vs horizon for
                naive BPTT | checkpointing (GPU) | checkpointing + CPU offload |
                + truncation window; gradient cosine vs naive (quality check).
                With --backend genesis also records naive Genesis BPTT memory.
  --part stab   Adaptive gradient truncation in a chaotic-contact cell:
                per-step grad-norm trace, grad-norm variance over perturbations,
                and downstream trajectory-optimisation quality
                (none | clip | relax).
  --part sched  Fast/slow dual-engine scheduling: control-loop latency and
                verdict staleness for fast-only | async (fast + background GT) |
                sync (GT in the hot loop, = gt_shadow).

Usage: python experiments/rq3_systems.py --task push --part all
"""
import os

import numpy as np
import torch

from _common import base_parser, load_policy, load_predictor, setup
from common import load_json, save_json
from systems.bptt import Stabilizer, checkpointed_bptt, measure, naive_bptt, step_cost

STRESS = {"push": {"friction": 0.2, "mass": 2.0}, "cloth": {"stiffness": 4.0, "mass": 2.0}}


@torch.no_grad()
def expert_actions(sim, params, s0, T):
    s, A = s0, []
    for _ in range(T):
        a = sim.expert(s)
        A.append(a)
        s, _ = sim.step(s, a, params)
    return torch.stack(A, 1)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


# ---------------------------------------------------------------------------
def part_mem(args, dev, sim, od):
    horizons = args.horizons or ([25, 50, 100, 200, 400] if sim.name == "push" else [10, 25, 50, 100, 200])
    if args.quick:
        horizons = horizons[:2]
    B = args.mem_envs or (1024 if sim.name == "push" else 16)
    if args.quick:
        B = 8
    params = sim.make_params(B, {})
    s0 = sim.init_state(B, torch.Generator().manual_seed(0))
    modes = {
        "naive": lambda A: naive_bptt(sim, params, s0, A),
        "ckpt_gpu": lambda A: checkpointed_bptt(sim, params, s0, A, seg=args.seg, offload=False),
        "ckpt_offload": lambda A: checkpointed_bptt(sim, params, s0, A, seg=args.seg, offload=True),
        "ckpt_offload_trunc": lambda A: checkpointed_bptt(sim, params, s0, A, seg=args.seg,
                                                          offload=True, trunc_window=args.trunc),
    }
    rows = []
    for T in horizons:
        A = expert_actions(sim, params, s0, T)
        ref = None
        for name, fn in modes.items():
            row = {"horizon": T, "mode": name}
            try:
                (loss, g, _), mem, sec = measure(lambda: fn(A), dev)
                row.update(peak_mem_mb=mem, seconds=sec, loss=float(loss.mean()))
                if name == "naive":
                    ref = g
                row["grad_cos_vs_naive"] = cos(g, ref) if ref is not None else float("nan")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row.update(peak_mem_mb=float("nan"), seconds=float("nan"), oom=True)
            rows.append(row)
            print(f"[rq3/mem] T={T:4d} {name:20s} mem={row.get('peak_mem_mb', float('nan')):9.1f}MB "
                  f"t={row.get('seconds', float('nan')):7.2f}s cos={row.get('grad_cos_vs_naive', float('nan')):.4f}")
    out = {"rows": rows, "seg": args.seg, "trunc": args.trunc, "n_envs": B}
    if args.backend == "genesis" and sim.name == "push":
        from sims.genesis_push import genesis_grad_probe
        out["genesis_naive"] = []
        for T in horizons:
            r = genesis_grad_probe(sim, {}, horizon=T * 5)
            r["horizon"] = T
            out["genesis_naive"].append(r)
            print(f"[rq3/mem] genesis naive T={T}: {r['status']} mem={r['peak_mem_mb']}MB {r['msg'][:80]}")
    return out


# ---------------------------------------------------------------------------
def part_stab(args, dev, sim, od):
    T = 80 if sim.name == "push" else 40
    K = 4 if args.quick else args.n_perturb
    iters = 3 if args.quick else args.opt_iters
    cell = STRESS[sim.name]
    B = 8
    params = sim.make_params(B, cell)
    s0 = sim.init_state(B, torch.Generator().manual_seed(1))
    A0 = expert_actions(sim, params, s0, T)
    g = torch.Generator().manual_seed(2)
    modes = {"none": None, "clip": "clip", "relax": "relax"}
    out = {"cell": cell, "T": T, "modes": {}}
    for name, m in modes.items():
        stab = (lambda: Stabilizer(mode=m, rho=args.rho)) if m else (lambda: None)
        norms, traces = [], None
        for k in range(K):
            A = (A0 + 0.1 * sim.max_vel * torch.randn(A0.shape, generator=g).to(dev)).clamp(-sim.max_vel, sim.max_vel)
            _, ga, info = checkpointed_bptt(sim, params, s0, A, seg=args.seg, stabilizer=stab())
            norms.append(float(ga.norm()))
            if traces is None:
                traces = {"raw": info["raw_norms"], "used": info["used_norms"], "n_spikes": info["n_spikes"]}
        # downstream: gradient-based trajectory optimisation, evaluated on the GT
        A = A0.clone()
        opt_curve = []
        for it in range(iters):
            loss, ga, _ = checkpointed_bptt(sim, params, s0, A, seg=args.seg, stabilizer=stab())
            opt_curve.append(float(loss.mean()))
            ga = torch.nan_to_num(ga)
            A = (A - args.opt_lr * sim.max_vel * ga / (ga.flatten(1).norm(dim=1)[:, None, None] + 1e-8)
                 ).clamp(-sim.max_vel, sim.max_vel)
        with torch.no_grad():
            s, tot, viol = s0, 0, torch.zeros(B, dtype=torch.bool, device=dev)
            for t in range(T):
                s, r = sim.step(s, A[:, t], params)
                tot = tot + step_cost(sim, s, r)
                viol |= r.amax(1) > 1
        ln = np.log(np.array(norms) + 1e-12)
        out["modes"][name] = {
            "grad_norms": norms, "log_grad_norm_var": float(ln.var()),
            "trace_raw": traces["raw"], "trace_used": traces["used"], "n_spikes": traces["n_spikes"],
            "opt_curve": opt_curve, "final_cost": float(tot.mean()),
            "final_success": float(sim.success(s).float().mean()), "final_CVR": float(viol.float().mean()),
        }
        print(f"[rq3/stab] {name:6s} log-norm var={ln.var():.3f} spikes={traces['n_spikes']} "
              f"final cost={float(tot.mean()):.4f} CVR={float(viol.float().mean()):.2f}")
    return out


# ---------------------------------------------------------------------------
def part_sched(args, dev, sim, od):
    from checkvla.verifier import Controller, run_episodes, summarize
    from systems.scheduler import AsyncEngine
    from registry import build_verifier
    policy = load_policy(args, od, dev, sim)
    pred = load_predictor(args, od, "orbisim", dev, sim)
    tau = load_json(os.path.join(od, "taus.json"))["tau"]["orbisim"]
    ver = build_verifier(args, pred, sim, tau)
    cell = STRESS[sim.name]
    n = 8 if args.quick else args.n_envs
    from _common import env_for_cell
    env, params = env_for_cell(args, n, dev, cell)
    out = {"cell": cell, "slow_delay_ms": args.slow_delay_ms, "modes": {}}
    eng = AsyncEngine(args.task, args.backend, n, cell, params, dev, args.slow_delay_ms)
    ctrls = {
        "fast_only": Controller(policy, sim, "checkvla", verifier=ver),
        "async": Controller(policy, sim, "checkvla", verifier=ver, async_engine=eng),
        "sync": Controller(policy, sim, "gt_shadow"),
    }
    try:
        for name, c in ctrls.items():
            eng.reset_stats()
            res = run_episodes(env, sim, c, params, seed=4000)
            s = summarize(res)
            s["latency_ms_all"] = res["latency_ms"]
            if name == "async":
                s["staleness_mean"] = float(np.mean(eng.staleness)) if eng.staleness else float("nan")
                s["staleness_p95"] = float(np.percentile(eng.staleness, 95)) if eng.staleness else float("nan")
                s["slow_engine_ms_mean"] = float(np.mean(eng.slow_ms)) if eng.slow_ms else float("nan")
                s["n_dropped"] = eng.n_dropped
            out["modes"][name] = s
            print(f"[rq3/sched] {name:9s} latency mean={s['latency_ms_mean']:.2f}ms "
                  f"p95={s['latency_ms_p95']:.2f}ms SR={s['SR']:.2f} CVR={s['CVR']:.2f}"
                  + (f" staleness={s.get('staleness_mean', float('nan')):.2f}" if name == "async" else ""))
    finally:
        eng.close()
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--part", default="all", choices=["all", "mem", "stab", "sched"])
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--mem_envs", type=int, default=None,
                   help="batch size for --part mem (default push 1024, cloth 16)")
    p.add_argument("--seg", type=int, default=10)
    p.add_argument("--trunc", type=int, default=25)
    p.add_argument("--rho", type=float, default=5.0)
    p.add_argument("--n_perturb", type=int, default=16)
    p.add_argument("--opt_iters", type=int, default=30)
    p.add_argument("--opt_lr", type=float, default=0.05)
    p.add_argument("--slow_delay_ms", type=float, default=0.0)
    args = p.parse_args()
    dev, sim, od = setup(args)
    path = os.path.join(od, "rq3.json")
    out = load_json(path) if os.path.exists(path) else {}
    for part, fn in (("mem", part_mem), ("stab", part_stab), ("sched", part_sched)):
        if args.part in ("all", part):
            out[part] = fn(args, dev, sim, od)
            save_json(out, path)


if __name__ == "__main__":
    main()
