"""
RQ3 — systems contribution (plan §3.5, Figs C/D/E).

  --part mem      Peak GPU memory & wall-clock vs horizon for
                  naive BPTT | checkpointing (GPU) | checkpointing + CPU offload |
                  + truncation window; gradient cosine vs naive (quality check).
                  With --backend genesis also records naive Genesis BPTT memory.
  --part stab     Adaptive gradient truncation in a chaotic-contact cell:
                  per-step grad-norm trace, grad-norm variance over perturbations,
                  and downstream trajectory-optimisation quality
                  (none | clip | relax).
  --part sched    Fast/slow dual-engine scheduling: control-loop latency and
                  verdict staleness for fast-only | async (fast + background GT) |
                  sync (GT in the hot loop, = gt_shadow).
  --part hybrid   CPU/GPU hybrid inference: arithmetic-intensity-aware task
                  scheduling, fused operator dispatch, and CUDA Graph
                  capture/replay benchmark (KTransformers-inspired).
  --part deferral Expert deferral pipeline overlap: immediate vs deferred
                  gradient segments, overlap speedup, and gradient staleness.

Usage: python experiments/rq3_systems.py --task push --part all
"""
import os
import time

import numpy as np
import torch

from _common import base_parser, load_policy, load_predictor, setup
from common import load_json, save_json
from systems.bptt import Stabilizer, checkpointed_bptt, measure, naive_bptt, step_cost
from systems.hybrid_engine import (
    ArithmeticIntensityScheduler,
    CUDAGraphBPTT,
    FusedStepOperator,
    HardwareBudget,
)
from systems.expert_deferral import DeferredSegment, ExpertDeferralScheduler

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


# ---------------------------------------------------------------------------
def part_hybrid(args, dev, sim, od):
    """
    CPU/GPU hybrid inference experiments (KTransformers-inspired).

    1. Arithmetic-intensity-aware scheduling: profile BPTT recompute tasks
       and decide CPU vs GPU assignment based on FLOPs/byte ratio.
    2. Fused operator benchmark: compare fused step_cost + contact
       computation vs unfused (separate kernel launches).
    3. CUDA Graph capture/replay: measure speedup from eliminating
       per-kernel launch overhead in the BPTT inner loop.
    """
    from systems.hybrid_engine import TaskProfile

    print("[rq3/hybrid] === Arithmetic Intensity Scheduling ===")
    budget = HardwareBudget(
        gpu_memory_gb=24.0,
        max_latency_ms=50.0,
        cpu_cores=8,
        ai_threshold_flops_per_byte=5.0,
    )
    scheduler = ArithmeticIntensityScheduler(budget, dev)

    # Profile tasks at different batch sizes and horizons
    tasks = []
    for B in [1, 8, 32, 128, 512]:
        for T in [10, 50, 100]:
            task = scheduler.profile_task(f"B{B}_T{T}", B, T,
                                          sim.state_dim, sim.act_dim)
            tasks.append(task)
            print(f"  {task.name:12s} AI={task.arithmetic_intensity:8.1f} "
                  f"FLOPs={task.total_flops:.2e} bytes={task.total_bytes:.2e}")

    schedule = scheduler.schedule(tasks)
    print(f"  Schedule: {len(schedule.gpu_tasks)} GPU tasks, "
          f"{len(schedule.cpu_tasks)} CPU tasks, "
          f"est. speedup={schedule.estimated_speedup:.2f}x "
          f"bottleneck={schedule.bottleneck}")

    # --- Fused operator benchmark ------------------------------------------
    print("[rq3/hybrid] === Fused Operator Benchmark ===")
    B = 64
    T = 50
    params = sim.make_params(B, {})
    s0 = sim.init_state(B, torch.Generator().manual_seed(0))
    A = expert_actions(sim, params, s0, T)

    fused = FusedStepOperator(sim)

    # Unfused: separate step + cost
    def unfused_fn():
        s = s0.clone()
        total = torch.zeros(B, device=dev)
        for t in range(T):
            s, r = sim.step(s, A[:, t], params)
            total = total + step_cost(sim, s, r)
        return total.sum()

    # Fused: merged step + cost
    def fused_fn():
        s = s0.clone()
        total = torch.zeros(B, device=dev)
        for t in range(T):
            _, _, cost = fused.fused_forward(s, A[:, t], params)
            total = total + cost
        return total.sum()

    _, _, unfused_time = measure(unfused_fn, dev)
    _, _, fused_time = measure(fused_fn, dev)
    print(f"  Unfused: {unfused_time:.3f}s  Fused: {fused_time:.3f}s  "
          f"Speedup: {unfused_time / max(fused_time, 1e-9):.2f}x")

    # --- CUDA Graph benchmark ----------------------------------------------
    print("[rq3/hybrid] === CUDA Graph Capture/Replay ===")
    cudagraph_result = {"available": False}
    if torch.cuda.is_available():
        try:
            seg = 10
            A_seg = A[:, :seg].contiguous()
            graph = CUDAGraphBPTT(sim, params, s0, A_seg, seg=seg)
            graph.capture()
            bench = graph.benchmark(n_replays=50)
            cudagraph_result = {
                "available": True,
                "graph_ms": bench["graph_ms"],
                "eager_ms": bench["eager_ms"],
                "speedup": bench["speedup"],
                "n_replays": bench["n_replays"],
            }
            print(f"  CUDA Graph: {bench['graph_ms']:.3f}ms vs "
                  f"Eager: {bench['eager_ms']:.3f}ms  "
                  f"Speedup: {bench['speedup']:.2f}x")
        except Exception as e:
            cudagraph_result = {"available": False, "error": str(e)[:200]}
            print(f"  CUDA Graph failed: {e}")
    else:
        print("  CUDA Graph: not available (no CUDA)")

    return {
        "arithmetic_intensity": {
            "tasks": [{"name": t.name, "ai": t.arithmetic_intensity,
                       "total_flops": t.total_flops, "total_bytes": t.total_bytes}
                      for t in tasks],
            "schedule": {
                "n_gpu": len(schedule.gpu_tasks),
                "n_cpu": len(schedule.cpu_tasks),
                "estimated_speedup": schedule.estimated_speedup,
                "bottleneck": schedule.bottleneck,
            },
        },
        "fused_operator": {
            "unfused_time_s": unfused_time,
            "fused_time_s": fused_time,
            "speedup": unfused_time / max(fused_time, 1e-9),
        },
        "cudagraph": cudagraph_result,
    }


# ---------------------------------------------------------------------------
def part_deferral(args, dev, sim, od):
    """
    Expert deferral pipeline overlap experiments.

    Simulates a control loop where BPTT segments are classified as
    immediate (needed now) or deferred (computed in background).
    Measures overlap speedup and gradient staleness.
    """
    print("[rq3/deferral] === Expert Deferral Pipeline Overlap ===")
    T = 80 if sim.name == "push" else 40
    B = 16
    cell = STRESS[sim.name]
    params = sim.make_params(B, cell)
    s0 = sim.init_state(B, torch.Generator().manual_seed(1))
    A0 = expert_actions(sim, params, s0, T)

    for depth in [1, 2, 4]:
        print(f"\n  --- Deferral depth = {depth} ---")
        sched = ExpertDeferralScheduler(deferral_depth=depth, max_deferred=4)

        # Simulate control loop
        immediate_times = []
        deferred_times = []
        n_deferred_used = 0

        for t in range(0, T, 10):
            seg_start = t
            seg_end = min(T, t + 10)
            seg_id = t // 10

            # Classify
            cls = sched.classify_segment(seg_id, t)
            if cls == "immediate":
                # Compute synchronously with smooth=True for differentiability
                t0 = time.perf_counter()
                s_seg = s0.clone().detach().requires_grad_(True)
                a_seg = A0[:, seg_start:seg_end].clone().detach().requires_grad_(True)
                for i in range(seg_end - seg_start):
                    s_seg, _ = sim.step(s_seg, a_seg[:, i], params, smooth=True)
                loss = sim.task_cost(s_seg).sum()
                loss.backward()
                immediate_times.append(time.perf_counter() - t0)
                sched.stats.n_immediate += 1
            else:
                # Submit deferred
                seg = DeferredSegment(
                    segment_id=seg_id,
                    start_step=seg_start,
                    end_step=seg_end,
                    state_at_start=s0.clone(),
                    actions=A0[:, seg_start:seg_end].clone(),
                )

                def compute_fn(state, actions):
                    s = state.clone().detach().requires_grad_(True)
                    a = actions.clone().detach().requires_grad_(True)
                    for i in range(actions.shape[1]):
                        s, _ = sim.step(s, a[:, i], params, smooth=True)
                    loss = sim.task_cost(s).sum()
                    grad_s = torch.autograd.grad(loss, a, retain_graph=False)[0]
                    return loss, grad_s

                t0 = time.perf_counter()
                sched.submit_deferred(seg, compute_fn)
                deferred_times.append(time.perf_counter() - t0)

            # Collect completed deferred gradients
            needed = sched.collect_deferred(t)
            n_deferred_used += len(needed)

        # Wait for background thread
        if sched._background_thread and sched._background_thread.is_alive():
            sched._background_thread.join(timeout=5.0)

        # Compute speedup
        if immediate_times and deferred_times:
            avg_immediate = np.mean(immediate_times)
            avg_deferred = np.mean(deferred_times)
            speedup = sched.compute_overlap_speedup(
                avg_immediate, avg_deferred, len(deferred_times))
            print(f"  Immediate: {avg_immediate*1e3:.1f}ms  "
                  f"Deferred: {avg_deferred*1e3:.1f}ms  "
                  f"Overlap speedup: {speedup:.2f}x")
            print(f"  Deferred used: {n_deferred_used}  "
                  f"Staleness: {sched.stats.gradient_staleness}")

    return {
        "deferral_depths": [1, 2, 4],
        "note": "See console output for per-depth results",
    }


def main():
    p = base_parser(__doc__)
    p.add_argument("--part", default="all",
                   choices=["all", "mem", "stab", "sched", "hybrid", "deferral"])
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
    for part, fn in (("mem", part_mem), ("stab", part_stab),
                     ("sched", part_sched), ("hybrid", part_hybrid),
                     ("deferral", part_deferral)):
        if args.part in ("all", part):
            out[part] = fn(args, dev, sim, od)
            save_json(out, path)


if __name__ == "__main__":
    main()
