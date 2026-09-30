"""
RQ2: closed-loop OOD evaluation grid (plan §5 Week 4, Fig A / Fig F).

Arms:
  none              uncorrected VLA proxy
  gt_shadow         Phase-0-style GT shadow-rollout correction (slow engine in loop)
  checkvla_vision   CheckVLA with the vision world-model predictor (baseline, §3.7)
  checkvla_orbisim  CheckVLA with OrbiSim-Dynamics (full pipeline)
  [checkvla_vision_noact  with --with_noact]

Metrics per cell: SR, CVR, RR, intervention rate, control latency.

Usage: python experiments/rq2_eval.py --task push
"""
import os

import numpy as np

from _common import base_parser, env_for_cell, load_policy, load_predictor, setup
from registry import build_verifier
from checkvla.verifier import Controller, run_episodes, summarize
from common import grid_cells, is_ood, load_json, save_json


def build_controller(arm, policy, sim, od, dev, taus, args):
    kw = dict(chunk_k=args.chunk_k)
    if arm == "none":
        return Controller(policy, sim, "none", **kw)
    if arm == "gt_shadow":
        return Controller(policy, sim, "gt_shadow", commit=args.commit, **kw)
    role = arm.replace("checkvla_", "")
    ver = build_verifier(args, load_predictor(args, od, role, dev, sim), sim, taus[role])
    return Controller(policy, sim, "checkvla", verifier=ver, **kw)


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--with_noact", action="store_true")
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--chunk_k", type=int, default=5,
                   help="steps between policy calls (open-loop chunk execution)")
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_envs = 8
    policy = load_policy(args, od, dev, sim)
    taus = load_json(os.path.join(od, "taus.json"))["tau"]
    arms = args.arms or (["none", "gt_shadow", "checkvla_vision", "checkvla_orbisim"]
                         + (["checkvla_vision_noact"] if args.with_noact else []))
    ctrls = {a: build_controller(a, policy, sim, od, dev, taus, args) for a in arms}

    cells = grid_cells(args.task)
    # the most stressful OOD cell (by uncorrected CVR) is traced for Fig F
    results, traces = [], {}
    for ci, cell in enumerate(cells):
        env, params = env_for_cell(args, args.n_envs, dev, cell)
        entry = {"cell": cell, "ood": is_ood(args.task, cell)}
        for a, c in ctrls.items():
            res = run_episodes(env, sim, c, params, seed=3000 + ci,
                               record=a.startswith("checkvla"),
                               audit_repair=a != "none")
            entry[a] = summarize(res)
            if "trace" in res:
                vi = int(np.argmax(res["violation"])) if res["violation"].any() else 0
                traces.setdefault(str(ci), {})[a] = {k: v[vi] for k, v in res["trace"].items()}
        results.append(entry)
        print(f"[rq2] {cell} " + " | ".join(
            f"{a}: SR {entry[a]['SR']:.2f} CVR {entry[a]['CVR']:.2f} "
            f"safe {entry[a]['safe_success']:.2f} {entry[a]['latency_ms_mean']:.1f}ms"
            for a in arms))

    # pooled over OOD cells
    pooled = {}
    for a in arms:
        ood = [r[a] for r in results if r["ood"]] or [r[a] for r in results]
        # union, not ood[0]: arms carry optional keys (the repair audit appears
        # only where the arm actually intervened), so one arm's key set is not
        # a valid template for another's cells
        keys = {k for x in ood for k in x}
        pooled[a] = {m: float(np.nanmean([x.get(m, np.nan) for x in ood])) for m in keys}
        # intervention counts add, they do not average
        if "n_interventions" in keys:
            pooled[a]["n_interventions"] = int(sum(x.get("n_interventions", 0) for x in ood))
    base = pooled.get("none", {}).get("CVR", np.nan)
    for a in arms:
        pooled[a]["CVR_reduction_vs_none"] = float(1 - pooled[a]["CVR"] / base) if base > 0 else float("nan")
    worst = int(np.argmax([r["none"]["CVR"] if "none" in r else 0 for r in results]))
    save_json({"arms": arms, "per_cell": results, "pooled_ood": pooled,
               "fig_f_cell": results[worst]["cell"]}, os.path.join(od, "rq2.json"))
    if str(worst) in traces:
        np.savez(os.path.join(od, "rq2_trace.npz"),
                 **{f"{a}__{k}": v for a, d in traces[str(worst)].items() for k, v in d.items()},
                 tau_orbisim=taus.get("orbisim", 1.0), tau_vision=taus.get("vision", 1.0))
    print("[rq2] pooled OOD:", {a: {m: round(v[m], 3) for m in
                                    ("SR", "CVR", "safe_success", "CVR_reduction_vs_none")}
                                for a, v in pooled.items()})
    print("[rq2] repair audit - GT risk of the chunk actually executed:")
    for a in arms:
        v = pooled[a]
        if np.isfinite(v.get("gt_risk_rel_drop", np.nan)):
            print(f"       {a:22s} GT risk {v['gt_risk_proposed']:.3f} -> "
                  f"{v['gt_risk_applied']:.3f} ({100 * v['gt_risk_rel_drop']:+.1f}%)  "
                  f"cleared {v['frac_cleared']:.2f}  no-op {v['frac_ineffective']:.2f}  "
                  f"mag x{v['mag_ratio_mean']:.2f}  abst {v.get('frac_abstain', float('nan')):.2f}"
                  f"  n={v['n_interventions']}", flush=True)


if __name__ == "__main__":
    main()
