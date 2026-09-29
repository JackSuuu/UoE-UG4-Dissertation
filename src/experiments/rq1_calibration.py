"""
RQ1 (narrowed): inside the CheckVLA trigger, does OrbiSim-Dynamics give better
calibrated risk signals / earlier repair timing than a vision world model?

For every OOD cell the *uncorrected* policy is rolled out in the GT simulator;
each predictor scores every proposed chunk and is compared against the GT
shadow-rollout label. Reports per predictor (pooled, per cell, and split by
the gradient-valid / gradient-invalid audit label):
  precision, recall, false-alarm rate, AUROC, timely recall, lead time,
plus OrbiSim fidelity-vs-horizon and gradient agreement (cosine of dJ/dchunk
between OrbiSim and the differentiable GT) restricted to gradient-valid states.

Usage: python experiments/rq1_calibration.py --task push
"""
import os

import numpy as np

from _common import CHUNK_H, base_parser, env_for_cell, load_policy, load_verifiers, setup
from _verif import auroc, controllability, trigger_metrics, verifier_rollout
from common import grid_cells, is_ood, load_json, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--grad_every", type=int, default=5)
    p.add_argument("--chunk_k", type=int, default=5,
                   help="steps between policy calls (open-loop chunk execution)")
    p.add_argument("--ctrl_scales", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0],
                   help="chunk scales used to measure repair controllability")
    p.add_argument("--ctrl_every", type=int, default=5,
                   help="measure controllability every Nth step (a GT shadow "
                        "rollout costs ~5 steps, so probing every step is unaffordable)")
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_envs, args.grad_every = 8, 20
    policy = load_policy(args, od, dev, sim)
    taus = load_json(os.path.join(od, "taus.json"))["tau"]
    preds = load_verifiers(args, od, dev, sim, taus)
    ge = args.grad_every if args.backend == "torch" else 0

    per_cell, pooled = [], {k: {"s": [], "g": [], "r": []} for k in preds}
    ctrl = {}
    split = {k: {"valid": ([], []), "invalid": ([], [])} for k in preds}
    fid, cos_valid, cos_by_regime = [], [], {}
    trace = None
    for ci, cell in enumerate(grid_cells(args.task)):
        env, params = env_for_cell(args, args.n_envs, dev, cell)
        r = verifier_rollout(env, sim, policy, params, preds, seed=2000 + ci, grad_every=ge,
                             chunk_k=args.chunk_k, ctrl_scales=tuple(args.ctrl_scales),
                             ctrl_every=args.ctrl_every)
        entry = {"cell": cell, "ood": is_ood(args.task, cell),
                 "policy_SR": float(r["success"].mean()),
                 "policy_CVR": float((r["step_risk"] > 1).any(1).mean())}
        for k in preds:
            entry[k] = trigger_metrics(r["scores"][k], r["gt_chunk_risk"], r["step_risk"],
                                       taus[k], CHUNK_H)
            entry[k]["risk_mae"] = float(np.abs(np.minimum(r["pred_max_risk"][k], 3)
                                                - np.minimum(r["gt_chunk_risk"], 3)).mean())
            pooled[k]["s"].append(r["scores"][k])
            pooled[k]["g"].append(r["gt_chunk_risk"])
            pooled[k]["r"].append(r["step_risk"])
            ctrl.setdefault(k, {"p": [], "g": []})
            # only chunks the trigger would act on: controllability of a chunk
            # that is never repaired is irrelevant. ctrl_pred/ctrl_gt are
            # subsampled to every ctrl_every'th step, so index them with the
            # same mask rather than the full-length scores.
            sc_all = r["scores"][k]
            act = sc_all[:, ::args.ctrl_every] > taus[k]
            ctrl[k]["p"].append(r["ctrl_pred"][k][act])
            ctrl[k]["g"].append(r["ctrl_gt"][act])
        if "probe" in r:
            pr = r["probe"]
            t_idx = pr["t"]
            valid = pr["valid"]
            lab = r["gt_chunk_risk"][:, t_idx] > 1
            for k in preds:
                sc = r["scores"][k][:, t_idx]
                split[k]["valid"][0].append(sc[valid]); split[k]["valid"][1].append(lab[valid])
                split[k]["invalid"][0].append(sc[~valid]); split[k]["invalid"][1].append(lab[~valid])
            fid.append(pr["fidelity"][valid])                         # (n_valid, H)
            cos_valid.append(pr["cos"][valid])
            for reg_i, name in enumerate(sim.regime_names):
                m = valid & (pr["regime"] == reg_i)
                cos_by_regime.setdefault(name, []).append(pr["cos"][m])
            entry["grad_valid_frac"] = float(valid.mean())
            entry["grad_cos_mean"] = float(pr["cos"][valid].mean()) if valid.any() else float("nan")
        if trace is None or (r["step_risk"] > 1).any():
            # keep one violating episode trace for Fig F
            vi = int(np.argmax((r["step_risk"] > 1).any(1))) if (r["step_risk"] > 1).any() else 0
            if trace is None or not trace.get("violating", False):
                trace = {"cell": cell, "violating": bool((r["step_risk"][vi] > 1).any()),
                         "step_risk": r["step_risk"][vi], "gt_chunk_risk": r["gt_chunk_risk"][vi],
                         **{f"score_{k}": r["scores"][k][vi] for k in preds}}
        per_cell.append(entry)
        print(f"[rq1] {cell} CVR={entry['policy_CVR']:.2f} | " + " | ".join(
            f"{k}: AUROC {entry[k]['auroc']:.2f} timely {entry[k]['timely_recall']:.2f}"
            for k in preds))

    summary = {}
    for k in preds:
        S = np.concatenate(pooled[k]["s"]); G = np.concatenate(pooled[k]["g"])
        Rk = np.concatenate(pooled[k]["r"])
        summary[k] = trigger_metrics(S, G, Rk, taus[k], CHUNK_H)
        for part in ("valid", "invalid"):
            sc, lb = split[k][part]
            if sc:
                sc, lb = np.concatenate(sc), np.concatenate(lb)
                summary[k][f"auroc_grad_{part}"] = auroc(sc, lb)
                summary[k][f"n_grad_{part}"] = int(len(sc))
        if ctrl.get(k):
            summary[k].update(controllability(np.concatenate(ctrl[k]["p"]),
                                              np.concatenate(ctrl[k]["g"])))
    out = {"tau": taus, "summary": summary, "per_cell": per_cell,
           "ctrl_scales": list(args.ctrl_scales), "ctrl_every": args.ctrl_every,
           "ctrl_curve": {"scales": list(args.ctrl_scales),
                          **{k: {"pred": np.mean(np.concatenate(ctrl[k]["p"]), 0).tolist(),
                                 "gt": np.mean(np.concatenate(ctrl[k]["g"]), 0).tolist()}
                             for k in preds if ctrl.get(k)}}}
    if fid:
        F_ = np.concatenate(fid)
        C = np.concatenate(cos_valid)
        out["fidelity_vs_horizon"] = {"mean": F_.mean(0), "std": F_.std(0)}
        out["grad_agreement"] = {"cos_mean": float(C.mean()), "cos_median": float(np.median(C)),
                                 "n": int(len(C)),
                                 "by_regime": {k: float(np.concatenate(v).mean()) if sum(len(x) for x in v) else float("nan")
                                               for k, v in cos_by_regime.items()}}
    save_json(out, os.path.join(od, "rq1.json"))
    # raw controllability samples, so the Fig G response curve can be
    # regenerated without paying for another 40-minute rollout
    if ctrl:
        np.savez(os.path.join(od, "rq1_ctrl.npz"),
                 scales=np.array(args.ctrl_scales),
                 **{f"pred_{k}": np.concatenate(ctrl[k]["p"]) for k in ctrl},
                 **{f"gt_{k}": np.concatenate(ctrl[k]["g"]) for k in ctrl})
    np.savez(os.path.join(od, "rq1_trace.npz"), **{k: np.asarray(v) for k, v in trace.items()
                                                    if k not in ("cell",)})
    print("[rq1] pooled:", {k: {m: round(v[m], 3) for m in ("auroc", "precision", "recall",
                                                              "timely_recall")} for k, v in summary.items()})
    print("[rq1] repair controllability (chunks the trigger acts on):")
    for k, v in summary.items():
        if "pred_monotone" in v:
            print(f"       {k:14s} predicted score descends on {v['pred_monotone']:.2f} of the "
                  f"ladder, Spearman {v['pred_spearman']:+.2f} "
                  f"| GT {v['gt_monotone']:.2f}, Spearman {v['gt_spearman']:+.2f} "
                  f"(n={v['n']})", flush=True)


if __name__ == "__main__":
    main()
