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
from _verif import auroc, trigger_metrics, verifier_rollout
from common import grid_cells, is_ood, load_json, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--grad_every", type=int, default=5)
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_envs, args.grad_every = 8, 20
    policy = load_policy(args, od, dev, sim)
    taus = load_json(os.path.join(od, "taus.json"))["tau"]
    preds = load_verifiers(args, od, dev, sim, taus)
    ge = args.grad_every if args.backend == "torch" else 0

    per_cell, pooled = [], {k: {"s": [], "g": [], "r": []} for k in preds}
    split = {k: {"valid": ([], []), "invalid": ([], [])} for k in preds}
    fid, cos_valid, cos_by_regime = [], [], {}
    trace = None
    for ci, cell in enumerate(grid_cells(args.task)):
        env, params = env_for_cell(args, args.n_envs, dev, cell)
        r = verifier_rollout(env, sim, policy, params, preds, seed=2000 + ci, grad_every=ge)
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
    out = {"tau": taus, "summary": summary, "per_cell": per_cell}
    if fid:
        F_ = np.concatenate(fid)
        C = np.concatenate(cos_valid)
        out["fidelity_vs_horizon"] = {"mean": F_.mean(0), "std": F_.std(0)}
        out["grad_agreement"] = {"cos_mean": float(C.mean()), "cos_median": float(np.median(C)),
                                 "n": int(len(C)),
                                 "by_regime": {k: float(np.concatenate(v).mean()) if sum(len(x) for x in v) else float("nan")
                                               for k, v in cos_by_regime.items()}}
    save_json(out, os.path.join(od, "rq1.json"))
    np.savez(os.path.join(od, "rq1_trace.npz"), **{k: np.asarray(v) for k, v in trace.items()
                                                    if k not in ("cell",)})
    print("[rq1] pooled:", {k: {m: round(v[m], 3) for m in ("auroc", "precision", "recall",
                                                              "timely_recall")} for k, v in summary.items()})


if __name__ == "__main__":
    main()
