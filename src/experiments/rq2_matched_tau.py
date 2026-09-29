"""Matched-trigger comparison: is the gap calibration, or the verifier?

run_all_v5 / drive.log gave a result that contradicts the obvious reading.
The repair audit says, per intervention:

    gt_shadow          GT risk 2.289 -> 0.741  (+63.2%)  no-op 0.01  n= 665
    checkvla_vision    GT risk 0.691 -> 0.408  (+41.4%)  no-op 0.24  n=3721
    checkvla_orbisim   GT risk 0.650 -> 0.392  (+28.9%)  no-op 0.49  n=1394

So vision is the *better* repairer per intervention -- it lowers the true risk
of the chunk it executes by more than orbisim does -- and yet it buys 0.4% CVR
reduction against orbisim's 24.6%. The difference is how often they fire:
vision intervenes 3721 times to orbisim's 1394, and it fires at recall 0.934
with an 8.8-step lead, i.e. on chunks that were never going to violate.

That makes the comparison confounded. A verifier that wins partly by having a
looser threshold is not a result; it is a tuned constant. This experiment
removes the confound by sweeping tau for each predictor and comparing at
*matched intervention rate* rather than at each predictor's own calibrated tau.

The calibrated tau is included as a reference point, marked ``*``.

Usage:
    python experiments/rq2_matched_tau.py --task push --backend torch --chunk_k 5
"""
import os

import numpy as np
import torch

from _common import base_parser, env_for_cell, load_policy, load_predictor, setup
from _verif import summary_keys
from checkvla.verifier import Controller, run_episodes, summarize
from common import grid_cells, is_ood, load_json, save_json
from registry import build_verifier

# Rate-matched targets. The per-predictor calibrated rate is 0.019 (orbisim) and
# 0.052 (vision) per control step; the sweep brackets both so each predictor is
# measured on both sides of the other's operating point.
TARGET_RATES = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]


def build_controller(arm, policy, sim, od, dev, taus, args, tau=None):
    if arm == "none":
        return Controller(policy, sim, "none", chunk_k=args.chunk_k)
    role = arm.replace("checkvla_", "")
    ver = build_verifier(args, load_predictor(args, od, role, dev, sim),
                         sim, taus[role] if tau is None else tau)
    return Controller(policy, sim, "checkvla", verifier=ver, chunk_k=args.chunk_k)


def sweep_at(args, dev, sim, od, policy, arm, taus, cells, rates, envs, params):
    """Run every cell at every tau, returning per-(cell, tau) summaries.

    ``env``/``params`` are built once per cell and reused across taus: resetting
    is the only thing that differs, and ``run_episodes`` resets internally.
    """
    out = []
    for tau in rates:
        c = build_controller(arm, policy, sim, od, dev, taus, args, tau=tau)
        per_cell = []
        for ci, cell in enumerate(cells):
            res = run_episodes(envs[ci], sim, c, params[ci], seed=3000 + ci,
                               audit_repair=True)
            per_cell.append(summarize(res))
        pooled = {k: float(np.nanmean([x.get(k, np.nan) for x in per_cell]))
                  for k in summary_keys(per_cell)}
        pooled["n_interventions"] = int(sum(x.get("n_interventions", 0) for x in per_cell))
        out.append({"tau": tau, "pooled": pooled, "per_cell": per_cell})
        p = pooled
        print(f"  [{arm:18s}] tau {tau:6.3f}  rate {p['intervention_rate']:.4f}  "
              f"CVR {p['CVR']:.3f}  SR {p['SR']:.3f}  safe {p['safe_success']:.3f}  "
              f"n={p['n_interventions']}", flush=True)
    return out


def at_rate(curve, target):
    """Linear interpolation of the (rate, metric) curve at a target rate.

    Returns None when the sweep does not bracket the target, so a caller can
    never silently report a value extrapolated past the measured range.
    """
    rates = np.array([c["pooled"]["intervention_rate"] for c in curve])
    order = np.argsort(rates)
    rates = rates[order]
    keys = [k for k in curve[0]["pooled"] if k != "n_interventions"]
    vals = {k: np.array([c["pooled"][k] for c in curve])[order] for k in keys}
    if target < rates.min() - 1e-9 or target > rates.max() + 1e-9:
        return None
    return {k: float(np.interp(target, rates, vals[k])) for k in keys}


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--chunk_k", type=int, default=5)
    p.add_argument("--arms", nargs="+", default=["checkvla_orbisim", "checkvla_vision"])
    p.add_argument("--rates", type=float, nargs="+", default=TARGET_RATES)
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_envs, args.rates = 8, [0.02, 0.08]
    policy = load_policy(args, od, dev, sim)
    taus = load_json(os.path.join(od, "taus.json"))["tau"]

    cells = [c for c in grid_cells(args.task) if is_ood(args.task, c)]
    envs, params = [], []
    for cell in cells:
        e, pr = env_for_cell(args, args.n_envs, dev, cell)
        envs.append(e); params.append(pr)

    results = {a: sweep_at(args, dev, sim, od, policy, a, taus, cells,
                           args.rates, envs, params) for a in args.arms}

    # where each predictor's own calibrated tau lands on its own curve
    calib = {a: taus[a.replace("checkvla_", "")] for a in args.arms}
    rows = []
    for a in args.arms:
        c = results[a]
        for target in args.rates:
            row = {"arm": a, "target_rate": target, "calibrated": False}
            v = at_rate(c, target)
            if v is not None:
                row.update({k: v[k] for k in
                            ("intervention_rate", "CVR", "SR", "safe_success",
                             "gt_risk_rel_drop", "frac_ineffective", "mag_ratio_mean")})
            rows.append(row)
        # the calibrated operating point, for reference
        row = {"arm": a, "target_rate": None, "calibrated": True, "tau": calib[a]}
        v = at_rate(c, calib[a])
        if v is not None:
            row.update({k: v[k] for k in ("intervention_rate", "CVR", "SR", "safe_success")})
        rows.append(row)
    save_json({"arms": args.arms, "cells": cells, "taus": taus,
               "calibrated_tau": calib, "sweep": results, "matched": rows,
               "target_rates": args.rates}, os.path.join(od, "rq2_matched_tau.json"))

    print("\n[matched] at a common intervention rate (interpolated):")
    print(f"{'rate':>7}  " + "  ".join(f"{a.replace('checkvla_', ''):>22}" for a in args.arms))
    for target in args.rates:
        cells_out = []
        for a in args.arms:
            r = next((x for x in rows if x["arm"] == a
                      and x["target_rate"] == target), None)
            cells_out.append("  CVR --  safe --   " if r is None or r["CVR"] != r["CVR"]
                             else f"  CVR {r['CVR']:.3f} safe {r['safe_success']:.3f}")
        print(f"{target:7.3f}  " + "  ".join(f"{c:>22}" for c in cells_out))
    print("\n[matched] each predictor at its own calibrated tau (*):")
    for a in args.arms:
        r = next(x for x in rows if x["arm"] == a and x["calibrated"])
        print(f"  {a:22s} tau {r.get('tau', float('nan')):.3f}  "
              f"rate {r.get('intervention_rate', float('nan')):.4f}  "
              f"CVR {r.get('CVR', float('nan')):.3f}  safe {r.get('safe_success', float('nan')):.3f}")


if __name__ == "__main__":
    main()
