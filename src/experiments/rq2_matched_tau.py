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

``--rates`` are **intervention rates, not thresholds**, and the thresholds are
found by measurement rather than assumed. Passing the targets straight in as tau
values is a category error: the calibrated taus are 0.597 (orbisim) and 0.457
(vision), so a target of 0.02 sits far below every chunk score and the trigger
saturates -- a 4x change in tau moved the measured rate by 5% (0.179 -> 0.169),
so the sweep measured nothing. Round 0 takes a quantile of the predictor's own
score distribution on the OOD cells. That undershoots too, because intervening
damps the actions and lowers every later score: the measured rate came out at
20-25% of the requested one (target 0.12 -> 0.027 for orbisim). Round 1 runs with
the repair on and the reported thresholds invert a power law fitted to round 1's
*measured* rates, which converges in one step; the re-pooled quantile is printed
as a cross-check. The interpolation still uses measured rates, and each
predictor's own calibrated tau is *measured* as an extra point rather than
interpolated to, so it is on the same footing as every other row -- a useful
check, since orbisim's calibrated tau comes out at rate 0.0192, matching the
0.019 recorded by the independent RQ2 run.

Usage:

Usage:
    python experiments/rq2_matched_tau.py --task push --backend torch --chunk_k 5
"""
import os

import numpy as np
import torch

from _common import (base_parser, env_for_cell, load_policy, load_predictor,
                     load_verifiers, setup)
from _verif import summary_keys, verifier_rollout
from checkvla.verifier import Controller, run_episodes, summarize
from common import grid_cells, is_ood, load_json, save_json
from registry import build_verifier

# Target intervention rates, per control step. The per-predictor calibrated rate
# is 0.019 (orbisim) and 0.052 (vision), so this brackets both and puts each
# predictor on both sides of the other's operating point.
TARGET_RATES = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]


def build_controller(arm, policy, sim, od, dev, taus, args, tau=None):
    if arm == "none":
        return Controller(policy, sim, "none", chunk_k=args.chunk_k)
    role = arm.replace("checkvla_", "")
    ver = build_verifier(args, load_predictor(args, od, role, dev, sim),
                         sim, taus[role] if tau is None else tau)
    return Controller(policy, sim, "checkvla", verifier=ver, chunk_k=args.chunk_k)


def collect_scores_noop(args, dev, sim, od, policy, roles, cells, envs, params):
    """One uncorrected pass over the OOD cells, pooling every chunk score.

    Round 0 of the threshold search. Sampling the cells matters -- a quantile
    taken in-distribution would be applied to an OOD score distribution and land
    nowhere near the requested rate.
    """
    preds = load_verifiers(args, od, dev, sim)          # tau=inf, no repair
    pooled = {r: [] for r in roles}
    for ci in range(len(cells)):
        r = verifier_rollout(envs[ci], sim, policy, params[ci], preds,
                             seed=3000 + ci, chunk_k=args.chunk_k)
        for role in roles:
            pooled[role].append(np.asarray(r["scores"][role]).ravel())
    return {r: np.concatenate(v) for r, v in pooled.items()}


def sweep_at(args, dev, sim, od, policy, arm, calib_taus, cells, points, envs,
             params, record_scores=False):
    """Run every cell at every sweep point, returning per-(cell, point) summaries.

    ``env``/``params`` are built once per cell and reused across points:
    resetting is the only thing that differs, and ``run_episodes`` resets
    internally. ``points`` is a list of ``(label, target_rate_or_None, tau)``,
    where ``target_rate`` is ``None`` for a point that is measured rather than
    aimed at (the calibrated operating point).

    ``record_scores`` returns the proposed-chunk scores observed *on the repaired
    trajectory*, which is what round 1 of the threshold search needs: intervening
    changes the trajectory, and therefore the score distribution the next
    round's quantile has to be taken from.
    """
    out, score_pool = [], []
    for label, target, tau in points:
        c = build_controller(arm, policy, sim, od, dev, calib_taus, args, tau=tau)
        per_cell = []
        for ci, cell in enumerate(cells):
            res = run_episodes(envs[ci], sim, c, params[ci], seed=3000 + ci,
                               audit_repair=True, record=record_scores)
            per_cell.append(summarize(res))
            if record_scores:
                score_pool.append(res["trace"]["score"].ravel())
        pooled = {k: float(np.nanmean([x.get(k, np.nan) for x in per_cell]))
                  for k in summary_keys(per_cell)}
        pooled["n_interventions"] = int(sum(x.get("n_interventions", 0) for x in per_cell))
        out.append({"label": label, "target_rate": target, "tau": float(tau),
                    "pooled": pooled, "per_cell": per_cell})
        p = pooled
        print(f"  [{arm:18s}] {label:>10s}  tau {tau:6.3f}  "
              f"rate {p['intervention_rate']:.4f}  "
              f"CVR {p['CVR']:.3f}  SR {p['SR']:.3f}  safe {p['safe_success']:.3f}  "
              f"n={p['n_interventions']}", flush=True)
    return out, (np.concatenate(score_pool) if score_pool else None)


def solve_taus(taus, measured_rates, targets, tol=1e-9):
    """Threshold per target rate, by inverting a power law fitted to measurements.

    Re-quantileing the score pool is the obvious estimator and it is the wrong
    one here: the pool it reads was collected at a *different* tau, so it
    converges only as slowly as the pool refreshes (two rounds left a factor-3
    shortfall: target 0.12 -> measured 0.027 -> 0.044). The quantity actually
    needing control is the measured rate, and the measured rate is already
    available at every point tried, so fitting it directly uses strictly more
    information for no extra rollouts.

    The score pools are heavy-tailed (orbisim p99/median = 11), which is what
    makes rate-vs-tau close to a straight line in log-log over the range that
    matters. The fit is ``log r = intercept + slope * log tau`` with a negative
    slope, and the thresholds come from *inverting* it -- returning the fitted
    rates instead is the obvious transcription slip and silently turns every
    target into a rate above 1.

    Returns ``(taus, fit, extrapolated)``, where ``extrapolated`` flags targets
    whose solution falls outside the tau range the fit was measured over, so a
    row can never look measured when it is a projection. A flat fit (rate barely
    moving with tau, which is what a saturated trigger or a dead predictor looks
    like) carries no information to invert, so it degrades to the median tau with
    everything flagged rather than dividing by a slope that is nearly zero.
    """
    taus = np.asarray(taus, dtype=float)
    n = len(targets)
    lx = np.log(taus)
    ly = np.log(np.maximum(np.asarray(measured_rates, dtype=float), 1e-4))
    slope, intercept = np.polyfit(lx, ly, 1)
    if not np.isfinite(slope) or slope >= -0.1:
        return (np.full(n, float(np.median(taus))),
                (float(intercept), float(slope)), np.ones(n, dtype=bool))
    ltau = (np.log(np.asarray(targets, dtype=float)) - intercept) / slope
    out = np.exp(np.clip(ltau, np.log(1e-6), np.log(1e3)))
    out = np.clip(out, float(taus.min()) * 1e-3, float(taus.max()) * 10.0)
    extrapolated = (out < taus.min() - tol) | (out > taus.max() + tol)
    return out, (float(intercept), float(slope)), extrapolated


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
        args.n_envs, args.rates = 8, [0.02, 0.05, 0.12]
    policy = load_policy(args, od, dev, sim)
    taus = load_json(os.path.join(od, "taus.json"))["tau"]

    cells = [c for c in grid_cells(args.task) if is_ood(args.task, c)]
    envs, params = [], []
    for cell in cells:
        e, pr = env_for_cell(args, args.n_envs, dev, cell)
        envs.append(e); params.append(pr)

    roles = sorted({a.replace("checkvla_", "") for a in args.arms})
    calib = {a: taus[a.replace("checkvla_", "")] for a in args.arms}

    # Round 0: thresholds from the *uncorrected* score distribution.
    pool0 = collect_scores_noop(args, dev, sim, od, policy, roles, cells, envs, params)
    qs = np.asarray(args.rates)
    tau_r0 = {r: np.quantile(pool0[r], 1.0 - qs) for r in roles}
    for r in roles:
        q = np.quantile(pool0[r], [0.5, 0.9, 0.99])
        print(f"[matched] {r:12s} uncorrected pool n={pool0[r].size}  "
              f"median {q[0]:.3f}  p90 {q[1]:.3f}  p99 {q[2]:.3f}  "
              f"-> tau {np.round(tau_r0[r], 3).tolist()}", flush=True)

    # Round 1 runs with the repair on. A single quantile from the uncorrected
    # pool undershoots badly -- intervening damps the actions, which lowers every
    # later score, so the measured rate came out at 20-25% of the requested one
    # (target 0.12 -> 0.027 for orbisim). The reported thresholds come from
    # inverting a power law fitted to round 1's *measured* rates, which
    # converges in one step; the re-pooled quantile is kept as a cross-check.
    r1, pool1, tau_r1, fits = {}, {}, {}, {}
    for a in args.arms:
        role = a.replace("checkvla_", "")
        pts = [(f"r1@{t:.3f}", float(t), float(tau_r0[role][i]))
               for i, t in enumerate(qs)]
        r1[a], pool1[a] = sweep_at(args, dev, sim, od, policy, a, taus, cells, pts,
                                   envs, params, record_scores=True)
        meas = [p["pooled"]["intervention_rate"] for p in r1[a]]
        tau_r1[role], fits[role], extrap = solve_taus(tau_r0[role], meas, qs)
        quant = np.quantile(pool1[a], 1.0 - qs)
        print(f"[matched] {role:12s} measured {np.round(meas, 4).tolist()}  "
              f"fit r = {np.exp(fits[role][0]):.2e} * tau^{fits[role][1]:.3f}  "
              f"-> tau {np.round(tau_r1[role], 3).tolist()}"
              f"{'  EXTRAP ' + str(extrap.tolist()) if extrap.any() else ''}  "
              f"(repooled quantile {np.round(quant, 3).tolist()})", flush=True)

    # Round 2 is the reported curve, and it also *measures* each predictor at its
    # own calibrated tau rather than interpolating to it, so the reference point
    # is on the same footing as every other row.
    results = {}
    for a in args.arms:
        role = a.replace("checkvla_", "")
        pts = [(f"{t:.3f}", float(t), float(tau_r1[role][i]))
               for i, t in enumerate(qs)]
        pts.append(("calib", None, float(calib[a])))
        results[a], _ = sweep_at(args, dev, sim, od, policy, a, taus, cells, pts,
                                 envs, params)

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
        pt = next(p for p in c if p["target_rate"] is None)
        p = pt["pooled"]
        rows.append({"arm": a, "target_rate": None, "calibrated": True,
                     "tau": pt["tau"], "intervention_rate": p["intervention_rate"],
                     "CVR": p["CVR"], "SR": p["SR"], "safe_success": p["safe_success"]})
    save_json({"arms": args.arms, "cells": cells, "calibrated_tau": calib,
               "score_pool_n": {r: int(pool0[r].size) for r in roles},
               "tau_round0": {r: tau_r0[r].tolist() for r in roles},
               "tau_reported": {r: tau_r1[r].tolist() for r in roles},
               "rate_tau_fit": {r: fits[r] for r in roles},
               "round1": r1, "sweep": results, "matched": rows,
               "target_rates": args.rates}, os.path.join(od, "rq2_matched_tau.json"))

    print("\n[matched] at a common intervention rate (interpolated):")
    print(f"{'rate':>7}  " + "  ".join(f"{a.replace('checkvla_', ''):>22}" for a in args.arms))
    for target in args.rates:
        cells_out = []
        for a in args.arms:
            r = next((x for x in rows if x["arm"] == a
                      and x["target_rate"] == target), None)
            v = r.get("CVR") if r else None
            cells_out.append("  CVR --  safe --   " if v is None or v != v
                             else f"  CVR {v:.3f} safe {r['safe_success']:.3f}")
        print(f"{target:7.3f}  " + "  ".join(f"{c:>22}" for c in cells_out))
    print("\n[matched] each predictor at its own calibrated tau (*):")
    for a in args.arms:
        r = next(x for x in rows if x["arm"] == a and x["calibrated"])
        print(f"  {a:22s} tau {r.get('tau', float('nan')):.3f}  "
              f"rate {r.get('intervention_rate', float('nan')):.4f}  "
              f"CVR {r.get('CVR', float('nan')):.3f}  safe {r.get('safe_success', float('nan')):.3f}")


if __name__ == "__main__":
    main()
