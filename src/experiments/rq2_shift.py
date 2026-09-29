"""
RQ2b: verifier performance as a function of *distribution shift* (plan §5, Fig G).

The OOD grid in ``rq2_eval`` asks "does the verifier help at a fixed set of
perturbed cells". That conflates two things: how far the cell is from the
predictor's training range, and whether the verifier helps at all. This
experiment makes the shift magnitude the independent variable, which is the
question a practitioner actually faces: *how much physics shift can this
verifier absorb before it stops being useful, and what happens past that?*

One parameter is varied at a time along each OOD axis, the other is held at
nominal, and the ladder spans the training range and continues past it. For
every rung we report the uncorrected baseline, the GT-verifier upper bound and
each learned predictor, so the answer is a curve rather than a single point.

``shift`` is reported as the (signed) log-distance from the training range:
0 while the cell is inside the range the predictor was trained and calibrated
on, and growing as it leaves. Everything at shift = 0 is by construction
in-distribution; the interesting behaviour is on the right of that line.

Usage: python experiments/rq2_shift.py --task push --backend torch --chunk_k 5
"""
import os

import numpy as np

from _common import base_parser, env_for_cell, load_policy, load_predictor, setup
from registry import build_verifier
from checkvla.verifier import Controller, run_episodes, summarize
from common import PREDICTOR_TRAIN_RANGES, load_json, save_json

# Ladders. Each is (axis, value); the other OOD axis stays at nominal 1.0.
# Values below 1.0 mean *less* friction / *lighter* box, which is the direction
# that actually breaks the task: a slippery box slides into the wall, a heavy
# one resists the pusher and jams.
LADDERS = {
    "push": {
        "friction": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0],
        "mass": [0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5],
    },
    "cloth": {
        "stiffness": [0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0],
        "mass": [0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5],
    },
}


def shift_of(task: str, cell: dict) -> float:
    """Signed log-distance of the cell from the predictor's training range.

    0 if every parameter is inside its training range, otherwise the largest
    log-ratio by which any single parameter sits outside it.
    """
    rng = PREDICTOR_TRAIN_RANGES[task]
    worst = 0.0
    for k, v in cell.items():
        lo, hi = rng.get(k, (0.0, float("inf")))
        if v < lo:
            worst = max(worst, float(np.log(lo / v)))
        elif v > hi:
            worst = max(worst, float(np.log(v / hi)))
    return worst


def _reduction(cvr: float, base: float) -> str:
    """CVR reduction against the uncorrected arm, as a signed percentage.

    A cell the baseline already solves has nothing to reduce, so we say so
    rather than divide by zero.
    """
    if base <= 1e-6:
        return "(-)" if cvr > 1e-6 else "(=0)"
    return f"({100 * (1 - cvr / base):+.0f}%)"


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
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--chunk_k", type=int, default=5,
                   help="steps between policy calls (open-loop chunk execution)")
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_envs = 8
    policy = load_policy(args, od, dev, sim)
    taus = load_json(os.path.join(od, "taus.json"))["tau"]
    arms = args.arms or ["none", "gt_shadow", "checkvla_orbisim", "checkvla_vision"]
    ctrls = {a: build_controller(a, policy, sim, od, dev, taus, args) for a in arms}

    ladders = LADDERS[args.task]
    results = []
    for axis, values in ladders.items():
        # hold the *other* OOD axis at nominal so that one parameter moves at a
        # time. Derived from the ladder keys, not hard-coded: getting this wrong
        # silently collapses {"mass": v, "mass": 1.0} to nominal and the whole
        # second ladder silently re-evaluates the same cell.
        other = [k for k in ladders if k != axis]
        assert not other or len(other) == 1, f"{args.task} ladder must have 2 axes"
        for ci, v in enumerate(values):
            cell = {axis: v}
            if other:
                cell[other[0]] = 1.0
            env, params = env_for_cell(args, args.n_envs, dev, cell)
            entry = {"axis": axis, "value": v, "cell": cell,
                     "shift": shift_of(args.task, cell)}
            for a, c in ctrls.items():
                entry[a] = summarize(run_episodes(env, sim, c, params, seed=4000 + ci))
            results.append(entry)
            base = entry["none"]["CVR"]
            print(f"[shift] {axis}={v:<4} shift {entry['shift']:.2f}  " + " | ".join(
                f"{a} safe {entry[a]['safe_success']:.2f} CVR {entry[a]['CVR']:.2f} "
                f"{_reduction(entry[a]['CVR'], base)}"
                for a in arms), flush=True)

    # The headline: safe success vs shift, pooled over both axes, bucketed so
    # that rungs inside and outside the training range can be compared.
    buckets = {"in-dist": (0.0, 1e-9), "near": (1e-9, 0.2), "far": (0.2, 1e9)}
    summary = {}
    for name, (lo, hi) in buckets.items():
        rows = [r for r in results if lo <= r["shift"] < hi]
        if not rows:
            continue
        summary[name] = {a: {
            "safe_success": float(np.mean([r[a]["safe_success"] for r in rows])),
            "CVR": float(np.mean([r[a]["CVR"] for r in rows])),
            "SR": float(np.mean([r[a]["SR"] for r in rows])),
            "intervention_rate": float(np.mean([r[a]["intervention_rate"] for r in rows])),
        } for a in arms}
        summary[name]["n_cells"] = len(rows)
    save_json({"arms": arms, "ladders": ladders, "results": results,
               "by_shift": summary, "train_ranges": PREDICTOR_TRAIN_RANGES[args.task],
               "taus": taus},
              os.path.join(od, "rq2_shift.json"))
    print("\n[shift] safe success by distance from the training range:")
    for name, v in summary.items():
        print(f"  {name:12s} n={v['n_cells']:2d}  " + "  ".join(
            f"{a} {v[a]['safe_success']:.2f}" for a in arms))


if __name__ == "__main__":
    main()
