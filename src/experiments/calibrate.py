"""
Calibrate CheckVLA trigger thresholds (split-conformal, plan §3.3/§3.7).

For each predictor, roll out the uncorrected policy (with small execution noise)
on random IN-DISTRIBUTION cells, keep the scores of chunks whose GT shadow
rollout is safe, and set tau so that P(score > tau | safe) <= alpha.

Usage: python experiments/calibrate.py --task push
"""
import os

import numpy as np
import torch

from _common import (train_ranges, base_parser, calibration_path, env_for_cell,
                     experiment_metadata, load_policy, load_verifiers, setup)
from _verif import verifier_rollout
from common import PREDICTOR_TRAIN_RANGES, load_json, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n_cells", type=int, default=6)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--chunk_k", type=int, default=5,
                    help="steps between policy calls (open-loop chunk execution)")
    p.add_argument("--act_noise", type=float, default=0.1,
                   help="execution-noise fraction; set 0 for deployment-matched calibration")
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_cells, args.n_envs = 2, 8
    policy = load_policy(args, od, dev, sim)
    preds = load_verifiers(args, od, dev, sim)          # tau=inf while calibrating
    rng = np.random.default_rng(args.seed + 1)
    ranges = train_ranges(args)
    scores = {k: [] for k in preds}
    cells = []
    for c in range(args.n_cells):
        cell = {k: float(rng.uniform(*v)) for k, v in ranges.items()}
        cells.append(cell)
        env, params = env_for_cell(args, args.n_envs, dev, cell)
        r = verifier_rollout(env, sim, policy, params, preds, seed=args.seed + 1000 + c,
                              act_noise=args.act_noise,
                              chunk_k=args.chunk_k)
        safe = (r["gt_chunk_risk"] <= 1.0) & r["decision_mask"]
        for k in preds:
            scores[k].append(r["scores"][k][safe])
        print(f"[calib] cell {c} {cell} safe decision-chunk frac "
              f"{safe.sum() / max(r['decision_mask'].sum(), 1):.3f}", flush=True)
    from checkvla.reference import calibrate_tau
    pooled = {k: np.concatenate(v) for k, v in scores.items()}
    for k, values in pooled.items():
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{k}: calibration needs nonempty finite safe decision scores")
    taus = {k: calibrate_tau(v, args.alpha) for k, v in pooled.items()}
    print("[calib] tau:", taus)
    path = calibration_path(args, od)
    old = load_json(path)["tau"] if os.path.exists(path) else {}
    old.update(taus)                                     # keep taus of roles not re-run
    save_json({"alpha": args.alpha, "tau": old,
               "sample_unit": "safe_policy_decision_chunk",
               "n_safe_chunks": {k: len(v) for k, v in pooled.items()},
               "cells": cells,
               "metadata": experiment_metadata(args, od, policy),
               "impl": {"orbisim": args.orbisim_impl, "vision": args.vision_impl,
                         "verifier": args.verifier, "policy": args.policy}}, path)
    print(f"[calib] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
