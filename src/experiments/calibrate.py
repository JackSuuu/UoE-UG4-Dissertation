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

from _common import base_parser, env_for_cell, load_policy, load_verifiers, setup
from _verif import verifier_rollout
from common import PREDICTOR_TRAIN_RANGES, load_json, save_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n_cells", type=int, default=6)
    p.add_argument("--n_envs", type=int, default=64)
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_cells, args.n_envs = 2, 8
    policy = load_policy(args, od, dev, sim)
    preds = load_verifiers(args, od, dev, sim)          # tau=inf while calibrating
    rng = np.random.default_rng(args.seed + 1)
    ranges = PREDICTOR_TRAIN_RANGES[args.task]
    scores = {k: [] for k in preds}
    for c in range(args.n_cells):
        cell = {k: float(rng.uniform(*v)) for k, v in ranges.items()}
        env, params = env_for_cell(args, args.n_envs, dev, cell)
        r = verifier_rollout(env, sim, policy, params, preds, seed=1000 + c, act_noise=0.1)
        safe = r["gt_chunk_risk"] <= 1.0
        for k in preds:
            scores[k].append(r["scores"][k][safe])
        print(f"[calib] cell {c} {cell} safe-chunk frac {safe.mean():.3f}")
    from checkvla.reference import calibrate_tau
    taus = {k: calibrate_tau(np.concatenate(v), args.alpha) for k, v in scores.items()}
    print("[calib] tau:", taus)
    path = os.path.join(od, "taus.json")
    old = load_json(path)["tau"] if os.path.exists(path) else {}
    old.update(taus)                                     # keep taus of roles not re-run
    save_json({"alpha": args.alpha, "tau": old,
               "impl": {"orbisim": args.orbisim_impl, "vision": args.vision_impl,
                        "verifier": args.verifier, "policy": args.policy}}, path)


if __name__ == "__main__":
    main()
