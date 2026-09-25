"""
Week 1-2: collect ground-truth data.

* bc_data.pt   — clean expert demos at NOMINAL physics (trains the VLA proxy).
* dyn_data.pt  — diverse rollouts over PREDICTOR_TRAIN_RANGES with expert-speed
                 gain and temporally-correlated action noise, so that the
                 verifiers see both safe and violating transitions.
                 (Trains OrbiSim-Dynamics and the vision world model.)

Usage: python experiments/collect_data.py --task push [--backend genesis]
"""
import os

import numpy as np
import torch

from _common import base_parser, setup, env_for_cell
from common import PREDICTOR_TRAIN_RANGES


@torch.no_grad()
def collect(env, sim, params, T, gain, sigma, seed):
    g = torch.Generator().manual_seed(seed + 12345)
    env.reset(params, seed)
    S, A, R = [env.get_state().clone()], [], []
    n = S[0].shape[0]
    dev = S[0].device
    noise = torch.zeros(n, sim.act_dim, device=dev)
    for _ in range(T):
        a = sim.expert(S[-1], gain=gain)
        eps = torch.randn(n, sim.act_dim, generator=g).to(dev)
        noise = 0.8 * noise + 0.6 * sigma[:, None] * eps
        a = (a + noise).clamp(-sim.max_vel, sim.max_vel)
        _, r, _ = env.step(a)
        S.append(env.get_state().clone()); A.append(a); R.append(r)
    return (torch.stack(S, 1).cpu(), torch.stack(A, 1).cpu(), torch.stack(R, 1).cpu())


def main():
    p = base_parser(__doc__)
    p.add_argument("--n_bc", type=int, default=512)
    p.add_argument("--n_dyn", type=int, default=2048)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.n_bc, args.n_dyn, args.batch = 32, 64, 32
    T = sim.T
    rng = np.random.default_rng(args.seed)

    # ---------------- BC demos at nominal ----------------
    env, params = env_for_cell(args, args.batch, dev, {})
    out = []
    for b in range(0, args.n_bc, args.batch):
        zero = torch.zeros(args.batch, device=dev)
        out.append(collect(env, sim, params, T, None, zero, seed=args.seed * 1000 + b))
    S, A, R = (torch.cat(x) for x in zip(*out))
    torch.save({"states": S, "actions": A, "risk": R}, os.path.join(od, "bc_data.pt"))
    print(f"[bc] {S.shape[0]} eps, expert viol rate at nominal: "
          f"{(R.amax((1, 2)) > 1).float().mean():.3f}")

    # ---------------- predictor data over a param range ----------------
    ranges = PREDICTOR_TRAIN_RANGES[args.task]
    out, pars = [], []
    for b in range(0, args.n_dyn, args.batch):
        n = args.batch
        if args.backend == "torch":
            gen = torch.Generator().manual_seed(args.seed * 7 + b)
            params = sim.sample_params(n, ranges, gen)
            mult = {k: (params[k] / sim.nominal_params()[k]).cpu() for k in params}
        else:   # genesis: params baked per scene -> one random cell per batch
            cell = {k: float(rng.uniform(*v)) for k, v in ranges.items()}
            env, params = env_for_cell(args, n, dev, cell)
            mult = {k: torch.full((n,), cell.get(k, 1.0)) for k in sim.param_names}
        gain = torch.as_tensor(rng.uniform(0.7, 2.0, n), dtype=torch.float32, device=dev)
        sigma = torch.as_tensor(rng.uniform(0.0, 0.3, n) * sim.max_vel,
                                dtype=torch.float32, device=dev)
        out.append(collect(env, sim, params, T, gain, sigma, seed=args.seed * 1000 + 500 + b))
        pars.append(torch.stack([mult[k] for k in sim.param_names], 1))
        print(f"[dyn] {b + n}/{args.n_dyn}")
    S, A, R = (torch.cat(x) for x in zip(*out))
    P = torch.cat(pars)
    torch.save({"states": S, "actions": A, "risk": R, "param_mults": P,
                "param_names": sim.param_names}, os.path.join(od, "dyn_data.pt"))
    print(f"[dyn] {S.shape[0]} eps, step-violation rate "
          f"{(R.amax(-1) > 1).float().mean():.3f}, episode-violation rate "
          f"{(R.amax((1, 2)) > 1).float().mean():.3f}")


if __name__ == "__main__":
    main()
