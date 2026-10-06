"""Predictor training data from the states the OpenVLA actor actually visits.

The physics predictor (OrbiSim-style MLP) was trained only on noisy-expert
rollouts. The OpenVLA actor reaches states the expert never did (it pushes
harder, and in heavy cells it approaches the wall where the expert never
violated), and the learned verifier's errors concentrate exactly there. This
collects the same (states, actions, risk, param_mults) format as
experiments/collect_data.py's dyn_data.pt, but with OpenVLA driving: open-loop
chunks every ``chunk_k`` steps, plus the same temporally correlated action noise
the expert data uses, so both safe and violating transitions are covered.
Physics is sampled over the predictor's training ranges (mass widened to
0.5-2.0, friction unchanged 0.5-1.5, so friction 0.2x stays out of range).

    cd src && CUDA_VISIBLE_DEVICES=2 setsid nohup python -u vla/collect_vla_dyn.py \
        > ~/scratch/collect_vla_dyn.log 2>&1 < /dev/null &
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from registry import build_policy  # noqa: E402
from sims.base import TorchEnv, make_sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="~/scratch/openvla_chunk_head_vis_h10.pt")
    ap.add_argument("--n_episodes", type=int, default=1024)
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--chunk_k", type=int, default=5)
    ap.add_argument("--friction", type=float, nargs=2, default=(0.5, 1.5))
    ap.add_argument("--mass", type=float, nargs=2, default=(0.5, 2.0))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="~/scratch/dyn_data_vla.pt")
    args = ap.parse_args()

    dev = "cuda:0"
    sim = make_sim("push", torch.device(dev))
    od = os.path.join(SRC, "results", "push_torch")
    pol = build_policy(SimpleNamespace(policy="openvla", chunk_k=args.chunk_k,
                                       vla_head_path=args.head, action_mode="planar_head"),
                       od, dev, sim)
    rng = np.random.default_rng(args.seed)
    ranges = {"friction": tuple(args.friction), "mass": tuple(args.mass)}
    S_all, A_all, R_all, P_all = [], [], [], []
    K, T = args.chunk_k, sim.T
    for b in range(-(-args.n_episodes // args.n_envs)):
        n = min(args.n_envs, args.n_episodes - b * args.n_envs)
        gen = torch.Generator().manual_seed(args.seed * 7919 + b)
        params = sim.sample_params(n, ranges, gen)
        mult = torch.stack([(params[k] / sim.nominal_params()[k]).cpu()
                            for k in sim.param_names], 1)
        env = TorchEnv(sim, n, camera=True, cam_res=224, cam_ss=1)
        env.reset(params, seed=args.seed * 1000 + b)
        sigma = torch.as_tensor(rng.uniform(0.0, 0.3, n) * sim.max_vel,
                                dtype=torch.float32, device=dev)
        g = torch.Generator().manual_seed(args.seed + 12345 + b)
        noise = torch.zeros(n, sim.act_dim, device=dev)
        S, A, R = [env.state.clone()], [], []
        with torch.no_grad():
            for t in range(T):
                if t % K == 0:
                    chunk = pol(sim.obs(env.state), env.render_rgb())
                a = chunk[:, t % K]
                eps = torch.randn(n, sim.act_dim, generator=g).to(dev)
                noise = 0.8 * noise + 0.6 * sigma[:, None] * eps
                a = (a + noise).clamp(-sim.max_vel, sim.max_vel)
                env.state, r = sim.step(env.state, a, env.params)
                S.append(env.state.clone()); A.append(a); R.append(r)
        S_all.append(torch.stack(S, 1).cpu()); A_all.append(torch.stack(A, 1).cpu())
        R_all.append(torch.stack(R, 1).cpu()); P_all.append(mult)
        Rb = torch.stack(R, 1)
        print(f"  {b * args.n_envs + n}/{args.n_episodes} episodes, episode-violation "
              f"rate {(Rb.amax((1, 2)) > 1).float().mean():.2f}", flush=True)

    S, A, R, P = (torch.cat(x) for x in (S_all, A_all, R_all, P_all))
    torch.save({"states": S, "actions": A, "risk": R, "param_mults": P,
                "param_names": sim.param_names}, os.path.expanduser(args.out))
    print(f"done: {S.shape[0]} eps, step-violation rate {(R.amax(-1) > 1).float().mean():.3f}, "
          f"episode-violation rate {(R.amax((1, 2)) > 1).float().mean():.3f} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
