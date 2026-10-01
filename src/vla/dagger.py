"""DAgger for the OpenVLA chunk head: label the states the policy actually visits.

Behaviour cloning on expert states fails in closed loop when the policy's own
small errors take it to states the expert never visited (covariate shift). The
llm-readout head shows exactly this: held-out R^2 0.947 open loop, but SR 0.50 at
the nominal physics its demos were collected under. DAgger fixes the data, not
the model: roll the current policy out, have the scripted expert label every
state the policy reaches with the H-step chunk the expert would execute from
there, add those pairs, retrain, repeat.

Cheap here for two reasons: the expert is a state-feedback controller, so it can
label any state by rolling a cloned copy of the sim forward H steps; and the
policy computes the frozen-backbone feature on its own way through, so the
feature is saved directly with no second 7B pass.

Rollouts use nominal physics only. The actor learns the task skill; physical OOD
remains the verifier's job, so the two stay cleanly separated.

    cd src && CUDA_VISIBLE_DEVICES=0 setsid nohup python -u vla/dagger.py \
        --head ~/scratch/openvla_chunk_head_llm_keepall.pt --round 1 \
        > ~/scratch/dagger_r1.log 2>&1 < /dev/null &
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from adapters.openvla_policy import OpenVLAPolicy  # noqa: E402
from sims.base import TorchEnv, make_sim  # noqa: E402


@torch.no_grad()
def expert_chunk(sim, state, params, H):
    """The expert's next H actions from ``state``, on a cloned copy of the sim."""
    s = state.clone()
    out = []
    for _ in range(H):
        a = sim.expert(s, params)
        out.append(a)
        s, _ = sim.step(s, a, params)
    return torch.stack(out, 1)                          # (n, H, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n_episodes", type=int, default=1024)
    ap.add_argument("--n_envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80, help="sim.T, the eval length")
    ap.add_argument("--chunk_k", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=50000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda:0"
    sim = make_sim("push", torch.device(dev))
    pol = OpenVLAPolicy(sim, dev, H=args.chunk_k, head_path=args.head)
    H, K = pol.head.H, args.chunk_k
    out = os.path.expanduser(args.out or f"~/scratch/dagger_r{args.round}.h5")
    feats, tgts, eps, sts = [], [], [], []
    n_succ = 0
    t0 = time.time()
    seed_base = args.seed_start + 100000 * args.round       # fresh episodes each round
    for b in range(-(-args.n_episodes // args.n_envs)):
        n = min(args.n_envs, args.n_episodes - b * args.n_envs)
        env = TorchEnv(sim, n, camera=True, cam_res=224, cam_ss=1)
        env.reset(sim.make_params(n, {}), seed=seed_base + b)
        with torch.no_grad():
            for t in range(0, args.steps, K):
                img = env.render_rgb()
                f = pol.features(img)
                chunk = pol.head((f - pol.mu) / pol.sd)[:, :K].clamp(-sim.max_vel, sim.max_vel)
                feats.append(f.half().cpu().numpy())
                tgts.append(expert_chunk(sim, env.state, env.params, H).cpu().numpy())
                eps.append(np.arange(b * args.n_envs, b * args.n_envs + n) + 10 ** 6 * args.round)
                sts.append(np.full(n, t))
                for k in range(K):                      # the policy drives, open loop
                    env.state, _ = sim.step(env.state, chunk[:, k], env.params)
        n_succ += int(sim.success(env.state).sum()) if hasattr(sim, "success") else 0
        done = b * args.n_envs + n
        print(f"  {done}/{args.n_episodes} episodes, policy SR so far "
              f"{n_succ / done:.2f}, {time.time() - t0:.0f}s", flush=True)

    with h5py.File(out, "w") as f:
        f.create_dataset("llm" if pol.feature == "llm" else "vis", data=np.concatenate(feats))
        f.create_dataset("target", data=np.concatenate(tgts).astype(np.float32))
        f.create_dataset("episode", data=np.concatenate(eps))
        f.create_dataset("step", data=np.concatenate(sts))
        f.attrs.update(H=H, round=args.round, head=args.head, policy_sr=n_succ / args.n_episodes)
    print(f"done: {sum(len(x) for x in feats)} labelled policy states -> {out}  "
          f"(policy SR at nominal during collection {n_succ / args.n_episodes:.2f})", flush=True)


if __name__ == "__main__":
    main()
