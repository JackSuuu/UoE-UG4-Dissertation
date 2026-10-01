"""Collect scripted-expert demos for the 2-D planar chunk head.

Runs ``n_envs`` environments in parallel (the camera renders a batch in one
call), and appends each batch of episodes to the HDF5 file as soon as it is
done, so a partial run still leaves usable data.

Per frame we store the RGB image, the expert's action at that step, and the
episode id / step index. The chunk target (the next H actions) is assembled
at training time from consecutive steps of the same episode, so H is not
baked into the file.

Long runs must be detached from the launching shell, or they are killed when
it exits:
    cd src && CUDA_VISIBLE_DEVICES=0 setsid nohup python -u vla/collect_demos.py \
        --n_episodes 3000 > ~/scratch/collect_demos.log 2>&1 < /dev/null &
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

from sims.base import TorchEnv, make_sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=3000)
    ap.add_argument("--n_envs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cam_res", type=int, default=224)
    ap.add_argument("--out", default="~/scratch/openvla_push_demos.h5")
    ap.add_argument("--seed_start", type=int, default=10000)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    sim = make_sim("push", dev)
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R, T, B = args.cam_res, args.steps, args.n_envs
    n_batches = -(-args.n_episodes // B)

    with h5py.File(out, "w") as f:
        f.create_dataset("obs", (0, R, R, 3), maxshape=(None, R, R, 3), dtype=np.uint8,
                         chunks=(T, R, R, 3), compression="gzip", compression_opts=4)
        f.create_dataset("act", (0, 2), maxshape=(None, 2), dtype=np.float32)
        f.create_dataset("episode", (0,), maxshape=(None,), dtype=np.int64)
        f.create_dataset("step", (0,), maxshape=(None,), dtype=np.int64)
        f.attrs.update(cam_res=R, dt=sim.dt, steps_per_episode=T,
                       action_unit="m/s world-frame planar velocity")

    print(f"{args.n_episodes} episodes = {n_batches} batches x {B} envs x {T} steps",
          flush=True)
    t0, done = time.time(), 0
    for b in range(n_batches):
        nb = min(B, args.n_episodes - done)
        env = TorchEnv(sim, nb, camera=True, cam_res=R, cam_ss=1)
        env.reset(sim.make_params(nb, {}), seed=args.seed_start + b)
        obs = np.empty((nb, T, R, R, 3), np.uint8)
        act = np.empty((nb, T, 2), np.float32)
        with torch.no_grad():
            for t in range(T):
                a = sim.expert(env.state, env.params)
                img = env.render_rgb().clamp(0, 1).mul(255).round().to(torch.uint8)
                obs[:, t] = img.permute(0, 2, 3, 1).cpu().numpy()
                act[:, t] = a.cpu().numpy()
                env.state, _ = sim.step(env.state, a, env.params)

        ep = np.repeat(np.arange(done, done + nb), T)
        st = np.tile(np.arange(T), nb)
        with h5py.File(out, "a") as f:
            n0 = f["obs"].shape[0]
            n1 = n0 + nb * T
            for k, v in (("obs", obs.reshape(-1, R, R, 3)), ("act", act.reshape(-1, 2)),
                         ("episode", ep), ("step", st)):
                f[k].resize(n1, axis=0)
                f[k][n0:n1] = v
        done += nb
        el = time.time() - t0
        print(f"  batch {b + 1}/{n_batches}: {done} episodes, {done * T} frames, "
              f"{el:.0f}s elapsed, ~{el / done * (args.n_episodes - done):.0f}s left",
              flush=True)

    print(f"done: {done} episodes, {done * T} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
