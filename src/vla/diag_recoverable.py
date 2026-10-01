"""Is the violation still avoidable when the verifier gets to act?

The headline OpenVLA run showed the oracle verifier (gt_shadow) clearing only 49%
of its interventions at friction 0.2x, against 94% for the bc actor. Hypothesis:
on low friction the peg coasts, the pusher can only push (never brake), so by the
chunk boundary where the verifier first acts, the strike may already be
committed by momentum built up in earlier chunks. If so, no repair of the
*current* chunk can help, and the verifier would need to act earlier (a longer
horizon), not better.

Test, with the true simulator: at every chunk boundary of a policy-only rollout,
replay from a cloned state (a) the policy's proposed chunk, (b) the chunk scaled
to 25%, (c) an all-zero chunk (pusher stops). A boundary where (a) violates and
(c) also violates is *unrecoverable at that boundary*. Reported for the OpenVLA
vis actor and the bc actor on the same cells and seeds.

    cd src && CUDA_VISIBLE_DEVICES=0 python -u vla/diag_recoverable.py
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from registry import build_policy  # noqa: E402
from sims.base import TorchEnv, make_sim  # noqa: E402


@torch.no_grad()
def chunk_risk(sim, state, params, chunk):
    s, r = state.clone(), torch.zeros(len(state), device=state.device)
    for k in range(chunk.shape[1]):
        s, risk = sim.step(s, chunk[:, k], params)
        r = torch.maximum(r, risk[:, 0])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="~/scratch/openvla_chunk_head_vis_keepall.pt")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda:0"
    sim = make_sim("push", torch.device(dev))
    od = os.path.join(SRC, "results", "push_torch")
    actors = {
        "openvla_vis": build_policy(SimpleNamespace(policy="openvla", chunk_k=5,
                                                    vla_head_path=args.head,
                                                    action_mode="planar_head"), od, dev, sim),
        "bc": build_policy(SimpleNamespace(policy="bc"), od, dev, sim),
    }
    K = 5
    for cell in ({"friction": 0.2, "mass": 1.0}, {"friction": 0.2, "mass": 2.0},
                 {"friction": 1.8, "mass": 2.0}):
        print(f"\ncell {cell}")
        for name, pol in actors.items():
            env = TorchEnv(sim, args.n, camera=True, cam_res=224, cam_ss=1)
            env.reset(sim.make_params(args.n, cell), seed=3000)
            n_viol = n_unrec = n_scal = 0
            first_seen = torch.zeros(args.n, dtype=torch.bool, device=dev)
            first_rec = first_unrec_n = 0
            first_t = []
            ever = torch.zeros(args.n, dtype=torch.bool, device=dev)
            with torch.no_grad():
                for t in range(0, sim.T, K):
                    obs = sim.obs(env.state)
                    img = env.render_rgb() if getattr(pol, "needs_rgb", False) else None
                    chunk = pol(obs, img)[:, :K].clamp(-sim.max_vel, sim.max_vel)
                    r_prop = chunk_risk(sim, env.state, env.params, chunk)
                    r_q = chunk_risk(sim, env.state, env.params, 0.25 * chunk)
                    r_0 = chunk_risk(sim, env.state, env.params, torch.zeros_like(chunk))
                    v = r_prop > 1
                    n_viol += int(v.sum())
                    n_unrec += int((v & (r_0 > 1)).sum())
                    n_scal += int((v & (r_q <= 1)).sum())
                    new = v & ~first_seen                    # this episode's FIRST violating chunk
                    first_rec += int((new & (r_0 <= 1)).sum())
                    first_unrec_n += int((new & (r_0 > 1)).sum())
                    first_t += [t] * int(new.sum())
                    first_seen |= v
                    ever |= v & (r_0 > 1)
                    for k in range(K):
                        env.state, _ = sim.step(env.state, chunk[:, k], env.params)
            nv = max(n_viol, 1)
            print(f"  {name:12s} violating chunks {n_viol:4d} | unrecoverable even with "
                  f"zero action {n_unrec / nv:5.2f} | fixed by 25% scale {n_scal / nv:5.2f}"
                  f" | episodes with an unrecoverable chunk {float(ever.float().mean()):.2f}")
            nf = max(first_rec + first_unrec_n, 1)
            print(f"  {'':12s} FIRST violation per episode: {nf} episodes, recoverable "
                  f"{first_rec / nf:.2f}, at step {sum(first_t) / nf:.1f} on average")


if __name__ == "__main__":
    main()
