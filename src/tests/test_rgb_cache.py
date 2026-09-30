"""The RGB frame must be cached across a chunk without changing any action.

A cache that changes behaviour is worse than no cache, and the failure is quiet:
the run still produces SR/CVR, just for a slightly different system. So this
asserts equivalence rather than measuring a saving.

Three things have to hold, and the third is the one that is easy to get wrong:

  1. With needs_rgb, actions must equal the uncached implementation exactly.
     Reconstructed by forcing a render on every step (env.render_rgb called
     outside the cache) and comparing the executed action tensors bit for bit.
  2. The render must actually be skipped. n_render_calls must equal
     n_policy_calls, and n_render_rows strictly less than n_steps * B whenever
     any row is mid-chunk. A cache that never hits costs more than no cache.
  3. The frame a mid-chunk row sees must be the one rendered at its replan, not
     a stale frame from some earlier step. This is the subtle one: if the
     policy read a cached frame from a previous chunk, the chunk-variance
     premise of the whole thesis (a chunk-emitting VLA is checked chunk by
     chunk) would quietly break. So a row that replans must see a *new* frame,
     and the test pins that by making the scene move between replans.

Run: cd src && CUDA_VISIBLE_DEVICES=0 python -u tests/test_rgb_cache.py
"""
import os
import sys

import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from checkvla.verifier import Controller                        # noqa: E402
from experiments._common import env_for_cell                # noqa: E402
from experiments._common import base_parser               # noqa: E402
from sims.base import make_env, make_sim                        # noqa: E402


class _CamPolicy(torch.nn.Module):
    """A policy that reads the RGB frame, so needs_rgb is genuinely exercised.

    The action is a function of the frame's mean colour, which is enough to make
    the cache observable: a stale frame produces a different action, so any
    behaviour change shows up as a tensor mismatch rather than as a metric that
    drifts within noise.
    """

    needs_rgb = True
    H, act_dim = 5, 2

    def __init__(self, dev):
        super().__init__()
        self.lin = torch.nn.Linear(3, 2, bias=False).to(dev)
        with torch.no_grad():
            # action = frame means scaled; keep it inside the sim's +-0.5 clamp
            self.lin.weight.copy_(torch.tensor([[0.4, 0.0, 0.0],
                                                [0.0, 0.4, 0.0]], device=dev))

    def forward(self, obs, rgb, instruction=None):
        m = rgb.mean(dim=(2, 3))                      # (n,3)
        a = self.lin(m).clamp(-0.5, 0.5)              # (n,2)
        return a[:, None, :].expand(-1, self.H, -1).contiguous()   # a chunk


class _UncachedController(Controller):
    """Render every step, ignoring the cache -- the reference implementation."""

    def _ctx(self, env, obs, rows=None):
        self.rgb = env.render_rgb()                  # unconditional
        self.n_render_calls += 1
        self.n_render_rows += obs.shape[0]
        ctx = {"obs": obs, "obs_prev": self.obs_prev, "a_prev": self.a_prev,
               "state": env.get_state(), "instruction": self.instruction,
               "rgb": self.rgb}
        return ctx


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = base_parser(__doc__).parse_args([])
    args.policy, args.camera = "openvla", 1
    args.cam_res, args.cam_ss, args.chunk_k = 32, 1, 5
    B, T = 16, 30

    cell = {"friction": 0.25, "mass": 0.5}
    acts = {}
    counts = {}
    for name, cls in (("cached", Controller), ("uncached", _UncachedController)):
        env, params = env_for_cell(args, B, dev, cell)
        pol = _CamPolicy(dev)
        ctl = cls(pol, env.sim, verifier=None, mode="none", chunk_k=args.chunk_k)
        obs = env.reset(params, seed=7)
        ctl.reset(env, obs)
        out = []
        for t in range(T):
            a, _ = ctl.act(env, obs)
            out.append(a.clone())
            # move the scene so a stale frame would differ from a fresh one
            env.set_state(env.get_state() + _nudge(t, env))
            obs = env.sim.obs(env.get_state())
        acts[name] = torch.stack(out)
        counts[name] = (ctl.n_render_calls, ctl.n_render_rows, ctl.n_policy_calls,
                        ctl.n_policy_rows)

    same = torch.equal(acts["cached"], acts["uncached"])
    print(f"equivalence  cached vs render-every-step: "
          f"{'bit-identical' if same else 'DIFFER'}")
    if not same:
        d = (acts["cached"] - acts["uncached"]).abs()
        print(f"             max |diff| {d.max():.3e} over {T} steps")
    assert same, "caching the frame changed the executed actions"

    rc, rr, pc, pr = counts["cached"]
    _, ur, _, _ = counts["uncached"]
    print(f"calls        policy {pc}/{T} steps, render {rc}/{T} steps "
          f"(they must match: a frame is rendered for a policy call)")
    assert rc == pc, f"render calls {rc} != policy calls {pc}"
    print(f"rows         policy {pr}, render {rr} vs uncached {ur} "
          f"({ur / max(rr, 1):.1f}x fewer)")
    assert rr <= ur, "the cache is not saving anything"
    assert rr < T * B or pc >= T, "expected mid-chunk steps to reuse the frame"

    # ---- 3. the cache must update exactly on replans, and nowhere else
    #
    # The contract has two halves and asserting only the interesting one passes
    # for the wrong reason. At a replan step the frame must be *new*; mid-chunk it
    # must be *identical*. "The frame changes sometimes" would be satisfied by a
    # cache that refreshed on the wrong steps, and "the frame never changes" by no
    # cache at all. An earlier draft of this test asserted min-over-all-steps > 0
    # and therefore rejected the cache for being correct.
    env, params = env_for_cell(args, B, dev, cell)
    pol = _CamPolicy(dev)
    ctl = Controller(pol, env.sim, verifier=None, mode="none",
                     chunk_k=args.chunk_k)
    obs = env.reset(params, seed=7)
    ctl.reset(env, obs)
    seen = []
    for t in range(args.chunk_k * 3):
        a, _ = ctl.act(env, obs)
        seen.append(ctl.rgb.clone())
        env.set_state(env.get_state() + _nudge(t, env, big=True))
        obs = env.sim.obs(env.get_state())
    d = [(seen[i] - seen[i - 1]).abs().mean().item() for i in range(1, len(seen))]
    k = args.chunk_k
    # step i-1 -> i is a replan iff i is a multiple of k
    replan = [x for i, x in enumerate(d, start=1) if i % k == 0]
    mid = [x for i, x in enumerate(d, start=1) if i % k != 0]
    print(f"freshness    at replans   n={len(replan)}  frame delta "
          f"min {min(replan):.5f} (must be > 0)")
    print(f"             mid-chunk    n={len(mid)}  frame delta "
          f"max {max(mid):.2e} (must be exactly 0)")
    assert min(replan) > 0, "a replanning row reused the previous chunk's pixels"
    assert max(mid) == 0.0, ("a mid-chunk step changed the cached frame; the frame "
                             "is only read at the replan, so this is wasted render")
    print("\nall checks passed")
    return 0


def _nudge(t, env, big=False):
    """A deterministic scene change big enough to move the rendered pixels.

    ``big`` is 2 cm/step. The default 4 mm is ~0.25 px at res=32, which is
    sub-pixel: after coverage antialiasing and the gamma curve the frame
    quantises to a bit-identical image, so a freshness probe at that size reports
    "the cache never updates" for a cache that updates correctly. Calibrating a
    check needs a stimulus above its own resolution, which is the same trap as the
    1 cm camera-scale probe in the observability test.
    """
    d = torch.zeros_like(env.get_state())
    k = 0.02 if big else 0.004
    d[:, 0] = k * (t + 1)
    d[:, 1] = 0.5 * k * (t + 1)
    return d


if __name__ == "__main__":
    sys.exit(main())
