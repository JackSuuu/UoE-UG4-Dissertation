"""Does the head ignore the renderer's appearance, or is the argmax just robust?

The paired measurement came back 0 of 7 action tokens changed when a scene was
re-rendered under a different domain-randomised appearance seed, while pure noise
frames produced 9 of 10 distinct sequences -- as much variation as the real
frames. Those two facts sit awkwardly together, and they have opposite
consequences:

  BLIND. The head never looks at rendered appearance, only at coarse image
  structure. Then demos can be collected at a single appearance, and a
  fine-tune cannot learn the renderer's lighting -- but it also cannot be
  reading the peg, and something else is driving the output.

  ROBUST ARGMAX. Appearance does move the head, but by less than the margin
  between the top two action tokens, so a 1-of-255 greedy sample does not flip.
  Then fine-tuning on a single appearance would let the head latch onto texture
  it is already sensitive to, and the demos have to span appearances.

A greedy action is the wrong instrument for this question and cannot answer it: it
is one deterministic sample from a 255-way categorical, so "the action did not
change" and "the head did not move" are indistinguishable. The instrument has to
be the logit row, restricted to the action token block (31745..31999) and
renormalised, which is what the head is actually choosing among. Then the
comparison is scale-free:

  appearance shift  = distance between two appearance seeds of the same scene
  state shift       = distance between two scenes at the same appearance seed
  noise shift       = distance between a real frame and a noise frame

and the argmax margin is reported alongside, because it is what converts a small
appearance shift into an unchanged action, and reporting one without the other
would leave the 0-of-7 result unexplained.

Run: CUDA_VISIBLE_DEVICES=2 python -u vla/vla_appearance_invariance.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from adapters import tf5_compat                                    # noqa: E402

DEVICE = os.environ.get("OV_DEVICE", "cuda:0")
PROMPT = ("In: What action should the robot take to push the block into the green "
          "seat without hitting the wall?\nOut:")
N_SCENES = 8
APPEAR_SEEDS = (3, 4, 5)
CAM_RES = 224
N_STEPS = 6


def render(n, state_seed, appear_seed):
    """Frames for fixed initial states under a chosen appearance.

    The two seeds are separate because in ``TorchEnv`` they are not: ``reset``
    seeds one generator, and ``reset_appearance`` draws from *that* generator, so
    the episode seed fixes the pixels along with the initial states and
    ``cam_seed`` only has an effect before the first reset. Varying only the
    ``reset`` seed therefore changes the state as well, and "same scene, different
    appearance" is not expressible through the public entry point. Calling
    ``cam.reset_appearance(n, gen)`` directly with its own generator is the way
    out, and it re-randomises appearance without touching the state -- the
    camera's own API, so there is no second implementation of it here to drift.
    """
    from sims.base import make_sim, TorchEnv
    sim = make_sim("push", torch.device(DEVICE))
    env = TorchEnv(sim, n, camera=True, cam_res=CAM_RES, cam_ss=1)
    env.reset(sim.make_params(n, {}), seed=state_seed)
    env.cam.reset_appearance(n, torch.Generator().manual_seed(appear_seed))
    for _ in range(N_STEPS):
        a = sim.expert(env.state, env.params)
        env.state, _ = sim.step(env.state, a, env.params)
    px = env.render_rgb().clamp(0, 1)
    img = (px * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return img, px, env.state.detach().cpu().numpy()


def main():
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoConfig

    tf5_compat.setup_hf_offline()
    snap = tf5_compat.snapshot_dir()
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
    model, _ = tf5_compat.build_architecture(cfg, snap)
    sd = {}
    for f in sorted(os.listdir(snap)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(snap, f), device="cpu"))
    model.load_state_dict(sd, strict=False, assign=True)
    del sd
    model = model.to(DEVICE).eval()
    proc = tf5_compat.load_processor(snap)

    act_slice = tf5_compat.action_token_slice(model.vocab_size)
    print(f"loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B, "
          f"action block {act_slice[0]}..{act_slice[-1]} ({len(act_slice)} tokens)")

    def head(im):
        """Greedy-decode, return the action-block log-probabilities per step and
        the argmax margin. Renormalised over the action block, because that is the
        set the head is choosing among and the other ~31.8k logits are text."""
        with torch.no_grad():
            inp = proc(PROMPT, Image.fromarray(im)).to(DEVICE, dtype=torch.bfloat16)
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            inp = {k: (v.unsqueeze(0) if v.dim() == 1 else v)
                   for k, v in inp.items()}
            toks, lg = tf5_compat.greedy_decode(
                model, inp["input_ids"], inp["pixel_values"],
                n_tokens=7, return_logits=True)
        p = torch.softmax(lg[:, act_slice], dim=-1).numpy()
        top2 = np.sort(p, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return p, margin, toks.tolist()

    # ---- collect: every scene under every appearance seed, plus noise ----------
    P, M, T = {}, {}, {}
    px_by, st_by = {}, {}
    for ap in APPEAR_SEEDS:
        imgs, px, st = render(N_SCENES, state_seed=11, appear_seed=ap)
        px_by[ap] = px.cpu().numpy()
        st_by[ap] = st
        for i in range(N_SCENES):
            p, m, t = head(imgs[i])
            P[(ap, i)], M[(ap, i)], T[(ap, i)] = p, m, t
        print(f"   appearance {ap}: decoded {N_SCENES} scenes")

    rng = np.random.default_rng(0)
    noise = [rng.integers(0, 255, (CAM_RES, CAM_RES, 3), dtype=np.uint8)
             for _ in range(N_SCENES)]
    PN, MN, TN = [], [], []
    for im in noise:
        p, m, t = head(im)
        PN.append(p)
        MN.append(m)
        TN.append(t)
    print(f"   noise: decoded {N_SCENES} frames")

    # ---- 0. the manipulation has to have done something -----------------------
    # This check is the whole reason the previous run of this script produced a
    # confident and completely false "the head is blind to appearance": the
    # appearance seeds rendered bit-identical frames, so it was measuring a no-op
    # and reporting 0.00. Any harness that varies a factor has to prove the factor
    # moved before it is allowed to interpret a downstream null.
    print("\n0. did the manipulation actually do anything?")
    ok_px = True
    for ap in APPEAR_SEEDS[1:]:
        d = np.abs(px_by[ap] - px_by[APPEAR_SEEDS[0]])
        frac = float((d > 1 / 255).mean())
        ok_px &= frac > 0.01
        print(f"   appearance {ap} vs {APPEAR_SEEDS[0]}: "
              f"mean |dpx| {d.mean():.4f}, max {d.max():.3f}, "
              f"pixels changed >1/255: {100 * frac:.1f}%")
    st_same = all(np.array_equal(st_by[APPEAR_SEEDS[0]], st_by[ap])
                  for ap in APPEAR_SEEDS)
    print(f"   states identical across appearance seeds: {st_same}")
    if not ok_px:
        print("\n   ABORT: the appearance seeds produced (near-)identical frames, so")
        print("   every appearance number below is a measurement of nothing. Fix the")
        print("   harness before reading them.")
        return 2
    if not st_same:
        print("\n   ABORT: the states differ across appearance seeds, so the 'state")
        print("   only' and 'appearance only' columns would not be comparable.")
        return 2
    print("   -> pixels differ, states do not. The two factors are separated.")

    def dist(p, q):
        """Mean per-step L1 distance between two action distributions."""
        return float(np.abs(p - q).sum(axis=-1).mean())

    # ---- the three shifts, all on the same scale -----------------------------
    appear = [dist(P[(APPEAR_SEEDS[0], i)], P[(ap, i)])
              for i in range(N_SCENES) for ap in APPEAR_SEEDS[1:]]
    state = [dist(P[(APPEAR_SEEDS[0], i)], P[(APPEAR_SEEDS[0], j)])
             for i in range(N_SCENES) for j in range(i + 1, N_SCENES)]
    noise_d = [dist(P[(APPEAR_SEEDS[0], i)], PN[j])
               for i in range(N_SCENES) for j in range(N_SCENES)]

    marg = np.concatenate([M[(ap, i)] for ap in APPEAR_SEEDS
                          for i in range(N_SCENES)] + MN)

    print("\n1. action-distribution shift (mean L1 over the 255 action tokens)")
    print(f"   appearance only, same scene : {np.mean(appear):.4f}  "
          f"(n={len(appear)})")
    print(f"   state only, same appearance: {np.mean(state):.4f}  "
          f"(n={len(state)})")
    print(f"   real frame vs noise frame   : {np.mean(noise_d):.4f}  "
          f"(n={len(noise_d)})")
    print(f"\n   appearance / state  = {np.mean(appear) / np.mean(state):.3f}")
    print(f"   appearance / noise   = {np.mean(appear) / np.mean(noise_d):.3f}")

    print("\n2. the argmax margin, which is what turns a small shift into no change")
    print(f"   margin (top1 - top2 prob) over the action block: "
          f"median {np.median(marg):.3f}, 10th pct {np.percentile(marg, 10):.3f}, "
          f"min {marg.min():.4f}")
    n_changed = sum(sum(a != b for a, b in zip(T[(APPEAR_SEEDS[0], i)],
                                               T[(ap, i)]))
                    for i in range(N_SCENES) for ap in APPEAR_SEEDS[1:])
    n_tot = N_SCENES * len(APPEAR_SEEDS[1:]) * 7
    print(f"   tokens changed by appearance alone: {n_changed}/{n_tot}")

    print("\n" + "=" * 72)
    r = np.mean(appear) / np.mean(state)
    if r < 0.2:
        print(f"VERDICT: BLIND to appearance. Appearance moves the head {r:.2f}x as")
        print("much as a change of scene does, so the renderer is not being read.")
        print("Demos can be collected at a single appearance, and a fine-tune on")
        print("them cannot be learning the lighting -- though it is also not clear")
        print("what visual evidence the head is using.")
    elif r < 0.7:
        print(f"VERDICT: partially appearance-sensitive, and the argmax is robust.")
        print(f"A change of appearance shifts the action distribution {r:.2f}x as much")
        print("as a change of scene, so the head is reading appearance, but by less")
        print("than the top-two margin, so a greedy sample does not flip. This")
        print("resolves the earlier 0-of-7: it was the sampler, not the model.")
        print("\nConsequence for the demos: the head is sensitive to rendered")
        print("appearance, so a fine-tune on frames from a single seed would let it")
        print("learn that seed's lighting. Demos must span appearance seeds, or the")
        print("fine-tuned actor will fail under the same domain randomisation the")
        print("verifier and the camera are evaluated with.")
    else:
        print(f"VERDICT: appearance moves the head as much as the scene does "
              f"({r:.2f}x).")
        print("The renderer is a dominant input, which is a problem for a task that")
        print("is supposed to be about geometry. Check that the appearance")
        print("randomisation is not stronger than intended before fine-tuning on it.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
