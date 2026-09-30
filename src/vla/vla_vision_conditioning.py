"""Does OpenVLA know anything about this task? Re-run with the decode that works.

The first run of this question used predict_action, and its answer was void:
generate() emits one token, 31872, for every input on transformers 5.17, which
unnormalises to the middle bin of the bridge range and looks exactly like a
256-bin head collapsing on an out-of-distribution task. tests/test_decode_
agreement.py pins that. Re-run against greedy_decode, which is the ground truth.

The instrument is chosen deliberately. A greedy decode is one deterministic sample
from a 255-way categorical per dimension, so a per-dimension standard deviation
across ten inputs is quantisation noise sitting on top of whatever the head wants
to say, and the two are not separable from the spread alone. The noise-free
statistic is the *exact sequence*: greedy decoding is deterministic, so if the
model ignored the image, every input would give a bit-identical 7-token sequence
and the count of distinct sequences would be exactly 1. That has no sampling
component at all.

Three conditions, because action spread across real frames is not by itself
evidence of conditioning -- those frames also differ in domain-randomised
appearance, and a model ignoring the image entirely would still produce a spread:

  A  state varies, appearance fixed   -> scene-conditioned signal
  B  state fixed, appearance varies   -> appearance noise
  C  pure noise frames                -> what a model ignoring the image emits

The load-bearing comparison is A against B *paired*: the same scenes rendered
under several appearance seeds. A model reading texture rather than the task
produces as much variation in B as in A. And the reference point for "no
conditioning at all" is C.

What this cannot settle, and is not claimed to: whether the emitted directions
are *good*. The units are Bridge end-effector deltas in metres, the task, table
and camera are absent from pretraining, and this measures whether the model
responds to the scene, not whether it can do the task. That needs demos.

Run: CUDA_VISIBLE_DEVICES=2 python -u vla/vla_vision_conditioning.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from adapters import tf5_compat                                    # noqa: E402

DEVICE = os.environ.get("OV_DEVICE", "cuda:0")
UNNORM = "bridge_orig"
PROMPT = ("In: What action should the robot take to push the block into the green "
          "seat without hitting the wall?\nOut:")
N_STATES = 10
N_APPEAR = 3
N_PAIRED = 4          # scenes re-rendered under N_APPEAR appearance seeds
CAM_RES = 224
N_STEPS = 6           # expert steps before rendering, so frames are not all poses


def render(n, seed, cam_seed):
    """Real frames from the torch ground truth, plus the states that made them."""
    from sims.base import make_sim, TorchEnv
    dev = torch.device(DEVICE)
    sim = make_sim("push", dev)
    env = TorchEnv(sim, n, camera=True, cam_res=CAM_RES, cam_ss=1, cam_seed=cam_seed)
    env.reset(sim.make_params(n, {}), seed=seed)
    for _ in range(N_STEPS):
        a = sim.expert(env.state, env.params)
        env.state, _ = sim.step(env.state, a, env.params)
    img = env.render_rgb().clamp(0, 1)
    img = (img * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return img, env.state.detach().cpu().numpy(), sim


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
    miss, unexp = model.load_state_dict(sd, strict=False, assign=True)
    miss = [k for k in miss if "rotary" not in k and "inv_freq" not in k]
    assert not miss and not unexp, (miss[:4], unexp[:4])
    del sd
    model = model.to(DEVICE).eval()
    proc = tf5_compat.load_processor(snap)

    n_bins = len(tf5_compat.make_bin_centers(256))
    stats = cfg.norm_stats[UNNORM]["action"]
    q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
    print(f"loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B, "
          f"{n_bins} bins, unnorm_key {UNNORM}")
    print(f"decode = greedy_decode, ground truth per test_decode_agreement.py "
          f"(NOT predict_action)")

    def decode_batch(imgs):
        """Greedy-decode each frame. 7 uncached forwards per frame, so this is
        ~1 s a frame; the cost buys an answer that is not a single sample."""
        toks, bins, acts = [], [], []
        for im in imgs:
            with torch.no_grad():
                inp = proc(PROMPT, Image.fromarray(im)).to(DEVICE, dtype=torch.bfloat16)
                inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
                inp = {k: (v.unsqueeze(0) if v.dim() == 1 else v)
                       for k, v in inp.items()}
            t = tf5_compat.greedy_decode(model, inp["input_ids"],
                                         inp["pixel_values"], n_tokens=7)
            b = tf5_compat.tokens_to_bins(t.cpu().numpy(), model.vocab_size)
            toks.append(t.tolist())
            bins.append(b.tolist())
            acts.append(tf5_compat.bins_to_action(b, q01, q99,
                                                  mask=stats.get("mask")).tolist())
        return toks, np.array(bins), np.array(acts)

    def report(label, toks, bins):
        uniq = {tuple(t) for t in toks}
        sd = bins.std(axis=0)
        const = [i for i, v in enumerate(sd) if v <= 0]
        print(f"   {label}")
        print(f"      distinct 7-token sequences: {len(uniq)} of {len(toks)}")
        print(f"      per-dim bin std: {np.array2string(sd, precision=1)}")
        if const:
            names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
            print(f"      constant dims: {[names[i] for i in const]}")
        return len(uniq), sd

    print("\nA. real frames, state varies, appearance fixed")
    imgA, stA, sim = render(N_STATES, seed=11, cam_seed=3)
    print(f"   {len(imgA)} frames; peg x in [{stA[:, 0].min():.3f}, "
          f"{stA[:, 0].max():.3f}] m, target y in [{stA[:, 6].min():+.3f}, "
          f"{stA[:, 6].max():+.3f}] m")
    tA, bA, aA = decode_batch(imgA)
    nA, sdA = report("A", tA, bA)

    print("\nB. the same scenes, appearance re-randomised (paired)")
    per_scene_tokens = []
    for ap in (4, 5, 6):
        imgB, stB, _ = render(N_PAIRED, seed=11, cam_seed=ap)
        # same first N_PAIRED scenes, different appearance
        tB, bB, _ = decode_batch(imgB)
        per_scene_tokens.append((tB, bB))
    # tokens that change when only the appearance changes
    changed = [sum(a != b for a, b in zip(per_scene_tokens[0][0][i],
                                          per_scene_tokens[1][0][i]))
               for i in range(N_PAIRED)]
    print(f"   tokens changed by appearance alone, per scene: {changed} of 7")
    print(f"   mean {np.mean(changed):.2f} of 7 positions")
    sdB = np.mean([per_scene_tokens[k][1].std(axis=0) for k in range(N_APPEAR)],
                  axis=0)
    print(f"   per-dim bin std across appearances: {np.array2string(sdB, precision=1)}")

    print("\nC. pure noise frames, matched in count")
    rng = np.random.default_rng(0)
    noise = [rng.integers(0, 255, (CAM_RES, CAM_RES, 3), dtype=np.uint8)
             for _ in range(N_STATES)]
    tC, bC, aC = decode_batch(noise)
    nC, sdC = report("C", tC, bC)

    print("\n" + "=" * 72)
    print(f"   distinct sequences:  real states {nA}/{N_STATES}   "
          f"noise {nC}/{N_STATES}")
    print(f"   tokens moved by appearance alone: {np.mean(changed):.2f} of 7")
    print(f"   per-dim bin std   state {sdA.mean():.1f}   "
          f"appearance {sdB.mean():.1f}   noise {sdC.mean():.1f}")
    print("=" * 72)
    if nA == 1 and nC == 1:
        print("\nVERDICT: one sequence for every input in every condition. The model")
        print("is not conditioning on the image at all, and a fourth transformers")
        print("5.x breakage would be the likely cause -- the compat shim is")
        print("incomplete and the '7B VLA runs' claim does not stand.")
    elif nA > 1 and nA > nC:
        print("\nVERDICT: the head is alive and reads the scene. Distinct sequences")
        print("on real frames, more than on noise, so the response is not a fixed")
        print("function of the prompt alone.")
        if np.mean(changed) >= 5:
            print("\n   But note: appearance alone moves most of the tokens, so the")
            print("   response is dominated by rendered appearance rather than by the")
            print("   task geometry. That is a statement about what the head is")
            print("   keying on, and it is exactly the confound the camera's")
            print("   domain randomisation introduces. It also means a fine-tune on")
            print("   frames from a *fixed* appearance would overfit the texture, so")
            print("   demos have to span appearances or the head will learn the")
            print("   renderer's lighting instead of the peg.")
        print("\n   Still not a capability claim. The units are Bridge deltas, and")
        print("   whether the emitted direction is *useful* needs demos and a")
        print("   success rate, not a spread.")
    else:
        print("\nVERDICT: inconclusive -- read the counts above for which condition")
        print("collapsed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
