"""OpenVLA's generate() is silently wrong on transformers 5.x, and here is the proof.

The most dangerous finding of the VLA work, recorded as a test because the
evidence is the kind that evaporates: a plausible, finite, in-range 7-D action
that is wrong, and that looks exactly like a legitimate negative result.

What generate() does on 5.17, from real weights and a real 224x224 frame:

    generate(..., max_new_tokens=7)          ->  [31872] * 7
    argmax of its own scores, per step       ->  31872 at every step
    max logit of its own scores, per step    ->  4.719 at every step, 3 dp
    the model's own last-position logit      ->  11.062 at step 0
    an uncached greedy loop, same forward    ->  [31744, 31999, 31955, 31842,
                                                  31856, 31871, 31872]

Seven autoregressive steps over seven different prefixes cannot yield an
identical distribution to three decimal places, and the peak does not match the
model's own logits either, so generate is scoring a constant from the wrong
tensor. The shape of it is OpenVLA's cached-generation branch, entered when
``input_ids.shape[1] == 1``: it runs the LLM on a single token with no cache and
ignores ``pixel_values`` entirely.

It does not raise. It returns a plausible action. And the plausible reading of
that action -- "a 256-bin head collapses to the middle bin on an
out-of-distribution task, so zero-shot is worthless here and fine-tuning is
mandatory" -- is a fabrication. The real bins from a correct decode are
[255, 0, 44, 157, 143, 128, 127]: widely spread, not a collapse. The two
conclusions imply opposite experiments, so this is checked rather than noted.

The checks, cheapest first:

  1. tokens_to_bins reproduces the vendored token -> bin arithmetic on hand-made
     inputs, so the reference decode is comparable to predict_action's output
     without a 15 GB model.
  2. greedy_decode on the real model is *not* constant: it must produce at least
     two distinct tokens, and its bins must span more than one bin. This is the
     claim that generate fails, stated so that a future transformers which fixes
     generate cannot quietly make this test pass for the wrong reason.
  3. check_decode_agreement reports generate as unusable, with the evidence.
     Asserted explicitly rather than skipped when it happens to pass: a
     transformers upgrade that fixes generate should *fail* this test and force
     the note to be revisited, not silently retire it.

Loads the 14 GiB checkpoint, so this is the slow test. The fast checks live in
test_tf5_compat.py.

Run: cd src && CUDA_VISIBLE_DEVICES=2 python -u tests/test_decode_agreement.py
"""
import os
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

PROMPT = ("In: What action should the robot take to push the block into the green "
          "seat without hitting the wall?\nOut:")
FAILED = []


def check(cond, label, detail=""):
    print(f"   {'ok  ' if cond else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def main():
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from adapters import tf5_compat

    def stage(m):
        print(f"\n{'=' * 72}\n{m}\n{'=' * 72}", flush=True)

    stage("1. the token -> bin map, on hand-made inputs")
    # From the vendored predict_action:
    #   discretized = vocab_size - token ; clip to [0, n_bins-1] ; then -1
    # Reproduced here so the reference decode is comparable to predict_action
    # without loading the model, and so a change to the arithmetic is visible.
    V, NB = 32000, 256
    n_centres = len(tf5_compat.make_bin_centers(NB))
    print(f"   bin_centers has {n_centres} entries for n_action_bins={NB} "
          f"(the {NB} numbers are edges, not centres), so the clip is to "
          f"{n_centres - 1}")
    check(n_centres == NB - 1,
          "make_bin_centers returns n_bins - 1 values, so the clip tops out at "
          "n_bins - 2",
          "clipping to n_bins - 1 reaches past the end of the array")
    centres = tf5_compat.make_bin_centers(NB)
    check(abs(centres[0] - (-1 + 1 / 255)) < 1e-12
          and abs(centres[-1] - (1 - 1 / 255)) < 1e-12,
          "bin_centers span (-1, 1) as interval midpoints",
          f"[{centres[0]:+.6f}, {centres[-1]:+.6f}]")

    def want(tok):                      # the vendored arithmetic, spelled out
        return int(np.clip(V - tok - 1, 0, n_centres - 1))

    cases = {
        0: want(0),            # 31999 -> clamps to the top
        255: want(255),        # 31744 -> clamps to the top
        257: want(257),        # 31742 -> first unclamped value, 253
        31872: want(31872),    # 127: the token generate() emits, the middle
        31744: want(31744),    # 255 -> clamps to 254
        31745: want(31745),    # 254, reachable without clamping
        31999: want(31999),    # 0, the bottom
        32063: want(32063),    # past vocab_size, clamps to 0 and not negative
        40000: want(40000),    # far past, must not go negative
    }
    for tok, exp in cases.items():
        got = int(tf5_compat.tokens_to_bins([tok], V, NB)[0])
        check(got == exp, f"bin({tok}) == {exp}", f"got {got}")
    # The clip is the part that is easy to get wrong and invisible when it is:
    # without it, low token ids map to bins of ~31744 and blow straight through the
    # 256-bin unnormalisation, i.e. an action 128x outside the dataset's range.
    check(int(tf5_compat.tokens_to_bins([0], V, NB)[0]) == n_centres - 1,
          "a low token id clamps to the top bin instead of exceeding the range")
    check(int(tf5_compat.tokens_to_bins([31872], V, NB)[0]) == n_centres // 2,
          "the collapsed token maps to the middle bin (127 of 255 centres)")

    print("\n   and bins -> the pretraining dataset's units")
    q01 = np.array([-0.028, -0.041, -0.04, -0.082, -0.078, -0.204, 0.0])
    q99 = np.array([0.028, 0.041, 0.04, 0.082, 0.078, 0.204, 1.0])
    lo_a = tf5_compat.bins_to_action([0] * 7, q01, q99)
    hi_a = tf5_compat.bins_to_action([n_centres - 1] * 7, q01, q99)
    mid_a = tf5_compat.bins_to_action([n_centres // 2] * 7, q01, q99)
    # The end bins are *interval midpoints*, so they stop half a bin short of
    # q01/q99 rather than reaching them. The offset is exactly (hi-lo)/510, i.e.
    # half of one bin's width. Asserting equality with q01 here would be wrong;
    # what matters is that the offset is bounded by half a bin and the mapping is
    # monotone, because anything larger would mean the bin grid does not cover the
    # dataset range.
    off_lo = (q99 - q01) / (2 * n_centres)
    check(np.allclose(lo_a, q01 + off_lo) and np.allclose(hi_a, q99 - off_lo),
          "the end bins land within half a bin width of q01 and q99",
          f"low {lo_a[0]:+.6f} vs q01 {q01[0]:+.4f} (offset {off_lo[0]:.2e}), "
          f"high {hi_a[0]:+.6f} vs q99 {q99[0]:+.4f}")
    check(np.all(hi_a > mid_a) and np.all(mid_a > lo_a),
          "the unnormalisation is monotone in the bin index")
    check(np.allclose(mid_a, 0.5 * (q01 + q99)),
          "the middle bin lands mid-range, which is why generate()'s constant "
          "produced a small non-zero action rather than a zero one",
          f"mid {mid_a[0]:+.6f}, |a| would be {np.abs(mid_a).mean():.5f}")
    # A masked dimension keeps the *normalised centre*, not zero and not the
    # unnormalised value -- predict_action's np.where(mask, unnorm, normalized).
    # For bridge_orig dim 6 is the gripper, masked, so this is not hypothetical.
    mask_a = tf5_compat.bins_to_action([0, 0, 0, 0, 0, 0, 0], q01, q99,
                                       mask=[1, 1, 1, 1, 1, 1, 0])
    check(abs(mask_a[6] - centres[0]) < 1e-12,
          "a masked dimension stays in normalised units, per predict_action",
          f"masked dim 6 = {mask_a[6]:+.6f} = bin_centers[0]")
    check(abs(mask_a[0] - lo_a[0]) < 1e-12,
          "an unmasked dimension is still unnormalised")

    print("\n   and the prompt terminator, which predict_action adds for us")
    ids = torch.tensor([[1, 450, 1234, 29901]])
    once = tf5_compat.prepare_prompt_ids(ids)
    twice = tf5_compat.prepare_prompt_ids(once)
    check(int(once[0, -1]) == tf5_compat.PROMPT_END_TOKEN,
          "a prompt not ending in 29871 gets it appended", f"last {int(once[0, -1])}")
    check(once.shape[1] == ids.shape[1] + 1, "exactly one token is added")
    check(twice.shape[1] == once.shape[1],
          "appending is idempotent, so a processor that already added it is safe")
    check(int(ids[0, -1]) != tf5_compat.PROMPT_END_TOKEN,
          "the fixture genuinely lacked the terminator, or these checks are vacuous")

    stage("2. load the real weights")
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
    check(not miss and not unexp, "checkpoint loaded completely",
          f"{len(miss)} missing, {len(unexp)} unexpected")
    del sd
    dev = os.environ.get("OV_DEVICE", "cuda:0")
    model = model.to(dev).eval()
    proc = tf5_compat.load_processor(snap)

    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    with torch.no_grad():
        inp = proc(PROMPT, im).to(dev, dtype=torch.bfloat16)
        if hasattr(inp, "pixel_values"):
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        inp = {k: (v.unsqueeze(0) if v.dim() == 1 else v) for k, v in inp.items()}

    stage("3. the reference decode is not constant -- generate's failure, inverted")
    ref = tf5_compat.greedy_decode(model, inp["input_ids"], inp["pixel_values"],
                                   n_tokens=7)
    ref_t = ref.tolist()
    ref_b = tf5_compat.tokens_to_bins(ref_t, model.vocab_size).tolist()
    print(f"   reference tokens {ref_t}")
    print(f"   reference bins   {ref_b}")
    n_distinct = len(set(ref_t))
    span = max(ref_b) - min(ref_b)
    check(n_distinct >= 2,
          "the reference decode is not one token repeated",
          f"{n_distinct} distinct of {len(ref_t)}")
    check(span > 1,
          "the reference bins span more than one value -- not a collapse",
          f"span {span}, bins {sorted(set(ref_b))}")
    # The specific claim that makes this a diagnosis rather than a curiosity: the
    # correct answer is *not* the middle bin everywhere, so "the head collapses on
    # OOD input" is not what is happening.
    check(sorted(set(ref_b)) != [127],
          "the correct decode is not the all-127 collapse generate() produces")

    stage("4. and a different image gives a different action")
    im2 = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    with torch.no_grad():
        inp2 = proc(PROMPT, im2).to(dev, dtype=torch.bfloat16)
        inp2["pixel_values"] = inp2["pixel_values"].to(torch.bfloat16)
        inp2 = {k: (v.unsqueeze(0) if v.dim() == 1 else v) for k, v in inp2.items()}
    ref2 = tf5_compat.greedy_decode(model, inp2["input_ids"], inp2["pixel_values"],
                                    n_tokens=7).tolist()
    print(f"   image 1 -> {ref_t}")
    print(f"   image 2 -> {ref2}")
    check(ref2 != ref_t, "two different images give two different token sequences",
          f"{sum(a != b for a, b in zip(ref_t, ref2))}/7 positions differ")

    stage("5. generate() is reported unusable, with the evidence")
    rep = tf5_compat.check_decode_agreement(model, inp["input_ids"],
                                            inp["pixel_values"], model.vocab_size)
    print(f"   reference: tokens {rep['reference_tokens']}  bins {rep['reference_bins']}")
    print(f"   generate : tokens {rep['generate_tokens']}  bins {rep['generate_bins']}")
    print(f"   agreeing positions {rep['agreeing_positions']}/{rep['of']}"
          + (f"  error: {rep['generate_error']}" if rep["generate_error"] else ""))
    # Asserted, not skipped. If a transformers upgrade fixes generate, this FAILS
    # on purpose: the note in tf5_compat claiming it is unusable is then stale and
    # has to be rewritten, rather than the test quietly ceasing to mean anything.
    check(not rep["generate_is_usable"],
          "generate() does NOT agree with the reference decode (expected)",
          "if this now fails, generate was fixed -- update the note in tf5_compat "
          "and re-check any conclusion drawn from predict_action output")
    if rep["generate_tokens"] is not None and len(set(rep["generate_tokens"])) == 1:
        print(f"   and it is degenerate: one token "
              f"{rep['generate_tokens'][0]} repeated, which is bin "
              f"{tf5_compat.tokens_to_bins([rep['generate_tokens'][0]], model.vocab_size)[0]}"
              f" of {n_centres} -- the exact middle")

    print("\n" + (f"{len(FAILED)} FAILED: {FAILED}" if FAILED
                  else "all decode checks passed"))
    print("\nConsequence for the record: any claim of the form \"OpenVLA has no "
          "zero-shot\nability on this task\" that rests on predict_action output is "
          "void. The\nsupported decode is tf5_compat.greedy_decode, and the policy "
          "must use it.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
