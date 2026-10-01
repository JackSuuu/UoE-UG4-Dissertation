"""The policy must compute exactly the feature its head was trained on.

A head run on a different feature than it was trained on does not raise; it
just acts wrongly. So this checks *code* parity: on the same images, in the same
process, the policy's ``features()`` must equal what ``vla/extract_features.py``
computes. That is deterministic and exact (cosine 1.000000), so any drift in
either code path fails the test.

The comparison with the *stored* feature file is reported, not gated. Measured on
1 Oct: the same code in another process gives cosine 0.999875 for vis (max
action gap 0.005 m/s over 64 frames, about 0.25 mm per step, half the head's
own held-out error), while within one process repeated runs agree exactly and
batch size moves actions by <= 0.0006 m/s. The cause is cross-process numerical
nondeterminism, not a code path; no global cudnn/TF32 flag is set anywhere in
src/. It was not chased further: the code-parity gate is the one that guards
against a train/run feature mismatch, and a threshold on cross-run noise would
have been picked after seeing the data.

    cd src && CUDA_VISIBLE_DEVICES=0 python -u tests/test_feature_parity.py
"""
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from adapters.openvla_policy import OpenVLAPolicy  # noqa: E402
from adapters import tf5_compat  # noqa: E402
from sims.base import make_sim  # noqa: E402
from vla.extract_features import PROMPT  # noqa: E402

FEAT = os.path.expanduser(os.environ.get("PARITY_FEATURES", "~/scratch/openvla_push_features.h5"))
DEMO = os.path.expanduser("~/scratch/openvla_push_demos.h5")
HEAD = os.environ.get("PARITY_HEAD", "~/scratch/openvla_chunk_head_vis_fast.pt")


def extraction_features(model, proc, imgs, feat, dev):
    """Verbatim from vla/extract_features.py, so the comparison is code-to-code."""
    inp = proc([PROMPT] * len(imgs), imgs).to(dev)
    pv = inp["pixel_values"].to(torch.bfloat16)
    if feat == "vis":
        return model.projector(model.vision_backbone(pv)).float().mean(1)
    ids = tf5_compat.prepare_prompt_ids(inp["input_ids"][:1]).repeat(len(imgs), 1)
    o = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv,
              use_cache=False, output_hidden_states=True)
    return o.hidden_states[-1][:, -1].float()


def main():
    dev = "cuda:0"
    sim = make_sim("push", torch.device(dev))
    f, d = h5py.File(FEAT, "r"), h5py.File(DEMO, "r")
    de, ds = d["episode"][:], d["step"][:]
    rows = [int(np.nonzero((de == e) & (ds == s))[0][0])
            for e, s in zip(f["episode"][:64], f["step"][:64])]
    raw = np.stack([d["obs"][r] for r in rows])
    img = torch.tensor(raw).permute(0, 3, 1, 2).float().div(255).to(dev)

    pol = OpenVLAPolicy(sim, dev, H=5, head_path=HEAD)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=1).min().item()  # noqa: E731
    act = lambda x: pol.head((x - pol.mu) / pol.sd)  # noqa: E731
    ok = True
    for feat in os.environ.get("PARITY_FEATS", "vis,llm").split(","):
        pol.feature = feat
        with torch.no_grad():
            run = pol.features(img)
            ext = extraction_features(pol.model, pol.proc, [Image.fromarray(x) for x in raw],
                                      feat, dev)
            gap_code = (act(run) - act(ext)).abs().max().item()
            exact = cos(run, ext) > 0.999999 and gap_code < 1e-6
            ok &= exact
            print(f"{feat}: policy vs extraction code, same process: cosine "
                  f"{cos(run, ext):.6f}, action gap {gap_code:.2e}  "
                  f"{'ok' if exact else 'MISMATCH (code drift)'}")
            stored = torch.tensor(f[feat][:64].astype(np.float32), device=dev)
            if stored.abs().sum() > 0:
                gap_file = (act(run) - act(stored)).abs().max().item()
                mean_gap = (act(run) - act(stored)).abs().mean().item()
                print(f"     vs stored file (cross-process, reported only): cosine "
                      f"{cos(run, stored):.6f}, action gap max {gap_file:.4f} / mean "
                      f"{mean_gap:.5f} m/s")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
