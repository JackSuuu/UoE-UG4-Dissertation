"""Check that the policy computes exactly the features the head was trained on.

A head trained on one feature and run on another does not raise, it just acts
wrongly, so this compares the policy's run-time features against the cached
training features on the same demo frames, for both readouts, and times them.

    cd src && CUDA_VISIBLE_DEVICES=0 python -u tests/test_feature_parity.py
"""
import os
import sys
import time

import h5py
import numpy as np
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from adapters.openvla_policy import OpenVLAPolicy  # noqa: E402
from sims.base import make_sim  # noqa: E402

FEAT = os.path.expanduser("~/scratch/openvla_push_features.h5")
DEMO = os.path.expanduser("~/scratch/openvla_push_demos.h5")


def main():
    dev = "cuda:0"
    sim = make_sim("push", torch.device(dev))
    f, d = h5py.File(FEAT, "r"), h5py.File(DEMO, "r")
    ep, st = f["episode"][:16], f["step"][:16]
    de, ds = d["episode"][:], d["step"][:]
    rows = [int(np.nonzero((de == e) & (ds == s))[0][0]) for e, s in zip(ep, st)]
    img = torch.tensor(np.stack([d["obs"][r] for r in rows])).permute(0, 3, 1, 2).float() / 255
    img = img.to(dev)
    obs = torch.zeros(len(rows), 10, device=dev)

    pol = OpenVLAPolicy(sim, dev, H=5, head_path="~/scratch/openvla_chunk_head_vis_keepall.pt")
    ok = True
    for feat in ("vis", "llm"):
        pol.feature = feat
        x = pol.features(img)
        ref = torch.tensor(f[feat][:16].astype(np.float32), device=dev)
        cos = torch.nn.functional.cosine_similarity(x, ref, dim=1).min().item()
        # The criterion that matters is the action, not the raw feature. bf16
        # kernels differ slightly between the vision-only path and the full
        # forward, so features are not bit-identical; that is harmless if the
        # resulting action gap is far below the head's own prediction error
        # (~0.01 m/s held-out). Both features go through the same head here,
        # so the comparison isolates the feature difference.
        with torch.no_grad():
            a_run = pol.head((x - pol.mu) / pol.sd)
            a_ref = pol.head((ref - pol.mu) / pol.sd)
        gap = (a_run - a_ref).abs().max().item()
        good = cos > 0.999 and gap < 0.002
        ok &= good
        print(f"{feat}: min cosine {cos:.6f}, max action gap {gap:.5f} m/s "
              f"(head error ~0.01)  {'ok' if good else 'MISMATCH'}")

    for feat in ("vis", "llm"):
        pol.feature = feat
        big = img.repeat(4, 1, 1, 1)                          # 64 frames
        pol.features(big[:8])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pol.features(big)
        torch.cuda.synchronize()
        print(f"{feat}: {(time.perf_counter() - t0) * 1e3 / len(big):.1f} ms per frame at B=64")

    pol.feature = "vis"
    a = pol(obs, img)
    print(f"policy output {tuple(a.shape)}, |a| max {a.abs().max():.3f} m/s")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
