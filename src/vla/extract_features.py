"""Run the frozen OpenVLA-7B once over the demo frames and cache its features.

The backbone is frozen, so its features are the same in every epoch. Running a
forward pass per epoch costs ~100 ms per image, more than 10 h for a few epochs
over 120k frames. Running it once and training the head on the cache takes
minutes. Two features are cached so the choice of readout can be measured
rather than assumed:

  llm  last-layer LLM hidden state at the final prompt token (image + text fused)
  vis  mean of the 256 projected image patches (vision only, pre-LLM)

Frames are subsampled with ``--stride`` on the step axis and only steps that
have a full H-step future are kept, so each cached frame has a valid chunk
target. Images go through OpenVLA's own processor (normalisation and the
6-channel DINOv2/SigLIP stacking); feeding raw pixels would be silently wrong.

    cd src && CUDA_VISIBLE_DEVICES=0 setsid nohup python -u vla/extract_features.py \
        > ~/scratch/extract_features.log 2>&1 < /dev/null &
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)

from adapters import tf5_compat  # noqa: E402

PROMPT = ("In: What action should the robot take to push the block into the green "
          "seat without hitting the wall?\nOut:")


def load_model(dev):
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
    return model.to(dev).eval(), tf5_compat.load_processor(snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="~/scratch/openvla_push_demos.h5")
    ap.add_argument("--out", default="~/scratch/openvla_push_features.h5")
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    src = h5py.File(os.path.expanduser(args.demos), "r")
    ep_all, st_all, act_all = src["episode"][:], src["step"][:], src["act"][:]
    T = int(src.attrs["steps_per_episode"])
    keep = (st_all % args.stride == 0) & (st_all <= T - args.H)
    if args.max_episodes is not None:
        keep &= ep_all < args.max_episodes
    idx = np.nonzero(keep)[0]
    # chunk target: actions at steps t..t+H-1 of the same episode (rows are contiguous)
    tgt = np.stack([act_all[idx + k] for k in range(args.H)], axis=1)
    assert np.all(ep_all[idx + args.H - 1] == ep_all[idx]), "chunk crosses an episode"
    print(f"{len(idx)} frames from {np.unique(ep_all[idx]).size} episodes, "
          f"H={args.H}, stride={args.stride}", flush=True)

    model, proc = load_model(dev)
    out = os.path.expanduser(args.out)
    with h5py.File(out, "w") as f:
        f.create_dataset("llm", (len(idx), 4096), dtype=np.float16)
        f.create_dataset("vis", (len(idx), 4096), dtype=np.float16)
        f.create_dataset("target", data=tgt.astype(np.float32))
        f.create_dataset("episode", data=ep_all[idx])
        f.create_dataset("step", data=st_all[idx])
        f.attrs.update(H=args.H, stride=args.stride, prompt=PROMPT)

    t0 = time.time()
    for s in range(0, len(idx), args.batch):
        rows = idx[s:s + args.batch]
        imgs = [Image.fromarray(src["obs"][r]) for r in rows]
        inp = proc([PROMPT] * len(rows), imgs).to(dev)
        ids = tf5_compat.prepare_prompt_ids(inp["input_ids"][:1]).repeat(len(rows), 1)
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      pixel_values=inp["pixel_values"].to(torch.bfloat16),
                      use_cache=False, output_hidden_states=True,
                      output_projector_features=True)
        llm = o.hidden_states[-1][:, -1].float().cpu().numpy()
        vis = o.projector_features.float().mean(1).cpu().numpy()
        with h5py.File(out, "a") as f:
            f["llm"][s:s + len(rows)] = llm
            f["vis"][s:s + len(rows)] = vis
        done = s + len(rows)
        if (s // args.batch) % 100 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(idx)}  {el:.0f}s elapsed, "
                  f"~{el / done * (len(idx) - done) / 60:.0f} min left", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
