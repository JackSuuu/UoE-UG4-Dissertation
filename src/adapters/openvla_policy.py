"""OpenVLA-7B as a chunk policy for Controller.act.

Two action modes:

  planar_head (supported)  frozen OpenVLA backbone -> cached-style feature ->
                           ChunkHead trained by vla/train_chunk_head.py ->
                           (B, H, 2) world-frame pusher velocity in m/s, the
                           sim's own action. No mapping, no constants.
  native_head (ablation)   OpenVLA's 7-D Bridge end-effector delta, mapped with
                           explicit axes / metres_per_unit / control_dt. No
                           defaults: a guessed constant is a silent bug.

The planar feature must be computed exactly as vla/extract_features.py computed
it for training. That means the same prompt, the 29871 terminator, the same
readout (llm: last-layer hidden state at the last token; vis: mean projected
patch), and the same normalisation, all read from the checkpoint. The head class
is imported from the trainer, not redefined here, so the two cannot drift.

OpenVLA's own predict_action/generate is never used. On transformers 5.x it
returns one constant token for every input (tests/test_decode_agreement.py).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Pretrained 7-D EE deltas are [dx, dy, dz, droll, dpitch, dyaw, gripper].
EE_DELTA_ORDER = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


class ActionMapError(RuntimeError):
    """Raised when an action convention is used without being pinned down."""


class OpenVLAPolicy:
    needs_img = False
    needs_rgb = True

    def __init__(self, sim, device, H=5, head_path="~/scratch/openvla_chunk_head.pt",
                 instruction=None, action_mode="planar_head", axes=("dx", "dy"),
                 metres_per_unit=None, control_dt=None, unnorm_key="bridge_orig",
                 **_ignored):
        # Validate the cheap arguments before touching a 15 GB checkpoint, so a
        # wrong convention fails in microseconds (and tests run without a GPU).
        self.sim, self.dev, self.H = sim, torch.device(device), H
        self.act_dim = sim.act_dim
        if action_mode not in ("planar_head", "native_head"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        if action_mode == "native_head":
            for ax in axes:
                if ax not in EE_DELTA_ORDER:
                    raise ValueError(f"unknown EE axis {ax!r}")
            if metres_per_unit is None or control_dt is None:
                raise ActionMapError(
                    "action_mode='native_head' requires an explicit metres_per_unit "
                    f"and control_dt (got {metres_per_unit!r}, {control_dt!r}). Pin "
                    "them with tests/test_action_map.py, or use planar_head.")
        else:
            self.act_dim = 2
            head_path = os.path.expanduser(head_path)
            if not os.path.exists(head_path):
                raise FileNotFoundError(
                    f"no chunk head at {head_path}; run vla/extract_features.py "
                    "then vla/train_chunk_head.py")
        self.action_mode, self.axes = action_mode, axes
        self.metres_per_unit, self.control_dt = metres_per_unit, control_dt
        self.unnorm_key = unnorm_key

        from vla.extract_features import PROMPT, load_model
        self.prompt = PROMPT
        if instruction is not None:
            self.prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        self.model, self.proc = load_model(self.dev)
        for p in self.model.parameters():
            p.requires_grad_(False)

        if action_mode == "planar_head":
            from vla.train_chunk_head import ChunkHead
            ck = torch.load(head_path, map_location="cpu")
            self.feature = ck["feature"]
            self.head = ChunkHead(ck["feat_mu"].numel(), ck["H"]).to(self.dev).eval()
            self.head.load_state_dict(ck["head_state"])
            self.mu = ck["feat_mu"].to(self.dev)
            self.sd = ck["feat_sd"].to(self.dev)
            if ck["H"] < H:
                raise ActionMapError(f"head emits {ck['H']} steps, chunk needs {H}")

    # -- planar_head -------------------------------------------------------
    @torch.no_grad()
    def features(self, img):
        """(B,3,h,w) float [0,1] -> (B,4096), computed as in extract_features.py."""
        from PIL import Image
        from adapters import tf5_compat
        frames = (img.clamp(0, 1) * 255).round().to(torch.uint8)
        frames = frames.permute(0, 2, 3, 1).cpu().numpy()
        pil = [Image.fromarray(f) for f in frames]
        inp = self.proc([self.prompt] * len(pil), pil).to(self.dev)
        ids = tf5_compat.prepare_prompt_ids(inp["input_ids"][:1]).repeat(len(pil), 1)
        o = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       pixel_values=inp["pixel_values"].to(torch.bfloat16),
                       use_cache=False, output_hidden_states=True,
                       output_projector_features=True)
        if self.feature == "llm":
            return o.hidden_states[-1][:, -1].float()
        return o.projector_features.float().mean(1)

    # -- native_head -------------------------------------------------------
    def _map_action(self, a7) -> torch.Tensor:
        """Pretrained 7-D EE delta -> (..., act_dim) world-frame velocity."""
        if self.action_mode != "native_head":
            raise ActionMapError(
                "_map_action is only used by action_mode='native_head'; with "
                "'planar_head' the network already emits the sim's action.")
        a7 = torch.as_tensor(a7, dtype=torch.float32, device=getattr(self, "dev", None))
        if a7.dim() == 0 or a7.shape[-1] < len(EE_DELTA_ORDER):
            raise ActionMapError(
                f"expected a {len(EE_DELTA_ORDER)}-D action, got {a7.shape[-1]}")
        idx = [EE_DELTA_ORDER.index(ax) for ax in self.axes]
        if len(idx) != self.act_dim:
            raise ActionMapError(
                f"axes {self.axes!r} give {len(idx)} components but the task "
                f"needs act_dim={self.act_dim}")
        vel = a7[..., idx] * self.metres_per_unit / self.control_dt
        return vel.clamp(-self.sim.max_vel, self.sim.max_vel)

    @torch.no_grad()
    def __call__(self, obs, img=None, instruction=None):
        if img is None:
            raise RuntimeError("OpenVLAPolicy needs RGB frames: env.render_rgb()")
        if self.action_mode == "planar_head":
            x = (self.features(img) - self.mu) / self.sd
            a = self.head(x)[:, : self.H]
        else:
            # Single action per image via the reference decode, repeated over the
            # chunk. Kept only as an ablation: a repeated action has no chunk
            # variance, so the verifier has nothing to choose between.
            from adapters import tf5_compat
            from PIL import Image
            stats = self.model.config.norm_stats[self.unnorm_key]["action"]
            acts = []
            for f in (img.clamp(0, 1) * 255).round().to(torch.uint8):
                inp = self.proc(self.prompt, Image.fromarray(
                    f.permute(1, 2, 0).cpu().numpy())).to(self.dev)
                tok = tf5_compat.greedy_decode(
                    self.model, inp["input_ids"], inp["pixel_values"].to(torch.bfloat16))
                bins = tf5_compat.tokens_to_bins(tok.cpu().numpy(), self.model.vocab_size)
                a7 = tf5_compat.bins_to_action(bins, stats["q01"], stats["q99"],
                                               stats.get("mask"))
                acts.append(self._map_action(a7).repeat(self.H, 1))
            a = torch.stack(acts)
        return a.clamp(-self.sim.max_vel, self.sim.max_vel).to(obs.device)


# Old name kept so registry/imports keep working.
OpenVLAChunkPolicyWrapper = OpenVLAPolicy
