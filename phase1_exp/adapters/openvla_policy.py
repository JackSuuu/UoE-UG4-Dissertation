"""
Real-VLA policy adapter (PolicyAdapter, interfaces.py) — SKELETON, NOT TESTED.

Default target: OpenVLA-7B via HuggingFace (github.com/openvla/openvla).
The loading / predict_action calls follow the OpenVLA README; verify against
the version you install.

What you must decide / fill in on the server (marked TODO):
  1. Camera: a real VLA needs RGB frames (e.g. 224x224). Our sims' ``render()``
     is a 32x32 synthetic image for the vision-WM baseline, so the env must
     provide ``render_rgb()`` (GenesisPushEnv has a camera hook, see there).
  2. Action mapping: OpenVLA outputs a single 7-D end-effector delta
     [dx,dy,dz,droll,dpitch,dyaw,gripper] (un-normalised with ``unnorm_key``).
     Our tasks expect velocities: push -> (vx,vy), cloth -> (vx,vy,vz).
     ``_map_action`` converts delta-position / control-dt -> velocity; the scale
     and axis convention depend on the camera frame and the fine-tuning data.
  3. Chunking: OpenVLA has no action chunks. We repeat the single action H
     times so the verifier can score a chunk (replace with a chunked VLA such as
     pi0 / OpenVLA-OFT if available — then return its native chunk).
  4. Fine-tuning: zero-shot OpenVLA will not solve these tasks; fine-tune on
     demos collected with ``collect_data.py`` rendered through ``render_rgb()``.
"""
from __future__ import annotations

import numpy as np
import torch


class OpenVLAPolicy:
    needs_img = False
    needs_rgb = True

    def __init__(self, sim, device, H=10, model_id="openvla/openvla-7b",
                 unnorm_key="bridge_orig", instruction=None, action_scale=1.0):
        from transformers import AutoModelForVision2Seq, AutoProcessor  # lazy import
        self.sim, self.dev, self.H = sim, torch.device(device), H
        self.act_dim = sim.act_dim
        self.unnorm_key = unnorm_key
        self.action_scale = action_scale
        self.default_instruction = instruction or {
            "push": "push the cylinder into the slot",
            "cloth": "fold the cloth corner to the opposite corner",
        }[sim.name]
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).to(self.dev)
        self.vla.eval()

    def _map_action(self, a7: np.ndarray) -> torch.Tensor:
        # TODO(server): verify axis convention / scale for your camera + fine-tune data.
        vel = torch.as_tensor(a7[: self.act_dim], dtype=torch.float32) / self.sim.dt
        return (vel * self.action_scale).clamp(-self.sim.max_vel, self.sim.max_vel)

    @torch.no_grad()
    def __call__(self, obs, img=None, instruction=None):
        if img is None:
            raise RuntimeError("OpenVLAPolicy needs RGB frames: env.render_rgb() must be available")
        from PIL import Image
        instr = instruction or self.default_instruction
        prompt = f"In: What action should the robot take to {instr}?\nOut:"
        acts = []
        for b in range(img.shape[0]):                    # OpenVLA is not batched here
            frame = img[b]
            if frame.dtype != torch.uint8:
                frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
            pil = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
            inputs = self.processor(prompt, pil).to(self.dev, dtype=torch.bfloat16)
            a7 = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
            acts.append(self._map_action(np.asarray(a7)))
        a = torch.stack(acts).to(obs.device)
        return a[:, None].repeat(1, self.H, 1)            # TODO: native chunk if available
