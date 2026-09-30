"""
Real-VLA policy adapter (PolicyAdapter, interfaces.py) -- SKELETON, NOT TESTED.

Status 30 Sep, after the camera landed. Two of the four original TODOs are
closed, and the important remaining decision is now a decision rather than a
mystery:

  1. Camera: **DONE.** ``sims/camera.py`` gives the torch GT a perspective RGB
     camera (224x224 by default), so ``env.render_rgb()`` works without Genesis.
     Measured useful, not just present: held-out R^2 0.974 for the peg's x and
     0.974 for the clearance to the end-stop that the constraint depends on
     (src/tests/test_camera_observability.py). It is a pure renderer, so it
     cannot perturb the dynamics.
  2. Action mapping: **do not guess it -- remove it.** See the note below; this
     is the one substantive change in this file.
  3. Chunking: still required, and it is not optional for this thesis. A policy
     that emits one action and repeats it has ~zero chunk variance, so there is
     no chunk for the verifier to choose between and the mechanism under test
     never engages. Use pi0 or OpenVLA-OFT, which emit chunks natively, and
     return the native chunk.
  4. Fine-tuning: unchanged. Zero-shot OpenVLA will not solve this; collect
     demos with the scripted expert through the camera and fine-tune.

Why the action mapping should be removed rather than resolved
------------------------------------------------------------
This sim's action is a **world-frame pusher velocity** (vx, vy), clipped to
+-0.5 m/s and integrated as ``p += a * dt`` with dt = 0.05 s -- so one control
step at full speed moves the pusher 25 mm, about 40% of the 60 mm peg. A 7-D
end-effector-delta action space has a scale, an axis order and an unnormalisation
key that are all properties of the *pretraining data*, not of the task. Mapping
one to the other means guessing three coupled constants, and a wrong guess is
not a crash: it produces a policy that looks trained, moves in roughly the
right direction, and plateaus at some arbitrary success rate that then gets
attributed to the method.

So: keep the VLA's vision and language, and replace its action head with a
**2-D world-frame planar-velocity head**, fine-tuned on demos from the scripted
expert. The convention is then fixed by construction -- it is the same
convention the expert and the ``bc`` stand-in already use, so every arm speaks
the same action language and the verifier is untouched. This is also what makes
the comparison fair: arms differ in how the action is *chosen*, not in what an
action means.

The alternative, if a native 7-D head has to be kept, is ``_map_action`` below
made explicit and configurable rather than a bare TODO: an axis permutation, a
metres-per-unit scale, and a control dt, all named arguments, all asserted
against a measured demo (replay one expert action through the map and check the
pusher tracks). Guessing silently is the failure mode; a stated convention with
a test that pins it is not.
"""
from __future__ import annotations

import numpy as np
import torch

# Pretrained 7-D EE deltas are [dx, dy, dz, droll, dpitch, dyaw, gripper]. The
# first two are the only ones a planar task can use; keeping the permutation
# explicit rather than assuming a7[:2] is the sim's (x, y) is the whole point of
# the TODO this replaced.
EE_DELTA_ORDER = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


class ActionMapError(RuntimeError):
    """Raised when an action convention is used without being pinned down."""


class OpenVLAPolicy:
    """Adapter skeleton. Requires a 2-D planar-velocity action head.

    This class deliberately refuses to guess. ``action_mode="native_head"`` with
    an explicit, tested convention is the supported path; anything else raises,
    because a silently mis-scaled action is the failure mode that costs a
    semester.
    """

    needs_img = False
    needs_rgb = True

    def __init__(self, sim, device, H=5, model_id="openvla/openvla-7b",
                 unnorm_key="bridge_orig", instruction=None,
                 action_mode="planar_head", axes=("dx", "dy"),
                 metres_per_unit=None, control_dt=None):
        # Validate the cheap arguments *before* the lazy transformers import, so
        # a wrong convention is rejected in microseconds rather than after a 7B
        # checkpoint has been read off disk. That ordering is what makes the
        # guard in tests/test_action_map.py runnable without transformers.
        self.sim, self.dev, self.H = sim, torch.device(device), H
        self.act_dim = sim.act_dim
        self.unnorm_key = unnorm_key
        if action_mode not in ("planar_head", "native_head"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        if action_mode == "planar_head":
            # The 2-D head outputs the sim's own action directly: world-frame
            # pusher velocity in m/s, clamped by the sim. No mapping, no scale,
            # no axis question -- and therefore no way for a normalisation
            # constant to silently cap the policy.
            self.act_dim = 2
        else:
            for ax in axes:
                if ax not in EE_DELTA_ORDER:
                    raise ValueError(f"unknown EE axis {ax!r}")
            # No defaults here, deliberately. These two are properties of the
            # *pretraining dataset*, not of the task, so a default would be a
            # guess -- and a wrong guess is not a crash, it is a policy that
            # trains, moves plausibly and plateaus at an arbitrary success rate
            # which then gets attributed to the verifier. The supported path,
            # planar_head, needs neither.
            if metres_per_unit is None or control_dt is None:
                raise ActionMapError(
                    "action_mode='native_head' requires an explicit "
                    f"metres_per_unit and control_dt (got {metres_per_unit!r}, "
                    f"{control_dt!r}). Read them off the dataset the checkpoint "
                    "was trained on -- OpenVLA's bridge unnormalised actions are "
                    "already metres, so metres_per_unit=1.0 -- and pin them with "
                    "tests/test_action_map.py against a replayed expert demo. "
                    "Or use action_mode='planar_head', which needs no constants.")
        self.action_mode = action_mode
        self.axes, self.metres_per_unit, self.control_dt = axes, metres_per_unit, control_dt
        self.default_instruction = instruction or {
            "push": "push the block into the green seat without hitting the wall",
            "cloth": "fold the cloth corner to the opposite corner",
        }[sim.name]
        # Everything above is free. Only now pay for the checkpoint.
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).to(self.dev)
        self.vla.eval()

    def _map_action(self, a7) -> torch.Tensor:
        """Pretrained 7-D EE delta -> (B, act_dim) world-frame velocity.

        Only for ``action_mode="native_head"``. Every constant is an explicit
        constructor argument so it can be *pinned by a test* against a measured
        demo rather than discovered by watching a success rate plateau. The
        convention is: ``a7`` is a position delta in the pretraining dataset's
        units; multiply by ``metres_per_unit`` for metres, divide by
        ``control_dt`` for a velocity, take the axes named by ``axes``.
        """
        if self.action_mode != "native_head":
            raise ActionMapError(
                "_map_action is only used by action_mode='native_head'; with "
                "'planar_head' the network already emits the sim's action.")
        # Accept numpy arrays, CPU lists, or torch tensors on any device.
        # torch.as_tensor handles all three; going through np.asarray first would
        # reject a CUDA tensor outright, which is a trap for a caller who has
        # already batched the actions. On self.dev from the start, because
        # __call__ ends with a .to(obs.device) that would hide a CPU result here.
        a7 = torch.as_tensor(a7, dtype=torch.float32,
                             device=getattr(self, "dev", None))
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
        from PIL import Image
        instr = instruction or self.default_instruction
        prompt = f"In: What action should the robot take to {instr}?\nOut:"
        acts = []
        for b in range(img.shape[0]):                    # OpenVLA is not batched
            frame = img[b]
            if frame.dtype != torch.uint8:
                frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
            pil = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
            inputs = self.processor(prompt, pil).to(self.dev, dtype=torch.bfloat16)
            out = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key,
                                          do_sample=False)
            if self.action_mode == "planar_head":
                a = torch.as_tensor(np.asarray(out), dtype=torch.float32)  # (H,2)
                if a.dim() != 2 or a.shape[0] < self.H:
                    raise ActionMapError(
                        f"the 2-D head must emit a chunk of >= {self.H} actions, "
                        f"got shape {tuple(a.shape)}; if this model has no action "
                        "chunk, use pi0 / OpenVLA-OFT -- a repeated single action "
                        "has no chunk variance for the verifier to act on")
                acts.append(a[: self.H])
            else:
                acts.append(self._map_action(out))
        a = torch.stack(acts).to(obs.device)
        if a.shape[1] != self.H:
            raise ActionMapError(f"policy returned {a.shape[1]} actions, expected {self.H}")
        return a
