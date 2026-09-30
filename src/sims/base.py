"""
Simulator abstractions.

Two layers:

* ``FunctionalSim`` — a *pure, differentiable* ``step(state, action, params)``
  written in PyTorch. Used as (i) the ground-truth engine when Genesis is not
  available / not differentiable for a regime, and (ii) the substrate for the
  RQ3 systems experiments (checkpointed BPTT needs a re-playable pure step).

* ``Env`` — a *stateful*, batched environment API shared by the torch backend
  and the Genesis backend so every experiment script is backend-agnostic:

      obs = env.reset(params, seed)
      obs, risk, success = env.step(action)
      s = env.get_state(); env.set_state(s)
      risk_seq = env.shadow_rollout(chunk)     # non-destructive GT rollout

``risk`` is always a (B, 2) tensor ``[force / force_limit, deform / deform_limit]``
so a value > 1 in either channel is a constraint violation.
"""
from __future__ import annotations

import torch


class FunctionalSim:
    name: str = "base"
    state_dim: int
    obs_dim: int
    act_dim: int
    T: int              # episode length (control steps)
    max_vel: float
    param_names: tuple

    def __init__(self, device):
        self.device = torch.device(device)

    # --- to implement -----------------------------------------------------
    def nominal_params(self) -> dict:
        raise NotImplementedError

    def init_state(self, n: int, gen: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def step(self, s, a, params, smooth: bool = False):
        """Return (next_state, risk(B,2)). Must be differentiable."""
        raise NotImplementedError

    def obs(self, s) -> torch.Tensor:
        raise NotImplementedError

    def success(self, s) -> torch.Tensor:
        raise NotImplementedError

    def render(self, s, res: int = 32) -> torch.Tensor:
        raise NotImplementedError

    def expert(self, s, params=None, gain=None) -> torch.Tensor:
        raise NotImplementedError

    def task_cost(self, s) -> torch.Tensor:
        """Per-sample differentiable task cost of a state (lower is better)."""
        raise NotImplementedError

    def contact_regime(self, s) -> torch.Tensor:
        """Integer regime label per sample (for the gradient path audit)."""
        raise NotImplementedError

    regime_names: tuple = ()

    # --- helpers ------------------------------------------------------------
    def make_params(self, n: int, mults: dict) -> dict:
        """Multipliers (scalars or (n,) tensors) -> absolute param tensors."""
        nom = self.nominal_params()
        out = {}
        for k, v in nom.items():
            m = mults.get(k, 1.0)
            m = torch.as_tensor(m, dtype=torch.float32, device=self.device)
            out[k] = (v * m).expand(n).clone() if m.dim() == 0 else v * m
        return out

    def sample_params(self, n: int, ranges: dict, gen: torch.Generator) -> dict:
        mults = {}
        for k, (lo, hi) in ranges.items():
            u = torch.rand(n, generator=gen, device="cpu").to(self.device)
            mults[k] = lo + (hi - lo) * u
        return self.make_params(n, mults)


class Env:
    """Stateful batched env interface (see module docstring)."""

    sim: FunctionalSim
    n: int

    def reset(self, params: dict, seed: int = 0) -> torch.Tensor: ...
    def step(self, a: torch.Tensor): ...
    def get_state(self) -> torch.Tensor: ...
    def set_state(self, s: torch.Tensor): ...

    def obs(self):
        return self.sim.obs(self.get_state())

    def render(self):
        return self.sim.render(self.get_state())

    def render_rgb(self, res=None, ss=None, return_mat=False, rows=None):
        """Perspective RGB frames (B,3,h,w) float in [0,1] for a real VLA policy.

        Backed by :mod:`sims.camera`, which is a pure renderer: read-only on the
        state, so it cannot perturb the dynamics or the reported risk.
        ``return_mat`` additionally returns the (B,h,w) material map, for tests.
        ``rows`` renders only a subset of the batch, so the Controller can spend
        the render on the rows that are about to call the policy and reuse the
        cached frame elsewhere.
        """
        if self.cam is None:
            raise NotImplementedError(
                f"{type(self).__name__} was built without a camera; pass "
                "camera=True (registry --policy openvla does this).")
        st = self.get_state()
        if rows is not None:
            st = st[rows]
        return self.cam.render(st, res=res, ss=ss, return_mat=return_mat, rows=rows)

    @torch.no_grad()
    def shadow_rollout(self, chunk: torch.Tensor, mask=None) -> torch.Tensor:
        """Non-destructive GT rollout of an action chunk (B,H,A) -> risk (B,H,2).

        ``mask`` restricts the rollout to a subset of the batch: a cheaper and
        cleaner way to audit an intervention than cloning the whole environment
        for one env. The subset is stepped on its own copy of the state, so the
        live state is never touched. Without a mask the whole-batch form is used,
        which snapshots and restores.
        """
        if mask is not None:
            st = self.get_state()
            sub = st[mask] if torch.is_tensor(st) else \
                {k: v[mask] for k, v in st.items()}
            pr = self.params
            if isinstance(pr, dict):
                pr = {k: (v[mask] if torch.is_tensor(v) and v.dim() > 0
                          and v.shape[0] == self.n else v) for k, v in pr.items()}
            risks = []
            for h in range(chunk.shape[1]):
                sub, r = self.sim.step(sub, chunk[:, h], pr)
                risks.append(r)
            return torch.stack(risks, 1)
        snap = self.get_state().clone()
        risks = []
        for h in range(chunk.shape[1]):
            _, r, _ = self.step(chunk[:, h])
            risks.append(r)
        self.set_state(snap)
        return torch.stack(risks, 1)


class TorchEnv(Env):
    def __init__(self, sim: FunctionalSim, n: int, camera: bool = False,
                 cam_res: int = 224, cam_ss: int = 1, cam_seed: int = 0):
        self.sim, self.n = sim, n
        self.state = None
        self.params = None
        # The camera is a pure renderer (sims/camera.py): it reads the state and
        # returns pixels, so switching it on cannot move any physics result. It
        # is off by default because a VLA is the only consumer and rendering
        # every step would be pure waste for every other arm.
        self.cam = None
        if camera:
            from sims.camera import PushCamera
            self.cam = PushCamera(sim, sim.device, res=cam_res, ss=cam_ss,
                                  seed=cam_seed, n=n)

    def reset(self, params, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.params = params
        self.state = self.sim.init_state(self.n, g)
        if self.cam is not None:
            # same generator, so one seed fixes the appearance along with the
            # initial states and a rerun reproduces the pixels exactly
            self.cam.reset_appearance(self.n, g)
        return self.sim.obs(self.state)

    @torch.no_grad()
    def step(self, a):
        self.state, risk = self.sim.step(self.state, a, self.params)
        return self.sim.obs(self.state), risk, self.sim.success(self.state)

    def get_state(self):
        return self.state

    def set_state(self, s):
        self.state = s.clone()


def make_sim(task: str, device):
    if task == "push":
        from sims.torch_push import PushSim
        return PushSim(device)
    if task == "cloth":
        from sims.torch_cloth import ClothSim
        return ClothSim(device)
    raise ValueError(task)


def make_env(task: str, backend: str, n: int, device, mults: dict | None = None,
             camera: bool = False, cam_res: int = 224, cam_ss: int = 1,
             cam_seed: int = 0):
    """Build a GT environment. ``mults`` = physical-parameter multipliers.

    Genesis scenes bake physical parameters in at build time, so the Genesis
    env is built per OOD cell; the torch env takes params at reset().
    ``camera*`` configure the pure RGB renderer (see sims/camera.py); the torch
    camera follows the episode seed, so ``cam_seed`` only matters before the
    first ``reset``.
    """
    sim = make_sim(task, device)
    if backend == "torch":
        return TorchEnv(sim, n, camera=camera, cam_res=cam_res, cam_ss=cam_ss,
                        cam_seed=cam_seed)
    if backend == "genesis":
        if task != "push":
            raise NotImplementedError(
                "Genesis backend is implemented for Task A (push/insertion). "
                "Task B cloth uses the torch mass-spring GT (Genesis PBD cloth is "
                "not differentiable; see audit_gradients.py probe).")
        from sims.genesis_push import GenesisPushEnv
        return GenesisPushEnv(sim, n, mults or {}, camera=camera)
    raise ValueError(backend)
