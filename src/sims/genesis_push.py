"""
Task A on Genesis (``genesis-world``) — GPU ground-truth backend.

Mirrors ``sims/torch_push.py`` exactly in geometry and in the *state layout*
``[bx, by, vx, vy, px, py, ty, t]`` so obs / render / success / expert are
re-used from ``PushSim`` and every experiment script is backend-agnostic.

Scene: plane + dynamic cylinder peg (free joint) + kinematically driven
sphere pusher (free joint, velocity set every sub-step) + fixed end-stop box.

Physical parameters (friction / mass multipliers) are baked in at build time,
so one ``GenesisPushEnv`` is built per OOD cell (n_envs parallel episodes).

Genesis API notes — checked against genesis-world 0.2.x/0.3.x. Anything that is
version-sensitive is wrapped with a fallback and logged, not silently ignored.
If your installed version differs, the functions to adapt are
``_build`` / ``_read`` / ``_write`` / ``_contact_force``.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from sims.base import Env

_GS_INITIALISED = False


def gs_init(requires_grad_hint: bool = False):
    global _GS_INITIALISED
    import genesis as gs
    if not _GS_INITIALISED:
        backend = gs.gpu if torch.cuda.is_available() else gs.cpu
        gs.init(backend=backend, precision="32", logging_level="warning")
        _GS_INITIALISED = True
    return gs


class GenesisPushEnv(Env):
    def __init__(self, sim, n: int, mults: dict, requires_grad: bool = False,
                 dt_sub: float = 0.01, camera: bool = False, cam_res=(224, 224)):
        self.camera, self.cam_res = camera, cam_res
        self.sim, self.n, self.mults = sim, n, dict(mults)
        self.requires_grad = requires_grad
        self.dt_sub = dt_sub
        self.n_sub = int(round(sim.dt / dt_sub))
        self.dev = sim.device
        nom = sim.nominal_params()
        self.mu = float(nom["friction"]) * mults.get("friction", 1.0)
        self.mass = float(nom["mass"]) * mults.get("mass", 1.0)
        self.params = sim.make_params(n, mults)
        self.t = torch.zeros(n, device=self.dev)
        self.ty = torch.zeros(n, device=self.dev)
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        gs = gs_init()
        s = self.sim
        self.gs = gs
        h = 0.03                              # peg height
        self.h = h
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt_sub, substeps=4,
                                              requires_grad=self.requires_grad),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane(),
                              material=gs.materials.Rigid(friction=self.mu))
        # density chosen so the cylinder has the requested mass
        vol = np.pi * s.R_b ** 2 * h
        self.peg = self.scene.add_entity(
            gs.morphs.Cylinder(radius=s.R_b, height=h, pos=(0.0, 0.0, h / 2)),
            material=gs.materials.Rigid(rho=self.mass / vol, friction=self.mu))
        self.pusher = self.scene.add_entity(
            gs.morphs.Sphere(radius=s.r_p, pos=(-0.05, 0.0, s.r_p)),
            material=gs.materials.Rigid(rho=5000.0, friction=0.01))
        self.wall = self.scene.add_entity(
            gs.morphs.Box(size=(0.02, 0.6, 0.06),
                          pos=(s.x_w + 0.01, 0.0, 0.03), fixed=True),
            material=gs.materials.Rigid(friction=self.mu))
        if self.camera:
            # top-down-ish camera for real-VLA policies (render_rgb)
            self.cam = self.scene.add_camera(res=self.cam_res, pos=(0.15, -0.45, 0.45),
                                             lookat=(0.15, 0.0, 0.0), fov=45, GUI=False)
        self.scene.build(n_envs=self.n)

    # ------------------------------------------------------------------
    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32).to(self.dev)

    def _read(self):
        pp = self._t(self.peg.get_pos())
        pv = self._t(self.peg.get_vel())
        qp = self._t(self.pusher.get_pos())
        return pp, pv, qp

    def _write(self, s):
        n = self.n
        z = torch.zeros(n, 1, device=self.dev)
        peg_pos = torch.cat([s[:, 0:2], z + self.h / 2], 1)
        pus_pos = torch.cat([s[:, 4:6], z + self.sim.r_p], 1)
        self.peg.set_pos(peg_pos)
        self.peg.set_quat(torch.tensor([1.0, 0, 0, 0], device=self.dev).repeat(n, 1))
        self.pusher.set_pos(pus_pos)
        vel = torch.zeros(n, 6, device=self.dev)
        vel[:, 0:2] = s[:, 2:4]
        self.peg.set_dofs_velocity(vel)
        self.pusher.set_dofs_velocity(torch.zeros(n, 6, device=self.dev))

    def _contact_force(self):
        """Horizontal net contact force magnitude on the peg (excludes floor normal)."""
        try:
            f = self._t(self.peg.get_links_net_contact_force())   # (n, links, 3)
            f = f.sum(1) if f.dim() == 3 else f
            return torch.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2)
        except Exception as e:                                   # pragma: no cover
            if not getattr(self, "_warned_cf", False):
                warnings.warn(f"contact force API unavailable ({e}); "
                              "using momentum-change proxy")
                self._warned_cf = True
            return None

    def render_rgb(self):
        """(B,3,H,W) uint8 camera frames. UNTESTED: Genesis cameras may render only
        env 0 in batched scenes depending on version — if so, loop over envs
        (set per-env state, render) or use n_envs=1 for VLA evaluation."""
        if not self.camera:
            raise RuntimeError("build GenesisPushEnv(camera=True) to use render_rgb()")
        rgb = self.cam.render(rgb=True)[0]
        rgb = torch.as_tensor(np.asarray(rgb)).to(self.dev)
        if rgb.dim() == 3:                                   # single env frame
            rgb = rgb[None].expand(self.n, -1, -1, -1)
        return rgb.permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------
    def get_state(self):
        pp, pv, qp = self._read()
        return torch.stack([pp[:, 0], pp[:, 1], pv[:, 0], pv[:, 1],
                            qp[:, 0], qp[:, 1], self.ty, self.t], 1)

    def set_state(self, s):
        self.ty = s[:, 6].clone()
        self.t = s[:, 7].clone()
        self._write(s)

    def reset(self, params=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        s0 = self.sim.init_state(self.n, g)
        self.scene.reset()
        self.set_state(s0)
        return self.sim.obs(s0)

    def step(self, a, record_grad: bool = False):
        a = a.clamp(-self.sim.max_vel, self.sim.max_vel)
        fmax = torch.zeros(self.n, device=self.dev)
        v_prev = None
        for _ in range(self.n_sub):
            vel = torch.zeros(self.n, 6, device=self.dev)
            vel[:, 0:2] = a
            self.pusher.set_dofs_velocity(vel)
            self.scene.step()
            f = self._contact_force()
            if f is None:
                _, pv, _ = self._read()
                if v_prev is not None:
                    f = self.mass * torch.norm(pv[:, :2] - v_prev, dim=1) / self.dt_sub
                else:
                    f = torch.zeros_like(fmax)
                v_prev = pv[:, :2].clone()
            fmax = torch.maximum(fmax, f)
        self.t = self.t + 1
        s = self.get_state()
        risk = torch.stack([fmax / self.sim.force_limit, torch.zeros_like(fmax)], 1)
        return self.sim.obs(s), risk, self.sim.success(s)


# ---------------------------------------------------------------------------
# Differentiability probe (used by experiments/audit_gradients.py & rq3)
# ---------------------------------------------------------------------------
def genesis_grad_probe(sim, mults: dict, horizon: int, n: int = 1, seed: int = 0):
    """Try to backprop a task loss through a Genesis rigid rollout.

    Returns dict(status in {valid, zero, nan, error}, grad_norm, msg, peak_mem_mb).
    The gradient is taken w.r.t. the peg's initial velocity (the pattern used in
    Genesis's differentiable-simulation docs). Unsupported paths raise or
    return zeros — both are recorded, which *is* the audit result.
    """
    out = {"status": "error", "grad_norm": float("nan"), "msg": "", "peak_mem_mb": float("nan")}
    try:
        gs = gs_init()
        env = GenesisPushEnv(sim, n, mults, requires_grad=True)
        env.reset(seed=seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        v0 = gs.tensor(np.tile([[0.3, 0.0, 0, 0, 0, 0]], (n, 1)).astype(np.float32),
                       requires_grad=True)
        env.peg.set_dofs_velocity(v0)
        for _ in range(horizon):
            vel = gs.tensor(np.tile([[0.3, 0.0, 0, 0, 0, 0]], (n, 1)).astype(np.float32))
            env.pusher.set_dofs_velocity(vel)
            env.scene.step()
        pos = env.peg.get_pos()
        loss = ((pos[:, 0] - sim.target_x) ** 2).sum()
        loss.backward()
        g = v0.grad
        if g is None:
            out.update(status="zero", grad_norm=0.0, msg="grad is None")
        else:
            gn = float(torch.as_tensor(g).norm())
            out["grad_norm"] = gn
            out["status"] = "nan" if not np.isfinite(gn) else ("zero" if gn < 1e-10 else "valid")
        if torch.cuda.is_available():
            out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    except Exception as e:
        out["msg"] = f"{type(e).__name__}: {e}"[:300]
    return out
