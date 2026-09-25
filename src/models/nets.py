"""
Learned components.

STAND-INS. These are the in-repo placeholder implementations of the plug-in
interfaces in ``interfaces.py``; replace them via ``registry.py``.

* ``OrbiSimDynamics`` — STAND-IN for OrbiSim-Dynamics: fast, differentiable, *action-conditioned, object-centric*
  dynamics predictor (plan §3.2). An ensemble of E MLPs predicts the next
  object-centric state delta and the per-step constraint-risk signal. It is
  distilled from ground-truth rollouts (Genesis or the torch GT). Ensemble
  disagreement is the confidence signal used by CheckVLA (confidence collapse).

  NOTE: this is the pragmatic Phase-1 instantiation of the OrbiSim-Dynamics
  role (Li et al., arXiv:2605.16395) — a distilled, object-centric,
  end-to-end-differentiable predictor — not a re-implementation of the paper's
  architecture.

* ``VisionWM`` — STAND-IN for CheckVLA's visual world model: the baseline predictor (plan §3.7): a pixel-based,
  (optionally) action-conditioned latent world model with a risk read-out, in
  the spirit of CheckVLA's visual world model (Liu et al., arXiv:2607.26789).
  ``action_conditioned=False`` gives the observation-only ablation.

* ``BCPolicy`` — STAND-IN for a real VLA: VLA proxy: obs -> action chunk (H x A), trained by behaviour
  cloning at *nominal* physics only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x):
        x = x.reshape(-1, x.shape[-1])
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp(min=1e-4))

    def forward(self, x):
        return (x - self.mean) / self.std

    def inv(self, x):
        return x * self.std + self.mean


class EnsembleLinear(nn.Module):
    def __init__(self, E, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(E, i, o) / math.sqrt(i))
        self.b = nn.Parameter(torch.zeros(E, 1, o))

    def forward(self, x):            # x: (E, B, i)
        return torch.baddbmm(self.b, x, self.w)


class OrbiSimDynamics(nn.Module):
    # PredictorAdapter attributes (interfaces.py)
    needs_img = False
    differentiable = True

    @property
    def obs_scale(self):
        return self.dobs_n.std

    def __init__(self, obs_dim, act_dim, E=5, hidden=256, max_vel=1.0):
        super().__init__()
        self.obs_dim, self.act_dim, self.E = obs_dim, act_dim, E
        self.max_vel = max_vel
        i = 2 * obs_dim + 2 * act_dim
        self.l1 = EnsembleLinear(E, i, hidden)
        self.l2 = EnsembleLinear(E, hidden, hidden)
        self.l3 = EnsembleLinear(E, hidden, hidden)
        self.out = EnsembleLinear(E, hidden, obs_dim + 2)
        self.obs_n = Normalizer(obs_dim)
        self.dobs_n = Normalizer(obs_dim)

    def _one(self, o, o_prev, a_prev, a):
        """o etc: (E,B,*) -> (d_obs (E,B,O), risk (E,B,2))."""
        x = torch.cat([self.obs_n(o), (o - o_prev) / self.dobs_n.std,
                       a_prev / self.max_vel, a / self.max_vel], -1)
        h = F.silu(self.l1(x))
        h = F.silu(self.l2(h)) + h
        h = F.silu(self.l3(h)) + h
        y = self.out(h)
        d_obs = y[..., : self.obs_dim] * self.dobs_n.std + self.dobs_n.mean
        risk = F.softplus(y[..., self.obs_dim:])
        return d_obs, risk

    def rollout(self, obs, obs_prev, a_prev, actions):
        """obs/obs_prev: (B,O), a_prev: (B,A), actions: (B,H,A)
        -> obs_seq (E,B,H,O) [obs_{t+1..t+H}], risk (E,B,H,2)."""
        E = self.E
        o = obs[None].expand(E, -1, -1)
        op = obs_prev[None].expand(E, -1, -1)
        ap = a_prev[None].expand(E, -1, -1)
        os_, rs = [], []
        for h in range(actions.shape[1]):
            a = actions[:, h][None].expand(E, -1, -1)
            d, r = self._one(o, op, ap, a)
            op, o, ap = o, o + d, a
            os_.append(o)
            rs.append(r)
        return torch.stack(os_, 2), torch.stack(rs, 2)

    def risk(self, ctx, actions):
        _, r = self.rollout(ctx["obs"], ctx["obs_prev"], ctx["a_prev"], actions)
        return r.mean(0), r.std(0)


class _VisionNet(nn.Module):
    def __init__(self, act_dim, hidden=256, action_conditioned=True):
        super().__init__()
        self.ac = action_conditioned
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 16, hidden), nn.SiLU())
        self.cell = nn.GRUCell(act_dim if self.ac else 1, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, img, img_prev, actions):
        h = self.enc(torch.cat([img, img_prev], 1))
        out = []
        for k in range(actions.shape[1]):
            u = actions[:, k] if self.ac else torch.zeros_like(actions[:, k, :1])
            h = self.cell(u, h)
            out.append(F.softplus(self.head(h)))
        return torch.stack(out, 1)


class VisionWM(nn.Module):
    needs_img = True
    differentiable = True

    def __init__(self, act_dim, E=3, hidden=256, action_conditioned=True, max_vel=1.0):
        super().__init__()
        self.E, self.max_vel = E, max_vel
        self.action_conditioned = action_conditioned
        self.nets = nn.ModuleList([_VisionNet(act_dim, hidden, action_conditioned)
                                   for _ in range(E)])

    def forward(self, img, img_prev, actions):
        a = actions / self.max_vel
        return torch.stack([n(img, img_prev, a) for n in self.nets], 0)   # (E,B,H,2)

    def risk(self, ctx, actions):
        r = self(ctx["img"], ctx["img_prev"], actions)
        return r.mean(0), r.std(0)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, H, hidden=256, max_vel=1.0):
        super().__init__()
        self.H, self.act_dim, self.max_vel = H, act_dim, max_vel
        self.obs_n = Normalizer(obs_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, H * act_dim))

    def forward(self, obs):
        y = self.net(self.obs_n(obs)).view(-1, self.H, self.act_dim)
        return self.max_vel * torch.tanh(y)


# ---------------------------------------------------------------------------
def save_model(model, path, **meta):
    torch.save({"state": model.state_dict(), "meta": meta}, path)


def load_model(cls, path, device, **kw):
    ck = torch.load(path, map_location=device)
    kw = {**ck["meta"], **kw}
    m = cls(**kw).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m
