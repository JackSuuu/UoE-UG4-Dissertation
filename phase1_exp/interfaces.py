"""
Plug-in interfaces for the four swappable components of the Phase-1 system.

Every experiment script talks to components ONLY through these interfaces, so a
real component can replace a stand-in by writing one adapter class and
registering it in ``registry.py`` — no experiment code changes.

    Component          Interface             Stand-in (in repo)            Real target
    -----------------  --------------------  ----------------------------  ------------------------------
    Base policy (VLA)  PolicyAdapter         models.nets.BCPolicy (MLP)    OpenVLA / pi0 / ...
    Fast predictor     PredictorAdapter      models.nets.OrbiSimDynamics   OrbiSim-Dynamics (2605.16395)
                                             (distilled MLP ensemble)
    Baseline predictor PredictorAdapter      models.nets.VisionWM          CheckVLA's visual world model
    Verifier           VerifierAdapter       checkvla.reference.RefCheckVLA CheckVLA (2607.26789)
    GT simulator       sims.base.Env         sims.base.TorchEnv            sims.genesis_push.GenesisPushEnv
                                                                           (written, NOT yet run)

Shared conventions
------------------
B = batch of parallel episodes, H = chunk length, A = action dim, O = obs dim.

``ctx`` (verifier/predictor context) is a dict with keys
    obs (B,O)       object-centric state features (sim.obs)
    obs_prev (B,O)  previous step's obs
    a_prev (B,A)    previously executed action
    img, img_prev   (B,3,32,32) synthetic frames — only if some component sets needs_img
    rgb             (B,3,H,W) camera frames — only if the policy sets needs_rgb
    state (B,S)     raw simulator state (for predictors that re-simulate)
    instruction     str | None (language instruction for VLA policies)

``risk`` is always (B,H,2) = [force/force_limit, deform/deform_limit] per step;
> 1 in either channel is a constraint violation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PolicyAdapter(Protocol):
    H: int                  # chunk length the policy emits
    act_dim: int            # must equal sim.act_dim (adapter maps its native action space)
    needs_img: bool         # if True, the 32x32 synthetic sim.render() frame is passed as img
    needs_rgb: bool         # if True, env.render_rgb() camera frames are passed as img instead

    def __call__(self, obs: torch.Tensor, img: torch.Tensor | None = None,
                 instruction: str | None = None) -> torch.Tensor:
        """-> action chunk (B,H,A), already clipped to ±sim.max_vel."""
        ...


@runtime_checkable
class PredictorAdapter(Protocol):
    needs_img: bool
    differentiable: bool    # True -> risk() is differentiable w.r.t. actions

    def risk(self, ctx: dict, actions: torch.Tensor):
        """actions (B,H,A) -> (mean (B,H,2), std (B,H,2)); std=0 if no uncertainty."""
        ...


class RolloutPredictor(PredictorAdapter, Protocol):
    """Optional extension used by RQ1 fidelity / gradient-agreement metrics."""

    def rollout(self, obs, obs_prev, a_prev, actions):
        """-> (obs_seq (E,B,H,O), risk (E,B,H,2)); E = ensemble size (1 if none)."""
        ...


@runtime_checkable
class VerifierAdapter(Protocol):
    predictor: PredictorAdapter
    tau: float
    latency: int            # actions already committed while repairing
    commit: int             # steps the repaired suffix is executed open-loop

    def score(self, ctx: dict, chunk: torch.Tensor) -> torch.Tensor:
        """-> (B,) risk score; higher = riskier."""
        ...

    def check(self, ctx: dict, chunk: torch.Tensor):
        """-> (trigger mask (B,) bool, score (B,))."""
        ...

    def repair(self, ctx: dict, chunk: torch.Tensor) -> torch.Tensor:
        """Called only on triggered rows. -> repaired chunk (b,H,A) whose first
        ``latency`` actions equal the input's."""
        ...
