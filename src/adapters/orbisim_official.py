"""
Real OrbiSim-Dynamics adapter (PredictorAdapter, interfaces.py) — SKELETON.

Target: OrbiSim (Li et al., arXiv:2605.16395), OrbiSim-Dynamics module.
I have NOT checked whether code/weights are public or what their API is, so
everything below is a template with the exact contract the rest of the
pipeline needs. Fill in the TODOs on the server.

Contract (what the verifier / RQ1 need from this class)
-------------------------------------------------------
risk(ctx, actions)  -> (mean (B,H,2), std (B,H,2))
    Normalised constraint risk per future step for the given action chunk:
    channel 0 = contact force / sim.force_limit, channel 1 = deformation /
    sim.deform_limit (0 for the rigid task). Must be differentiable w.r.t.
    ``actions`` if ``differentiable=True`` (used by gradient suffix repair).
rollout(obs, obs_prev, a_prev, actions) -> (obs_seq (E,B,H,O), risk (E,B,H,2))
    Optional — only for RQ1 fidelity / gradient-agreement. ``obs_seq`` must be
    in the same feature space as ``sim.obs`` so it can be compared to GT.
obs_scale (O,)  optional per-feature scale used to normalise fidelity errors.

Two ways to get ``risk`` out of a physics-state predictor
---------------------------------------------------------
(a) Analytic: predict future physical state with OrbiSim-Dynamics, then
    compute force / strain from it (e.g. contact penetration x stiffness,
    spring strain from predicted particle positions). Implement in
    ``_risk_from_states``.
(b) Learned head: freeze OrbiSim-Dynamics and train a small risk head on its
    latent/state rollouts using ``dyn_data.pt`` (same data and loss as
    ``train.py::train_orbisim`` risk term). Keeps RQ1 comparison fair.
"""
from __future__ import annotations

import os

import torch


class OrbiSimOfficial:
    needs_img = False          # set True if you feed OrbiSim-Vision with frames
    differentiable = True

    def __init__(self, sim, device, ckpt=None, repo=None):
        self.sim, self.dev = sim, torch.device(device)
        repo = repo or os.environ.get("ORBISIM_ROOT")
        ckpt = ckpt or os.environ.get("ORBISIM_CKPT")
        if repo is None:
            raise RuntimeError("Set ORBISIM_ROOT (path to the OrbiSim repo) and ORBISIM_CKPT")
        import sys
        sys.path.insert(0, repo)
        # TODO(server): import and build the OrbiSim-Dynamics model, load ``ckpt``.
        # self.model = orbisim.build_dynamics(...).to(self.dev).eval()
        raise NotImplementedError("OrbiSimOfficial: fill in model construction")

    # -- state conversion ---------------------------------------------------
    def _to_orbisim_state(self, ctx):
        """TODO: map ctx['state'] (raw sim state, see sims/*.py docstrings) or
        ctx['obs'] into OrbiSim's object-centric state format."""
        raise NotImplementedError

    def _to_orbisim_action(self, actions):
        """TODO: map our velocity actions (B,H,A) into OrbiSim's action space."""
        raise NotImplementedError

    def _risk_from_states(self, states):
        """TODO (route a): predicted physical states -> normalised risk (B,H,2)."""
        raise NotImplementedError

    # -- PredictorAdapter ---------------------------------------------------
    def risk(self, ctx, actions):
        s0 = self._to_orbisim_state(ctx)
        u = self._to_orbisim_action(actions)
        states = self.model.rollout(s0, u)            # TODO: real API
        r = self._risk_from_states(states)
        return r, torch.zeros_like(r)                 # no ensemble -> std 0

    def rollout(self, obs, obs_prev, a_prev, actions):
        raise NotImplementedError("optional: needed only for RQ1 fidelity / grad agreement")
