"""
Real CheckVLA adapter (VerifierAdapter, interfaces.py) — SKELETON.

Target: CheckVLA (Liu et al., arXiv:2607.26789). I have NOT checked whether
code is public. The RQ1/RQ2 manipulation is "which predictor sits inside the
verifier", so the official verifier must accept an arbitrary PredictorAdapter
(OrbiSim-Dynamics vs. its own visual world model). If the official code hard-
wires its world model, wrap it as a PredictorAdapter (see
``CheckVLAVisualWM`` below) and keep using this class — or ``RefCheckVLA`` —
for trigger/repair logic.

Contract
--------
score(ctx, chunk)   -> (B,)            risk score, higher = riskier
check(ctx, chunk)   -> (mask (B,), score (B,))   uses the calibrated self.tau
repair(ctx, chunk)  -> (b,H,A)         first ``latency`` actions unchanged
attributes: predictor, tau, latency, commit
calibrate.py sets ``tau`` from scores of GT-safe chunks (FAR <= alpha); if the
official method has its own calibration, override ``tau`` after construction
and pass ``--no_recalibrate`` semantics by editing taus.json.
"""
from __future__ import annotations

import os

import torch


class CheckVLAOfficial:
    def __init__(self, predictor, tau=1.0, latency=1, commit=5, sim=None, **kw):
        self.predictor, self.tau = predictor, tau
        self.latency, self.commit = latency, commit
        self.sim = sim
        repo = os.environ.get("CHECKVLA_ROOT")
        if repo is None:
            raise RuntimeError("Set CHECKVLA_ROOT to the CheckVLA repo")
        import sys
        sys.path.insert(0, repo)
        # TODO(server): import CheckVLA's trigger + suffix-repair modules.
        raise NotImplementedError("CheckVLAOfficial: fill in")

    def score(self, ctx, chunk):
        # TODO: official risk score from self.predictor.risk(ctx, chunk)
        raise NotImplementedError

    def check(self, ctx, chunk):
        s = self.score(ctx, chunk)
        return s > self.tau, s

    def repair(self, ctx, chunk):
        # TODO: official latency-aware suffix repair
        raise NotImplementedError


class CheckVLAVisualWM:
    """PredictorAdapter wrapper for CheckVLA's own visual world model — SKELETON.
    Use as the real baseline predictor (replaces the VisionWM stand-in)."""
    needs_img = True        # set to match what the WM consumes (may need render_rgb)
    differentiable = False  # set True if gradients w.r.t. actions are available

    def __init__(self, sim, device, ckpt=None):
        self.sim, self.dev = sim, torch.device(device)
        # TODO(server): load CheckVLA's action-conditioned world model + risk readout.
        raise NotImplementedError("CheckVLAVisualWM: fill in")

    def risk(self, ctx, actions):
        # TODO: -> (mean (B,H,2), std (B,H,2)) in normalised risk units
        raise NotImplementedError
