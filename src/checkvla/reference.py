"""
Reference (STAND-IN) implementation of the CheckVLA verifier — written from the
paper's high-level description only (Liu et al., arXiv:2607.26789):
action-conditioned world model + calibrated-threshold trigger + latency-aware
suffix repair. It is NOT the authors' code; details (score definition, repair
optimiser) are our own design choices. Replace with
``adapters/checkvla_official.py`` via ``--verifier official``.

Score of a chunk  = max over horizon & channels of  mean_risk + beta * std
Trigger           = score > tau   (tau: split-conformal, FAR on safe chunks <= alpha)
Repair            = keep first ``latency`` actions; optimise the suffix by
                    gradient descent through the predictor to push risk below
                    ``margin`` while staying close to the proposal. Falls back to
                    uniform down-scaling if the predictor is not differentiable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def chunk_score(predictor, ctx, chunk, beta=1.0):
    mean, std = predictor.risk(ctx, chunk)
    return (mean + beta * std).amax(dim=(1, 2)), mean, std


def calibrate_tau(scores_safe, alpha: float = 0.05) -> float:
    """Split-conformal threshold: P(score > tau | safe) <= alpha."""
    s = np.sort(np.asarray(scores_safe))
    n = len(s)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(max(k, 0), n - 1)])


def suffix_repair(predictor, ctx, chunk, latency=1, iters=25, lr=0.1, lam=0.5,
                  margin=0.8, beta=1.0, max_vel=1.0):
    prefix = chunk[:, :latency].detach()
    orig = chunk[:, latency:].detach()
    suffix = orig.clone().requires_grad_(True)
    opt = torch.optim.Adam([suffix], lr=lr * max_vel)
    with torch.enable_grad():
        for _ in range(iters):
            acts = torch.cat([prefix, suffix], 1)
            mean, std = predictor.risk(ctx, acts)
            r = mean + beta * std
            loss = F.relu(r - margin).pow(2).sum((1, 2)) \
                + lam * ((suffix - orig) / max_vel).pow(2).mean((1, 2))
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                suffix.clamp_(-max_vel, max_vel)
    return torch.cat([prefix, suffix.detach()], 1)


@torch.no_grad()
def scale_repair(predictor, ctx, chunk, latency=1, beta=1.0, margin=0.8,
                 scales=(0.75, 0.5, 0.3, 0.15)):
    """Gradient-free fallback: pick the largest suffix scale predicted safe."""
    best = chunk.clone()
    todo = torch.ones(chunk.shape[0], dtype=torch.bool, device=chunk.device)
    for sc in scales:
        cand = chunk.clone()
        cand[:, latency:] *= sc
        s, _, _ = chunk_score(predictor, ctx, cand, beta)
        ok = todo & (s <= margin)
        best[ok] = cand[ok]
        todo &= ~ok
    best[todo, latency:] = chunk[todo, latency:] * scales[-1]
    return best


class RefCheckVLA:
    """VerifierAdapter (interfaces.py) — reference stand-in."""

    def __init__(self, predictor, tau=1.0, beta=1.0, latency=1, commit=5,
                 repair_iters=25, max_vel=1.0, margin=0.8):
        self.predictor, self.tau, self.beta = predictor, tau, beta
        self.latency, self.commit = latency, commit
        self.repair_iters, self.max_vel, self.margin = repair_iters, max_vel, margin

    def score(self, ctx, chunk):
        return chunk_score(self.predictor, ctx, chunk, self.beta)[0]

    def check(self, ctx, chunk):
        s = self.score(ctx, chunk)
        return s > self.tau, s

    def repair(self, ctx, chunk):
        if getattr(self.predictor, "differentiable", False):
            return suffix_repair(self.predictor, ctx, chunk, self.latency, self.repair_iters,
                                 beta=self.beta, margin=self.margin, max_vel=self.max_vel)
        return scale_repair(self.predictor, ctx, chunk, self.latency, self.beta, self.margin)
