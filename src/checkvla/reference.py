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

# Fractional risk reduction that counts as a certified-safe down-scale (see
# ``scale_repair``). 0.25 keeps roughly three quarters of the commanded motion
# while still cutting the predicted contact risk substantially.
DROP = 0.25


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
                 scales=None, iters=6, floor=0.05):
    """Gradient-free repair: uniform down-scale, bisected to the largest factor
    whose predicted score clears ``target``.

    ``target = min(margin, (1 - DROP) * score(chunk))``: a down-scale only counts
    as a repair if the predictor certifies a *relative* reduction, because the
    predictor carries a near-constant offset (predicted risk 0.412/0.413/0.415/
    0.421 at scales 1.0/0.9/0.75/0.5) and an absolute margin alone would accept
    the very first rung and leave the true risk untouched.

    Three design points, each measured on friction=0.2 / mass=1.5:
      * uniform over the whole chunk, not just the suffix -- scaling only the
        suffix leaves the committed prefix's contact force in place, and the GT
        risk of a chunk scaled 1.0/0.75/0.5/0.15/0.0 falls
        0.34/0.27/0.20/0.09/0.00.
      * bisection rather than a fixed ladder. With a ladder the repair was
        discontinuous in the trigger threshold: at friction=1.0 tau 0.60/0.45/
        0.35/0.25 gave CVR 0.00/0.26/0.00/0.00, i.e. lowering the threshold
        sometimes *raised* the violation rate. The bisection makes the applied
        factor a smooth function of the score, so the trade-off is monotone.
      * largest factor that clears the target, never the argmin -- argmin pins
        every chunk to the bottom rung (95% down-scaling), which stops the task
        from completing.

    ``latency`` is kept in the signature for interface compatibility.
    """
    s0, _, _ = chunk_score(predictor, ctx, chunk, beta)
    target = torch.minimum(s0 * (1.0 - DROP), s0.new_full((), margin))
    lo = torch.full_like(s0, floor)          # known to be conservative enough
    hi = torch.ones_like(s0)                 # known to be too aggressive
    for _ in range(iters):
        mid = (lo + hi) / 2
        s, _, _ = chunk_score(predictor, ctx, chunk * mid[:, None, None], beta)
        safe = s <= target
        lo = torch.where(safe, mid, lo)
        hi = torch.where(safe, hi, mid)
    return chunk * lo[:, None, None]


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
        """Gradient suffix repair, with a monotone down-scale as a safety net.

        The gradient step alone is unsafe. On out-of-distribution cells the
        predictor's ranking of action magnitudes can invert: measured on
        friction=0.2, mass=1.5, the GT risk of a chunk scaled by
        1.0/0.75/0.5/0.15/0.0 falls 0.34/0.27/0.20/0.09/0.00, while the
        predicted risk *rises* over 1.0->0.5 (0.412 -> 0.421). A pure gradient
        descent therefore walks toward larger actions and raised the true
        violation rate from 0.00 to 0.27. We keep the gradient result but also
        score a ladder of uniform down-scales and take whichever the predictor
        rates safest, so the repair degrades to "be more conservative" rather
        than "optimise a surrogate that has stopped tracking reality".
        """
        cand = scale_repair(self.predictor, ctx, chunk, self.latency, self.beta, self.margin)
        if not getattr(self.predictor, "differentiable", False):
            return cand
        grad = suffix_repair(self.predictor, ctx, chunk, self.latency, self.repair_iters,
                             beta=self.beta, margin=self.margin, max_vel=self.max_vel)
        s_grad, _, _ = chunk_score(self.predictor, ctx, grad, self.beta)
        s_cand, _, _ = chunk_score(self.predictor, ctx, cand, self.beta)
        return torch.where((s_grad <= s_cand)[:, None, None], grad, cand)
