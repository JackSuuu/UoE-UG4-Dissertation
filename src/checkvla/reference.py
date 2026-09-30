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
                 scales=None, iters=6, floor=0.05, hard_prefix=False,
                 use_grad=False, abstain=True, return_mask=False, resp_frac=0.02):
    """Gradient-free repair: down-scale, bisected to the largest factor whose
    predicted score clears ``target``.

    ``use_grad`` (default off) enables the gradient branch. It is off because it
    is **structurally inert here**: ``suffix_repair`` holds a hard prefix, and the
    violating contact is the first step of the chunk, so the gradient cannot
    change the one thing that matters. Measured with 128 envs at friction
    0.2/mass 1.5, tau forced to 0, same proposed chunk:

        scale_repair  (bisection)  GT 1.193 -> 0.125
        suffix_repair (gradient)   GT 1.193 -> 1.193   <- unchanged

    Yet the gradient branch scores *lower* on the predictor (0.300 vs 0.256 --
    it wins on 29% of envs), so the old ``repair()`` picked it and threw away
    the bisection's result: 1.193 -> 0.419. The gradient is not noisy here, it
    is inert, and nothing in the selection step knew the difference.

    Do not conflate this with the RQ3 ``stab`` table, which is a different code
    path: that optimises the whole action sequence by direct BPTT through the
    simulator, with no prefix and no verifier in the loop, and finds that
    *unregularised* BPTT is the best of the three (final CVR 0.88, against 1.00
    for both clipping and relaxation) -- over-regularising makes the steps
    worse. Two distinct mechanisms, one shared conclusion: gradients through
    this contact model are not worth optimising against.

    ``target = min(margin, (1 - DROP) * score(chunk))``: a down-scale only counts
    as a repair if the predictor certifies a *relative* reduction, because the
    predictor carries a near-constant offset (predicted risk 0.412/0.413/0.415/
    0.421 at scales 1.0/0.9/0.75/0.5) and an absolute margin alone would accept
    the very first rung and leave the true risk untouched.

    Design points, each measured on friction=0.2 / mass=1.5:
      * down-scale the *whole* chunk when ``hard_prefix=False`` (legacy): the GT
        risk of a chunk scaled 1.0/0.75/0.5/0.15/0.0 falls
        0.34/0.27/0.20/0.09/0.00, and scaling only the suffix leaves the
        committed prefix's contact force in place.
      * bisection rather than a fixed ladder. With a ladder the repair was
        discontinuous in the trigger threshold: at friction=1.0 tau 0.60/0.45/
        0.35/0.25 gave CVR 0.00/0.26/0.00/0.00, i.e. lowering the threshold
        sometimes *raised* the violation rate. The bisection makes the applied
        factor a smooth function of the score, so the trade-off is monotone.
      * largest factor that clears the target, never the argmin -- argmin pins
        every chunk to the bottom rung (95% down-scaling), which stops the task
        from completing.

    ``hard_prefix`` implements CheckVLA's latency-aware constraint (hold the
    first ``latency`` dispatched actions, damp only the suffix) and is **off by
    default, because it measures worse.** CheckVLA's constraint assumes the
    already-dispatched actions are not the problem. On this task they are: the
    risk label is the max over the chunk, and the violating contact is *at* the
    first step of the chunk, so holding the prefix holds the violation. With
    128 envs, predictor `orbisim`, friction 0.2/mass 1.5, tau forced to 0:

        hard_prefix=False   GT 1.004 -> 0.110  (-89.0%)  cleared 0.99
        hard_prefix=True    GT 1.004 -> 0.708  (-29.5%)  cleared 0.68
        (friction 1.0/mass 1.0: 1.020 -> 0.136 vs 1.020 -> 0.540)

    In both cases the bisection drives the suffix to x0.10, nearly a dead stop,
    and still cannot clear the limit -- because it cannot touch the step that
    violates. Keep it as an ablation, not as the default.

    Note the gradient path (``suffix_repair``) has a hard prefix by construction
    and fails for the identical reason, which is why the branch is opt-in.

    ``abstain`` (default on) is a correctness guard, not a tuning knob. The
    bisection seeds ``lo = floor`` on the assumption that scaling to ``floor`` is
    already conservative enough. That assumption is false for a predictor that
    does not respond to the action: the score is then constant in the scale, no
    rung clears ``target``, the accept test never passes, and the search falls
    straight through to the floor. Measured on that arm (``vision_noact``):
    magnitude ratio x0.05, and CVR 0.278 against a 0.272 baseline with SR -0.057 --
    it stops the task without preventing anything.

    **The guard tests responsiveness, not certification.** The obvious predicate
    is "no rung cleared the target, so abstain", and it is wrong: ``margin`` (0.8)
    is deliberately stricter than the constraint limit (1.0) so that the repair
    *disarms* the trigger, and a predictor carrying a large near-constant offset
    cannot always reach it. That version was implemented and measured, and it
    abstained on 45% of orbisim's own interventions -- magnitude ratio 0.63
    instead of 0.34, CVR 0.92 instead of 0.70 on the stress cell. It silently
    disabled the verifier it was meant to protect. Adding a second accept
    criterion at ``tau`` did not rescue it either, because a damped chunk the
    predictor still scores above ``tau`` is frequently the best available action
    (the v6 audit puts orbisim's floor damping at -28.9% true risk).

    What actually distinguishes the broken arm is that the search has no signal:
    the objective does not vary with the variable being searched. So the guard
    measures the score at the floor and abstains when the scale moves it by less
    than ``resp_eps`` relative. One extra predictor call per intervention.

        orbisim      score 0.431 / 0.429 / 0.201 at scale 1.0 / 0.5 / 0.05
        vision       score 0.395 / 0.299 / 0.237
        vision_noact score 0.453 / 0.453 / 0.453   <- unresponsive, abstain

    Abstention is then the honest response: the search certified nothing, so the
    runtime gets its proposed chunk back rather than an arbitrary near-stop that
    a reader would score as a maximal repair. The repair audit surfaces the
    change without extra plumbing -- an abstaining arm shows ``mag_ratio_mean``
    1.00 and ``frac_ineffective`` 1.00. And because the predicate is
    responsiveness, the guard is provably inert on any predictor that responds,
    so the working arms are bit-unchanged.

    ``return_mask`` hands back the per-row flag so a caller holding several
    candidate repairs can make the abstention dominate all of them rather than
    only the bisection's.
    """
    lat = int(min(latency, chunk.shape[1]))
    pre, suf = chunk[:, :lat], chunk[:, lat:]

    def rebuild(f):
        if hard_prefix and lat > 0:
            return torch.cat([pre, suf * f[:, None, None]], 1)
        return chunk * f[:, None, None]

    s0, _, _ = chunk_score(predictor, ctx, chunk, beta)
    target = torch.minimum(s0 * (1.0 - DROP), s0.new_full((), margin))
    lo = torch.full_like(s0, floor)          # assumed conservative enough
    hi = torch.ones_like(s0)                 # known to be too aggressive
    for _ in range(iters):
        mid = (lo + hi) / 2
        s, _, _ = chunk_score(predictor, ctx, rebuild(mid), beta)
        safe = s <= target
        lo = torch.where(safe, mid, lo)
        hi = torch.where(safe, hi, mid)
    out = rebuild(lo)

    # Responsiveness of the objective to the variable being searched, as a
    # fraction of the score itself. An absolute epsilon does not work: the
    # action-blind arm still wobbles by ~0.002, which clears any epsilon small
    # enough to be safe and lets it through. The measured separation is wide
    # (53% / 40% / 0.0%), so the threshold is not knife-edge.
    sf, _, _ = chunk_score(predictor, ctx, rebuild(torch.full_like(s0, floor)), beta)
    responsive = (s0 - sf) > resp_frac * s0.abs()
    if return_mask:                     # report the true flag, not the applied one
        return out, responsive
    return torch.where(responsive[:, None, None], out, chunk) if abstain else out


class RefCheckVLA:
    """VerifierAdapter (interfaces.py) — reference stand-in."""

    def __init__(self, predictor, tau=1.0, beta=1.0, latency=1, commit=5,
                 repair_iters=25, max_vel=1.0, margin=0.8, hard_prefix=False,
                 use_grad=False, abstain=True, resp_frac=0.02):
        self.predictor, self.tau, self.beta = predictor, tau, beta
        self.latency, self.commit = latency, commit
        self.repair_iters, self.max_vel, self.margin = repair_iters, max_vel, margin
        # CheckVLA's latency-aware constraint and its gradient branch, both
        # switchable so the ablations are one flag. Both default off; see the
        # docstrings for the measurements that put them there.
        self.hard_prefix, self.use_grad = hard_prefix, use_grad
        # Abstention guard: return the proposed chunk unchanged when the search
        # certifies no scale. A correctness fix, not a knob -- see scale_repair.
        self.abstain = abstain
        self.resp_frac = resp_frac
        # Interventions where the guard fired. Reported so abstention is visible
        # rather than inferred from SR/CVR, which cannot resolve a change this
        # small on 64 episodes.
        self.n_abstain = 0

    def score(self, ctx, chunk):
        return chunk_score(self.predictor, ctx, chunk, self.beta)[0]

    def check(self, ctx, chunk):
        s = self.score(ctx, chunk)
        return s > self.tau, s

    def repair(self, ctx, chunk):
        """Bisected down-scale, with the gradient suffix repair as an optional
        candidate (off by default -- see ``scale_repair``).

        History, because it is where two separate defects were stacked. The
        gradient step alone is unsafe: on out-of-distribution cells the
        predictor's ranking of action magnitudes can invert, so a pure gradient
        descent walks toward larger actions and raised the true violation rate
        from 0.00 to 0.27. Adding a down-scale candidate and keeping whichever
        the predictor rates safest fixed that -- and then a second problem
        surfaced underneath. The gradient branch is *inert* rather than merely
        wrong, because its hard prefix holds the one step that violates
        (measured GT 1.193 -> 1.193, against 0.125 for the bisection), yet it
        still scores lower on the predictor, so the selection picked it and
        discarded the bisection (1.193 -> 0.419). Neither branch is trustworthy
        on its own score: one is inert and over-claims, the other is sound. That
        is why the gradient branch is now opt-in.
        """
        cand, cleared = scale_repair(self.predictor, ctx, chunk, self.latency,
                                     self.beta, self.margin,
                                     hard_prefix=self.hard_prefix, abstain=False,
                                     return_mask=True,
                                     resp_frac=self.resp_frac)
        best = cand
        if self.use_grad and getattr(self.predictor, "differentiable", False):
            grad = suffix_repair(self.predictor, ctx, chunk, self.latency, self.repair_iters,
                                 beta=self.beta, margin=self.margin, max_vel=self.max_vel)
            s_grad, _, _ = chunk_score(self.predictor, ctx, grad, self.beta)
            s_cand, _, _ = chunk_score(self.predictor, ctx, cand, self.beta)
            best = torch.where((s_grad <= s_cand)[:, None, None], grad, cand)
        # The guard is applied here, after the selection, so it covers the
        # gradient branch too. If the predictor does not respond to the
        # down-scale, the gradient candidate is not a repair either -- it only
        # ever scores lower because it barely moves the chunk.
        self.n_abstain += int((~cleared).sum())
        return torch.where(cleared[:, None, None], best, chunk) if self.abstain else best
