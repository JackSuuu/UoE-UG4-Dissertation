"""
RQ3 systems optimisations 1 & 2 (plan §3.5):

1. Truncated differentiable checkpointing — forward pass without autograd,
   storing only segment-boundary physics states (offloaded to CPU); the
   backward pass reloads one boundary at a time, recomputes that segment with
   autograd and back-propagates through it, so live GPU memory is
   O(segment length) instead of O(horizon). Optional block truncation (TBPTT
   window W) bounds gradient path length.

2. Adaptive gradient truncation ("gradient stabilizer") — during the reverse
   sweep we see dL/ds_t for every step. When its norm spikes above
   ``rho * running-median`` (chaotic contact window) the step's gradient is
   either clipped ("clip") or replaced by the gradient of a *smooth relaxation*
   of the contact model ("relax", sim.step(..., smooth=True)).

Both operate on any ``FunctionalSim`` (pure step function). ``naive_bptt`` is
the reference full backprop-through-time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


def step_cost(sim, s, risk, w_risk=5.0, margin=0.8):
    return sim.task_cost(s) + w_risk * F.relu(risk - margin).pow(2).sum(-1)


def naive_bptt(sim, params, s0, actions, smooth=False):
    a = actions.detach().clone().requires_grad_(True)
    s, total = s0, 0.0
    for t in range(a.shape[1]):
        s, r = sim.step(s, a[:, t], params, smooth=smooth)
        total = total + step_cost(sim, s, r)
    total.sum().backward()
    return total.detach(), a.grad.detach(), {}


@dataclass
class Stabilizer:
    mode: str = "relax"          # "relax" | "clip"
    rho: float = 5.0
    warmup: int = 5
    norms: list = field(default_factory=list)

    def reset(self):
        self.norms = []

    def is_spike(self, gnorm: float) -> bool:
        if len(self.norms) < self.warmup:
            return False
        med = float(np.median(self.norms))
        return gnorm > self.rho * max(med, 1e-12)

    def cap(self):
        return self.rho * max(float(np.median(self.norms)), 1e-12)


def checkpointed_bptt(sim, params, s0, actions, seg=10, offload=True,
                      trunc_window=None, stabilizer: Stabilizer | None = None):
    """Returns (loss (B,), dL/da (B,T,A), info{raw_norms, used_norms, n_spikes})."""
    dev = s0.device
    T = actions.shape[1]
    acts = actions.detach()
    # ---- forward, no graph ------------------------------------------------
    bounds = []
    total = torch.zeros(s0.shape[0], device=dev)
    with torch.no_grad():
        s = s0.detach()
        for t in range(T):
            if t % seg == 0:
                bounds.append(s.to("cpu", non_blocking=True) if offload else s.clone())
            s, r = sim.step(s, acts[:, t], params)
            total += step_cost(sim, s, r)
    # ---- backward, segment by segment -------------------------------------
    grad_s = torch.zeros_like(s)
    act_grad = torch.zeros_like(acts)
    raw_norms = np.zeros(T)
    used_norms = np.zeros(T)
    n_spikes = 0
    if stabilizer:
        stabilizer.reset()
    for k in reversed(range(len(bounds))):
        t0, t1 = k * seg, min(T, (k + 1) * seg)
        s_in = bounds[k].to(dev, non_blocking=True)
        leaves, outs, losses, a_leaves = [], [], [], []
        with torch.enable_grad():
            for i in range(t1 - t0):
                sl = s_in.detach().requires_grad_(True)
                al = acts[:, t0 + i].clone().requires_grad_(True)
                so, r = sim.step(sl, al, params)
                leaves.append(sl); outs.append(so); a_leaves.append(al)
                losses.append(step_cost(sim, so, r).sum())
                s_in = so
        for i in reversed(range(t1 - t0)):
            t = t0 + i
            torch.autograd.backward([outs[i], losses[i]], [grad_s, torch.ones_like(losses[i])])
            g, ga = leaves[i].grad, a_leaves[i].grad
            gn = float(g.norm())
            raw_norms[t] = gn
            if stabilizer is not None and stabilizer.is_spike(gn):
                n_spikes += 1
                if stabilizer.mode == "relax":
                    with torch.enable_grad():
                        sl2 = leaves[i].detach().requires_grad_(True)
                        al2 = acts[:, t].clone().requires_grad_(True)
                        so2, r2 = sim.step(sl2, al2, params, smooth=True)
                        l2 = step_cost(sim, so2, r2).sum()
                        torch.autograd.backward([so2, l2], [grad_s, torch.ones_like(l2)])
                    g, ga = sl2.grad, al2.grad
                    if float(g.norm()) > stabilizer.cap():       # relaxation still explosive
                        sc = stabilizer.cap() / (float(g.norm()) + 1e-12)
                        g, ga = g * sc, ga * sc
                else:
                    sc = stabilizer.cap() / (gn + 1e-12)
                    g, ga = g * sc, ga * sc
            if stabilizer is not None:
                stabilizer.norms.append(min(gn, stabilizer.cap()) if len(stabilizer.norms) >= stabilizer.warmup else gn)
            used_norms[t] = float(g.norm())
            act_grad[:, t] = ga
            grad_s = g.detach()
            if trunc_window and t % trunc_window == 0:
                grad_s = torch.zeros_like(grad_s)
        del leaves, outs, losses, a_leaves
    return total, act_grad, {"raw_norms": raw_norms, "used_norms": used_norms,
                             "n_spikes": n_spikes}


def measure(fn, device):
    """Run fn() and return (result, peak_mem_MB above baseline, seconds)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    else:
        peak = float("nan")
    return res, peak, time.perf_counter() - t0
