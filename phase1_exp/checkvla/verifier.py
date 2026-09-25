"""
Closed-loop controller + episode runner.

The controller is component-agnostic: it only uses
  * a PolicyAdapter   (interfaces.py)  — proposes action chunks
  * a VerifierAdapter (interfaces.py)  — check() / repair()   [mode "checkvla"]
  * the GT Env's shadow_rollout()                              [mode "gt_shadow"]

Modes:
  "none"       uncorrected policy
  "gt_shadow"  Phase-0 style: shadow-roll every chunk in the GT simulator (slow
               engine in the hot loop), scale the chunk down until GT-safe
  "checkvla"   verifier check -> (if triggered) latency-aware repair; the repaired
               chunk is executed open-loop for ``verifier.commit`` steps
"""
from __future__ import annotations

import time

import numpy as np
import torch

# re-exported for backwards compatibility
from checkvla.reference import calibrate_tau, chunk_score, suffix_repair  # noqa: F401


class Controller:
    def __init__(self, policy, sim, mode="none", verifier=None, commit=5,
                 async_engine=None, instruction=None):
        assert mode in ("none", "gt_shadow", "checkvla")
        assert mode != "checkvla" or verifier is not None
        self.policy, self.sim, self.mode = policy, sim, mode
        self.verifier = verifier
        self.commit = verifier.commit if verifier is not None else commit
        self.async_engine = async_engine
        self.instruction = instruction
        self.needs_img = bool(getattr(policy, "needs_img", False)) or bool(
            verifier is not None and getattr(verifier.predictor, "needs_img", False))
        self.needs_rgb = bool(getattr(policy, "needs_rgb", False))

    def reset(self, env, obs):
        B, dev = obs.shape[0], obs.device
        H, A = self.policy.H, self.policy.act_dim
        self.obs_prev = obs.clone()
        self.a_prev = torch.zeros(B, A, device=dev)
        self.img_prev = env.render() if self.needs_img else None
        self.plan = torch.zeros(B, H, A, device=dev)
        self.plan_ptr = torch.zeros(B, dtype=torch.long, device=dev)
        self.plan_left = torch.zeros(B, dtype=torch.long, device=dev)

    def _ctx(self, env, obs):
        ctx = {"obs": obs, "obs_prev": self.obs_prev, "a_prev": self.a_prev,
               "state": env.get_state(), "instruction": self.instruction}
        if self.needs_img:
            ctx["img"], ctx["img_prev"] = env.render(), self.img_prev
        if self.needs_rgb:
            ctx["rgb"] = env.render_rgb()          # real VLA camera frames
        return ctx

    def _set_plan(self, mask, chunks):
        self.plan[mask] = chunks
        self.plan_ptr[mask] = 0
        self.plan_left[mask] = min(self.commit, self.plan.shape[1])

    @torch.no_grad()
    def act(self, env, obs):
        B, dev = obs.shape[0], obs.device
        ar = torch.arange(B, device=dev)
        ctx = self._ctx(env, obs)
        chunk = self.policy(obs, ctx.get("rgb" if self.needs_rgb else "img"), self.instruction)
        info = {"trig": torch.zeros(B, dtype=torch.bool, device=dev),
                "score": torch.zeros(B, device=dev)}
        using_plan = self.plan_left > 0

        if self.mode == "checkvla":
            trig, score = self.verifier.check(ctx, chunk)
            info["score"] = score
            trig = trig & ~using_plan
            if self.async_engine is not None:
                late = self.async_engine.poll(B, dev)          # stale slow-engine verdicts
                trig = trig | (late & ~using_plan)
                self.async_engine.submit(ctx["state"], chunk)
            if trig.any():
                idx = trig.nonzero().squeeze(1)
                sub = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in ctx.items()}
                self._set_plan(trig, self.verifier.repair(sub, chunk[idx]))
            info["trig"] = trig

        elif self.mode == "gt_shadow":
            risk = env.shadow_rollout(chunk)
            viol = (risk.amax(dim=(1, 2)) > 1.0) & ~using_plan
            info["score"] = risk.amax(dim=(1, 2))
            if viol.any():
                best, todo = chunk.clone(), viol.clone()
                for sc in (0.75, 0.5, 0.3, 0.15):
                    cand = chunk * sc
                    ok = todo & (env.shadow_rollout(cand).amax(dim=(1, 2)) <= 1.0)
                    best[ok] = cand[ok]
                    todo &= ~ok
                best[todo] = chunk[todo] * 0.15
                self._set_plan(viol, best[viol])
            info["trig"] = viol

        use = self.plan_left > 0
        a = torch.where(use[:, None],
                        self.plan[ar, self.plan_ptr.clamp(max=chunk.shape[1] - 1)], chunk[:, 0])
        self.plan_ptr = torch.where(use, self.plan_ptr + 1, self.plan_ptr)
        self.plan_left = torch.where(use, self.plan_left - 1, self.plan_left)
        self.obs_prev, self.a_prev = obs.clone(), a.clone()
        if self.needs_img:
            self.img_prev = ctx["img"]
        info["chunk"] = chunk
        return a, info


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_episodes(env, sim, controller, params=None, seed=0, T=None, record=False):
    """Roll out env.n parallel episodes. Returns per-episode metric arrays."""
    T = T or sim.T
    obs = env.reset(params, seed)
    controller.reset(env, obs)
    B, dev = obs.shape[0], obs.device
    viol_any = torch.zeros(B, dtype=torch.bool, device=dev)
    first_viol = torch.full((B,), -1, dtype=torch.long, device=dev)
    first_trig = torch.full((B,), -1, dtype=torch.long, device=dev)
    n_int = torch.zeros(B, device=dev)
    lat, rec = [], {"risk": [], "score": [], "trig": []}
    success = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(T):
        _sync(dev)
        t0 = time.perf_counter()
        a, info = controller.act(env, obs)
        _sync(dev)
        lat.append((time.perf_counter() - t0) * 1e3)
        obs, risk, success = env.step(a)
        v = risk.amax(1) > 1.0
        first_viol = torch.where(v & (first_viol < 0), torch.full_like(first_viol, t), first_viol)
        first_trig = torch.where(info["trig"] & (first_trig < 0), torch.full_like(first_trig, t), first_trig)
        viol_any |= v
        n_int += info["trig"].float()
        if record:
            rec["risk"].append(risk.cpu())
            rec["score"].append(info["score"].cpu())
            rec["trig"].append(info["trig"].cpu())
    out = {
        "success": success.cpu().numpy(),
        "violation": viol_any.cpu().numpy(),
        "first_viol": first_viol.cpu().numpy(),
        "first_trig": first_trig.cpu().numpy(),
        "n_interventions": n_int.cpu().numpy(),
        "latency_ms": np.array(lat),
    }
    if record:
        out["trace"] = {k: torch.stack(v, 1).numpy() for k, v in rec.items()}
    return out


def summarize(res: dict) -> dict:
    s, v, ni = res["success"], res["violation"], res["n_interventions"]
    intervened = ni > 0
    return {
        "SR": float(s.mean()),
        "CVR": float(v.mean()),
        "RR": float(s[intervened].mean()) if intervened.any() else float("nan"),
        "intervention_rate": float(intervened.mean()),
        "latency_ms_mean": float(res["latency_ms"].mean()),
        "latency_ms_p95": float(np.percentile(res["latency_ms"], 95)),
    }
