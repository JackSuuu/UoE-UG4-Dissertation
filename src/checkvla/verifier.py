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
                 async_engine=None, instruction=None, chunk_k=None):
        assert mode in ("none", "gt_shadow", "checkvla")
        assert mode != "checkvla" or verifier is not None
        self.policy, self.sim, self.mode = policy, sim, mode
        self.verifier = verifier
        self.commit = verifier.commit if verifier is not None else commit
        self.async_engine = async_engine
        self.instruction = instruction
        # chunk_k: steps between policy calls (open-loop chunk execution).
        # All arms use the same value so the comparison is fair. Default: commit.
        self.chunk_k = chunk_k if chunk_k is not None else self.commit
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

    def _set_plan(self, mask, chunks, steps=None):
        self.plan[mask] = chunks
        self.plan_ptr[mask] = 0
        self.plan_left[mask] = min(steps or self.commit, self.plan.shape[1])

    @torch.no_grad()
    def act(self, env, obs):
        B, dev = obs.shape[0], obs.device
        ar = torch.arange(B, device=dev)
        ctx = self._ctx(env, obs)
        chunk = self.policy(obs, ctx.get("rgb" if self.needs_rgb else "img"), self.instruction)
        info = {"trig": torch.zeros(B, dtype=torch.bool, device=dev),
                "score": torch.zeros(B, device=dev),
                "applied": None}
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
                rep = self.verifier.repair(sub, chunk[idx])
                self._set_plan(trig, rep)
                # the chunk that will actually be executed, for the repair audit
                # (kept out of the hot path: the audit itself runs in run_episodes)
                ap = chunk.clone()
                ap[idx] = rep
                info["applied"] = ap
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
                ap = chunk.clone()
                ap[viol] = best[viol]
                info["applied"] = ap
            info["trig"] = viol

        elif self.mode == "none" and (~using_plan).any():
            # Open-loop chunk execution: run the chunk for chunk_k steps before
            # replanning. This matches the CheckVLA setting where a VLA executes
            # action chunks open-loop.
            self._set_plan(~using_plan, chunk[~using_plan], steps=self.chunk_k)

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
def run_episodes(env, sim, controller, params=None, seed=0, T=None, record=False,
                 audit_repair=False):
    """Roll out env.n parallel episodes. Returns per-episode metric arrays.

    audit_repair: on every intervention, shadow-roll the chunk that was
    *executed* alongside the chunk the policy *proposed*. The ratio of their GT
    risks is the only direct measure of whether the repair changed anything on
    the real dynamics -- a predictor can fire often, look controllable, and
    still leave the true risk of the executed chunk untouched. Kept out of
    Controller.act so it never contaminates the reported latency.
    """
    T = T or sim.T
    obs = env.reset(params, seed)
    controller.reset(env, obs)
    B, dev = obs.shape[0], obs.device
    viol_any = torch.zeros(B, dtype=torch.bool, device=dev)
    first_viol = torch.full((B,), -1, dtype=torch.long, device=dev)
    first_trig = torch.full((B,), -1, dtype=torch.long, device=dev)
    n_int = torch.zeros(B, device=dev)
    lat, rec = [], {"risk": [], "score": [], "trig": []}
    audit = {"gt_applied": [], "gt_proposed": [], "mag_ratio": []}
    success = torch.zeros(B, dtype=torch.bool, device=dev)
    for t in range(T):
        _sync(dev)
        t0 = time.perf_counter()
        a, info = controller.act(env, obs)
        _sync(dev)
        lat.append((time.perf_counter() - t0) * 1e3)
        if audit_repair and info.get("applied") is not None:
            m = info["trig"]
            ap, pr = info["applied"][m], info["chunk"][m]
            audit["gt_applied"].append(env.shadow_rollout(ap, mask=m).amax(dim=(1, 2)).cpu())
            audit["gt_proposed"].append(env.shadow_rollout(pr, mask=m).amax(dim=(1, 2)).cpu())
            audit["mag_ratio"].append((ap.norm(dim=(1, 2))
                                       / pr.norm(dim=(1, 2)).clamp_min(1e-8)).cpu())
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
        "n_steps": T,
        "latency_ms": np.array(lat),
    }
    if record:
        out["trace"] = {k: torch.stack(v, 1).numpy() for k, v in rec.items()}
    if audit_repair and audit["gt_applied"]:
        ga = torch.cat(audit["gt_applied"]).numpy()
        gp = torch.cat(audit["gt_proposed"]).numpy()
        mr = torch.cat(audit["mag_ratio"]).numpy()
        out["repair_audit"] = {
            "n_interventions": int(ga.size),
            "gt_risk_proposed": float(gp.mean()),
            "gt_risk_applied": float(ga.mean()),
            # relative reduction of the TRUE risk of the chunk actually executed
            "gt_risk_rel_drop": float((gp - ga).mean() / max(gp.mean(), 1e-8)),
            # fraction of interventions that pushed the true risk below the limit
            "frac_cleared": float((ga <= 1.0).mean()),
            # fraction of interventions that were no-ops on the true dynamics
            "frac_ineffective": float((ga >= gp - 1e-3).mean()),
            "mag_ratio_mean": float(mr.mean()),
        }
    elif audit_repair:
        # Always emit the keys, even with nothing to report. A cell where the
        # arm never triggered has no repair_audit dict at all, and any pooling
        # that walks one arm's key set over every cell then dies with a KeyError
        # -- i.e. exactly the cells where the arm was most conservative are the
        # ones that disappear.
        out["repair_audit"] = {"n_interventions": 0, "gt_risk_proposed": float("nan"),
                               "gt_risk_applied": float("nan"), "gt_risk_rel_drop": float("nan"),
                               "frac_cleared": float("nan"), "frac_ineffective": float("nan"),
                               "mag_ratio_mean": float("nan")}
    return out


def summarize(res: dict) -> dict:
    s, v, ni = res["success"], res["violation"], res["n_interventions"]
    intervened = ni > 0
    T = res["n_interventions"].size and max(1, int(res["n_steps"]))
    out = {
        "SR": float(s.mean()),
        "CVR": float(v.mean()),
        "safe_success": float((s & ~v).mean()),
        "RR": float(s[intervened].mean()) if intervened.any() else float("nan"),
        # fraction of CONTROL STEPS that triggered, not the fraction of episodes
        # that ever did -- with a per-step rate of ~0.15 over 80 steps nearly
        # every episode triggers at least once, which made the two identical
        # (1.00) and hid the real trigger frequency.
        "intervention_rate": float(ni.sum() / (T * len(ni))),
        "episode_trigger_rate": float(intervened.mean()),
        "latency_ms_mean": float(res["latency_ms"].mean()),
        "latency_ms_p95": float(np.percentile(res["latency_ms"], 95)),
    }
    if "repair_audit" in res:
        out.update(res["repair_audit"])
    return out
