"""
Verifier-evaluation rollouts shared by calibrate.py and rq1_calibration.py.

The *uncorrected* policy is rolled out in the GT simulator. At every step we
record, for the policy's proposed chunk:
  * each predictor's risk score,
  * the GT chunk risk (non-destructive shadow rollout),
  * optionally (every ``grad_every`` steps, torch backend only): GT and predictor
    gradients of a common objective J(chunk) w.r.t. the chunk, the gradient-path
    validity flag, and OrbiSim's per-horizon obs error (fidelity curve).
"""
import numpy as np
import torch
import torch.nn.functional as F


GOAL_DIMS = {"push": 2, "cloth": 3}


def J_pred(model, ctx, chunk, goal_dims):
    o, r = model.rollout(ctx["obs"], ctx["obs_prev"], ctx["a_prev"], chunk)
    o, r = o.mean(0), r.mean(0)
    return (o[..., :goal_dims] ** 2).sum((1, 2)) + F.relu(r - 0.8).pow(2).sum((1, 2))


def J_gt(sim, s, params, chunk, goal_dims):
    tot = 0.0
    obs = []
    for h in range(chunk.shape[1]):
        s, r = sim.step(s, chunk[:, h], params)
        o = sim.obs(s)
        obs.append(o)
        tot = tot + (o[:, :goal_dims] ** 2).sum(1) + F.relu(r - 0.8).pow(2).sum(1)
    return tot, torch.stack(obs, 1)


def grad_probe(sim, params, state, chunk, orbisim, ctx, goal_dims):
    """-> dict with gt_grad, pred_grad (B,H*A), valid(B), cos(B), fidelity(B,H)."""
    with torch.enable_grad():
        c1 = chunk.detach().clone().requires_grad_(True)
        jg, gt_obs = J_gt(sim, state.detach(), params, c1, goal_dims)
        gg, = torch.autograd.grad(jg.sum(), c1)
        c2 = chunk.detach().clone().requires_grad_(True)
        jp = J_pred(orbisim, ctx, c2, goal_dims)
        gp, = torch.autograd.grad(jp.sum(), c2)
    gg, gp = gg.flatten(1), gp.flatten(1)
    gn = gg.norm(dim=1)
    valid = torch.isfinite(gn) & (gn > 1e-8)
    cos = F.cosine_similarity(torch.nan_to_num(gg), gp, dim=1)
    with torch.no_grad():
        o, _ = orbisim.rollout(ctx["obs"], ctx["obs_prev"], ctx["a_prev"], chunk)
        scale = getattr(orbisim, "obs_scale", None)
        scale = torch.ones_like(gt_obs[0, 0]) if scale is None else scale
        err = ((o.mean(0) - gt_obs.detach()) / scale).pow(2).mean(-1).sqrt()
    return {"valid": valid, "cos": cos, "fidelity": err,
            "regime": sim.contact_regime(state)}


@torch.no_grad()
def verifier_rollout(env, sim, policy, params, verifiers: dict, seed=0,
                     grad_every=0, act_noise=0.0):
    """verifiers: {role: VerifierAdapter}. Uses verifier.score for trigger scores
    and verifier.predictor.risk for the raw predicted risk."""
    obs = env.reset(params, seed)
    needs_rgb = bool(getattr(policy, "needs_rgb", False))
    probe_pred = verifiers["orbisim"].predictor if "orbisim" in verifiers else None
    can_probe = (probe_pred is not None and hasattr(probe_pred, "rollout")
                 and getattr(probe_pred, "differentiable", False))
    B, dev = obs.shape[0], obs.device
    obs_prev = obs.clone()
    a_prev = torch.zeros(B, sim.act_dim, device=dev)
    img_prev = env.render()
    rec = {k: [] for k in verifiers}
    rec_max = {k: [] for k in verifiers}
    gt_chunk, step_risk, succ = [], [], None
    probes = []
    g = torch.Generator().manual_seed(seed + 99)
    for t in range(sim.T):
        img = env.render()
        ctx = {"obs": obs, "obs_prev": obs_prev, "a_prev": a_prev,
               "img": img, "img_prev": img_prev, "state": env.get_state(), "instruction": None}
        chunk = policy(obs, env.render_rgb() if needs_rgb else img)
        for k, v in verifiers.items():
            rec[k].append(v.score(ctx, chunk))
            mean, _ = v.predictor.risk(ctx, chunk)
            rec_max[k].append(mean.amax(dim=(1, 2)))
        gt_chunk.append(env.shadow_rollout(chunk).amax(dim=(1, 2)))
        if grad_every and t % grad_every == 0 and params is not None and can_probe:
            pr = grad_probe(sim, params, env.get_state(), chunk, probe_pred,
                            ctx, GOAL_DIMS[sim.name])
            pr["t"] = t
            probes.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in pr.items()})
        a = chunk[:, 0]
        if act_noise > 0:
            a = (a + act_noise * sim.max_vel * torch.randn(a.shape, generator=g).to(dev)
                 ).clamp(-sim.max_vel, sim.max_vel)
        obs_prev, a_prev, img_prev = obs, a, img
        obs, r, succ = env.step(a)
        step_risk.append(r.amax(1))
    out = {
        "scores": {k: torch.stack(v, 1).cpu().numpy() for k, v in rec.items()},
        "pred_max_risk": {k: torch.stack(v, 1).cpu().numpy() for k, v in rec_max.items()},
        "gt_chunk_risk": torch.stack(gt_chunk, 1).cpu().numpy(),
        "step_risk": torch.stack(step_risk, 1).cpu().numpy(),
        "success": succ.cpu().numpy(),
    }
    if probes:
        out["probe"] = {
            "valid": torch.stack([p["valid"] for p in probes], 1).numpy(),
            "cos": torch.stack([p["cos"] for p in probes], 1).numpy(),
            "fidelity": torch.stack([p["fidelity"] for p in probes], 1).numpy(),
            "regime": torch.stack([p["regime"] for p in probes], 1).numpy(),
            "t": np.array([p["t"] for p in probes]),
        }
    return out


# ---------------------------------------------------------------------------
def auroc(scores, labels):
    scores, labels = np.asarray(scores).ravel(), np.asarray(labels).ravel().astype(bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def trigger_metrics(scores, gt_chunk_risk, step_risk, tau, H, lead=1):
    """Step-level precision/recall/FAR + episode-level timely recall & lead time."""
    trig = scores > tau
    lab = gt_chunk_risk > 1.0
    tp, fp = (trig & lab).sum(), (trig & ~lab).sum()
    fn, tn = (~trig & lab).sum(), (~trig & ~lab).sum()
    viol = step_risk > 1.0
    ep_v = viol.any(1)
    first_v = np.where(ep_v, viol.argmax(1), -1)
    timely, leads = [], []
    for i in np.where(ep_v)[0]:
        fv = first_v[i]
        win = trig[i, max(0, fv - H + 1): max(0, fv - lead + 1)]
        ok = bool(win.any())
        timely.append(ok)
        if ok:
            leads.append(fv - (max(0, fv - H + 1) + int(np.argmax(win))))
    safe_ep = ~ep_v
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "auroc": auroc(scores, lab),
        "timely_recall": float(np.mean(timely)) if timely else float("nan"),
        "mean_lead_steps": float(np.mean(leads)) if leads else float("nan"),
        "episode_false_alarm": float(trig[safe_ep].any(1).mean()) if safe_ep.any() else float("nan"),
        "n_viol_episodes": int(ep_v.sum()),
        "n_steps_pos": int(lab.sum()),
    }
