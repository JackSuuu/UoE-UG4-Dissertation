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
                     grad_every=0, act_noise=0.0, chunk_k=1, ctrl_scales=(),
                     ctrl_every=5):
    """verifiers: {role: VerifierAdapter}. Uses verifier.score for trigger scores
    and verifier.predictor.risk for the raw predicted risk.

    chunk_k: steps between policy calls (open-loop chunk execution). chunk_k=1
    replans every step (closed-loop); chunk_k>1 executes each chunk open-loop.

    ctrl_scales: if non-empty, additionally score every proposed chunk scaled by
    each of these factors and record both the predicted and the GT chunk risk.
    Measures *controllability* -- whether the signal repair's bisection search
    actually walks down -- which AUROC does not capture.

    ctrl_every: measure it on every ctrl_every'th step only. Controllability is
    a property of the predictor's response curve, not of individual chunks, so
    subsampling costs almost no precision -- and it has to be subsampled,
    because a GT shadow rollout is ~5x a real step, so probing every step would
    add several times the cost of the whole rest of the experiment.
    """
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
    rec_ctrl = {k: [] for k in verifiers} if ctrl_scales else {}
    gt_chunk, gt_ctrl, step_risk, succ = [], [], [], None
    probes = []
    g = torch.Generator().manual_seed(seed + 99)
    plan = torch.zeros(B, policy.H, policy.act_dim, device=dev)
    plan_ptr, plan_left = 0, 0                      # uniform across envs: plain ints
    chunk = None
    for t in range(sim.T):
        img = env.render()
        if plan_left == 0:
            chunk = policy(obs, env.render_rgb() if needs_rgb else img)
            plan = chunk.clone()
            plan_ptr, plan_left = 0, min(chunk_k, chunk.shape[1])
        ctx = {"obs": obs, "obs_prev": obs_prev, "a_prev": a_prev,
               "img": img, "img_prev": img_prev, "state": env.get_state(), "instruction": None}
        a = plan[:, min(plan_ptr, plan.shape[1] - 1)]
        plan_ptr, plan_left = plan_ptr + 1, plan_left - 1
        for k, v in verifiers.items():
            rec[k].append(v.score(ctx, chunk))
            mean, _ = v.predictor.risk(ctx, chunk)
            rec_max[k].append(mean.amax(dim=(1, 2)))
        probe_ctrl = bool(ctrl_scales) and (t % ctrl_every == 0)
        if probe_ctrl:
            for k, v in verifiers.items():
                # Repair controllability: does the score actually fall when the
                # chunk is scaled down? AUROC only measures detection, but the
                # repair search bisects on exactly this signal, so a predictor
                # can rank violations well and still be uncontrollable. Stacked
                # within the timestep so the array is (B, T', n_scales), in the
                # order of ctrl_scales.
                rec_ctrl[k].append(torch.stack([v.score(ctx, chunk * sc)
                                                for sc in ctrl_scales], 1))
        gt_chunk.append(env.shadow_rollout(chunk).amax(dim=(1, 2)))
        if probe_ctrl:
            gt_ctrl.append(torch.stack(
                [env.shadow_rollout(chunk * sc).amax(dim=(1, 2)) for sc in ctrl_scales], 1))
        if grad_every and t % grad_every == 0 and params is not None and can_probe:
            pr = grad_probe(sim, params, env.get_state(), chunk, probe_pred,
                            ctx, GOAL_DIMS[sim.name])
            pr["t"] = t
            probes.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in pr.items()})
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
    if ctrl_scales:
        out["ctrl_scales"] = list(ctrl_scales)
        out["ctrl_pred"] = {k: torch.stack(v, 1).cpu().numpy() for k, v in rec_ctrl.items()}
        out["ctrl_gt"] = torch.stack(gt_ctrl, 1).cpu().numpy()   # (B, T, n_scales)
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


def controllability(ctrl_pred, ctrl_gt):
    """Can the repair search actually walk the predicted score down?

    AUROC says "a violating chunk scores high". It says nothing about whether
    halving the chunk lowers the score, which is the only thing the bisection in
    ``scale_repair`` relies on. A predictor can therefore rank violations well
    and still be useless for repair -- its response to action magnitude is flat
    or even increasing, so the search converges on a scale that is not actually
    safer.

    ``ctrl_pred`` / ``ctrl_gt`` are (n, n_scales) arrays ordered by decreasing
    chunk scale (1.0 first). Returns the fraction of monotone descents and a
    Spearman correlation of score against scale (negative = falls as the chunk
    shrinks, which is what the bisection needs), next to the same two measured
    on GT.

    A *relative* drop is deliberately not reported. The predicted score carries
    a near-constant offset and can be near zero at full scale on individual
    chunks, so (s[0]-s[-1])/s[0] has an unbounded denominator: on real chunks
    it produced means of -2127 and -593, which say nothing. Spearman and the
    monotone fraction are scale-free and stable.
    """
    p = np.asarray(ctrl_pred, dtype=float)
    g = np.asarray(ctrl_gt, dtype=float)
    n = p.shape[0]
    if n == 0 or p.shape[1] < 2:
        return {"pred_monotone": float("nan"), "pred_spearman": float("nan"),
                "gt_monotone": float("nan"), "gt_spearman": float("nan"), "n": int(n)}

    def _stats(x):
        mono = float((x[:, :-1] > x[:, 1:]).mean())
        rho = []
        for row in x:
            if np.ptp(row) > 0:
                rho.append(_spearman(row, np.arange(len(row))))
        return mono, float(np.mean(rho)) if rho else float("nan")

    pm, pr = _stats(p)
    gm, gr = _stats(g)
    return {"pred_monotone": pm, "pred_spearman": pr,
            "gt_monotone": gm, "gt_spearman": gr, "n": int(n)}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")
