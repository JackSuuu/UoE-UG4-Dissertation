"""Offline decision diagnostics; no controller decisions or model updates."""
import numpy as np
import torch

SCALES = (1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.15, 0.05)


@torch.no_grad()
def label_chunk(sim, state, params, chunk, execution_horizon):
    """Pure functional labels, not a live Env rollout or runtime verifier input."""
    state = state.clone()
    risks, distances = [], []
    for t in range(chunk.shape[1]):
        state, risk = sim.step(state, chunk[:, t], params)
        risks.append(risk.amax(1))
        if t + 1 in (execution_horizon, chunk.shape[1]):
            distances.append(sim.obs(state)[:, :2].norm(dim=1))
    if execution_horizon == chunk.shape[1]:
        distances.append(distances[0])
    return torch.stack(risks, -1), torch.stack(distances, -1)


@torch.no_grad()
def predict_chunk(predictor, ctx, chunk, execution_horizon, beta=1.0):
    mean, std = predictor.risk(ctx, chunk)
    risk = (mean + beta * std).amax(-1)
    obs_seq, _ = predictor.rollout(ctx['obs'], ctx['obs_prev'], ctx['a_prev'], chunk)
    distances = torch.stack([
        obs_seq[:, :, t - 1, :2].norm(dim=-1).mean(0)
        for t in (execution_horizon, chunk.shape[1])], -1)
    return risk, distances


def decision_summary(data, execution_horizon):
    """Counts are decision-local evidence, not exclusive episode root causes.

    Pre-first-violation includes the decision whose execution contains that
    violation; it excludes decisions after irrecoverable episode safety loss.
    A safe ten-step forecast describes continuing that candidate open-loop,
    not the subsequent replanned policy. Candidate sufficiency refers only to
    this finite set. Ground-truth progress uses goal distance, not full success.
    """
    gt = data['candidate_gt_risk']
    pred = data['candidate_pred_risk']
    trigger = data['trigger'].astype(bool)
    first = data['first_viol']
    pre = (first[None, :] < 0) | (data['decision_t'][:, None] <= first[None, :])
    active = ~data['seated_at_decision'].astype(bool)
    # Keep seated/zero-action stages: a proposal can introduce a new hazard
    # after reaching the goal. Report stage counts rather than dropping them.
    scope = pre
    result = {'n_decisions': int(trigger.size),
              'n_triggered_in_scope': int((scope & trigger).sum()), 'horizons': {}}
    result['n_pre_first_violation_all_stages'] = int(pre.sum())
    result['n_pre_first_violation_unseated'] = int((pre & active).sum())
    result['n_pre_first_violation_seated'] = int((pre & ~active).sum())
    for name, h in (('prefix', execution_horizon), ('lookahead', gt.shape[-1])):
        g, p = gt[..., :h].max(-1), pred[..., :h].max(-1)
        unsafe = g[..., 0] > 1
        applied_g = data['applied_gt_risk'][..., :h].max(-1)
        applied_p = data['applied_pred_risk'][..., :h].max(-1)
        safe_exists = (g <= 1).any(-1)
        target = np.minimum(0.8, 0.75 * pred.max(-1)[..., 0])
        # Deployment candidate modes filter full-H score, even when evaluating
        # whether their accepted candidates were safe over the executed prefix.
        accepted = np.isfinite(pred.max(-1)) & (pred.max(-1) <= target[..., None])
        accepted_in_scope = accepted & scope[..., None]
        inversions = comparable = 0
        for i in range(len(SCALES)):
            for j in range(i + 1, len(SCALES)):
                dg, dp = g[..., i] - g[..., j], p[..., i] - p[..., j]
                valid = scope & np.isfinite(dp) & (np.abs(dg) > 1e-4) & (np.abs(dp) > 1e-4)
                comparable += int(valid.sum())
                inversions += int((valid & (dg * dp < 0)).sum())
        finite = np.isfinite(p[..., 0]) & scope
        first_idx = gt[..., 0, :h].argmax(-1)
        miss = scope & unsafe & ~trigger
        r = {
            'n_unsafe_proposals': int((scope & unsafe).sum()),
            'true_trigger': int((scope & unsafe & trigger).sum()),
            'missed_trigger': int(miss.sum()),
            'safe_proposal_trigger': int((scope & ~unsafe & trigger).sum()),
            'safe_proposal_no_trigger': int((scope & ~unsafe & ~trigger).sum()),
            'trigger_recall': float((scope & unsafe & trigger).sum() / (scope & unsafe).sum()) if (scope & unsafe).any() else None,
            'trigger_precision': float((scope & unsafe & trigger).sum() / (scope & trigger).sum()) if (scope & trigger).any() else None,
            'missed_unsafe_with_safe_scale_available': int((miss & safe_exists).sum()),
            'triggered_applied_unsafe_with_safe_scale_available': int((scope & trigger & (applied_g > 1) & safe_exists).sum()),
            'triggered_applied_unsafe_without_safe_scale': int((scope & trigger & (applied_g > 1) & ~safe_exists).sum()),
            'triggered_safe_proposal_made_unsafe_locally': int((scope & trigger & ~unsafe & (applied_g > 1)).sum()),
            'triggered_applied_above_relative_target': int((scope & trigger & (applied_p > target)).sum()),
            'n_predicted_accepted_candidates': int(accepted_in_scope.sum()),
            'n_predicted_accepted_but_GT_unsafe': int((accepted_in_scope & (g > 1)).sum()),
            'risk_ranking_inversions': inversions,
            'risk_ranking_comparable_pairs': comparable,
            'risk_ranking_inversion_rate': inversions / max(comparable, 1),
            'proposal_score_mae': float(np.abs(p[..., 0][finite] - g[..., 0][finite]).mean()) if finite.any() else None,
            'missed_unsafe_peak_step_0': int((miss & (first_idx == 0)).sum()),
            'missed_unsafe_peak_later': int((miss & (first_idx > 0)).sum()),
            'missed_trigger_while_seated': int((miss & ~active).sum()),
            'local_induced_hazard_while_seated': int((scope & ~active & trigger & ~unsafe & (applied_g > 1)).sum()),
        }
        dist_i = 0 if name == 'prefix' else 1
        pd, gd = data['candidate_pred_goal_dist'][..., dist_i], data['candidate_gt_goal_dist'][..., dist_i]
        # Compare progress rankings only where both progress choices pass the
        # fixed learned-risk filter and are genuinely safe in this label scope.
        eligible = accepted & (g <= 1) & np.isfinite(pd) & np.isfinite(gd)
        enough = scope & (eligible.sum(-1) >= 2)
        pred_choice = np.where(eligible, pd, np.inf).argmin(-1)
        true_best = np.where(eligible, gd, np.inf).min(-1)
        chosen_g = np.take_along_axis(gd, pred_choice[..., None], -1)[..., 0]
        regret = chosen_g[enough] - true_best[enough]
        r['progress_comparable_decisions'] = int(enough.sum())
        r['progress_choice_GT_regret_mean_m'] = float(regret.mean()) if regret.size else None
        r['progress_choice_GT_regret_gt_1mm_count'] = int((regret > 0.001).sum())
        result['horizons'][name] = r
    if 'order_probe_pred_risk' in data:
        original_p = pred[..., 0, :].max(-1)
        original_g = gt[..., 0, :].max(-1)
        probe_p = data['order_probe_pred_risk'].max(-1)
        probe_g = data['order_probe_gt_risk'].max(-1)
        preserved = np.abs(probe_p - original_p[..., None]) <= 1e-5
        changed_label = (probe_g > 1) != (original_g[..., None] > 1)
        examined = np.broadcast_to(scope[..., None], probe_p.shape)
        result['order_probe'] = {
            'n_pairs': int(examined.sum()),
            'score_preserved_at_1e_minus_5': int((examined & preserved).sum()),
            'GT_safety_label_changed_with_score_preserved': int((examined & preserved & changed_label).sum()),
            'GT_risk_changed_gt_0_1_with_score_preserved': int((examined & preserved & (np.abs(probe_g - original_g[..., None]) > 0.1)).sum()),
            'interpretation': 'offline reversed/half-shifted chunks; not proposed runtime repairs'}
    return result


def verify_replay(actual, expected):
    """Fail closed if an audit did not reproduce the source trajectories."""
    for key in ('success', 'violation', 'first_viol', 'first_trig', 'n_interventions', 'first_success'):
        if key in expected and not np.array_equal(actual[key], np.asarray(expected[key])):
            raise RuntimeError(f'replay differs from source episode outcomes: {key}')
    if 'final_task_cost' in expected and not np.allclose(
            actual['final_task_cost'], expected['final_task_cost'], rtol=1e-5, atol=1e-7):
        raise RuntimeError('replay differs from source final task costs')
