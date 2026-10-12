"""Replay a frozen validation run with offline decision/candidate simulator labels.

No model fitting, threshold search, extra replanning, or runtime GT feedback.
All original episode rows are replayed to preserve appearance and batch math.
Raw per-cell/per-arm NPZs precede a summary; source outcome agreement is required.
"""
import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from _common import (env_for_cell, experiment_metadata, load_policy, load_predictor, setup)
import numpy as np
import torch
from common import load_json, save_json
from rq2_eval import build_controller
from _audit import SCALES, decision_summary, label_chunk, predict_chunk, verify_replay


def atomic_json(data, path):
    save_json(data, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def pooled_counts(cells, arms):
    """Counts, not means of per-cell ratios; stages/trajectories vary by arm."""
    result = {}
    for arm in arms:
        rows = [c['arms'][arm]['summary'] for c in cells if c['ood']]
        if not rows:
            continue
        out = {'n_cells': len(rows), 'horizons': {},
               'decision_scope': 'pre_first_violation_all_stages_on_each_arm_own_trajectory'}
        for key in ('n_decisions', 'n_pre_first_violation_all_stages',
                    'n_pre_first_violation_unseated', 'n_pre_first_violation_seated', 'n_triggered_in_scope'):
            out[key] = sum(r[key] for r in rows)
        for horizon in ('prefix', 'lookahead'):
            first = rows[0]['horizons'][horizon]
            h = {key: sum(r['horizons'][horizon][key] for r in rows)
                 for key, value in first.items() if type(value) is int}
            h['trigger_recall'] = h['true_trigger'] / h['n_unsafe_proposals'] if h['n_unsafe_proposals'] else None
            n_trigger = h['true_trigger'] + h['safe_proposal_trigger']
            h['trigger_precision'] = h['true_trigger'] / n_trigger if n_trigger else None
            h['risk_ranking_inversion_rate'] = h['risk_ranking_inversions'] / max(h['risk_ranking_comparable_pairs'], 1)
            out['horizons'][horizon] = h
        out['order_probe'] = {key: sum(r['order_probe'][key] for r in rows)
                              for key, value in rows[0]['order_probe'].items() if type(value) is int}
        result[arm] = out
    return result


@torch.no_grad()
def replay(env, sim, controller, predictor, params, seed, expected, execution_horizon):
    obs = env.reset(params, seed)
    controller.reset(env, obs)
    B = obs.shape[0]
    decision_t, trace, live_risk = [], {}, []
    first_viol = torch.full((B,), -1, dtype=torch.long, device=obs.device)
    first_trig, first_success = first_viol.clone(), first_viol.clone()
    n_interventions = torch.zeros(B, device=obs.device)
    violation = torch.zeros(B, dtype=torch.bool, device=obs.device)

    def append(key, value):
        trace.setdefault(key, []).append(value.detach().cpu().numpy().copy())

    for t in range(sim.T):
        # Capture context before Controller.act updates its observation/action
        # history. No extra camera rendering or policy calls are introduced.
        free = controller.plan_left <= 0
        if free.any() and not free.all():
            raise RuntimeError('audit expects synchronous matched chunk decisions')
        if free.all():
            state = env.get_state().clone()
            ctx = {'obs': obs.clone(), 'obs_prev': controller.obs_prev.clone(),
                   'a_prev': controller.a_prev.clone()}
        action, info = controller.act(env, obs)
        if free.all():
            decision_t.append(t)
            proposal, applied = info['chunk'].clone(), controller.plan.clone()
            append('state', state)
            for key, value in ctx.items():
                append(key, value)
            append('proposal', proposal)
            append('applied', applied)
            append('trigger', info['trig'])
            append('seated_at_decision', sim.success(state))
            cr, cg, pr, pg = [], [], [], []
            for scale in SCALES:
                candidate = proposal * scale
                gr, gg = label_chunk(sim, state, params, candidate, execution_horizon)
                rr, rg = predict_chunk(predictor, ctx, candidate, execution_horizon)
                cr.append(gr); cg.append(gg); pr.append(rr); pg.append(rg)
            append('candidate_gt_risk', torch.stack(cr, 1))
            append('candidate_gt_goal_dist', torch.stack(cg, 1))
            append('candidate_pred_risk', torch.stack(pr, 1))
            append('candidate_pred_goal_dist', torch.stack(pg, 1))
            order_g, order_p = [], []
            # Same action multiset, different temporal order. The current
            # static-context max-risk formulation should be order-invariant;
            # GT risk need not be. These are diagnostic probes, never actions.
            for probe in (proposal.flip(1), proposal.roll(execution_horizon, 1)):
                gr, _ = label_chunk(sim, state, params, probe, execution_horizon)
                rr, _ = predict_chunk(predictor, ctx, probe, execution_horizon)
                order_g.append(gr); order_p.append(rr)
            append('order_probe_gt_risk', torch.stack(order_g, 1))
            append('order_probe_pred_risk', torch.stack(order_p, 1))
            gr, gg = label_chunk(sim, state, params, applied, execution_horizon)
            rr, rg = predict_chunk(predictor, ctx, applied, execution_horizon)
            append('applied_gt_risk', gr)
            append('applied_gt_goal_dist', gg)
            append('applied_pred_risk', rr)
            append('applied_pred_goal_dist', rg)
            if not torch.equal(env.get_state(), state):
                raise RuntimeError('offline diagnostic modified live simulator state')
        obs, risk, success = env.step(action)
        live_risk.append(risk.amax(1).cpu().numpy())
        v = risk.amax(1) > 1
        first_viol = torch.where(v & (first_viol < 0), t, first_viol)
        first_trig = torch.where(info['trig'] & (first_trig < 0), t, first_trig)
        first_success = torch.where(success & (first_success < 0), t, first_success)
        n_interventions += info['trig'].float()
        violation |= v
    data = {key: np.stack(value) for key, value in trace.items()}
    data.update(decision_t=np.asarray(decision_t), live_risk=np.stack(live_risk),
                success=success.cpu().numpy(), violation=violation.cpu().numpy(),
                first_viol=first_viol.cpu().numpy(), first_trig=first_trig.cpu().numpy(),
                first_success=first_success.cpu().numpy(),
                n_interventions=n_interventions.cpu().numpy(),
                final_task_cost=sim.task_cost(env.get_state()).cpu().numpy())
    verify_replay(data, expected)
    expected_calls = int(np.ceil(sim.T / execution_horizon))
    if controller.n_policy_calls != expected_calls or controller.n_policy_rows != expected_calls * B:
        raise RuntimeError('audit replay changed policy-call cadence')
    # The first five GT-labeled actions must reproduce the ACTUAL executed
    # segment; full-H labels are counterfactual and never get that claim.
    for di, t in enumerate(decision_t):
        length = min(execution_horizon, sim.T - t)
        if not np.allclose(data['applied_gt_risk'][di, :, :length].T,
                           data['live_risk'][t:t + length], rtol=1e-5, atol=1e-6):
            raise RuntimeError('offline prefix labels do not match actual executed risk')
    return data


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source_result', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--cell_indices', type=int, nargs='+', default=None)
    cli = p.parse_args()
    source_path = Path(cli.source_result).expanduser().resolve()
    source = load_json(source_path)
    if not source.get('complete') or source['metadata']['args']['split'] != 'validation':
        raise ValueError('audit source must be a completed validation run, not confirmation/test')
    args = SimpleNamespace(**source['metadata']['args'])
    if args.task != 'push' or args.backend != 'torch' or args.hard_prefix or args.use_grad:
        raise ValueError('this audit requires frozen push/torch, no gradient/hard-prefix repair')
    if args.repair_mode != 'bisection' or args.chunk_k != 5:
        raise ValueError('audit supports the matched five-step candidate validation protocol')
    for name, artifact in source['metadata']['artifacts'].items():
        file = Path(artifact['path'])
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != artifact['sha256']:
            raise RuntimeError(f'source artifact/code has changed: {name}')
    args.device = cli.device
    dev, sim, od = setup(args)
    output = Path(od) / f'audit_{cli.tag}'
    output.mkdir(exist_ok=False)
    policy = load_policy(args, od, dev, sim)
    predictor = load_predictor(args, od, 'orbisim', dev, sim)
    taus = load_json(args.calibration_path)['tau']
    arms = [a for a in source['arms'] if a == 'none' or a.startswith('checkvla_orbisim')]
    ctrls = {a: build_controller(a, policy, sim, od, dev, taus, args) for a in arms}
    source_indices = source['metadata']['evaluated_cell_indices']
    chosen = cli.cell_indices if cli.cell_indices is not None else source_indices
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen).issubset(source_indices):
        raise ValueError('audit indices must be a distinct subset of source grid indices')
    metadata = experiment_metadata(args, od, policy, args.calibration_path)
    metadata.update(source_result=str(source_path),
                    source_result_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    audit_cell_indices=chosen, candidate_scales=SCALES,
                    audit_code_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                       for f in (Path(__file__), Path(__file__).with_name('_audit.py'))},
                    scope='offline diagnostic, no algorithm/threshold selection',
                    raw_array_layout='decision, episode, candidate, horizon (where applicable)',
                    ranking_tie_tolerance=1e-4, progress_regret_threshold_m=0.001,
                    order_probes=['reverse', 'circular_shift_by_execution_horizon'],
                    order_score_tolerance=1e-5, order_GT_risk_difference_threshold=0.1,
                    primary_decision_scope='before_first_violation_all_stages_including_seated',
                    reported_latency='none; diagnostic runtime is not controller latency')
    summary = {'complete': False, 'metadata': metadata, 'per_cell': []}
    atomic_json(summary, output / 'summary.partial.json')
    execution_horizon = min(args.chunk_k, policy.H)
    for ci in chosen:
        entry = source['per_cell'][source_indices.index(ci)]
        env, params = env_for_cell(args, args.n_envs, dev, entry['cell'])
        cell_result = {'grid_index': ci, 'cell': entry['cell'], 'ood': entry['ood'], 'arms': {}}
        for arm, ctrl in ctrls.items():
            data = replay(env, sim, ctrl, predictor, params, entry['episode_seed'],
                          entry['episodes'][arm], execution_horizon)
            base = entry['episodes']['none']
            safe_b = np.asarray(base['success']) & ~np.asarray(base['violation'])
            safe_m = data['success'] & ~data['violation']
            data['safe_rescue_vs_none'] = ~safe_b & safe_m
            data['safe_harm_vs_none'] = safe_b & ~safe_m
            raw_path = output / f'cell_{ci:02d}_{arm}.npz'
            with open(str(raw_path) + '.tmp', 'wb') as f:
                np.savez_compressed(f, **data)
            os.replace(str(raw_path) + '.tmp', raw_path)
            cell_result['arms'][arm] = {
                'replay_matches_source': True, 'executed_prefix_matches_offline_labels': True,
                'raw_path': str(raw_path), 'summary': decision_summary(data, execution_horizon)}
            atomic_json(cell_result, output / f'cell_{ci:02d}.partial.json')
            print(f'[audit] cell={ci} {entry["cell"]} arm={arm} replay=matched '
                  f'decisions={len(data["decision_t"])} saved={raw_path}', flush=True)
        summary['per_cell'].append(cell_result)
        atomic_json(summary, output / 'summary.partial.json')
    summary['complete'] = True
    summary['pooled_ood_counts'] = pooled_counts(summary['per_cell'], arms)
    atomic_json(summary, output / 'summary.json')
    for arm, counts in summary['pooled_ood_counts'].items():
        h = counts['horizons']['prefix']
        print(f'[audit-summary] {arm}: prefix true_trigger={h["true_trigger"]} '
              f'miss={h["missed_trigger"]} safe_proposal_trigger={h["safe_proposal_trigger"]} '
              f'local_induced_hazard={h["triggered_safe_proposal_made_unsafe_locally"]}', flush=True)
    print(f'[audit] complete; raw labels and summaries: {output}', flush=True)


if __name__ == '__main__':
    main()
