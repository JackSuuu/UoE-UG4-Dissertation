"""CPU-only attribution of completed validation audits; no policy or GT control.

Decision counts are correlated, arm-specific local witnesses, not exclusive
episode causes. Full-H labels are counterfactual beyond the execution prefix.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope_mask(data):
    return ((data['first_viol'][None] < 0) |
            (data['decision_t'][:, None] <= data['first_viol'][None]))


def local_counts(data, scope, horizon, distance_index):
    gt = data['candidate_gt_risk'][..., :horizon].max(-1)
    pred = data['candidate_pred_risk'].max(-1)
    original = gt[..., 0]
    applied = data['applied_gt_risk'][..., :horizon].max(-1)
    trigger = data['trigger'].astype(bool)
    safe_exists = (gt <= 1).any(-1)
    target = np.minimum(0.8, 0.75 * pred[..., 0])
    accepted = np.isfinite(pred) & (pred <= target[..., None])
    eligible = accepted & (gt <= 1)
    unsafe = original > 1

    def count(mask):
        return int((scope & mask).sum())

    pd = data['candidate_pred_goal_dist'][..., distance_index]
    gd = data['candidate_gt_goal_dist'][..., distance_index]
    eligible &= np.isfinite(pd) & np.isfinite(gd)
    comparable = scope & (eligible.sum(-1) >= 2)
    selected = np.where(eligible, pd, np.inf).argmin(-1)
    chosen = np.take_along_axis(gd, selected[..., None], -1)[..., 0]
    best = np.where(eligible, gd, np.inf).min(-1)
    regret = chosen[comparable] - best[comparable]
    changed = np.max(np.abs(data['applied'][..., :5, :] -
                            data['proposal'][..., :5, :]), axis=(-1, -2)) > 1e-7
    return {
        'decisions': int(scope.sum()),
        'unsafe_proposals': count(unsafe),
        'unsafe_with_safe_scale': count(unsafe & safe_exists),
        'missed': count(unsafe & ~trigger),
        'missed_with_safe_scale': count(unsafe & ~trigger & safe_exists),
        'missed_without_safe_scale': count(unsafe & ~trigger & ~safe_exists),
        'true_triggers': count(unsafe & trigger),
        'triggered_unsafe_made_safe': count(unsafe & trigger & (applied <= 1)),
        'triggered_applied_unsafe_with_safe_scale': count(trigger & (applied > 1) & safe_exists),
        'triggered_applied_unsafe_without_safe_scale': count(trigger & (applied > 1) & ~safe_exists),
        'safe_proposal_triggers': count(~unsafe & trigger),
        'safe_proposal_trigger_changed_prefix_actions': count(~unsafe & trigger & changed),
        'local_induced_hazards': count(~unsafe & trigger & (applied > 1)),
        'triggered_unsafe_with_safe_scale_but_none_accepted': count(
            unsafe & trigger & safe_exists & ~eligible.any(-1)),
        'accepted_candidates': int((accepted & scope[..., None]).sum()),
        'accepted_unsafe_candidates': int((accepted & (gt > 1) & scope[..., None]).sum()),
        'progress_comparable': int(comparable.sum()),
        'progress_regret_sum_m': float(regret.sum()),
        'progress_regret_gt_1mm': int((regret > 0.001).sum()),
        'triggered_progress_comparable': int((comparable & trigger).sum()),
        'triggered_progress_regret_gt_1mm': int((comparable & trigger & (chosen - best > 0.001)).sum()),
    }


def episode_witness(data, baseline, episode, execution_horizon=5):
    """Classify terminal differences without pretending a local branch is causal."""
    i = episode
    safe_b = bool(baseline['success'][i] and not baseline['violation'][i])
    safe_m = bool(data['success'][i] and not data['violation'][i])
    changed = np.max(np.abs(data['applied'][:, i, :execution_horizon] -
                            baseline['applied'][:, i, :execution_horizon]), axis=(-1, -2)) > 1e-7
    indices = np.flatnonzero(changed)
    divergence = None
    if indices.size:
        di = int(indices[0])
        divergence = {
            'decision_index': di, 't': int(data['decision_t'][di]),
            'state_matches_baseline': bool(np.allclose(data['state'][di, i], baseline['state'][di, i], rtol=0, atol=1e-6)),
            'proposal_matches_baseline': bool(np.allclose(data['proposal'][di, i], baseline['proposal'][di, i], rtol=0, atol=1e-6)),
            'trigger': bool(data['trigger'][di, i]),
            'original_prefix_risk': float(data['candidate_gt_risk'][di, i, 0, :execution_horizon].max()),
            'applied_prefix_risk': float(data['applied_gt_risk'][di, i, :execution_horizon].max()),
            'original_prefix_distance_m': float(data['candidate_gt_goal_dist'][di, i, 0, 0]),
            'applied_prefix_distance_m': float(data['applied_gt_goal_dist'][di, i, 0]),
        }
    fv = int(data['first_viol'][i])
    first_violation = None
    if fv >= 0:
        di = int(np.searchsorted(data['decision_t'], fv, side='right') - 1)
        g = data['candidate_gt_risk'][di, i, :, :execution_horizon].max(-1)
        tr = bool(data['trigger'][di, i])
        if g[0] <= 1:
            kind = 'safe_proposal_applied_unsafe'
        else:
            kind = ('triggered' if tr else 'missed') + ('_with_safe_scale' if (g <= 1).any() else '_without_safe_scale')
        first_violation = {
            'decision_index': di, 'decision_t': int(data['decision_t'][di]), 'violation_t': fv,
            'category': kind, 'trigger': tr,
            'earlier_trigger': bool(data['trigger'][:di, i].any()),
            'seated': bool(data['seated_at_decision'][di, i]),
            'original_prefix_risk': float(g[0]), 'min_scale_prefix_risk': float(g.min()),
            'original_peak_step': int(data['candidate_gt_risk'][di, i, 0, :execution_horizon].argmax()),
            'original_first_unsafe_step': int(np.flatnonzero(data['candidate_gt_risk'][di, i, 0, :execution_horizon] > 1)[0]) if g[0] > 1 else None,
        }
    return {
        'episode_row': i, 'baseline_success': bool(baseline['success'][i]),
        'baseline_violation': bool(baseline['violation'][i]),
        'success': bool(data['success'][i]), 'violation': bool(data['violation'][i]),
        'safe_rescue': not safe_b and safe_m, 'safe_harm': safe_b and not safe_m,
        'harm_success_loss': safe_b and not safe_m and not bool(data['success'][i]),
        'harm_safety_loss': safe_b and not safe_m and bool(data['violation'][i]),
        'first_divergence': divergence, 'first_violation': first_violation,
        'final_task_cost_delta': float(data['final_task_cost'][i] - baseline['final_task_cost'][i]),
    }


def add_counts(total, counts):
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def analyze(audit_path):
    audit_path = Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text())
    source_path = Path(audit['metadata']['source_result'])
    source = json.loads(source_path.read_text())
    if not audit['complete'] or not source['complete'] or source['metadata']['args']['split'] != 'validation':
        raise ValueError('requires a completed audit of completed validation, never confirmation/test')
    if sha256(source_path) != audit['metadata']['source_result_sha256']:
        raise ValueError('source validation hash mismatch')
    if set(audit['metadata']['audit_cell_indices']) != set(source['metadata']['evaluated_cell_indices']):
        raise ValueError('requires full-grid replay, not an interesting subset')
    if source['metadata']['args']['chunk_k'] != 5:
        raise ValueError('this analysis requires five-step execution')
    out = {'complete': True, 'audit_sha256': sha256(audit_path),
           'analysis_sha256': sha256(__file__), 'source_result': str(source_path),
           'source_sha256': sha256(source_path), 'raw_sha256': {},
           'pooled_ood': {}, 'ood_episode_witnesses': [], 'per_cell': [],
           'source_paired_intervals': source['paired_vs_none'],
           'limitations': ['decision/probe samples are correlated and arm-trajectory-specific',
                           'full-H beyond step 5 is counterfactual, not observed execution',
                           'no-safe-scale describes only the eight tested scales, not all possible actions',
                           'safe lookahead triggers are not automatically causally harmful',
                           'progress diagnostics condition on the learned filter and true local safety',
                           'terminal harm/rescue is paired but local witnesses are not causal attribution']}
    source_indices = source['metadata']['evaluated_cell_indices']
    for cell in audit['per_cell']:
        ci = cell['grid_index']
        source_cell = source['per_cell'][source_indices.index(ci)]
        if cell['cell'] != source_cell['cell'] or cell['ood'] != source_cell['ood']:
            raise ValueError('audit/source cell mismatch')
        arrays = {}
        for arm, entry in cell['arms'].items():
            if not entry['replay_matches_source'] or not entry['executed_prefix_matches_offline_labels']:
                raise ValueError('replay consistency check failed')
            raw = Path(entry['raw_path'])
            out['raw_sha256'][raw.name] = sha256(raw)
            with np.load(raw, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            expected = source_cell['episodes'][arm]
            for key in ('success', 'violation', 'first_viol', 'first_trig', 'n_interventions', 'first_success'):
                if not np.array_equal(data[key], expected[key]):
                    raise ValueError(f'raw/source outcome mismatch: {raw.name} {key}')
            for di, t in enumerate(data['decision_t']):
                if not np.allclose(data['applied_gt_risk'][di, :, :5].T, data['live_risk'][t:t+5], rtol=1e-5, atol=1e-6):
                    raise ValueError('raw prefix/live risk mismatch')
            arrays[arm] = data
        cell_out = {'grid_index': ci, 'cell': cell['cell'], 'ood': cell['ood'], 'arms': {}}
        for arm, data in arrays.items():
            scope = scope_mask(data)
            stages = {'all': scope, 'unseated': scope & ~data['seated_at_decision'],
                      'seated': scope & data['seated_at_decision']}
            counts = {stage: {h: local_counts(data, mask, n, j)
                             for h, n, j in (('prefix', 5, 0), ('lookahead', 10, 1))}
                      for stage, mask in stages.items()}
            rows = [episode_witness(data, arrays['none'], i) for i in range(len(data['success']))]
            terminal = {'episodes': len(rows), 'success': int(data['success'].sum()),
                        'violation': int(data['violation'].sum()),
                        'safe_success': int((data['success'] & ~data['violation']).sum()),
                        'safe_rescue': sum(r['safe_rescue'] for r in rows),
                        'safe_harm': sum(r['safe_harm'] for r in rows),
                        'harm_success_loss': sum(r['harm_success_loss'] for r in rows),
                        'harm_safety_loss': sum(r['harm_safety_loss'] for r in rows),
                        'first_violation_categories': {}}
            for row in rows:
                if row['first_violation']:
                    kind = row['first_violation']['category']
                    terminal['first_violation_categories'][kind] = terminal['first_violation_categories'].get(kind, 0) + 1
            cell_out['arms'][arm] = {'decisions': counts, 'terminal': terminal}
            if cell['ood']:
                pooled = out['pooled_ood'].setdefault(arm, {'decisions': {}, 'terminal': {}})
                for stage, horizons in counts.items():
                    for h, values in horizons.items():
                        add_counts(pooled['decisions'].setdefault(stage, {}).setdefault(h, {}), values)
                for key, value in terminal.items():
                    if isinstance(value, dict):
                        add_counts(pooled['terminal'].setdefault(key, {}), value)
                    else:
                        pooled['terminal'][key] = pooled['terminal'].get(key, 0) + value
                for row in rows:
                    row.update(grid_index=ci, cell=cell['cell'], arm=arm)
                    out['ood_episode_witnesses'].append(row)
        out['per_cell'].append(cell_out)
    for arm, pooled in out['pooled_ood'].items():
        for stage in pooled['decisions'].values():
            for h in stage.values():
                h['progress_regret_mean_m'] = h['progress_regret_sum_m'] / h['progress_comparable'] if h['progress_comparable'] else None
        if arm != 'none':
            paired = source['paired_vs_none'][arm]
            for key in ('safe_rescue', 'safe_harm'):
                if pooled['terminal'][key] != paired[key + '_count']:
                    raise ValueError('paired harm/rescue mismatch')
    return out


def markdown_report(out):
    lines = ['# Frozen validation audit attribution', '',
             'Full grid: 20 cells × 32 episodes × 4 arms. Main tables: 14 OOD cells, 448 episodes/arm.',
             'No weights, thresholds, controller, candidate family or cadence changed.', '',
             '## Outcomes (paired validation, not new efficacy experiments)', '',
             '| arm | SR % | CVR % | safe % | rescue / harm | harm: success / safety loss |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, p in out['pooled_ood'].items():
        t = p['terminal']; n = t['episodes']
        lines.append(f"| {arm} | {100*t['success']/n:.2f} | {100*t['violation']/n:.2f} | {100*t['safe_success']/n:.2f} | {t['safe_rescue']} / {t['safe_harm']} | {t['harm_success_loss']} / {t['harm_safety_loss']} |")
    lines += ['', 'Success/safety loss can overlap. Fixed-grid paired safe-success intervals (percentage points):']
    for arm, pair in out['source_paired_intervals'].items():
        interval = pair['paired_intervals']['safe_success']
        lo, hi = interval['ci95']
        lines.append(f"- {arm}: {100*interval['estimate']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}].")
    lines += ['', '## Local diagnostics before first violation (including its decision)', '',
              '| arm / horizon | unsafe | true trigger / miss | misses with safe scale | triggered applied unsafe: safe scale / none |',
              '|---|---:|---:|---:|---:|']
    for arm, p in out['pooled_ood'].items():
        for horizon, h in p['decisions']['all'].items():
            lines.append(f"| {arm} / {horizon} | {h['unsafe_proposals']} | {h['true_triggers']} / {h['missed']} | {h['missed_with_safe_scale']} | {h['triggered_applied_unsafe_with_safe_scale']} / {h['triggered_applied_unsafe_without_safe_scale']} |")
    lines += ['', '## First actual violation (one witness per violating episode)', '']
    for arm, p in out['pooled_ood'].items():
        lines.append(f"- {arm}: `{json.dumps(p['terminal']['first_violation_categories'], sort_keys=True)}`")
    lines += ['', '## Stage and progress checks', '']
    for arm, p in out['pooled_ood'].items():
        h = p['decisions']['all']['prefix']
        s = p['decisions']['seated']['prefix']; u = p['decisions']['unseated']['prefix']
        lines.append(f"- {arm}: safe-prefix triggers seated/unseated = {s['safe_proposal_triggers']}/{u['safe_proposal_triggers']}; >1 mm progress regret among learned-accepted, truly safe choices = {h['progress_regret_gt_1mm']}/{h['progress_comparable']}; triggered subset = {h['triggered_progress_regret_gt_1mm']}/{h['triggered_progress_comparable']}.")
    lines += ['', '## Paired harm/rescue witnesses', '',
              'All paired rows and first divergent execution-prefix/first-violation details are in analysis.json.',
              'A safe first divergent prefix does not certify subsequent replanned execution.']
    for arm in out['pooled_ood']:
        if arm == 'none':
            continue
        for label in ('safe_harm', 'safe_rescue'):
            rows = [r for r in out['ood_episode_witnesses'] if r['arm'] == arm and r[label]]
            ds = [r['first_divergence'] for r in rows if r['first_divergence']]
            lines.append(f"- {arm} {label}: {len(rows)}; first divergent prefix applied unsafe = {sum(d['applied_prefix_risk'] > 1 for d in ds)}; same state/proposal = {sum(d['state_matches_baseline'] and d['proposal_matches_baseline'] for d in ds)}/{len(ds)}.")
    lines += ['', '## Interpretation limits', ''] + ['- ' + s for s in out['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit_summary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = analyze(args.audit_summary)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    (output / 'report.md').write_text(markdown_report(out))
    print(markdown_report(out))


if __name__ == '__main__':
    main()
