"""Actual-repair premise audit and automatic full-grid transition analysis."""
import argparse
import os
from pathlib import Path

from _audit import verify_replay
from _paired import pooled_paired
from audit_verifier import atomic_json
from common import load_json
from validate_sequence_risk import digest
import numpy as np


def audit_actual(audit_path, factorial_path, output):
    if output.exists():
        raise FileExistsError(output)
    audit, factorial = load_json(audit_path), load_json(factorial_path)
    assert audit['complete'] and factorial['complete']
    rows = []
    for row in audit['per_cell']:
        ci = row['grid_index']
        assert row['cell'] == factorial['per_cell'][ci]['cell']
        meta = row['arms']['checkvla_orbisim']
        with np.load(meta['raw_path'], allow_pickle=False) as d:
            verify_replay(d, factorial['per_cell'][ci]['episodes']['frozen_ll'])
            changed = np.max(np.abs(d['applied'][:, :, :5] - d['proposal'][:, :, :5]), axis=(2, 3)) > 1e-7
            post = np.concatenate([np.zeros_like(changed[:1]), changed[:-1]])
            release = post & ~changed
            first, t = d['first_viol'][None], d['decision_t'][:, None]
            pre = (first < 0) | (t <= first)
            incident = (first >= t) & (first < t + 5)
            harm = d['safe_harm_vs_none'][None]
            rows.append({'grid_index': ci, 'ood': row['ood'], 'raw_sha256': digest(meta['raw_path']),
                         'episodes': len(d['success']), 'violations': int(d['violation'].sum()),
                         'safe_harms': int(harm.sum()), 'postrepair_pre_decisions': int((post & pre).sum()),
                         'postrepair_first_violations': int((post & incident).sum()),
                         'release_first_violations': int((release & incident).sum()),
                         'release_first_step_violations': int((release & (first == t)).sum()),
                         'harm_postrepair_first_violations': int((harm & post & incident).sum()),
                         'harm_release_first_violations': int((harm & release & incident).sum())})
    keys = [k for k in rows[0] if k not in ('grid_index', 'ood', 'raw_sha256')]
    report = {'complete': True, 'audit_summary_sha256': digest(audit_path),
              'factorial_summary_sha256': digest(factorial_path), 'per_cell': rows,
              'pooled_ood': {k: sum(r[k] for r in rows if r['ood']) for k in keys},
              'scope': 'temporal attribution on actual verifier trajectories, not proof of transition mechanism'}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, output)
    print('[actual-repair-audit]', report['pooled_ood'], flush=True)
    return report


def decision_counts(d):
    post = d['postrepair'].astype(bool)
    pre = ~d['prior_violation'].astype(bool)
    scope = post & pre
    incident = (d['first_viol'][None] >= d['decision_t'][:, None]) & (d['first_viol'][None] < d['decision_t'][:, None] + 5)
    raw = d['raw_risk'][:, :, :5].max(-1) > 1
    prepared = d['prepared_risk'][:, :, :5].max(-1) > 1
    applied = d['applied_risk'][:, :, :5].max(-1) > 1
    before_jump = np.linalg.norm(d['raw_proposal'][:, :, 0] - d['a_prev'], axis=-1)
    after_jump = np.linalg.norm(d['proposal'][:, :, 0] - d['a_prev'], axis=-1)
    executed_jump = np.linalg.norm(d['applied'][:, :, 0] - d['a_prev'], axis=-1)
    return {'postrepair_pre_decisions': int(scope.sum()),
            'postrepair_first_violations': int((scope & incident).sum()),
            'postrepair_first_step_violations': int((scope & (d['first_viol'][None] == d['decision_t'][:, None])).sum()),
            'raw_unsafe_prepared_safe_R5': int((scope & raw & ~prepared).sum()),
            'raw_safe_prepared_unsafe_R5': int((scope & ~raw & prepared).sum()),
            'postrepair_applied_unsafe_R5': int((scope & applied).sum()),
            'postrepair_transformed_decisions': int((scope & d['transition_changed']).sum()),
            'raw_jump_sum': float(before_jump[scope].sum()), 'prepared_jump_sum': float(after_jump[scope].sum()),
            'executed_jump_sum': float(executed_jump[scope].sum())}


def finalize(root, n_shards):
    """Each worker attempts once; the last completed worker performs aggregation."""
    paths = [root / f'shard_{i}/summary.json' for i in range(n_shards)]
    if not all(p.exists() for p in paths):
        return False
    lock = root / 'aggregate.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    os.close(fd)
    if (root / 'summary.json').exists():
        raise FileExistsError(root / 'summary.json')
    shards = [load_json(p) for p in paths]
    assert all(s['complete'] and s['metadata']['shard'] == i for i, s in enumerate(shards))
    metadata = shards[0]['metadata']
    for s in shards:
        for key in ('arms', 'smoke', 'protocol_sha256', 'code_hashes', 'source_status_sha256', 'source_evaluation_sha256', 'factorial_summary_sha256', 'actual_repair_audit_sha256'):
            assert s['metadata'][key] == metadata[key]
    rows = sorted([r for s in shards for r in s['per_cell']], key=lambda r: r['grid_index'])
    assert [r['grid_index'] for r in rows] == list(range(1 if metadata['smoke'] else 20))
    arms = metadata['arms']
    counts, hashes = {}, 0
    for row in rows:
        for arm in arms:
            raw = row['raw'][arm]
            assert digest(raw['path']) == raw['sha256']
            hashes += 1
            with np.load(raw['path'], allow_pickle=False) as d:
                for key in ('success', 'violation', 'first_viol', 'n_interventions'):
                    np.testing.assert_array_equal(d[key], row['episodes'][arm][key])
                np.testing.assert_array_equal(d['violation'], d['live_risk'].max(0) > 1)
                n = len(d['success'])
                prior = np.concatenate([np.zeros((1, n), bool), np.maximum.accumulate(d['live_risk'] > 1, axis=0)[:-1]])[::5]
                np.testing.assert_array_equal(d['prior_violation'], prior)
                np.testing.assert_allclose(d['applied_risk'][:, :, :5], d['live_risk'].reshape(16, 5, n).transpose(0, 2, 1), rtol=1e-5, atol=1e-6)
                counts[row['grid_index'], arm] = decision_counts(d)
    selected = [r for r in rows if r['ood']] or rows
    result = {'complete': True, 'metadata': metadata, 'per_cell': rows,
              'verified_raw_hashes': hashes, 'shard_summary_hashes': {str(p): digest(p) for p in paths},
              'pooled_ood': {arm: {k: float(np.mean([r[arm][k] for r in selected])) for k in ('SR', 'CVR', 'safe_success')} for arm in arms},
              'paired_vs_none': {arm: pooled_paired(rows, arm, seed=2300) for arm in arms if arm != 'none'},
              'paired_vs_direct': {}, 'paired_ramp_vs_controls': {}, 'retention_vs_direct': {},
              'decision_counts_ood': {}}
    for gate in ('learned', 'gt'):
        direct = gate + '_direct'
        entries = [{'ood': r['ood'], 'episodes': {**r['episodes'], 'none': r['episodes'][direct]}} for r in rows]
        for mode in ('ramp', 'flat', 'pause'):
            arm = gate + '_' + mode
            result['paired_vs_direct'][arm] = pooled_paired(entries, arm, seed=2400)
            retention = {'direct_safe_harms': 0, 'direct_safe_harms_reverted': 0,
                         'direct_safe_rescues': 0, 'direct_safe_rescues_retained': 0}
            for r in selected:
                def safe(name):
                    e = r['episodes'][name]
                    return np.array(e['success'], bool) & ~np.array(e['violation'], bool)
                b, d, m = safe('none'), safe(direct), safe(arm)
                harm, rescue = b & ~d, ~b & d
                retention['direct_safe_harms'] += int(harm.sum())
                retention['direct_safe_harms_reverted'] += int((harm & m).sum())
                retention['direct_safe_rescues'] += int(rescue.sum())
                retention['direct_safe_rescues_retained'] += int((rescue & m).sum())
            result['retention_vs_direct'][arm] = retention
        for control in ('flat', 'pause'):
            entries = [{'ood': r['ood'], 'episodes': {**r['episodes'], 'none': r['episodes'][gate + '_' + control]}} for r in rows]
            result['paired_ramp_vs_controls'][f'{gate}_ramp_vs_{control}'] = pooled_paired(entries, gate + '_ramp', seed=2500)
    for arm in arms:
        example = counts[selected[0]['grid_index'], arm]
        pooled = {k: sum(counts[r['grid_index'], arm][k] for r in selected) for k in example}
        for k in ('raw', 'prepared', 'executed'):
            pooled[k + '_jump_mean'] = pooled[k + '_jump_sum'] / pooled['postrepair_pre_decisions'] if pooled['postrepair_pre_decisions'] else None
        result['decision_counts_ood'][arm] = pooled
    primary = result['paired_vs_none']['learned_ramp']
    original = result['paired_vs_direct']['learned_ramp']
    result['primary_screen'] = {
        'method': 'learned_ramp (fixed primary; controls are not selected as replacement winners)',
        'incremental_algorithm_evidence': bool(not metadata['smoke'] and
            original['paired_intervals']['safe_success']['ci95'][0] > 0 and original['delta_SR'] >= -.02),
        'passes_point_and_paired_CI_screen': bool(not metadata['smoke'] and primary['delta_safe_success'] >= .03
            and primary['paired_intervals']['safe_success']['ci95'][0] > 0
            and original['paired_intervals']['safe_success']['ci95'][0] > 0 and primary['delta_SR'] >= -.02),
        'stress_cells_needing_review': [r['grid_index'] for r in rows if r['ood'] and
             (r['learned_ramp']['safe_success'] < r['none']['safe_success'] or r['learned_ramp']['SR'] < r['none']['SR'] - .02)],
        'independent_confirmation_still_required': True}
    atomic_json(result, root / 'summary.json')
    print('[transition-analysis] complete', result['pooled_ood'], result['primary_screen'], flush=True)
    return True


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--n_shards', type=int, default=3)
    p.add_argument('--audit_summary')
    p.add_argument('--factorial_summary')
    cli = p.parse_args()
    root = Path(cli.run).resolve()
    if cli.audit_summary:
        audit_actual(Path(cli.audit_summary), Path(cli.factorial_summary), root / 'actual_repair_audit.json')
    else:
        finalize(root, cli.n_shards)


if __name__ == '__main__':
    main()
