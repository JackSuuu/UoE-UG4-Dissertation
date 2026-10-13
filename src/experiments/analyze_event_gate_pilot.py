"""Read-only artifact audit and collector attribution for the coverage pilot."""
import argparse
from pathlib import Path

from collect_event_gate_pilot import COLLECTORS, coverage, passes_pilot
from validate_sequence_risk import digest
from audit_verifier import atomic_json
from common import load_json
from _paired import paired_outcomes, bootstrap_delta
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source_run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    source, output = Path(args.source_run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    summary = load_json(source / 'summary.json')
    assert summary['complete'] and not summary['smoke']
    assert len(summary['batches']) == sum(summary['protocol']['batch_counts'].values()) * len(COLLECTORS)
    rows, outcomes, initial = {}, {}, {}
    hashes = 0
    for meta in summary['batches']:
        for key in ('path', 'outcomes_path'):
            hash_key = 'sha256' if key == 'path' else 'outcomes_sha256'
            assert digest(meta[key]) == meta[hash_key]
            hashes += 1
        d = torch.load(meta['path'], map_location='cpu', weights_only=True)
        o = torch.load(meta['outcomes_path'], map_location='cpu', weights_only=True)
        split, kind, batch = meta['split'], meta['collector'], meta['batch']
        n = len(o['success'])
        assert coverage(d) == meta['coverage']
        assert meta['n_policy_calls'] == 16 and meta['n_policy_rows'] == 16 * n
        torch.testing.assert_close(o['live_risk'].amax(0) > 1, o['violation'], rtol=0, atol=0)
        risk = o['live_risk'].reshape(16, 5, n).permute(0, 2, 1).reshape(-1, 5)
        torch.testing.assert_close(d['applied_risk'][:, :5], risk, rtol=1e-5, atol=1e-6)
        prior = torch.cat([torch.zeros(1, n, dtype=torch.bool),
                           (o['live_risk'] > 1).cummax(0).values[:-1]])[::5].reshape(-1)
        torch.testing.assert_close(d['prior_violation'], prior, rtol=0, atol=0)
        snapshot = {k: d[k][:n] for k in ('state', 'obs', 'actions', 'physical_mults', 'episode_id')}
        if (split, batch) in initial:
            for k in snapshot:
                torch.testing.assert_close(snapshot[k], initial[split, batch][k], rtol=0, atol=0)
        else:
            initial[split, batch] = snapshot
        rows.setdefault((split, kind), []).append(d)
        outcomes.setdefault((split, kind), []).append(o)
    result = {'source_summary_sha256': digest(source / 'summary.json'),
              'analysis_code_sha256': digest(__file__), 'verified_artifact_hashes': hashes,
              'integrity': 'coverage, pairing, cadence, prior-violation and executed-prefix checks passed',
              'scope': 'training/design-dev collector diagnosis, not verifier efficacy', 'splits': {}}
    ids = {}
    for split in summary['coverage']:
        merged_arms, output_arms = {}, {}
        for kind in COLLECTORS:
            ds, os = rows[split, kind], outcomes[split, kind]
            d = {k: torch.cat([x[k] for x in ds]) for k in ds[0]}
            o = {k: torch.cat([x[k] for x in os]) for k in ('success', 'violation')}
            merged_arms[kind], output_arms[kind] = d, o
        all_data = {k: torch.cat([d[k] for d in merged_arms.values()]) for k in merged_arms['none']}
        cov = coverage(all_data)
        assert cov == summary['coverage'][split]['all_collectors']
        assert passes_pilot(cov, split) == summary['coverage'][split]['passes_pilot']
        ids[split] = set(all_data['episode_id'].tolist())
        report = {'coverage': cov, 'collectors': {}}
        for kind, d in merged_arms.items():
            r = d['risk'].amax(-1)
            pre = ~d['prior_violation']
            anticipation = (r[:, :, :5].amax(-1) <= 1) & (r.amax(-1) > 1)
            pulse = d['collector_pulse']
            immediate_applied = d['applied_risk'][:, :5].amax(-1) > 1
            o, base = output_arms[kind], output_arms['none']
            safe = o['success'] & ~o['violation']
            pair = paired_outcomes(base, o)
            pair['paired_intervals'] = {}
            for metric, b, m in [('SR', base['success'], o['success']),
                                  ('CVR', base['violation'], o['violation']),
                                  ('safe_success', base['success'] & ~base['violation'], safe)]:
                interval = bootstrap_delta([(m.float() - b.float()).numpy()])
                interval['interval_scope'] = 'root-paired bootstrap within this training/design split; sampled in-range physics'
                pair['paired_intervals'][metric] = interval
            report['collectors'][kind] = {
                'n_roots': len(safe), 'SR': float(o['success'].float().mean()),
                'CVR': float(o['violation'].float().mean()), 'safe_success': float(safe.float().mean()),
                'paired_vs_none': pair, 'pulse_count': int(pulse.sum()),
                'pre_violation_pulses': int((pulse & pre).sum()),
                'first_violation_during_pulse': int((pulse & pre & immediate_applied).sum()),
                'anticipation_original_pre': int((anticipation[:, 0] & pre).sum()),
                'anticipation_original_post': int((anticipation[:, 0] & ~pre).sum()),
                'anticipation_any_scale_pre_decisions': int((anticipation.any(-1) & pre).sum()),
                'anticipation_any_scale_pre_roots': int(d['episode_id'][anticipation.any(-1) & pre].unique().numel()),
            }
        result['splits'][split] = report
    assert not ids['train'] & ids['dev']
    result['split_isolation'] = True
    atomic_json(result, output)
    for split, report in result['splits'].items():
        print(split)
        for kind, row in report['collectors'].items():
            print(kind, row)


if __name__ == '__main__':
    main()
