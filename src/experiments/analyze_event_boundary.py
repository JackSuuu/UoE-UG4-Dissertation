"""Root-grouped boundary coverage, collector outcomes and gate diagnostics."""
import argparse
from pathlib import Path

from _paired import paired_outcomes, bootstrap_delta
from audit_verifier import atomic_json
from checkvla.reference import calibrate_tau
from collect_event_gate_pilot import coverage
from common import load_json
from validate_sequence_risk import digest
import numpy as np
import torch


def detection(scores, y, roots, tau):
    """Root bootstrap preserves all correlated decisions of a paired branch."""
    scores, y, roots = np.asarray(scores), np.asarray(y, bool), np.asarray(roots)
    pred = scores > tau
    unique = np.unique(roots)
    counts = np.array([[np.sum((roots == r) & y & pred), np.sum((roots == r) & y),
                        np.sum((roots == r) & ~y & pred), np.sum((roots == r) & ~y),
                        np.sum((roots == r) & pred)] for r in unique], dtype=float).reshape(-1, 5)
    sums = counts.sum(0)
    out = {'threshold': float(tau), 'positive_decisions': int(y.sum()),
           'positive_roots': len(np.unique(roots[y])), 'safe_roots': len(np.unique(roots[~y])),
           'false_triggers': int((pred & ~y).sum()), 'missed_events': int((~pred & y).sum())}
    out['adequate_for_diagnostic_comparison'] = out['positive_roots'] >= 12 and out['safe_roots'] >= 32
    rng = np.random.default_rng(1300)
    boot = counts[rng.integers(len(unique), size=(1000, len(unique)))].sum(1) if len(unique) else np.empty((0, 5))
    for name, numerator, denominator in [('recall', 0, 1), ('safe_trigger_rate', 2, 3), ('precision', 0, 4)]:
        valid = boot[:, denominator] > 0
        out[name] = float(sums[numerator] / sums[denominator]) if sums[denominator] else None
        out[name + '_root_ci95'] = np.quantile(boot[valid, numerator] / boot[valid, denominator], [.025, .975]).tolist() if valid.any() else None
    out['interval_scope'] = 'root-cluster bootstrap, conditional on this diagnostic collector and reused training roots'
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True)
    cli = p.parse_args()
    run = Path(cli.run)
    output = run / 'analysis.json'
    if output.exists():
        raise FileExistsError(output)
    s = load_json(run / 'summary.json')
    assert s['complete']
    arm_data, arm_outcomes = {}, {}
    for meta in s['batches']:
        assert digest(meta['path']) == meta['sha256']
        assert digest(meta['outcomes_path']) == meta['outcomes_sha256']
        d = torch.load(meta['path'], map_location='cpu', weights_only=True)
        o = torch.load(meta['outcomes_path'], map_location='cpu', weights_only=True)
        assert coverage(d) == meta['coverage']
        torch.testing.assert_close(o['violation'], o['live_risk'].amax(0) > 1, rtol=0, atol=0)
        n = len(o['violation'])
        live = o['live_risk'].reshape(16, 5, n).permute(0, 2, 1).reshape(-1, 5)
        torch.testing.assert_close(d['applied_risk'][:, :5], live, rtol=1e-5, atol=1e-6)
        prior = torch.cat([torch.zeros(1, n, dtype=torch.bool), (o['live_risk'] > 1).cummax(0).values[:-1]])[::5].reshape(-1)
        torch.testing.assert_close(d['prior_violation'], prior, rtol=0, atol=0)
        arm_data.setdefault(meta['arm'], []).append(d)
        arm_outcomes.setdefault(meta['arm'], []).append(o)
    data = {arm: {k: torch.cat([d[k] for d in ds]) for k in ds[0]} for arm, ds in arm_data.items()}
    outcomes = {arm: {k: torch.cat([o[k] for o in os]) for k in ('success', 'violation')}
                for arm, os in arm_outcomes.items()}
    # This split is for diagnostic threshold fitting only, not new model training
    # or independent validation. All branches of the root use the same half.
    root_ids = data['none']['episode_id'].unique(sorted=True)
    fit_ids = root_ids[:len(root_ids) // 2]
    fit_data = {k: torch.cat([d[k][torch.isin(d['episode_id'], fit_ids) & ~d['prior_violation']]
                             for d in data.values()]) for k in ('scores', 'risk')}
    result = {'complete': True, 'smoke': s['smoke'], 'source_summary_sha256': digest(run / 'summary.json'),
              'analysis_code_sha256': digest(__file__), 'verified_artifact_hashes': 2 * len(s['batches']),
              'fit_roots': fit_ids.tolist(), 'diagnostic_check_roots': root_ids[len(root_ids) // 2:].tolist(),
              'scope': 'training-root design diagnosis; no efficacy claim, promotion or automatic training',
              'arms': {}}
    for arm, d in data.items():
        cov = coverage(d)
        pre = ~d['prior_violation']
        r = d['risk'][:, 0].amax(-1)
        post = (d['pulse_t'] >= 0) & (d['decision_t'] > d['pulse_t']) & pre
        next_decision = post & (d['decision_t'] == d['pulse_t'] + 5)
        anticip = (r[:, :5].amax(-1) <= 1) & (r.amax(-1) > 1)
        pulse = d['collector_pulse'] & pre
        survived = pulse & (d['applied_risk'][:, :5].amax(-1) <= 1)
        bins = (d['physical_mults'][:, 0] >= 1.).long() * 2 + (d['physical_mults'][:, 1] >= 1.1).long()
        per_bin = [int(d['episode_id'][post & anticip & (bins == b)].unique().numel()) for b in range(4)]
        n_anticip = int(d['episode_id'][post & anticip].unique().numel())
        survival = float(survived.sum() / pulse.sum()) if pulse.any() else None
        o, b = outcomes[arm], outcomes['none']
        pair = paired_outcomes(b, o)
        pair['paired_intervals'] = {}
        for metric, bv, mv in [('SR', b['success'], o['success']), ('CVR', b['violation'], o['violation']),
                               ('safe_success', b['success'] & ~b['violation'], o['success'] & ~o['violation'])]:
            interval = bootstrap_delta([(mv.float() - bv.float()).numpy()])
            interval['interval_scope'] = 'paired roots under sampled in-range physics; reused training cohort'
            pair['paired_intervals'][metric] = interval
        report = {'coverage': cov, 'SR': float(o['success'].float().mean()),
                  'CVR': float(o['violation'].float().mean()),
                  'safe_success': float((o['success'] & ~o['violation']).float().mean()),
                  'paired_vs_none': pair, 'pre_violation_pulses': int(pulse.sum()),
                  'pulse_survival_rate': survival,
                  'post_pulse_anticipation_roots': n_anticip,
                  'post_pulse_anticipation_roots_per_bin': per_bin,
                  'next_decision_anticipation_roots': int(d['episode_id'][next_decision & anticip].unique().numel()),
                  'reachability_screen_pass': bool(not s['smoke'] and n_anticip >= 24 and min(per_bin) >= 2
                                                    and survival is not None and survival >= .9),
                  'gates': {}}
        check = pre & ~torch.isin(d['episode_id'], fit_ids)
        for mi, name in enumerate(s['protocol']['model_order']):
            score = d['scores'][:, mi].amax(-1)
            fit_score = fit_data['scores'][:, mi].amax(-1)
            safe_fit = fit_data['risk'][:, 0].amax((-1, -2)) <= 1
            thresholds = {'frozen': s['protocol']['frozen_taus'][name]}
            if safe_fit.any():
                for alpha in (.01, .05):
                    thresholds[f'design_safe_fpr_{alpha}'] = calibrate_tau(fit_score[safe_fit].numpy(), alpha)
            gate = {'event_detection': {}, 'order_probe': {}}
            for horizon, label in [('R5', r[:, :5].amax(-1) > 1), ('H10', r.amax(-1) > 1)]:
                gate['event_detection'][horizon] = {key: detection(score[check], label[check], d['episode_id'][check], tau)
                                                    for key, tau in thresholds.items()}
            # Anticipation recall alone; false positives are defined against all
            # H10-safe originals, not against immediate hazards.
            anti_scope = check & ((r.amax(-1) <= 1) | anticip)
            gate['event_detection']['anticipation_vs_H10_safe'] = {
                key: detection(score[anti_scope], anticip[anti_scope], d['episode_id'][anti_scope], tau)
                for key, tau in thresholds.items()}
            changed = (r.amax(-1) > 1) != (d['reverse_risk'].amax(-1) > 1)
            same_score = (score - d['reverse_scores'][:, mi].amax(-1)).abs() <= 1e-5
            gate['order_probe'] = {'pre_decisions': int(pre.sum()),
                                  'gt_H10_event_flips': int((pre & changed).sum()),
                                  'event_flip_with_score_invariant': int((pre & changed & same_score).sum()),
                                  'event_flip_roots': int(d['episode_id'][pre & changed].unique().numel()),
                                  'scope': 'synthetic reversed actions, offline only; does not prove prevalence in actor proposals'}
            report['gates'][name] = gate
        result['arms'][arm] = report
        print(f'[boundary-analysis] {arm} anticipation_roots={n_anticip} pulse_survival={survival} screen={report["reachability_screen_pass"]}', flush=True)
    atomic_json(result, output)


if __name__ == '__main__':
    main()
