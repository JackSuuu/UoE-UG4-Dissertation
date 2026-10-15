"""Read-only, post-launch A/B audit of M1; does not redefine its endpoint.

A: trace contract observations and baseline failures. B: paired full-task effects.
Only hash-checked, replay-verified completed episodes enter outcome denominators.
Interrupted/unverified cases remain explicit. Timing is a partial lower bound.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path


ARMS = ('vla', 'gate_hold', 'timed_repair', 'phase_repair', 'verified_repair')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def timing(values):
    ordered = sorted(values)
    if not ordered:
        return None
    # Empirical nearest-rank percentiles; not a worst-case execution bound.
    return dict(n=len(values), median_s=ordered[math.ceil(.5*len(values))-1],
        p95_s=ordered[math.ceil(.95*len(values))-1], max_s=ordered[-1],
        over_50ms=sum(v > .05 for v in values), deadline_reference_only_s=.05)


def episode(directory):
    meta = json.loads((directory/'manifest.json').read_text())
    proof_path = directory/'verification.json'
    if meta['status'] != 'complete' or not proof_path.exists():
        return None, dict(path=str(directory), status=meta['status'],
            replay_verified=False, reason='incomplete_or_missing_replay_proof')
    proof = json.loads(proof_path.read_text())
    assert proof['status'] == 'verified' and not proof['fixture_only']
    required = {'manifest.json', 'steps.jsonl', 'result.json'}
    assert required <= proof['artifact_sha256'].keys()
    for name, sha in proof['artifact_sha256'].items():
        assert digest(directory/name) == sha, directory/name
    rows = [json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
    result = json.loads((directory/'result.json').read_text())
    assert len(rows) == result['steps']
    streak = maximum = 0
    for i, row in enumerate(rows):
        assert row['step'] == i
        checks = row['measured']['conditions']
        assert row['measured']['candidate'] == all(checks.values())
        streak = streak+1 if all(checks.values()) else 0
        maximum = max(maximum, streak)
        assert row['tracking']['final_streak'] == streak
        assert (streak >= 20) == row['tracking']['complete']
        if streak >= 20:
            assert i == len(rows)-1
    assert result['complete'] == (streak >= 20)
    assert result['complete'] or len(rows) == meta['max_steps'] == 600
    live = [r for r in rows if r['live_query']]
    changed = [r['step'] for r in rows if r['correction']]
    # This excludes state measurement, gate, capture, queueing and actuator latency.
    # Forecasts on control arms are shadow instrumentation, not active algorithm time.
    lower_bound = [r['actor_seconds'] + (
        r['forecast']['wall_seconds'] if r['forecast'] else 0) for r in live]
    handoff = result['handoff']
    return dict(case_id=result['case_id'], arm=result['arm'], root=meta['case']['root'],
        steps=result['steps'], complete=result['complete'], native_ever=result['native_ever'],
        endpoint='satisfied' if result['complete'] else 'full_task_timeout',
        longest_completion_dwell_intervals=maximum,
        terminal_unmet_checks=[k for k,v in rows[-1]['measured']['conditions'].items() if not v],
        terminal_checks_are_not_causal_failure_labels=True,
        trigger=result['trigger'], first_changed_action=changed[0] if changed else None,
        trigger_to_first_change_intervals=None if not changed or result['trigger'] is None
            else changed[0]-result['trigger'],
        changed_actions=len(changed), handoff=handoff,
        task_completed_after_handoff=None if handoff is None else result['complete'],
        actor_wall=timing([r['actor_seconds'] for r in live]),
        instrumented_actor_plus_forecast_wall_lower_bound=timing(lower_bound),
        end_to_end_latency=None, hard_realtime_proven=False,
        safety_compliance=None, validated_collision_rate=None,
        proof_sha256=digest(proof_path)), None


def audit(run):
    grid = json.loads((run/'cases.json').read_text())
    launch = json.loads((run/'launch.json').read_text())
    reports, missing = {}, []
    for case in grid['cases']:
        for arm in ARMS:
            directory = run/case['case_id']/arm
            if not (directory/'manifest.json').exists():
                missing.append(dict(case_id=case['case_id'], arm=arm, status='not_started'))
                continue
            report, issue = episode(directory)
            if report:
                reports[case['case_id'], arm] = report
            else:
                missing.append(dict(case_id=case['case_id'], arm=arm, **issue))
    baseline = []
    for label in sorted({c['label'] for c in grid['cases']}):
        cases = [c for c in grid['cases'] if c['label'] == label]
        available = [reports[c['case_id'], 'vla'] for c in cases
                     if (c['case_id'], 'vla') in reports]
        n = len(available)
        failures = sum(not r['complete'] for r in available)
        baseline.append(dict(configuration=label, registered_roots=len(cases),
            verified_roots=n, full_task_failures=failures,
            observed_failure_fraction=None if not n else failures/n,
            development_30_70_flag=None if n != len(cases) else .3 <= failures/n <= .7,
            flag_is_not_eligibility_or_confirmation=True))
    comparisons = []
    for method, control in itertools.combinations(ARMS, 2):
        # Later arm is the candidate; baseline ordering has no effect on inclusion.
        method, control = control, method
        for family in ('all_available', 'sham', 'release_delay', 'pose_bias'):
            bins = {k: [] for k in ('both_success', 'rescue', 'harm', 'both_failure')}
            for c in grid['cases']:
                if family != 'all_available' and not (
                    (family == 'sham' and c['duration'] == 0) or
                    (family == c['fault'] and c['duration'] > 0)):
                    continue
                a, b = reports.get((c['case_id'], method)), reports.get((c['case_id'], control))
                if a is None or b is None:
                    continue
                x, y = a['complete'], b['complete']
                bins['both_success' if x and y else 'rescue' if x else 'harm' if y else 'both_failure'].append(c['case_id'])
            n = sum(map(len, bins.values()))
            comparisons.append(dict(method=method, control=control, family=family,
                verified_pairs=n, cases=bins,
                paired_delta=None if not n else (len(bins['rescue'])-len(bins['harm']))/n))
    return dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        analysis_registration='post-launch descriptive audit; frozen M1 endpoint unchanged',
        supervisor_status=launch['status'], supervisor_error=launch.get('error'),
        grid_sha256=digest(run/'cases.json'), verified_episodes=len(reports),
        line_A=dict(episodes=list(reports.values()), baseline_configuration_map=baseline),
        line_B=dict(paired_full_task_effects=comparisons), unresolved_or_unstarted=missing,
        limitations=['three consumed development roots, correlated variants',
            'partial pairs, not an estimate for all 39 cases',
            'release opportunity is not an independently labeled failure alarm',
            'terminal unmet checks are not failure causes',
            'safety and end-to-end realtime unvalidated'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['supervisor_status'], verified=report['verified_episodes'],
        outcomes_by_arm={arm: dict(Counter('pass' if r['complete'] else 'fail'
            for r in report['line_A']['episodes'] if r['arm'] == arm)) for arm in ARMS}), indent=2))
