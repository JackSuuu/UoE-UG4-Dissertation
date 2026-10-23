"""Export execution-verified recovery data and matched candidate coverage."""
import argparse
import json
from pathlib import Path
import sys

import run_framework_comparison as comparison
sys.path.insert(0, str(comparison.REPO))
from src.training.recovery_dataset import record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = comparison.load(args.run/'results.json')
    training = comparison.load(args.run/'protocol.json').get('data_split') == 'training'
    report = dict(schema='repair_candidate_coverage_report_v1', cases=[], contrasts=[],
        development_only=True, learned_ranking_tested=False, infrastructure_failures=[],
        limitations=['single H1 alternative at first repair only', 'two reused failed development roots',
            'root1 sentinel does not test repair selection because no repair is expected',
            'coverage is an offline upper bound, not a deployable oracle selector'])
    if training:
        report.update(data_split='training', limitations=[
            'one microwave task, ten training initial states; no held-out efficacy claim',
            'single H1 alternative at first admitted repair; siblings grouped by root',
            'bare nominal pays no auxiliary queries; candidate arms pay equal pool cost'])
        report['skipped_siblings'] = comparison.load(args.run/'skipped_siblings.json')
    records, tapes, statuses = [], {}, {}
    for job in jobs:
        source = Path(job['source'])
        status = comparison.load(source/'status.json') if (source/'status.json').exists() else {}
        steps = comparison.rows(source/'steps.jsonl')
        controls = comparison.rows(source/'controller.jsonl')
        rp = args.run/'replays'/job['name']/'verification.json'
        ap = source/'joint_recomputation.json'
        replay = comparison.load(rp) if rp.exists() else {}
        audit = comparison.load(ap) if ap.exists() else {}
        replay_passed = replay.get('status') == 'exact_replay_passed'
        audit_passed = (audit.get('status') == 'joint_recomputation_passed' or
            (training and job['slot'] is None and status.get('cell', {}).get('arm') == 'baseline'))
        valid = bool(status.get('valid_episode') and replay_passed and audit_passed)
        case = dict(name=job['name'], root=job['root'], slot=job['slot'], valid=valid,
            success=status.get('any_native_success') if valid else None,
            actions=status.get('steps_executed'), queries=status.get('actor_queries'),
            repair_actions=status.get('controller_summary', {}).get('repair_actions'),
            pools=status.get('candidate_pools'), replay_frames=replay.get('frames_checked', 0),
            joint_observations=audit.get('observations_recomputed', 0),
            handoff_scores=audit.get('handoff_scores_recomputed', 0))
        report['cases'].append(case)
        if not valid:
            report['infrastructure_failures'].append(job['name'])
        if any('candidate_selection' in r for r in steps):
            records.append(record(status, steps, controls, str(source), replay_passed, audit_passed))
        tapes[job['name']], statuses[job['name']] = steps, status
    for root in sorted({j['root'] for j in jobs}):
        control = tapes[f'candidate0_root{root}']
        controls = [c for c in report['cases'] if c['root'] == root]
        control_case = next(c for c in controls if c['slot'] == 0)
        for case in controls:
            if case['slot'] in (None, 0) or not case['valid'] or not control_case['valid']:
                continue
            other = tapes[case['name']]
            probes = [r for r in control if 'candidate_selection' in r]
            first = probes[0]['step'] if probes else None
            matched_pool = None
            if first is not None:
                comparison.require(len(other) > first and 'candidate_selection' in other[first], 'Probe schedule drift')
                def pool_signature(row):
                    return dict(before=row['before'], context=row['candidate_selection']['context'], candidates=[
                        {k: c['decode'][k] for k in ('source', 'seed', 'action', 'native_action', 'tokens',
                            'policy_png_sha256', 'policy_rgb_sha256', 'prompt', 'sampling')}
                        for c in row['candidate_selection']['candidates']])
                matched_pool = pool_signature(control[first]) == pool_signature(other[first])
                comparison.require(matched_pool, 'Candidate pool differs at the matched pre-action state')
                comparison.require(all(a['action'] == b['action'] and a['before'] == b['before']
                    for a, b in zip(control[:first], other[:first])), 'Pre-intervention prefix drift')
            else:
                comparison.require(len(control) == len(other) and all(a['action'] == b['action'] and
                    a['before'] == b['before'] for a, b in zip(control, other)), 'Inactive candidate path changed tape')
            report['contrasts'].append(dict(root=root, slot=case['slot'], first_probe=first,
                matched_pool=matched_pool, rescue=not control_case['success'] and case['success'],
                harm=control_case['success'] and not case['success'],
                action_tape_equal=len(control) == len(other) and all(a['action'] == b['action'] for a,b in zip(control, other))))
    report['candidate_coverage'] = [dict(root=root,
        unique_first_actions=len({tuple(c['decode']['action']) for r in records if r['root'] == root
            for c in r['candidates']}),
        evaluated=sum(c['valid'] for c in report['cases'] if c['root'] == root and c['slot'] is not None),
        successful_slots=[c['slot'] for c in report['cases'] if c['root'] == root and c['slot'] is not None and c['success'] is True])
        for root in sorted({r['root'] for r in records})]
    if training:
        report['nominal_contrasts'] = []
        for case in report['cases']:
            if case['slot'] is None:
                continue
            base = next(c for c in report['cases'] if c['root'] == case['root'] and c['slot'] is None)
            valid = base['valid'] and case['valid']
            report['nominal_contrasts'].append(dict(root=case['root'], slot=case['slot'], valid=valid,
                rescue=(not base['success'] and case['success']) if valid else None,
                harm=(base['success'] and not case['success']) if valid else None))
    report['totals'] = {key: sum(c[key] or 0 for c in report['cases']) for key in
        ('actions', 'queries', 'repair_actions', 'replay_frames', 'joint_observations', 'handoff_scores')}
    report['status'] = 'passed' if not report['infrastructure_failures'] else 'contains_invalid_cases'
    with (args.out/'recovery_dataset.jsonl').open('x') as stream:
        for r in records:
            stream.write(json.dumps(r, allow_nan=False)+'\n')
    report['dataset_sha256'] = comparison.sha(args.out/'recovery_dataset.jsonl')
    comparison.save(args.out/'report.json', report)
    lines = ['# Repair-only candidate coverage', '',
        '|Root|Slot|Valid|Complete success|Actions|Queries|Repair actions|', '|---|---|---|---|---|---|---|']
    lines += ['|'+'|'.join(str(c[k]) for k in ('root','slot','valid','success','actions','queries','repair_actions'))+'|'
        for c in report['cases']]
    lines += ['', 'Candidate coverage: `'+json.dumps(report['candidate_coverage'])+'`', '',
        'Selection was fixed-slot collection, not learned ranking. All outcomes are development evidence.']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
