"""Read-only A1 progress and replay-qualified paired outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


def main(root):
    launch = json.loads((root/'launch.json').read_text())
    data = list(root.glob('data_[0-9]*.json'))
    report = dict(status=launch['status'], phase=launch['phase'],
        supervisor=launch['supervisor'], collected_episodes=len(data), collection_target=40,
        planned=len(launch['planned']), verified=[], unresolved=[], paired=[])
    by_case = {}
    for case in launch['planned']:
        path = root/f"live_s{case['seed']}_c{case['case']}_{case['arm']}"
        if not (path/'status.json').exists():
            report['unresolved'].append(dict(**case, state='unstarted'))
            continue
        status = json.loads((path/'status.json').read_text())
        if not (path/'replay.json').exists():
            trace = path/'trace.jsonl'
            lines = trace.read_text().splitlines() if trace.exists() else []
            valid = []
            for line in lines:
                try: valid.append(json.loads(line))
                except json.JSONDecodeError: break  # actively written last line
            report['unresolved'].append(dict(**case, state=status['status'],
                logged_steps=len(valid), changed=sum(r.get('changed', False) for r in valid)))
            continue
        proof = json.loads((path/'replay.json').read_text())
        digest = hashlib.sha256((path/'trace.jsonl').read_bytes()).hexdigest()
        assert proof['status'] == 'exact_replay_passed' and digest == proof['trace_sha256'] == status['trace_sha256']
        result = dict(**case, complete=status['complete'], steps=status['steps'],
            changed=status['changed'], peak_load_N=status['peak_nut_peg_load_N'])
        report['verified'].append(result)
        by_case.setdefault((case['seed'], case['case']), {})[case['arm']] = status['complete']
    for key, outcomes in by_case.items():
        if 'vla' not in outcomes:
            continue
        for arm in ('framework', 'feedback'):
            if arm in outcomes:
                base, treatment = outcomes['vla'], outcomes[arm]
                label = ('both_success' if treatment else 'harm') if base else ('rescue' if treatment else 'both_failure')
                report['paired'].append(dict(seed=key[0], case=key[1], arm=arm, outcome=label))
    report['arm_totals'] = {arm: dict(verified=sum(r['arm'] == arm for r in report['verified']),
        successes=sum(r['arm'] == arm and r['complete'] for r in report['verified']))
        for arm in ('vla', 'framework', 'feedback')}
    training = root/'training.json'
    if training.exists():
        t = json.loads(training.read_text())
        report['predictor'] = dict(status=t['status'], tau=t['tau'],
            validation_xyz_mae_m=t['metrics']['validation']['xyz_mae_m'],
            validation_risk_mae=t['metrics']['validation']['risk_mae'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
