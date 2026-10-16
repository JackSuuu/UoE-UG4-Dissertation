"""Read-only A2 status: verified results, unresolved runs and paired outcomes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main(root):
    launch = json.loads((root/'launch.json').read_text())
    verified, unresolved = [], []
    for case in launch['planned']:
        folder = root/f"live_s{case['seed']}_c{case['case']}_{case['arm']}"
        if not (folder/'replay.json').exists():
            unresolved.append(dict(**case, started=folder.exists()))
            continue
        status = json.loads((folder/'status.json').read_text())
        replay = json.loads((folder/'replay.json').read_text())
        digest = hashlib.sha256((folder/'trace.jsonl').read_bytes()).hexdigest()
        assert digest == replay['trace_sha256'] == status['trace_sha256']
        assert replay['status'] == 'exact_replay_passed' and not status['fixture_only']
        verified.append(dict(**status, cohort=case['cohort']))
    groups, pairs = [], []
    for cohort in ('consumed_regression', 'prospective_development'):
        for arm in ('vla', 'feedback80', 'phase_geometry80', 'hybrid80'):
            rows = [r for r in verified if r['cohort'] == cohort and r['arm'] == arm]
            counts = Counter()
            for r in rows: counts.update(r['counters'])
            groups.append(dict(cohort=cohort, arm=arm, verified=len(rows), success=sum(r['complete'] for r in rows),
                corrections=sum(r['changed'] for r in rows), counters=dict(counts),
                mean_peak_load_N=sum(r['peak_nut_peg_load_N'] for r in rows)/len(rows) if rows else None))
        for comparison in ('vla', 'feedback80', 'phase_geometry80'):
            lookup = {(r['seed'], r['case'], r['arm']): r for r in verified if r['cohort'] == cohort}
            counts = Counter()
            for (seed, case, arm), result in lookup.items():
                if arm != 'hybrid80' or (seed, case, comparison) not in lookup: continue
                other = lookup[seed, case, comparison]
                label = ('both_success' if other['complete'] else 'rescue') if result['complete'] else ('harm' if other['complete'] else 'both_failure')
                counts[label] += 1
            pairs.append(dict(cohort=cohort, hybrid_vs=comparison, outcomes=dict(counts)))
    audit = root/'actual_state_audit.json'
    report = dict(status=launch['status'], phase=launch['phase'], supervisor=launch['supervisor'],
        audit_trajectories=len(list(root.glob('audit_s*.json'))), audit_complete=audit.exists(),
        verified=len(verified), planned=len(launch['planned']), groups=groups, pairs=pairs,
        unresolved=unresolved, failed_jobs=[j for j in launch['jobs'] if j.get('exit_code', 0) != 0])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
