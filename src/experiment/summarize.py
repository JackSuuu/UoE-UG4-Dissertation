"""Summarize paired episode directories; infrastructure errors are not failures.

Expected layout: ROOT/rootN/baseline and ROOT/rootN/<treatment>. Exact simulator
replay is not implemented; this checks recorded pairing metadata/goal equality.
"""
import argparse
import json
from pathlib import Path


def episode(directory):
    status_path = directory/'status.json'
    if not status_path.exists():
        return dict(status='missing')
    status = json.loads(status_path.read_text())
    if status['status'] != 'completed':
        return dict(status=status['status'], error=status.get('error', status.get('cleanup_error')))
    result = json.loads((directory/'result.json').read_text())
    native_task = status['instruction'] == status['benchmark_instruction']
    return dict(status='completed', generated_complete=result['controller']['complete'],
                native_success=result['native_success'], actions=result['controller']['budget']['actions'],
                primary_complete=result['native_success'] is True if native_task else result['controller']['complete'],
                primary_metric='native_success' if native_task else 'generated_complete',
                queries=result['controller']['budget']['queries'], termination=result['termination'],
                repairs=len(result['controller']['repairs']),
                verified_handoffs=sum(r['verified'] for r in result['controller']['repairs']),
                pair_key=dict(initial_state=status['initial_states_sha256'], bddl=status['bddl_sha256'],
                    init_index=status['args']['init_index'], seed=status['args']['seed'],
                    instruction=status['instruction'], policy=status.get('policy'), limits=status['limits'],
                    benchmark_instruction=status['benchmark_instruction'],
                    planner={k: status['args'][k] for k in ('planner_url', 'planner_model', 'planner_calls',
                                                           'planner_timeout', 'rollout_seconds')},
                    source=status['source_sha256'],
                    milestones=sorted(result['controller']['task']['milestones'], key=lambda p: json.dumps(p, sort_keys=True)),
                    goals=sorted(result['controller']['task']['goals'], key=lambda p: json.dumps(p, sort_keys=True))))


def summarize(directory, treatment):
    pairs = []
    for root in sorted(directory.glob('root*')):
        if not root.is_dir():
            continue
        baseline, framework = episode(root/'baseline'), episode(root/treatment)
        paired = baseline['status'] == framework['status'] == 'completed'
        if paired:
            paired = baseline['pair_key'] == framework['pair_key']
        outcome = 'unpaired_or_infrastructure'
        if paired:
            a, b = baseline['primary_complete'], framework['primary_complete']
            outcome = 'rescue' if b and not a else 'harm' if a and not b else 'both_success' if a else 'both_failure'
        pairs.append(dict(root=root.name, baseline=baseline, treatment=framework, paired=paired, outcome=outcome))
    return dict(treatment=treatment, pairs=pairs,
                counts={name: sum(p['outcome'] == name for p in pairs)
                        for name in ('rescue', 'harm', 'both_success', 'both_failure', 'unpaired_or_infrastructure')},
                replay_audited=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--treatment', choices=('measured', 'context', 'full'), default='context')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    report = json.dumps(summarize(args.root, args.treatment), indent=2)
    if args.out:
        with args.out.open('x') as stream:
            stream.write(report+'\n')
    else:
        print(report)


if __name__ == '__main__':
    main()
