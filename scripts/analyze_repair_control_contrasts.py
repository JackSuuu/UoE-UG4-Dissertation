"""Identify actual same-state request/action changes in the repair contrasts."""
import argparse
from collections import Counter
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def compare(a, b):
    shared = 0
    for x, y in zip(a, b):
        if x['pre_fingerprint'] != y['pre_fingerprint']:
            break
        shared += 1
    first = {}
    for key in ('language', 'owner', 'executed_action'):
        t = next((i for i, (x, y) in enumerate(zip(a, b)) if x[key] != y[key]), None)
        first[key] = dict(action_index=t, matched_pre_state=t is not None and t < shared)
        if t is not None:
            first[key].update(control=a[t][key], treatment=b[t][key])
    return dict(shared_pre_action_prefix=shared, first_changes=first,
        tapes_identical=len(a) == len(b) and first['executed_action']['action_index'] is None,
        before_first_intervention_match=next((r['action_index'] for r in b if r['owner']=='repair'), len(b)) < shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = load(args.analysis/'report.json')
    results = []
    for root in (0, 4):
        summaries, traces = {}, {}
        for arm in ('baseline', 'joint_admission', 'contract_stage', 'contract_progress'):
            summaries[arm] = next(e for e in report['episodes'] if e['root']==root and e['arm']==arm)
            traces[arm] = [json.loads(l) for l in (args.analysis/f'{arm}_root{root}.steps.jsonl').open()]
        contrasts = {}
        for control, treatment in (('baseline', 'joint_admission'), ('joint_admission', 'contract_stage'),
                                   ('contract_stage', 'contract_progress')):
            contrasts[f'{treatment}_vs_{control}'] = compare(traces[control], traces[treatment])
        arms = {}
        for arm, s in summaries.items():
            trace = traces[arm]
            arms[arm] = dict(success=s['success'], actions=s['actions'], repair_actions=s['repair_actions'],
                repairs=len(s['repairs']), verified_repairs=sum(t['verified'] for t in s['repairs']),
                milestones=s['milestones'],
                changed_from_joint_request_steps=sum(bool((r.get('request_diagnosis') or {}).get('request_changed_from_joint')) for r in trace),
                diagnostic_states=dict(Counter(r['repair_diagnosis']['diagnosis'] for r in trace if r.get('repair_diagnosis'))),
                exit_reasons=dict(Counter(t['exit_reason'] for t in s['repairs'])),
                handoffs=[{k:t.get(k) for k in ('repair_id','entry_step','used','exit_step','exit_reason','verified',
                    'handoff_step','post_handoff_goal_loss_step','completed_after_handoff')} for t in s['repairs']])
        results.append(dict(root=root, contrasts=contrasts, arms=arms))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/'report.json').open('x') as stream:
        json.dump(dict(scope='two_failed_root_development_contrasts_not_full_cohort_efficacy',roots=results), stream, indent=2)
    lines = ['# Repair-control contrasts', '', '|Root|Arm|Complete success|Actions|Repair actions|Verified repairs|', '|---|---|---|---|---|---|']
    for r in results:
        for arm, a in r['arms'].items():
            lines.append(f"|{r['root']}|{arm}|{a['success']}|{a['actions']}|{a['repair_actions']}|{a['verified_repairs']}|")
        lines += ['', f"## Root {r['root']}: first actual changes", '']
        for name, c in r['contrasts'].items():
            lines.append(f"- {name}: shared pre-action prefix={c['shared_pre_action_prefix']}; first changes={c['first_changes']}")
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
