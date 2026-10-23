"""Offline contract/progress diagnostics; never claims counterfactual outcomes.

Thresholds inherit the existing controller's contact_window and motion_m. The
input contains observed trajectories only. An annotated early stop does not
predict the subsequent nominal continuation; that requires a separate live run.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def measured_stage(s):
    if s['in_goal'] is True:
        return 'close' if s['grasp'] is False and s['mug_gripper'] is False else 'contain_release'
    return 'transfer' if s['held_clear'] is True else 'acquire'


def stagnation(window, checks, width=12, motion_m=.003):
    """Stationary AND no accepted progress; missing evidence is not stagnation."""
    if len(window) < width or any(any(c.values()) for c in checks[-width:]):
        return dict(stagnant=False, reason='short_window_or_acceptance_progress')
    states = window[-width:]
    keys = ('grasp', 'held_clear', 'in_goal', 'close_goal')
    if any(s.get('position') is None or any(s.get(k) is None for k in keys) for s in states):
        return dict(stagnant=False, reason='missing_measurements')
    changed = any(any(s[k] != states[0][k] for s in states) for k in keys)
    diameter = max(math.dist(a['position'], b['position']) for a in states for b in states)
    return dict(stagnant=not changed and diameter <= motion_m, reason='measured_window',
        diameter_m=diameter, semantic_change=changed, width=width, motion_m=motion_m)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.analysis/'report.json').read_text())
    episodes = []
    for ep in report['episodes']:
        trace = [json.loads(l) for l in (args.analysis/f"{ep['name']}.steps.jsonl").open()]
        records = []
        for repair in ep['repairs']:
            segment = trace[repair['entry_step']:repair['exit_step']+1]
            window, checks, decisions = [], [], []
            for row in segment:
                window.append(row['post'])
                checks.append(row['acceptance'])
                decisions.append(dict(action_index=row['action_index'], requested_stage=row['stage'],
                    measured_stage=measured_stage(row['pre']), stagnation=stagnation(window, checks)))
            records.append(dict(repair_id=repair['repair_id'], trigger=repair['trigger'],
                recovery_goal=repair['recovery_goal'], entry_step=repair['entry_step'],
                actual_exit_step=repair['exit_step'], verified=repair['verified'],
                first_stage_disagreement=next((d['action_index'] for d in decisions if
                    d['requested_stage'] != d['measured_stage']), None),
                first_stagnant_endpoint=next((d['action_index'] for d in decisions if d['stagnation']['stagnant']), None),
                requested_stages=dict(Counter(d['requested_stage'] for d in decisions)), decisions=decisions))
        with (args.out/f"{ep['name']}.json").open('x') as stream:
            json.dump(records, stream, indent=2)
        episodes.append(dict(name=ep['name'], repairs=[{k: v for k, v in r.items() if k != 'decisions'} for r in records]))
    result = dict(scope='observational_only_no_replayed_alternatives',
        thresholds=dict(width=12, motion_m=.003, source='existing controller contact_window and motion_m'), episodes=episodes)
    with (args.out/'report.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({e['name']: [(r['repair_id'],r['first_stage_disagreement'],r['first_stagnant_endpoint'])
        for r in e['repairs']] for e in episodes if e['repairs']}, indent=2))


if __name__ == '__main__':
    main()
