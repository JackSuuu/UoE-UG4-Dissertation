"""Matched fridge candidate pools; complete closed task vs open grasp diagnosis."""
import argparse
import json
from pathlib import Path
import run_framework_comparison as c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = c.load(a.run/'results.json')
    report = dict(schema='fridge_candidate_probe_analysis_v1', cases=[], comparisons=[], pool_checks=[],
        held_out=False, learned_selection_tested=False, source_sha256=c.sha(Path(__file__)),
        protocol_sha256=c.sha(a.run/'protocol.json'), invalid_cases=[])
    tapes, statuses, records = {}, {}, []
    for job in jobs:
        source = Path(job['source'])
        name = job['cell']['name']
        status = c.load(source/'status.json') if (source/'status.json').exists() else {}
        rp = a.run/'replays'/name/'status.json'
        replay = c.load(rp) if rp.exists() else {}
        valid = bool(job['valid'] and replay.get('status') == 'exact_replay_passed' and replay.get('actual_actor_queries') == 0)
        rows = c.rows(source/'steps.jsonl')
        if valid:
            q = c.rows(source/'queries.jsonl')
            requested = [r for r in q if r['event'] == 'requested']
            returned = [r for r in q if r['event'] == 'returned']
            c.require(len(rows) == status['steps_executed'] <= 600, 'Action budget/accounting')
            c.require(len(requested) == len(returned) == status['actor_queries'] <= 603, 'Query budget/accounting')
            c.require(status['actor_queries'] == len(rows)+3*status['candidate_pools'], 'Auxiliary query accounting')
            c.require(all(r['action'] == r['proposal'] == r['decode']['action'] and not r['action_modified'] for r in rows), 'Action edit')
        else:
            report['invalid_cases'].append(name)
        case = dict(name=name, door=job['cell']['door'], slot=job['cell']['slot'], valid=valid,
            native=status.get('any_native_success') if valid else None,
            grasp=status.get('any_grasp') if valid else None,
            strict_full=status.get('full_completion') if valid and job['cell']['door'] == 'closed' else None,
            actions=status.get('steps_executed'), queries=status.get('actor_queries'), repair=status.get('repair'))
        report['cases'].append(case)
        tapes[name], statuses[name] = rows, status
        probes = [r for r in rows if 'candidate_selection' in r]
        if probes:
            r = probes[0]
            repair = status.get('repair') or {}
            records.append(dict(schema='fridge_executed_candidate_outcome_v1', source=str(source),
                split='development', split_group='fridge_seed0', condition=job['cell']['door'], training_eligible=valid,
                selected_id=r['candidate_selection']['selected_id'], candidates=r['candidate_selection']['candidates'],
                context=r['candidate_selection']['context'], repair=repair,
                labels=dict(complete_native_task_success=case['native'] if case['door']=='closed' else None,
                    diagnostic_grasp=case['grasp'], verified_exit=repair.get('verified') if valid else None,
                    actual_handoff=(repair.get('handoff_step') is not None) if valid else None,
                    stable_nominal_continuation=repair.get('stable_nominal_continuation') if valid else None),
                unexecuted_labels=None))
    def signature(row):
        return dict(before=row['before'], context=row['candidate_selection']['context'], candidates=[
            {k:v['decode'][k] for k in ('action','tokens','source','seed','sampling','policy_png_sha256','prompt')}
            for v in row['candidate_selection']['candidates']])
    for door in ('closed','open'):
        cases = [v for v in report['cases'] if v['door'] == door]
        base = next(v for v in cases if v['slot'] is None)
        greedy = next(v for v in cases if v['slot'] == 0)
        for case in cases:
            if case['slot'] is None or not case['valid'] or not base['valid']:
                continue
            c.require(statuses[case['name']]['initial'] == statuses[base['name']]['initial'], 'Initial scene mismatch')
            for control in (base, greedy):
                if not control['valid'] or control == case:
                    continue
                key = 'native' if door == 'closed' else 'grasp'
                report['comparisons'].append(dict(door=door, endpoint=key, control=control['name'], treatment=case['name'],
                    rescue=not control[key] and case[key], harm=control[key] and not case[key]))
            if not greedy['valid'] or case['slot'] == 0:
                continue
            left, right = tapes[greedy['name']], tapes[case['name']]
            probes = [r for r in left if 'candidate_selection' in r]
            if probes:
                t = probes[0]['step']
                c.require(len(right)>t and 'candidate_selection' in right[t], 'Probe schedule mismatch')
                c.require(signature(left[t]) == signature(right[t]), 'Same-state candidate pool mismatch')
                c.require(all(x['before']==y['before'] and x['action']==y['action'] for x,y in zip(left[:t], right[:t])), 'Prefix mismatch')
                report['pool_checks'].append(dict(door=door,slot=case['slot'],step=t,
                    unique_actions=len({tuple(x['decode']['action']) for x in probes[0]['candidate_selection']['candidates']})))
            else:
                c.require(len(left)==len(right) and all(x['action']==y['action'] and x['before']==y['before'] for x,y in zip(left,right)), 'Inactive path mismatch')
    report['status'] = 'passed' if not report['invalid_cases'] else 'contains_invalid_cases'
    with (a.out/'recovery_dataset.jsonl').open('x') as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    c.save(a.out/'report.json',report)
    print(json.dumps(report,indent=2))
    return int(bool(report['invalid_cases']))


if __name__ == '__main__':
    raise SystemExit(main())
