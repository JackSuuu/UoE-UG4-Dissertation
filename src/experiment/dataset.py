"""Collect auditable generated-contract labels from completed training episodes.

No relabelling of old microwave traces, unexecuted candidates, failed backend
runs or held-out roots. Inputs are the pre-execution feature vectors saved by
the live runtime; only labels consult actual observed future endpoints.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..observation import Entity, Observation
from ..planning.schema import ARITY, RecoveryPlan, TaskSpec
from ..training.handoff_dataset import LABEL_SCHEMA, continuation_label
from ..training.recovery_dataset import recovery_label, regression_label
from ..training.train_fusion_mlp import validate_rows
from ..verification.features import FEATURE_SCHEMA


def observation(data):
    return Observation(data['index'], b'', tuple(Entity(**e) for e in data['entities']), data['facts'],
                       data['positions'], data['contacts'], data['eef_position'], data['native_success'])


def collect(directory, role):
    status = json.loads((directory/'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError(f'Not a completed runtime episode: {directory}')
    root = status['args']['init_index']
    if type(root) is not int or root not in range(10, 20):
        raise ValueError('Dataset collection is restricted to training roots 10–19')
    result = json.loads((directory/'result.json').read_text())
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    executions = [e for e in events if e['type'] == 'execution']
    if (len(executions) != result['controller']['budget']['actions']
            or sum(e['type'] == 'policy_query' for e in events) != result['controller']['budget']['queries']):
        raise ValueError('Recorded execution/query counts disagree with the completed episode')
    events_sha = hashlib.sha256((directory/'events.jsonl').read_bytes()).hexdigest()
    initial = observation(next(e['observation'] for e in events if e['type'] == 'observation'))
    ids = {e.id for e in initial.entities}
    generated = next(e['task'] for e in events if e['type'] == 'task_generated')
    spec = TaskSpec.parse({k: generated[k] for k in ('goals', 'milestones')}, generated['instruction'], ids, ARITY)
    decisions = {e['observation']['index']: e for e in events if e['type'] == 'decision'}
    monitors = {e['observation_index']: e for e in events if e['type'] == 'monitor'}
    future = [dict(owner=e['decision']['owner'], observation=observation(e['observation']),
                   complete=e['result']['complete']) for e in executions]
    repairs = {r['id']: r for r in result['controller']['repairs']}
    rows, censored = [], Counter()
    for i, executed in enumerate(executions):
        step = executed['decision']['step']
        decision = decisions[step]
        current = observation(decision['observation'])
        history = [initial, *(row['observation'] for row in future[:i])]
        history[-1] = current  # Include clauses enriched at this unchanged endpoint.
        monitor = monitors[step]
        features = monitor['features'] if role == 'risk' else executed.get('model_features')
        if role == 'risk':
            if executed['decision']['owner'] != 'nominal':
                continue
            label, reason = regression_label(spec, current, future[i:], history=history)
        elif role == 'handoff':
            if not executed['decision']['handoff']:
                continue
            plan = RecoveryPlan.parse(decision['repair'], ids, ARITY)
            label, reason = continuation_label(plan, current, future[i:], history=history)
        else:
            repair_id = executed['decision']['repair_id']
            if executed['decision']['owner'] != 'repair' or step != repairs[repair_id]['entry']:
                continue
            label, reason = recovery_label(repairs[repair_id])
        if label is None or features is None:
            censored['no_visual_features' if features is None else reason] += 1
            continue
        rows.append(dict(sample_id=f'{directory.resolve()}:{role}:{step}', root=root, split='training', role=role,
            schema=FEATURE_SCHEMA, label_schema=LABEL_SCHEMA, features=features, label=label, reason=reason,
            encoder_signature=monitor['encoder_signature'], source=str(directory.resolve()),
            source_events_sha256=events_sha))
    if rows:
        validate_rows(rows, role)
    return rows, censored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, nargs='+', required=True)
    parser.add_argument('--role', choices=('risk', 'recovery', 'handoff'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    rows, censored = [], Counter()
    for directory in args.episodes:
        new, counts = collect(directory, args.role)
        rows.extend(new)
        censored.update(counts)
    if not rows:
        parser.error(f'No usable observed labels; censored={dict(censored)}')
    validate_rows(rows, args.role)
    with args.out.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(dict(samples=len(rows), censored=dict(censored))))


if __name__ == '__main__':
    main()
