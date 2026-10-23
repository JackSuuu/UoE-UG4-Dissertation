"""Build predicate keyframes from successful, completed training episodes only."""
import argparse
import json
from pathlib import Path

from .dataset import observation
from ..memory.keyframe_bank import KeyframeBank
from ..planning.schema import ARITY, TaskSpec
from ..verification.physical_verifier import PhysicalVerifier


def build(directories):
    bank, verifier = None, PhysicalVerifier()
    for directory in directories:
        status = json.loads((directory/'status.json').read_text())
        root = status['args']['init_index']
        if status['status'] != 'completed' or type(root) is not int or root not in range(10, 20):
            raise ValueError('Only completed training roots 10–19 may enter memory')
        result = json.loads((directory/'result.json').read_text())
        if not result['controller']['complete']:
            continue
        events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        initial = observation(next(e['observation'] for e in events if e['type'] == 'observation'))
        generated = next(e['task'] for e in events if e['type'] == 'task_generated')
        spec = TaskSpec.parse({k: generated[k] for k in ('goals', 'milestones')}, generated['instruction'],
                              {e.id for e in initial.entities}, ARITY)
        monitors = {e['observation_index']: e for e in events if e['type'] == 'monitor'}
        history = [initial]
        for executed in (e for e in events if e['type'] == 'execution'):
            endpoint = observation(executed['observation'])
            history.append(endpoint)
            monitor = monitors.get(endpoint.index)
            if not monitor or monitor['visual'] is None:
                continue
            if bank is None:
                bank = KeyframeBank(encoder_signature=monitor['encoder_signature'])
            if bank.encoder_signature != monitor['encoder_signature']:
                raise ValueError('Cannot mix encoders in a memory bank')
            # One frame of apparent success is not enough to admit a keyframe.
            recent = history[-3:]
            predicates = [p for p in dict.fromkeys((*spec.goals, *spec.milestones))
                          if len(recent) == 3 and recent[0].index >= 1
                          and all(verifier.evaluate(p, o, history[:history.index(o)+1]) is True for o in recent)]
            if predicates:
                bank.write(monitor['visual'], predicates=predicates, instruction=spec.instruction,
                           root=root, split='training', measured_success=True)
    if bank is None or not bank._entries:
        raise ValueError('No train-supported measured successful visual keyframes; collect more episodes')
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.episodes).save(args.out)


if __name__ == '__main__':
    main()
