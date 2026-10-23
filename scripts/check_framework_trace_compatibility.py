"""Zero-actor logical replay of retained decisions with the current defaults.

This checks controller compatibility, not simulator or neural-model replay.
The independent original simulator/neural audits remain separate evidence.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import run_framework_comparison as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for episode in comparison.load(args.report)['episodes']:
        source = Path(episode['source'])
        status = comparison.load(source/'status.json')
        rows = comparison.rows(source/'controller.jsonl')
        before = {r['step']: r for r in rows if r['event'] == 'before'}
        after = {r['step']: r for r in rows if r['event'] == 'after'}
        controller, _ = comparison.controllers(status['cell']['arm'], status['language'],
            status['cell']['variant'], status['fusion_mode'], status['joint_mode'], status['recovery_options'],
            status.get('joint_admission', 'all'), status.get('repair_control', 'existing'))
        for raw in comparison.rows(source/'steps.jsonl'):
            t = raw['step']
            expected = before[t]['decision']
            decision = controller.before(t, before[t]['diagnostics'],
                joint_evidence=deepcopy(expected['evidence'].get('joint')))
            comparison.require(decision == expected, f'{source.name}: decision drift at {t}')
            stored = after[t]['verification']
            fusion = stored['verification'].get('fusion', {}).get('evidence')
            actual = controller.after(t, raw['diagnostics'], raw['action'], decision, fusion)
            comparison.require(actual == {k: v for k, v in stored.items() if k != 'joint_after'},
                f'{source.name}: after drift at {t}')
        comparison.require(controller.summary == status['controller_summary'], f'{source.name}: summary drift')
        records.append(dict(name=episode['name'], source=str(source), actions=controller.summary['actions'],
            status='exact_controller_trace_match', controller_sha256=comparison.sha(source/'source/src/controller.py')))
    result = dict(status='passed', actor_queries=0, simulator_replayed=False, models_recomputed=False,
        script_sha256=comparison.sha(Path(__file__)), new_controller_sha256=comparison.sha(comparison.REPO/'src/legacy/microwave_controller.py'),
        report_sha256=comparison.sha(args.report), episodes=records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(episodes=len(records), actions=sum(r['actions'] for r in records), status='passed')))


if __name__ == '__main__':
    main()
