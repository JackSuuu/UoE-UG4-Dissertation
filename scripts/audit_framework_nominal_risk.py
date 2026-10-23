"""Observational risk/no-lift evaluation on saved nominal tapes.

No actor, environment actions, fitting or threshold selection. Region-dependent
exit prediction is explicitly unavailable; it does not enter stage/risk/no-lift
features. This is a diagnostic on reused roots, not held-out model evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

import run_framework_comparison as comparison
sys.path.insert(0, str(comparison.REPO))
from src.training.train_joint_components import risk_label
from src.verification.joint_runtime import JointRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = comparison.load(args.plan)
    comparison.check_plan(plan)
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for source in args.source:
        status = comparison.load(source/'status.json')
        comparison.require(status['valid_episode'] and status['cell']['arm'] == 'baseline',
            'Only complete baseline tapes are eligible')
        controls = comparison.rows(source/'controller.jsonl')
        before = [r for r in controls if r['event'] == 'before']
        after = [dict(owner=r['decision']['owner'], state=r['verification']['evidence']['state'])
            for r in controls if r['event'] == 'after']
        comparison.require(all(r['decision']['owner'] == 'nominal' for r in before), 'Non-nominal action')
        runtime = JointRuntime(plan['fusion']['model'], plan['joint']['artifact'], device=args.device)
        records, history = [], []
        for row in before:
            t = row['step']
            state = row['decision']['evidence']['state']
            rgb = np.asarray(Image.open(source/'model_inputs'/f'{t:06d}.png').convert('RGB'))
            evidence = runtime.observe(rgb, state, t, region=None)
            history.append(state)
            label, reason = risk_label(state, history, after[t:t+24])
            risk = evidence['risk']
            records.append(dict(observation_index=t, selected_stage=evidence['selected_stage'],
                risk=risk, prediction=evidence['prediction'],
                risk_entry_eligible=risk['consecutive'] >= risk['required'],
                no_lift_entry_eligible=bool((evidence['prediction']['no_lift'] or {}).get('no_lift_predicted')),
                nominal24_label=label, nominal24_reason=reason, state=state,
                visual=evidence['visual'], memory=evidence['memory'],
                policy_rgb_sha256=evidence['policy_rgb_sha256']))
        root = status['cell']['root']
        with (args.out/f'baseline_root{root}.jsonl').open('x') as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        valid = [r for r in records if r['nominal24_label'] is not None]
        def confusion(key):
            return {name: sum(bool(r[key]) == pred and r['nominal24_label'] == actual for r in valid)
                for name, pred, actual in (('tp', True, 1), ('fp', True, 0), ('tn', False, 0), ('fn', False, 1))}
        result = dict(root=root, source=str(source.resolve()), actor_queries=0, device=args.device,
            complete_task_success=status['any_native_success'], observations=len(records),
            scope='risk/no-lift only; region-dependent exit is unavailable; reused development roots',
            labels=dict(Counter(r['nominal24_reason'] for r in records)),
            risk_entry_confusion=confusion('risk_entry_eligible'),
            no_lift_confusion=confusion('no_lift_entry_eligible'),
            first_risk_eligible=next((r['observation_index'] for r in records if r['risk_entry_eligible']), None),
            first_no_lift_eligible=next((r['observation_index'] for r in records if r['no_lift_entry_eligible']), None),
            risk_eligible_observations=sum(r['risk_entry_eligible'] for r in records),
            no_lift_eligible_observations=sum(r['no_lift_entry_eligible'] for r in records),
            source_controller_sha256=comparison.sha(source/'controller.jsonl'),
            source_actions_sha256=comparison.sha(source/'actions.npz'))
        reports.append(result)
        comparison.save(args.out/f'baseline_root{root}.summary.json', result)
        print(json.dumps(result), flush=True)
    comparison.save(args.out/'report.json', dict(episodes=reports,
        script_sha256=comparison.sha(Path(__file__)), plan_sha256=comparison.sha(args.plan),
        actor_queries=0, thresholds_reselected=False, trained=False,
        warning='Adjacent windows overlap. Confusion counts are per observation, not independent episode samples.'))


if __name__ == '__main__':
    raise SystemExit(main())
