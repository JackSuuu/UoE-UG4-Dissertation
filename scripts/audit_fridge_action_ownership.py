"""Descriptive R2 action provenance; no counterfactual efficacy or autonomy claim."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def main(root, out):
    report = dict(scope='descriptive_provenance_not_causal_verifier_ablation',
        near_zero_definition='L2 of three normalized translation commands <0.01; not measured immobility',
        runs={}, pairs={})
    tapes = {}
    for door in ('closed', 'open'):
        for arm in ('actor', 'framework'):
            name = f'seed0_{door}_{arm}'
            run = root/name
            rows = [json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
            tapes[name] = rows
            assert all(r['action'] == r['proposal'] for r in rows if r['owner'] == 'vla')
            assert arm != 'actor' or all(r['owner'] == 'vla' for r in rows)
            proposal = np.asarray([r['proposal'] for r in rows])
            first_grasp = next((r for r in rows if r['grasped']), None)
            first_extract = next((r for r in rows if r['task']['extracted_seen']), None)
            info = dict(steps=len(rows), owners=dict(Counter(r['owner'] for r in rows)),
                translation_near_zero_steps=int(np.count_nonzero(np.linalg.norm(proposal[:, :3], axis=1) < .01)),
                unique_token_sequences=len(set(tuple(r['tokens']) for r in rows)),
                geometry_primitive_steps=sum(r['owner'] == 'repair' and r['repair_phase'] is not None for r in rows),
                gripper_only_retention_steps=sum(r['event'] == 'retain_grip' for r in rows),
                phases_during_primitive=dict(Counter(r['repair_phase'] for r in rows
                                                    if r['owner'] == 'repair' and r['repair_phase'])),
                first_grasp={k:first_grasp[k] for k in ('step', 'owner', 'repair_phase')} if first_grasp else None,
                first_extraction={k:first_extract[k] for k in ('step', 'owner', 'repair_phase')} if first_extract else None,
                final_task=rows[-1]['task'])
            report['runs'][name] = info
        left, right = tapes[f'seed0_{door}_actor'], tapes[f'seed0_{door}_framework']
        first = next(r['step'] for r in right if r['owner'] == 'repair')
        same_initial = left[0]['before'] == right[0]['before']
        prefix_equal = all(all(left[i][k] == right[i][k] for k in ('before', 'proposal', 'action', 'after'))
                           for i in range(first))
        assert same_initial and prefix_equal
        report['pairs'][door] = dict(first_intervention=first, initial_exact=same_initial,
                                    pre_intervention_actor_physics_image_prefix_exact=prefix_equal)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.root.resolve(), a.out.resolve())
