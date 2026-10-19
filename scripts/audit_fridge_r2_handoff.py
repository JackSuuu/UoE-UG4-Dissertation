"""Post-hoc R2 exit-event audit; preserves frozen controller and original labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main(run, out):
    status = json.loads((run/'status.json').read_text())
    frozen = run/'source'
    for path, sha in status['sources'].items():
        assert hashlib.sha256((frozen/path).read_bytes()).hexdigest() == sha, path
    sys.path.insert(0, str(frozen/'scripts'))
    from run_fridge_framework import (create, sense, TaskMonitor, FridgeRepair,
                                     ContactRecorder, interval_released, fingerprint, save)
    args = argparse.Namespace(**status['args'])
    args.gpu = 2
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    env, obs = create(args, out)
    recorder = None
    try:
        for _ in range(20):
            obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
        tracker, repair = TaskMonitor(args.door), FridgeRepair()
        state = sense(env, obs)
        task = tracker.observe(state)
        recorder = ContactRecorder(env.sim, keywords=('cream_cheese', 'short_fridge', 'gripper'))
        exits = []
        for row in rows:
            for key in ('integration', 'controller'):
                assert fingerprint(env, obs)[key] == row['before'][key]
            active = repair.active
            verified_exit = bool(active and repair.stable >= 2 and repair.verified(state, active))
            action, gate = repair.decide(state, task['phase'], np.asarray(row['proposal']))
            assert action.tolist() == row['action']
            assert all(row[k] == v for k, v in gate.items())
            if verified_exit:
                following = next((r for r in rows[row['step']:] if r['owner'] == 'vla'), None)
                exits.append(dict(step=row['step'], verified_phase=active,
                    recorded_event=row['event'], actual_owner=row['owner'],
                    same_step_unmodified_vla=row['owner'] == 'vla',
                    first_later_unmodified_vla_step=following['step'] if following else None,
                    remaining_sequence_completed_later=any(
                        r['task']['remaining_sequence_complete'] for r in rows[row['step']:])))
            recorder.reset()
            obs, *_ = env.step(action.tolist())
            state = sense(env, obs)
            task = tracker.observe(state, interval_released(recorder.samples))
            assert task == row['task']
            for key in ('integration', 'controller'):
                assert fingerprint(env, obs)[key] == row['after'][key]
        report = dict(status='frozen_decision_physics_task_audit_passed', source=str(run),
            scope='posthoc_event_accounting_not_new_rollout_or_actor_decode', steps=len(rows),
            logged_verified_handoffs=status['verified_handoffs'],
            verified_local_exits=exits, full_completion=status['full_completion'],
            explanation='retain_grip overwrites verified_handoff on the same decision; '
                        'a verified local exit is not an immediate unmodified-policy handoff')
        save(out/'audit.json', report)
        print(json.dumps(report, indent=2))
    finally:
        if recorder:
            recorder.close()
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.run.resolve(), a.out.resolve())
