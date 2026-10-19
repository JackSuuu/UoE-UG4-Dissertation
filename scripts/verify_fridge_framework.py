"""Fresh-process replay of physics AND runtime decisions from recorded VLA proposals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main(run, out):
    status = json.loads((run/'status.json').read_text())
    assert status['status'] == 'completed'
    frozen = run/'source'
    for path, sha in status['sources'].items():
        assert hashlib.sha256((frozen/path).read_bytes()).hexdigest() == sha, path
    sys.path.insert(0, str(frozen/'scripts'))
    from run_fridge_framework import (create, sense, FridgeRepair, TaskMonitor,
        ContactRecorder, interval_released, fingerprint, save, measure, PROMPTS)
    args = argparse.Namespace(**status['args'])
    asset = Path(args.root)/'libero/libero/assets/articulated_objects/short_fridge.xml'
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == status['asset_sha256']
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', source=str(run), mismatches=[], image_differences=[],
                  max_state_gap=0., sources_verified=True)
    env = recorder = None
    try:
        env, obs = create(args, out)
        assert (out/'scene.bddl').read_bytes() == (run/'scene.bddl').read_bytes()
        for _ in range(20):
            obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
        tracker, repair = TaskMonitor(args.door), FridgeRepair()
        state = sense(env, obs)
        task = tracker.observe(state)
        recorder = ContactRecorder(env.sim, keywords=('cream_cheese', 'short_fridge', 'gripper'))
        rows = [json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
        states = np.load(run/'states.npz')['states']
        with np.load(run/'images.npz') as archive:
            images = {k:archive[k] for k in ('agent', 'wrist')}
        assert len(states) == len(rows)+1
        for i, row in enumerate(rows):
            report['max_state_gap'] = max(report['max_state_gap'], float(np.abs(states[i]-env.get_sim_state()).max()))
            for k,v in fingerprint(env, obs).items():
                if v != row['before'][k]:
                    report['mismatches'].append([i, 'before', k])
            candidate, gate = repair.decide(state, task['phase'], np.asarray(row['proposal']))
            if args.mode == 'actor':
                gate = dict(gate, shadow_owner=gate['owner'], shadow_event=gate['event'],
                            owner='vla', event=None, applied_corrections=0)
                candidate = np.asarray(row['proposal'])
            else:
                gate['applied_corrections'] = repair.used
            expected = dict(gate, phase=task['phase'], language=PROMPTS[task['phase']], action=candidate.tolist())
            for k,v in expected.items():
                if v != row[k]:
                    report['mismatches'].append([i, 'decision', k])
            recorder.reset()
            obs, _, _, _ = env.step(row['action'])
            state = sense(env, obs)
            task = tracker.observe(state, interval_released(recorder.samples))
            if task != row['task']:
                report['mismatches'].append([i, 'task'])
            for k,v in measure(env, obs).items():
                if v != row[k]:
                    report['mismatches'].append([i, 'diagnostic', k])
            for k,v in fingerprint(env, obs).items():
                if v != row['after'][k]:
                    report['mismatches'].append([i, 'after', k])
            for key, obs_key in [('agent','agentview_image'), ('wrist','robot0_eye_in_hand_image')]:
                diff = np.abs(images[key][i+1].astype(int)-obs[obs_key].astype(int))
                if diff.any():
                    report['image_differences'].append(dict(step=i, camera=key,
                        changed_channels=int(np.count_nonzero(diff)), max_difference=int(diff.max())))
        report['max_state_gap'] = max(report['max_state_gap'], float(np.abs(states[-1]-env.get_sim_state()).max()))
        report['physics_decisions_exact'] = not report['max_state_gap'] and not any(m[-1]!='images' for m in report['mismatches'])
        report['steps'] = len(rows)
        report['status'] = 'exact_replay_passed' if not report['mismatches'] and not report['max_state_gap'] else 'replay_mismatch'
    finally:
        save(out/'verification.json', report)
        if recorder:
            recorder.close()
        if env is not None:
            env.close()
    print(json.dumps(report, indent=2))
    return report['status'] == 'exact_replay_passed'


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    a=p.parse_args()
    raise SystemExit(0 if main(a.run.resolve(), a.out.resolve()) else 1)
