"""Independent frozen-source replay for actor-only failure diagnostics."""
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
    for name, sha in status['sources'].items():
        assert hashlib.sha256((frozen/name).read_bytes()).hexdigest() == sha, name
    sys.path.insert(0, str(frozen/'scripts'))
    from diagnose_vla_failure import (make_scene, observe, fingerprint, save,
                                      CompletionMonitor, ContactRecorder)
    args = argparse.Namespace(**status['args'])
    out.mkdir(parents=True, exist_ok=False)
    env, obs = make_scene(args, out)
    report = dict(status='started', source=str(run), mismatches=[], max_state_gap=0.)
    recorder = ContactRecorder(env.sim, keywords=('black_book_1' if args.scene == 'book' else
                                                'cream_cheese_1', 'desk_caddy', 'gripper'))
    target = 'black_book_1' if args.scene == 'book' else 'cream_cheese_1'
    completion = CompletionMonitor(env) if args.scene == 'book' else None
    try:
        rows = [json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
        states = np.load(run/'states.npz')['states']
        images = np.load(run/'images.npz')['agent']
        assert len(states) == len(images) == len(rows)+1
        for i, row in enumerate(rows):
            report['max_state_gap'] = max(report['max_state_gap'], float(np.abs(env.get_sim_state()-states[i]).max()))
            if not np.array_equal(images[i], obs['agentview_image']):
                report['mismatches'].append([i, 'agent_image'])
            for k, v in fingerprint(env, obs).items():
                if v != row['before'][k]:
                    report['mismatches'].append([i, 'before', k])
            assert row['action'] == row['proposal'] and row['owner'] == 'vla'
            recorder.reset()
            obs, *_ = env.step(row['action'])
            expected = observe(env, obs, target)
            if completion:
                expected['completion'] = completion.measure(i, recorder.samples)
            for k, v in expected.items():
                if v != row[k]:
                    report['mismatches'].append([i, 'diagnostic', k])
            for k, v in fingerprint(env, obs).items():
                if v != row['after'][k]:
                    report['mismatches'].append([i, 'after', k])
        report['max_state_gap'] = max(report['max_state_gap'], float(np.abs(env.get_sim_state()-states[-1]).max()))
        if not np.array_equal(images[-1], obs['agentview_image']):
            report['mismatches'].append([len(rows), 'agent_image'])
        report.update(status='exact_replay_passed' if not report['mismatches'] and not report['max_state_gap'] else 'replay_mismatch', steps=len(rows))
    finally:
        save(out/'verification.json', report)
        recorder.close()
        env.close()
    print(json.dumps(report, indent=2))
    return report['status'] == 'exact_replay_passed'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    raise SystemExit(0 if main(a.run.resolve(), a.out.resolve()) else 1)
