"""Independently replay an R0 action tape; never infer actor/task efficacy.

Reconstructs controller history with real steps. Numerical and image differences
are retained separately, without widening tolerances after observing failures.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_fridge_task import REPO, create_env, diagnostics, fingerprint, save


def main(run, out):
    status = json.loads((run/'status.json').read_text())
    assert status['status'] == 'completed'
    for path, expected in status['sources'].items():
        assert hashlib.sha256((REPO/path).read_bytes()).hexdigest() == expected, path
    args = argparse.Namespace(**status['args'])
    out.mkdir(parents=True, exist_ok=False)
    env = None
    records = [json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    saved = np.load(run/'states.npz')['states']
    report = dict(status='started', source_hashes_match=True, steps=len(records),
        source=str(run), fingerprint_mismatches=[], diagnostic_mismatches=[],
        max_abs_sim_state_difference=0.0)
    try:
        env, obs = create_env(args, out)
        assert (out/'scene.bddl').read_bytes() == (run/'scene.bddl').read_bytes()
        for _ in range(20):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        for i, row in enumerate(records):
            for key, value in fingerprint(env, obs).items():
                if value != row['before'][key]:
                    report['fingerprint_mismatches'].append(dict(step=i, when='before', field=key))
            gap = float(np.abs(env.get_sim_state()-saved[i]).max())
            report['max_abs_sim_state_difference'] = max(report['max_abs_sim_state_difference'], gap)
            obs, _, _, _ = env.step(row['action'])
            for key, value in fingerprint(env, obs).items():
                if value != row['after'][key]:
                    report['fingerprint_mismatches'].append(dict(step=i, when='after', field=key))
            for key, value in diagnostics(env, obs).items():
                if value != row[key]:
                    report['diagnostic_mismatches'].append(dict(step=i, field=key))
        gap = float(np.abs(env.get_sim_state()-saved[-1]).max())
        report['max_abs_sim_state_difference'] = max(report['max_abs_sim_state_difference'], gap)
        passed = not (report['fingerprint_mismatches'] or report['diagnostic_mismatches'] or
                      report['max_abs_sim_state_difference'])
        report['status'] = 'exact_replay_passed' if passed else 'replay_mismatch'
    finally:
        save(out/'verification.json', report)
        if env is not None:
            env.close()
    print(json.dumps(report, indent=2))
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    raise SystemExit(0 if main(a.run.resolve(), a.out.resolve()) else 1)
