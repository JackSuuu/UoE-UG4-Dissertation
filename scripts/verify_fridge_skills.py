"""Replay R1 using the run's frozen sources, without loading or querying the actor."""
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
    # Early geometry probes omitted this transitive, unchanged dependency. Explicitly
    # record that fallback; new source freezes include it.
    fallback = {}
    dependency = 'src/adapters/libero_completion.py'
    if not (frozen/dependency).exists():
        repo = Path(__file__).resolve().parents[1]
        sys.path.append(str(repo/'src'))
        fallback[dependency] = hashlib.sha256((repo/dependency).read_bytes()).hexdigest()
    from diagnose_fridge_skills import create, measure, fingerprint, save
    args = argparse.Namespace(**status['args'])
    asset = Path(args.root)/'libero/libero/assets/articulated_objects/short_fridge.xml'
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == status['asset_sha256']
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', source=str(run), source_hashes_match=True,
                  supplemental_dependency_hashes=fallback,
                  fingerprint_mismatches=[], diagnostic_mismatches=[], max_state_difference=0.,
                  image_pixel_differences=[])
    env = None
    try:
        env, obs = create(args, out)
        for _ in range(20):
            obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
        rows = [json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
        states = np.load(run/'states.npz')['states']
        images = None
        if (run/'images.npz').exists():
            with np.load(run/'images.npz') as archive:
                images = {key: archive[key] for key in ('agent','wrist')}
        assert len(states) == len(rows)+1
        assert (out/'scene.bddl').read_bytes() == (run/'scene.bddl').read_bytes()
        for i, row in enumerate(rows):
            for k,v in fingerprint(env, obs).items():
                if v != row['before'][k]:
                    report['fingerprint_mismatches'].append([i,'before',k])
            gap = float(np.max(np.abs(states[i]-env.get_sim_state())))
            report['max_state_difference'] = max(gap, report['max_state_difference'])
            obs, _, _, _ = env.step(row['action'])
            for k,v in fingerprint(env, obs).items():
                if v != row['after'][k]:
                    report['fingerprint_mismatches'].append([i,'after',k])
            for k,v in measure(env, obs).items():
                if v != row[k]:
                    report['diagnostic_mismatches'].append([i,k])
            if images is not None:
                for key, obs_key in [('agent','agentview_image'),('wrist','robot0_eye_in_hand_image')]:
                    diff = np.abs(images[key][i+1].astype(int)-obs[obs_key].astype(int))
                    if diff.any():
                        report['image_pixel_differences'].append(dict(step=i,camera=key,
                            changed_channels=int(np.count_nonzero(diff)), max_channel_difference=int(diff.max())))
        report['max_state_difference'] = max(report['max_state_difference'],
            float(np.max(np.abs(states[-1]-env.get_sim_state()))))
        report['steps'] = len(rows)
        report['physics_controller_exact'] = not (report['max_state_difference'] or
            report['diagnostic_mismatches'] or
            any(x[2] != 'images' for x in report['fingerprint_mismatches']))
        passed = not (report['max_state_difference'] or report['fingerprint_mismatches']
                      or report['diagnostic_mismatches'])
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
