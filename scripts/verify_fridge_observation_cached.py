"""Independent replay retry: materialize compressed image arrays once, not per step.

The original frozen verifier repeatedly indexes NPZ members inside its frame loop,
decompressing whole image tapes and exceeding its300s timeout. Keep those timeouts
and source snapshots unchanged; this version only fixes verification I/O cost.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def main(run, out, gpu):
    start = time.monotonic()
    status = json.loads((run/'status.json').read_text())
    assert status['status']=='completed'
    for name, sha in status['sources'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==sha
    script = run/'source/scripts/fridge_observation_experiment.py'
    sys.path.insert(0,str(script.parent))
    spec = importlib.util.spec_from_file_location('frozen_fridge_observation',script)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    args = argparse.Namespace(**status['args'])
    args.gpu = gpu
    if args.camera_config:
        args.camera_config = run/'camera_config.json'
    out.mkdir(parents=True,exist_ok=False)
    rows = [json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()]
    with np.load(run/'states.npz') as data:
        states = data['states']
    with np.load(run/'images.npz') as data:
        images = {key:data[key] for key in ('agent','wrist')}
    assert all(len(v)==len(rows)+1 for v in images.values()) and len(states)==len(rows)+1
    env,obs = frozen.make_scene(args,out)
    report = dict(status='started',source=str(run),mismatches=[],max_state_gap=0.,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verification_change='cache decoded NPZ image arrays once; frozen scene/measure/actions unchanged')
    try:
        for i in range(len(rows)+1):
            report['max_state_gap'] = max(report['max_state_gap'],float(np.abs(env.get_sim_state()-states[i]).max()))
            for key,field in [('agent','agentview_image'),('wrist','robot0_eye_in_hand_image')]:
                if not np.array_equal(images[key][i],obs[field]):
                    report['mismatches'].append([i,key])
            if i==len(rows):
                break
            row = rows[i]
            if frozen.fingerprint(env,obs)!=row['before']:
                report['mismatches'].append([i,'before'])
            if frozen.measure(env,obs)!=row['trace']['evidence']:
                report['mismatches'].append([i,'pre_action_evidence'])
            assert row['action']==row['proposal']==row['trace']['executed_action']
            obs,_,done,_ = env.step(row['action'])
            if frozen.fingerprint(env,obs)!=row['after']:
                report['mismatches'].append([i,'after'])
            for key,value in frozen.measure(env,obs).items():
                if value!=row[key]:
                    report['mismatches'].append([i,key])
            assert bool(done)==row['done']
        report.update(status='exact_replay_passed' if not report['mismatches'] and not report['max_state_gap'] else 'replay_mismatch',
                      steps=len(rows),elapsed_s=time.monotonic()-start)
    finally:
        frozen.save(out/'verification.json',report)
        env.close()
    print(json.dumps(report,indent=2))
    return report['status']=='exact_replay_passed'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--gpu',type=int,default=1)
    a=p.parse_args()
    raise SystemExit(0 if main(a.run.resolve(),a.out.resolve(),a.gpu) else 1)
