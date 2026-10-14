"""Frozen-actor, H1/k1 placement-preservation mechanism study (privileged)."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import imageio.v2 as imageio
import numpy as np
import torch
from adapters.libero_completion import CompletionMonitor, ContactRecorder
from adapters.libero_openvla import LiberoOpenVLA
from adapters.libero_preservation import (ARMS, NAMES, SETTLE, MAX_STEPS, TAIL_STEPS,
    PreservationWindow, candidates, fingerprint, forecast, settled_env)
from run_libero_baseline import write_json


def episode(actor, args, index, arm):
    out = args.out / f'root_{index}' / arm
    out.mkdir(parents=True, exist_ok=False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), repo/'scripts/run_libero_baseline.py'] + [repo/'src/adapters'/f for f in
        ['libero_preservation.py', 'libero_completion.py', 'libero_runtime.py', 'libero_openvla.py', 'tf5_compat.py']]
    manifest = dict(status='started', index=index, arm=arm, start=time.time(), actor=actor.build,
        git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(f.relative_to(repo)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        protocol_sha256=hashlib.sha256((repo/'exp_record/libero_preservation_protocol_20261014.md').read_bytes()).hexdigest(),
        versions={k: importlib.metadata.version(k) for k in ['torch','transformers','numpy','mujoco','robosuite']},
        privileged=True, learned_predictor=False, H=1, chunk_k=1, max_steps=MAX_STEPS,
        forecast='K6; 5-step pulse then settle; execute first only', post_completion_actor_steps=TAIL_STEPS,
        GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), CVR=None, safe_success=None)
    write_json(out/'manifest.json', manifest)
    env = recorder = None
    try:
        env, obs = settled_env(args.root, index)
        monitor, window = CompletionMonitor(env), PreservationWindow()
        recorder = ContactRecorder(env.sim)
        manifest.update(bounds=monitor.bounds.tolist(), floor=monitor.floor)
        write_json(out/'manifest.json', manifest)
        (out/'model.xml').write_text(env.sim.model.get_xml())
        states, actions, rows = [], [], []
        previous = None
        corrections = simulator_queries = replay_steps = 0
        with imageio.get_writer(out/'rollout.mp4', fps=20, codec='libx264',
                               ffmpeg_params=['-movflags','+faststart']) as video, \
                (out/'steps.jsonl').open('w') as log, (out/'substeps.jsonl').open('w') as physics, \
                (out/'decisions.jsonl').open('w') as decisions:
            for step in range(MAX_STEPS):
                states.append(env.get_sim_state().copy())
                video.append_data(obs['agentview_image'][::-1, ::-1])
                proposal, decode, image = actor.act(obs['agentview_image'], env.language_instruction)
                if step == 0:
                    image.save(out/'policy_input.png')
                active = window.active(step, previous, monitor.first_complete)
                action, choice = proposal.copy(), 0
                if active:
                    plans = candidates(proposal, actions[-1] if actions else SETTLE)
                    prediction = None
                    if arm == 'hold':
                        choice = 3
                    elif arm == 'retreat':
                        choice = 4
                    elif arm == 'physics':
                        before = fingerprint(env, obs)
                        choice, prediction = forecast(args.root, index, actions, plans, before, previous)
                        assert fingerprint(env, obs) == before, 'Forecast mutated live environment'
                        simulator_queries += prediction['forecast_control_steps']
                        replay_steps += prediction['replay_control_steps']
                    action = plans[choice, 0].copy()
                    decisions.write(json.dumps(dict(step=step, selected=choice,
                        first_actions=plans[:, 0].tolist(), prediction=prediction))+'\n')
                    decisions.flush()
                changed = not np.array_equal(action, proposal)
                corrections += int(changed)
                recorder.reset()
                obs, _, native_done, _ = env.step(action.tolist())
                assert len(recorder.samples) == 25
                measured = monitor.measure(step, recorder.samples)
                if active and arm == 'physics':
                    assert measured == prediction['candidates'][choice]['rows'][0], 'Forecast/actual first interval differs'
                row = dict(step=step, action=action.tolist(), proposal=proposal.tolist(), **decode,
                    active=active, correction=changed, selected=NAMES[choice], native_done=bool(native_done), **measured)
                log.write(json.dumps(row)+'\n')
                physics.write(json.dumps(dict(step=step, substeps=recorder.samples))+'\n')
                log.flush()
                physics.flush()
                actions.append(action.copy())
                rows.append(row)
                previous = measured
                if step % 20 == 0 or active or measured['complete']:
                    print(json.dumps(dict(root=index, arm=arm, step=step, selected=NAMES[choice],
                        streak=monitor.streak, first_complete=monitor.first_complete)), flush=True)
                # Acquisition uses unchanged frozen acceptance. Then turn correction
                # OFF and stress-test 40 additional real actor actions on new images.
                if monitor.first_complete is not None and step >= monitor.first_complete + TAIL_STEPS:
                    break
            states.append(env.get_sim_state().copy())
            video.append_data(obs['agentview_image'][::-1, ::-1])
        np.savez_compressed(out/'states.npz', states=np.asarray(states))
        first = monitor.first_complete
        tail = rows[first+1:first+1+TAIL_STEPS] if first is not None else []
        enough = len(tail) == TAIL_STEPS
        write_json(out/'result.json', dict(index=index, arm=arm, policy_steps=len(rows),
            native_ever=monitor.first_native is not None, first_native_step=monitor.first_native,
            released_placement=first is not None, first_complete_step=first, max_streak=monitor.max_streak,
            tail_steps=len(tail), tail_final_dwell=all(r['candidate'] for r in tail[-20:]) if enough else None,
            tail_continuous=all(r['candidate'] for r in tail) if enough else None,
            trigger=window.trigger, opportunities=window.used, corrections=corrections,
            VLA_queries=len(rows), simulator_forecast_steps=simulator_queries,
            simulator_replay_steps=replay_steps, final=measured, CVR=None, safe_success=None))
        manifest['status'] = 'complete'
    except BaseException:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()
        manifest['end'] = time.time()
        write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--snapshot', required=True)
    p.add_argument('--indices', type=int, nargs='+', choices=range(6), required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    actor = LiberoOpenVLA(a.snapshot)
    for index in a.indices:
        for arm in a.arms:
            episode(actor, a, index, arm)
