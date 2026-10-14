"""One frozen native-H1 actor, four matched release-to-close development arms."""
import argparse
import hashlib
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
from adapters.libero_completion import ContactRecorder
from adapters.libero_drawer import (TASK, ARMS, MAX_STEPS, DrawerMonitor,
    SubgoalTracker, TransitionGate, decision_trace, settled_drawer)
from adapters.libero_openvla import LiberoOpenVLA
from adapters.libero_preservation import fingerprint
from run_libero_baseline import write_json


def reset_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def episode(actor, args, index, arm):
    out = args.out / f'root_{index}' / arm
    out.mkdir(parents=True, exist_ok=False)
    reset_seeds()
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), repo/'scripts/run_libero_baseline.py'] + [repo/'src/adapters'/f for f in
        ['libero_drawer.py', 'libero_preservation.py', 'libero_completion.py',
         'libero_runtime.py', 'libero_openvla.py', 'tf5_compat.py']]
    task_base = Path(args.root)/'libero/libero'
    manifest = dict(status='started', start=time.time(), index=index, arm=arm, task=TASK,
        actor=actor.build, H=1, chunk_k=1, privileged=True, learned_predictor=False,
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        source_sha256={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        protocol_sha256=hashlib.sha256((repo/'exp_record/libero_drawer_target_protocol_20261014.md').read_bytes()).hexdigest(),
        task_sha256={str(p.relative_to(task_base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [task_base/'bddl_files/libero_10'/(TASK+'.bddl'), task_base/'init_files/libero_10'/(TASK+'.pruned_init')]},
        GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), max_steps=MAX_STEPS, CVR=None, safe_success=None)
    write_json(out/'manifest.json', manifest)
    env = recorder = None
    try:
        env, obs = settled_drawer(args.root, index)
        monitor, tracker, gate = DrawerMonitor(env), SubgoalTracker(), TransitionGate(arm)
        recorder = ContactRecorder(env.sim, keywords=('akita_black_bowl', 'white_cabinet'))
        (out/'model.xml').write_text(env.sim.model.get_xml())
        previous = monitor.measure([])
        tracking = tracker.update(previous, -1)
        states, rows = [], []
        corrections = 0
        with imageio.get_writer(out/'rollout.mp4', fps=20, codec='libx264',
                               ffmpeg_params=['-movflags','+faststart']) as video, \
                (out/'steps.jsonl').open('w') as log, (out/'substeps.jsonl').open('w') as raw, \
                (out/'decision_trace.jsonl').open('w') as trace_log:
            for step in range(MAX_STEPS):
                states.append(env.get_sim_state().copy())
                before = fingerprint(env, obs)
                video.append_data(obs['agentview_image'][::-1, ::-1])
                proposal, decode, image = actor.act(obs['agentview_image'], env.language_instruction)
                if step == 0:
                    image.save(out/'policy_input.png')
                action, decision = gate.decide(proposal, previous, tracking, step)
                trace = decision_trace(step, arm, previous, tracking, proposal, action, decision)
                assert fingerprint(env, obs) == before, 'Read-only monitor/actor changed simulator'
                recorder.reset()
                obs, _, _, _ = env.step(action.tolist())
                assert len(recorder.samples) == 25
                after = fingerprint(env, obs)
                measured = monitor.measure(recorder.samples)
                assert fingerprint(env, obs) == after, 'Monitor mutated simulator'
                post_tracking = tracker.update(measured, step)
                changed = not np.array_equal(action, proposal)
                corrections += int(changed)
                row = dict(step=step, before=before, proposal=proposal.tolist(), action=action.tolist(),
                    **decode, decision=decision, tracking_before=tracking, measured=measured,
                    tracking=post_tracking, correction=changed)
                log.write(json.dumps(row)+'\n')
                raw.write(json.dumps(dict(step=step, substeps=recorder.samples))+'\n')
                trace['observed_after_execution'] = dict(
                    retained=measured['retained'], bowl_gripper=measured['bowl_gripper'],
                    cabinet_gripper=measured['cabinet_gripper'], clear=measured['clear'],
                    drawer_q=measured['drawer_q'], native=measured['native_success'],
                    full_completion=post_tracking['complete'], phase=post_tracking['phase'])
                trace_log.write(json.dumps(trace)+'\n')
                log.flush()
                raw.flush()
                trace_log.flush()
                rows.append(row)
                previous, tracking = measured, post_tracking
                if step % 20 == 0 or decision['mode'] != 'nominal' or tracking['complete']:
                    print(json.dumps(dict(root=index, arm=arm, step=step, phase=tracking['phase'],
                        mode=decision['mode'], reason=decision['reason'], native=measured['native_success'],
                        dwell=tracking['final_streak'])), flush=True)
                # Shared whole-task endpoint; never stop at placement alone.
                if tracking['complete']:
                    break
            states.append(env.get_sim_state().copy())
            video.append_data(obs['agentview_image'][::-1, ::-1])
        np.savez_compressed(out/'states.npz', states=np.asarray(states))
        write_json(out/'result.json', dict(index=index, arm=arm, steps=len(rows),
            complete=tracking['complete'], native_ever=any(r['measured']['native_success'] for r in rows),
            milestone_steps=tracker.milestones, trigger=gate.trigger, intervention_steps=gate.used,
            changed_actions=corrections, handoff=gate.handoff, VLA_queries=len(rows),
            forecast_queries=0, final=measured, CVR=None, safe_success=None,
            post_handoff_bowl_contact_intervals=sum(r['measured']['bowl_gripper'] for r in rows
                if gate.handoff is not None and r['step'] >= gate.handoff['step'])))
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
    p.add_argument('--indices', type=int, nargs='+', choices=range(3), required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    reset_seeds()
    actor = LiberoOpenVLA(args.snapshot)
    for index in args.indices:
        for arm in args.arms:
            episode(actor, args, index, arm)
