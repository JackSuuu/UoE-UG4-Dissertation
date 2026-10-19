"""Actual frozen VLA / five-layer runtime comparison, R2 development version."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

from diagnose_fridge_skills import REPO, SOURCES as R1_SOURCES, create, measure, save, fingerprint
from adapters.fridge_runtime import FridgeRepair, TaskMonitor, PROMPTS, sense
from adapters.libero_completion import ContactRecorder

PROTOCOL = 'exp_record/agent_notes/fridge_r2_framework_protocol_20261019.md'
SOURCES = R1_SOURCES+[PROTOCOL, 'src/adapters/fridge_runtime.py',
    'scripts/run_fridge_framework.py', 'scripts/verify_fridge_framework.py']


def interval_released(samples):
    return not any('cream_cheese' in str(c) and 'gripper' in str(c)
                   for sample in samples for c in sample['contacts'])


def main(args):
    import imageio.v2 as imageio
    import torch
    from adapters.libero_openvla import LiberoOpenVLA
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for path in SOURCES:
        dest = out/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO/path).read_bytes())
    asset = Path(args.root)/'libero/libero/assets/articulated_objects/short_fridge.xml'
    contract = dict(protocol_sha256=hashlib.sha256((REPO/PROTOCOL).read_bytes()).hexdigest(),
        max_actions=args.steps, max_queries=args.steps, max_episode_wall_s=1200,
        control_hz=20, state_access='same privileged feedback available; shared phase prompts',
        termination='20-interval verified remaining-sequence completion or budget',
        actor_snapshot=args.snapshot, disturbance='none', runtime='synchronous')
    status = dict(status='started', scope='R2_five_layer_development',
        hybrid=args.mode == 'framework', args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        started=time.time(), comparison_contract=contract, full_completion=False,
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        sources={p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in SOURCES},
        steps_executed=0, queries=0)
    save(out/'status.json', status)
    env = recorder = None
    states, images, wrists, results = [], [], [], []
    try:
        env, obs = create(args, out)
        (out/'model.xml').write_text(env.sim.model.get_xml())
        for _ in range(20):
            obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
        imageio.imwrite(out/'settled.png', obs['agentview_image'][::-1,::-1])
        save(out/'initial.json', measure(env, obs))
        tracker = TaskMonitor(args.door)
        repair = FridgeRepair()
        state = sense(env, obs)
        task = tracker.observe(state)
        actor = LiberoOpenVLA(args.snapshot)
        status['actor_build'] = actor.build
        recorder = ContactRecorder(env.sim, keywords=('cream_cheese', 'short_fridge', 'gripper'))
        start = time.monotonic()
        save(out/'status.json', status)
        with imageio.get_writer(out/'raw.mp4', fps=20) as video, (out/'steps.jsonl').open('w') as rows:
            for step in range(args.steps):
                if time.monotonic()-start >= 1200:
                    status['termination'] = 'episode_wall_budget'
                    break
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                wrists.append(obs['robot0_eye_in_hand_image'].copy())
                video.append_data(obs['agentview_image'][::-1,::-1])
                before = fingerprint(env, obs)
                phase, language = task['phase'], PROMPTS[task['phase']]
                t = time.monotonic()
                proposal, decode, image = actor.act(obs['agentview_image'], language)
                inference = time.monotonic()-t
                if step == 0:
                    image.save(out/'policy_input.png')
                t = time.monotonic()
                candidate, gate = repair.decide(state, phase, proposal)
                if args.mode == 'actor':
                    gate = dict(gate, shadow_owner=gate['owner'], shadow_event=gate['event'],
                                owner='vla', event=None, applied_corrections=0)
                    action = proposal.copy()
                else:
                    action = candidate
                    gate['applied_corrections'] = repair.used
                decision_time = time.monotonic()-t
                recorder.reset()
                obs, _, done, _ = env.step(action.tolist())
                state = sense(env, obs)
                task = tracker.observe(state, interval_released(recorder.samples))
                diag = measure(env, obs)
                row = dict(step=step, phase=phase, next_phase=task['phase'], language=language,
                    action=action.tolist(), proposal=proposal.tolist(), inference_s=inference,
                    decision_s=decision_time, episode_elapsed_s=time.monotonic()-start,
                    before=before, after=fingerprint(env, obs), done=bool(done),
                    task=task.copy(), **gate, **diag, **decode)
                rows.write(json.dumps(row)+'\n')
                rows.flush()
                results.append(row)
                status.update(steps_executed=step+1, queries=step+1, full_completion=task['full_completion'])
                if step%20 == 0:
                    save(out/'status.json', status)
                    print(json.dumps({k:row[k] for k in ('step','phase','owner','event','grasped','task')}), flush=True)
                if done or task['remaining_sequence_complete']:
                    status['termination'] = 'verified_remaining_sequence' if task['remaining_sequence_complete'] else 'environment_done'
                    break
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
            wrists.append(obs['robot0_eye_in_hand_image'].copy())
            video.append_data(obs['agentview_image'][::-1,::-1])
        imageio.imwrite(out/'final.png', obs['agentview_image'][::-1,::-1])
        status.update(status='completed', final=measure(env, obs), final_task=task,
            any_grasp=any(r['grasped'] for r in results),
            applied_corrections=sum(r['owner']=='repair' for r in results),
            verified_handoffs=sum(r['event']=='verified_handoff' for r in results),
            unverified_exits=sum(r['event']=='repair_budget_exit_unverified' for r in results),
            nominal_actions=sum(r['owner']=='vla' for r in results),
            termination=status.get('termination','action_budget'), episode_wall_s=time.monotonic()-start)
    except BaseException:
        status.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        if states:
            np.savez_compressed(out/'states.npz', states=np.asarray(states))
        if images:
            np.savez_compressed(out/'images.npz', agent=np.asarray(images), wrist=np.asarray(wrists))
        status['ended'] = time.time()
        save(out/'status.json', status)
        if recorder:
            recorder.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--mode', choices=['actor', 'framework'], required=True)
    p.add_argument('--door', choices=['open', 'closed'], default='closed')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--prompt', default='shared_phase')
    p.add_argument('--yaw', type=float, default=-np.pi/2)
    p.add_argument('--fridge-y', type=float, default=.35)
    p.add_argument('--camera', default='robot_side')
    p.add_argument('--open-angle', type=float, default=np.deg2rad(130))
    p.add_argument('--item-forward', type=float, default=.04)
    p.add_argument('--item-lateral', type=float, default=.035)
    args = p.parse_args()
    assert args.steps > 0
    main(args)
