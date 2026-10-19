"""Actor-only causal diagnostics: task, camera and pinned-inference controls.

No repair, action-route scripting, online simulator state input to actor or model
training. Observations other than actor RGB and language are offline diagnostics.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_fridge_skills import (REPO, SOURCES as R1_SOURCES, create as fridge_create,
    configure, patch_joint_addresses, patch_mass_matrix, bind_renderer_context,
    validate_controller, fingerprint, save)
from adapters.libero_completion import CompletionMonitor, ContactRecorder

TASKS = {
    'book': 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy',
    'cheese': 'LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket',
}
PROTOCOL = 'exp_record/agent_notes/vla_failure_audit_protocol_20261019.md'
SOURCES = R1_SOURCES+['scripts/diagnose_vla_failure.py', 'scripts/openvla_tf440_worker.py',
                     'scripts/verify_vla_failure.py', PROTOCOL]


def make_scene(args, out):
    if args.scene == 'fridge':
        env, obs = fridge_create(args, out)
    else:
        import torch
        base = configure(args.root)
        patch_joint_addresses()
        patch_mass_matrix()
        bind_renderer_context()
        from libero.libero.envs import OffScreenRenderEnv
        task = TASKS[args.scene]
        bddl = base/'bddl_files/libero_10'/(task+'.bddl')
        (out/'scene.bddl').write_bytes(bddl.read_bytes())
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
            camera_widths=256, horizon=args.steps+40, render_gpu_device_id=args.gpu)
        env.seed(0)
        env.reset()
        initial_file = base/'init_files/libero_10'/(task+'.pruned_init')
        states = torch.load(initial_file, weights_only=False)
        obs = env.set_init_state(states[args.init_index])
        save(out/'initial_source.json', dict(path=str(initial_file), index=args.init_index,
            sha256=hashlib.sha256(initial_file.read_bytes()).hexdigest()))
        if args.camera == 'robot_side':
            m = env.sim.model
            cid = m.camera_name2id('agentview')
            position, target = np.array([-.55, .85, 1.55]), np.array([0., .35, 1.05])
            forward = (target-position)/np.linalg.norm(target-position)
            right = np.cross(forward, [0., 0., 1.])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            quat = Rotation.from_matrix(np.column_stack([right, up, -forward])).as_quat()
            m.cam_pos[cid], m.cam_quat[cid] = position, quat[[3, 0, 1, 2]]
            env.sim.forward()
    validate_controller(env)
    for _ in range(20):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
    return env, obs


class PinnedActor:
    def __init__(self, args, out):
        from adapters.tf5_compat import snapshot_dir
        self.log = (out/'pinned_worker.log').open('w')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        self.process = subprocess.Popen([args.pinned_python, '-u',
            str(Path(__file__).with_name('openvla_tf440_worker.py')), args.snapshot, snapshot_dir()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            bufsize=1, env=env)
        self.build = self.receive()
        assert self.build['ready']

    def receive(self):
        for line in self.process.stdout:
            if line.startswith('RESULT '):
                return json.loads(line[7:])
            self.log.write(line)
            self.log.flush()
        raise RuntimeError(f'Pinned actor exited: {self.process.poll()}')

    def act(self, raw, language):
        from adapters.libero_openvla import policy_image
        image = policy_image(raw)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        self.process.stdin.write(json.dumps(dict(language=language,
            png=base64.b64encode(buffer.getvalue()).decode()))+'\n')
        self.process.stdin.flush()
        result = self.receive()
        action = np.asarray(result.pop('action'))
        return action, result, image

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()


def observe(env, obs, target):
    item = env.env.objects_dict[target]
    return dict(eef_position=obs['robot0_eef_pos'].tolist(),
        item_position=obs[target+'_pos'].tolist(),
        grasped=bool(env.env._check_grasp(env.env.robots[0].gripper, item.contact_geoms)),
        native_success=bool(env.check_success()))


def main(args):
    import torch
    import imageio.v2 as imageio
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='started', scope='actor_only_causal_diagnosis_no_framework',
        args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        sources={}, started=time.time())
    for path in SOURCES:
        data = (REPO/path).read_bytes()
        status['sources'][path] = hashlib.sha256(data).hexdigest()
        dest = out/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    save(out/'status.json', status)
    env = actor = recorder = None
    states, images, records = [], [], []
    try:
        env, obs = make_scene(args, out)
        target = 'black_book_1' if args.scene == 'book' else 'cream_cheese_1'
        language = 'pick up the cream cheese box inside the fridge' if args.scene == 'fridge' else env.language_instruction
        status.update(language=language, initial=fingerprint(env, obs), initial_diagnostics=observe(env, obs, target))
        imageio.imwrite(out/'initial.png', obs['agentview_image'][::-1, ::-1])
        (out/'model.xml').write_text(env.sim.model.get_xml())
        if args.smoke:
            status.update(status='scene_smoke_passed', no_actor_queries=True)
            return
        if args.decoder == 'pinned':
            actor = PinnedActor(args, out)
        else:
            from adapters.libero_openvla import LiberoOpenVLA
            actor = LiberoOpenVLA(args.snapshot)
        status['actor_build'] = actor.build
        recorder = ContactRecorder(env.sim, keywords=(target, 'desk_caddy', 'gripper'))
        completion = CompletionMonitor(env) if args.scene == 'book' else None
        save(out/'status.json', status)
        start = time.monotonic()
        with imageio.get_writer(out/'raw.mp4', fps=20) as video, (out/'steps.jsonl').open('w') as log:
            for step in range(args.steps):
                if time.monotonic()-start > 1200:
                    status['termination'] = 'wall_budget'
                    break
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                video.append_data(obs['agentview_image'][::-1, ::-1])
                before = fingerprint(env, obs)
                t = time.monotonic()
                action, decode, image = actor.act(obs['agentview_image'], language)
                inference = time.monotonic()-t
                if step == 0:
                    image.save(out/'policy_input.png')
                recorder.reset()
                obs, _, done, _ = env.step(action.tolist())
                row = dict(step=step, action=action.tolist(), proposal=action.tolist(),
                    owner='vla', inference_s=inference, before=before, after=fingerprint(env, obs),
                    **decode, **observe(env, obs, target))
                if completion:
                    row['completion'] = completion.measure(step, recorder.samples)
                records.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                status['steps_executed'] = step+1
                if step%40 == 0:
                    save(out/'status.json', status)
                    print(json.dumps({k:row[k] for k in ('step', 'grasped', 'native_success')}), flush=True)
                if done:
                    status['termination'] = 'environment_done'
                    break
            video.append_data(obs['agentview_image'][::-1, ::-1])
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
        status.update(status='completed', any_grasp=any(r['grasped'] for r in records),
            any_native_success=any(r['native_success'] for r in records),
            any_complete=any(r.get('completion', {}).get('complete', False) for r in records) if completion else None,
            final=observe(env, obs, target), termination=status.get('termination', 'action_budget'))
    except BaseException:
        status.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        if states:
            np.savez_compressed(out/'states.npz', states=np.asarray(states))
            np.savez_compressed(out/'images.npz', agent=np.asarray(images))
        status['ended'] = time.time()
        save(out/'status.json', status)
        if isinstance(actor, PinnedActor):
            actor.close()
        if recorder:
            recorder.close()
        if env:
            env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--scene', choices=['book', 'cheese', 'fridge'], required=True)
    p.add_argument('--camera', choices=['native', 'robot_side'], required=True)
    p.add_argument('--decoder', choices=['current', 'pinned'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--pinned-python', default='/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--init-index', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--door', default='open')
    p.add_argument('--yaw', type=float, default=-np.pi/2)
    p.add_argument('--fridge-y', type=float, default=.35)
    p.add_argument('--open-angle', type=float, default=np.deg2rad(130))
    p.add_argument('--item-forward', type=float, default=.04)
    p.add_argument('--item-lateral', type=float, default=.035)
    main(p.parse_args())
