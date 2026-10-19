"""R1 versioned geometry/skill diagnostics; never a framework efficacy score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from probe_fridge_task import (REPO, configure, patch_joint_addresses, patch_mass_matrix,
                              bind_renderer_context, validate_controller, fingerprint, save)
import numpy as np

SOURCES = ['scripts/diagnose_fridge_skills.py', 'scripts/probe_fridge_task.py',
           'configs/tasks/fridge_retrieval_dev.bddl', 'src/adapters/libero_runtime.py',
           'src/adapters/libero_openvla.py', 'src/adapters/tf5_compat.py',
           'src/adapters/libero_preservation.py', 'src/adapters/libero_completion.py',
           'exp_record/agent_notes/closed_loop_recovery_r1_20261019.md']


def patch_asset(open_angle, item_forward, item_lateral):
    """Change only this process's constructed asset, before MuJoCo compilation."""
    from libero.libero.envs.objects.articulated_objects import ShortFridge
    if getattr(ShortFridge, '_r1', False):
        return
    original = ShortFridge.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        joints = [j for j in self.get_obj().iter('joint')
                  if j.get('name', '').endswith('fridge_door_joint_0')]
        assert len(joints) == 1
        joints[0].set('limited', 'true')
        self.object_properties['articulation']['default_close_ranges'] = [0., 0.]
        self.object_properties['articulation']['default_open_ranges'] = [open_angle, open_angle]
        sites = [s for s in self.get_obj().iter('site') if s.get('name','').endswith('middle_region')]
        assert len(sites) == 1
        pos = np.fromstring(sites[0].get('pos'), sep=' ')
        pos[0] += item_forward
        pos[2] += item_lateral
        sites[0].set('pos', ' '.join(map(str, pos)))
    ShortFridge.__init__ = init
    ShortFridge.is_close = lambda self, q: bool(abs(q) <= np.deg2rad(5))
    ShortFridge._r1 = True


def create(args, out):
    configure(args.root)
    patch_joint_addresses()
    patch_mass_matrix()
    bind_renderer_context()
    patch_asset(args.open_angle, args.item_forward, args.item_lateral)
    from libero.libero.envs import OffScreenRenderEnv
    text = (REPO/'configs/tasks/fridge_retrieval_dev.bddl').read_text()
    text = text.replace('1.5707963267948966 1.5707963267948966', f'{args.yaw} {args.yaw}')
    text = text.replace('-0.01 0.14 0.01 0.16', f'-0.01 {args.fridge_y-.01} 0.01 {args.fridge_y+.01}')
    if args.door == 'open':
        text = text.replace('(Close short_fridge_1))\n  (:goal', '(Open short_fridge_1))\n  (:goal')
    (out/'scene.bddl').write_text(text)
    env = OffScreenRenderEnv(bddl_file_name=str(out/'scene.bddl'), camera_heights=256,
                            camera_widths=256, horizon=args.steps+40,
                            render_gpu_device_id=args.gpu)
    env.seed(args.seed)
    obs = env.env.reset()
    if args.camera == 'robot_side':
        from scipy.spatial.transform import Rotation
        model = env.sim.model
        cid = model.camera_name2id('agentview')
        position = np.array([-.55, .85, 1.55])
        target = np.array([0., args.fridge_y, 1.05])
        forward = target-position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.,0.,1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        quat = Rotation.from_matrix(np.column_stack([right,up,-forward])).as_quat()
        model.cam_pos[cid] = position
        model.cam_quat[cid] = quat[[3,0,1,2]]
        env.sim.forward()
    validate_controller(env)
    return env, obs


def inventory(env, obs):
    m, d = env.sim.model, env.sim.data
    robot = env.env.robots[0]
    bodies = {}
    for i in range(m.nbody):
        name = m.body_id2name(i)
        if name and ('short_fridge' in name or name in ('robot0_base', 'gripper0_eef', 'gripper0_hand')):
            bodies[name] = dict(position=d.body_xpos[i].tolist(), rotation=d.body_xmat[i].reshape(3,3).tolist())
    geoms = {}
    for i in range(m.ngeom):
        name = m.geom_id2name(i)
        if name and any(s in name for s in ('short_fridge', 'cream_cheese', 'gripper0')):
            rot = d.geom_xmat[i].reshape(3,3)
            geoms[name] = dict(id=i, type=int(m.geom_type[i]), position=d.geom_xpos[i].tolist(),
                size=m.geom_size[i].tolist(), rotation=rot.tolist(),
                box_world_half_extent=(np.abs(rot)@m.geom_size[i]).tolist(),
                contact_enabled=bool(m.geom_contype[i] or m.geom_conaffinity[i]),
                body=m.body_id2name(m.geom_bodyid[i]))
    sites = {m.site_id2name(i): d.site_xpos[i].tolist() for i in range(m.nsite)
             if any(s in (m.site_id2name(i) or '') for s in ('fridge', 'grip', 'eef'))}
    j = m.joint_name2id('short_fridge_1_fridge_door_joint_0')
    return dict(bodies=bodies, geoms=geoms, sites=sites,
                hinge=dict(anchor=d.xanchor[j].tolist(), axis=d.xaxis[j].tolist(),
                           limited=bool(m.jnt_limited[j]), range=m.jnt_range[j].tolist()),
                eef=obs['robot0_eef_pos'].tolist(), eef_quat=obs['robot0_eef_quat'].tolist(),
                gripper_qpos=obs['robot0_gripper_qpos'].tolist(),
                important_geoms=robot.gripper.important_geoms,
                controller=dict(output_min=robot.controller.output_min.tolist(),
                                output_max=robot.controller.output_max.tolist()))


def measure(env, obs):
    m, d = env.sim.model, env.sim.data
    inner = env.env
    j = m.joint_name2id('short_fridge_1_fridge_door_joint_0')
    item = inner.objects_dict['cream_cheese_1']
    contacts = [[m.geom_id2name(c.geom1), m.geom_id2name(c.geom2), float(c.dist)]
                for c in d.contact[:d.ncon]]
    return dict(door_angle_rad=float(d.qpos[m.jnt_qposadr[j]]),
                door_speed=float(d.qvel[m.jnt_dofadr[j]]),
                door_limited=bool(m.jnt_limited[j]),
                item_position=obs['cream_cheese_1_pos'].tolist(),
                eef_position=obs['robot0_eef_pos'].tolist(),
                eef_quat=obs['robot0_eef_quat'].tolist(),
                gripper_qpos=obs['robot0_gripper_qpos'].tolist(),
                grasped=bool(inner._check_grasp(inner.robots[0].gripper, item.contact_geoms)),
                native_success=bool(env.check_success()), contacts_endpoint=contacts)


def servo_action(obs, target, grip=-1., orientation=None):
    action = np.zeros(7)
    action[:3] = np.clip((np.asarray(target)-obs['robot0_eef_pos'])*12, -.5, .5)
    action[6] = grip
    if orientation is not None:
        from scipy.spatial.transform import Rotation
        current = Rotation.from_quat(obs['robot0_eef_quat'])
        error = (Rotation.from_matrix(np.asarray(orientation))*current.inv()).as_rotvec()
        action[3:6] = np.clip(error*2, -.5, .5)
    return action


def main(args):
    import imageio.v2 as imageio
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for path in SOURCES:
        dest = out/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO/path).read_bytes())
    if args.waypoints:
        (out/'waypoints.json').write_bytes(Path(args.waypoints).read_bytes())
    asset = Path(args.root)/'libero/libero/assets/articulated_objects/short_fridge.xml'
    status = dict(status='started', scope='R1_development_not_framework',
        args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        started=time.time(), full_completion=None, hybrid=False,
        physics_version='r1_limited_hinge_zero_closed_explicit_open_and_item_region_diagnostic_5deg',
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        sources={p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in SOURCES})
    if args.waypoints:
        status['waypoints_sha256'] = hashlib.sha256((out/'waypoints.json').read_bytes()).hexdigest()
    save(out/'status.json', status)
    states, images, wrists, env = [], [], [], None
    try:
        env, obs = create(args, out)
        (out/'model.xml').write_text(env.sim.model.get_xml())
        for _ in range(20):
            obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
        imageio.imwrite(out/'settled.png', obs['agentview_image'][::-1,::-1])
        imageio.imwrite(out/'wrist.png', obs['robot0_eye_in_hand_image'][::-1,::-1])
        save(out/'inventory.json', inventory(env, obs))
        initial = measure(env, obs)
        save(out/'initial.json', initial)
        actor = None
        if args.mode == 'actor':
            import torch
            from adapters.libero_openvla import LiberoOpenVLA
            torch.manual_seed(args.seed)
            torch.set_num_threads(4)
            actor = LiberoOpenVLA(args.snapshot)
            status['actor_build'] = actor.build
            save(out/'status.json', status)
        language = env.language_instruction if args.prompt == 'canonical' else (
            'open the fridge' if args.door == 'closed' else 'pick up the cream cheese box')
        waypoints = json.loads(Path(args.waypoints).read_text()) if args.waypoints else []
        assert args.mode != 'servo' or sum(w['steps'] for w in waypoints) == args.steps
        phase_end, phase_index = (waypoints[0]['steps'] if waypoints else 0), 0
        measurements = []
        with imageio.get_writer(out/'raw.mp4', fps=20) as video, (out/'steps.jsonl').open('w') as rows:
            for step in range(args.steps):
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                wrists.append(obs['robot0_eye_in_hand_image'].copy())
                video.append_data(obs['agentview_image'][::-1,::-1])
                before = fingerprint(env, obs)
                decode, phase = {}, args.mode
                t = time.monotonic()
                if actor:
                    action, decode, image = actor.act(obs['agentview_image'], language)
                    if step == 0:
                        image.save(out/'policy_input.png')
                elif args.mode == 'servo':
                    if step >= phase_end:
                        phase_index += 1
                        phase_end += waypoints[phase_index]['steps']
                    w = waypoints[phase_index]
                    phase = w['name']
                    action = servo_action(obs, w['target'], w.get('grip', -1), w.get('orientation'))
                elif args.mode == 'pulse':
                    action = np.zeros(7)
                    action[-1] = -1
                    if step < 18:
                        action[step//6] = .25 if step%6 < 3 else -.25
                else:
                    action = np.array([0,0,0,0,0,0,-1.])
                inference = time.monotonic()-t
                obs, _, done, _ = env.step(action.tolist())
                diag = measure(env, obs)
                measurements.append(diag)
                row = dict(step=step, phase=phase, language=language, action=action.tolist(),
                           inference_s=inference, before=before, after=fingerprint(env, obs),
                           done=bool(done), **diag, **decode)
                rows.write(json.dumps(row)+'\n')
                rows.flush()
                status['steps_executed'] = step+1
                if step%20 == 0:
                    save(out/'status.json', status)
                    print(json.dumps(dict(step=step, phase=phase, **diag)), flush=True)
                if done:
                    status['env_done'] = True
                    break
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
            wrists.append(obs['robot0_eye_in_hand_image'].copy())
            video.append_data(obs['agentview_image'][::-1,::-1])
        imageio.imwrite(out/'final.png', obs['agentview_image'][::-1,::-1])
        status.update(status='completed', final=measure(env, obs), language=language,
            any_grasp=any(x['grasped'] for x in measurements),
            max_item_displacement_m=max([float(np.linalg.norm(np.array(x['item_position'])-initial['item_position']))
                                         for x in measurements], default=0),
            max_door_displacement_rad=max([abs(x['door_angle_rad']-initial['door_angle_rad'])
                                           for x in measurements], default=0))
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
        if env is not None:
            env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--mode', choices=['inventory','pulse','servo','actor'], default='inventory')
    p.add_argument('--door', choices=['open','closed'], default='closed')
    p.add_argument('--yaw', type=float, default=np.pi/2)
    p.add_argument('--fridge-y', type=float, default=.15)
    p.add_argument('--camera', choices=['original','robot_side'], default='original')
    p.add_argument('--open-angle', type=float, default=np.pi/2)
    p.add_argument('--item-forward', type=float, default=0.)
    p.add_argument('--item-lateral', type=float, default=0.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--prompt', choices=['canonical','skill'], default='canonical')
    p.add_argument('--snapshot')
    p.add_argument('--waypoints')
    args = p.parse_args()
    assert args.steps > 0
    assert args.mode != 'actor' or args.snapshot
    main(args)
