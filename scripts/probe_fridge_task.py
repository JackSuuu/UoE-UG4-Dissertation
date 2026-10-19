"""Bounded scene/native-actor development probe, not a repair comparison.

Fresh robot reset; no scripted grasp/transport. Open-door diagnostics are separate
from closed-door full-sequence attempts. Stores physical states, images, predicates,
contacts and timing. Native success is diagnostic, not full stable completion.
"""
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

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
from adapters.libero_runtime import configure, patch_joint_addresses, patch_mass_matrix
from adapters.libero_preservation import bind_renderer_context, fingerprint, validate_controller


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def diagnostics(env, obs):
    m, d = env.sim.model, env.sim.data
    jid = m.joint_name2id('short_fridge_1_fridge_door_joint_0')
    contacts = []
    for c in d.contact[:d.ncon]:
        contacts.append([m.geom_id2name(c.geom1), m.geom_id2name(c.geom2), float(c.dist)])
    return dict(door_angle_rad=float(d.qpos[m.jnt_qposadr[jid]]),
        door_limited=bool(m.jnt_limited[jid]),
        item_position=obs['cream_cheese_1_pos'].tolist(),
        eef_position=obs['robot0_eef_pos'].tolist(),
        predicates={str(s): bool(env.env._eval_predicate(s)) for s in
            [*env.env.parsed_problem['goal_state'],
             ['in', 'cream_cheese_1', 'short_fridge_1_middle_region']]},
        native_success=bool(env.check_success()), contacts_endpoint=contacts)


def create_env(args, out):
    configure(args.root)
    patch_joint_addresses()
    patch_mass_matrix()
    bind_renderer_context()
    from libero.libero.envs import OffScreenRenderEnv
    text = (REPO / 'configs/tasks/fridge_retrieval_dev.bddl').read_text()
    if args.door == 'open':
        # Change only initialization, not the final close requirement.
        text = text.replace('(Close short_fridge_1))\n  (:goal', '(Open short_fridge_1))\n  (:goal')
    path = out / 'scene.bddl'
    path.write_text(text)
    env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256,
        camera_widths=256, horizon=args.steps + 40, render_gpu_device_id=args.gpu)
    env.seed(args.seed)
    # Bypass wrapper's unbounded RandomizationError retry; retain a failed reset.
    obs = env.env.reset()
    validate_controller(env)
    return env, obs


def main(args):
    import imageio.v2 as imageio
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = ['scripts/probe_fridge_task.py', 'configs/tasks/fridge_retrieval_dev.bddl',
        'src/adapters/libero_runtime.py', 'src/adapters/libero_openvla.py',
        'src/adapters/tf5_compat.py', 'src/adapters/libero_preservation.py',
        'exp_record/agent_notes/two_demo_experiment_design_20261019.md']
    status = dict(status='started', scope='scene_and_actor_development_only',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        start=time.time(), actor=args.snapshot is not None, hybrid=False,
        full_completion=None, simulated_time_pauses_during_inference=True,
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        sources={p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in sources})
    save(out/'status.json', status)
    env = None
    states = []
    try:
        env, obs = create_env(args, out)
        imageio.imwrite(out/'reset.png', obs['agentview_image'][::-1, ::-1])
        save(out/'reset.json', diagnostics(env, obs))
        (out/'model.xml').write_text(env.sim.model.get_xml())
        for _ in range(20):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        imageio.imwrite(out/'settled.png', obs['agentview_image'][::-1, ::-1])
        save(out/'settled.json', diagnostics(env, obs))
        actor = None
        if args.snapshot:
            import torch
            from adapters.libero_openvla import LiberoOpenVLA
            torch.manual_seed(args.seed)
            torch.set_num_threads(4)
            actor = LiberoOpenVLA(args.snapshot)
            status['actor_build'] = actor.build
            save(out/'status.json', status)
        status['native_ever'] = False
        with imageio.get_writer(out/'raw_baseline.mp4', fps=20) as video, (out/'steps.jsonl').open('w') as rows:
            for step in range(args.steps):
                video.append_data(obs['agentview_image'][::-1, ::-1])
                states.append(env.get_sim_state().copy())
                before = fingerprint(env, obs)
                t = time.monotonic()
                decode = {}
                if actor:
                    action, decode, image = actor.act(obs['agentview_image'], env.language_instruction)
                    if step == 0:
                        image.save(out/'policy_input.png')
                else:
                    action = np.array([0, 0, 0, 0, 0, 0, -1.])
                inference = time.monotonic() - t
                obs, _, done, _ = env.step(action.tolist())
                diag = diagnostics(env, obs)
                status['native_ever'] |= diag['native_success']
                row = dict(step=step, action=action.tolist(), inference_s=inference,
                    before=before, after=fingerprint(env, obs), done=bool(done), **diag, **decode)
                rows.write(json.dumps(row) + '\n')
                rows.flush()
                status['steps_executed'] = step + 1
                if step % 20 == 0:
                    print(json.dumps(dict(step=step, **diag)), flush=True)
                    save(out/'status.json', status)
                # Diagnostic native labels do not stop a full-sequence attempt.
                if done:
                    status['stopped_by_env_done'] = True
                    break
            states.append(env.get_sim_state().copy())
            video.append_data(obs['agentview_image'][::-1, ::-1])
        status.update(status='completed', final=diagnostics(env, obs))
    except BaseException:
        status.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        if states:
            np.savez_compressed(out/'states.npz', states=np.asarray(states))
        status['end'] = time.time()
        save(out/'status.json', status)
        if env is not None:
            env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--door', choices=['open', 'closed'], default='closed')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--snapshot')
    p.add_argument('--gpu', type=int, default=2)
    args = p.parse_args()
    assert args.steps > 0
    main(args)
