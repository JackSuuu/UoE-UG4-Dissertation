"""Camera-framing intervention only: no robot action repair or learned adaptation."""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from diagnose_vla_failure import PinnedActor, SOURCES as BASE_SOURCES, make_scene as base_scene
from diagnose_fridge_skills import REPO, measure, fingerprint, save
from run_fridge_r1 import SNAPSHOT

PROTOCOL = 'exp_record/agent_notes/fridge_observation_protocol_20261019.md'
SOURCES = list(dict.fromkeys(BASE_SOURCES+[
    'scripts/fridge_observation_experiment.py', 'scripts/run_fridge_observation_cohort.py',
    'scripts/run_fridge_r1.py',
    'src/adapters/libero_drawer.py', PROTOCOL]))


def points(env):
    """Static camera calibration geometry, never a policy/action input."""
    m, d = env.sim.model, env.sim.data
    result = []
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    names = []
    for g in range(m.ngeom):
        name = m.geom_id2name(g) or ''
        if not name.startswith(('short_fridge_1_', 'cream_cheese_1_', 'gripper0_')):
            continue
        rotation = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == 6:  # box
            local = signs*m.geom_size[g]
        elif m.geom_type[g] == 7:  # compiled mesh vertices
            mesh = int(m.geom_dataid[g])
            start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            local = m.mesh_vert[start:start+count]
        else:
            local = signs*float(m.geom_rbound[g])
        result.extend(local@rotation.T+d.geom_xpos[g])
        names.append(name)
    assert result and names
    return np.asarray(result), names


def camera_config(env):
    m, d = env.sim.model, env.sim.data
    cid = m.camera_name2id('agentview')
    vertices, names = points(env)
    rotation = d.cam_xmat[cid].reshape(3, 3).copy()
    center = (vertices.min(axis=0)+vertices.max(axis=0))/2
    projected = (vertices-center)@rotation
    tangent = np.tan(np.deg2rad(float(m.cam_fovy[cid]))/2)
    # sqrt(.9) is the fixed policy crop width; retain a further 10% margin.
    usable = float(np.sqrt(.9)*.9)
    distance = float(np.max(projected[:, 2]+np.max(np.abs(projected[:, :2]), axis=1)/(tangent*usable))+.02)
    position = center+rotation[:, 2]*distance
    assert int(m.cam_bodyid[cid]) == 0, 'This calibration assumes a world-fixed camera'
    return dict(rule='fit_seed0_fixture_item_hand_geometry_in_existing_orientation_then_freeze',
        calibration_seed=0, original_position=m.cam_pos[cid].tolist(),
        position=position.tolist(), quat=m.cam_quat[cid].tolist(), fovy=float(m.cam_fovy[cid]),
        center=center.tolist(), distance=distance, usable_sensor_half_fraction=usable,
        included_geoms=names, geometry_points=len(vertices),
        scope='privileged_reset_only_camera_calibration_not_occlusion_or_skill_guarantee')


def make_scene(args, out):
    shared = copy.copy(args)
    shared.camera = 'robot_side'
    env, obs = base_scene(shared, out)
    if args.camera == 'framed_side':
        config = json.loads(Path(args.camera_config).read_text())
        m = env.sim.model
        cid = m.camera_name2id('agentview')
        m.cam_pos[cid] = config['position']
        m.cam_quat[cid] = config['quat']
        assert float(m.cam_fovy[cid]) == config['fovy']
    # Both arms perform the same refresh, with no action or integration step.
    env.sim.forward()
    obs = env.env._get_observations(force_update=True)
    return env, obs


def main(args):
    import torch
    import imageio.v2 as imageio
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='started', scope='actor_only_camera_framing_diagnosis',
        args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        sources={}, started=time.time(), intervention_actions=0)
    for path in SOURCES:
        data = (REPO/path).read_bytes()
        target = out/'source'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        status['sources'][path] = hashlib.sha256(data).hexdigest()
    if args.camera_config:
        data = Path(args.camera_config).read_bytes()
        (out/'camera_config.json').write_bytes(data)
        status['camera_config_sha256'] = hashlib.sha256(data).hexdigest()
    save(out/'status.json', status)
    env = actor = None
    states, images, wrists, rows = [], [], [], []
    try:
        env, obs = make_scene(args, out)
        status.update(initial=fingerprint(env, obs), initial_diagnostics=measure(env, obs))
        (out/'model.xml').write_text(env.sim.model.get_xml())
        imageio.imwrite(out/'initial.png', obs['agentview_image'][::-1, ::-1])
        if args.calibrate:
            assert args.seed == 0 and args.camera == 'robot_side'
            save(out/'camera_config.json', camera_config(env))
        if args.smoke or args.calibrate:
            np.save(out/'initial_state.npy', env.get_sim_state())
            status.update(status='calibration_passed' if args.calibrate else 'scene_smoke_passed')
            return
        actor = PinnedActor(args, out)
        language = 'pick up the cream cheese box inside the fridge'
        status.update(actor_build=actor.build, language=language)
        start = time.monotonic()
        with imageio.get_writer(out/'raw.mp4', fps=20) as video, (out/'steps.jsonl').open('w') as log:
            for step in range(args.steps):
                if time.monotonic()-start >= 1200:
                    status['termination'] = 'wall_budget'
                    break
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                wrists.append(obs['robot0_eye_in_hand_image'].copy())
                video.append_data(obs['agentview_image'][::-1, ::-1])
                before = fingerprint(env, obs)
                evidence = measure(env, obs)
                t = time.monotonic()
                action, decode, processed = actor.act(obs['agentview_image'], language)
                inference = time.monotonic()-t
                if step == 0:
                    processed.save(out/'policy_input.png')
                trace = dict(schema='observation_actor_trace_v1', step=step,
                    schema_origin='libero_drawer.decision_trace evidence/decision/outcome separation',
                    evidence_through_step=step-1, evidence=evidence,
                    evidence_use='logging_only_not_action_selection',
                    subgoal='grasp cream cheese in open fridge',
                    proposal_source='frozen_openvla_official_predict_action',
                    nominal_action=action.tolist(), executed_action=action.tolist(),
                    gate=dict(mode='actor_only', reason='execute_unmodified_model_proposal',
                              repair_active=False, actual_handoff=False),
                    events=[], verification='not_applied_actor_control',
                    interpretation='observable execution evidence, not model internal reasoning or causal effect')
                obs, _, done, _ = env.step(action.tolist())
                diag = measure(env, obs)
                row = dict(step=step, owner='vla', phase='grasp', language=language,
                    action=action.tolist(), proposal=action.tolist(), inference_s=inference,
                    episode_elapsed_s=time.monotonic()-start, before=before,
                    after=fingerprint(env, obs), trace=trace, done=bool(done), **decode, **diag)
                rows.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                status['steps_executed'] = step+1
                if step%40 == 0:
                    save(out/'status.json', status)
                if done:
                    status['termination'] = 'environment_done'
                    break
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
            wrists.append(obs['robot0_eye_in_hand_image'].copy())
            video.append_data(obs['agentview_image'][::-1, ::-1])
        status.update(status='completed', any_grasp=any(r['grasped'] for r in rows),
            any_native_success=any(r['native_success'] for r in rows),
            full_completion=None, final=measure(env, obs),
            termination=status.get('termination', 'action_budget'))
    except BaseException:
        status.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        if states:
            np.savez_compressed(out/'states.npz', states=np.asarray(states))
            np.savez_compressed(out/'images.npz', agent=np.asarray(images), wrist=np.asarray(wrists))
        status['ended'] = time.time()
        save(out/'status.json', status)
        if actor:
            actor.close()
        if env:
            env.close()


def replay(run, out, gpu):
    status = json.loads((run/'status.json').read_text())
    assert status['status'] == 'completed'
    for name, sha in status['sources'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest() == sha
    sys.path.insert(0, str(run/'source/scripts'))
    # Run this entry point from its frozen source; subprocess launcher enforces it.
    assert Path(__file__).resolve() == (run/'source/scripts/fridge_observation_experiment.py').resolve()
    args = argparse.Namespace(**status['args'])
    args.gpu = gpu
    if args.camera_config:
        args.camera_config = run/'camera_config.json'
    out.mkdir(parents=True, exist_ok=False)
    env, obs = make_scene(args, out)
    rows = [json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()]
    states = np.load(run/'states.npz')['states']
    images = np.load(run/'images.npz')
    report = dict(status='started', source=str(run), mismatches=[], max_state_gap=0.)
    try:
        for i in range(len(rows)+1):
            gap = float(np.abs(env.get_sim_state()-states[i]).max())
            report['max_state_gap'] = max(report['max_state_gap'], gap)
            for key, field in [('agent','agentview_image'), ('wrist','robot0_eye_in_hand_image')]:
                if not np.array_equal(images[key][i], obs[field]):
                    report['mismatches'].append([i,key])
            if i == len(rows):
                break
            row = rows[i]
            if fingerprint(env,obs) != row['before']:
                report['mismatches'].append([i,'before'])
            if measure(env,obs) != row['trace']['evidence']:
                report['mismatches'].append([i,'pre_action_evidence'])
            assert row['action'] == row['proposal'] == row['trace']['executed_action']
            obs, _, done, _ = env.step(row['action'])
            if fingerprint(env,obs) != row['after']:
                report['mismatches'].append([i,'after'])
            for key, value in measure(env,obs).items():
                if value != row[key]:
                    report['mismatches'].append([i,key])
            assert bool(done) == row['done']
        report.update(status='exact_replay_passed' if not report['mismatches'] and not report['max_state_gap'] else 'replay_mismatch', steps=len(rows))
    finally:
        save(out/'verification.json', report)
        images.close()
        env.close()
    assert report['status'] == 'exact_replay_passed', report


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--scene', default='fridge')
    p.add_argument('--camera', choices=['robot_side', 'framed_side'], default='robot_side')
    p.add_argument('--camera-config', type=Path)
    p.add_argument('--decoder', default='pinned')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--snapshot', default=SNAPSHOT)
    p.add_argument('--pinned-python', default='/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--init-index', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--replay', type=Path)
    p.add_argument('--door', default='open')
    p.add_argument('--yaw', type=float, default=-np.pi/2)
    p.add_argument('--fridge-y', type=float, default=.35)
    p.add_argument('--open-angle', type=float, default=np.deg2rad(130))
    p.add_argument('--item-forward', type=float, default=.04)
    p.add_argument('--item-lateral', type=float, default=.035)
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if a.replay:
        replay(a.replay.resolve(), a.out.resolve(), a.gpu)
    else:
        main(a)
