"""Fridge-only candidate integration, with shared prompts and read-only evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np

from experiment3_candidate_run import CandidateActor
from experiment3_candidate_gate import ARMS, CAP, WALL_SECONDS, Budget
from experiment3_fridge_gate import FridgeCandidateGate, trace, executed_events
from fridge_observation_experiment import make_scene
from diagnose_fridge_skills import fingerprint, save
from adapters.fridge_runtime import PROMPTS, sense
from adapters.fridge_stage_evidence import FridgeStageEvidence, predicates
from adapters.libero_completion import ContactRecorder, box_corners


def read_state(env, obs):
    s = {k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in sense(env, obs).items()}
    m, d = env.sim.model, env.sim.data
    g = m.geom_name2id('cream_cheese_1_g1')
    world = box_corners(d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g])
    footprint_margin = float(min((world[:, :2]-[-.15, -.25]).min(), ([.10, -.05]-world[:, :2]).min()))
    seated_margin = .005-abs(float(world[:, 2].min())-s['floor'])
    s.update(release_footprint_margin_m=footprint_margin, release_seated_margin_m=seated_margin,
        local_release_support=bool(s['table'] and not s['shelf'] and footprint_margin >= 0 and
            seated_margin >= 0 and s['max_penetration'] <= .001))
    return s


def initial_stages(tracker, s):
    return dict(schema='fridge_stage_evidence_v1_initial', step=-1,
        phase='open' if tracker.initial_door == 'closed' else 'grasp',
        predicates=predicates(s, None), milestone_history=dict(tracker.milestones),
        consecutive_intervals=dict(tracker.streaks), complete_dwell=0,
        remaining_sequence_complete=False, full_closed_start_completion=False,
        events=[], unknown=['place'], interval_evidence='no task action executed yet')


def interval_release(samples):
    assert len(samples) == 25
    return not any('cream_cheese' in str(c) and 'gripper' in str(c)
        for sample in samples for c in sample['contacts'])


def scene_args():
    return dict(root='/home/ami/scratch/LIBERO', scene='fridge', camera='framed_side',
        steps=CAP, gpu=2, init_index=0, seed=0, door='closed', yaw=-np.pi/2,
        fridge_y=.35, open_angle=float(np.deg2rad(130)), item_forward=.04, item_lateral=.035)


def run(args):
    import torch
    import imageio.v2 as imageio
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['MUJOCO_EGL_DEVICE_ID'] == '2'
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for k,v in scene_args().items(): setattr(args, k, v)
    freeze = Path(__file__).resolve().parents[2]/'freeze.json'
    status = dict(status='started', arm=args.arm, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        started=time.time(), freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
        scope='full_closed_start_fridge_candidate_integration_exploratory_actor_readiness_unresolved',
        caps=dict(queries=CAP, actions=CAP, episode_wall_s=WALL_SECONDS),
        phase_prompt_shared_across_all_arms=True, no_synthesized_actions=True)
    env = actor = recorder = None
    states, images, rows = [], [], []
    budget = Budget()
    try:
        env, obs = make_scene(args, out)
        assert env.env.horizon == CAP+40 and not env.env.done
        tracker, gate = FridgeStageEvidence(args.door), FridgeCandidateGate(args.arm)
        s = read_state(env, obs)
        stages = initial_stages(tracker, s)
        status['initial'] = fingerprint(env, obs)
        status['initial_state'] = s
        status['camera_sha256'] = hashlib.sha256(Path(args.camera_config).read_bytes()).hexdigest()
        (out/'model.xml').write_text(env.sim.model.get_xml())
        imageio.imwrite(out/'initial.png', obs['agentview_image'][::-1, ::-1])
        actor = CandidateActor(args, out)
        status['actor_build'] = actor.build
        recorder = ContactRecorder(env.sim, keywords=('cream_cheese', 'short_fridge', 'gripper', 'table_collision'))
        start = time.monotonic()
        save(out/'status.json', status)
        with (out/'steps.jsonl').open('w') as log, (out/'queries.jsonl').open('w') as qlog:
            for step in range(CAP):
                if budget.queries >= CAP or time.monotonic()-start >= WALL_SECONDS:
                    status['termination'] = 'query_budget' if budget.queries >= CAP else 'wall_budget'
                    break
                monitor = gate.begin(step, s, stages)
                language = PROMPTS[stages['phase']]
                candidates = []
                for index in range(3 if gate.active else 1):
                    if budget.queries >= CAP or time.monotonic()-start >= WALL_SECONDS: break
                    t = time.monotonic()
                    budget.query()
                    c, image = actor.query(obs['agentview_image'], language,
                        'greedy' if index == 0 else 'sample', step*10+index)
                    c.update(query=budget.queries-1, step=step, candidate_index=index, language=language,
                        query_wall_s=time.monotonic()-t)
                    candidates.append(c)
                    qlog.write(json.dumps(c)+'\n')
                    qlog.flush()
                    if step == index == 0: image.save(out/'policy_input.png')
                if time.monotonic()-start >= WALL_SECONDS:
                    status['termination'] = 'wall_budget_after_query_no_action'
                    break
                action, decision = gate.choose(candidates, s, monitor)
                decision_trace = trace(step, args.arm, s, stages, candidates, action, monitor, decision, bool(env.check_success()))
                before = fingerprint(env, obs)
                before_state, before_stages = s, stages
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                recorder.reset()
                budget.action()
                obs, _, done, _ = env.step(action)
                assert not env.env.done
                s = read_state(env, obs)
                released = interval_release(recorder.samples)
                stages = tracker.observe(step, s, released)
                events = executed_events(decision, step, action, candidates[0]['action'], gate.transaction)
                row = dict(step=step, phase=before_stages['phase'], language=language,
                    before=before, after=fingerprint(env, obs), before_state=before_state, after_state=s,
                    before_stages=before_stages, after_stages=stages,
                    action=action, candidates=candidates, owner=decision['owner'], decision_trace=decision_trace,
                    events=events, stage_events=stages['events'], interval_released=released,
                    native_done_returned=bool(done), simulator_done_internal=bool(env.env.done),
                    actions_used=budget.actions, queries_used=budget.queries, episode_wall_s=time.monotonic()-start)
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                if stages['remaining_sequence_complete']:
                    status['termination'] = 'strict_sequence_20_step_dwell'
                    break
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
        status.update(status='completed', termination=status.get('termination', 'action_budget'),
            episode_wall_s=time.monotonic()-start, actions=budget.actions, queries=budget.queries,
            any_grasp=any(r['after_state']['grasp'] for r in rows),
            any_native=any(r['native_done_returned'] for r in rows),
            full_closed_start_completion=stages['full_closed_start_completion'],
            remaining_sequence_complete=stages['remaining_sequence_complete'],
            max_complete_dwell=max((r['after_stages']['complete_dwell'] for r in rows), default=0),
            final_stages=stages, final_state=s,
            interventions=sum(r['action'] != r['candidates'][0]['action'] for r in rows),
            local_entries=sum(e['type']=='local_entry' for r in rows for e in r['events']),
            actual_handoffs=sum(e['type']=='handoff_executed' for r in rows for e in r['events']))
    except BaseException:
        status.update(status='failed', error=traceback.format_exc(), actions=budget.actions, queries=budget.queries)
        raise
    finally:
        if states:
            np.savez_compressed(out/'states.npz', states=np.asarray(states))
            np.savez_compressed(out/'images.npz', agent=np.asarray(images))
        status['ended'] = time.time()
        save(out/'status.json', status)
        if actor: actor.close()
        if recorder: recorder.close()
        if env: env.close()


def replay(run_path, out):
    out.mkdir(parents=True, exist_ok=False)
    status = json.loads((run_path/'status.json').read_text())
    args = argparse.Namespace(**status['args'])
    env, obs = make_scene(args, out)
    recorder = ContactRecorder(env.sim, keywords=('cream_cheese', 'short_fridge', 'gripper', 'table_collision'))
    tracker, gate = FridgeStageEvidence(args.door), FridgeCandidateGate(args.arm)
    s = read_state(env, obs)
    stages = initial_stages(tracker, s)
    rows = [json.loads(l) for l in (run_path/'steps.jsonl').read_text().splitlines()]
    states, images = np.load(run_path/'states.npz')['states'], np.load(run_path/'images.npz')['agent']
    report = dict(status='started', steps=len(rows), mismatches=[], max_state_gap=0.)
    try:
        assert len(states) == len(images) == len(rows)+1
        for i, row in enumerate(rows):
            report['max_state_gap'] = max(report['max_state_gap'], float(np.abs(states[i]-env.get_sim_state()).max()))
            if not np.array_equal(images[i], obs['agentview_image']): report['mismatches'].append([i, 'image'])
            if fingerprint(env, obs) != row['before'] or s != row['before_state'] or stages != row['before_stages']:
                report['mismatches'].append([i, 'before'])
            monitor = gate.begin(i, s, stages)
            action, decision = gate.choose(row['candidates'], s, monitor)
            t = trace(i, args.arm, s, stages, row['candidates'], action, monitor, decision, bool(env.check_success()))
            if t != row['decision_trace'] or action != row['action'] or PROMPTS[stages['phase']] != row['language']:
                report['mismatches'].append([i, 'decision'])
            recorder.reset()
            obs, _, done, _ = env.step(action)
            s = read_state(env, obs)
            released = interval_release(recorder.samples)
            stages = tracker.observe(i, s, released)
            events = executed_events(decision, i, action, row['candidates'][0]['action'], gate.transaction)
            if s != row['after_state'] or stages != row['after_stages'] or events != row['events'] or released != row['interval_released']:
                report['mismatches'].append([i, 'measured_after'])
            if fingerprint(env, obs) != row['after'] or bool(done) != row['native_done_returned'] or env.env.done:
                report['mismatches'].append([i, 'physics_after'])
        report['max_state_gap'] = max(report['max_state_gap'], float(np.abs(states[-1]-env.get_sim_state()).max()))
        if not np.array_equal(images[-1], obs['agentview_image']): report['mismatches'].append(['final', 'image'])
        report['status'] = 'exact_replay_passed' if not report['mismatches'] and report['max_state_gap'] == 0 else 'mismatch'
    finally:
        save(out/'verification.json', report)
        recorder.close()
        env.close()
    if report['status'] != 'exact_replay_passed': raise RuntimeError(report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--snapshot')
    p.add_argument('--camera-config')
    p.add_argument('--replay', type=Path)
    a = p.parse_args()
    replay(a.replay, a.out) if a.replay else run(a)
