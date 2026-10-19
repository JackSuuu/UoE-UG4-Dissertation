"""Frozen three-arm native-book development pilot and independent action replay."""
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

from diagnose_vla_failure import make_scene, observe, fingerprint, save, PinnedActor
from adapters.libero_completion import CompletionMonitor, ContactRecorder
from experiment3_candidate_gate import ARMS, CAP, WALL_SECONDS, Budget, CandidateGate, observable_trace, executed_events

PINNED = '/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python'


class CandidateActor(PinnedActor):
    def __init__(self, args, out):
        from adapters.tf5_compat import snapshot_dir
        self.log = (out/'worker.log').open('w')
        self.process = subprocess.Popen([PINNED, '-u', str(Path(__file__).with_name('experiment3_candidate_worker.py')),
            args.snapshot, snapshot_dir()], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1,
            env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        self.build = self.receive()
        assert self.build['ready']

    def query(self, raw, language, source, seed):
        from adapters.libero_openvla import policy_image
        image = policy_image(raw)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        data = buffer.getvalue()
        self.process.stdin.write(json.dumps(dict(language=language, png=base64.b64encode(data).decode(),
            source=source, seed=seed))+'\n')
        self.process.stdin.flush()
        result = self.receive()
        result['policy_png_sha256'] = hashlib.sha256(data).hexdigest()
        return result, image


def args_for_scene():
    return dict(root='/home/ami/scratch/LIBERO', scene='book', camera='native', steps=CAP,
        gpu=2, init_index=0, seed=0)


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
    for k, v in args_for_scene().items(): setattr(args, k, v)
    status = dict(status='started', arm=args.arm, args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        started=time.time(), cap=dict(queries=CAP, actions=CAP, episode_wall_s=WALL_SECONDS),
        scope='single_native_book_init0_development_exploration_no_fridge_transfer_claim',
        freeze_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/'freeze.json').read_bytes()).hexdigest())
    save(out/'status.json', status)
    env = actor = recorder = None
    states, images, rows = [], [], []
    budget = Budget()
    try:
        env, obs = make_scene(args, out)
        # Native done is only the returned success predicate. robosuite self.done
        # remains horizon-based; 20 settle + <=600 actions < horizon 640.
        assert env.env.horizon == CAP+40 and not env.env.done
        status['continuation'] = dict(horizon=env.env.horizon, settle_steps=20,
            native_done='milestone_only', strict_stop='released_supported_stationary_20_steps',
            ignores_native_return_without_reset=True)
        status['initial'] = fingerprint(env, obs)
        status['language'] = env.language_instruction
        (out/'model.xml').write_text(env.sim.model.get_xml())
        imageio.imwrite(out/'initial.png', obs['agentview_image'][::-1, ::-1])
        actor = CandidateActor(args, out)
        status['actor_build'] = actor.build
        recorder = ContactRecorder(env.sim, keywords=('black_book', 'desk_caddy', 'gripper'))
        completion = CompletionMonitor(env)
        gate = CandidateGate(args.arm)
        before_evidence = observe(env, obs, 'black_book_1')
        start = time.monotonic()
        save(out/'status.json', status)
        with (out/'steps.jsonl').open('w') as log, (out/'queries.jsonl').open('w') as qlog:
            for step in range(CAP):
                if budget.queries >= CAP or time.monotonic()-start >= WALL_SECONDS:
                    status['termination'] = 'query_budget' if budget.queries >= CAP else 'wall_budget'
                    break
                monitor = gate.begin(step, before_evidence)
                candidates = []
                for index in range(3 if gate.active else 1):
                    if budget.queries >= CAP or time.monotonic()-start >= WALL_SECONDS: break
                    t = time.monotonic()
                    budget.query()
                    c, image = actor.query(obs['agentview_image'], env.language_instruction,
                        'greedy' if index == 0 else 'sample', step*10+index)
                    c.update(query=budget.queries-1, step=step, candidate_index=index,
                        query_wall_s=time.monotonic()-t)
                    candidates.append(c)
                    qlog.write(json.dumps(c)+'\n')
                    qlog.flush()
                    if step == 0 and index == 0: image.save(out/'policy_input.png')
                if time.monotonic()-start >= WALL_SECONDS:
                    status['termination'] = 'wall_budget_after_query_no_action'
                    break
                action, decision = gate.choose(candidates, before_evidence, monitor)
                trace = observable_trace(step, args.arm, before_evidence, candidates, action, monitor, decision)
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                before = fingerprint(env, obs)
                recorder.reset()
                budget.action()
                obs, _, done, _ = env.step(action)
                assert not env.env.done, 'underlying_horizon_ended'
                assert len(recorder.samples) == 25
                after_evidence = observe(env, obs, 'black_book_1')
                after_evidence['completion'] = completion.measure(step, recorder.samples)
                events = executed_events(decision, step, action, candidates[0]['action'], gate.transaction)
                row = dict(step=step, before=before, after=fingerprint(env, obs), before_evidence=before_evidence,
                    after_evidence=after_evidence, action=action, candidates=candidates, owner=decision['owner'],
                    decision_trace=trace, events=events, native_done_returned=bool(done),
                    simulator_done_internal=bool(env.env.done), queries_used=budget.queries,
                    actions_used=budget.actions, episode_wall_s=time.monotonic()-start)
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                before_evidence = after_evidence
                if after_evidence['completion']['complete']:
                    status['termination'] = 'strict_complete_20_step_dwell'
                    break
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
        status.update(status='completed', termination=status.get('termination', 'action_budget'),
            episode_wall_s=time.monotonic()-start, actions=budget.actions, queries=budget.queries,
            any_grasp=any(r['after_evidence']['grasped'] for r in rows),
            any_native=any(r['after_evidence']['native_success'] for r in rows),
            strict_complete=any(r['after_evidence']['completion']['complete'] for r in rows),
            max_dwell=completion.max_streak, first_native=completion.first_native,
            first_complete=completion.first_complete,
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
    recorder = ContactRecorder(env.sim, keywords=('black_book', 'desk_caddy', 'gripper'))
    completion = CompletionMonitor(env)
    gate = CandidateGate(args.arm)
    rows = [json.loads(l) for l in (run_path/'steps.jsonl').read_text().splitlines()]
    states = np.load(run_path/'states.npz')['states']
    images = np.load(run_path/'images.npz')['agent']
    report = dict(status='started', steps=len(rows), mismatches=[], max_state_gap=0.)
    evidence = observe(env, obs, 'black_book_1')
    try:
        assert len(states) == len(images) == len(rows)+1
        for i, row in enumerate(rows):
            report['max_state_gap'] = max(report['max_state_gap'], float(np.abs(states[i]-env.get_sim_state()).max()))
            if not np.array_equal(images[i], obs['agentview_image']): report['mismatches'].append([i, 'image'])
            for k, expected, actual in [('before', row['before'], fingerprint(env, obs)), ('before_evidence', row['before_evidence'], evidence)]:
                if expected != actual: report['mismatches'].append([i, k])
            monitor = gate.begin(i, evidence)
            action, decision = gate.choose(row['candidates'], evidence, monitor)
            trace = observable_trace(i, args.arm, evidence, row['candidates'], action, monitor, decision)
            if trace != row['decision_trace'] or action != row['action']:
                report['mismatches'].append([i, 'decision_recompute'])
            recorder.reset()
            obs, _, done, _ = env.step(action)
            if executed_events(decision, i, action, row['candidates'][0]['action'], gate.transaction) != row['events']:
                report['mismatches'].append([i, 'actual_handoff_events'])
            evidence = observe(env, obs, 'black_book_1')
            evidence['completion'] = completion.measure(i, recorder.samples)
            if evidence != row['after_evidence'] or fingerprint(env, obs) != row['after'] or bool(done) != row['native_done_returned']:
                report['mismatches'].append([i, 'after'])
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
    p.add_argument('--replay', type=Path)
    a = p.parse_args()
    replay(a.replay, a.out) if a.replay else run(a)
