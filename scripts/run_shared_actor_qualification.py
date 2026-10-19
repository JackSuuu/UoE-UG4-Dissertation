"""Experiment 2 only: frozen actor qualification, exact replay and review video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from diagnose_vla_failure import (SOURCES as AUDIT_SOURCES, make_scene, PinnedActor,
    observe, fingerprint, save, CompletionMonitor, ContactRecorder)
from run_fridge_r1 import REPO, GPUS, SNAPSHOT

PINNED = '/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python'
PROTOCOL = 'exp_record/agent_notes/shared_actor_qualification_protocol_20261019.md'
SELF = 'scripts/run_shared_actor_qualification.py'
SOURCES = list(dict.fromkeys(AUDIT_SOURCES + ['scripts/run_fridge_r1.py', SELF, PROTOCOL]))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def gpu_check():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '3'
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                  '--format=csv,noheader'], text=True)
    actual = {int(s.split(',')[0]): s.split(',')[1].strip() for s in raw.splitlines()}
    assert actual[3] == GPUS[3]


def clock(env):
    return dict(sim_time=float(env.sim.data.time), timestep=int(env.env.timestep),
                internal_done=bool(env.env.done))


def transition(env, before):
    after = clock(env)
    assert not before['internal_done'], before
    assert after['timestep'] == before['timestep'] + 1, (before, after)
    assert abs(after['sim_time'] - before['sim_time'] - env.env.control_timestep) < 1e-9
    return after


def episode(out, index, manifest):
    import random
    import torch
    import imageio.v2 as imageio
    gpu_check()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(scene='book', camera='native', decoder='pinned',
        root='/home/ami/scratch/LIBERO', steps=600, gpu=3, init_index=index,
        seed=0, pinned_python=PINNED, snapshot=SNAPSHOT)
    status = dict(status='started', started=time.time(), scope='shared_frozen_actor_qualification',
        args=vars(args), sources=manifest['sources'], freeze_root=str(REPO),
        intervention_actions=0, setup_settling_calls=20, steps_executed=0,
        queries=0, native_done_continuations=0)
    save(out/'status.json', status)
    env = actor = recorder = None
    states, images, rows = [], [], []
    try:
        env, obs = make_scene(args, out)
        language = env.language_instruction
        status.update(language=language, initial=fingerprint(env, obs), initial_clock=clock(env))
        (out/'model.xml').write_text(env.sim.model.get_xml())
        imageio.imwrite(out/'initial.png', obs['agentview_image'][::-1, ::-1])
        actor = PinnedActor(args, out)
        status['actor_build'] = actor.build
        assert actor.build['transformers'] == '4.40.1' and actor.build['H'] == 1
        completion = CompletionMonitor(env)
        recorder = ContactRecorder(env.sim, keywords=('black_book', 'desk_caddy', 'gripper'))
        # Any accidental post-setup reset/state initialization is a hard error.
        def forbidden(*a, **kw):
            raise RuntimeError('Post-setup reset/state initialization forbidden')
        env.reset = env.set_init_state = env.env.reset = forbidden
        save(out/'status.json', status)
        start = time.monotonic()
        previous_done = False
        with (out/'steps.jsonl').open('w') as log, (out/'contacts.jsonl').open('w') as contacts:
            for step in range(600):
                if time.monotonic() - start > 1200:
                    status['termination'] = 'wall_budget'
                    break
                states.append(env.get_sim_state().copy())
                images.append(obs['agentview_image'].copy())
                before, tick = fingerprint(env, obs), clock(env)
                assert not tick['internal_done']
                t = time.monotonic()
                action, decode, image = actor.act(obs['agentview_image'], language)
                status['queries'] += 1
                inference = time.monotonic() - t
                assert action.shape == (7,) and np.isfinite(action).all()
                if step == 0:
                    image.save(out/'policy_input.png')
                recorder.reset()
                t = time.monotonic()
                obs, reward, done, _ = env.step(action.tolist())
                simulation = time.monotonic() - t
                after_tick = transition(env, tick)
                measurement = completion.measure(step, recorder.samples)
                assert bool(done) == measurement['native_success']
                evidence = dict(input_language=language, input_rgb_sha256=digest(images[-1].tobytes()),
                    processed_rgb_sha256=digest(np.asarray(image).tobytes()),
                    input_scope='RGB_and_language_only', selection='identity_of_model_proposal',
                    hidden_reasoning_available=False, monitor_role='logging_and_strict_stop_only')
                row = dict(step=step, action=action.tolist(), proposal=action.tolist(), owner='vla',
                    inference_s=inference, simulation_s=simulation, elapsed_s=time.monotonic()-start,
                    before=before, after=fingerprint(env, obs), clock_before=tick, clock_after=after_tick,
                    returned_done=bool(done), action_after_native_done=previous_done,
                    reward=float(reward), completion=measurement, decision_trace=evidence,
                    contact_samples_sha256=digest(json.dumps(recorder.samples, sort_keys=True).encode()),
                    **decode, **observe(env, obs, 'black_book_1'))
                if previous_done:
                    status['native_done_continuations'] += 1
                previous_done = bool(done)
                rows.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                contacts.write(json.dumps(dict(step=step, samples=recorder.samples))+'\n')
                contacts.flush()
                status['steps_executed'] = step+1
                if step % 40 == 0 or done or measurement['complete']:
                    save(out/'status.json', status)
                    print(json.dumps(dict(step=step, native=bool(done), strict=measurement['complete'],
                        streak=measurement['streak'], continuation=previous_done)), flush=True)
                if measurement['complete']:
                    status['termination'] = 'strict_completion'
                    break
            states.append(env.get_sim_state().copy())
            images.append(obs['agentview_image'].copy())
        status.update(status='completed', termination=status.get('termination', 'action_budget'),
            any_grasp=any(r['grasped'] for r in rows),
            first_grasp=next((r['step'] for r in rows if r['grasped']), None),
            first_native=completion.first_native, first_complete=completion.first_complete,
            max_strict_streak=completion.max_streak,
            any_native_success=completion.first_native is not None,
            any_complete=completion.first_complete is not None, final_clock=clock(env),
            final=observe(env, obs, 'black_book_1'), rollout_wall_s=time.monotonic()-start)
    except BaseException:
        status.update(status='failed', termination='error', error=traceback.format_exc())
        raise
    finally:
        if states:
            np.savez_compressed(out/'states.npz', states=np.asarray(states))
            np.savez_compressed(out/'images.npz', agent=np.asarray(images))
        status['ended'] = time.time()
        save(out/'status.json', status)
        if actor:
            actor.close()
        if recorder:
            recorder.close()
        if env:
            env.close()


def replay(run, out):
    gpu_check()
    status = json.loads((run/'status.json').read_text())
    assert status['status'] == 'completed'
    for name, sha in status['sources'].items():
        assert digest((REPO/name).read_bytes()) == sha, name
    out.mkdir(exist_ok=False)
    env, obs = make_scene(argparse.Namespace(**status['args']), out)
    recorder = ContactRecorder(env.sim, keywords=('black_book', 'desk_caddy', 'gripper'))
    completion = CompletionMonitor(env)
    report = dict(status='started', mismatches=[], max_state_gap=0., continuations=0,
                  actor_requeried=False, fresh_process=True)
    try:
        rows = [json.loads(s) for s in (run/'steps.jsonl').read_text().splitlines()]
        states = np.load(run/'states.npz')['states']
        images = np.load(run/'images.npz')['agent']
        assert len(states) == len(images) == len(rows)+1
        previous_done = False
        for i, row in enumerate(rows):
            gap = float(np.max(np.abs(env.get_sim_state()-states[i])))
            report['max_state_gap'] = max(gap, report['max_state_gap'])
            expected = dict(before=fingerprint(env, obs), clock_before=clock(env))
            assert row['action'] == row['proposal'] and row['owner'] == 'vla'
            if not np.array_equal(images[i], obs['agentview_image']):
                report['mismatches'].append([i, 'image'])
            recorder.reset()
            obs, reward, done, _ = env.step(row['action'])
            expected.update(after=fingerprint(env, obs), clock_after=transition(env, expected['clock_before']),
                completion=completion.measure(i, recorder.samples), returned_done=bool(done),
                reward=float(reward), action_after_native_done=previous_done,
                contact_samples_sha256=digest(json.dumps(recorder.samples, sort_keys=True).encode()),
                **observe(env, obs, 'black_book_1'))
            report['continuations'] += int(previous_done)
            previous_done = bool(done)
            for key, value in expected.items():
                if value != row[key]:
                    report['mismatches'].append([i, key])
        report['max_state_gap'] = max(report['max_state_gap'],
            float(np.max(np.abs(env.get_sim_state()-states[-1]))))
        if not np.array_equal(images[-1], obs['agentview_image']):
            report['mismatches'].append([len(rows), 'final_image'])
        assert completion.first_native == status['first_native']
        assert completion.first_complete == status['first_complete']
        assert report['continuations'] == status['native_done_continuations']
        report.update(status='exact_replay_passed' if not report['mismatches'] and
            report['max_state_gap'] == 0 else 'replay_mismatch', steps=len(rows),
            first_native=completion.first_native, first_complete=completion.first_complete,
            max_strict_streak=completion.max_streak)
    finally:
        save(out/'verification.json', report)
        recorder.close()
        env.close()
    assert report['status'] == 'exact_replay_passed', report


def review(out):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    runs = []
    for index in range(3):
        path = out/f'book_init{index}'
        status = json.loads((path/'status.json').read_text())
        rows = [json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
        frames = np.load(path/'images.npz')['agent']
        runs.append((status, rows, frames))
    max_frames = max(len(x[2]) for x in runs)
    with imageio.get_writer(out/'actor_only_qualification_annotated.mp4', fps=20,
                           codec='libx264', quality=8, macro_block_size=16) as video:
        for frame in range(max_frames+40):
            canvas = Image.new('RGB', (1152, 640), '#121721')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), 'EXPERIMENT 2 | frozen OpenVLA TF4 | actor-only native-book qualification', font=font)
            draw.text((12, 30), '20 fps simulation time; inference pauses physics | zero repair/human motion actions', font=font)
            for index, (status, rows, frames) in enumerate(runs):
                x = index*384
                f = min(frame, len(frames)-1)
                canvas.paste(Image.fromarray(frames[f][::-1, ::-1]).resize((384,384)), (x, 64))
                row = rows[min(max(0, f-1), len(rows)-1)]
                ended = frame >= len(frames)-1
                lines = [f'Init {index} | owner: VLA | actions {min(f,len(rows))}/600',
                    f'Native first step: {status["first_native"]}',
                    f'Strict first step: {status["first_complete"]}',
                    f'Strict streak: {row["completion"]["streak"]}/20',
                    f'Native now: {row["native_success"]} | strict: {row["completion"]["complete"]}',
                    f'Stop: {status["termination"]}' if ended else 'Running: model query -> model action',
                    'END HOLD (no physics / no extra dwell)' if ended else f'Inference: {row["inference_s"]:.3f}s',
                    f'Post-native-done actions: {status["native_done_continuations"]}']
                for j, text in enumerate(lines):
                    draw.text((x+7, 454+j*22), text, fill='#83e1ae' if j==0 else 'white', font=font)
            video.append_data(np.asarray(canvas))
    summary = []
    for status, rows, frames in runs:
        summary.append({**{k:status[k] for k in ('args','steps_executed','queries','first_grasp',
            'first_native','first_complete','max_strict_streak','native_done_continuations','termination')},
            'median_inference_s': float(np.median([r['inference_s'] for r in rows])),
            'final_conditions': rows[-1]['completion']['conditions'],
            'any_strict_candidate': any(r['completion']['candidate'] for r in rows),
            'unique_token_sequences': len({tuple(r['tokens']) for r in rows})})
    save(out/'summary.json', summary)


def supervise(out):
    gpu_check()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'source'
    hashes = {}
    for name in SOURCES:
        data = (REPO/name).read_bytes()
        dest = frozen/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hashes[name] = digest(data)
    import robosuite
    external = [Path('/home/ami/scratch/LIBERO/libero/libero/envs')/name for name in
                ('env_wrapper.py', 'bddl_base_domain.py')]
    external.append(Path(robosuite.__file__).parent/'environments/base.py')
    external_hashes = {}
    for i, source in enumerate(external):
        data = source.read_bytes()
        dest = out/'external_semantics'/f'{i}_{source.name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(data)
        external_hashes[str(source)] = digest(data)
    manifest = dict(status='running', started=time.time(), pid=os.getpid(), sources=hashes,
        external_sources=external_hashes, gpu=3, gpu_uuid=GPUS[3], checkpoint=SNAPSHOT,
        pinned_python=PINNED, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],
        cwd=REPO,text=True).strip(), jobs=[])
    save(out/'launch.json', manifest)
    def bounded(cmd, log, timeout):
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, cwd=frozen, stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                return proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
    for index in range(3):
        job = dict(index=index, started=time.time(), status='running')
        manifest['jobs'].append(job)
        save(out/'launch.json', manifest)
        try:
            assert all(digest((frozen/p).read_bytes()) == h for p,h in hashes.items())
            assert all(digest(Path(p).read_bytes()) == h for p,h in external_hashes.items())
            run = out/f'book_init{index}'
            cmd = [sys.executable, '-u', str(frozen/SELF), 'episode', str(run),
                   '--index', str(index), '--manifest', str(out/'launch.json')]
            job['command'] = cmd
            job['actor_returncode'] = bounded(cmd, out/f'book_init{index}.log', 1800)
            assert job['actor_returncode'] == 0
            cmd = [sys.executable, '-u', str(frozen/SELF), 'replay', str(run),
                   '--replay-out', str(out/f'replay_init{index}')]
            job['replay_returncode'] = bounded(cmd, out/f'replay_init{index}.log', 360)
            assert job['replay_returncode'] == 0
            job['status'] = 'exact_verified'
        except Exception:
            job.update(status='failed', error=traceback.format_exc())
        finally:
            job['ended'] = time.time()
            save(out/'launch.json', manifest)
    if all(j['status'] == 'exact_verified' for j in manifest['jobs']):
        manifest['review_returncode'] = bounded([sys.executable, str(frozen/SELF), 'review', str(out)],
                                                out/'review.log', 360)
    manifest.update(status='complete_verified' if all(j['status'] == 'exact_verified' for j in
        manifest['jobs']) and manifest.get('review_returncode') == 0 else 'complete_with_issues', ended=time.time())
    save(out/'launch.json', manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['supervise', 'episode', 'replay', 'review'])
    p.add_argument('out', type=Path)
    p.add_argument('--index', type=int)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--replay-out', type=Path)
    a = p.parse_args()
    if a.mode == 'supervise':
        supervise(a.out.resolve())
    elif a.mode == 'episode':
        episode(a.out.resolve(), a.index, json.loads(a.manifest.read_text()))
    elif a.mode == 'replay':
        replay(a.out.resolve(), a.replay_out.resolve())
    else:
        review(a.out.resolve())
