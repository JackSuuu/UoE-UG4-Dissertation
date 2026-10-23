"""Canonical-runner fridge transfer pilot: unchanged sampled H1 repair proposals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import run_framework_comparison as core

sys.path[:0] = [str(core.REPO), str(core.REPO/'src')]
from src.repair import candidate_pool
from src.repair.fridge_probe import CONFIG, Probe

MONITOR = Path('/home/ami/scratch/fridge_r2_framework_20261019_a/seed0_closed_framework/source/src/adapters/fridge_runtime.py')
REFERENCE = Path('/home/ami/scratch/native_long_l1_20261019_a/microwave_root0')
SELF = 'scripts/run_fridge_repair_probe.py'
FRIDGE_CANDIDATES = dict(candidate_pool.CONFIG, action_cap=600, nominal_query_cap=600,
                        total_query_cap=603, continuation='single_measured_fridge_repair_probe')
# Legacy simulator helpers were intentionally archived out of the working tree.
# Reuse their immutable implementations instead of restoring/deleting user files.
import adapters
adapters.__path__.insert(0, str(REFERENCE/'source/src/adapters'))


def prepare(out):
    manifest = core.load(REFERENCE/'source_manifest.json')
    sources = list(dict.fromkeys([*core.SOURCES, SELF,
        'scripts/diagnose_vla_failure.py', 'scripts/diagnose_fridge_skills.py',
        'scripts/probe_fridge_task.py', 'configs/tasks/fridge_retrieval_dev.bddl',
        'scripts/test_fridge_repair_probe.py', 'scripts/run_native_long_episode.py',
        'scripts/audit_openvla_long_tasks.py', 'src/repair/fridge_probe.py']))
    pins = {str(core.REPO/p): core.sha(core.REPO/p) for p in sources}
    for path in (REFERENCE/'source/src/adapters').glob('*.py'):
        pins[str(path)] = core.sha(path)
    pins[str(MONITOR)] = core.sha(MONITOR)
    pins['/home/ami/scratch/LIBERO/libero/libero/assets/articulated_objects/short_fridge.xml'] = core.sha(
        '/home/ami/scratch/LIBERO/libero/libero/assets/articulated_objects/short_fridge.xml')
    for section in ('runtime', 'assets', 'actor_code'):
        for p, digest in manifest[section].items():
            core.require(core.sha(p) == digest, f'Runtime drift: {p}')
            pins[p] = digest
    weights = {}
    for entry in manifest['checkpoint']['files'].values():
        if 'sha256' in entry:
            core.require(core.sha(entry['path']) == entry['sha256'], 'Model metadata drift')
            pins[entry['path']] = entry['sha256']
        else:
            weights[entry['path']] = {k:entry[k] for k in ('resolved_path', 'bytes', 'cached_blob_id_not_rehashed')}
    original = core.load(REFERENCE/'status.json')['args']
    plan = dict(schema='fridge_frozen_policy_repair_probe_v1', source_sha256=pins, weights=weights,
        actor=dict(snapshot=original['snapshot'], pinned_python=original['pinned_python'], checkpoint=manifest['checkpoint']),
        endpoint_monitor=str(MONITOR), endpoint_monitor_sha256=core.sha(MONITOR),
        controller=CONFIG, candidates=FRIDGE_CANDIDATES,
        budget=dict(actions=600, queries=603, initialization_noops=20, max_episodes=10,
            maximum_queries=6030, rollout_seconds=1200, outer_seconds=1800),
        cells=[dict(name=f'{door}_{"baseline" if slot is None else "candidate"+str(slot)}',
            door=door, slot=slot, seed=0, lane=deepcopy(core.LANES[index]))
            for index, door in enumerate(('closed', 'open')) for slot in (None, 0, 1, 2, 3)],
        scope='Reused seed0 development fridge; closed full task and open grasp diagnostic reported separately',
        primary='Closed: native task success, strict full sequence separately; open: measured grasp, not full task',
        control='constant historical full instruction when closed; constant historical grasp instruction when open',
        repair='first measured stagnation/grasp loss only; archived phase instruction,32unchanged VLA actions maximum',
        verification='archived measured predicates;3consecutive observations plus preservation;8nominal-action retention',
        selection='fixed four-slot collection at first repair only; no trained scorer',
        transfer='new measured task adapter; microwave learned verifier/risk/fusion not transferred',
        stopping='all ten episodes irrespective of outcomes; no automatic retries; invalid episodes kept separate',
        limitations=['one reused seed, not held-out generalization', 'single transaction/H1 candidate coverage',
            'native predicate and strict full sequence differ; open-door setup is not actor opening'])
    out.mkdir(parents=True, exist_ok=False)
    for path in sources:
        dest = out/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((core.REPO/path).read_bytes())
    (out/'archived_endpoint_monitor.py').write_bytes(MONITOR.read_bytes())
    core.save(out/'plan.json', plan)
    print(json.dumps(dict(plan=str(out/'plan.json'), sha256=core.sha(out/'plan.json'))))


def check(plan):
    core.require(plan['controller'] == CONFIG and plan['candidates'] == FRIDGE_CANDIDATES, 'Protocol drift')
    for p, digest in plan['source_sha256'].items():
        core.require(core.sha(p) == digest, f'Frozen source drift: {p}')
    for p, entry in plan['weights'].items():
        core.require(str(Path(p).resolve()) == entry['resolved_path'] and Path(p).stat().st_size == entry['bytes'],
            'Frozen checkpoint blob drift')


def endpoint(plan):
    spec = importlib.util.spec_from_file_location('frozen_fridge_endpoint', plan['endpoint_monitor'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_args(plan, cell):
    import numpy as np
    return argparse.Namespace(scene='fridge', camera='robot_side', root='/home/ami/scratch/LIBERO',
        steps=600, gpu=cell['lane']['egl_device'], seed=cell['seed'], door=cell['door'],
        yaw=-np.pi/2, fridge_y=.35, open_angle=float(np.deg2rad(130)), item_forward=.04, item_lateral=.035,
        snapshot=plan['actor']['snapshot'], pinned_python=plan['actor']['pinned_python'])


def serial(s):
    return {k:v.tolist() if hasattr(v, 'tolist') else v for k,v in s.items()}


def write(stream, row):
    stream.write(json.dumps(row, allow_nan=False)+'\n')
    stream.flush()


def run_episode(plan_path, cell_name, out, replay=None, smoke=False):
    import numpy as np
    import torch
    from diagnose_vla_failure import make_scene, fingerprint, ContactRecorder
    from run_native_long_branch import CandidateActor, controller_fields, validate_decode
    from run_native_long_episode import integration_state
    from adapters.libero_openvla import policy_image
    plan = core.load(plan_path)
    check(plan)
    cell = next(c for c in plan['cells'] if c['name'] == cell_name)
    core.require(core.os.environ.get('CUDA_VISIBLE_DEVICES') == cell['lane']['physical_uuid'], 'Wrong GPU lane')
    out.mkdir(parents=True, exist_ok=False)
    core.save(out/'plan.json', plan)
    args = scene_args(plan, cell)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    status = dict(status='started', valid_episode=False, cell=cell, plan_sha256=core.sha(plan_path),
        actor_queries=0, steps_executed=0, candidate_pools=0, repair=None,
        any_native_success=False, any_grasp=False, full_completion=False, remaining_completion=False,
        actor_construction_attempts=0, mismatches=[], replay_frames=0, started=time.time())
    core.save(out/'status.json', status)
    env = actor = recorder = None
    frames, expected = [], core.rows(replay/'steps.jsonl') if replay else None
    expected_status = core.load(replay/'status.json') if replay else None
    try:
        env, obs = make_scene(args, out)
        monitor_module = endpoint(plan)
        tracker = monitor_module.TaskMonitor(args.door)
        state = serial(monitor_module.sense(env, obs))
        task = tracker.observe(state)
        language = env.language_instruction if args.door == 'closed' else monitor_module.PROMPTS['grasp']
        probe = Probe(language, monitor_module.PROMPTS, cell['slot'] is not None)
        recorder = ContactRecorder(env.sim, keywords=('cream_cheese', 'short_fridge', 'gripper', 'table_collision'))
        (out/'model.xml').write_text(env.sim.model.get_xml())
        status.update(initial=fingerprint(env, obs), language=language)
        def forbid_reset(*a, **kw):
            raise RuntimeError('Post-initialization reset/state installation forbidden')
        env.reset = env.set_init_state = env.env.reset = forbid_reset
        if smoke:
            core.require(np.isfinite(integration_state(env)).all(), 'Invalid integration state')
            core.require(bool(controller_fields(env)), 'Missing controller state')
            status.update(status='scene_smoke_passed', sensed=state, task=task, actual_actor_queries=0)
            return
        if replay:
            core.require(expected_status['valid_episode'], 'Only complete valid tapes can be replayed')
            core.require(expected_status['plan_sha256'] == status['plan_sha256'], 'Replay plan differs')
            core.require(core.sha(out/'model.xml') == core.sha(replay/'model.xml'), 'Replay scene differs')
            def forbidden(*a, **kw):
                raise RuntimeError('Model use forbidden in replay')
            CandidateActor.__init__ = CandidateActor.query = forbidden
        else:
            actor = CandidateActor.__new__(CandidateActor)
            status['actor_construction_attempts'] += 1
            actor_args = argparse.Namespace(gpu=cell['lane']['gpu'], egl_device=cell['lane']['egl_device'],
                readiness_seconds=180, query_seconds=30)
            CandidateActor.__init__(actor, None, args, actor_args, out, cell['lane'])
            status['actor_build'] = actor.build
            (out/'frames').mkdir()
            (out/'model_inputs').mkdir()
        def capture(i):
            frame = dict(state=env.get_sim_state(), integration=integration_state(env),
                agent=obs['agentview_image'], wrist=obs['robot0_eye_in_hand_image'])
            if replay:
                with np.load(replay/'frames'/f'{i:06d}.npz') as old:
                    core.require(all(np.array_equal(v, old[k]) for k,v in frame.items()), f'Physical/image replay mismatch at {i}')
                status['replay_frames'] += 1
            else:
                np.savez_compressed(out/'frames'/f'{i:06d}.npz', **frame)
        start = time.monotonic()
        with (out/'steps.jsonl').open('x') as log, (out/'queries.jsonl').open('x') as query_log:
            count = len(expected) if replay else plan['budget']['actions']
            for step in range(count):
                if time.monotonic()-start >= plan['budget']['rollout_seconds']:
                    raise TimeoutError('Fridge rollout/replay wall budget')
                capture(step)
                before = fingerprint(env, obs)
                controls_before = controller_fields(env)
                decision = probe.before(step, state, task)
                do_pool = decision['owner'] == 'repair' and status['candidate_pools'] == 0
                requests = candidate_pool.requests(decision, 100 if args.door == 'closed' else 101, step) if do_pool else [
                    dict(language=decision['language'], source='greedy', seed=0)]
                buffer = io.BytesIO()
                image = policy_image(obs['agentview_image'])
                image.save(buffer, format='PNG')
                png = buffer.getvalue()
                if not replay:
                    (out/'model_inputs'/f'{step:06d}.png').write_bytes(png)
                candidates = []
                core.require(status['actor_queries']+len(requests) <= plan['budget']['queries'], 'Query budget')
                for index, request in enumerate(requests):
                    metadata = dict(step=step, query_index=status['actor_queries'], **request,
                        policy_png_sha256=hashlib.sha256(png).hexdigest(),
                        policy_rgb_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
                    status['actor_queries'] += 1
                    write(query_log, dict(event='requested_recorded' if replay else 'requested', **metadata))
                    decode = (expected[step]['candidate_selection']['candidates'][index]['decode'] if do_pool else
                        expected[step]['decode']) if replay else actor.query(png, request['language'], request['source'], request['seed'], metadata['query_index'])
                    validate_decode(decode, metadata)
                    core.require(decode['valid'], 'Invalid generation')
                    native = np.asarray(decode['native_action'])
                    core.require(native.shape == (7,) and np.isfinite(native).all() and len(decode['tokens']) == 7, 'Invalid proposal')
                    mapped = native.copy()
                    mapped[-1] = -np.sign(2*mapped[-1]-1)
                    core.require(np.array_equal(mapped, decode['action']), 'Numerically modified proposal')
                    write(query_log, dict(event='returned_recorded' if replay else 'returned', **metadata, decode=decode))
                    candidates.append(dict(candidate_id=index, metadata=metadata, decode=decode))
                selection = None
                if do_pool:
                    context = candidate_pool.context(decision, state, hashlib.sha256(png).hexdigest(),
                        [r['action'] for r in frames[-6:]])
                    decode, selection = candidate_pool.select(candidates, context, collection_slot=cell['slot'])
                    status['candidate_pools'] += 1
                else:
                    decode = candidates[0]['decode']
                recorder.reset()
                obs, reward, done, _ = env.step(decode['action'])
                state = serial(monitor_module.sense(env, obs))
                released = not any('cream_cheese' in str(c) and 'gripper' in str(c)
                    for sample in recorder.samples for c in sample['contacts'])
                task = tracker.observe(state, interval_released=released)
                repair = probe.after(step, state, decision)
                row = dict(step=step, before=before, after=fingerprint(env, obs),
                    controller_before=controls_before, controller_after=controller_fields(env),
                    decision=decision, decode=decode, action=decode['action'], proposal=decode['action'],
                    action_modified=False, sensed=state, task=deepcopy(task), repair=repair,
                    native_success=bool(env.check_success()), done=bool(done), reward=float(reward),
                    contacts_sha256=hashlib.sha256(json.dumps(recorder.samples, sort_keys=True).encode()).hexdigest())
                if selection is not None:
                    row['candidate_selection'] = selection
                if replay:
                    core.require(row == expected[step], f'Independent controller/endpoint replay mismatch at {step}')
                write(log, row)
                frames.append(row)
                status.update(steps_executed=step+1, repair=repair,
                    any_native_success=status['any_native_success'] or row['native_success'],
                    any_grasp=status['any_grasp'] or state['grasp'],
                    full_completion=status['full_completion'] or task['full_completion'],
                    remaining_completion=status['remaining_completion'] or task['remaining_sequence_complete'])
                if step % 20 == 0:
                    core.save(out/'status.json', status)
                # Native done is terminal; never step after it to manufacture dwell.
                if done or row['native_success'] or task['remaining_sequence_complete']:
                    status['termination'] = 'native_success' if row['native_success'] else 'environment_done' if done else 'strict_completion'
                    break
            capture(len(frames))
        if replay:
            core.require(len(frames) == len(expected) and status['actor_queries'] == expected_status['actor_queries'], 'Truncated replay')
        status.update(status='exact_replay_passed' if replay else 'completed', valid_episode=True,
            termination=status.get('termination', 'action_budget'),
            actual_actor_queries=0 if replay else status['actor_queries'], rollout_wall_s=time.monotonic()-start)
    except BaseException:
        status.update(status='invalid', error=traceback.format_exc())
        raise
    finally:
        status['ended'] = time.time()
        core.save(out/'status.json', status)
        if actor is not None and hasattr(actor, 'process'):
            actor.process.stdin.close()
            try:
                actor.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                actor.process.kill()
                actor.process.wait()
            actor.log.close()
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()


def study(plan_path, out):
    plan = core.load(plan_path)
    check(plan)
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    (out/'episodes').mkdir()
    (out/'replays').mkdir()
    core.save(out/'protocol.json', plan)
    def lane(gpu):
        results = []
        for cell in plan['cells']:
            if cell['lane']['gpu'] != gpu:
                continue
            source = out/'episodes'/cell['name']
            result = dict(cell=cell, source=str(source))
            for phase in ('episode', 'replay'):
                if phase == 'replay' and result['episode_returncode'] != 0:
                    break
                dest = source if phase == 'episode' else out/'replays'/cell['name']
                command = [core.PYTHON, '-B', '-u', str(core.SCRIPT), '--fridge-probe',
                    '--'+phase, cell['name'] if phase == 'episode' else str(source),
                    '--plan', str(plan_path), '--out', str(dest)]
                record = dict(command=command, started=time.time())
                with (out/'logs'/f"{cell['name']}.{phase}.log").open('xb') as stream:
                    p = subprocess.Popen(command, cwd=core.REPO, env=core.environment(cell['lane']),
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    record['pid'] = p.pid
                    core.save(out/f"{cell['name']}.{phase}.json", record)
                    try:
                        code = p.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        core.terminate_group(p)
                        code = -15
                    record.update(returncode=code, ended=time.time())
                    core.save(out/f"{cell['name']}.{phase}.json", record)
                    result[phase+'_returncode'] = code
            status = core.load(source/'status.json') if (source/'status.json').exists() else {}
            result.update({k:status.get(k) for k in ('any_native_success', 'any_grasp', 'full_completion',
                'remaining_completion', 'steps_executed', 'actor_queries', 'repair', 'candidate_pools')})
            result['valid'] = bool(status.get('valid_episode') and result['episode_returncode'] == 0 and result.get('replay_returncode') == 0)
            core.save(out/f"{cell['name']}.result.json", result)
            print(json.dumps(result), flush=True)
            results.append(result)
        return results
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [r for lane_results in pool.map(lane, (2, 3)) for r in lane_results]
    core.save(out/'results.json', results)
    return 0 if all(r['valid'] for r in results) else 1


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--study', action='store_true')
    mode.add_argument('--episode')
    mode.add_argument('--smoke', help='Cell identity; scene and endpoint preflight with zero actor queries')
    mode.add_argument('--replay', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    if a.prepare:
        prepare(a.prepare.resolve())
        return 0
    if not a.plan or not a.out:
        p.error('--plan and --out required')
    if a.study:
        return study(a.plan.resolve(), a.out.resolve())
    name = core.load(a.replay/'status.json')['cell']['name'] if a.replay else a.episode or a.smoke
    run_episode(a.plan.resolve(), name, a.out.resolve(), a.replay.resolve() if a.replay else None, bool(a.smoke))
    return 0
