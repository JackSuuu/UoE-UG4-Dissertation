"""Bounded, detached three-GPU execution of an explicitly frozen L1 job plan.

The plan contains argument arrays, never shell commands. One infrastructure or
replay failure blocks that GPU lane pending inspection; native task failures do
not stop screening. No retries, adaptive root selection or model changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from run_fridge_r1 import REPO, GPUS
from audit_openvla_long_tasks import PRIORITY

# Audited via eglQueryDeviceAttribEXT(EGL_CUDA_DEVICE_NV), then CUDA UUIDs.
# EGL index order is REVERSED on this host; matching numeric IDs is incorrect.
EGL_DEVICES = {1: 2, 2: 1, 3: 0}
TASK_ARGS = {'microwave': 'microwave', 'two_pots': 'moka', 'two_mugs': 'mugs', 'drawer': 'drawer'}
TASK_NAMES = dict(zip(TASK_ARGS, PRIORITY))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argument(command, flag):
    require(command.count(flag) == 1, f'Expected one {flag}')
    index = command.index(flag) + 1
    require(index < len(command), f'Missing value for {flag}')
    return command[index]


def main(plan_path, out, validate_only=False):
    plan = json.loads(plan_path.read_text())
    jobs = plan['jobs']
    require(len(jobs) == 24 and len({j['name'] for j in jobs}) == 24, 'Expected24unique jobs')
    require(sorted({j['gpu'] for j in jobs}) == [1, 2, 3], 'Expected GPUs1/2/3')
    tasks = sorted({j['task'] for j in jobs})
    require(set(tasks) == set(TASK_ARGS), 'Wrong native task cohort')
    for task in tasks:
        require(sorted(j['init_index'] for j in jobs if j['task'] == task) == list(range(6)), task)
    paths = []
    for job in jobs:
        require(job['command'] and job['replay_command'], 'Missing command')
        require(job['gpu'] == 1 + job['init_index'] % 3, 'Wrong GPU lane')
        require(job['egl_device'] == EGL_DEVICES[job['gpu']], 'Wrong EGL device')
        for key in ('episode_dir', 'replay_dir'):
            path = Path(job[key]).resolve()
            require(path.parent == out and not path.exists(), f'Output not fresh: {path}')
            paths.append(path)
        command, replay = job['command'], job['replay_command']
        require(str(REPO / 'scripts/run_native_long_episode.py') in command and
                str(REPO / 'scripts/run_native_long_episode.py') in replay, 'Wrong episode entry point')
        for flag, expected in {'--task': TASK_ARGS[job['task']], '--init-index': str(job['init_index']),
                               '--gpu': str(job['gpu']), '--egl-device': str(job['egl_device']),
                               '--steps': '520', '--seed': '0', '--out': job['episode_dir']}.items():
            require(argument(command, flag) == expected, f'Episode command disagrees with {flag}')
        require('--smoke' not in command and '--replay' not in command, 'Not a full actor command')
        require(argument(replay, '--replay') == job['episode_dir'] and
                argument(replay, '--out') == job['replay_dir'], 'Wrong replay command identities')
    require(len(set(paths)) == 48, 'Output directories must be unique')
    require(plan.get('report_command'), 'Reporting is mandatory')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                   '--format=csv,noheader'], text=True)
    actual = {int(line.split(',')[0]): line.split(',')[1].strip() for line in raw.splitlines()}
    require(all(actual[g] == GPUS[g] for g in (1, 2, 3)), 'Physical GPU identity mismatch')
    hashes = plan['source_sha256']
    require(all(sha(REPO / name) == value for name, value in hashes.items()), 'Source drift')
    if validate_only:
        print(json.dumps(dict(status='plan_validated', jobs=len(jobs), gpu_to_egl=EGL_DEVICES,
                              model_queries=0, robot_actions=0)), flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_bytes(plan_path.read_bytes())
    for name in hashes:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / name).read_bytes())
    status = dict(status='running', planned=24, scope='L1_native_actor_screening',
                  pid=os.getpid(), started=time.time(), plan_sha256=sha(plan_path),
                  git=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              cwd=REPO, text=True).strip(),
                  source_sha256=hashes, jobs=[dict(j, status='planned') for j in jobs])
    lock = threading.RLock()
    active = set()
    active_lock = threading.Lock()
    stopping = threading.Event()

    def cleanup_group(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def terminate(signum, frame):
        stopping.set()
        with lock:
            status.update(status='interrupted', signal=signum, interrupted=time.time())
            save_locked()
        with active_lock:
            for pid in list(active):
                cleanup_group(pid)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def save_locked():
        tmp = out / 'launch.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(out / 'launch.json')

    def change(job, **fields):
        with lock:
            job.update(fields)
            save_locked()

    def run_command(command, env, log_path, timeout, job, phase):
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with active_lock:
                active.add(child.pid)
            try:
                change(job, **{phase + '_pid': child.pid})
                return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            finally:
                # Also clean descendants when a parent crashes or exits normally.
                cleanup_group(child.pid)
                child.wait()
                with active_lock:
                    active.discard(child.pid)

    def lane(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu],
                   MUJOCO_EGL_DEVICE_ID=str(EGL_DEVICES[gpu]),
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                   TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', PYTHONOPTIMIZE='0')
        blocked = None
        for job in status['jobs']:
            if job['gpu'] != gpu:
                continue
            if blocked or stopping.is_set():
                change(job, status='blocked_after_infrastructure_issue', blocked_by=blocked)
                continue
            change(job, status='running', started=time.time())
            try:
                require(all(sha(REPO / name) == value for name, value in hashes.items()), 'Source drift')
                code = run_command(job['command'], env, out / (job['name'] + '.log'),
                                   1800, job, 'episode')
                change(job, actor_returncode=code)
                if code:
                    raise RuntimeError(f'Actor worker returned {code}')
                episode = json.loads((Path(job['episode_dir']) / 'status.json').read_text())
                require(episode['status'] == 'completed', episode['status'])
                args = episode['args']
                require(args['task'] == TASK_NAMES[job['task']] and
                        args['init_index'] == job['init_index'] and args['gpu'] == gpu,
                        'Episode task/root/GPU identity mismatch')
                require(args['steps'] == 520 and args['seed'] == 0, 'Episode protocol mismatch')
                require(episode['initialization_actions'] == 10 and
                        0 <= episode['steps_executed'] <= episode['actor_queries'] <= 520,
                        'Episode action/query budget mismatch')
                require(Path(args['out']).resolve() == Path(job['episode_dir']).resolve(),
                        'Episode output identity mismatch')
                change(job, status='replaying', episode_status=episode)
                code = run_command(job['replay_command'], env,
                                   out / ('replay_' + job['name'] + '.log'), 900, job, 'replay')
                change(job, replay_returncode=code)
                if code:
                    raise RuntimeError(f'Replay worker returned {code}')
                replay = json.loads((Path(job['replay_dir']) / 'verification.json').read_text())
                require(replay['status'] == 'exact_replay_passed', 'No exact replay proof')
                require(not replay.get('mismatches') and replay.get('max_state_gap') == 0,
                        'Replay mismatches or missing state evidence')
                require(replay.get('max_integration_gap') == 0 and
                        replay['steps'] == episode['steps_executed'], 'Replay integration/step mismatch')
                require(Path(replay['source']).resolve() == Path(job['episode_dir']).resolve(),
                        'Replay source mismatch')
                change(job, status='exact_verified', ended=time.time())
            except Exception:
                blocked = job['name']
                change(job, status='infrastructure_or_replay_issue', error=traceback.format_exc(),
                       ended=time.time())

    with lock:
        save_locked()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lane, gpu) for gpu in (1, 2, 3)]
        for future in futures:
            future.result()
    with lock:
        status.update(status='reporting', rollouts_verified=all(j['status'] == 'exact_verified'
                      for j in status['jobs']), rollouts_ended=time.time())
        save_locked()
    report_command = plan.get('report_command')
    try:
        with (out / 'report.log').open('w') as log:
            result = subprocess.run(report_command, cwd=REPO, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        with lock:
            status['report_returncode'] = result.returncode
            save_locked()
        if result.returncode == 0:
            report = json.loads((out / 'report' / 'summary.json').read_text())
            identities = {(case['task'], case['root']) for case in report['episodes']}
            require(len(report['episodes']) == 24 and identities ==
                    {(task, index) for task in TASK_ARGS for index in range(6)},
                    'Report omitted or duplicated cohort identities')
    except Exception:
        status['report_error'] = traceback.format_exc()
    with lock:
        status.update(status='complete_verified' if status['rollouts_verified'] and
                      status.get('report_returncode') == 0 and not status.get('report_error')
                      else 'complete_with_issues', ended=time.time())
        save_locked()
    print(json.dumps({'status': status['status'], 'run': str(out)}, indent=2), flush=True)
    return 0 if status['status'] == 'complete_verified' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    raise SystemExit(main(args.plan.resolve(), args.out.resolve(), args.validate_only))
