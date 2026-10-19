"""Detached actor-only audit; frozen task/camera/decoder conditions, no repair."""
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

from run_fridge_r1 import REPO, GPUS, SNAPSHOT
from diagnose_vla_failure import SOURCES


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    actual = {int(line.split(',')[0]):line.split(',')[1].strip() for line in gpu_lines.splitlines()}
    assert all(actual[g] == GPUS[g] for g in (1, 2, 3))
    hashes = {p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in
              SOURCES+['scripts/run_vla_failure_audit.py']}
    status = dict(status='running', started=time.time(), pid=os.getpid(), planned=5,
        scope='actor_only_task_camera_decode_diagnosis', sources=hashes, jobs=[],
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
    lock = threading.Lock()

    def save():
        tmp = out/'launch.tmp'
        tmp.write_text(json.dumps(status, indent=2)+'\n')
        tmp.replace(out/'launch.json')

    def bounded(cmd, env, log, timeout):
        with log.open('w') as f:
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                return p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                raise

    def lane(gpu, jobs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        for scene, camera, decoder in jobs:
            name = f'{scene}_{camera}_{decoder}'
            job = dict(name=name, gpu=gpu, status='running', started=time.time())
            with lock:
                status['jobs'].append(job)
                save()
            try:
                assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest() == h for p,h in hashes.items())
                cmd = [sys.executable, '-u', str(REPO/'scripts/diagnose_vla_failure.py'),
                    '--out', str(out/name), '--scene', scene, '--camera', camera,
                    '--decoder', decoder, '--gpu', str(gpu), '--snapshot', SNAPSHOT, '--steps', '600']
                job['command'] = cmd
                job['returncode'] = bounded(cmd, env, out/f'{name}.log', 1800)
                if job['returncode']:
                    job['status'] = 'actor_failed'
                    continue
                run = json.loads((out/name/'status.json').read_text())
                job['result'] = {k:run.get(k) for k in ('steps_executed', 'any_grasp', 'any_native_success', 'any_complete', 'termination')}
                verify = [sys.executable, str(REPO/'scripts/verify_vla_failure.py'),
                          str(out/name), str(out/f'replay_{name}')]
                job['replay_returncode'] = bounded(verify, env, out/f'replay_{name}.log', 300)
                job['status'] = 'exact_verified' if job['replay_returncode'] == 0 else 'replay_issue'
            except Exception:
                job.update(status='job_error', error=traceback.format_exc())
            finally:
                with lock:
                    job['ended'] = time.time()
                    save()

    save()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lane, 1, [('book', 'native', 'pinned'), ('fridge', 'robot_side', 'pinned')]),
                   pool.submit(lane, 2, [('book', 'robot_side', 'pinned'), ('cheese', 'native', 'pinned')]),
                   pool.submit(lane, 3, [('book', 'native', 'current')])]
        for future in futures:
            future.result()
    status.update(status='complete_verified' if all(j['status'] == 'exact_verified' for j in status['jobs'])
                  else 'complete_with_issues', ended=time.time())
    save()
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.out.resolve())
