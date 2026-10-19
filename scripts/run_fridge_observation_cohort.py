"""Six paired actor-only camera trials, GPU1, bounded detached execution."""
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

from fridge_observation_experiment import SOURCES, REPO, save
from run_fridge_r1 import GPUS


def main(out, calibration):
    out.mkdir(parents=True, exist_ok=False)
    gpu = 1
    gpu_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    assert {int(x.split(',')[0]):x.split(',')[1].strip() for x in gpu_rows.splitlines()}[gpu] == GPUS[gpu]
    config = calibration.read_bytes()
    (out/'camera_config.json').write_bytes(config)
    hashes = {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in SOURCES}
    status = dict(status='running', started=time.time(), pid=os.getpid(), gpu=gpu,
        planned=6, sources=hashes, camera_config_sha256=hashlib.sha256(config).hexdigest(),
        scope='actor_only_fixed_framing_experiment1', jobs=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', MUJOCO_EGL_DEVICE_ID='1', MUJOCO_GL='egl',
               PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false')

    def bounded(command, log, timeout):
        with log.open('w') as f:
            process = subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise

    save(out/'launch.json',status)
    for seed in range(3):
        for camera in ('robot_side','framed_side'):
            name = f'seed{seed}_{camera}'
            job = dict(name=name,seed=seed,camera=camera,status='running',started=time.time())
            status['jobs'].append(job)
            save(out/'launch.json',status)
            try:
                assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
                command = [sys.executable,'-u',str(REPO/'scripts/fridge_observation_experiment.py'),
                    '--out',str(out/name),'--seed',str(seed),'--camera',camera,
                    '--camera-config',str(out/'camera_config.json'),'--steps','600','--gpu','1']
                job['command'] = command
                job['returncode'] = bounded(command,out/(name+'.log'),1800)
                if job['returncode']:
                    job['status'] = 'actor_failed'
                    continue
                run = json.loads((out/name/'status.json').read_text())
                job['result'] = {key:run[key] for key in ('steps_executed','any_grasp','any_native_success','termination')}
                verifier = out/name/'source/scripts/fridge_observation_experiment.py'
                job['replay_returncode'] = bounded([sys.executable,str(verifier),'--replay',str(out/name),
                    '--out',str(out/('replay_'+name)),'--gpu','1'],out/('replay_'+name+'.log'),300)
                job['status'] = 'exact_verified' if job['replay_returncode']==0 else 'replay_issue'
            except Exception:
                job.update(status='job_error', error=traceback.format_exc())
            finally:
                job['ended'] = time.time()
                save(out/'launch.json',status)
    status.update(status='complete_verified' if all(j['status']=='exact_verified' for j in status['jobs'])
                  else 'complete_with_issues', ended=time.time())
    save(out/'launch.json',status)
    print(json.dumps(status,indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('calibration',type=Path)
    a = p.parse_args()
    main(a.out.resolve(),a.calibration.resolve())
