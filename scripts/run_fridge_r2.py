"""Detached R2 matched-pair supervisor: actual VLA in both arms, replay, paired video."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from run_fridge_r1 import GPUS, SNAPSHOT, REPO


def main(out):
    from run_fridge_framework import SOURCES
    out.mkdir(parents=True, exist_ok=False)
    gpu_lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], text=True)
    actual = {int(line.split(',')[0]):line.split(',')[1].strip() for line in gpu_lines.splitlines()}
    assert all(actual[g] == GPUS[g] for g in (1,3))
    files = SOURCES+['scripts/run_fridge_r2.py','scripts/render_fridge_pair.py']
    hashes = {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files}
    status = dict(status='running', started=time.time(), pid=os.getpid(), planned=4,
        scope='actual_VLA_plus_five_layer_wrapper_development', sources=hashes, jobs=[], pairs=[],
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip())
    lock = threading.Lock()

    def save():
        temp = out/'launch.tmp'
        temp.write_text(json.dumps(status,indent=2)+'\n')
        temp.replace(out/'launch.json')

    def bounded(cmd, env, log, timeout):
        with log.open('w') as f:
            p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            try:
                return p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                raise

    def lane(mode, gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false')
        for door in ('closed','open'):
            name=f'seed0_{door}_{mode}'
            job=dict(name=name,mode=mode,door=door,gpu=gpu,status='running',started=time.time())
            with lock:
                status['jobs'].append(job)
                save()
            try:
                assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
                cmd=[sys.executable,'-u',str(REPO/'scripts/run_fridge_framework.py'),
                    '--out',str(out/name),'--mode',mode,'--door',door,'--steps','600',
                    '--seed','0','--gpu',str(gpu),'--snapshot',SNAPSHOT]
                job['command']=cmd
                job['actor_returncode']=bounded(cmd,env,out/f'{name}.log',1800)
                if job['actor_returncode']:
                    job['status']='actor_failed'
                    continue
                verify=[sys.executable,str(REPO/'scripts/verify_fridge_framework.py'),
                        str(out/name),str(out/f'replay_{name}')]
                job['replay_returncode']=bounded(verify,env,out/f'replay_{name}.log',300)
                job['status']='exact_verified' if not job['replay_returncode'] else 'replay_issue'
                run=json.loads((out/name/'status.json').read_text())
                job['result']={k:run.get(k) for k in ('full_completion','any_grasp','steps_executed',
                    'applied_corrections','verified_handoffs','unverified_exits','nominal_actions','final_task')}
            except Exception:
                job.update(status='job_error',error=traceback.format_exc())
            finally:
                with lock:
                    job['ended']=time.time()
                    save()

    save()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,'actor',1),pool.submit(lane,'framework',3)]
        for future in futures:
            future.result()
    for door in ('closed','open'):
        pair=dict(door=door,status='started')
        status['pairs'].append(pair)
        try:
            left,right=[out/f'seed0_{door}_{mode}' for mode in ('actor','framework')]
            video=out/'paired'/f'{door}_baseline_left_framework_right.mp4'
            cmd=[sys.executable,str(REPO/'scripts/render_fridge_pair.py'),str(left),str(right),str(video),'--scope','framework']
            pair['returncode']=bounded(cmd,dict(os.environ),out/f'render_{door}.log',180)
            pair['status']='rendered' if pair['returncode']==0 else 'render_failed'
            pair['video']=str(video)
        except Exception:
            pair.update(status='render_error',error=traceback.format_exc())
        save()
    status.update(status='complete_verified' if all(j['status']=='exact_verified' for j in status['jobs'])
                  and all(p['status']=='rendered' for p in status['pairs']) else 'complete_with_issues',ended=time.time())
    save()
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    a=p.parse_args()
    main(a.out.resolve())
