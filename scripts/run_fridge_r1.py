"""Detached, bounded R1 actor diagnostics with independent fresh-process replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
GPUS = {1:'GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0',
        2:'GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8',
        3:'GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b'}
SNAPSHOT = '/home/ami/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/80970322773f81baa2e22fe495d0487b93a05cfa'


def main(out, gpus):
    out.mkdir(parents=True,exist_ok=False)
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    found = {int(line.split(',')[0]):line.split(',')[1].strip() for line in raw.splitlines()}
    assert gpus and len(set(gpus)) == len(gpus)
    assert all(g in GPUS and found[g] == GPUS[g] for g in gpus)
    from diagnose_fridge_skills import SOURCES
    files = SOURCES+['scripts/verify_fridge_skills.py','scripts/run_fridge_r1.py']
    hashes = {p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files}
    state = dict(status='running',start=time.time(),pid=os.getpid(),gpus={g:GPUS[g] for g in gpus},
        git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        sources=hashes,planned=12,scope='development_basic_skills_not_framework',jobs=[])
    lock = threading.Lock()

    def save():
        tmp = out/'launch.tmp'
        tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(out/'launch.json')

    def run(seed, gpu):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                   MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        for door in ['closed','open']:
            for prompt in (['canonical','skill'] if seed%2 == 0 else ['skill','canonical']):
                assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
                name=f'seed{seed}_{door}_{prompt}'
                cmd=[sys.executable,'-u',str(REPO/'scripts/diagnose_fridge_skills.py'),
                     '--out',str(out/name),'--mode','actor','--door',door,'--prompt',prompt,
                     '--seed',str(seed),'--steps','160','--gpu',str(gpu),'--snapshot',SNAPSHOT,
                     '--yaw','-1.5707963267948966','--fridge-y','0.35','--camera','robot_side',
                     '--open-angle','2.2689280275926285','--item-forward','0.04','--item-lateral','0.035']
                job=dict(name=name,seed=seed,gpu=gpu,door=door,prompt=prompt,command=cmd,status='running',start=time.time())
                with lock:
                    state['jobs'].append(job)
                    save()
                try:
                    with (out/f'{name}.log').open('w') as log:
                        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                        with lock:
                            job['pid']=proc.pid
                            save()
                        try:
                            code=proc.wait(timeout=900)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                            raise
                    job['actor_returncode']=code
                    if code != 0:
                        job['status']='actor_failed'
                        continue
                    verify=[sys.executable,str(REPO/'scripts/verify_fridge_skills.py'),str(out/name),str(out/f'replay_{name}')]
                    with (out/f'replay_{name}.log').open('w') as log:
                        result=subprocess.run(verify,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    job['replay_returncode']=result.returncode
                    job['status']='exact_verified' if result.returncode==0 else 'replay_failed'
                    job['result']=json.loads((out/name/'status.json').read_text())
                except Exception as exc:
                    job.update(status='job_error',error=repr(exc))
                finally:
                    with lock:
                        job['end']=time.time()
                        save()
    save()
    def lane(index):
        for seed in range(index,3,len(gpus)):
            run(seed,gpus[index])
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lane,range(len(gpus))))
    state['status']='complete_verified' if all(j['status']=='exact_verified' for j in state['jobs']) else 'complete_with_issues'
    state['end']=time.time()
    save()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--gpus',type=int,nargs='+',default=[1,2,3])
    a=p.parse_args()
    main(a.out.resolve(),a.gpus)
