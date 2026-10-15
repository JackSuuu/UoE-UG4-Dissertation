"""Detached M1 supervisor: all39cases × five arms, no baseline-success veto."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from launch_libero_preservation import EXPECTED


def launch(args):
    repo=Path(__file__).resolve().parents[1]
    preflight=json.loads(args.smoke.read_text())
    assert preflight['status']=='passed' and preflight['fixture_only']
    assert preflight['exact_sham'] and preflight['full_action_and_forecast_replay']
    pose=json.loads(args.pose_smoke.read_text())
    assert pose['status']=='passed' and pose['fixture_only']
    assert pose['source_sha256']==preflight['source_sha256']
    for path,sha in preflight['source_sha256'].items():
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==sha,path
    grid=json.loads(args.cases.read_text())
    assert grid['development_only'] and grid['final_roots_used']==[]
    assert len(grid['cases'])==39 and grid['total_episodes']==195
    assert {c['root'] for c in grid['cases']}=={0,1,2}
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
                                      '--format=csv,noheader,nounits'],text=True)
    available={}
    for line in hardware.splitlines():
        i,uuid,mem=map(str.strip,line.split(','));available[int(i)]=[uuid,int(mem)]
    for gpu,uuid in EXPECTED.items():
        assert available[gpu][0]==uuid and available[gpu][1]<1024,available
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.smoke,args.out/'preflight_smoke.json')
    shutil.copy2(args.pose_smoke,args.out/'preflight_pose_smoke.json')
    shutil.copy2(args.cases,args.out/'cases.json')
    protocol=repo/'exp_record/libero_failure_mining_protocol_20261015.md'
    shutil.copy2(protocol,args.out/'protocol.md')
    meta=dict(status='running',start=time.time(),phase='all_case_paired_discovery',supervisor=os.getpid(),
        git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        GPUs=available,workers=[],episodes_planned=195,source_sha256=preflight['source_sha256'],
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        development_only=True,baseline_success_cancels_interventions=False)
    def save():
        tmp=args.out/'launch.json.tmp';tmp.write_text(json.dumps(meta,indent=2)+'\n');tmp.replace(args.out/'launch.json')
    processes,handles=[],[]
    save()
    try:
        for index,gpu in enumerate((1,2,3)):
            handle=(args.out/f'worker_gpu_{gpu}.log').open('w');handles.append(handle)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                     OMP_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')
            command=[sys.executable,'-u',str(repo/'scripts/run_libero_mining.py'),
                '--cases',str(args.out/'cases.json'),'--out',str(args.out),'--root',str(args.root),
                '--snapshot',str(args.snapshot),'--index',str(index)]
            proc=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(proc);meta['workers'].append(dict(index=index,gpu=gpu,pid=proc.pid,command=command));save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(p.wait):i for i,p in enumerate(processes)}
            for future in as_completed(futures):
                i,code=futures[future],future.result();meta['workers'][i]['exit_code']=code;save()
                if code:
                    for p in processes:
                        if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
                    raise RuntimeError(f'worker {i} failed ({code})')
        meta['phase']='aggregate_verified_discovery';save()
        subprocess.run([sys.executable,str(repo/'scripts/verify_libero_mining.py'),str(args.out),
            '--cases',str(args.out/'cases.json')],cwd=repo,check=True)
        meta.update(status='complete_verified',phase='finished')
    except BaseException as error:
        meta.update(status='failed',error=repr(error));raise
    finally:
        for p in processes:
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
            p.wait()
        for info,p in zip(meta['workers'],processes): info['exit_code']=p.returncode
        for handle in handles: handle.close()
        meta['end']=time.time();save()
    subprocess.run([sys.executable,str(repo/'scripts/archive_libero_run.py'),str(args.out),
                    str(repo/'backups'/args.out.name)],cwd=repo,check=True)
    print(json.dumps(dict(status=meta['status'],run=str(args.out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--pose-smoke',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/home/ami/scratch/LIBERO'))
    p.add_argument('--snapshot',type=Path,default=Path('/home/ami/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/80970322773f81baa2e22fe495d0487b93a05cfa'))
    launch(p.parse_args())
