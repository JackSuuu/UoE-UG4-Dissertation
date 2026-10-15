"""Detached budget probe supervisor with independent replay of each trajectory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main(root, out):
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    meta = dict(status='running', start=time.time(), supervisor=os.getpid(),
        protocol='failure-only A1 budget80, same horizon160; exact common-prefix check', jobs=[])
    lock = threading.Lock()
    def save():
        p = out/'launch.tmp'
        p.write_text(json.dumps(meta, indent=2)+'\n'); p.replace(out/'launch.json')
    def lane(gpu, cases):
        for seed, case in cases:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS='4')
            for stage in ('live', 'verify'):
                if stage == 'live':
                    command = [sys.executable, '-u', str(repo/'scripts/probe_assembly_budget.py'), '--root', str(root),
                        '--out', str(out), '--seed', str(seed), '--case', str(case)]
                else:
                    command = [sys.executable, '-u', str(repo/'scripts/run_assembly_development.py'), 'verify',
                        '--out', str(out), '--seed', str(seed), '--case', str(case), '--arm', 'framework']
                with (out/f'{stage}_s{seed}_c{case}.log').open('x') as log:
                    p = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    job = dict(stage=stage, seed=seed, case=case, gpu=gpu, pid=p.pid)
                    with lock:
                        meta['jobs'].append(job); save()
                    code = p.wait()
                    with lock:
                        job['exit_code'] = code; save()
                    if code: raise RuntimeError(f'{stage} {seed}/{case} exited {code}')
    save()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lane, g, cases) for g, cases in
                [(1, [(3000, 1), (3001, 3)]), (2, [(3000, 3)]), (3, [(3001, 2)])]]
            for f in futures: f.result()
        meta['status'] = 'complete_verified'
    except BaseException as error:
        meta.update(status='failed', error=repr(error))
        raise
    finally:
        meta['end'] = time.time(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.out)
