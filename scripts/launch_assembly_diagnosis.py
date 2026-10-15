"""Detached read-only counterfactual diagnostics, three independent process lanes."""
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
    repo = Path(__file__).resolve().parents[1]
    meta = dict(status='running', start=time.time(), supervisor=os.getpid(), jobs=[])
    lock = threading.Lock()
    def save():
        temp = out/'launch.tmp'
        temp.write_text(json.dumps(meta, indent=2)+'\n')
        temp.replace(out/'launch.json')
    save()
    def lane(gpu, cases):
        for seed, case in cases:
            command = [sys.executable, '-u', str(repo/'scripts/audit_assembly_failures.py'), 'branches',
                '--root', str(root), '--out', str(out), '--seed', str(seed), '--case', str(case)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS='4')
            with (out/f'branches_s{seed}_c{case}.log').open('x') as log:
                process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job = dict(seed=seed, case=case, gpu=gpu, pid=process.pid)
                with lock:
                    meta['jobs'].append(job); save()
                code = process.wait()
                with lock:
                    job['exit_code'] = code; save()
                if code:
                    raise RuntimeError(f'{seed}/{case} failed with {code}')
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lane, gpu, cases) for gpu, cases in
                [(1, [(3000, 1), (3001, 3)]), (2, [(3000, 3)]), (3, [(3001, 2)])]]
            for f in futures:
                f.result()
        meta['status'] = 'completed'
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
