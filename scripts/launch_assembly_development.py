"""Detached A1 supervisor; subprocess-isolated episodes and exact replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from launch_libero_preservation import EXPECTED


def main(out, preflight):
    repo = Path(__file__).resolve().parents[1]
    checked = json.loads(preflight.read_text())
    assert checked['status'] == 'passed'
    for rel, digest in checked['sources'].items():
        assert hashlib.sha256((repo/rel).read_bytes()).hexdigest() == digest, rel
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    gpu_info = {int(parts[0]): (parts[1].strip(), int(parts[2]))
        for parts in (line.split(',') for line in hardware.splitlines())}
    for gpu in (1, 2, 3):
        assert gpu_info[gpu][0] == EXPECTED[gpu] and gpu_info[gpu][1] < 1024
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preflight, out/'preflight.json')
    shutil.copy2(repo/'exp_record/assembly_a1_protocol_20261015.md', out/'protocol.md')
    for rel in checked['sources']:
        path = out/'source'/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/rel, path)
    meta = dict(status='running', phase='collect', start=time.time(), supervisor=os.getpid(),
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        sources=checked['sources'], hardware=gpu_info, jobs=[], completed=[], development_only=True,
        planned=[dict(seed=s, case=c, arm=a) for s in (3000, 3001) for c in range(5)
                 for a in ('vla', 'framework', 'feedback')])
    lock = threading.Lock()
    stopped = threading.Event()
    def save():
        temp = out/'launch.tmp'
        temp.write_text(json.dumps(meta, indent=2)+'\n')
        temp.replace(out/'launch.json')
    def run(stage, gpu, extra=(), name=None):
        if stopped.is_set():
            raise RuntimeError('Other worker failed; no further episode launch')
        for rel, digest in checked['sources'].items():
            assert hashlib.sha256((repo/rel).read_bytes()).hexdigest() == digest, rel
        command = [sys.executable, '-u', str(repo/'scripts/run_assembly_development.py'), stage, '--out', str(out), *extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4')
        with (out/f'{name or stage}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job = dict(stage=stage, gpu=gpu, name=name, pid=process.pid, command=command)
            with lock:
                meta['jobs'].append(job); save()
            code = process.wait()
            with lock:
                job['exit_code'] = code; save()
            if code:
                stopped.set()
                raise RuntimeError(f'{name or stage} exited {code}; preserve incomplete artifacts')
    save()
    try:
        run('collect', 2)
        meta['phase'] = 'train'; save()
        run('train', 2)
        meta['phase'] = 'paired_live'; save()
        def worker(gpu, arm):
            for seed in (3000, 3001):
                for case in range(5):
                    extra = ['--seed', str(seed), '--case', str(case), '--arm', arm]
                    name = f's{seed}_c{case}_{arm}'
                    run('live', gpu, extra, 'live_'+name)
                    run('verify', gpu, extra, 'verify_'+name)
                    with lock:
                        meta['completed'].append(dict(seed=seed, case=case, arm=arm)); save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, gpu, arm) for gpu, arm in
                       [(1, 'vla'), (2, 'framework'), (3, 'feedback')]]
            for future in as_completed(futures):
                future.result()
        summary = []
        for entry in meta['planned']:
            dest = out/f"live_s{entry['seed']}_c{entry['case']}_{entry['arm']}"
            result = json.loads((dest/'status.json').read_text())
            proof = json.loads((dest/'replay.json').read_text())
            assert proof['trace_sha256'] == result['trace_sha256']
            summary.append(result)
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        meta.update(status='complete_verified', phase='finished')
    except BaseException as error:
        meta.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        meta['end'] = time.time(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    a = p.parse_args()
    main(a.out, a.preflight)
