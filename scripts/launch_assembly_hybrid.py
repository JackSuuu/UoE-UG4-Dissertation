"""Detached A2 actual-state audit and four-arm mixed-controller development."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

from launch_libero_preservation import EXPECTED

ARMS = ('vla', 'feedback80', 'phase_geometry80', 'hybrid80')


def main(source, out, preflight):
    repo = Path(__file__).resolve().parents[1]
    checked = json.loads(preflight.read_text())
    assert checked['status'] == 'passed' and checked['fixture_only']
    for path, digest in checked['sources'].items():
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == digest, path
    gpu_info = {}
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    for line in hardware.splitlines():
        i, uuid, memory = map(str.strip, line.split(',')); gpu_info[int(i)] = [uuid, int(memory)]
    for gpu in (1, 2, 3):
        assert gpu_info[gpu][0] == EXPECTED[gpu] and gpu_info[gpu][1] < 1024
    out.mkdir(parents=True, exist_ok=False)
    for name in ('training.json', 'assembly_dynamics.pt'):
        shutil.copy2(source/name, out/name)
    for path in checked['sources']:
        dest = out/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/path, dest)
    shutil.copy2(preflight, out/'preflight.json')
    shutil.copy2(repo/'exp_record/assembly_a2_protocol_20261016.md', out/'protocol.md')
    cases = [dict(seed=seed, case=case, arm=arm,
        cohort='consumed_regression' if seed < 3100 else 'prospective_development')
        for seed in (3000, 3001, 3100, 3101) for case in range(5) for arm in ARMS]
    meta = dict(status='running', phase='actual_state_audit', start=time.time(), supervisor=os.getpid(),
        git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        source_run=str(source), sources=checked['sources'], hardware=gpu_info,
        planned=cases, jobs=[], completed=[], experiment='A2 development, not final confirmation')
    lock, stopped = threading.Lock(), threading.Event()
    def save():
        temp = out/'launch.tmp'; temp.write_text(json.dumps(meta, indent=2)+'\n'); temp.replace(out/'launch.json')
    def run(stage, gpu, case=None):
        if stopped.is_set(): raise RuntimeError('Peer failure; no new scheduling')
        for path, digest in checked['sources'].items():
            assert hashlib.sha256((repo/path).read_bytes()).hexdigest() == digest, path
        command = [sys.executable, '-u', str(repo/'scripts/run_assembly_hybrid.py'), stage,
            '--source', str(source if stage == 'audit' else out), '--out', str(out)]
        name = stage
        if case:
            command += ['--seed', str(case['seed']), '--case', str(case['case']), '--arm', case['arm']]
            name += f"_s{case['seed']}_c{case['case']}_{case['arm']}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4')
        with (out/(name+'.log')).open('x') as log:
            p = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job = dict(stage=stage, gpu=gpu, pid=p.pid, name=name, case=case)
            with lock:
                meta['jobs'].append(job); save()
            code = p.wait()
            with lock:
                job['exit_code'] = code; save()
            if code:
                stopped.set(); raise RuntimeError(f'{name} exited {code}')
    save()
    try:
        run('audit', 2)
        meta['phase'] = 'paired_live'; save()
        pending = queue.Queue()
        for case in cases: pending.put(case)
        def worker(gpu):
            while not stopped.is_set():
                try: case = pending.get_nowait()
                except queue.Empty: return
                run('live', gpu, case)
                run('verify', gpu, case)
                with lock:
                    meta['completed'].append(case); save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, gpu) for gpu in (1, 2, 3)]
            for f in futures: f.result()
        assert len(meta['completed']) == len(cases)
        summary = []
        for case in cases:
            folder = out/f"live_s{case['seed']}_c{case['case']}_{case['arm']}"
            result = json.loads((folder/'status.json').read_text())
            proof = json.loads((folder/'replay.json').read_text())
            assert result['trace_sha256'] == proof['trace_sha256']
            summary.append(dict(**result, cohort=case['cohort']))
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        meta.update(status='complete_verified', phase='finished')
    except BaseException as error:
        meta.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        meta['end'] = time.time(); save()
    subprocess.run([sys.executable, str(repo/'scripts/archive_assembly_run.py'), str(out),
        str(repo/'backups'/out.name)], cwd=repo, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('/home/ami/scratch/assembly_a1_20261015_a'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    a = p.parse_args()
    main(a.source, a.out, a.preflight)
