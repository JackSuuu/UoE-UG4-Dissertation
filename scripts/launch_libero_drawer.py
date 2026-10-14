"""Detached D1 supervisor: targeted capability gate, matched arms, exact review."""
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
    repo = Path(__file__).resolve().parents[1]
    smoke = json.loads(args.smoke.read_text())
    assert smoke['status'] == 'passed' and smoke['fixture_only']
    assert smoke['source_sha256']['libero_drawer.py'] == hashlib.sha256(
        (repo/'src/adapters/libero_drawer.py').read_bytes()).hexdigest()
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
    available = {}
    for line in hardware.splitlines():
        i, uuid, mem = map(str.strip, line.split(','))
        available[int(i)] = [uuid, int(mem)]
    for gpu, uuid in EXPECTED.items():
        assert available[gpu][0] == uuid and available[gpu][1] < 1024, available
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.smoke, args.out/'preflight_smoke.json')
    manifest = dict(status='started', phase='baseline_target_coverage', start=time.time(),
        supervisor=os.getpid(), git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        GPUs=available, workers=[], roots=[0,1,2], target='release-clearance-handoff-close',
        protocol_sha256=hashlib.sha256((repo/'exp_record/libero_drawer_target_protocol_20261014.md').read_bytes()).hexdigest())
    def save():
        tmp = args.out/'launch.json.tmp'
        tmp.write_text(json.dumps(manifest, indent=2)+'\n')
        tmp.replace(args.out/'launch.json')

    def batch(arms, label):
        processes, handles, infos = [], [], []
        try:
            for gpu, index in [(1,0),(2,1),(3,2)]:
                log = (args.out/f'{label}_gpu_{gpu}.log').open('w')
                handles.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                           OMP_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
                command = [sys.executable,'-u',str(repo/'scripts/run_libero_drawer.py'),
                    '--indices',str(index),'--arms',*arms,'--snapshot',str(args.snapshot),
                    '--root',str(args.root),'--out',str(args.out)]
                proc = subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                info = dict(stage=label, gpu=gpu, index=index, pid=proc.pid, command=command)
                processes.append(proc)
                infos.append(info)
                manifest['workers'].append(info)
                save()
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(p.wait):i for i,p in enumerate(processes)}
                for f in as_completed(futures):
                    i, code = futures[f], f.result()
                    infos[i]['exit_code'] = code
                    save()
                    if code:
                        # Kill first, before exiting the executor (which waits for workers).
                        for p in processes:
                            if p.poll() is None:
                                os.killpg(p.pid, signal.SIGTERM)
                        raise RuntimeError(f'{label} worker {i} failed: {code}')
        finally:
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                p.wait()
            for info, p in zip(infos, processes):
                info['exit_code'] = p.returncode
            for log in handles:
                log.close()
            save()

    save()
    try:
        batch(['vla'], 'baseline')
        outcomes = [json.loads((args.out/f'root_{i}/vla/result.json').read_text()) for i in range(3)]
        coverage = [r['trigger'] is not None for r in outcomes]
        target_failure = [r['trigger'] is not None and not r['complete'] for r in outcomes]
        manifest['baseline_target_coverage'] = coverage
        manifest['baseline_post_event_failure'] = target_failure
        print(json.dumps(dict(baseline_target_coverage=coverage, baseline_post_event_failure=target_failure)), flush=True)
        arms = ['vla']
        if any(target_failure):
            manifest['phase'] = 'matched_interventions'
            save()
            batch(['gate_hold','timed_repair','phase_repair'], 'intervention')
            arms += ['gate_hold','timed_repair','phase_repair']
        else:
            manifest['stop_reason'] = 'target_unreachable' if not any(coverage) else 'no_observed_post_event_headroom'
        manifest['phase'] = 'verification_and_observable_review'
        save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', MUJOCO_EGL_DEVICE_ID='1')
        subprocess.run([sys.executable,'-u',str(repo/'scripts/summarize_libero_drawer.py'),str(args.out),
                        '--root',str(args.root),'--arms',*arms],cwd=repo,env=env,check=True)
        manifest['status'] = 'complete_verified' if any(target_failure) else manifest['stop_reason'] + '_verified'
        manifest['phase'] = 'finished'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['end'] = time.time()
        save()
    # Archive only after all files have stopped changing and final launch state is saved.
    subprocess.run([sys.executable,str(repo/'scripts/archive_libero_run.py'),str(args.out),
                    str(repo/'backups'/args.out.name)],cwd=repo,check=True)
    print(json.dumps(dict(status=manifest['status'], archive=str(repo/'backups'/args.out.name))),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/home/ami/scratch/LIBERO'))
    p.add_argument('--snapshot', type=Path, default=Path('/home/ami/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/80970322773f81baa2e22fe495d0487b93a05cfa'))
    launch(p.parse_args())
