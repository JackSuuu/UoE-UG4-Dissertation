"""Detached study supervisor; runs 3 GPU workers then verifies/renders all arms.

Launch this script itself with setsid nohup and detached stdin. Each worker also
has its own session and persistent log, so SSH disconnection cannot stop it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


EXPECTED = {
    1: 'GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0',
    2: 'GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8',
    3: 'GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b',
}


def launch(args):
    smoke = json.loads(args.smoke.read_text())
    assert smoke['status'] == 'passed' and smoke['branches'] == 6
    assert smoke['full_integration_controller_image_identity']
    assert smoke['live_nonmutation'] and smoke['selected_first_prediction_equal_actual']
    assert smoke.get('consecutive_decisions', 0) >= 3, 'Require repeated-render branch regression'
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
    available = {}
    for line in hardware.splitlines():
        index, uuid, memory = map(str.strip, line.split(','))
        available[int(index)] = (uuid, int(memory))
    for index, uuid in EXPECTED.items():
        assert available[index][0] == uuid and available[index][1] < 1024, available
    repo = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='started', start=time.time(), supervisor=os.getpid(),
        git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        smoke=str(args.smoke), GPUs=available, workers=[])
    path = args.out/'launch.json'
    def save():
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    processes, handles = [], []
    try:
        for gpu, indices in [(1, [0,3]), (2, [1,4]), (3, [2,5])]:
            log = (args.out/f'gpu_{gpu}.log').open('w')
            handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                       OMP_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
            command = [sys.executable, '-u', str(repo/'scripts/run_libero_preservation.py'),
                       '--indices', *map(str, indices), '--snapshot', str(args.snapshot),
                       '--root', str(args.root), '--out', str(args.out)]
            worker = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(worker)
            manifest['workers'].append(dict(gpu=gpu, roots=indices, pid=worker.pid, command=command))
            save()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(p.wait): i for i, p in enumerate(processes)}
            for future in as_completed(futures):
                i, code = futures[future], future.result()
                manifest['workers'][i]['exit_code'] = code
                save()
                if code:
                    for p in processes:
                        if p.poll() is None:
                            os.killpg(p.pid, signal.SIGTERM)
                    raise RuntimeError(f'Worker {i} failed: {code}; inspect gpu logs')
        subprocess.run([sys.executable, str(repo/'scripts/summarize_libero_preservation.py'),
                        str(args.out)], cwd=repo, check=True)
        manifest['status'] = 'complete_verified'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait()
        for info, process in zip(manifest['workers'], processes):
            info['exit_code'] = process.returncode
        for handle in handles:
            handle.close()
        manifest['end'] = time.time()
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/home/ami/scratch/LIBERO'))
    p.add_argument('--snapshot', type=Path, default=Path('/home/ami/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/80970322773f81baa2e22fe495d0487b93a05cfa'))
    launch(p.parse_args())
