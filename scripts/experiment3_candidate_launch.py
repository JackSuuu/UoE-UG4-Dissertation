"""GPU2-only detached bounded experiment3 supervisor and pre-run source freeze."""
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

REPO = Path(__file__).resolve().parents[1]
ARMS = ('nominal', 'simple_feedback', 'verification')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def bounded(command, logfile, seconds, env):
    with logfile.open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return -9


def supervise(out):
    freeze = json.loads((out/'freeze.json').read_text())
    state = dict(status='running', pid=os.getpid(), started=time.time(), jobs=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', MUJOCO_EGL_DEVICE_ID='2', MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', TOKENIZERS_PARALLELISM='false')
    try:
        for rel, value in freeze['sources'].items(): assert sha(out/'source'/rel) == value
        for path, value in freeze['external_sources'].items(): assert sha(Path(path)) == value
        runner = out/'source/scripts/experiment3_candidate_run.py'
        for arm in ARMS:
            job = dict(arm=arm, started=time.time())
            state['jobs'].append(job)
            save(out/'launch.json', state)
            cmd = [sys.executable, '-u', str(runner), '--out', str(out/arm), '--arm', arm, '--snapshot', freeze['snapshot']]
            job['command'] = cmd
            job['returncode'] = bounded(cmd, out/f'{arm}.log', 1500, env)
            if job['returncode'] == 0:
                job['replay_returncode'] = bounded([sys.executable, '-u', str(runner), '--out', str(out/f'replay_{arm}'),
                    '--replay', str(out/arm)], out/f'replay_{arm}.log', 240, env)
            job['ended'] = time.time()
            save(out/'launch.json', state)
        if all(j.get('returncode') == j.get('replay_returncode') == 0 for j in state['jobs']):
            state['review_returncode'] = bounded([sys.executable, str(out/'source/scripts/experiment3_candidate_review.py'), str(out)],
                out/'review.log', 300, env)
        state['status'] = 'completed' if state.get('review_returncode') == 0 else 'completed_with_issues'
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        state['ended'] = time.time()
        save(out/'launch.json', state)


def launch(out):
    from run_fridge_r1 import SNAPSHOT, GPUS
    out.mkdir(parents=True, exist_ok=False)
    found = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    assert any(line.strip() == f'2, {GPUS[2]}' for line in found.splitlines())
    files = [p.relative_to(REPO) for p in (REPO/'src/adapters').glob('*.py')]
    files += [p.relative_to(REPO) for p in (REPO/'scripts').glob('experiment3_candidate_*.py')]
    files += [Path('scripts')/name for name in ('test_experiment3_candidate.py', 'diagnose_vla_failure.py',
        'diagnose_fridge_skills.py', 'probe_fridge_task.py')]
    files += [Path('exp_record/agent_notes/experiment3_candidate_20261019/protocol.md')]
    sources = {}
    for relative in files:
        dest = out/'source'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO/relative).read_bytes())
        sources[str(relative)] = sha(dest)
    external = [Path('/home/ami/scratch/LIBERO/libero/libero/envs/bddl_base_domain.py'),
        Path('/home/ami/scratch/LIBERO/libero/libero/envs/env_wrapper.py'),
        Path('/home/ami/miniconda3/lib/python3.13/site-packages/robosuite/environments/base.py')]
    sys.path.insert(0, str(REPO/'src'))
    from adapters.tf5_compat import snapshot_dir
    model_files = list(Path(SNAPSHOT).glob('*.json'))+list(Path(SNAPSHOT).glob('*.safetensors'))
    code_files = list(Path(snapshot_dir()).glob('*.py'))
    freeze = dict(frozen_at=time.time(), gpu=2, gpu_uuid=GPUS[2], sources=sources,
        snapshot=SNAPSHOT, model_files={str(p):sha(p) for p in model_files},
        model_code={str(p):sha(p) for p in code_files}, external_sources={str(p):sha(p) for p in external},
        protocol='exp_record/agent_notes/experiment3_candidate_20261019/protocol.md',
        no_outcome_seen_before_freeze=True, planned=list(ARMS))
    save(out/'freeze.json', freeze)
    with (out/'supervisor.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(out/'source/scripts/experiment3_candidate_launch.py'),
            str(out), '--supervise'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(out/'detached.json', dict(pid=proc.pid, process_group=proc.pid, start_new_session=True,
            gpu=2, outer_wait_is_notification_only=True, freeze_sha256=sha(out/'freeze.json')))
        print(json.dumps(dict(detached_pid=proc.pid, out=str(out))), flush=True)
        return proc.wait()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('--supervise', action='store_true')
    a = p.parse_args()
    if a.supervise: supervise(a.out.resolve())
    else: raise SystemExit(launch(a.out.resolve()))
