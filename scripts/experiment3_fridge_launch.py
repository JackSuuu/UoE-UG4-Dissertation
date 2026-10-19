"""Detached frozen-source closed-start fridge v1 exploration on GPU2 only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import json
from experiment3_candidate_launch import bounded, sha, save

REPO = Path(__file__).resolve().parents[1]
ARMS = ('nominal', 'simple_feedback', 'verification')
CAMERA = Path('/home/ami/scratch/fridge_observation_preflight_20261019/calibration/camera_config.json')


def supervise(out):
    freeze = json.loads((out/'freeze.json').read_text())
    state = dict(status='running', pid=os.getpid(), started=time.time(), jobs=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', MUJOCO_EGL_DEVICE_ID='2', MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', TOKENIZERS_PARALLELISM='false')
    try:
        for rel,h in freeze['sources'].items(): assert sha(out/'source'/rel)==h
        assert sha(out/'camera_config.json')==freeze['camera_sha256']
        for path,h in freeze['external_sources'].items(): assert sha(Path(path))==h
        for arm in ARMS:
            job = dict(arm=arm, started=time.time())
            state['jobs'].append(job)
            save(out/'launch.json', state)
            runner = out/'source/scripts/experiment3_fridge_run.py'
            cmd = [sys.executable, '-u', str(runner), '--out', str(out/arm), '--arm', arm,
                '--snapshot', freeze['snapshot'], '--camera-config', str(out/'camera_config.json')]
            job['command'] = cmd
            job['returncode'] = bounded(cmd, out/f'{arm}.log', 1500, env)
            if job['returncode']==0:
                job['replay_returncode'] = bounded([sys.executable, '-u', str(runner), '--out', str(out/f'replay_{arm}'),
                    '--replay', str(out/arm)], out/f'replay_{arm}.log', 240, env)
            job['ended'] = time.time()
            save(out/'launch.json', state)
        if all(j.get('returncode')==j.get('replay_returncode')==0 for j in state['jobs']):
            state['review_returncode'] = bounded([sys.executable, str(out/'source/scripts/experiment3_fridge_review.py'), str(out)],
                out/'review.log', 300, env)
        state['status'] = 'completed' if state.get('review_returncode')==0 else 'completed_with_issues'
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        state['ended'] = time.time()
        save(out/'launch.json', state)


def launch(out):
    from run_fridge_r1 import SNAPSHOT, GPUS
    out.mkdir(parents=True, exist_ok=False)
    found=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],text=True)
    assert any(line.strip()==f'2, {GPUS[2]}' for line in found.splitlines())
    files=[p.relative_to(REPO) for p in (REPO/'src/adapters').glob('*.py')]
    files += [p.relative_to(REPO) for p in (REPO/'scripts').glob('experiment3_fridge_*.py')]
    files += [Path('scripts')/name for name in ('test_experiment3_fridge.py',
        'experiment3_candidate_gate.py', 'experiment3_candidate_run.py', 'experiment3_candidate_worker.py',
        'experiment3_candidate_launch.py', 'diagnose_vla_failure.py', 'diagnose_fridge_skills.py',
        'probe_fridge_task.py', 'fridge_observation_experiment.py', 'run_fridge_r1.py')]
    files += [Path('configs/tasks/fridge_retrieval_dev.bddl'),
        Path('exp_record/experiment3_fridge_candidate_20261019/protocol.md')]
    sources={}
    for rel in files:
        dest=out/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((REPO/rel).read_bytes())
        sources[str(rel)]=sha(dest)
    (out/'camera_config.json').write_bytes(CAMERA.read_bytes())
    sys.path.insert(0,str(REPO/'src'))
    from adapters.tf5_compat import snapshot_dir
    models=list(Path(SNAPSHOT).glob('*.json'))+list(Path(SNAPSHOT).glob('*.safetensors'))
    external=[Path('/home/ami/scratch/LIBERO/libero/libero/envs/bddl_base_domain.py'),
        Path('/home/ami/scratch/LIBERO/libero/libero/envs/env_wrapper.py'),
        Path('/home/ami/miniconda3/lib/python3.13/site-packages/robosuite/environments/base.py'),
        Path('/home/ami/scratch/LIBERO/libero/libero/assets/articulated_objects/short_fridge.xml')]
    save(out/'freeze.json',dict(frozen_at=time.time(),sources=sources,snapshot=SNAPSHOT,
        model_files={str(p):sha(p) for p in models},
        model_code={str(p):sha(p) for p in Path(snapshot_dir()).glob('*.py')},
        external_sources={str(p):sha(p) for p in external}, camera_sha256=sha(out/'camera_config.json'),
        camera_source=str(CAMERA), gpu=2,gpu_uuid=GPUS[2],planned=list(ARMS),
        condition='closed_start_seed0', scope='integration_exploration_not_formal_efficacy',
        no_outcome_seen_before_freeze=True))
    with (out/'supervisor.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(out/'source/scripts/experiment3_fridge_launch.py'),
            str(out),'--supervise'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'detached.json',dict(pid=proc.pid,start_new_session=True,gpu=2,freeze_sha256=sha(out/'freeze.json')))
        print(json.dumps(dict(detached_pid=proc.pid,out=str(out))),flush=True)
        return proc.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--supervise',action='store_true')
    a=p.parse_args()
    if a.supervise: supervise(a.out.resolve())
    else: raise SystemExit(launch(a.out.resolve()))
