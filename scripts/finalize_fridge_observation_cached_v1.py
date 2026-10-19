"""Add four bounded cached replays; preserve original cohort and timeout evidence."""
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


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def validate(root, name, verifier_hash):
    run = root/name
    status = json.loads((run/'status.json').read_text())
    report_path = root/f'cached_replay_{name}/verification.json'
    report = json.loads(report_path.read_text())
    assert status['status'] == 'completed' and status['steps_executed'] == 600
    for file, expected in status['sources'].items():
        assert digest(run/'source'/file) == expected
    assert report['status'] == 'exact_replay_passed' and report['steps'] == 600
    assert report['max_state_gap'] == 0 and report['mismatches'] == []
    assert report['source'] == str(run) and report['verifier_sha256'] == verifier_hash
    return dict(path=str(report_path), sha256=digest(report_path), report=report)


def main(root, out):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['MUJOCO_EGL_DEVICE_ID'] == '1'
    repo = Path(__file__).resolve().parents[1]
    verifier = repo/'scripts/verify_fridge_observation_cached.py'
    committed = subprocess.check_output(['git', 'show', 'e4944c7:scripts/verify_fridge_observation_cached.py'], cwd=repo)
    assert verifier.read_bytes() == committed
    vh = digest(verifier)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader'], text=True)
    assert '1, GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0,' in gpu
    assert not out.exists()
    original = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    out.mkdir(parents=True)
    source = out/'source'
    source.mkdir()
    frozen_verifier = source/verifier.name
    frozen_verifier.write_bytes(committed)
    (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(out/'historical_files_before.json', original)
    launch = json.loads((root/'launch.json').read_text())
    assert launch['status'] == 'complete_with_issues'
    assert len(launch['jobs']) == 6
    assert all(j['status'] == 'job_error' and j['returncode'] == 0 and
               'timed out after 300 seconds' in j['error'] for j in launch['jobs'])
    report = dict(status='running', started=time.time(), root=str(root), gpu=1,
        verifier_commit='e4944c7', verifier_sha256=vh, original_launch_sha256=digest(root/'launch.json'),
        original_status=launch['status'], original_timeout_count=6, original_timeout_seconds_each=300,
        actor_queries=0, actor_inference_processes=0, reused_reports={}, fresh_replays=[],
        gpu_before=gpu, finalizer_sha256=digest(Path(__file__)))
    for camera in ('robot_side', 'framed_side'):
        name = f'seed0_{camera}'
        report['reused_reports'][name] = validate(root, name, vh)
    save(out/'replay_finalization.json', report)
    try:
        for seed in (1, 2):
            for camera in ('robot_side', 'framed_side'):
                name = f'seed{seed}_{camera}'
                target = root/f'cached_replay_{name}'
                assert not target.exists()
                command = [sys.executable, '-u', str(frozen_verifier), str(root/name), str(target), '--gpu', '1']
                job = dict(name=name, command=command, started=time.time(), timeout_s=300)
                report['fresh_replays'].append(job)
                save(out/'replay_finalization.json', report)
                with (out/f'{name}.cached_replay.log').open('x') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    job['pid'] = child.pid
                    try:
                        job['returncode'] = child.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                        raise
                job['ended'] = time.time()
                assert job['returncode'] == 0, name
                job['verification'] = validate(root, name, vh)
                save(out/'replay_finalization.json', report)
        report['status'] = 'six_cached_exact_replays_validated'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['historical_file_changes'] = [name for name, expected in original.items()
            if not (root/name).is_file() or digest(root/name) != expected]
        report['ended'] = time.time()
        report['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
            '--format=csv,noheader'], text=True)
        save(out/'replay_finalization.json', report)
        assert not report['historical_file_changes']
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    main(a.root.resolve(), a.out.resolve())
