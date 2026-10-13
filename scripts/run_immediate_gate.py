"""Durable coordinator: split collection -> support -> paired training -> evaluation.

Launch this process with setsid nohup and /dev/null stdin. GPU workers are normal
child processes; wait() observes completion without a filesystem polling loop.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


GPUS = ('GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0',
        'GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8',
        'GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = repo / 'src/results/push_torch' / cli.run
    source = repo / 'src/results/push_torch/sequence_risk_20261012_a'
    base = [sys.executable, '-u', 'experiments/validate_immediate_gate.py',
            '--source_run', str(source), '--output', str(output)]
    if cli.smoke:
        base += ['--smoke', '--device', 'cpu']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    subprocess.run(base + ['--stage', 'init'], cwd=repo / 'src', env=env, check=True)
    launches = []

    def save(state):
        tmp = output / 'coordinator.json.tmp'
        tmp.write_text(json.dumps({'stage': state, 'smoke': cli.smoke, 'launches': launches}, indent=2))
        os.replace(tmp, output / 'coordinator.json')

    def parallel(stage, jobs):
        processes = []
        save(stage)
        for gpu, extra in jobs:
            log = output / f'{stage}_{len(launches):02d}.log'
            with log.open('xb') as f:
                proc = subprocess.Popen(base + ['--stage', stage] + extra, cwd=repo / 'src',
                    env=dict(env, CUDA_VISIBLE_DEVICES='' if cli.smoke else GPUS[gpu]),
                    stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
            launches.append({'stage': stage, 'pid': proc.pid, 'gpu': None if cli.smoke else GPUS[gpu],
                             'args': extra, 'log': str(log)})
            processes.append((proc, launches[-1]))
            save(stage)
        failed = False
        for proc, item in processes:
            item['exit_code'] = proc.wait()
            failed |= item['exit_code'] != 0
            save(stage)
        if failed:
            save(stage + '_failed')
            raise RuntimeError(f'{stage} failed; inspect worker logs')

    parallel('collect', [(i, ['--split', s]) for i, s in enumerate(('train', 'dev', 'calibration'))])
    save('screen')
    result = subprocess.run(base + ['--stage', 'screen'], cwd=repo / 'src', env=env)
    if result.returncode == 2:
        save('blocked_insufficient_support')
        print('[R5 coordinator] support failed; no training or evaluation launched', flush=True)
        return
    result.check_returncode()
    parallel('train', [(i, ['--seed', str(i)]) for i in range(1 if cli.smoke else 3)])
    parallel('evaluate', [(i, ['--shard', str(i)]) for i in range(1 if cli.smoke else 3)])
    save('aggregate')
    subprocess.run(base + ['--stage', 'aggregate'], cwd=repo / 'src', env=env, check=True)
    save('complete')
    print('[R5 coordinator] complete', output, flush=True)


if __name__ == '__main__':
    main()
