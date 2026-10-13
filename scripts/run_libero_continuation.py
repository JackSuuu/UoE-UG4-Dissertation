"""Same frozen H1 actor, continue past native success to released placement."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import imageio.v2 as imageio
import numpy as np
import torch
from adapters.libero_runtime import TASK, make_env
from adapters.libero_openvla import LiberoOpenVLA
from adapters.libero_completion import ContactRecorder, CompletionMonitor
from run_libero_baseline import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--snapshot', required=True)
    p.add_argument('--index', type=int, required=True, choices=[0, 1, 2])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'scripts/run_libero_baseline.py'] + [root / 'src/adapters' / f for f in
               ['libero_runtime.py', 'libero_openvla.py', 'libero_completion.py', 'tf5_compat.py']]
    manifest = dict(status='started', args=vars(a), task=TASK, start=time.time(),
                    git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    libero_git=subprocess.check_output(['git', '-C', a.root, 'rev-parse', 'HEAD'], text=True).strip(),
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    versions={k: importlib.metadata.version(k) for k in
                              ['torch', 'transformers', 'numpy', 'mujoco', 'robosuite']},
                    source_sha256={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
                    max_policy_steps=520, dwell_steps=20, CVR=None, safe_success=None)
    write_json(out / 'manifest.json', manifest)
    try:
        actor = LiberoOpenVLA(a.snapshot)
        env, obs = make_env(a.root, a.index)
        try:
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            monitor = CompletionMonitor(env)
            recorder = ContactRecorder(env.sim)
            manifest.update(actor=actor.build, bounds=monitor.bounds.tolist(), floor=monitor.floor)
            write_json(out / 'manifest.json', manifest)
            (out / 'model.xml').write_text(env.sim.model.get_xml())
            imageio.imwrite(out / 'initial.png', obs['agentview_image'][::-1, ::-1])
            states = []
            with imageio.get_writer(out / 'rollout.mp4', fps=20) as video, \
                    (out / 'steps.jsonl').open('w') as log, \
                    (out / 'substeps.jsonl').open('w') as physics:
                for step in range(520):
                    states.append(env.get_sim_state().copy())
                    video.append_data(obs['agentview_image'][::-1, ::-1])
                    action, decode, image = actor.act(obs['agentview_image'], env.language_instruction)
                    if step == 0:
                        image.save(out / 'policy_input.png')
                    recorder.reset()
                    obs, reward, native_done, _ = env.step(action.tolist())
                    assert len(recorder.samples) == 25, 'Unexpected physics/control cadence'
                    measured = monitor.measure(step, recorder.samples)
                    log.write(json.dumps(dict(step=step, action=action.tolist(), **decode,
                        native_done=bool(native_done), **measured)) + '\n')
                    physics.write(json.dumps(dict(step=step, substeps=recorder.samples)) + '\n')
                    log.flush()
                    physics.flush()
                    if step % 20 == 0 or measured['complete']:
                        print(json.dumps(dict(step=step, native=measured['native_success'],
                             conditions=measured['conditions'], streak=monitor.streak)), flush=True)
                    # LIBERO's returned done is its native predicate, not time-limit
                    # termination. Robosuite's internal horizon is1000 (>530total).
                    if measured['complete']:
                        break
                states.append(env.get_sim_state().copy())
                video.append_data(obs['agentview_image'][::-1, ::-1])
                imageio.imwrite(out / 'final.png', obs['agentview_image'][::-1, ::-1])
                np.savez_compressed(out / 'states.npz', states=np.asarray(states))
            write_json(out / 'result.json', dict(index=a.index, policy_steps=step+1,
                native_ever=monitor.first_native is not None, first_native_step=monitor.first_native,
                released_placement=monitor.first_complete is not None,
                first_complete_step=monitor.first_complete, max_streak=monitor.max_streak,
                final=measured, CVR=None, safe_success=None))
            recorder.close()
        finally:
            env.close()
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'failed'
        manifest['error'] = traceback.format_exc()
        raise
    finally:
        manifest['end'] = time.time()
        write_json(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
