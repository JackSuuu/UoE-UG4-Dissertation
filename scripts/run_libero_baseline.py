"""Bounded H1 feasibility pilot. Stores every episode, not a verifier evaluation."""
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


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/ami/scratch/LIBERO')
    p.add_argument('--snapshot', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=520)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    manifest = dict(task=TASK, args=vars(a), status='started', start=time.time(),
                    git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    libero_git=subprocess.check_output(['git', '-C', a.root, 'rev-parse', 'HEAD'], text=True).strip(),
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), protocol='native H1 feasibility; no safety threshold',
                    versions={k: importlib.metadata.version(k) for k in
                              ['torch', 'transformers', 'numpy', 'mujoco', 'robosuite', 'tensorflow-cpu']})
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'src/adapters/libero_runtime.py',
               root / 'src/adapters/libero_openvla.py', root / 'src/adapters/tf5_compat.py']
    manifest['source_sha256'] = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    write_json(out / 'manifest.json', manifest)
    results = []
    try:
        actor = LiberoOpenVLA(a.snapshot)
        manifest['actor'] = actor.build
        write_json(out / 'manifest.json', manifest)
        print('MODEL READY', flush=True)
        for index in a.indices:
            episode = out / f'episode_{index:03d}'
            episode.mkdir()
            env, obs = make_env(a.root, index)
            video = imageio.get_writer(episode / 'rollout.mp4', fps=20)
            rows = (episode / 'steps.jsonl').open('w')
            states = []
            start = time.time()
            try:
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                assert not env.check_success(), 'Initial task already complete'
                imageio.imwrite(episode / 'initial.png', obs['agentview_image'][::-1, ::-1])
                (episode / 'model.xml').write_text(env.sim.model.get_xml())
                success = False
                for step in range(a.steps):
                    video.append_data(obs['agentview_image'][::-1, ::-1])
                    states.append(env.get_sim_state().copy())
                    action, decode, image = actor.act(obs['agentview_image'], env.language_instruction)
                    if step == 0:
                        image.save(episode / 'policy_input.png')
                    obs, reward, done, info = env.step(action.tolist())
                    success = bool(env.check_success())
                    # Endpoint contact diagnostics only; NOT a substep peak/safety certificate.
                    import mujoco
                    contacts = []
                    for ci in range(env.sim.data.ncon):
                        c = env.sim.data.contact[ci]
                        force = np.zeros(6)
                        mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, ci, force)
                        contacts.append(dict(geom1=env.sim.model.geom_id2name(c.geom1),
                                             geom2=env.sim.model.geom_id2name(c.geom2),
                                             wrench=force.tolist(), distance=float(c.dist)))
                    rows.write(json.dumps(dict(step=step, action=action.tolist(), **decode,
                                                success=success, done=bool(done), reward=float(reward),
                                                book_pos=obs['black_book_1_pos'].tolist(),
                                                book_quat=obs['black_book_1_quat'].tolist(),
                                                eef_pos=obs['robot0_eef_pos'].tolist(),
                                                contacts_endpoint=contacts)) + '\n')
                    rows.flush()
                    if step % 20 == 0 or success:
                        print(f'episode={index} step={step} success={success} elapsed={time.time()-start:.1f}s', flush=True)
                    if success or done:
                        break
                video.append_data(obs['agentview_image'][::-1, ::-1])
                states.append(env.get_sim_state().copy())
                np.savez_compressed(episode / 'states.npz', states=np.asarray(states))
                result = dict(index=index, native_success=success, policy_steps=step + 1,
                              seconds=time.time()-start, stable_precision_success=None,
                              safety_violation=None)
                write_json(episode / 'result.json', result)
                results.append(result)
                write_json(out / 'summary.json', dict(episodes=results,
                    native_SR=float(np.mean([r['native_success'] for r in results])),
                    CVR=None, safe_success=None, feasibility_only=True))
            finally:
                rows.close()
                video.close()
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
