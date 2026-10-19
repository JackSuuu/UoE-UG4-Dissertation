"""U0 backend qualification only: actual deformable particles, not folding skill.

No VLA, learned model, target demonstration, grasp weld or particle teleportation.
Runs a tiny Genesis cloth drape over an asymmetric rigid obstacle and exports
physical positions for independent numerical review. This cannot establish robot
task success, cutting, peeling, washing or OOD policy generalization.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='started', scope='backend_physics_qualification_only', actor=False,
        hybrid=False, training=False, garment_folding_success=None,
        device=os.environ.get('CUDA_VISIBLE_DEVICES'), start=time.time(),
        installed={x: importlib.util.find_spec(x) is not None for x in
            ('genesis', 'omnigibson', 'isaacsim', 'mani_skill', 'softgym', 'pyflex', 'mujoco')},
        versions={x: importlib.metadata.version(x) for x in ('genesis-world', 'mujoco', 'torch')},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(out/'status.json', status)
    try:
        import genesis as gs
        import trimesh
        gs.init(backend=gs.gpu, seed=0, logging_level='warning')
        n = 17
        vertices = np.array([[x, y, .30] for y in np.linspace(-.12, .12, n) for x in np.linspace(-.12, .12, n)])
        faces = []
        for y in range(n-1):
            for x in range(n-1):
                i = y*n+x
                faces += [[i, i+1, i+n], [i+1, i+n+1, i+n]]
        mesh = out/'cloth.obj'
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh)
        scene = gs.Scene(show_viewer=False,
            sim_options=gs.options.SimOptions(dt=.005, substeps=2),
            pbd_options=gs.options.PBDOptions(particle_size=.008,
                lower_bound=(-.4, -.4, -.05), upper_bound=(.4, .4, .6)))
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())
        scene.add_entity(gs.morphs.Box(pos=(.035, 0, .06), size=(.06, .16, .12), fixed=True), material=gs.materials.Rigid())
        cloth = scene.add_entity(gs.morphs.Mesh(file=str(mesh.resolve()), convexify=False, decimate=False),
            material=gs.materials.PBD.Cloth())
        scene.build()
        states = [cloth.get_particles_pos().detach().cpu().numpy().copy()]
        costs = []
        for _ in range(120):
            t0 = time.monotonic()
            scene.step()
            states.append(cloth.get_particles_pos().detach().cpu().numpy().copy())
            costs.append(time.monotonic()-t0)
        states = np.array(states)
        assert np.isfinite(states).all()
        # Rigid translation alone would preserve all pairwise distances.
        initial, final = states[0], states[-1]
        initial_dist = np.linalg.norm(initial[:, None]-initial[None, :], axis=-1)
        final_dist = np.linalg.norm(final[:, None]-final[None, :], axis=-1)
        change = float(np.max(np.abs(final_dist-initial_dist)))
        np.savez_compressed(out/'particle_states.npz', positions=states, dt=.005)
        status.update(status='completed', frames=len(states), particles=cloth.n_particles,
            max_pairwise_distance_change_m=change, actual_nonrigid_change=change > .001,
            final_height_range_m=[float(final[:, 2].min()), float(final[:, 2].max())],
            measured_step_median_s=float(np.median(costs[10:])), measured_step_p95_s=float(np.quantile(costs[10:], .95)),
            particle_sha256=hashlib.sha256((out/'particle_states.npz').read_bytes()).hexdigest(),
            caveat='No contact-force/convergence/grasp or physical-material validation; no robot capability result.')
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status['end'] = time.time()
        save(out/'status.json', status)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
