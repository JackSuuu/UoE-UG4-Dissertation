"""Explicit local LIBERO setup; no global simulator/package edits."""
import os
import sys
from pathlib import Path

TASK = 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy'


def configure(root):
    import yaml
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    config = root / '.local_config'
    config.mkdir(exist_ok=True)
    base = root / 'libero/libero'
    paths = dict(benchmark_root=str(base), bddl_files=str(base / 'bddl_files'),
                 init_states=str(base / 'init_files'), assets=str(base / 'assets'),
                 datasets=str(root / 'datasets'))
    (root / 'datasets').mkdir(exist_ok=True)
    (config / 'config.yaml').write_text(yaml.safe_dump(paths))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    return base


def make_env(root, init_index=0, task=TASK):
    base = configure(root)
    import torch
    patch_joint_addresses()
    patch_mass_matrix()
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=str(base / 'bddl_files/libero_10' / (task + '.bddl')),
        camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    states = torch.load(base / 'init_files/libero_10' / (task + '.pruned_init'),
                        weights_only=False)
    obs = env.set_init_state(states[init_index])
    return env, obs


def patch_joint_addresses():
    """MuJoCo 3.14 enums compare asymmetrically with np.int32 in tuple membership.

    Preserve robosuite 1.4.1 joint address semantics using integer enum values.
    This changes only Python address lookup, never mechanics or solver settings.
    """
    from robosuite.utils.binding_utils import MjModel

    def address(model, name, velocity):
        j = model.joint_name2id(name)
        kind = int(model.jnt_type[j])
        dims = {0: 6 if velocity else 7, 1: 3 if velocity else 4, 2: 1, 3: 1}
        n = dims[kind]
        start = int((model.jnt_dofadr if velocity else model.jnt_qposadr)[j])
        return start if n == 1 else (start, start + n)

    MjModel.get_joint_qpos_addr = lambda self, name: address(self, name, False)
    MjModel.get_joint_qvel_addr = lambda self, name: address(self, name, True)


def patch_mass_matrix():
    """Adapt the one changed mj_fullM call, preserving upstream update logic."""
    import inspect
    import textwrap
    import mujoco
    from robosuite.controllers import base_controller
    if hasattr(mujoco.MjData, 'qM'):
        return
    cls = base_controller.Controller
    if getattr(cls.update, '_libero_mass_patch', False):
        return
    source = textwrap.dedent(inspect.getsource(cls.update))
    old = 'mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)'
    assert source.count(old) == 1, 'Unexpected robosuite controller source'
    source = source.replace(old, 'mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)')
    namespace = {}
    exec(compile(source, '<libero_mj_fullM_compat>', 'exec'), vars(base_controller), namespace)
    cls.update = namespace['update']
    cls.update._libero_mass_patch = True
