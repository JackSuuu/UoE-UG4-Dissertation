"""L1: one native LIBERO-Long episode, or exact fresh-scene action replay.

Only RGB and the complete native instruction reach PinnedActor (TF 4.40.1).
Diagnostics are observational. Success is exclusively env.check_success().
The coordinator owns cohort scheduling; this entry point always runs batch one.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import numpy as np

from audit_openvla_long_tasks import PRIORITY, REVISION, CHECKPOINT
from diagnose_vla_failure import PinnedActor
from diagnose_fridge_skills import (REPO, patch_joint_addresses, patch_mass_matrix,
    bind_renderer_context, validate_controller, fingerprint, save)
from adapters.libero_completion import ContactRecorder

TASKS = dict(zip(('microwave', 'moka', 'mugs', 'drawer'), PRIORITY))
TASKS.update(two_pots=PRIORITY[1], two_mugs=PRIORITY[2])
NOOP = [0., 0., 0., 0., 0., 0., -1.]
SETTLE, STEPS, WALL_SECONDS = 10, 520, 1200
# Audited EGL enumeration differs from NVIDIA physical indices on this host.
EGL_DEVICES = {1: 2, 2: 1, 3: 0}
DEVICE_PREFLIGHT = Path('/home/ami/scratch/native_long_l1_20261019_launch')
SNAPSHOT = (Path.home()/'.cache/huggingface/hub'/
            'models--openvla--openvla-7b-finetuned-libero-10/snapshots'/REVISION)
SOURCES = ['scripts/run_native_long_episode.py', 'scripts/audit_openvla_long_tasks.py',
    'scripts/diagnose_vla_failure.py', 'scripts/openvla_tf440_worker.py',
    'scripts/diagnose_fridge_skills.py', 'scripts/probe_fridge_task.py',
    'src/adapters/libero_runtime.py', 'src/adapters/libero_preservation.py',
    'src/adapters/libero_completion.py', 'src/adapters/libero_openvla.py',
    'src/adapters/tf5_compat.py']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False)+'\n')
    stream.flush()
    os.fsync(stream.fileno())


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def local_environment(args, out):
    """Same paths as libero_runtime.configure, with all config/cache writes in out."""
    import yaml
    base = Path(args.root).resolve()/'libero/libero'
    config = out/'libero_config'
    config.mkdir()
    (out/'tmp').mkdir()
    (config/'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(base),
        bddl_files=str(base/'bddl_files'), init_states=str(base/'init_files'),
        assets=str(base/'assets'), datasets=str(Path(args.root).resolve()/'datasets'))))
    sys.path.insert(0, str(Path(args.root).resolve()))
    os.environ.update(LIBERO_CONFIG_PATH=str(config), MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR=str(out/'mpl_cache'), NUMBA_CACHE_DIR=str(out/'numba_cache'),
        HF_MODULES_CACHE=str(out/'hf_modules'), TOKENIZERS_PARALLELISM='false',
        TMPDIR=str(out/'tmp'))
    # CUDA ordinal 0 in the worker and EGL index are distinct device namespaces.
    if args.gpu not in EGL_DEVICES:
        raise ValueError('Supported physical GPU lanes are 1, 2, 3')
    egl_device = getattr(args, 'egl_device', None)
    args.egl_device = EGL_DEVICES[args.gpu] if egl_device is None else egl_device
    if args.egl_device != EGL_DEVICES[args.gpu]:
        raise ValueError(f'Physical GPU{args.gpu} requires audited EGL{EGL_DEVICES[args.gpu]}')
    raw_identity = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=index,uuid,name,pci.bus_id', '--format=csv,noheader'], text=True).strip()
    physical_index, uuid, name, bus = [part.strip() for part in raw_identity.split(',')]
    if int(physical_index) != args.gpu:
        raise ValueError(f'Unexpected physical GPU identity: {raw_identity}')
    for key, expected in (('CUDA_VISIBLE_DEVICES', str(args.gpu)),
                          ('MUJOCO_EGL_DEVICE_ID', str(args.egl_device))):
        selected = os.environ.get(key, expected)
        if key == 'CUDA_VISIBLE_DEVICES' and selected.startswith('GPU-'):
            if selected != uuid:
                raise ValueError(f'{key}={selected!r} disagrees with physical --gpu {args.gpu}')
        elif selected != expected:
            raise ValueError(f'{key}={selected!r} disagrees with selected device index {expected}')
        os.environ[key] = selected
    identity = dict(physical_gpu=args.gpu, physical_uuid=uuid, name=name, pci_bus_id=bus,
        egl_device=args.egl_device, worker_cuda_ordinal=0,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        egl_mapping_provenance='coordinator_EGL_NV_device_cuda_and_unmasked_CUDA_UUID_audits',
        preflight_sources={})
    for filename in ('egl_device_preflight.json', 'cuda_ordinal_preflight.json'):
        source = DEVICE_PREFLIGHT/filename
        if source.is_file():
            (out/filename).write_bytes(source.read_bytes())
            identity['preflight_sources'][str(source)] = sha(source)
    save(out/'device_identity.json', identity)
    return base


def freeze_sources(out, base):
    manifest = dict(repository={}, runtime={}, packages={}, executable=sys.executable,
        python=sys.version, argv=sys.argv, environment={key: os.environ.get(key) for key in
            ('CUDA_VISIBLE_DEVICES', 'MUJOCO_EGL_DEVICE_ID', 'MUJOCO_GL',
             'PYOPENGL_PLATFORM', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'LIBERO_CONFIG_PATH', 'HF_MODULES_CACHE')})
    for name in SOURCES:
        source = REPO/name
        destination = out/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest['repository'][name] = dict(path=str(source), sha256=sha(source))
    for source in sorted((base/'envs').rglob('*.py')):
        relative = source.relative_to(base)
        destination = out/'source/runtime_libero'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest['runtime'][str(source)] = sha(source)
    for name in ('numpy', 'scipy', 'torch', 'transformers', 'mujoco', 'robosuite',
                 'imageio', 'imageio-ffmpeg'):
        try:
            manifest['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            manifest['packages'][name] = None
    save(out/'source_manifest.json', manifest)
    return manifest


def record_model_sources(args, out, manifest):
    snapshot = Path(args.snapshot).resolve()
    if snapshot.name != REVISION:
        raise ValueError(f'Expected existing checkpoint revision {REVISION}: {snapshot}')
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    entries = {}
    for source in sorted(snapshot.iterdir()):
        if not source.is_file():
            continue
        entry = dict(path=str(source), resolved_path=str(source.resolve()),
                     bytes=source.stat().st_size)
        if source.suffix in ('.safetensors', '.bin'):
            entry['cached_blob_id_not_rehashed'] = source.resolve().name
        else:
            entry['sha256'] = sha(source)
        entries[source.name] = entry
    manifest['checkpoint'] = dict(repository=CHECKPOINT, revision=REVISION, files=entries)
    # PinnedActor uses these exact official base-snapshot source files.
    from adapters.tf5_compat import snapshot_dir
    code = Path(snapshot_dir())
    manifest['actor_code'] = {}
    for source in sorted(code.glob('*.py')):
        destination = out/'source/actor_code'/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest['actor_code'][str(source)] = sha(source)
    save(out/'source_manifest.json', manifest)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def make_scene(args, out, base):
    """Native cameras/assets/language; returned observation precedes the 10 noops."""
    import torch
    patch_joint_addresses()
    patch_mass_matrix()
    bind_renderer_context()
    from libero.libero.envs import OffScreenRenderEnv
    bddl = base/'bddl_files/libero_10'/(args.task+'.bddl')
    initial = base/'init_files/libero_10'/(args.task+'.pruned_init')
    (out/'scene.bddl').write_bytes(bddl.read_bytes())
    states = torch.load(initial, weights_only=False, map_location='cpu')
    chosen = np.asarray(states[args.init_index])
    np.save(out/'selected_initial_state.npy', chosen, allow_pickle=False)
    save(out/'initial_source.json', dict(path=str(initial), index=args.init_index,
        sha256=sha(initial), selected_state_sha256=array_sha(chosen),
        selected_state_shape=list(chosen.shape), selected_state_dtype=str(chosen.dtype),
        bddl_path=str(bddl), bddl_sha256=sha(bddl), seed=args.seed))
    # Keep a reference even if construction/reset fails, for best-effort EGL cleanup.
    env = OffScreenRenderEnv.__new__(OffScreenRenderEnv)
    try:
        OffScreenRenderEnv.__init__(env, bddl_file_name=str(bddl), camera_heights=256,
            camera_widths=256, horizon=STEPS+SETTLE+1, render_gpu_device_id=args.egl_device)
        env.seed(args.seed)
        # Single native reset: wrapper.reset retries RandomizationError indefinitely.
        env.env.reset()
        obs = env.set_init_state(chosen)
        validate_controller(env)
        (out/'model.xml').write_text(env.sim.model.get_xml())
        return env, obs
    except BaseException:
        try:
            env.close()
        except Exception:
            pass
        raise


def record_scene_sources(env, out, manifest):
    import robosuite
    root = Path(inspect.getfile(robosuite)).parent
    for directory in ('controllers', 'utils', 'environments', 'robots', 'models/grippers'):
        for source in sorted((root/directory).rglob('*.py')):
            destination = out/'source/robosuite'/source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            manifest['runtime'][str(source)] = sha(source)
    manifest['assets'] = {}
    for node in ET.fromstring((out/'model.xml').read_text()).iter():
        if node.get('file'):
            source = Path(node.get('file'))
            if not source.is_file():
                raise FileNotFoundError(f'Compiled model asset: {source}')
            manifest['assets'][str(source)] = sha(source)
    manifest['model_xml_sha256'] = sha(out/'model.xml')
    save(out/'source_manifest.json', manifest)


def integration_state(env):
    import mujoco
    model, data = env.sim.model._model, env.sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


class Diagnostics:
    def __init__(self, env):
        self.env = env
        inner = env.env
        self.clauses = inner.parsed_problem['goal_state']
        self.keys = [json.dumps(c, separators=(',', ':')) for c in self.clauses]
        self.targets = sorted({c[1] for c in self.clauses
                               if len(c) == 3 and c[1] in inner.objects_dict})
        self.relevant = tuple(sorted(set(inner.obj_of_interest) | set(self.targets)))
        self.joints = [i for i in range(env.sim.model.njnt)
            if any(name in (env.sim.model.joint_id2name(i) or '')
                   for name in (*self.relevant, 'gripper'))]

    def measure(self, obs):
        inner, m, d = self.env.env, self.env.sim.model, self.env.sim.data
        predicates = {key: bool(inner._eval_predicate(clause))
                      for key, clause in zip(self.keys, self.clauses)}
        joints = {}
        for j in self.joints:
            kind = int(m.jnt_type[j])
            nq, nv = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[kind]
            q, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            joints[m.joint_id2name(j)] = dict(qpos=d.qpos[q:q+nq].tolist(),
                qvel=d.qvel[v:v+nv].tolist(), limited=bool(m.jnt_limited[j]),
                range=m.jnt_range[j].tolist(), type=kind)
        contacts = []
        for c in d.contact[:d.ncon]:
            a, b = m.geom_id2name(c.geom1) or '', m.geom_id2name(c.geom2) or ''
            if any(k in a+b for k in (*self.relevant, 'gripper')):
                contacts.append(dict(geom1=a, geom2=b, distance=float(c.dist)))
        return dict(predicates=predicates, native_success=bool(self.env.check_success()),
            grasped={name: bool(inner._check_grasp(inner.robots[0].gripper,
                inner.objects_dict[name].contact_geoms)) for name in self.targets},
            object_observations={key: np.asarray(value).tolist() for key, value in obs.items()
                if not key.endswith('_image') and any(key.startswith(n+'_') for n in self.relevant)},
            eef_position=obs['robot0_eef_pos'].tolist(), eef_quat=obs['robot0_eef_quat'].tolist(),
            gripper_qpos=obs['robot0_gripper_qpos'].tolist(),
            gripper_qvel=np.asarray(obs.get('robot0_gripper_qvel', [])).tolist(),
            gripper_current_action=inner.robots[0].gripper.current_action.tolist(),
            joints=joints, contacts_endpoint=contacts, sim_time=float(d.time))


def transitions(before, after):
    return dict(attained=[k for k, v in after['predicates'].items()
                          if v and not before['predicates'][k]],
                lost=[k for k, v in after['predicates'].items()
                      if not v and before['predicates'][k]],
                grasp_attained=[k for k, v in after['grasped'].items()
                                if v and not before['grasped'][k]],
                grasp_lost=[k for k, v in after['grasped'].items()
                            if not v and before['grasped'][k]])


class Tape:
    """Atomic per-frame spool survives interrupts; final NPZ files are built once."""
    def __init__(self, directory):
        self.directory = directory
        (directory/'frames').mkdir(parents=True)
        self.log = (directory/'frames.jsonl').open('w')
        self.count = 0
        self.last = None

    def capture(self, env, obs, diagnostic):
        frame = dict(index=self.count, fingerprint=fingerprint(env, obs), diagnostics=diagnostic)
        arrays = dict(state=env.get_sim_state().copy(), integration=integration_state(env),
                      **{k: v for k, v in obs.items() if k.endswith('_image')})
        path = self.directory/'frames'/f'{self.count:06d}.npz'
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        path.with_suffix('.tmp').replace(path)
        write_row(self.log, frame)
        self.count += 1
        self.last = frame
        return frame

    def close(self):
        self.log.close()
        if not self.count:
            return
        arrays = {}
        for i in range(self.count):
            with np.load(self.directory/'frames'/f'{i:06d}.npz', allow_pickle=False) as frame:
                for key in frame.files:
                    arrays.setdefault(key, []).append(frame[key])
        for filename, keys in (('states.npz', ['state', 'integration']),
                               ('images.npz', [k for k in arrays if k.endswith('_image')])):
            path = self.directory/filename
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, **{k: np.asarray(arrays[k]) for k in keys})
            path.with_suffix('.tmp').replace(path)


class WallBudget(BaseException):
    pass


@contextmanager
def deadline(seconds):
    def expired(signum, frame):
        raise WallBudget('1200s episode wall budget exhausted')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def cleanup_actor(actor):
    """Also works when PinnedActor.__init__ failed while waiting for readiness."""
    process = getattr(actor, 'process', None)
    if process is not None:
        # Terminate promptly even if a deadline interrupted a CUDA query.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()
    log = getattr(actor, 'log', None)
    if log is not None:
        log.close()


def step_once(env, obs, action, diag, recorder, tape, stream, index, phase, extra=None):
    before = tape.last
    recorder.reset()
    start = time.monotonic()
    obs, reward, done, info = env.step(action)
    simulator_s = time.monotonic()-start
    measured = diag.measure(obs)
    after = tape.capture(env, obs, measured)
    row = dict(step=index, phase=phase, owner='vla' if phase == 'actor' else 'initialization',
        action=action, executed_mapped_action=action, before=before['fingerprint'],
        after=after['fingerprint'], diagnostics=measured,
        events=transitions(before['diagnostics'], measured), reward=float(reward),
        done=bool(done), simulator_s=simulator_s, contacts_substeps=recorder.samples,
        **(extra or {}))
    write_row(stream, row)
    return obs, row


def summarize(initial, rows):
    history = [initial]+[row['diagnostics'] for row in rows]
    goals = {}
    for key in initial['predicates']:
        values = [d['predicates'][key] for d in history]
        goals[key] = dict(initial=values[0], final=values[-1], ever=any(values),
            first_true_frame=next((i for i, value in enumerate(values) if value), None),
            attainment_steps=[r['step'] for r in rows if key in r['events']['attained']],
            loss_steps=[r['step'] for r in rows if key in r['events']['lost']])
    return dict(goals=goals, any_grasp={k: any(d['grasped'][k] for d in history)
        for k in initial['grasped']}, any_native_success=any(d['native_success'] for d in history),
        final=history[-1])


def run_episode(args):
    import imageio.v2 as imageio
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='started', scope='L1_actor_only_native_LIBERO_Long',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        started=time.time(), pid=os.getpid(), actor_queries=0, steps_executed=0,
        initialization_actions=0, protocol=dict(seed=0, roots=list(range(6)),
            initialization_noops=SETTLE, max_model_actions=STEPS, wall_seconds=WALL_SECONDS,
            wall_scope='scene_setup_initialization_and_episode_excluding_model_load',
            camera='native_agentview_256', instruction='full_native', batch_size=1,
            success='native_env_check_success', checkpoint_revision=REVISION))
    save(out/'status.json', status)
    actor = env = recorder = video = tape = settle_tape = None
    handles, rows = [], []
    episode_start = None
    initial = None
    result = 1
    try:
        base = local_environment(args, out)
        status['args']['egl_device'] = args.egl_device
        manifest = freeze_sources(out, base)
        if not args.smoke:
            record_model_sources(args, out, manifest)
            status['phase'] = 'model_load'
            save(out/'status.json', status)
            start = time.monotonic()
            actor = PinnedActor.__new__(PinnedActor)
            PinnedActor.__init__(actor, args, out)
            status.update(actor_build=actor.build, model_load_s=time.monotonic()-start)
            if actor.build.get('transformers') != '4.40.1' or actor.build.get('H') != 1:
                raise RuntimeError(f'Unexpected pinned actor build: {actor.build}')
        # Loading/importing actor dependencies must not change fresh-scene RNG state.
        seed_all(args.seed)
        episode_start = time.monotonic()
        status['phase'] = 'scene_setup'
        save(out/'status.json', status)
        with deadline(WALL_SECONDS):
            start = time.monotonic()
            env, obs = make_scene(args, out, base)
            status['scene_setup_s'] = time.monotonic()-start
            record_scene_sources(env, out, manifest)
            diag = Diagnostics(env)
            status.update(language=env.language_instruction, goal_clauses=diag.clauses,
                          grasp_targets=diag.targets)
            recorder = ContactRecorder(env.sim, keywords=(*diag.relevant, 'gripper'))
            settle_tape = Tape(out/'initialization')
            settle_tape.capture(env, obs, diag.measure(obs))
            stream = (out/'initialization/steps.jsonl').open('w')
            handles.append(stream)
            status['phase'] = 'initialization'
            for step in range(SETTLE):
                obs, row = step_once(env, obs, NOOP, diag, recorder, settle_tape,
                                     stream, step, 'initialization')
                status['initialization_actions'] += 1
            tape = Tape(out)
            initial = diag.measure(obs)
            tape.capture(env, obs, initial)
            status.update(initial=tape.last['fingerprint'], initial_diagnostics=initial)
            imageio.imwrite(out/'initial.png', obs['agentview_image'][::-1, ::-1])
            video = imageio.get_writer(out/'raw.mp4', fps=20)
            video.append_data(obs['agentview_image'][::-1, ::-1])
            stream, queries = (out/'steps.jsonl').open('w'), (out/'queries.jsonl').open('w')
            handles.extend([stream, queries])
            status['phase'] = 'scene_smoke' if args.smoke else 'actor'
            save(out/'status.json', status)
            if args.smoke:
                status.update(status='scene_smoke_passed', termination='scene_smoke', no_actor_queries=True)
            elif initial['native_success']:
                status.update(status='completed', termination='native_success')
            else:
                for step in range(STEPS):
                    start = time.monotonic()
                    status['actor_queries'] += 1
                    # Durable query intent distinguishes an interrupted request from no request.
                    write_row(queries, dict(event='requested', step=step,
                        before=tape.last['fingerprint'], language=env.language_instruction,
                        episode_elapsed_s=start-episode_start))
                    action, decode, image = actor.act(obs['agentview_image'], env.language_instruction)
                    inference_s = time.monotonic()-start
                    action = np.asarray(action, dtype=float)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError(f'Invalid actor action: {action}')
                    native = np.asarray(decode['native_action'], dtype=float)
                    expected = native.copy()
                    expected[-1] = -np.sign(2*expected[-1]-1)
                    np.testing.assert_array_equal(action, expected)
                    if len(decode['tokens']) != 7:
                        raise ValueError('Expected seven official action tokens')
                    extra = dict(proposal=action.tolist(), native_proposal=native.tolist(),
                        tokens=decode['tokens'], decode=decode, inference_s=inference_s,
                        episode_elapsed_s=time.monotonic()-episode_start)
                    write_row(queries, dict(event='returned', step=step, **extra))
                    if step == 0:
                        image.save(out/'policy_input.png')
                    obs, row = step_once(env, obs, action.tolist(), diag, recorder, tape,
                                         stream, step, 'actor', extra)
                    rows.append(row)
                    status['steps_executed'] = len(rows)
                    video.append_data(obs['agentview_image'][::-1, ::-1])
                    save(out/'status.json', status)
                    if step % 20 == 0 or row['diagnostics']['native_success']:
                        print(json.dumps(dict(step=step, predicates=row['diagnostics']['predicates'],
                            grasped=row['diagnostics']['grasped'],
                            native_success=row['diagnostics']['native_success'])), flush=True)
                    if row['diagnostics']['native_success']:
                        status['termination'] = 'native_success'
                        break
                    if row['done']:
                        raise RuntimeError('Environment done before native success or action budget')
                status.update(status='completed', termination=status.get('termination', 'action_budget'))
            result = 0
    except WallBudget:
        # A model/step interrupted in flight is visible in queries and the frame spool.
        status.update(status='completed' if initial is not None and not args.smoke else 'infrastructure_failed',
                      termination='wall_budget', interrupted_phase=status.get('phase'))
        result = 0 if status['status'] == 'completed' else 1
    except BaseException:
        status.update(status='infrastructure_failed', termination='infrastructure_failure',
                      error=traceback.format_exc(), failed_phase=status.get('phase'))
    finally:
        if episode_start is not None:
            status['episode_wall_s'] = time.monotonic()-episode_start
        if initial is not None:
            status.update(summarize(initial, rows))
        # Cleanup is independent: one failed artifact must not strand the actor or EGL context.
        errors = []
        operations = [('actor', lambda: cleanup_actor(actor))]
        operations += [('stream', handle.close) for handle in handles]
        def save_actions():
            path = out/'actions.npz'
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,
                    mapped=np.asarray([r['action'] for r in rows], dtype=float).reshape(-1, 7),
                    native=np.asarray([r['native_proposal'] for r in rows], dtype=float).reshape(-1, 7),
                    tokens=np.asarray([r['tokens'] for r in rows], dtype=np.int64).reshape(-1, 7))
            path.with_suffix('.tmp').replace(path)
        operations.append(('action_tape', save_actions))
        operations += [(name, resource.close) for name, resource in
                       [('video', video), ('tape', tape), ('initialization_tape', settle_tape),
                        ('contacts', recorder), ('environment', env)] if resource is not None]
        for name, operation in operations:
            try:
                operation()
            except BaseException:
                errors.append(dict(resource=name, error=traceback.format_exc()))
        if errors:
            status.update(status='infrastructure_failed', cleanup_errors=errors)
            result = 1
        status['outcome'] = ('infrastructure_failure' if status['status'] == 'infrastructure_failed'
            else 'scene_smoke' if args.smoke else 'native_success'
            if status.get('any_native_success') else 'no_native_success')
        status['partial_outputs'] = status['status'] == 'infrastructure_failed' or status.get('termination') == 'wall_budget'
        status.update(ended=time.time(), total_wall_s=time.time()-status['started'])
        save(out/'status.json', status)
    print(json.dumps(dict(status=status['status'], termination=status.get('termination'),
        out=str(out), actor_queries=status['actor_queries'], steps_executed=status['steps_executed'])), flush=True)
    return result


class ReplayTape:
    def __init__(self, directory):
        self.frames = load_rows(directory/'frames.jsonl')
        # Cache each NPZ once: indexing np.load(...)[key] in a loop repeatedly decompresses it.
        self.arrays = {}
        for name in ('states.npz', 'images.npz'):
            with np.load(directory/name, allow_pickle=False) as archive:
                self.arrays.update({key: archive[key] for key in archive.files})
        if any(len(a) != len(self.frames) for a in self.arrays.values()):
            raise ValueError(f'Incomplete tape: {directory}')

    def compare(self, index, env, obs, diagnostic, report, phase):
        expected = self.frames[index]
        actual = dict(state=env.get_sim_state(), integration=integration_state(env),
                      **{k: v for k, v in obs.items() if k.endswith('_image')})
        mismatch = []
        for key, arrays in self.arrays.items():
            if not np.array_equal(arrays[index], actual[key]):
                mismatch.append(key)
            if key in ('state', 'integration'):
                gap = float(np.max(np.abs(arrays[index]-actual[key])))
                report['max_'+key+'_gap'] = max(report['max_'+key+'_gap'], gap)
        if fingerprint(env, obs) != expected['fingerprint']:
            mismatch.append('fingerprint')
        if diagnostic != expected['diagnostics']:
            mismatch.append('diagnostics')
        if mismatch:
            report['mismatches'].append(dict(phase=phase, frame=index, fields=mismatch))
        report['frames_checked'] += 1


def replay(args):
    """Fresh scene, recorded noops then mapped actions; no simulator-state installs after reset."""
    run, out = args.replay.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', source=str(run), started=time.time(), mismatches=[],
                  frames_checked=0, max_state_gap=0., max_integration_gap=0., actor_queries=0)
    env = recorder = None
    try:
        previous = json.loads((run/'status.json').read_text())
        original = argparse.Namespace(**previous['args'])
        if args.gpu is not None and args.gpu != original.gpu:
            raise ValueError('Replay must preserve the recorded physical GPU')
        recorded_egl = getattr(original, 'egl_device', None)
        original.egl_device = EGL_DEVICES[original.gpu] if recorded_egl is None else recorded_egl
        if args.egl_device is not None and args.egl_device != original.egl_device:
            raise ValueError('Replay must preserve the recorded EGL device')
        original.out = out
        base = local_environment(original, out)
        manifest = json.loads((run/'source_manifest.json').read_text())
        for name, entry in manifest['repository'].items():
            if sha(run/'source'/name) != entry['sha256'] or sha(REPO/name) != entry['sha256']:
                raise ValueError(f'Replay source mismatch: {name}')
        for section in ('runtime', 'assets'):
            for path, expected in manifest.get(section, {}).items():
                if sha(path) != expected:
                    raise ValueError(f'Replay {section} mismatch: {path}')
        seed_all(original.seed)
        env, obs = make_scene(original, out, base)
        if 'model_xml_sha256' in manifest and sha(out/'model.xml') != manifest['model_xml_sha256']:
            raise ValueError('Replay compiled model XML mismatch')
        if json.loads((out/'initial_source.json').read_text()) != json.loads((run/'initial_source.json').read_text()):
            raise ValueError('Replay initial-state source mismatch')
        diag = Diagnostics(env)
        recorder = ContactRecorder(env.sim, keywords=(*diag.relevant, 'gripper'))
        for phase, directory in (('initialization', run/'initialization'), ('actor', run)):
            tape = ReplayTape(directory)
            rows = load_rows(directory/'steps.jsonl')
            if len(tape.frames) != len(rows)+1:
                raise ValueError(f'Uncommitted step/frame in {directory}')
            if phase == 'initialization' and len(rows) != SETTLE:
                raise ValueError('Replay requires exactly ten initialization noops')
            tape.compare(0, env, obs, diag.measure(obs), report, phase)
            for index, row in enumerate(rows):
                if row['step'] != index or row['action'] != row['executed_mapped_action']:
                    raise ValueError('Invalid action tape')
                if phase == 'initialization':
                    if row['action'] != NOOP:
                        raise ValueError('Unexpected initialization action')
                elif row['owner'] != 'vla' or row['action'] != row['proposal']:
                    raise ValueError('Non-native actor action')
                before = fingerprint(env, obs)
                previous_diag = diag.measure(obs)
                recorder.reset()
                obs, reward, done, info = env.step(row['action'])
                measured = diag.measure(obs)
                tape.compare(index+1, env, obs, measured, report, phase)
                fields = []
                for key, value in dict(before=before, after=fingerprint(env, obs),
                    diagnostics=measured, events=transitions(previous_diag, measured),
                    reward=float(reward), done=bool(done), contacts_substeps=recorder.samples).items():
                    if row[key] != value:
                        fields.append(key)
                if fields:
                    report['mismatches'].append(dict(phase=phase, step=index, fields=fields))
                save(out/'verification.json', report)
        report.update(status='replay_mismatch' if report['mismatches'] else 'exact_replay_passed',
                      steps=len(rows), original_status=previous['status'])
    except BaseException:
        report.update(status='infrastructure_failed', error=traceback.format_exc())
    finally:
        for resource in (recorder, env):
            if resource is not None:
                try:
                    resource.close()
                except BaseException:
                    report.update(status='infrastructure_failed', cleanup_error=traceback.format_exc())
        report['ended'] = time.time()
        save(out/'verification.json', report)
    print(json.dumps(report), flush=True)
    return 0 if report['status'] == 'exact_replay_passed' else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New, unique output directory')
    parser.add_argument('--root', default='/home/ami/scratch/LIBERO')
    parser.add_argument('--task', choices=[*TASKS, *PRIORITY])
    parser.add_argument('--init-index', type=int, choices=range(6), default=0)
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    parser.add_argument('--gpu', type=int, choices=[1, 2, 3],
                        help='Physical GPU index (episode default: 1; replay: recorded GPU)')
    parser.add_argument('--egl-device', type=int, choices=[0, 1, 2],
                        help='EGL index (audited default: physical GPU1/2/3 -> EGL2/1/0)')
    parser.add_argument('--snapshot', default=str(SNAPSHOT))
    parser.add_argument('--pinned-python', default='/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python')
    parser.add_argument('--steps', type=int, choices=[STEPS], default=STEPS)
    parser.add_argument('--smoke', action='store_true', help='Native reset and ten noops; zero model calls')
    parser.add_argument('--replay', type=Path, help='Replay a run from a fresh native scene, zero model calls')
    args = parser.parse_args()
    if args.replay and args.smoke:
        parser.error('--replay and --smoke are mutually exclusive')
    if not args.replay and not args.task:
        parser.error('--task is required for an episode or scene smoke')
    if not args.replay and args.gpu is None:
        args.gpu = 1
    args.task = TASKS.get(args.task, args.task)
    return args


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    arguments = parse_args()
    raise SystemExit(replay(arguments) if arguments.replay else run_episode(arguments))
