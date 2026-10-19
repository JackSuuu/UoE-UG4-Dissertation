"""L2 native microwave diagnostic branch, exact prefix smoke, and zero-query replay.

snapshot-step is zero-based POST-ACTION t: replay 10 initialization noops and
baseline actions 0..t inclusive. Every new action is a fresh official H1 query.
The first executed continuation action is a forced diagnostic handoff, never a
verified-repair claim. One invocation owns one candidate and one model worker.
"""
import argparse
import base64
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
NEW_SOURCES = ('run_native_long_branch.py', 'native_long_candidate_worker.py')
STEPS, WALL_SECONDS = 520, 120
PROTOCOL = HERE.parent/'exp_record/agent_notes/native_microwave_l2_protocol_20261019.md'
SNAPSHOT_STEPS = (253, 266, 275, 316, 275, 272)


def generation_kwargs(source):
    settings = dict(do_sample=source == 'sample', top_k=50, num_beams=1, num_return_sequences=1)
    if source == 'sample':
        settings.update(temperature=.7, top_p=.95)
    return settings


def request_seed(args, index):
    return args.seed+index if args.candidate == 'sample' and index < args.local_horizon else 0


def validate_decode(decode, metadata):
    for key in ('source', 'seed', 'query_index', 'policy_png_sha256', 'policy_rgb_sha256'):
        require(decode[key] == metadata[key], f'Worker request/response mismatch: {key}')
    settings = generation_kwargs(metadata['source'])
    require(decode['sampling'] == settings, 'Unexpected generation kwargs')
    config = decode['resolved_generation_config']
    for key, value in dict(settings, max_new_tokens=7).items():
        require(config.get(key) == value, f'Resolved generation config mismatch: {key}')
    require(decode['prompt'] == f"In: What action should the robot take to {metadata['language'].lower()}?\nOut:",
            'Prompt differs from the original instruction')


class InvalidGeneration(ValueError):
    pass


class NominalMismatch(ValueError):
    pass


class WorkerTimeout(TimeoutError):
    def __init__(self, phase):
        self.phase = phase
        super().__init__(f'Worker {phase} deadline exhausted')


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_runner(baseline):
    """Import the baseline's archived implementation, not mutable working helpers."""
    manifest = load(baseline/'source_manifest.json')
    for name, entry in manifest['repository'].items():
        require(sha(baseline/'source'/name) == entry['sha256'], f'Frozen source mismatch: {name}')
    for section in ('runtime', 'assets'):
        for path, expected in manifest.get(section, {}).items():
            require(sha(path) == expected, f'Baseline {section} mismatch: {path}')
    sys.path[:0] = [str(baseline/'source/scripts'), str(baseline/'source/src')]
    runner = importlib.import_module('run_native_long_episode')
    require(Path(runner.__file__).resolve() == baseline/'source/scripts/run_native_long_episode.py',
            'Runner must come from the frozen baseline')
    return runner, manifest


def zero_query_guard(runner, proof):
    """Fail closed even if a future smoke/replay code path accidentally loads an actor."""
    class ForbiddenActor:
        def __new__(cls, *a, **kw):
            proof['actor_constructor_attempts'] += 1
            raise RuntimeError('Model construction forbidden in zero-query mode')

    runner.PinnedActor = ForbiddenActor
    import diagnose_vla_failure
    diagnose_vla_failure.PinnedActor = ForbiddenActor
    def forbidden_query(*a, **kw):
        proof['actor_query_attempts'] += 1
        raise RuntimeError('Model query forbidden in zero-query mode')
    CandidateActor.query = forbidden_query

    def audit(event, values):
        if event == 'subprocess.Popen':
            command = str(values[1])
            proof['subprocesses'].append(command)
            if 'worker.py' in command:
                proof['worker_spawn_attempts'] += 1
                raise RuntimeError('Model worker forbidden in zero-query mode')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = os.fsdecode(values[0])
            if path.endswith('.safetensors') or ('huggingface' in path and path.endswith('.bin')):
                proof['model_weight_open_attempts'] += 1
                raise RuntimeError('Model weights forbidden in zero-query mode')

    sys.addaudithook(audit)


def controller_fields(env):
    numeric = {}
    for key, value in vars(env.robots[0].controller).items():
        if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
            numeric[key] = value.tolist()
        elif isinstance(value, (bool, int, float, str)) or value is None:
            numeric[key] = value
    numeric.update(gripper_current_action=env.robots[0].gripper.current_action.tolist(),
        timestep=env.env.timestep, cur_time=env.env.cur_time, done=bool(env.env.done))
    return numeric


def evidence_tape(runner, directory, out):
    class EvidenceTape(runner.ReplayTape):
        def compare(self, index, env, obs, diagnostic, report, phase):
            count = len(report['mismatches'])
            super().compare(index, env, obs, diagnostic, report, phase)
            if len(report['mismatches']) == count:
                return
            destination = out/'mismatches'/phase/f'{index:06d}'
            destination.mkdir(parents=True, exist_ok=True)
            actual = dict(state=env.get_sim_state(), integration=runner.integration_state(env),
                          **{k: v for k, v in obs.items() if k.endswith('_image')})
            np.savez_compressed(destination/'expected.npz',
                **{k: v[index] for k, v in self.arrays.items()})
            np.savez_compressed(destination/'actual.npz', **actual)
            save(destination/'expected.json', self.frames[index])
            save(destination/'actual.json', dict(fingerprint=runner.fingerprint(env, obs),
                diagnostics=diagnostic, controller_fields=controller_fields(env)))
    return EvidenceTape(directory)


def validate_action(row, index, initialization=False):
    require(row['step'] == index, f'Nonsequential action at {index}')
    action = np.asarray(row['action'], dtype=float)
    require(action.shape == (7,) and np.isfinite(action).all(), 'Invalid action shape/value')
    require(row['action'] == row['executed_mapped_action'], 'Mapped tape differs from action')
    if initialization:
        require(row['action'] == [0., 0., 0., 0., 0., 0., -1.], 'Non-native initialization')
    else:
        require(row['action'] == row['proposal'], 'Action differs from model proposal')
        require(len(row['tokens']) == 7, 'Expected seven official tokens')
        native = np.asarray(row['native_proposal'], dtype=float)
        require(native.shape == (7,) and np.isfinite(native).all(), 'Invalid native proposal')
        mapped = native.copy()
        mapped[-1] = -np.sign(2*mapped[-1]-1)
        np.testing.assert_array_equal(action, mapped)


def validate_model_sources(runner, original, out, manifest, baseline_manifest):
    runner.record_model_sources(original, out, manifest)
    require(manifest['checkpoint'] == baseline_manifest['checkpoint'],
            'Checkpoint provenance differs from the frozen baseline')
    require(manifest['actor_code'] == baseline_manifest['actor_code'],
            'Official model code differs from the frozen baseline')


def validate_nominal_controls(args, source_hashes):
    require(len(args.nominal_controls) == 6, 'Sampling requires all six nominal controls')
    roots, evidence = set(), []
    for directory in args.nominal_controls:
        directory = directory.resolve()
        status, identity = load(directory/'status.json'), load(directory/'nominal_identity.json')
        original = load(Path(status['args']['baseline'])/'status.json')
        root = original['args']['init_index']
        require(root in range(6) and root not in roots, 'Nominal controls must cover unique roots 0..5')
        roots.add(root)
        require(status['status'] == 'completed' and not status['partial_outputs'] and
            status['args']['candidate'] == 'greedy' and status['args']['seed'] == 0 and
            status['args']['local_horizon'] == 32 and status['args']['continuation_steps'] == 0 and
            status['args']['snapshot_step'] == SNAPSHOT_STEPS[root] and
            status['branch_sources'] == source_hashes and status['protocol_sha256'] == sha(PROTOCOL),
            f'Ineligible nominal control: {directory}')
        require(identity['status'] == 'exact_nominal_identity_passed' and not identity['mismatches']
            and identity['steps_checked'] == status['steps_executed'] == status['actor_queries']
            and (identity['steps_checked'] == 32 or status['termination'] == 'native_success'),
            f'Nominal model re-query identity not established: {directory}')
        evidence.append(dict(root=root, directory=str(directory),
            status_sha256=sha(directory/'status.json'), identity_sha256=sha(directory/'nominal_identity.json')))
    return evidence


def nominal_query_check(runner, reference, baseline_rows, metadata, decode, png, out, identity):
    """Saved observations are used only by this observer, never as actor input."""
    from adapters.libero_openvla import policy_image
    step = metadata['step']
    require(step < len(baseline_rows), 'Nominal suffix ended before the fresh control')
    original = baseline_rows[step]
    buffer = io.BytesIO()
    policy_image(reference.arrays['agentview_image'][step]).save(buffer, format='PNG')
    expected_png = buffer.getvalue()
    actual = dict(tokens=decode['tokens'], native_action=decode['native_action'],
        mapped_action=decode['action'], policy_png_sha256=hashlib.sha256(png).hexdigest(),
        before=metadata['before'])
    expected = dict(tokens=original['tokens'], native_action=original['native_proposal'],
        mapped_action=original['action'], policy_png_sha256=hashlib.sha256(expected_png).hexdigest(),
        before=original['before'])
    differences = {k: dict(expected=expected[k], actual=actual[k]) for k in expected if actual[k] != expected[k]}
    if png != expected_png:
        destination = out/'mismatches/nominal_policy'/f'{step:06d}'
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'expected.png').write_bytes(expected_png)
        (destination/'actual.png').write_bytes(png)
    check = dict(step=step, branch_step=metadata['branch_step'], before_action_equal=not differences,
        expected=expected, actual=actual, differences=differences, action_executed=False)
    identity['checks'].append(check)
    if differences:
        identity['mismatches'].append(dict(step=step, phase='nominal_query', differences=differences))
    save(out/'nominal_identity.json', identity)


def nominal_step_check(reference, original, row, env, obs, out, identity):
    check = identity['checks'][-1]
    check['action_executed'] = True
    check['after_differences'] = {key: dict(expected=original[key], actual=row[key])
        for key in ('after', 'diagnostics', 'events', 'reward', 'done', 'contacts_substeps')
        if original[key] != row[key]}
    reference.compare(row['step']+1, env, obs, row['diagnostics'], identity, 'nominal_suffix')
    if check['after_differences']:
        identity['mismatches'].append(dict(step=row['step'], phase='nominal_after_action',
                                         differences=check['after_differences']))
    identity['steps_checked'] += 1
    save(out/'nominal_identity.json', identity)
    if identity['mismatches']:
        raise NominalMismatch('Fresh nominal model suffix differs; sampling must remain blocked')


def replay_segment(runner, env, obs, diag, recorder, source, rows, report, phase,
                   out, resources, handles, baseline_start=0):
    """Compare every frame, controller fingerprint, causal transition, and contact substep."""
    expected = evidence_tape(runner, source, out)
    require(len(expected.frames) >= len(rows)+1, f'Incomplete frame tape: {source}')
    actual = runner.Tape(out/'reconstruction'/phase)
    resources.append((phase+'_tape', actual))
    actual.capture(env, obs, diag.measure(obs))
    expected.compare(0, env, obs, diag.measure(obs), report, phase)
    stream = (actual.directory/'steps.jsonl').open('w')
    handles.append(stream)
    for index, row in enumerate(rows):
        validate_action(row, baseline_start+index, phase == 'initialization')
        if phase == 'baseline':
            require(row['owner'] == 'vla', 'Baseline contains a non-VLA action')
        require(not diag.measure(obs)['native_success'], 'Source tape continues past native success')
        obs, replayed = runner.step_once(env, obs, row['action'], diag, recorder, actual,
            stream, row['step'], row['phase'], dict(replay_only=True))
        expected.compare(index+1, env, obs, replayed['diagnostics'], report, phase)
        fields = [key for key in ('before', 'after', 'diagnostics', 'events', 'reward', 'done',
                  'contacts_substeps') if replayed[key] != row[key]]
        if fields:
            report['mismatches'].append(dict(phase=phase, step=row['step'], fields=fields))
            destination = out/'mismatches'/phase/f'step_{index:06d}'
            destination.mkdir(parents=True, exist_ok=True)
            save(destination/'expected.json', row)
            save(destination/'actual.json', replayed)
        save(out/'verification.json', report)
    return obs


class CandidateActor:
    def __init__(self, runner, original, args, out, identity):
        from adapters.tf5_compat import snapshot_dir
        self.log = (out/'worker.log').open('w')
        self.query_seconds = args.query_seconds
        self.pending = b''
        readiness_deadline = time.monotonic()+args.readiness_seconds
        self.process = subprocess.Popen([original.pinned_python, '-B', '-u',
            str(HERE/'native_long_candidate_worker.py'), original.snapshot, snapshot_dir(),
            '--gpu', str(args.gpu), '--physical-uuid', identity['physical_uuid'],
            '--egl-device', str(args.egl_device)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, bufsize=0,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=identity['physical_uuid'],
                     HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        for pipe in (self.process.stdin, self.process.stdout):
            os.set_blocking(pipe.fileno(), False)
        self.build = self.receive(readiness_deadline, 'readiness')
        require(self.build.get('ready') and self.build.get('transformers') == '4.40.1'
            and self.build.get('H') == 1 and self.build.get('batch') == 1
            and self.build.get('frozen') and self.build.get('official_predict_action'),
            f'Unexpected actor build: {self.build}')
        require(self.build['device_identity']['physical_uuid'] == identity['physical_uuid'],
                'Worker GPU UUID differs from scene lane')
        require(self.build['unnorm_key'] == 'libero_10', 'Wrong normalization key')
        require(self.build['generation_kwargs'] == {s: generation_kwargs(s) for s in ('greedy', 'sample')},
                'Worker decoding configuration differs from the frozen protocol')

    def receive(self, deadline, phase):
        # FD-level buffering handles partial lines and startup logs without a
        # blocking readline. No alarm is installed: the charged wall alarm remains active.
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout(phase)
            if b'\n' in self.pending:
                line, self.pending = self.pending.split(b'\n', 1)
                text = line.decode()
                if text.startswith('RESULT '):
                    return json.loads(text[7:])
                self.log.write(text+'\n')
                self.log.flush()
                continue
            readable, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not readable:
                raise WorkerTimeout(phase)
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f'Model worker exited during {phase}: {self.process.poll()}')
            self.pending += chunk

    def query(self, png, language, source, seed, query_index):
        deadline = time.monotonic()+self.query_seconds
        data = (json.dumps(dict(png=base64.b64encode(png).decode(), language=language,
            source=source, seed=seed, query_index=query_index))+'\n').encode()
        while data:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                raise WorkerTimeout('query')
            try:
                sent = os.write(self.process.stdin.fileno(), data)
            except BlockingIOError:
                continue
            data = data[sent:]
        return self.receive(deadline, 'query')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--snapshot-step', type=int, help='Zero-based POST-ACTION baseline step t')
    parser.add_argument('--candidate', choices=['greedy', 'sample'])
    parser.add_argument('--seed', type=int, help='Candidate seed; scene seed is inherited from baseline')
    parser.add_argument('--local-horizon', type=int)
    parser.add_argument('--continuation-steps', type=int)
    parser.add_argument('--gpu', type=int, choices=[2, 3])
    parser.add_argument('--egl-device', type=int, choices=[0, 1])
    parser.add_argument('--out', type=Path, required=True, help='New unique output directory')
    parser.add_argument('--wall-seconds', type=float, default=WALL_SECONDS,
                        help='Charged scene/reconstruction/action budget (L2A/replay 120; future L2B 300)')
    parser.add_argument('--readiness-seconds', type=float, default=180)
    parser.add_argument('--query-seconds', type=float, default=30)
    parser.add_argument('--nominal-controls', type=Path, nargs='+',
                        help='Six completed C0 directories required before sampled model calls')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='Restore exact prefix with zero model calls')
    mode.add_argument('--replay', type=Path, help='Fresh-process replay of a branch; zero model calls')
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.replay:
        recorded = load(args.replay/'status.json')['args']
        for key in ('baseline', 'snapshot_step', 'candidate', 'seed', 'local_horizon',
                    'continuation_steps', 'gpu', 'egl_device'):
            value = recorded[key]
            if key == 'baseline':
                value = Path(value)
            supplied = getattr(args, key)
            if supplied is not None and supplied != value:
                parser.error(f'Replay --{key.replace("_", "-")} differs from recorded branch')
            setattr(args, key, value)
    else:
        for key in ('baseline', 'snapshot_step', 'candidate', 'seed', 'local_horizon',
                    'continuation_steps', 'gpu', 'egl_device'):
            if getattr(args, key) is None:
                parser.error('--'+key.replace('_', '-')+' is required')
    if args.snapshot_step < 0 or args.snapshot_step >= STEPS:
        parser.error('--snapshot-step must be in [0, 519]')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    if args.candidate == 'greedy' and args.seed != 0:
        parser.error('C0 and nominal greedy use constant --seed 0')
    if not 1 <= args.local_horizon <= STEPS-(args.snapshot_step+1):
        parser.error('--local-horizon exceeds the remaining lifetime action budget')
    if not 0 <= args.continuation_steps <= STEPS:
        parser.error('--continuation-steps must be in [0, 520]')
    if args.seed+args.local_horizon-1 >= 2**32:
        parser.error('Sample seed base+k must fit in [0, 2**32) without wrapping')
    if not 0 < args.wall_seconds <= (120 if args.replay or not args.continuation_steps else 300):
        parser.error('Wall cap is 120 seconds for L2A/replay, 300 for a future continuation')
    if not 0 < args.readiness_seconds <= 180 or not 0 < args.query_seconds <= 30:
        parser.error('Readiness/query caps are 180/30 seconds')
    if not args.smoke and args.local_horizon != 32:
        parser.error('Frozen L2 local horizon is 32 (12 is a nested offline readout)')
    if args.candidate == 'sample' and not (args.smoke or args.replay) and not args.nominal_controls:
        parser.error('Sampled calls require --nominal-controls with all six exact C0 runs')
    if args.continuation_steps and not args.replay:
        parser.error('L2B is separately gated and requires saved H32-tape resume; launch L2A with --continuation-steps 0')
    if args.egl_device != {2: 1, 3: 0}[args.gpu]:
        parser.error('Physical GPU2 requires EGL1; physical GPU3 requires EGL0')
    args.baseline, args.out = args.baseline.resolve(), args.out.resolve()
    if args.replay:
        args.replay = args.replay.resolve()
    return args


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    recorded_args = json.loads(json.dumps({k: v for k, v in vars(args).items()
                                          if k != '_child'}, default=str))
    cap = min(args.continuation_steps, STEPS-(args.snapshot_step+1)-args.local_horizon)
    status = dict(status='started', scope='L2_native_microwave_diagnostic_branch', args=recorded_args,
        started=time.time(), pid=os.getpid(), actor_queries=0, steps_executed=0,
        candidate_id=out.name, candidate_count=1, batch_size=1, scene_resets=1,
        snapshot_convention='zero_based_post_action', baseline_prefix_actions=args.snapshot_step+1,
        initialization_actions=10, local_horizon=args.local_horizon, continuation_cap=cap,
        lifetime_action_cap=STEPS, wall_seconds=args.wall_seconds,
        readiness_seconds=args.readiness_seconds, query_seconds=args.query_seconds,
        phase_supervisor_seconds=7200, phase_supervisor_owner='external_coordinator',
        wall_scope='setup_reconstruction_and_branch_excluding_model_load',
        handoff=None, repair_verified=False, seed_schedule='greedy=0; sampled_local=base+k',
        protocol_path=str(PROTOCOL), protocol_sha256=sha(PROTOCOL),
        policy_inputs='native_agentview_RGB_and_full_original_instruction_only',
        termination=None)
    report = dict(status='started', source=str(args.baseline), snapshot_step=args.snapshot_step,
        frames_checked=0, max_state_gap=0., max_integration_gap=0., mismatches=[], actor_queries=0,
        reconstruction='fresh_native_scene_then_10_init_and_baseline_actions_0_through_t_inclusive')
    proof = dict(actor_constructor_attempts=0, actor_query_attempts=0, worker_spawn_attempts=0,
                 model_weight_open_attempts=0, actor_queries=0, subprocesses=[])
    nominal = dict(status='not_run', baseline=str(args.baseline), snapshot_step=args.snapshot_step,
        limit=32, steps_checked=0, checks=[], mismatches=[], frames_checked=0,
        max_state_gap=0., max_integration_gap=0.,
        policy_png_reference='frozen_original_preprocessing_of_original_suffix_RGB_for_comparison_only')
    save(out/'status.json', status)
    env = recorder = actor = runner = None
    resources, handles, rows = [], [], []
    initial = None
    charged = 0.
    clock = None
    result = 1
    try:
        clock = time.monotonic()
        # Setup, source validation and all reconstruction are charged to the wall budget.
        with budget(args.wall_seconds):
            previous = load(args.baseline/'status.json')
            original = argparse.Namespace(**previous['args'])
            prefix = [json.loads(line) for line in (args.baseline/'steps.jsonl').read_text().splitlines()]
            require(args.snapshot_step < len(prefix), '--snapshot-step exceeds source action length')
            require(previous['status'] == 'completed', 'Baseline must be completed')
            require(original.task == 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it',
                    'This diagnostic runner is restricted to the native microwave task')
            runner, baseline_manifest = frozen_runner(args.baseline)
            if args.smoke or args.replay:
                zero_query_guard(runner, proof)
            for name in NEW_SOURCES:
                destination = out/'branch_source'/name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes((HERE/name).read_bytes())
            status['branch_sources'] = {name: sha(HERE/name) for name in NEW_SOURCES}
            if not args.smoke and not args.replay:
                require(args.snapshot_step == SNAPSHOT_STEPS[original.init_index],
                        'Snapshot differs from the frozen root-specific boundary')
                if args.candidate == 'sample':
                    require(args.seed in (20261019+10000*original.init_index+100,
                                          20261019+10000*original.init_index+200),
                            'Sample seed base must identify the frozen C1 or C2 cell')
                    status['nominal_controls'] = validate_nominal_controls(args, status['branch_sources'])
            if args.replay:
                source_status = load(args.replay/'status.json')
                require(source_status['status'] == 'completed' and not source_status['args']['smoke'],
                        'Replay requires a completed model branch')
                require(source_status['branch_sources'] == status['branch_sources'], 'Branch source drift')
                require(source_status['protocol_sha256'] == status['protocol_sha256'], 'Frozen protocol drift')
                for name, expected in source_status['branch_sources'].items():
                    require(sha(args.replay/'branch_source'/name) == expected,
                            f'Archived branch source mismatch: {name}')
            status['baseline_device'] = dict(gpu=original.gpu, egl_device=original.egl_device)
            status['scene_seed'] = original.seed
            original.gpu, original.egl_device, original.out = args.gpu, args.egl_device, out
            base = runner.local_environment(original, out)
            identity = load(out/'device_identity.json')
            # Mask by UUID for every descendant; indices are only used for audited EGL selection.
            os.environ['CUDA_VISIBLE_DEVICES'] = identity['physical_uuid']
            status['device_identity'] = identity
            report.update(baseline_device=status['baseline_device'], actual_device=identity,
                cross_device=args.gpu != previous['args']['gpu'],
                exact_scope='all_prefix_frames_state_integration_controller_images_diagnostics_contacts')
            files = ('source_manifest.json', 'status.json', 'initial_source.json', 'model.xml',
                     'steps.jsonl', 'frames.jsonl', 'actions.npz', 'states.npz', 'images.npz',
                     'initialization/steps.jsonl', 'initialization/frames.jsonl',
                     'initialization/states.npz', 'initialization/images.npz')
            inputs = {name: sha(args.baseline/name) for name in files}
            save(out/'baseline_inputs.json', inputs)
            with np.load(args.baseline/'actions.npz', allow_pickle=False) as actions:
                for key, field in (('mapped', 'action'), ('native', 'native_proposal'), ('tokens', 'tokens')):
                    np.testing.assert_array_equal(actions[key][:args.snapshot_step+1],
                        np.asarray([r[field] for r in prefix[:args.snapshot_step+1]]).reshape(-1, 7))
            if args.replay:
                require(inputs == load(args.replay/'baseline_inputs.json'), 'Baseline tape changed after branch')
            manifest = runner.freeze_sources(out, base)
            runner.seed_all(original.seed)
            status['phase'] = 'reconstruction'
            save(out/'status.json', status)
            env, obs = runner.make_scene(original, out, base)
            require(sha(out/'model.xml') == baseline_manifest['model_xml_sha256'], 'Compiled XML mismatch')
            require(load(out/'initial_source.json') == load(args.baseline/'initial_source.json'),
                    'Initial-state provenance mismatch')
            require(env.language_instruction == previous['language'], 'Original instruction mismatch')
            runner.record_scene_sources(env, out, manifest)
            status.update(language=env.language_instruction, checkpoint_revision=runner.REVISION)
            diag = runner.Diagnostics(env)
            recorder = runner.ContactRecorder(env.sim, keywords=(*diag.relevant, 'gripper'))
            initialization = runner.load_rows(args.baseline/'initialization/steps.jsonl')
            require(len(initialization) == runner.SETTLE == 10, 'Expected ten initialization actions')
            obs = replay_segment(runner, env, obs, diag, recorder, args.baseline/'initialization',
                initialization, report, 'initialization', out, resources, handles)
            obs = replay_segment(runner, env, obs, diag, recorder, args.baseline,
                prefix[:args.snapshot_step+1], report, 'baseline', out, resources, handles)
            report.update(status='reconstruction_mismatch' if report['mismatches'] else 'exact_reconstruction_passed',
                root_fingerprint=runner.fingerprint(env, obs), baseline_frame=args.snapshot_step+1,
                baseline_prefix_actions=args.snapshot_step+1)
            save(out/'reconstruction.json', report)
            save(out/'verification.json', report)
            if report['mismatches']:
                raise ValueError('Exact reconstruction failed; see raw mismatches and reconstruction.json')
            initial = diag.measure(obs)
            status.update(initial_diagnostics=initial, initial=runner.fingerprint(env, obs))
            save(out/'snapshot_controller.json', controller_fields(env))
            if args.replay:
                branch_rows = runner.load_rows(args.replay/'steps.jsonl')
                validate_branch_tape(runner, args, source_status, branch_rows)
                source_tape = runner.ReplayTape(args.replay)
                require(len(source_tape.frames) == len(branch_rows)+1, 'Uncommitted branch frames')
                del source_tape
                obs = replay_segment(runner, env, obs, diag, recorder, args.replay, branch_rows,
                    report, 'branch', out, resources, handles, args.snapshot_step+1)
                rows = branch_rows
                report.update(status='replay_mismatch' if report['mismatches'] else 'exact_replay_passed',
                    branch_steps=len(branch_rows), branch_source=str(args.replay))
                save(out/'verification.json', report)
                require(not report['mismatches'], 'Branch replay mismatch; raw evidence preserved')
                status.update(status='completed', termination='exact_replay_passed', replayed_steps=len(branch_rows))
            elif args.smoke:
                status.update(status='completed', termination='exact_reconstruction_smoke_passed')
            else:
                # Even an already-successful root has a complete zero-action branch tape.
                tape = runner.Tape(out)
                resources.append(('branch_tape', tape))
                tape.capture(env, obs, initial)
                stream, queries = (out/'steps.jsonl').open('w'), (out/'queries.jsonl').open('w')
                handles.extend([stream, queries])
                (out/'policy_inputs').mkdir()
                if initial['native_success']:
                    status.update(status='completed', termination='native_success')
        charged += time.monotonic()-clock
        clock = None
        if status['status'] != 'completed':
            # Worker readiness has its own FD deadline; the coordinator owns the phase watchdog.
            status['phase'] = 'model_load'
            save(out/'status.json', status)
            start = time.monotonic()
            validate_model_sources(runner, original, out, manifest, baseline_manifest)
            actor = CandidateActor.__new__(CandidateActor)
            CandidateActor.__init__(actor, runner, original, args, out, identity)
            status.update(model_load_s=time.monotonic()-start, actor_build=actor.build)
            clock = time.monotonic()
            with budget(args.wall_seconds-charged):
                from adapters.libero_openvla import policy_image
                reference = evidence_tape(runner, args.baseline, out) if args.candidate == 'greedy' else None
                if reference is not None:
                    nominal['status'] = 'in_progress'
                status['phase'] = 'branch'
                for index in range(args.local_horizon+cap):
                    step = args.snapshot_step+1+index
                    local = index < args.local_horizon
                    source = args.candidate if local else 'greedy'
                    seed = request_seed(args, index)
                    owner = 'local_candidate' if local else 'nominal_vla'
                    image = policy_image(obs['agentview_image'])
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    png = buffer.getvalue()
                    (out/'policy_inputs'/f'{index:06d}.png').write_bytes(png)
                    metadata = dict(step=step, branch_step=index, candidate_id=out.name,
                        candidate_owner=owner, branch_phase='local' if local else 'continuation',
                        source=source, seed=seed, query_index=index,
                        policy_png_sha256=hashlib.sha256(png).hexdigest(),
                        policy_rgb_sha256=runner.array_sha(np.asarray(image)),
                        raw_rgb_sha256=runner.array_sha(obs['agentview_image']),
                        language=env.language_instruction, before=tape.last['fingerprint'],
                        checkpoint_revision=runner.REVISION, worker_pid=actor.process.pid,
                        requested_at=time.time(), episode_elapsed_at_request_s=charged+time.monotonic()-clock)
                    start = time.monotonic()
                    runner.write_row(queries, dict(event='requested', **metadata))
                    status['actor_queries'] += 1
                    save(out/'status.json', status)
                    decode = actor.query(png, env.language_instruction, source, seed, index)
                    inference_s = time.monotonic()-start
                    # Preserve the unmodified response, including invalid/nonfinite values,
                    # before any validation. Such a response is never executed or redrawn.
                    with (out/'worker_responses.jsonl').open('a') as raw:
                        raw.write(json.dumps(dict(query_index=index, response=decode))+'\n')
                        raw.flush()
                        os.fsync(raw.fileno())
                    validate_decode(decode, metadata)
                    if not decode.get('valid'):
                        raise InvalidGeneration(decode.get('invalid_reason', 'worker_returned_invalid_generation'))
                    extra = dict(**metadata, proposal=decode['action'], native_proposal=decode['native_action'],
                        tokens=decode['tokens'], decode=decode, inference_s=inference_s,
                        forced_diagnostic_handoff=index == args.local_horizon,
                        repair_verified=False, episode_elapsed_s=charged+time.monotonic()-clock)
                    try:
                        validate_action(dict(**extra, action=decode['action'],
                                             executed_mapped_action=decode['action']), step)
                    except (ValueError, AssertionError, TypeError, KeyError) as error:
                        raise InvalidGeneration(str(error)) from error
                    runner.write_row(queries, dict(event='returned', **extra))
                    if reference is not None and index < 32:
                        nominal_query_check(runner, reference, prefix, metadata, decode, png, out, nominal)
                    obs, row = runner.step_once(env, obs, decode['action'], diag, recorder, tape,
                        stream, step, 'actor', {k: v for k, v in extra.items() if k not in ('step', 'before')})
                    rows.append(row)
                    status['steps_executed'] = len(rows)
                    if reference is not None and index < 32:
                        nominal_step_check(reference, prefix[step], row, env, obs, out, nominal)
                    if row['forced_diagnostic_handoff']:
                        status['handoff'] = dict(step=step, branch_step=index, executed=True,
                            action=row['action'], query_index=index, kind='forced_diagnostic', repair_verified=False)
                    save(out/'status.json', status)
                    if row['diagnostics']['native_success']:
                        status['termination'] = 'native_success'
                        break
                    require(not row['done'], 'Environment done before native success or lifetime budget')
                status.update(status='completed', termination=status.get('termination') or 'action_budget')
            charged += time.monotonic()-clock
            clock = None
        result = 0
    except BudgetExpired:
        status.update(status='censored', termination='wall_budget', interrupted_phase=status.get('phase'))
        report.update(status='censored', termination='wall_budget')
    except WorkerTimeout as error:
        status.update(status='timed_out', termination=error.phase+'_timeout',
                      interrupted_phase=status.get('phase'), error=str(error))
    except InvalidGeneration as error:
        status.update(status='invalid_generation', termination='invalid_generation', error=str(error))
    except NominalMismatch as error:
        status.update(status='nominal_mismatch', termination='nominal_identity_mismatch', error=str(error))
    except (KeyboardInterrupt, SystemExit):
        status.update(status='censored', termination='interrupted', interrupted_phase=status.get('phase'),
                      error=traceback.format_exc())
    except BaseException:
        status.update(status='infrastructure_failed', termination='infrastructure_failure',
            failed_phase=status.get('phase'), error=traceback.format_exc())
        report.update(status='infrastructure_failed' if not report['mismatches'] else 'replay_mismatch',
                      error=status['error'])
    finally:
        if clock is not None:
            charged += time.monotonic()-clock
        status['episode_wall_s'] = charged
        if args.candidate == 'greedy' and not args.replay:
            nominal['status'] = ('not_run_zero_queries' if args.smoke else 'nominal_identity_mismatch'
                if nominal['mismatches'] else 'exact_nominal_identity_passed'
                if status['status'] == 'completed' and (nominal['steps_checked'] == 32 or
                    status['termination'] == 'native_success') else 'censored_unestablished')
            nominal['branch_status'] = status['status']
            save(out/'nominal_identity.json', nominal)
        if initial is not None and runner is not None:
            status.update(runner.summarize(initial, rows))
        errors = []
        operations = []
        if runner is not None:
            operations.append(('actor', lambda: runner.cleanup_actor(actor)))
        operations += [('stream', handle.close) for handle in handles]
        operations += [(name, resource.close) for name, resource in resources]
        operations += [(name, resource.close) for name, resource in
                       [('contacts', recorder), ('environment', env)] if resource is not None]
        for name, operation in operations:
            try:
                operation()
            except BaseException:
                errors.append(dict(resource=name, error=traceback.format_exc()))
        if not (args.smoke or args.replay):
            np.savez_compressed(out/'actions.npz',
                mapped=np.asarray([r['action'] for r in rows], dtype=float).reshape(-1, 7),
                native=np.asarray([r['native_proposal'] for r in rows], dtype=float).reshape(-1, 7),
                tokens=np.asarray([r['tokens'] for r in rows], dtype=np.int64).reshape(-1, 7))
        if args.smoke or args.replay:
            proof['actor_queries'] = status['actor_queries']
            save(out/'zero_query_proof.json', proof)
        if errors:
            status.update(status='infrastructure_failed', cleanup_errors=errors)
            result = 1
            if args.candidate == 'greedy' and not args.replay:
                nominal.update(status='censored_unestablished', branch_status=status['status'])
                save(out/'nominal_identity.json', nominal)
        status.update(ended=time.time(), total_wall_s=time.time()-status['started'],
            lifetime_actions=args.snapshot_step+1+len(rows),
            partial_outputs=status['status'] != 'completed' or status['termination'] == 'wall_budget')
        save(out/'verification.json', report)
        save(out/'status.json', status)
    print(json.dumps(dict(status=status['status'], termination=status['termination'], out=str(out),
        actor_queries=status['actor_queries'], steps_executed=status['steps_executed'],
        frames_checked=report['frames_checked'], mismatches=len(report['mismatches']))), flush=True)
    return result


def validate_branch_tape(runner, args, status, rows):
    from PIL import Image
    require(len(rows) == status['steps_executed'] <= args.local_horizon+status['continuation_cap'],
            'Branch action count exceeds its budget')
    require(status['continuation_cap'] == min(args.continuation_steps,
            STEPS-(args.snapshot_step+1)-args.local_horizon), 'Continuation cap mismatch')
    require(args.snapshot_step+1+len(rows) <= STEPS, 'Lifetime action budget exceeded')
    queries = runner.load_rows(args.replay/'queries.jsonl')
    returned = {r['query_index']: r for r in queries if r['event'] == 'returned'}
    requested = {r['query_index']: r for r in queries if r['event'] == 'requested'}
    require(len(returned)+len(requested) == len(queries), 'Duplicate or unknown query events')
    require(len(requested) == status['actor_queries'], 'Query accounting mismatch')
    require(len(rows) == len(requested) == len(returned),
            'Completed branch must have one requested/returned query per executed action')
    if rows:
        build = status['actor_build']
        require(build['transformers'] == '4.40.1' and build['unnorm_key'] == 'libero_10'
            and build['H'] == build['batch'] == 1 and build['frozen'] and build['official_predict_action'],
            'Replay actor provenance mismatch')
        require(build['generation_kwargs'] == {s: generation_kwargs(s) for s in ('greedy', 'sample')},
                'Replay actor decode defaults mismatch')
    with np.load(args.replay/'actions.npz', allow_pickle=False) as tape:
        for key, field in (('mapped', 'action'), ('native', 'native_proposal'), ('tokens', 'tokens')):
            np.testing.assert_array_equal(tape[key], np.asarray([r[field] for r in rows]).reshape(-1, 7))
    for index, row in enumerate(rows):
        local = index < args.local_horizon
        require(row['branch_step'] == row['query_index'] == index, 'Branch/query index mismatch')
        require(row['source'] == (args.candidate if local else 'greedy'), 'Incorrect branch policy')
        require(row['owner'] == 'vla' and row['candidate_owner'] ==
                ('local_candidate' if local else 'nominal_vla'), 'Incorrect action owner')
        require(row['phase'] == 'actor' and row['branch_phase'] ==
                ('local' if local else 'continuation'), 'Incorrect action phase')
        require(row['seed'] == request_seed(args, index), 'Per-step seed mismatch')
        require(row['forced_diagnostic_handoff'] == (index == args.local_horizon)
                and row['repair_verified'] is False, 'Incorrect handoff claim')
        require(row['candidate_id'] == status['candidate_id'], 'Candidate ownership mismatch')
        require(index in returned and index in requested, 'Action without fresh query provenance')
        for key in returned[index].keys()-{'event'}:
            require(row[key] == returned[index][key], f'Query/action provenance mismatch: {key}')
        for key in requested[index].keys()-{'event'}:
            require(row[key] == requested[index][key], f'Query intent mismatch: {key}')
        require(sha(args.replay/'policy_inputs'/f'{index:06d}.png') == row['policy_png_sha256'],
                'Saved policy input hash mismatch')
        with Image.open(args.replay/'policy_inputs'/f'{index:06d}.png') as image:
            require(runner.array_sha(np.asarray(image.convert('RGB'))) == row['policy_rgb_sha256'],
                    'Saved policy RGB hash mismatch')
        require(row['raw_rgb_sha256'] == row['before']['images']['agentview_image'],
                'Policy raw RGB differs from the pre-action observation')
        require(row['language'] == status['language'] and row['checkpoint_revision'] == runner.REVISION,
                'Policy language/checkpoint provenance mismatch')
        decode = row['decode']
        require(decode['valid'] is True, 'Invalid generation in executed tape')
        require(decode['action'] == row['action'] and decode['native_action'] == row['native_proposal']
                and decode['tokens'] == row['tokens'], 'Worker decode/action mismatch')
        validate_decode(decode, row)
    if args.candidate == 'greedy':
        identity = load(args.replay/'nominal_identity.json')
        require(identity['status'] == 'exact_nominal_identity_passed' and not identity['mismatches']
            and identity['steps_checked'] == min(len(rows), 32), 'Nominal identity proof missing or incomplete')
    handoffs = [r for r in rows if r['forced_diagnostic_handoff']]
    require(len(handoffs) == int(len(rows) > args.local_horizon), 'Executed handoff count mismatch')
    if handoffs:
        first = handoffs[0]
        require(status['handoff'] == dict(step=first['step'], branch_step=first['branch_step'],
            executed=True, action=first['action'], query_index=first['query_index'],
            kind='forced_diagnostic', repair_verified=False), 'Handoff summary mismatch')
    else:
        require(status['handoff'] is None, 'Handoff claimed without executed continuation')


class BudgetExpired(BaseException):
    pass


class budget:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds <= 0:
            raise BudgetExpired('Charged wall budget exhausted')
        def expired(signum, frame):
            raise BudgetExpired('Charged wall budget exhausted')
        self.previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


def supervised(args):
    """Fresh child and descendant cleanup; the coordinator owns the 7200s phase watchdog."""
    require(not args.out.exists(), f'Output already exists: {args.out}')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
        PYTHONOPTIMIZE='0')
    child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
        *sys.argv[1:], '--_child'], env=env, start_new_session=True)
    try:
        return child.wait()
    except (KeyboardInterrupt, SystemExit):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        if args.out.exists():
            save(args.out/'supervisor_interruption.json', dict(status='censored',
                termination='external_supervisor_interruption', child_pid=child.pid, ended=time.time()))
        return 1
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    arguments = parse_args()
    raise SystemExit(run(arguments) if arguments._child else supervised(arguments))
