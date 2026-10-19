"""Frozen, bounded native microwave L2A supervisor (no selector or continuation).

Prepare with --prepare NEW_DIR, inspect/commit the implementation and plan, then
explicitly --launch --plan NEW_DIR/plan.json --out DISTINCT_NEW_DIR.  Launch is
foreground and nohup/setsid compatible; stdin is never used. --validate-only
does not write files, construct a scene, query a model, or access a GPU.
"""
import argparse
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent
DESIGN = Path('/home/ami/scratch/native_microwave_l2_design_20261019')
PROTOCOL = REPO/'exp_record/agent_notes/native_microwave_l2_protocol_20261019.md'
PROTOCOL_ID = 'native_microwave_l2_v1_20261019'
FREEZE_SHA = 'c1fb0739a51d07e73a4ba6a861771b31beb4a5d51aa46e1745382fda284c9f49'
LANES = [dict(gpu=2, egl_device=1,
              physical_uuid='GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8'),
         dict(gpu=3, egl_device=0,
              physical_uuid='GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b')]
LIMITS = dict(cells=18, calls=576, local_horizon=32, continuation_steps=0,
              parallel=2, batch=1, readiness_seconds=180, query_seconds=30,
              wall_seconds=120, child_outer_seconds=480, replay_outer_seconds=150,
              phase_seconds=7200, retries=0)
SOURCES = ('scripts/run_native_microwave_l2.py', 'scripts/run_native_long_branch.py',
           'scripts/native_long_candidate_worker.py')
BASELINE_FILES = ('source_manifest.json', 'status.json', 'initial_source.json',
                  'selected_initial_state.npy', 'scene.bddl', 'model.xml',
                  'steps.jsonl', 'frames.jsonl', 'queries.jsonl', 'actions.npz',
                  'states.npz', 'images.npz', 'initialization/steps.jsonl',
                  'initialization/frames.jsonl', 'initialization/states.npz',
                  'initialization/images.npz')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rows(path, partial=False):
    with Path(path).open() as stream:
        for line in stream:
            # Only a final uncommitted line may be ignored during live monitoring.
            if partial and not line.endswith('\n'):
                return
            yield json.loads(line)


def row_at(path, index):
    for position, row in enumerate(rows(path)):
        if position == index:
            return row
    raise ValueError(f'Missing row {index}: {path}')


def environment(lane):
    return dict(CUDA_VISIBLE_DEVICES=lane['physical_uuid'],
                NVIDIA_VISIBLE_DEVICES=lane['physical_uuid'],
                MUJOCO_EGL_DEVICE_ID=str(lane['egl_device']), MUJOCO_GL='egl',
                PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1',
                PYTHONOPTIMIZE='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
                TOKENIZERS_PARALLELISM='false')


def identities(snapshots):
    result = []
    for candidate in range(3):
        for snapshot in snapshots:
            root = snapshot['root']
            seed = 0 if candidate == 0 else 20261019+10000*root+100*candidate
            lane = LANES[root % 2]
            result.append(dict(name=f'C{candidate}_root{root}', root=root,
                candidate=f'C{candidate}', source='greedy' if candidate == 0 else 'sample',
                baseline=snapshot['source'], snapshot_step=snapshot['post_action_step'],
                state_index=snapshot['state_index'], fingerprint=snapshot['fingerprint'],
                seed=seed, seeds=[0 if candidate == 0 else seed+k for k in range(32)],
                lane=deepcopy(lane), environment=environment(lane), batch=1,
                local_horizon=32, continuation_steps=0))
    return result


def boundary_hashes(snapshots, python):
    """CPU-only original preprocessing, isolated from supervisor/scene imports.

    Execute just the pinned policy_image function, avoiding its module's torch
    import. The subprocess has no CUDA devices and disables TF's GPU backend.
    No images or caches are written; return only hashes of the original inputs.
    """
    program = r'''
import ast, hashlib, io, json, os, sys
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from pathlib import Path
snapshots = json.loads(sys.stdin.read())
result = []
for snapshot in snapshots:
    base = Path(snapshot['source'])
    tree = ast.parse((base/'source/src/adapters/libero_openvla.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'policy_image')
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen-policy-image>', 'exec'))
    with np.load(base/'frames'/f"{snapshot['state_index']:06d}.npz", allow_pickle=False) as frame:
        raw = frame['agentview_image']
        image = policy_image(raw)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    result.append(dict(root=snapshot['root'], policy_png_sha256=hashlib.sha256(buffer.getvalue()).hexdigest()))
print(json.dumps(result))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='none',
               PYTHONDONTWRITEBYTECODE='1', TF_CPP_MIN_LOG_LEVEL='3',
               TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2',
               OPENBLAS_NUM_THREADS='1')
    completed = subprocess.run([python, '-B', '-c', program], input=json.dumps(snapshots),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        timeout=120, check=True)
    return json.loads(completed.stdout)


def build_plan():
    frozen, weights = {}, {}

    def pin(path, expected=None):
        path = Path(path).absolute()
        value = frozen.get(str(path)) or sha(path)
        require(expected is None or value == expected, f'Frozen source drift: {path}')
        frozen[str(path)] = value

    freeze_path = DESIGN/'design_freeze.json'
    pin(freeze_path, FREEZE_SHA)
    freeze = load(freeze_path)
    require(freeze['protocol'] == PROTOCOL_ID, 'Wrong design protocol')
    for path, expected in freeze['files'].items():
        pin(path, expected)
    manifest = load(DESIGN/'snapshot_manifest.json')
    require(manifest['protocol'] == PROTOCOL_ID and manifest['local_horizon'] == 32
            and manifest['budgets']['local_model_calls'] == 576, 'Wrong L2A design')
    snapshots = manifest['snapshots']
    require([s['root'] for s in snapshots] == list(range(6)), 'Expected six ordered roots')
    require([s['post_action_step'] for s in snapshots] == [253, 266, 275, 316, 275, 272],
            'Frozen boundaries changed')
    for name in SOURCES:
        pin(REPO/name)
    python = None
    for snapshot in snapshots:
        base = Path(snapshot['source'])
        for name in BASELINE_FILES:
            pin(base/name, snapshot['source_sha256'].get(name))
        pin(base/'frames'/f"{snapshot['state_index']:06d}.npz")
        status = load(base/'status.json')
        require(status['status'] == 'completed' and status['args']['seed'] == 0
                and status['args']['init_index'] == snapshot['root']
                and status['language'] == manifest['instruction'], 'Baseline identity mismatch')
        s = snapshot['state_index']
        require(s == snapshot['post_action_step']+1 and 520-s == snapshot['remaining_actions'],
                'Boundary indexing mismatch')
        frame = row_at(base/'frames.jsonl', s)
        require(frame['fingerprint'] == snapshot['fingerprint']
                == row_at(base/'steps.jsonl', s-1)['after']
                == row_at(base/'steps.jsonl', s)['before'], 'Boundary fingerprint mismatch')
        sources = load(base/'source_manifest.json')
        initial = load(base/'initial_source.json')
        pin(initial['path'], initial['sha256'])
        pin(initial['bddl_path'], initial['bddl_sha256'])
        require(sources['checkpoint']['revision'] == manifest['checkpoint_revision'],
                'Checkpoint revision mismatch')
        executable = sources['executable']
        require(python is None or python == executable, 'Mixed scene interpreters')
        python = executable
        pin(python)
        pin(status['args']['pinned_python'])
        # Both mutable repository copies and the actually imported baseline
        # archive must match the original native implementation.
        for name, entry in sources['repository'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                    'Unsafe archived source name')
            pin(base/'source'/name, entry['sha256'])
            pin(REPO/name, entry['sha256'])
        for section in ('runtime', 'assets', 'actor_code'):
            for path, expected in sources[section].items():
                pin(path, expected)
                if section == 'actor_code':
                    pin(base/'source/actor_code'/Path(path).name, expected)
                elif section == 'runtime':
                    if '/robosuite/' in path:
                        relative = Path('robosuite')/path.split('/robosuite/', 1)[1]
                    else:
                        relative = Path('runtime_libero')/path.split('/libero/libero/', 1)[1]
                    pin(base/'source'/relative, expected)
        for entry in sources['checkpoint']['files'].values():
            path = Path(entry['path'])
            require(str(path.resolve()) == entry['resolved_path']
                    and path.stat().st_size == entry['bytes'], f'Checkpoint file drift: {path}')
            if 'sha256' in entry:
                pin(path, entry['sha256'])
            else:
                # Preserve the original explicit non-rehashed weight provenance;
                # never present a cache blob identifier as a freshly computed hash.
                weights[str(path)] = dict(entry, stat_mtime_ns=path.stat().st_mtime_ns)
    prompt = f"In: What action should the robot take to {manifest['instruction'].lower()}?\nOut:"
    inputs = boundary_hashes(snapshots, python)
    for value in inputs:
        value.update(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    return dict(schema_version=1, protocol=PROTOCOL_ID, scope='L2A_only',
        repo=str(REPO), python=python, limits=deepcopy(LIMITS), lanes=deepcopy(LANES),
        phase_order=['C0', 'C1', 'C2'], snapshots=snapshots, cells=identities(snapshots),
        boundary_inputs=inputs, sampling=manifest['sampling'],
        checkpoint_revision=manifest['checkpoint_revision'],
        base_code_revision=manifest['base_code_revision'], instruction=manifest['instruction'],
        frozen_files_sha256=frozen, checkpoint_weight_metadata=weights,
        archive_sources={name: frozen[str(REPO/name)] for name in SOURCES},
        design_freeze_sha256=frozen[str(freeze_path)],
        snapshot_manifest_sha256=frozen[str(DESIGN/'snapshot_manifest.json')],
        protocol_sha256=frozen[str(PROTOCOL)])


def check_pins(plan, plan_path=None):
    for path, expected in plan['frozen_files_sha256'].items():
        require(sha(path) == expected, f'Frozen source drift: {path}')
    for path, entry in plan['checkpoint_weight_metadata'].items():
        file = Path(path)
        require(str(file.resolve()) == entry['resolved_path']
                and file.stat().st_size == entry['bytes']
                and file.stat().st_mtime_ns == entry['stat_mtime_ns'],
                f'Checkpoint weight metadata drift: {path}')
    if plan_path is not None:
        for name, expected in plan['archive_sources'].items():
            require(sha(plan_path.parent/'source'/name) == expected,
                    f'Prepared source archive drift: {name}')


def validate_plan(plan, plan_path):
    require(plan['schema_version'] == 1 and plan['protocol'] == PROTOCOL_ID
            and plan['scope'] == 'L2A_only' and plan['repo'] == str(REPO), 'Wrong plan')
    require(plan['limits'] == LIMITS and plan['lanes'] == LANES
            and plan['phase_order'] == ['C0', 'C1', 'C2'], 'Plan limits/order/device drift')
    require(plan['cells'] == identities(plan['snapshots']), 'Frozen cell identities changed')
    require(plan['archive_sources'] == {name: plan['frozen_files_sha256'][str(REPO/name)]
                                      for name in SOURCES}, 'Archive/live pins disagree')
    require(plan['design_freeze_sha256'] == FREEZE_SHA, 'Wrong frozen design')
    check_pins(plan, plan_path)
    original = load(DESIGN/'snapshot_manifest.json')
    require(plan['snapshots'] == original['snapshots'], 'Snapshot plan drift')
    for key in ('sampling', 'checkpoint_revision', 'base_code_revision', 'instruction'):
        require(plan[key] == original[key], f'Design drift: {key}')
    # Rebuild the complete read-only inventory: deleting a pin is not a way to
    # bypass source verification. Also independently recheck CPU input hashes.
    require(plan == build_plan(), 'Prepared plan differs from frozen inputs/implementation')


def prepare(directory):
    require(not directory.exists(), f'Preparation directory already exists: {directory}')
    plan = build_plan()
    directory.mkdir(parents=True, exist_ok=False)
    for name, expected in plan['archive_sources'].items():
        destination = directory/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO/name).read_bytes())
        require(sha(destination) == expected, f'Source changed while archiving: {name}')
    save(directory/'plan.json', plan)
    check_pins(plan, directory/'plan.json')
    return directory/'plan.json'


def commands(plan, cell, out):
    branch = out/cell['name']
    replay = out/('replay_'+cell['name'])
    prefix = [plan['python'], '-B', '-u', str(REPO/'scripts/run_native_long_branch.py')]
    command = prefix + ['--baseline', cell['baseline'], '--snapshot-step', str(cell['snapshot_step']),
        '--candidate', cell['source'], '--seed', str(cell['seed']), '--local-horizon', '32',
        '--continuation-steps', '0', '--gpu', str(cell['lane']['gpu']), '--egl-device',
        str(cell['lane']['egl_device']), '--wall-seconds', '120', '--readiness-seconds', '180',
        '--query-seconds', '30', '--out', str(branch)]
    if cell['candidate'] != 'C0':
        command += ['--nominal-controls', *[str(out/f'C0_root{root}') for root in range(6)]]
    replay_command = prefix + ['--replay', str(branch), '--out', str(replay), '--wall-seconds', '120']
    return command, replay_command


def counts(directory, final=False):
    status_path, queries_path = directory/'status.json', directory/'queries.jsonl'
    status = load(status_path) if status_path.exists() else {}
    requested = list(r for r in rows(queries_path, partial=not final)
                     if r.get('event') == 'requested') if queries_path.exists() else []
    reported = status.get('actor_queries', 0)
    require(type(reported) is int and reported >= 0, 'Invalid final/request counter')
    if final:
        require(reported == len(requested), 'Final/request call counts disagree')
        require([r['query_index'] for r in requested] == list(range(len(requested))),
                'Repeated or missing query identity')
    return max(reported, len(requested))


def budget_available(cells, out, new_jobs=0):
    total = 0
    reserved = 0
    for cell in cells:
        used = counts(out/cell['name'])
        require(used <= 32, f"Per-cell call cap exceeded: {cell['name']}")
        total += used
        if cell['status'] == 'branch_running':
            reserved += 32-used
    require(total+reserved+32*new_jobs <= 576, 'Aggregate call reservation exceeds 576')
    return total


def exact_report(report, status):
    require(report.get('status') == status and report.get('mismatches') == [],
            f'Expected {status} with an explicit empty mismatch list')
    require(report.get('max_state_gap') == 0 and report.get('max_integration_gap') == 0,
            'Nonzero or missing exact replay gaps')
    require(type(report.get('frames_checked')) is int and report['frames_checked'] > 0,
            'No frames independently checked')


def check_branch(plan, cell, directory):
    status = load(directory/'status.json')
    require(status['status'] == 'completed' and status['termination'] in ('action_budget', 'native_success')
            and status.get('partial_outputs') is False, 'Incomplete/invalid/infrastructure branch')
    n = counts(directory, final=True)
    require(0 < n <= 32 and status['steps_executed'] == n, 'Branch action/query count mismatch')
    require(status['termination'] != 'action_budget' or n == 32, 'Truncated local horizon')
    expected_args = dict(baseline=cell['baseline'], snapshot_step=cell['snapshot_step'],
        candidate=cell['source'], seed=cell['seed'], local_horizon=32, continuation_steps=0,
        gpu=cell['lane']['gpu'], egl_device=cell['lane']['egl_device'], wall_seconds=120,
        readiness_seconds=180, query_seconds=30, out=str(directory))
    require(all(status['args'].get(k) == v for k, v in expected_args.items()), 'Branch CLI identity drift')
    require(status['batch_size'] == 1 and status['continuation_cap'] == 0, 'Wrong branch/batch scope')
    identity = load(directory/'device_identity.json')
    require(identity['physical_gpu'] == cell['lane']['gpu']
            and identity['physical_uuid'] == cell['lane']['physical_uuid']
            and identity['egl_device'] == cell['lane']['egl_device']
            and identity['cuda_visible_devices'] == cell['lane']['physical_uuid'], 'Wrong branch GPU')
    build = status['actor_build']
    require(build['batch'] == 1 and build['H'] == 1 and build['transformers'] == '4.40.1'
            and build['device_identity']['physical_uuid'] == cell['lane']['physical_uuid'],
            'Worker build/device mismatch')
    require(status['initial'] == cell['fingerprint'], 'Reconstructed boundary mismatch')
    require(status['branch_sources'] == {Path(name).name: plan['archive_sources'][name]
            for name in SOURCES if name != 'scripts/run_native_microwave_l2.py'},
            'Executed branch source hashes differ from prepared plan')
    require(status['protocol_sha256'] == plan['protocol_sha256'], 'Executed protocol hash drift')
    reconstruction = load(directory/'reconstruction.json')
    exact_report(reconstruction, 'exact_reconstruction_passed')
    inputs = plan['boundary_inputs'][cell['root']]
    require(sha(directory/'policy_inputs/000000.png') == inputs['policy_png_sha256'],
            'First policy PNG differs from frozen original boundary')
    for request in rows(directory/'queries.jsonl'):
        if request['event'] == 'requested':
            k = request['query_index']
            require(request['seed'] == cell['seeds'][k] and request['source'] == cell['source']
                    and request['language'] == plan['instruction'], 'Query seed/source/instruction drift')
    if cell['candidate'] == 'C0':
        identity_report = load(directory/'nominal_identity.json')
        exact_report(identity_report, 'exact_nominal_identity_passed')
        require(identity_report.get('steps_checked') == n
                and identity_report.get('baseline') == cell['baseline']
                and identity_report.get('snapshot_step') == cell['snapshot_step'],
                'Greedy nominal identity failed')
    return n


def check_replay(cell, branch, replay):
    status = load(replay/'status.json')
    verification = load(replay/'verification.json')
    require(status['status'] == 'completed' and status['termination'] == 'exact_replay_passed'
            and status['actor_queries'] == 0 and status.get('partial_outputs') is False,
            'Replay incomplete or queried actor')
    exact_report(verification, 'exact_replay_passed')
    require(verification['source'] == cell['baseline'] and verification['branch_source'] == str(branch)
            and verification['branch_steps'] == load(branch/'status.json')['steps_executed'],
            'Replay source/action identity mismatch')
    require(verification['frames_checked'] >= 11+cell['state_index']+1+verification['branch_steps']+1,
            'Incomplete initialization/prefix/local replay coverage')
    proof = load(replay/'zero_query_proof.json')
    for key in ('actor_constructor_attempts', 'actor_query_attempts', 'worker_spawn_attempts',
                'model_weight_open_attempts', 'actor_queries'):
        require(proof.get(key) == 0, f'Replay zero-query proof failed: {key}')
    require(isinstance(proof.get('subprocesses'), list)
            and not any('worker.py' in call for call in proof['subprocesses']),
            'Replay spawned a model worker')
    identity = load(replay/'device_identity.json')
    require(identity['physical_uuid'] == cell['lane']['physical_uuid']
            and identity['physical_gpu'] == cell['lane']['gpu']
            and identity['egl_device'] == cell['lane']['egl_device'], 'Replay device drift')


def controls_passed(cells):
    controls = [c for c in cells if c['candidate'] == 'C0']
    return (len(cells) == 18 and len(controls) == 6
            and {c['root'] for c in controls} == set(range(6))
            and all(c['status'] == 'exact_verified' for c in controls))


def pin_result(cell, out):
    files = ['status.json', 'steps.jsonl', 'queries.jsonl', 'actions.npz', 'frames.jsonl',
             'states.npz', 'images.npz', 'reconstruction.json', 'device_identity.json',
             'policy_inputs/000000.png']
    if cell['candidate'] == 'C0':
        files.append('nominal_identity.json')
    paths = [out/cell['name']/name for name in files]
    paths += [out/('replay_'+cell['name'])/name for name in
              ('status.json', 'verification.json', 'zero_query_proof.json', 'device_identity.json')]
    cell['artifact_sha256'] = {str(path): sha(path) for path in paths}


def check_result_pins(cells):
    for cell in cells:
        if cell['status'] == 'exact_verified':
            require(bool(cell.get('artifact_sha256')), 'Missing verified artifact pins')
            for path, expected in cell['artifact_sha256'].items():
                require(sha(path) == expected, f'Verified result drift: {path}')


def process_table():
    table = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            fields = text[text.rfind(')')+2:].split()
            table[int(path.parent.name)] = (int(fields[1]), int(fields[2]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return table


class Children:
    """Track detached grandchildren as well as Popen groups, including after exit.

    Linux subreaping keeps a branch's double-forked/session-owning model worker
    reachable even when its immediate parent dies between monitor iterations.
    PID start ticks prevent killing an unrelated reused PID.
    """
    def __init__(self):
        require(sys.platform == 'linux', 'Linux process-tree supervision required')
        require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0,
                'Cannot enable child subreaper')
        self.known = {}
        self.jobs = []

    def scan(self):
        table = process_table()
        parents = {os.getpid()} | {pid for pid, start in self.known.items()
                                  if pid in table and table[pid][2] == start}
        changed = True
        while changed:
            changed = False
            for pid, (ppid, group, start) in table.items():
                if ppid in parents and pid not in parents:
                    parents.add(pid)
                    self.known[pid] = start
                    changed = True
        return table

    def start(self, command, env, log_path):
        log = log_path.open('x')
        try:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **env),
                start_new_session=True)
        finally:
            log.close()
        self.jobs.append(process)
        self.scan()
        return process

    def cleanup(self):
        self.scan()
        # Stop spawning before discovering and killing the final descendants.
        for sig in (signal.SIGSTOP, signal.SIGKILL):
            table = self.scan()
            groups = set()
            for pid, start in self.known.items():
                if pid not in table or table[pid][2] != start:
                    continue
                group = table[pid][1]
                if group != os.getpgrp():
                    groups.add(group)
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            for group in groups:
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        for process in self.jobs:
            process.wait(timeout=5)
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                time.sleep(.05)
        raise RuntimeError('Descendants did not exit within cleanup deadline')


class Supervisor:
    def __init__(self, plan, plan_path, out):
        self.plan, self.plan_path, self.out = plan, plan_path, out
        self.state = dict(plan=deepcopy(plan), plan_path=str(plan_path), plan_sha256=sha(plan_path),
            status='started', phase=None, started=time.time(), pid=os.getpid(),
            cells=[dict(c, status='planned') for c in deepcopy(plan['cells'])],
            actor_queries=0, sampled_stage_enabled=False)
        self.children = Children()
        self.deadline = None

    def write(self):
        self.state['updated'] = time.time()
        self.state['actor_queries'] = budget_available(self.state['cells'], self.out)
        save(self.out/'status.json', self.state)
        save(self.out/'launch.json', self.state)

    def pair(self, cells, replay=False):
        self.children = Children()
        jobs = []
        try:
            check_pins(self.plan, self.plan_path)
            check_result_pins(self.state['cells'])
            require(sha(self.plan_path) == self.state['plan_sha256'], 'Plan file changed during launch')
            budget_available(self.state['cells'], self.out, new_jobs=0 if replay else len(cells))
            for cell in cells:
                require(time.monotonic() < self.deadline, 'Phase watchdog expired')
                if not replay and cell['candidate'] != 'C0':
                    require(controls_passed(self.state['cells']), 'All six C0 controls must pass before sampling')
                command, replay_command = commands(self.plan, cell, self.out)
                kind = 'replay' if replay else 'branch'
                cell.update(status=kind+'_running')
                cell[kind+'_command'] = replay_command if replay else command
                cell[kind+'_started'] = time.time()
                self.write()
                process = self.children.start(cell[kind+'_command'], cell['environment'],
                    self.out/(kind+'_'+cell['name']+'.log'))
                cell[kind+'_pid'] = process.pid
                jobs.append((cell, process, time.monotonic()))
                self.write()
            pending = list(jobs)
            cap = 150 if replay else 480
            while pending:
                self.children.scan()
                require(time.monotonic() < self.deadline, 'Phase watchdog expired')
                self.write()
                for cell, process, started in list(pending):
                    require(time.monotonic()-started < cap-10, f"Child outer watchdog: {cell['name']}")
                    code = process.poll()
                    if code is None:
                        continue
                    kind = 'replay' if replay else 'branch'
                    cell[kind+'_returncode'] = code
                    cell[kind+'_ended'] = time.time()
                    require(code == 0, f"{kind} failed: {cell['name']} ({code})")
                    branch = self.out/cell['name']
                    if replay:
                        check_replay(cell, branch, self.out/('replay_'+cell['name']))
                        pin_result(cell, self.out)
                        cell['status'] = 'exact_verified'
                    else:
                        cell['actor_queries'] = check_branch(self.plan, cell, branch)
                        cell['status'] = 'branch_verified'
                    pending.remove((cell, process, started))
                if pending:
                    time.sleep(.25)
        finally:
            self.children.cleanup()

    def run(self):
        self.out.mkdir(parents=True, exist_ok=False)
        save(self.out/'plan.json', self.plan)
        self.write()
        result = 1
        previous = {}

        def interrupted(signum, frame):
            raise InterruptedError(f'Supervisor received signal {signum}')

        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        # A closed terminal must not terminate an explicitly launched batch.
        previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        previous[signal.SIGALRM] = signal.signal(signal.SIGALRM, interrupted)
        try:
            self.deadline = time.monotonic()+LIMITS['phase_seconds']-10
            signal.setitimer(signal.ITIMER_REAL, LIMITS['phase_seconds']-10)
            for phase in self.plan['phase_order']:
                self.state['phase'] = phase
                if phase != 'C0':
                    require(controls_passed(self.state['cells']), 'C0 gate failed')
                    self.state['sampled_stage_enabled'] = True
                cells = [c for c in self.state['cells'] if c['candidate'] == phase]
                for index in range(0, 6, 2):
                    pair = cells[index:index+2]
                    self.pair(pair)
                    self.pair(pair, replay=True)
                    self.write()
                require(all(c['status'] == 'exact_verified' for c in cells), 'Phase incomplete')
            check_pins(self.plan, self.plan_path)
            check_result_pins(self.state['cells'])
            self.state['status'] = 'completed'
            result = 0
        except BaseException:
            self.state.update(status='aborted', error=traceback.format_exc())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            # Do not let a second termination interrupt group cleanup/status retention.
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            try:
                self.children.cleanup()
            except BaseException:
                self.state.update(status='aborted', cleanup_error=traceback.format_exc())
                result = 1
            for cell in self.state['cells']:
                for kind, directory in (('branch', self.out/cell['name']),
                                        ('replay', self.out/('replay_'+cell['name']))):
                    path = directory/'status.json'
                    if path.exists():
                        try:
                            observed = load(path)
                            cell[kind+'_observed'] = {key: observed.get(key) for key in
                                ('status', 'termination', 'actor_queries', 'steps_executed',
                                 'partial_outputs', 'pid', 'ended')}
                        except (OSError, ValueError) as error:
                            cell[kind+'_artifact_error'] = str(error)
                if cell['status'] != 'exact_verified':
                    cell['status'] = ('blocked_not_started' if cell['status'] == 'planned'
                                      else 'infrastructure_or_integrity_failure')
                    cell['blocked_by'] = self.state.get('error', self.state.get('cleanup_error'))
            self.state['ended'] = time.time()
            try:
                self.write()
            except BaseException:
                # Broken accounting is itself retained; never discard all 18 cells
                # because a malformed partial child artifact prevents final counting.
                self.state.update(status='aborted', accounting_error=traceback.format_exc())
                save(self.out/'status.json', self.state)
                save(self.out/'launch.json', self.state)
                result = 1
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path, metavar='NEW_DIR')
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--launch', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        if args.plan or args.out:
            parser.error('--prepare accepts neither --plan nor --out')
        print(json.dumps(dict(status='prepared', plan=str(prepare(args.prepare.resolve())))))
        return 0
    if args.launch and (args.plan is None or args.out is None):
        parser.error('--launch requires --plan and --out')
    if args.validate_only and args.out is not None:
        parser.error('--validate-only does not accept --out or write files')
    plan_path = args.plan.resolve() if args.plan else None
    plan = load(plan_path) if plan_path else build_plan()
    if plan_path:
        validate_plan(plan, plan_path)
    if args.validate_only:
        print(json.dumps(dict(status='validated', cells=18, limits=LIMITS,
            design_freeze_sha256=plan['design_freeze_sha256'],
            snapshot_manifest_sha256=plan['snapshot_manifest_sha256'],
            protocol_sha256=plan['protocol_sha256'], model_queries=0, simulator_actions=0)))
        return 0
    out = args.out.resolve()
    require(not out.exists() and out != plan_path.parent and plan_path.parent not in out.parents,
            'Launch output must be new and distinct from the prepared plan directory')
    return Supervisor(plan, plan_path, out).run()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'L2A supervisor refused: {error}', file=sys.stderr)
        raise SystemExit(1)
