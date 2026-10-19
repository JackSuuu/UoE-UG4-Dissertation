"""Explicit g+120 adapter over the frozen v1 native branch implementation.

Only the versioned design/boundary and provenance plumbing change. The imported
runner still owns decoding, causal reconstruction, nominal comparison and replay.
This module is also the lightweight, model-free version configuration shared by
the v2 supervisor and auditor. Importing it does not import the branch runner.
"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent
DESIGN = Path('/home/ami/scratch/native_microwave_l2_v2_design_20261019')
PROTOCOL = REPO/'exp_record/agent_notes/native_microwave_l2_v2_protocol_20261019.md'
PROTOCOL_ID = 'native_microwave_l2_v2_20261019'
# Independent design anchor supplied by the coordinator before any v2 run.
FREEZE_SHA = '2d9abef6b0d0a185d74ac0bfa9fd815e631fc4d466c0647df6f29d7c776e5245'
SNAPSHOT_STEPS = (193, 206, 215, 256, 215, 212)
V1_HASHES = {
    'run_native_long_branch.py': '292592b77952ce1b2985da4db186f3e27b04d5a281e70ff741aceda4843eb643',
    'run_native_microwave_l2.py': 'c75287d1f252bd6141c5011997a522612ec51a31c25b8aaef68aff6aba5bec08',
    'audit_native_microwave_l2.py': '89c85d3266927ceadc8f49a10dd61dae4ca79913dc2aa1b24ba05cbce4a6ade1',
    'native_long_candidate_worker.py': '04afbdbeea70eb23551aa524c4de59ef583469260880e9139462661724091db2',
}
BRANCH_SOURCES = ('run_native_long_branch_v2.py', 'run_native_long_branch.py',
                  'native_long_candidate_worker.py')
ALL_SOURCES = tuple('scripts/'+name for name in (
    *V1_HASHES, 'run_native_long_branch_v2.py', 'run_native_microwave_l2_v2.py',
    'audit_native_microwave_l2_v2.py'))


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


def verify_v1_sources():
    for name, expected in V1_HASHES.items():
        require(sha(SCRIPT.parent/name) == expected, f'Frozen v1 implementation drift: {name}')


def load_frozen(name):
    """A private module instance: never mutate a normally imported v1 module.

    No source transformation or string evaluation is used. Each adapter changes
    only documented module configuration/callbacks on its own isolated instance.
    """
    require(name in V1_HASHES, 'Unpinned implementation import')
    path = SCRIPT.parent/name
    require(sha(path) == V1_HASHES[name], f'Frozen v1 implementation drift: {name}')
    spec = importlib.util.spec_from_file_location('_l2_v2_frozen_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_design():
    require(isinstance(FREEZE_SHA, str) and len(FREEZE_SHA) == 64,
            'v2 final design SHA-256 has not been supplied by the coordinator')
    verify_v1_sources()
    freeze_path = DESIGN/'design_freeze.json'
    require(sha(freeze_path) == FREEZE_SHA, 'v2 design freeze anchor mismatch')
    freeze = load(freeze_path)
    require(freeze['protocol'] == PROTOCOL_ID, 'Foreign design version')
    for path, expected in freeze['files'].items():
        require(sha(path) == expected, f'v2 frozen design drift: {path}')
    manifest_path = DESIGN/'snapshot_manifest.json'
    for path in (PROTOCOL, manifest_path):
        require(freeze['files'].get(str(path)) == sha(path), f'v2 design does not bind {path}')
    manifest = load(manifest_path)
    require(manifest['protocol'] == PROTOCOL_ID, 'Foreign snapshot version')
    snapshots = manifest['snapshots']
    require([s['root'] for s in snapshots] == list(range(6))
            and tuple(s['post_action_step'] for s in snapshots) == SNAPSHOT_STEPS,
            'v2 requires exactly the six frozen g+120 boundaries')
    for snap in snapshots:
        require(snap['post_action_step'] == snap['first_grasp_step']+120
                and snap['state_index'] == snap['post_action_step']+1
                and snap['remaining_actions'] == 520-snap['state_index'],
                'v2 causal timer/indexing mismatch')
    return manifest


def branch_hashes():
    return {name: sha(SCRIPT.parent/name) for name in BRANCH_SOURCES}


def version_status(status, directory=None):
    """Reject v1 evidence even if somebody gives it a v2 directory name."""
    require(status.get('protocol_path') == str(PROTOCOL)
            and status.get('protocol_sha256') == sha(PROTOCOL), 'Foreign branch protocol')
    require(status.get('branch_sources') == branch_hashes(), 'Foreign branch implementation')
    args = status['args']
    root = load(Path(args['baseline'])/'status.json')['args']['init_index']
    require(type(root) is int and root in range(6)
            and args['snapshot_step'] == SNAPSHOT_STEPS[root], 'Foreign branch boundary')
    if directory is not None:
        for name, expected in status['branch_sources'].items():
            require(sha(Path(directory)/'branch_source'/name) == expected,
                    f'Archived v2 implementation drift: {name}')
    return root


def control_replay(directory):
    """The fixed supervisor names each independent replay replay_<cell>.

    Check its zero-query/exact marker before allowing even a standalone sampled
    invocation. Full independent tape/label audit remains the unchanged auditor.
    """
    directory = Path(directory).resolve()
    original = load(directory/'status.json')
    root = version_status(original, directory)
    replay = directory.parent/('replay_'+directory.name)
    status = load(replay/'status.json')
    require(version_status(status, replay) == root, 'Control replay root differs')
    require(Path(status['args']['replay']).resolve() == directory
            and status['status'] == 'completed' and status['termination'] == 'exact_replay_passed'
            and status['pid'] != original['pid'] and status['actor_queries'] == 0
            and status['replayed_steps'] == original['steps_executed']
            and status.get('partial_outputs') is False, 'Control lacks fresh completed v2 replay')
    report = load(replay/'verification.json')
    require(report['status'] == 'exact_replay_passed' and report['mismatches'] == []
            and report['max_state_gap'] == report['max_integration_gap'] == 0
            and report['source'] == original['args']['baseline']
            and report['branch_source'] == str(directory)
            and report['snapshot_step'] == SNAPSHOT_STEPS[root]
            and report['branch_steps'] == original['steps_executed']
            and report['frames_checked'] == 11+(SNAPSHOT_STEPS[root]+2)+original['steps_executed']+1,
            'Control replay exactness/coverage mismatch')
    proof = load(replay/'zero_query_proof.json')
    require(all(proof.get(key) == 0 for key in ('actor_constructor_attempts', 'actor_query_attempts',
            'worker_spawn_attempts', 'model_weight_open_attempts', 'actor_queries'))
            and isinstance(proof.get('subprocesses'), list)
            and not any('worker.py' in call for call in proof['subprocesses']),
            'Control replay zero-query proof failed')
    return dict(directory=str(replay), status_sha256=sha(replay/'status.json'),
                verification_sha256=sha(replay/'verification.json'))


def configure():
    runner = load_frozen('run_native_long_branch.py')
    runner.PROTOCOL = PROTOCOL
    runner.SNAPSHOT_STEPS = SNAPSHOT_STEPS
    runner.NEW_SOURCES = BRANCH_SOURCES
    original_parse, original_run = runner.parse_args, runner.run
    original_controls = runner.validate_nominal_controls

    def parse_args(argv=None):
        args = original_parse(argv)
        root = load(args.baseline/'status.json')['args']['init_index']
        require(type(root) is int and root in range(6)
                and args.snapshot_step == SNAPSHOT_STEPS[root], 'Not a v2 g+120 boundary')
        if args.replay:
            version_status(load(args.replay/'status.json'), args.replay)
        return args

    def controls(args, source_hashes):
        evidence = original_controls(args, source_hashes)
        for item, directory in zip(evidence, args.nominal_controls):
            item['strict_replay'] = control_replay(directory)
        return evidence

    def run(args):
        require_design()
        if args.replay:
            version_status(load(args.replay/'status.json'), args.replay)
        return original_run(args)

    runner.parse_args = parse_args
    runner.validate_nominal_controls = controls
    runner.run = run
    return runner


def supervised(runner, args, argv):
    """Explicit v2 child entrypoint; do not inherit v1's __file__ relaunch.

    The child and its unchanged worker share a fresh process group. The bank
    supervisor additionally subreaps detached descendants and owns the phase cap.
    """
    require(not args.out.exists(), f'Output already exists: {args.out}')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2')
    child = subprocess.Popen([sys.executable, '-B', '-u', str(SCRIPT), *argv, '--_child'],
                             env=env, start_new_session=True)
    cap = 150 if args.smoke or args.replay else 480
    try:
        return child.wait(timeout=cap-10)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit):
        if args.out.exists():
            runner.save(args.out/'supervisor_interruption.json', dict(status='censored',
                termination='v2_outer_interruption_or_timeout', child_pid=child.pid, cap=cap))
        return 1
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runner = configure()
    args = runner.parse_args(argv)
    require_design()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    return runner.run(args) if args._child else supervised(runner, args, argv)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'v2 branch refused: {error}', file=sys.stderr)
        raise SystemExit(1)
