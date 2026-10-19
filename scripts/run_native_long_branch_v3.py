"""Frozen H96 branch, with an exact fresh-v2-prefix gate before request 33.

The local execution loop is explicit because the frozen v1 loop hard-codes
32-action nominal verification. Scene restoration, worker, mapping, nominal
comparisons and independent replay primitives are imported without rewriting.
Importing this configuration module does not import model/simulator libraries.
"""
import argparse
import hashlib
import importlib.util
import io
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
DESIGN = Path('/home/ami/scratch/native_microwave_l2_v3_design_20261019')
PROTOCOL = REPO/'exp_record/agent_notes/native_microwave_l2_v3_protocol_20261019.md'
PROTOCOL_ID = 'native_microwave_l2_v3_20261019'
FREEZE_SHA = '47a14701541615f76803ca5e8303b7c69ef334e98444ac0bc5a6e554fbfc5e86'
HORIZON, PREFIX = 96, 32
PREVIOUS_RUN = Path('/home/ami/scratch/native_microwave_l2a_20261019_v2')
SNAPSHOT_STEPS = (193, 206, 215, 256, 215, 212)
OLD_HASHES = {
    'run_native_long_branch.py': '292592b77952ce1b2985da4db186f3e27b04d5a281e70ff741aceda4843eb643',
    'run_native_microwave_l2.py': 'c75287d1f252bd6141c5011997a522612ec51a31c25b8aaef68aff6aba5bec08',
    'audit_native_microwave_l2.py': '89c85d3266927ceadc8f49a10dd61dae4ca79913dc2aa1b24ba05cbce4a6ade1',
    'native_long_candidate_worker.py': '04afbdbeea70eb23551aa524c4de59ef583469260880e9139462661724091db2',
    'run_native_long_branch_v2.py': '3c2c216eaed409058488f961e73f4c007adaca4035ff62d72125ffa63bee0bc5',
    'run_native_microwave_l2_v2.py': '858706e220aacec18e9fb85a885bba744af704bf2d73e9574d8bb92975b3d8b3',
    'audit_native_microwave_l2_v2.py': '5c4e8b360b0accfc376cb89536a861f99928fc71fe2d0ff7661085c5efc7ba02',
}
BRANCH_SOURCES = ('run_native_long_branch_v3.py', 'run_native_long_branch.py',
                  'native_long_candidate_worker.py')
ALL_SOURCES = tuple('scripts/'+name for name in (*OLD_HASHES,
    'run_native_long_branch_v3.py', 'run_native_microwave_l2_v3.py', 'audit_native_microwave_l2_v3.py'))
LIMITS = dict(cells=18, calls=1728, local_horizon=96, continuation_steps=0, parallel=2, batch=1,
              readiness_seconds=180, query_seconds=30, wall_seconds=300, replay_wall_seconds=180,
              child_outer_seconds=540, replay_outer_seconds=210, phase_seconds=14400, retries=0)
PREFIX_FIELDS = ('step', 'tokens', 'native_proposal', 'proposal', 'action', 'executed_mapped_action',
                 'before', 'after', 'diagnostics', 'events', 'reward', 'done', 'contacts_substeps',
                 'source', 'seed', 'query_index', 'branch_step', 'language', 'checkpoint_revision',
                 'policy_png_sha256', 'policy_rgb_sha256', 'raw_rgb_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_hashes(pins):
    for path, expected in pins.items():
        require(sha(path) == expected, f'Frozen input/source drift: {path}')


def verify_old_sources():
    check_hashes({str(SCRIPT.parent/name): expected for name, expected in OLD_HASHES.items()})


def load_frozen(name):
    require(name in OLD_HASHES, 'Unpinned implementation import')
    path = SCRIPT.parent/name
    require(sha(path) == OLD_HASHES[name], f'Frozen implementation drift: {name}')
    spec = importlib.util.spec_from_file_location('_l2_v3_frozen_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_design():
    require(isinstance(FREEZE_SHA, str) and len(FREEZE_SHA) == 64,
            'v3 final design hash not supplied by coordinator')
    verify_old_sources()
    require(sha(DESIGN/'design_freeze.json') == FREEZE_SHA, 'v3 freeze anchor mismatch')
    freeze = load(DESIGN/'design_freeze.json')
    require(freeze['protocol'] == PROTOCOL_ID, 'Foreign frozen design')
    check_hashes(freeze['files'])
    for path in (PROTOCOL, DESIGN/'snapshot_manifest.json'):
        require(freeze['files'].get(str(path)) == sha(path), 'Design must bind protocol and manifest')
    manifest = load(DESIGN/'snapshot_manifest.json')
    import jsonschema
    jsonschema.Draft202012Validator(load(DESIGN/'snapshot_manifest.schema.json')).validate(manifest)
    require(manifest['protocol'] == PROTOCOL_ID and manifest['local_horizon'] == HORIZON
            and manifest['nested_measurements'] == [12, 32, 96], 'Not frozen H96 design')
    previous = load_frozen('run_native_long_branch_v2.py').require_design()
    for key in ('snapshots', 'sampling', 'sources', 'seed_formula', 'checkpoint_revision',
                'base_code_revision', 'instruction', 'label_contract', 'candidate_count_per_snapshot',
                'selector', 'retries'):
        require(manifest[key] == previous[key], f'Horizon-only design changed {key}')
    for key, expected in dict(local_model_calls=1728, local_branch_wall_seconds=300,
            worker_readiness_seconds=180, generation_seconds=30, replay_seconds=180,
            child_outer_seconds=540, replay_outer_seconds=210, phase_job_wall_seconds=14400,
            continuation_calls_upper_bound=0, parallel=2, batch=1, retries=0).items():
        require(manifest['budgets'][key] == expected, f'v3 budget mismatch: {key}')
    require(manifest['continuation_steps'] == 0 and manifest['automatic_continuation'] is False
            and manifest['hard_stop'] == 'A valid negative ends this source/g+120/horizon-extension line; no further duration/seed/root-specific tuning',
            'Final horizon stopping contract differs')
    require(manifest['prefix_reference_manifest'] == str(DESIGN/'v2_prefix_manifest.json')
            and manifest['prefix_reference_manifest_sha256'] == sha(DESIGN/'v2_prefix_manifest.json'),
            'Prefix manifest is not bound by snapshot design')
    references = load(DESIGN/'v2_prefix_manifest.json')
    require(references['protocol'] == PROTOCOL_ID
            and references['reference_protocol'] == 'native_microwave_l2_v2_20261019'
            and len(references['cells']) == 18
            and [(c['root'], c['candidate']) for c in references['cells']] ==
                [(r, f'C{c}') for c in range(3) for r in range(6)], 'Foreign prefix-reference protocol/identities')
    return manifest


def branch_hashes():
    return {name: sha(SCRIPT.parent/name) for name in BRANCH_SOURCES}


def cell_identity(args):
    root = load(Path(args.baseline)/'status.json')['args']['init_index']
    require(type(root) is int and root in range(6) and args.snapshot_step == SNAPSHOT_STEPS[root],
            'Foreign snapshot/root')
    candidate = 0 if args.candidate == 'greedy' else next((c for c in (1, 2)
        if args.seed == 20261019+10000*root+100*c), None)
    require(candidate is not None and (candidate != 0 or args.seed == 0), 'Foreign candidate seed')
    return root, candidate


def version_status(status, directory=None):
    require(status.get('protocol_path') == str(PROTOCOL)
            and status.get('protocol_sha256') == sha(PROTOCOL)
            and status.get('design_freeze_sha256') == FREEZE_SHA, 'Foreign v3 protocol/design')
    require(status.get('branch_sources') == branch_hashes(), 'Foreign v3 branch implementation')
    args = argparse.Namespace(**status['args'])
    require(args.local_horizon == HORIZON and args.continuation_steps == 0, 'Foreign H32/handoff evidence')
    identity = cell_identity(args)
    if directory is not None:
        check_hashes({str(Path(directory)/'branch_source'/name): expected
                      for name, expected in status['branch_sources'].items()})
        device = load(Path(directory)/'fresh_device_preflight.json')
        lane = {2: (1, 'GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8'),
                3: (0, 'GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b')}[args.gpu]
        require(device['physical_gpu'] == args.gpu and device['physical_uuid'] == lane[1]
                and device['visible_cuda']['count'] == 1 and device['visible_cuda']['ordinal'] == 0
                and device['visible_cuda']['uuid'] == device['renderer']['cuda']['uuid'] == lane[1]
                and device['renderer']['egl_device'] == args.egl_device == lane[0]
                and device['model_queries'] == 0, 'Missing/nonexact fresh GPU/EGL alignment proof')
    return identity


def validate_plan_input(path):
    manifest = require_design()
    plan = load(path)
    require(plan['protocol'] == PROTOCOL_ID and plan['design_freeze_sha256'] == FREEZE_SHA
            and plan['protocol_sha256'] == sha(PROTOCOL)
            and plan['snapshot_manifest_sha256'] == sha(DESIGN/'snapshot_manifest.json')
            and plan['limits'] == LIMITS, 'Foreign v3 prepared plan')
    require(set(plan['archive_sources']) == set(ALL_SOURCES), 'Incomplete source inventory')
    for name, expected in plan['archive_sources'].items():
        require(sha(REPO/name) == expected and sha(Path(path).parent/'source'/name) == expected,
                f'Prepared implementation drift: {name}')
    freeze = load(DESIGN/'design_freeze.json')
    require(all(plan['frozen_files_sha256'].get(name) == expected for name, expected in freeze['files'].items()),
            'Prepared plan omits design/reference pins')
    require(plan['snapshots'] == manifest['snapshots'] and plan['lanes'] == manifest['devices']['reference_lanes']
            and plan['phase_order'] == ['C0', 'C1', 'C2'], 'Prepared cohort/lane/order drift')
    for key in ('sampling', 'instruction', 'checkpoint_revision', 'base_code_revision'):
        require(plan[key] == manifest[key], f'Prepared actor drift: {key}')
    from run_native_microwave_l2_v3 import configure
    engine = configure()
    require(plan['cells'] == engine.identities(manifest['snapshots']), 'Prepared cell/seed schedule differs')
    require(set(plan['prefix_sources']) == {c['name'] for c in plan['cells']}, 'Incomplete prefix bank')
    for cell in plan['cells']:
        require(plan['prefix_sources'][cell['name']] == prefix_record(cell['root'], int(cell['candidate'][1:])),
                'Prepared prefix evidence differs from frozen v2 reference')
    engine.check_pins(plan, Path(path))
    return plan


def prefix_record(root, candidate):
    """Freeze only the unique completed v2 counterpart, never choose a prefix."""
    previous = load_frozen('run_native_long_branch_v2.py')
    directory = PREVIOUS_RUN/f'C{candidate}_root{root}'
    status = load(directory/'status.json')
    require(previous.version_status(status, directory) == root, 'Foreign v2 prefix root')
    require(cell_identity(argparse.Namespace(**status['args'])) == (root, candidate)
            and status['status'] == 'completed' and status['termination'] == 'action_budget'
            and status['actor_queries'] == status['steps_executed'] == PREFIX
            and status['args']['local_horizon'] == PREFIX and not status['partial_outputs'],
            'v2 prefix must be the complete original 32-query cell')
    previous.control_replay(directory)  # Works for sampled tapes too; no model calls.
    references = load(DESIGN/'v2_prefix_manifest.json')
    require(references['reference_protocol'] == previous.PROTOCOL_ID and references['protocol'] == PROTOCOL_ID,
            'Foreign v2 prefix protocol')
    entry = next(c for c in references['cells'] if c['candidate_id'] == f'C{candidate}_root{root}')
    require(entry['directory'] == str(directory) and entry['root'] == root and entry['candidate'] == f'C{candidate}'
            and entry['required_fresh_queries'] == 32 and entry['required_frames_including_boundary'] == 33,
            'Foreign v2 prefix manifest entry')
    pins = {str(directory/name): expected for name, expected in entry['artifacts_sha256'].items()}
    replay = directory.parent/('replay_'+directory.name)
    pins.update({str(replay/name): expected for name, expected in entry['replay_artifacts_sha256'].items()})
    for path in (replay/'zero_query_proof.json', PREVIOUS_RUN/'plan.json', PREVIOUS_RUN/'launch.json'):
        pins[str(path)] = sha(path)
    check_hashes(pins)
    return dict(root=root, candidate=f'C{candidate}', directory=str(directory),
                reference_protocol=previous.PROTOCOL_ID, reference_protocol_sha256=sha(previous.PROTOCOL),
                reference_design_sha256=previous.FREEZE_SHA,
                prefix_manifest_sha256=sha(DESIGN/'v2_prefix_manifest.json'),
                semantic_prefix_sha256=entry['semantic_prefix_sha256'], sha256=pins)


class PrefixMismatch(ValueError):
    pass


def compare_prefix(reference, actual, count=PREFIX, start=0):
    """Compare committed tapes and raw frame bytes; valid during a live branch.

    No arrays need finalization: native Tape.capture atomically spools each frame
    before step_once returns. Check policy PNG bytes, not just logged hashes.
    """
    import numpy as np
    reference, actual = Path(reference), Path(actual)
    expected_rows, actual_rows = rows(reference/'steps.jsonl'), rows(actual/'steps.jsonl')
    expected_frames, actual_frames = rows(reference/'frames.jsonl'), rows(actual/'frames.jsonl')
    events = rows(actual/'queries.jsonl')
    responses = rows(actual/'worker_responses.jsonl') if count else []
    require(0 <= count <= PREFIX and len(expected_rows) >= count and len(actual_rows) >= count,
            'Incomplete fresh prefix actions')
    require(len(expected_frames) > count and len(actual_frames) > count, 'Incomplete fresh prefix frames')
    for i in range(start, count):
        require(len(events) >= 2*(i+1) and events[2*i]['event'] == 'requested'
                and events[2*i+1]['event'] == 'returned', 'Incomplete/nonsequential prefix query evidence')
        for event in events[2*i:2*i+2]:
            require(all(actual_rows[i].get(key) == value for key, value in event.items() if key != 'event'),
                    'Prefix query/transition attribution differs')
        require(responses[i]['query_index'] == i and responses[i]['response'] == actual_rows[i]['decode'],
                'Prefix raw worker response differs from action decode')
        for key in PREFIX_FIELDS:
            if expected_rows[i][key] != actual_rows[i][key]:
                raise PrefixMismatch(f'v2 prefix action {i} differs: {key}')
        for key in ('prompt', 'sampling', 'resolved_generation_config', 'valid'):
            if expected_rows[i]['decode'][key] != actual_rows[i]['decode'][key]:
                raise PrefixMismatch(f'v2 prefix decode {i} differs: {key}')
        require(actual_rows[i]['before'] == actual_frames[i]['fingerprint']
                and actual_rows[i]['after'] == actual_frames[i+1]['fingerprint']
                and actual_rows[i]['diagnostics'] == actual_frames[i+1]['diagnostics'], 'Prefix transition/frame mismatch')
        old_png, new_png = reference/'policy_inputs'/f'{i:06d}.png', actual/'policy_inputs'/f'{i:06d}.png'
        if old_png.read_bytes() != new_png.read_bytes() or sha(new_png) != actual_rows[i]['policy_png_sha256']:
            raise PrefixMismatch(f'v2 prefix policy PNG differs: {i}')
    for i in range(start, count+1):
        if expected_frames[i] != actual_frames[i]:
            raise PrefixMismatch(f'v2 prefix full frame metadata differs: {i}')
        with np.load(reference/'frames'/f'{i:06d}.npz', allow_pickle=False) as old, \
                np.load(actual/'frames'/f'{i:06d}.npz', allow_pickle=False) as new:
            if set(old.files) != set(new.files) or not {'state', 'integration', 'agentview_image',
                                                       'robot0_eye_in_hand_image'} <= set(old.files):
                raise PrefixMismatch(f'v2 prefix frame array inventory differs: {i}')
            for key in old.files:
                a, b = old[key], new[key]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise PrefixMismatch(f'v2 prefix raw frame array differs: {i}/{key}')
    # Reference aggregate arrays are frozen too; compare their actual contents
    # with the per-frame files used by the live barrier, not just their names.
    if count == PREFIX and start == 0:
        with np.load(reference/'actions.npz', allow_pickle=False) as actions:
            for key, field in (('mapped', 'action'), ('native', 'native_proposal'), ('tokens', 'tokens')):
                require(np.array_equal(actions[key], np.asarray([r[field] for r in expected_rows])),
                        'Pinned reference actions archive disagrees with rows')
        for filename in ('states.npz', 'images.npz'):
            with np.load(reference/filename, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
                for i in range(PREFIX+1):
                    with np.load(reference/'frames'/f'{i:06d}.npz', allow_pickle=False) as frame:
                        for key in arrays:
                            a, b = arrays[key][i], frame[key]
                            require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                                    f'Pinned reference aggregate/spooled array differs: {i}/{key}')
    return dict(actions_checked=count, frames_checked=count+1, mismatches=[], exact=True)


class PrefixGuard:
    def __init__(self, record, out, plan_path):
        check_hashes(record['sha256'])
        self.record, self.out, self.passed = record, Path(out), False
        self.plan_path, self.plan_sha256 = Path(plan_path), sha(plan_path)
        self.expected = rows(Path(record['directory'])/'steps.jsonl')

    def input(self, index, metadata, png):
        if index >= PREFIX:
            return
        expected = self.expected[index]
        for key in ('step', 'branch_step', 'query_index', 'source', 'seed', 'language', 'before',
                    'checkpoint_revision', 'policy_png_sha256', 'policy_rgb_sha256', 'raw_rgb_sha256'):
            if metadata[key] != expected[key]:
                raise PrefixMismatch(f'v2 prefix query input {index} differs: {key}')
        if png != (Path(self.record['directory'])/'policy_inputs'/f'{index:06d}.png').read_bytes():
            raise PrefixMismatch(f'v2 prefix input PNG differs: {index}')

    def output(self, index, decode):
        if index >= PREFIX:
            return
        expected = self.expected[index]['decode']
        for key in ('tokens', 'native_action', 'action', 'prompt', 'sampling', 'resolved_generation_config',
                    'source', 'seed', 'query_index', 'policy_png_sha256', 'policy_rgb_sha256', 'valid'):
            if decode[key] != expected[key]:
                raise PrefixMismatch(f'v2 prefix query output {index} differs: {key}')

    def after_step(self, index):
        if index < PREFIX:
            compare_prefix(self.record['directory'], self.out, index+1, start=index)

    def before_request(self, index):
        require(0 <= index < HORIZON, 'Request outside H96')
        if index < PREFIX:
            return
        if self.passed:
            require(sha(self.out/'prefix_identity.json') == self.certificate_sha256,
                    'Missing/changed pre-query-33 certificate')
            return
        report = dict(status='prefix_mismatch', reference=self.record['directory'],
                      reference_sha256=self.record['sha256'], before_query_index=index,
                      plan_path=str(self.plan_path), plan_sha256=self.plan_sha256,
                      design_freeze_sha256=FREEZE_SHA,
                      prefix_manifest_sha256=self.record['prefix_manifest_sha256'],
                      semantic_prefix_sha256=self.record['semantic_prefix_sha256'])
        try:
            require(sha(self.plan_path) == self.plan_sha256, 'Plan changed before query 33')
            check_hashes(self.record['sha256'])
            report.update(compare_prefix(self.record['directory'], self.out))
            require(index == PREFIX, 'Prefix gate was skipped before request 33')
            # Inspect the actual request tape: a retrospective marker is invalid.
            requested = [r for r in rows(self.out/'queries.jsonl') if r['event'] == 'requested']
            require([r['query_index'] for r in requested] == list(range(PREFIX)),
                    'Prefix gate must precede the 33rd requested event')
            report.update(status='exact_v2_prefix_passed', actor_queries_at_check=len(requested),
                          checked_at=time.time())
            self.passed = True
        except BaseException:
            report['error'] = traceback.format_exc()
            raise
        finally:
            save(self.out/'prefix_identity.json', report)
        self.certificate_sha256 = sha(self.out/'prefix_identity.json')
        require(load(self.out/'prefix_identity.json') == report, 'Prefix certificate was not persisted exactly')


def validate_prefix_marker(directory, record, status):
    marker = load(Path(directory)/'prefix_identity.json')
    require(marker['status'] == 'exact_v2_prefix_passed' and marker['mismatches'] == []
            and marker['actor_queries_at_check'] == PREFIX and marker['before_query_index'] == PREFIX
            and marker['actions_checked'] == PREFIX and marker['frames_checked'] == PREFIX+1
            and marker['reference'] == record['directory'] and marker['reference_sha256'] == record['sha256'],
            'Missing/foreign/nonexact pre-query-33 prefix proof')
    require(marker['plan_path'] == status['args']['plan'] and marker['plan_sha256'] == sha(status['args']['plan'])
            and marker['design_freeze_sha256'] == FREEZE_SHA
            and marker['prefix_manifest_sha256'] == record['prefix_manifest_sha256']
            and marker['semantic_prefix_sha256'] == record['semantic_prefix_sha256'], 'Prefix certificate plan/design binding differs')
    check_hashes(record['sha256'])
    compare_prefix(record['directory'], directory)
    requested = [r for r in rows(Path(directory)/'queries.jsonl') if r['event'] == 'requested']
    require(status['actor_queries'] == len(requested), 'Prefix/final query count mismatch')
    if len(requested) > PREFIX:
        require(marker['checked_at'] <= requested[PREFIX]['requested_at'], 'Prefix checked after request 33')
    return marker


def validate_tape(directory, status, tape_rows=None):
    """Horizon-explicit version of v1 query/action validation, also used offline."""
    import numpy as np
    from PIL import Image
    directory = Path(directory)
    tape_rows = rows(directory/'steps.jsonl') if tape_rows is None else tape_rows
    args = argparse.Namespace(**status['args'])
    root, candidate = cell_identity(args)
    require(args.local_horizon == HORIZON and args.continuation_steps == status['continuation_cap'] == 0,
            'Not local-only H96')
    events = rows(directory/'queries.jsonl')
    responses = rows(directory/'worker_responses.jsonl')
    requested = [r for r in events if r['event'] == 'requested']
    returned = [r for r in events if r['event'] == 'returned']
    n = len(tape_rows)
    require(n == status['steps_executed'] == status['actor_queries'] == len(requested) == len(returned)
            and 0 < n <= HORIZON and args.snapshot_step+1+n <= 520, 'H96 query/action accounting')
    native = [i for i, row in enumerate(tape_rows) if row['diagnostics']['native_success']]
    require((status['termination'] == 'native_success' and native == [n-1]) or
            (status['termination'] == 'action_budget' and n == HORIZON and not native), 'Invalid H96/native stopping boundary')
    require([r['event'] for r in events] == ['requested', 'returned']*n, 'Nonsequential query events')
    require(len(responses) == n, 'Raw worker response count differs')
    helper = load_frozen('run_native_long_branch.py')
    with np.load(directory/'actions.npz', allow_pickle=False) as archive:
        for key, field in (('mapped', 'action'), ('native', 'native_proposal'), ('tokens', 'tokens')):
            np.testing.assert_array_equal(archive[key], np.asarray([r[field] for r in tape_rows]).reshape(-1, 7))
    for i, row in enumerate(tape_rows):
        helper.validate_action(row, args.snapshot_step+1+i)
        require(row['query_index'] == row['branch_step'] == requested[i]['query_index'] == returned[i]['query_index'] == i,
                'Nonsequential request/action identity')
        require(row['source'] == args.candidate and row['seed'] == (0 if candidate == 0 else args.seed+i),
                'H96 policy/seed drift')
        require(row['owner'] == 'vla' and row['candidate_owner'] == 'local_candidate'
                and row['phase'] == 'actor' and row['branch_phase'] == 'local'
                and row['forced_diagnostic_handoff'] is False and row['repair_verified'] is False
                and row['candidate_id'] == status['candidate_id'], 'Wrong local action ownership')
        for event in (requested[i], returned[i]):
            require(all(row.get(k) == value for k, value in event.items() if k != 'event'),
                    'Request/response/action provenance mismatch')
        png = directory/'policy_inputs'/f'{i:06d}.png'
        require(sha(png) == row['policy_png_sha256'], 'Saved policy PNG drift')
        with Image.open(png) as image:
            rgb = hashlib.sha256(np.ascontiguousarray(image.convert('RGB')).tobytes()).hexdigest()
        require(rgb == row['policy_rgb_sha256'] and row['raw_rgb_sha256'] == row['before']['images']['agentview_image'],
                'Policy image provenance mismatch')
        require(row['language'] == status['language'] and row['checkpoint_revision'] == status['checkpoint_revision'],
                'Instruction/checkpoint drift')
        decode = row['decode']
        require(responses[i]['query_index'] == i and responses[i]['response'] == decode, 'Raw worker response differs')
        require(decode['valid'] is True and decode['action'] == row['action']
                and decode['native_action'] == row['native_proposal'] and decode['tokens'] == row['tokens'],
                'Invalid decode/action attribution')
        require(all(type(token) is int for token in row['tokens']), 'Noninteger generated tokens')
        helper.validate_decode(decode, row)
    for filename in ('states.npz', 'images.npz'):
        with np.load(directory/filename, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
            required = {'state', 'integration'} if filename == 'states.npz' else {'agentview_image', 'robot0_eye_in_hand_image'}
            require(required <= set(arrays) and all(len(value) == n+1 for value in arrays.values()), 'Aggregate frame count/inventory differs')
            for i in range(n+1):
                with np.load(directory/'frames'/f'{i:06d}.npz', allow_pickle=False) as frame:
                    for key in arrays:
                        a, b = arrays[key][i], frame[key]
                        require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                                'Aggregate/spooled frame arrays differ')
    if candidate == 0:
        nominal = load(directory/'nominal_identity.json')
        require(nominal['status'] == 'exact_nominal_identity_passed' and nominal['mismatches'] == []
                and nominal['steps_checked'] == n and len(nominal['checks']) == n,
                'C0 must reproduce its entire H96/early-success suffix')
    return dict(requested=n, returned=n, executed=n)


def control_replay(directory):
    directory = Path(directory).resolve()
    original = load(directory/'status.json')
    identity = version_status(original, directory)
    replay = directory.parent/('replay_'+directory.name)
    status = load(replay/'status.json')
    require(version_status(status, replay) == identity and status['pid'] != original['pid']
            and Path(status['args']['replay']).resolve() == directory
            and status['status'] == 'completed' and status['termination'] == 'exact_replay_passed'
            and status['actor_queries'] == 0 and status['replayed_steps'] == original['steps_executed']
            and status['partial_outputs'] is False, 'Fresh exact v3 replay required')
    report = load(replay/'verification.json')
    require(report['status'] == 'exact_replay_passed' and report['mismatches'] == []
            and report['max_state_gap'] == report['max_integration_gap'] == 0
            and report['branch_source'] == str(directory) and report['source'] == original['args']['baseline']
            and report['branch_steps'] == original['steps_executed']
            and report['frames_checked'] == 11+original['args']['snapshot_step']+2+original['steps_executed']+1,
            'v3 replay exactness/coverage failed')
    proof = load(replay/'zero_query_proof.json')
    require(all(proof.get(k) == 0 for k in ('actor_constructor_attempts', 'actor_query_attempts',
            'worker_spawn_attempts', 'model_weight_open_attempts', 'actor_queries'))
            and isinstance(proof.get('subprocesses'), list)
            and not any('worker.py' in x for x in proof['subprocesses']), 'Replay queried a model')
    return dict(directory=str(replay), verification_sha256=sha(replay/'verification.json'))


def validate_controls(directories, plan):
    require(directories is not None and len(directories) == 6, 'Six fresh v3 C0 controls required')
    evidence, roots = [], set()
    for directory in directories:
        directory = Path(directory).resolve()
        status = load(directory/'status.json')
        root, candidate = version_status(status, directory)
        require(candidate == 0 and root not in roots and status['status'] == 'completed'
                and not status['partial_outputs'], 'Foreign/duplicate/incomplete nominal control')
        roots.add(root)
        validate_tape(directory, status)
        require(status['steps_executed'] == HORIZON or status['termination'] == 'native_success',
                'Truncated H96 nominal control')
        validate_prefix_marker(directory, plan['prefix_sources'][f'C0_root{root}'], status)
        from audit_native_microwave_l2_v3 import configure
        audit = configure()
        snapshot = plan['snapshots'][root]
        require(status['args']['baseline'] == snapshot['source'], 'Foreign C0 original tape')
        context = audit.baseline_context(snapshot)
        tape_rows = audit.load_rows(directory/'steps.jsonl')
        audit.nominal_identity(directory, snapshot, context, tape_rows,
                               audit.labels(context['initial'], tape_rows, context['first_grasp_z']))
        evidence.append(dict(root=root, directory=str(directory), status_sha256=sha(directory/'status.json'),
                             strict_replay=control_replay(directory)))
    require(roots == set(range(6)), 'Incomplete nominal control coverage')
    return evidence


def cuda_identity(ordinal):
    """Context-free driver UUID lookup, used only during authorized execution."""
    import ctypes
    import uuid
    driver = ctypes.CDLL('libcuda.so.1')
    count, device = ctypes.c_int(), ctypes.c_int()
    identifier = (ctypes.c_ubyte * 16)()
    require(driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0
            and driver.cuDeviceGet(ctypes.byref(device), ordinal) == 0, 'CUDA preflight enumeration failed')
    get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', driver.cuDeviceGetUuid)
    require(get_uuid(ctypes.byref(identifier), device) == 0, 'CUDA preflight UUID failed')
    return dict(count=count.value, ordinal=ordinal, uuid='GPU-'+str(uuid.UUID(bytes=bytes(identifier))))


def egl_identity(index):
    """Resolve the selected EGL device to its unmasked CUDA UUID, no context."""
    import ctypes
    egl = ctypes.CDLL('libEGL.so.1')
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    query_address = egl.eglGetProcAddress(b'eglQueryDevicesEXT')
    attribute_address = egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT')
    require(query_address and attribute_address, 'EGL device enumeration unavailable')
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                            ctypes.POINTER(ctypes.c_int))(query_address)
    attribute = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_ssize_t))(attribute_address)
    devices, count, ordinal = (ctypes.c_void_p*32)(), ctypes.c_int(), ctypes.c_ssize_t()
    require(query(32, devices, ctypes.byref(count)) and 0 <= index < count.value, 'EGL device index unavailable')
    require(attribute(devices[index], 0x323A, ctypes.byref(ordinal)), 'EGL_CUDA_DEVICE_NV unavailable')
    return dict(egl_device=index, egl_cuda_ordinal=ordinal.value, cuda=cuda_identity(ordinal.value))


def fresh_device_preflight(args, out, design):
    lane = next(lane for lane in design['devices']['reference_lanes'] if lane['gpu'] == args.gpu)
    require(args.egl_device == lane['egl_device'], 'Frozen EGL lane differs')
    for key, value in dict(CUDA_VISIBLE_DEVICES=lane['physical_uuid'],
                           MUJOCO_EGL_DEVICE_ID=str(args.egl_device)).items():
        require(os.environ.get(key, value) == value, f'Execution environment differs: {key}')
        os.environ[key] = value
    raw = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=index,uuid,pci.bus_id',
                                   '--format=csv,noheader'], text=True, timeout=15).strip()
    index, uuid, bus = [item.strip() for item in raw.split(',')]
    require(int(index) == args.gpu and uuid == lane['physical_uuid'], 'Fresh physical GPU UUID differs')
    visible = cuda_identity(0)
    require(visible['count'] == 1 and visible['uuid'] == uuid, 'Visible CUDA ordinal is not the frozen GPU')
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--_egl-identity', str(args.egl_device)],
                            env=env, text=True, capture_output=True, timeout=20, check=True)
    renderer = json.loads(result.stdout)
    require(renderer['cuda']['uuid'] == uuid, 'Fresh EGL renderer UUID differs from scene/actor lane')
    proof = dict(physical_gpu=args.gpu, physical_uuid=uuid, pci_bus_id=bus,
                 visible_cuda=visible, renderer=renderer, checked_at=time.time(), model_queries=0)
    save(out/'fresh_device_preflight.json', proof)
    return proof


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'out', 'plan'):
        parser.add_argument('--'+name, type=Path, required=name == 'out')
    parser.add_argument('--snapshot-step', type=int)
    parser.add_argument('--candidate', choices=('greedy', 'sample'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--local-horizon', type=int)
    parser.add_argument('--continuation-steps', type=int)
    parser.add_argument('--gpu', type=int, choices=(2, 3))
    parser.add_argument('--egl-device', type=int, choices=(0, 1))
    parser.add_argument('--wall-seconds', type=float)
    parser.add_argument('--readiness-seconds', type=float, default=180)
    parser.add_argument('--query-seconds', type=float, default=30)
    parser.add_argument('--nominal-controls', type=Path, nargs='+')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--replay', type=Path)
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    fields = ('baseline', 'snapshot_step', 'candidate', 'seed', 'local_horizon', 'continuation_steps',
              'gpu', 'egl_device', 'plan')
    if args.replay:
        args.replay = args.replay.resolve()
        status = load(args.replay/'status.json')
        version_status(status, args.replay)
        for key in fields:
            value = status['args'][key]
            if key in ('baseline', 'plan'):
                value = Path(value)
            require(getattr(args, key) is None or getattr(args, key) == value, f'Replay --{key} differs')
            setattr(args, key, value)
    for key in fields:
        if key != 'plan' or not args.smoke:
            require(getattr(args, key) is not None, f'--{key.replace("_", "-")} required')
    args.baseline, args.out = args.baseline.resolve(), args.out.resolve()
    if args.plan:
        args.plan = args.plan.resolve()
    cell_identity(args)
    require(args.local_horizon == HORIZON and args.continuation_steps == 0, 'Only local H96; no handoff')
    require(args.egl_device == {2: 1, 3: 0}[args.gpu], 'Wrong physical GPU/EGL lane')
    cap = 180 if args.replay or args.smoke else 300
    args.wall_seconds = cap if args.wall_seconds is None else args.wall_seconds
    require(0 < args.wall_seconds <= cap and 0 < args.readiness_seconds <= 180
            and 0 < args.query_seconds <= 30, 'Wall/readiness/query cap exceeded')
    if args.candidate == 'sample' and not (args.smoke or args.replay):
        require(args.nominal_controls is not None, 'Sampling requires all six fresh v3 controls')
    return args


def run(args):
    import numpy as np
    design = require_design()
    helper = load_frozen('run_native_long_branch.py')
    helper.PROTOCOL, helper.SNAPSHOT_STEPS = PROTOCOL, SNAPSHOT_STEPS
    plan = validate_plan_input(args.plan) if args.plan else None
    root, candidate = cell_identity(args)
    require(str(args.baseline) == design['snapshots'][root]['source'], 'Foreign original baseline path')
    prefix_record_value = None if args.smoke else plan['prefix_sources'][f'C{candidate}_root{root}']
    if prefix_record_value:
        check_hashes(prefix_record_value['sha256'])
    control_evidence = validate_controls(args.nominal_controls, plan) if candidate and not (args.smoke or args.replay) else None
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out
    recorded = json.loads(json.dumps({k: value for k, value in vars(args).items() if k != '_child'}, default=str))
    status = dict(status='started', scope='L2A_v3_H96_local_only', args=recorded, started=time.time(),
        pid=os.getpid(), actor_queries=0, steps_executed=0, candidate_id=out.name, candidate_count=1,
        batch_size=1, scene_resets=1, snapshot_convention='zero_based_post_action',
        baseline_prefix_actions=args.snapshot_step+1, initialization_actions=10, local_horizon=HORIZON,
        continuation_cap=0, lifetime_action_cap=520, wall_seconds=args.wall_seconds,
        readiness_seconds=args.readiness_seconds, query_seconds=args.query_seconds, phase_supervisor_seconds=14400,
        wall_scope='setup_reconstruction_and_branch_excluding_model_load', handoff=None, repair_verified=False,
        seed_schedule='greedy=0; sampled_local=base+k', protocol_path=str(PROTOCOL), protocol_sha256=sha(PROTOCOL),
        design_freeze_sha256=FREEZE_SHA, branch_sources=branch_hashes(), nominal_controls=control_evidence,
        policy_inputs='native_agentview_RGB_and_full_original_instruction_only', termination=None)
    report = dict(status='started', source=str(args.baseline), snapshot_step=args.snapshot_step,
                  frames_checked=0, max_state_gap=0., max_integration_gap=0., mismatches=[], actor_queries=0)
    nominal = dict(status='not_run', baseline=str(args.baseline), snapshot_step=args.snapshot_step,
                  limit=HORIZON, steps_checked=0, checks=[], mismatches=[], frames_checked=0,
                  max_state_gap=0., max_integration_gap=0.)
    proof = dict(actor_constructor_attempts=0, actor_query_attempts=0, worker_spawn_attempts=0,
                 model_weight_open_attempts=0, actor_queries=0, subprocesses=[])
    save(out/'status.json', status)
    env = recorder = actor = runner = initial = None
    resources, handles, executed = [], [], []
    charged, clock, result = 0., None, 1
    try:
        clock = time.monotonic()
        with helper.budget(args.wall_seconds):
            previous = load(args.baseline/'status.json')
            original = argparse.Namespace(**previous['args'])
            baseline_rows = rows(args.baseline/'steps.jsonl')
            require(previous['status'] == 'completed' and original.seed == 0
                    and previous['language'] == design['instruction'], 'Invalid native baseline')
            runner, baseline_manifest = helper.frozen_runner(args.baseline)
            if args.smoke or args.replay:
                helper.zero_query_guard(runner, proof)
            fresh_device_preflight(args, out, design)
            for name in BRANCH_SOURCES:
                destination = out/'branch_source'/name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes((SCRIPT.parent/name).read_bytes())
            source_status = None
            if args.replay:
                source_status = load(args.replay/'status.json')
                version_status(source_status, args.replay)
                require(source_status['status'] == 'completed' and not source_status['partial_outputs'], 'Incomplete replay source')
                validate_prefix_marker(args.replay, prefix_record_value, source_status)
            status.update(baseline_device=dict(gpu=original.gpu, egl_device=original.egl_device), scene_seed=original.seed)
            original.gpu, original.egl_device, original.out = args.gpu, args.egl_device, out
            base = runner.local_environment(original, out)
            identity = load(out/'device_identity.json')
            os.environ['CUDA_VISIBLE_DEVICES'] = identity['physical_uuid']
            status['device_identity'] = identity
            report.update(actual_device=identity, exact_scope='all_prefix_frames_state_integration_controller_images_diagnostics_contacts')
            files = ('source_manifest.json', 'status.json', 'initial_source.json', 'model.xml', 'steps.jsonl',
                     'frames.jsonl', 'actions.npz', 'states.npz', 'images.npz', 'initialization/steps.jsonl',
                     'initialization/frames.jsonl', 'initialization/states.npz', 'initialization/images.npz')
            inputs = {name: sha(args.baseline/name) for name in files}
            save(out/'baseline_inputs.json', inputs)
            if args.replay:
                require(inputs == load(args.replay/'baseline_inputs.json'), 'Replay baseline input drift')
            with np.load(args.baseline/'actions.npz', allow_pickle=False) as archive:
                for key, field in (('mapped', 'action'), ('native', 'native_proposal'), ('tokens', 'tokens')):
                    np.testing.assert_array_equal(archive[key][:args.snapshot_step+1],
                        np.asarray([row[field] for row in baseline_rows[:args.snapshot_step+1]]).reshape(-1, 7))
            manifest = runner.freeze_sources(out, base)
            runner.seed_all(original.seed)
            status['phase'] = 'reconstruction'
            save(out/'status.json', status)
            env, obs = runner.make_scene(original, out, base)
            require(sha(out/'model.xml') == baseline_manifest['model_xml_sha256']
                    and load(out/'initial_source.json') == load(args.baseline/'initial_source.json'), 'Native scene drift')
            require(env.language_instruction == design['instruction'], 'Instruction drift')
            runner.record_scene_sources(env, out, manifest)
            status.update(language=env.language_instruction, checkpoint_revision=runner.REVISION)
            diag = runner.Diagnostics(env)
            recorder = runner.ContactRecorder(env.sim, keywords=(*diag.relevant, 'gripper'))
            init_rows = rows(args.baseline/'initialization/steps.jsonl')
            require(len(init_rows) == runner.SETTLE == 10, 'Expected ten original initialization actions')
            obs = helper.replay_segment(runner, env, obs, diag, recorder, args.baseline/'initialization',
                init_rows, report, 'initialization', out, resources, handles)
            obs = helper.replay_segment(runner, env, obs, diag, recorder, args.baseline,
                baseline_rows[:args.snapshot_step+1], report, 'baseline', out, resources, handles)
            report.update(status='reconstruction_mismatch' if report['mismatches'] else 'exact_reconstruction_passed',
                root_fingerprint=runner.fingerprint(env, obs), baseline_frame=args.snapshot_step+1,
                baseline_prefix_actions=args.snapshot_step+1)
            save(out/'reconstruction.json', report)
            require(not report['mismatches'] and report['root_fingerprint'] == design['snapshots'][root]['fingerprint'],
                    'Nonexact original boundary reconstruction')
            initial = diag.measure(obs)
            status.update(initial_diagnostics=initial, initial=runner.fingerprint(env, obs))
            save(out/'snapshot_controller.json', helper.controller_fields(env))
            if args.replay:
                executed = rows(args.replay/'steps.jsonl')
                validate_tape(args.replay, source_status, executed)
                require(len(runner.ReplayTape(args.replay).frames) == len(executed)+1, 'Incomplete replay tape')
                obs = helper.replay_segment(runner, env, obs, diag, recorder, args.replay, executed,
                    report, 'branch', out, resources, handles, args.snapshot_step+1)
                report.update(status='replay_mismatch' if report['mismatches'] else 'exact_replay_passed',
                              branch_steps=len(executed), branch_source=str(args.replay))
                require(not report['mismatches'], 'Independent replay mismatch')
                status.update(status='completed', termination='exact_replay_passed', replayed_steps=len(executed))
            elif args.smoke:
                status.update(status='completed', termination='exact_reconstruction_smoke_passed')
            else:
                require(not initial['native_success'], 'Frozen local boundary must be nonterminal')
                tape = runner.Tape(out)
                resources.append(('branch_tape', tape))
                tape.capture(env, obs, initial)
                stream, queries = (out/'steps.jsonl').open('w'), (out/'queries.jsonl').open('w')
                handles.extend([stream, queries])
                (out/'policy_inputs').mkdir()
                guard = PrefixGuard(prefix_record_value, out, args.plan)
                compare_prefix(prefix_record_value['directory'], out, 0)
        charged += time.monotonic()-clock
        clock = None
        if status['status'] != 'completed':
            status['phase'] = 'model_load'
            save(out/'status.json', status)
            start = time.monotonic()
            helper.validate_model_sources(runner, original, out, manifest, baseline_manifest)
            actor = helper.CandidateActor.__new__(helper.CandidateActor)
            helper.CandidateActor.__init__(actor, runner, original, args, out, identity)
            status.update(actor_build=actor.build, model_load_s=time.monotonic()-start)
            clock = time.monotonic()
            with helper.budget(args.wall_seconds-charged):
                from adapters.libero_openvla import policy_image
                reference = helper.evidence_tape(runner, args.baseline, out) if candidate == 0 else None
                status['phase'] = 'branch'
                for index in range(HORIZON):
                    # This is before PNG construction, request logging/counting,
                    # or writing anything to the model worker for query 33.
                    guard.before_request(index)
                    image = policy_image(obs['agentview_image'])
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    png = buffer.getvalue()
                    (out/'policy_inputs'/f'{index:06d}.png').write_bytes(png)
                    step = args.snapshot_step+1+index
                    metadata = dict(step=step, branch_step=index, candidate_id=out.name,
                        candidate_owner='local_candidate', branch_phase='local', source=args.candidate,
                        seed=0 if candidate == 0 else args.seed+index, query_index=index,
                        policy_png_sha256=hashlib.sha256(png).hexdigest(),
                        policy_rgb_sha256=runner.array_sha(np.asarray(image)),
                        raw_rgb_sha256=runner.array_sha(obs['agentview_image']),
                        language=env.language_instruction, before=tape.last['fingerprint'],
                        checkpoint_revision=runner.REVISION, worker_pid=actor.process.pid,
                        requested_at=time.time(), episode_elapsed_at_request_s=charged+time.monotonic()-clock)
                    guard.input(index, metadata, png)
                    runner.write_row(queries, dict(event='requested', **metadata))
                    status['actor_queries'] += 1
                    save(out/'status.json', status)
                    start = time.monotonic()
                    decode = actor.query(png, env.language_instruction, args.candidate, metadata['seed'], index)
                    inference_s = time.monotonic()-start
                    with (out/'worker_responses.jsonl').open('a') as raw:
                        raw.write(json.dumps(dict(query_index=index, response=decode))+'\n')
                        raw.flush()
                        os.fsync(raw.fileno())
                    helper.validate_decode(decode, metadata)
                    if not decode.get('valid'):
                        raise helper.InvalidGeneration(decode.get('invalid_reason', 'invalid_generation'))
                    guard.output(index, decode)
                    extra = dict(**metadata, proposal=decode['action'], native_proposal=decode['native_action'],
                        tokens=decode['tokens'], decode=decode, inference_s=inference_s,
                        forced_diagnostic_handoff=False, repair_verified=False,
                        episode_elapsed_s=charged+time.monotonic()-clock)
                    try:
                        helper.validate_action(dict(**extra, action=decode['action'], executed_mapped_action=decode['action']), step)
                    except (ValueError, AssertionError, TypeError, KeyError) as error:
                        raise helper.InvalidGeneration(str(error)) from error
                    runner.write_row(queries, dict(event='returned', **extra))
                    if reference is not None:
                        helper.nominal_query_check(runner, reference, baseline_rows, metadata, decode, png, out, nominal)
                    obs, row = runner.step_once(env, obs, decode['action'], diag, recorder, tape, stream,
                        step, 'actor', {k: value for k, value in extra.items() if k not in ('step', 'before')})
                    executed.append(row)
                    status['steps_executed'] = len(executed)
                    if reference is not None:
                        helper.nominal_step_check(reference, baseline_rows[step], row, env, obs, out, nominal)
                    guard.after_step(index)
                    save(out/'status.json', status)
                    if row['diagnostics']['native_success']:
                        if not guard.passed:
                            raise PrefixMismatch('Early prefix termination differs from completed nonterminal v2 tape')
                        status['termination'] = 'native_success'
                        break
                    require(not row['done'], 'Environment done before native success')
                status.update(status='completed', termination=status['termination'] or 'action_budget')
            charged += time.monotonic()-clock
            clock = None
        result = 0
    except helper.BudgetExpired:
        status.update(status='censored', termination='wall_budget')
    except helper.WorkerTimeout as error:
        status.update(status='timed_out', termination=error.phase+'_timeout', error=str(error))
    except helper.InvalidGeneration as error:
        status.update(status='invalid_generation', termination='invalid_generation', error=str(error))
    except PrefixMismatch as error:
        status.update(status='invalid_identity', termination='prefix_mismatch', error=str(error))
    except helper.NominalMismatch as error:
        status.update(status='nominal_mismatch', termination='nominal_mismatch', error=str(error))
    except BaseException:
        status.update(status='infrastructure_failed', termination='infrastructure_failure', error=traceback.format_exc())
    finally:
        if clock is not None:
            charged += time.monotonic()-clock
        status['episode_wall_s'] = charged
        if candidate == 0 and not args.replay:
            nominal.update(status='exact_nominal_identity_passed' if status['status'] == 'completed'
                and not args.smoke and not nominal['mismatches'] and nominal['steps_checked'] == len(executed)
                else 'censored_unestablished', branch_status=status['status'])
            save(out/'nominal_identity.json', nominal)
        if initial is not None and runner is not None:
            status.update(runner.summarize(initial, executed))
        operations = [('actor', lambda: runner.cleanup_actor(actor))] if runner is not None else []
        operations += [('stream', handle.close) for handle in handles]
        operations += [(name, resource.close) for name, resource in resources]
        operations += [(name, resource.close) for name, resource in
                       [('contacts', recorder), ('environment', env)] if resource is not None]
        errors = []
        for name, operation in operations:
            try:
                operation()
            except BaseException:
                errors.append(dict(resource=name, error=traceback.format_exc()))
        if not (args.smoke or args.replay):
            np.savez_compressed(out/'actions.npz',
                mapped=np.asarray([r['action'] for r in executed], dtype=float).reshape(-1, 7),
                native=np.asarray([r['native_proposal'] for r in executed], dtype=float).reshape(-1, 7),
                tokens=np.asarray([r['tokens'] for r in executed], dtype=np.int64).reshape(-1, 7))
        if args.smoke or args.replay:
            proof['actor_queries'] = status['actor_queries']
            save(out/'zero_query_proof.json', proof)
        if errors:
            status.update(status='infrastructure_failed', cleanup_errors=errors)
            result = 1
        status.update(ended=time.time(), lifetime_actions=args.snapshot_step+1+len(executed),
                      partial_outputs=status['status'] != 'completed')
        save(out/'verification.json', report)
        save(out/'status.json', status)
    print(json.dumps(dict(status=status['status'], termination=status['termination'], out=str(out),
                          actor_queries=status['actor_queries'], frames_checked=report['frames_checked'])), flush=True)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == '--_egl-identity':
        require(argv[1] in ('0', '1') and 'CUDA_VISIBLE_DEVICES' not in os.environ, 'Invalid EGL identity probe')
        print(json.dumps(egl_identity(int(argv[1]))))
        return 0
    args = parse_args(argv)
    require_design()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    if args._child:
        return run(args)
    require(not args.out.exists(), 'Output must be unique')
    child = subprocess.Popen([sys.executable, '-B', '-u', str(SCRIPT), *argv, '--_child'],
        start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2'))
    try:
        return child.wait(timeout=(210 if args.replay or args.smoke else 540)-10)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit):
        return 1
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'v3 branch refused: {error}', file=sys.stderr)
        raise SystemExit(1)
