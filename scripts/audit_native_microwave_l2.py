"""Offline L2A audit of the frozen 6 x 3 microwave candidate bank.

Reads saved JSON/NPZ/PNG evidence only. No runner, model, simulator, or GPU
library is imported. Labels are evaluation-only offline oracle measurements.
Missing/censored/invalid cells are retained, never counted as native failures.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

sys.dont_write_bytecode = True

OBJ = 'white_yellow_mug_1'
IN = '["in","white_yellow_mug_1","microwave_1_heating_region"]'
CLOSE = '["close","microwave_1"]'
REVISION = '80970322773f81baa2e22fe495d0487b93a05cfa'
BASE_REVISION = '47a0ec7fc4ec123775a391911046cf33cf9ed83f'
DESIGN_FREEZE_SHA = 'c1fb0739a51d07e73a4ba6a861771b31beb4a5d51aa46e1745382fda284c9f49'
LANGUAGE = 'put the yellow and white mug in the microwave and close it'
PROMPT = f'In: What action should the robot take to {LANGUAGE}?\nOut:'
PHYSICAL_FIELDS = ('step', 'action', 'executed_mapped_action', 'before', 'after',
                   'diagnostics', 'events', 'reward', 'done', 'contacts_substeps')
NOMINAL_FIELDS = PHYSICAL_FIELDS + ('tokens', 'native_proposal', 'proposal')


def load(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=4096)
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_rows(path):
    """Stream large contact logs; retain their exact canonical content digest."""
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if 'contacts_substeps' in row:
                    row['contacts_substeps'] = canonical_sha(row['contacts_substeps'])
                rows.append(row)
    return rows


def requested_count(path):
    """Retain attempted-call accounting even with a torn final JSONL write."""
    count, exact = 0, True
    if not path.exists():
        return count, exact
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                count += json.loads(line).get('event') == 'requested'
            except (ValueError, AttributeError):
                # Unknown event: charge a possible request, never silently erase it.
                count += 1
                exact = False
    return count, exact


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(result, name, operation):
    """A missing/corrupt artifact fails its check without hiding other cells."""
    try:
        detail = operation()
        result['checks'][name] = dict(passed=True, detail=detail)
        return detail
    except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError,
            AttributeError, StopIteration, EOFError, BadZipFile) as error:
        result['checks'][name] = dict(passed=False, error=f'{type(error).__name__}: {error}')
        return None


def table_contact(diagnostic):
    return any((OBJ in c['geom1'] and 'table_collision' in c['geom2']) or
               (OBJ in c['geom2'] and 'table_collision' in c['geom1'])
               for c in diagnostic['contacts_endpoint'])


def held_clear(diagnostic, first_grasp_z):
    return bool(diagnostic['grasped'][OBJ] and
                diagnostic['object_observations'][OBJ+'_pos'][2] > first_grasp_z + .02
                and not table_contact(diagnostic))


def local_labels(initial, diagnostics, first_grasp_z, horizon):
    """Nested h-action labels, with success absorbing only inside this prefix.

    All attainment/loss times are ONE-BASED local action counts. Snapshot-only
    attainment is zero. Short non-success prefixes produce unknown U and p_h.
    """
    prefix = diagnostics[:horizon]
    success = next((i+1 for i, d in enumerate(prefix) if d['native_success']), None)
    if success is not None:
        prefix = prefix[:success]
    known = len(prefix) == horizon or success is not None
    base = 2 if initial['predicates'][IN] else int(held_clear(initial, first_grasp_z))
    durable = prefix[-8:]
    in_dwell = len(durable) == 8 and all(d['predicates'][IN] for d in durable)
    held_dwell = len(durable) == 8 and all(held_clear(d, first_grasp_z) for d in durable)
    observed = 3 if success is not None else 2 if in_dwell else 1 if held_dwell else 0
    lost = {key: [i+1 for i, d in enumerate(prefix) if not d['predicates'][key]]
            for key in (IN, CLOSE) if initial['predicates'][key]}
    predicate_lost = any(lost.values())
    signals = {IN: lambda d: d['predicates'][IN], CLOSE: lambda d: d['predicates'][CLOSE],
               'grasp': lambda d: d['grasped'][OBJ],
               'held_clear': lambda d: held_clear(d, first_grasp_z),
               'native_success': lambda d: d['native_success']}
    components = {}
    for name, signal in signals.items():
        values = [bool(signal(d)) for d in [initial, *prefix]]
        components[name] = dict(initial=values[0], final=values[-1],
            first_attainment=next((i for i, value in enumerate(values) if value), None),
            loss_actions=[i for i in range(1, len(values)) if values[i-1] and not values[i]])
    return dict(requested_horizon=horizon, observed_branch_actions=len(prefix),
        available_branch_actions=len(diagnostics), snapshot_level=base,
        terminal_level=observed if known else None, observed_prefix_level=observed,
        useful_proxy=(observed > base and not predicate_lost) if known else None,
        censored=not known, native_success=success is not None,
        first_native_action=success, in_dwell_8=in_dwell, held_clear_dwell_8=held_dwell,
        snapshot_predicate_lost=predicate_lost, snapshot_predicate_false_actions=lost,
        components=components, short_reason='native_success' if success is not None and
        len(prefix) < horizon else 'censored' if not known else None)


def labels(initial, rows, z):
    return {str(h): local_labels(initial, [r['diagnostics'] for r in rows], z, h)
            for h in (12, 32)}


def identity(args, snapshots):
    for root, snap in snapshots.items():
        if (Path(args.get('baseline', '')).resolve() == Path(snap['source']).resolve()
                and args.get('snapshot_step') == snap['post_action_step']):
            for candidate in range(3):
                seed = 0 if candidate == 0 else 20261019 + 10000*root + 100*candidate
                if args.get('candidate') == ('greedy' if candidate == 0 else 'sample') and args.get('seed') == seed:
                    return root, candidate
    return None


def job_records(value):
    """Find launch records without assuming candidate-directory names."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from job_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from job_records(child)


def discover(run, snapshots):
    branches, replays, unknown = defaultdict(list), defaultdict(list), []
    statuses = {}
    for path in sorted(run.rglob('status.json')):
        if any(part in ('source', 'branch_source', 'reconstruction') for part in path.relative_to(run).parts[:-1]):
            continue
        try:
            status = load(path)
        except (OSError, ValueError) as error:
            unknown.append(dict(path=str(path), error=str(error)))
            continue
        args = status.get('args', {})
        if 'baseline' not in args or args.get('smoke'):
            continue
        directory = path.parent.resolve()
        statuses[str(directory)] = status
        if args.get('replay'):
            source = Path(args['replay'])
            if not source.is_absolute():
                source = run/source
            replays[str(source.resolve())].append(directory)
        else:
            key = identity(args, snapshots)
            if key is None:
                unknown.append(dict(path=str(directory), error='unplanned branch identity',
                                    actor_queries=status.get('actor_queries')))
            else:
                branches[key].append(directory)
    # Prefer explicit launcher associations when present. A record's source
    # pointer is still checked against replay status before accepting evidence.
    launch_links = defaultdict(list)
    for name in ('launch.json', 'launches.json', 'jobs.json', 'status.json'):
        path = run/name
        if not path.exists():
            continue
        try:
            records = job_records(load(path))
            for record in records:
                command = record.get('replay_command', record.get('command', record.get('argv', record.get('cmd'))))
                if isinstance(command, list) and '--replay' in command and '--out' in command:
                    source = Path(command[command.index('--replay')+1])
                    output = Path(command[command.index('--out')+1])
                elif record.get('replay') and isinstance(record.get('replay'), str):
                    source = Path(record['replay'])
                    output = Path(record.get('out', record.get('directory', '')))
                else:
                    continue
                source = source if source.is_absolute() else run/source
                output = output if output.is_absolute() else run/output
                launch_links[str(source.resolve())].append(output.resolve())
        except (OSError, ValueError, TypeError, IndexError):
            unknown.append(dict(path=str(path), error='malformed launch job evidence'))
    for source, outputs in launch_links.items():
        replays[source] = list(dict.fromkeys(outputs + replays[source]))
    return branches, replays, statuses, unknown


@lru_cache(maxsize=4096)
def frame_arrays(directory, index):
    """Hash raw array bytes, dtype and shape; never import the simulator."""
    import numpy as np
    directory = Path(directory)
    spool = directory/'frames'/f'{index:06d}.npz'
    arrays = {}
    if spool.exists():
        with np.load(spool, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    else:
        for name in ('states.npz', 'images.npz'):
            with np.load(directory/name, allow_pickle=False) as archive:
                arrays.update({key: archive[key][index] for key in archive.files})
    require({'state', 'integration', 'agentview_image', 'robot0_eye_in_hand_image'} <= arrays.keys(),
            f'Incomplete frame arrays: {directory}:{index}')
    return {key: dict(shape=list(a.shape), dtype=str(a.dtype),
                     sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
            for key, a in arrays.items()}


def verify_frames(directory, frames):
    for index, frame in enumerate(frames):
        require(frame['index'] == index, f'Frame index {index}')
        arrays = frame_arrays(str(directory), index)
        require(arrays['integration']['sha256'] == frame['fingerprint']['integration'],
                f'Integration array/fingerprint at {index}')
        for key in ('agentview_image', 'robot0_eye_in_hand_image'):
            require(arrays[key]['sha256'] == frame['fingerprint']['images'][key],
                    f'Image array/fingerprint {key} at {index}')
    return len(frames)


def compare_tapes(expected, actual, count, start=0, fields=PHYSICAL_FIELDS):
    expected, actual = Path(expected), Path(actual)
    left, right = load_rows(expected/'steps.jsonl')[start:start+count], load_rows(actual/'steps.jsonl')
    require(len(left) == len(right) == count, f'Action tape lengths: {actual}')
    for i, (a, b) in enumerate(zip(left, right)):
        require(all(a[key] == b[key] for key in fields),
                f'Tape step {i}: {[key for key in fields if a.get(key) != b.get(key)]}')
    lf = load_rows(expected/'frames.jsonl')[start:start+count+1]
    rf = load_rows(actual/'frames.jsonl')
    require(len(lf) == len(rf) == count+1, f'Frame lengths: {actual}')
    for i, (a, b) in enumerate(zip(lf, rf)):
        require(a['fingerprint'] == b['fingerprint'] and a['diagnostics'] == b['diagnostics'],
                f'Frame fingerprint/diagnostics mismatch {i}: {actual}')
        require(frame_arrays(str(expected), start+i) == frame_arrays(str(actual), i),
                f'Frame state/integration/images mismatch {i}: {actual}')
    return dict(actions=count, frames=count+1, raw_arrays_exact=True)


def report_exact(path, expected_status):
    report = load(path)
    require(report.get('status') == expected_status, f'{path.name}: {report.get("status")}')
    require(report.get('mismatches') == [] and report.get('max_state_gap') == 0
            and report.get('max_integration_gap') == 0 and report.get('actor_queries') == 0,
            f'Nonexact or nonzero-query report: {path}')
    return report


def reconstruction(directory, snap):
    report = report_exact(directory/'reconstruction.json', 'exact_reconstruction_passed')
    require(report['root_fingerprint'] == snap['fingerprint'] and
            report['baseline_frame'] == snap['state_index'], 'Reconstruction boundary differs')
    source = Path(snap['source'])
    compare_tapes(source/'initialization', directory/'reconstruction/initialization', 10)
    compare_tapes(source, directory/'reconstruction/baseline', snap['state_index'])
    require(report['frames_checked'] == 10+1+snap['state_index']+1,
            'Reconstruction frame accounting')
    return dict(exact=True, actor_queries=0, prefix_actions=snap['state_index'])


def baseline_context(snap):
    source = Path(snap['source'])
    for name, expected in snap['source_sha256'].items():
        require(sha(source/name) == expected, f'Frozen baseline hash: {name}')
    rows, frames = load_rows(source/'steps.jsonl'), load_rows(source/'frames.jsonl')
    status = load(source/'status.json')
    g, t, s = snap['first_grasp_step'], snap['post_action_step'], snap['state_index']
    require([r['step'] for r in rows] == list(range(len(rows))), 'Baseline step sequence')
    require(next(i for i, r in enumerate(rows) if r['diagnostics']['grasped'][OBJ]) == g,
            'Original first grasp differs')
    require(t == g+180 and s == t+1 and not any(r['diagnostics']['native_success'] for r in rows[:s]),
            'Noncausal or terminal snapshot')
    require(frames[s]['fingerprint'] == rows[t]['after'] == rows[s]['before'] == snap['fingerprint'],
            'Snapshot fingerprint differs')
    initial = rows[t]['diagnostics']
    require(frames[s]['diagnostics'] == initial, 'Snapshot diagnostics differ')
    require(status['status'] == 'completed' and status['args']['seed'] == 0 and
            status['args']['init_index'] == snap['root'] and status['language'] == LANGUAGE,
            'Baseline scene/instruction contract')
    z = rows[g]['diagnostics']['object_observations'][OBJ+'_pos'][2]
    return dict(initial=initial, first_grasp_z=z, original_labels=labels(initial, rows[s:s+32], z),
                original_count=min(32, len(rows)-s), manifest=load(source/'source_manifest.json'))


def validate_queries(directory, status, rows, root, candidate):
    import numpy as np
    from PIL import Image
    events = load_rows(directory/'queries.jsonl')
    requests = [r for r in events if r.get('event') == 'requested']
    returns = [r for r in events if r.get('event') == 'returned']
    require(len(requests)+len(returns) == len(events), 'Unknown query event')
    require([r['query_index'] for r in requests] == list(range(len(requests))), 'Repeated/skipped requests')
    require([r['query_index'] for r in returns] == list(range(len(returns))), 'Repeated/skipped returns')
    require(len(rows) == len(returns) == len(requests) == status['actor_queries'] <= 32,
            'Completed cell query/action accounting')
    require([r['event'] for r in events] == ['requested', 'returned']*len(rows),
            'Queries must be sequential closed-loop calls')
    source = 'greedy' if candidate == 0 else 'sample'
    settings = dict(do_sample=candidate != 0, top_k=50, num_beams=1, num_return_sequences=1)
    if candidate:
        settings.update(temperature=.7, top_p=.95)
    with np.load(directory/'actions.npz', allow_pickle=False) as tape:
        for key, field in (('mapped', 'action'), ('native', 'native_proposal'), ('tokens', 'tokens')):
            require(np.array_equal(tape[key], np.asarray([r[field] for r in rows]).reshape(-1, 7)),
                    f'Actions archive disagrees: {key}')
    for i, row in enumerate(rows):
        seed = 0 if candidate == 0 else 20261019+10000*root+100*candidate+i
        require(row['query_index'] == row['branch_step'] == i and row['seed'] == seed
                and row['source'] == source, f'Source/seed schedule at {i}')
        require(row['owner'] == 'vla' and row['candidate_owner'] == 'local_candidate'
                and row['phase'] == 'actor' and row['branch_phase'] == 'local'
                and not row['forced_diagnostic_handoff'] and row['repair_verified'] is False,
                f'Local action ownership at {i}')
        require(row['candidate_id'] == status['candidate_id'], 'Candidate attribution')
        for event in (requests[i], returns[i]):
            require(all(row.get(key) == value for key, value in event.items() if key != 'event'),
                    f'Query/action attribution at {i}')
        action, native = np.asarray(row['action']), np.asarray(row['native_proposal'])
        require(action.shape == native.shape == (7,) and np.isfinite(action).all()
                and np.isfinite(native).all(), f'Invalid seven-coordinate action at {i}')
        expected = native.copy()
        expected[-1] = -np.sign(2*expected[-1]-1)
        require(np.array_equal(action, expected) and row['action'] == row['proposal'] ==
                row['executed_mapped_action'], f'Official mapping at {i}')
        require(len(row['tokens']) == 7 and all(type(t) is int for t in row['tokens']), 'Seven integer tokens')
        require(row['language'] == LANGUAGE and row['checkpoint_revision'] == REVISION,
                'Instruction/checkpoint differs')
        png = directory/'policy_inputs'/f'{i:06d}.png'
        require(sha(png) == row['policy_png_sha256'], f'Policy PNG hash at {i}')
        with Image.open(png) as image:
            rgb_hash = hashlib.sha256(np.ascontiguousarray(image.convert('RGB')).tobytes()).hexdigest()
        require(rgb_hash == row['policy_rgb_sha256'] and row['raw_rgb_sha256'] ==
                row['before']['images']['agentview_image'], f'Policy input provenance at {i}')
        decode = row['decode']
        require(decode['sampling'] == settings, f'Explicit generation kwargs at {i}')
        config = decode['resolved_generation_config']
        require(all(config.get(k) == v for k, v in settings.items()) and config['max_new_tokens'] == 7,
                f'Resolved generation configuration at {i}')
        require(decode['prompt'] == PROMPT and decode.get('valid') is True, 'Prompt/valid generation')
        require(decode['action'] == row['action'] and decode['native_action'] == row['native_proposal']
                and decode['tokens'] == row['tokens'], f'Decode action mismatch at {i}')
        for key in ('source', 'seed', 'query_index', 'policy_png_sha256', 'policy_rgb_sha256'):
            require(decode[key] == row[key], f'Decode provenance {key} at {i}')
    return dict(requested=len(requests), returned=len(returns), executed=len(rows))


def provenance(directory, status, snap, context, pinned_sources):
    inputs = load(directory/'baseline_inputs.json')
    require(inputs and all(sha(Path(snap['source'])/name) == expected for name, expected in inputs.items()),
            'Baseline input hashes differ')
    require(all(inputs.get(name) == expected for name, expected in snap['source_sha256'].items()),
            'Baseline input manifest lacks frozen source pins')
    manifest = load(directory/'source_manifest.json')
    require(manifest['repository'].keys() == context['manifest']['repository'].keys(),
            'Archived baseline code inventory differs')
    for name, entry in manifest['repository'].items():
        require(entry['sha256'] == context['manifest']['repository'][name]['sha256'] ==
                sha(directory/'source'/name), f'Archived baseline code differs: {name}')
    for section in ('checkpoint', 'actor_code', 'runtime', 'assets'):
        require(manifest[section] == context['manifest'][section], f'{section} provenance differs')
    for path, expected in manifest['actor_code'].items():
        require(sha(directory/'source/actor_code'/Path(path).name) == expected, 'Archived official actor code drift')
    require(load(directory/'initial_source.json') == load(Path(snap['source'])/'initial_source.json'),
            'Official initial-state provenance differs')
    require(sha(directory/'model.xml') == context['manifest']['model_xml_sha256'], 'Compiled scene differs')
    require(status['scene_seed'] == 0 and status['language'] == LANGUAGE and
            status['checkpoint_revision'] == REVISION, 'Scene seed/language/checkpoint')
    build = status['actor_build']
    require(build['transformers'] == '4.40.1' and build['H'] == build['batch'] == 1
            and build['frozen'] and build['official_predict_action'] and build['unnorm_key'] == 'libero_10'
            and Path(build['snapshot']).name == REVISION and Path(build['code_snapshot']).name == BASE_REVISION,
            'Frozen actor/runtime build differs')
    sources = status['branch_sources']
    require(sources, 'Missing archived branch sources')
    for name, expected in sources.items():
        require(sha(directory/'branch_source'/name) == expected, f'Archived branch source drift: {name}')
        if pinned_sources is not None:
            require(pinned_sources.get(Path(name).name) == expected, f'Launcher source pin differs: {name}')
    return dict(baseline_pinned=True, branch_sources=sources, launch_plan_pin=pinned_sources is not None)


def nominal_identity(directory, snap, context, rows, computed_labels):
    marker = load(directory/'nominal_identity.json')
    require(marker.get('mismatches') == [] and marker.get('status') == 'exact_nominal_identity_passed'
            and marker.get('steps_checked') == len(rows), 'Nominal identity marker did not pass')
    require(len(marker['checks']) == len(rows), 'Incomplete nominal per-query evidence')
    original = load_rows(Path(snap['source'])/'steps.jsonl')
    for row, item in zip(rows, marker['checks']):
        expected = original[row['step']]
        values = dict(tokens=expected['tokens'], native_action=expected['native_proposal'],
                      mapped_action=expected['action'], before=expected['before'],
                      policy_png_sha256=row['policy_png_sha256'])
        require(item['expected'] == item['actual'] == values and item['action_executed'] is True
                and item['differences'] == {} and item['after_differences'] == {},
                'Nominal per-query values disagree with original/fresh tapes')
    require(len(rows) == context['original_count'], 'Nominal suffix stopping boundary differs')
    compared = compare_tapes(snap['source'], directory, len(rows), snap['state_index'], NOMINAL_FIELDS)
    require(computed_labels == context['original_labels'], 'Fresh/original nominal labels differ')
    return dict(exact=True, independent_comparison=compared, labels_exact=True, marker=marker)


def replay_check(directory, source, snap, context, status, rows, computed_labels):
    replay_status = load(directory/'status.json')
    require(Path(replay_status['args']['replay']).resolve() == source.resolve(), 'Replay source pointer differs')
    require(replay_status['status'] == 'completed' and replay_status['termination'] == 'exact_replay_passed',
            'Fresh replay incomplete')
    require(replay_status['pid'] != status['pid'], 'Replay is not a fresh process')
    require(replay_status['actor_queries'] == 0 and replay_status['replayed_steps'] == len(rows),
            'Replay query/action accounting')
    proof = load(directory/'zero_query_proof.json')
    require(all(proof[key] == 0 for key in ('actor_constructor_attempts', 'actor_query_attempts',
            'worker_spawn_attempts', 'model_weight_open_attempts', 'actor_queries')), 'Nonzero replay actor use')
    require(isinstance(proof['subprocesses'], list) and
            not any('worker.py' in call for call in proof['subprocesses']), 'Replay spawned a model worker')
    report = report_exact(directory/'verification.json', 'exact_replay_passed')
    require(Path(report['branch_source']).resolve() == source.resolve() and report['branch_steps'] == len(rows),
            'Replay report source/count differs')
    require(replay_status['branch_sources'] == status['branch_sources'], 'Replay implementation differs')
    for name, expected in status['branch_sources'].items():
        require(sha(directory/'branch_source'/name) == expected, 'Replay archived source drift')
    reconstruction(directory, snap)
    actual = directory/'reconstruction/branch'
    compared = compare_tapes(source, actual, len(rows))
    recomputed = labels(context['initial'], load_rows(actual/'steps.jsonl'), context['first_grasp_z'])
    require(recomputed == computed_labels, 'Independently recomputed replay labels differ')
    require(report['frames_checked'] == 10+1+snap['state_index']+1+len(rows)+1,
            'Replay frame accounting')
    return dict(exact=True, directory=str(directory), zero_queries=True,
                independent_comparison=compared, labels=recomputed)


def completed_tape(directory, status, snap, context, rows):
    require(status['status'] == 'completed', 'Branch did not complete')
    require(status['args']['local_horizon'] == 32 and status['args']['continuation_steps'] == 0
            and status['continuation_cap'] == 0, 'Not the frozen L2A horizon')
    require(len(rows) == status['steps_executed'] <= status['actor_queries'] <= 32,
            'Steps/query cap or accounting differs')
    require(status['initial'] == snap['fingerprint'] and status['initial_diagnostics'] == context['initial'],
            'Initial boundary differs')
    require([r['step'] for r in rows] == list(range(snap['state_index'], snap['state_index']+len(rows))),
            'Nonsequential local actions')
    native = [i for i, r in enumerate(rows) if r['diagnostics']['native_success']]
    require((status['termination'] == 'native_success' and native == [len(rows)-1]) or
            (status['termination'] == 'action_budget' and len(rows) == 32 and not native),
            'Requires 32-action completion or actual native early termination')
    require(not status.get('partial_outputs') and status.get('handoff') is None, 'Partial tape or handoff')
    frames = load_rows(directory/'frames.jsonl')
    require(len(frames) == len(rows)+1 and frames[0]['fingerprint'] == snap['fingerprint']
            and frames[0]['diagnostics'] == context['initial'], 'Incomplete frame tape/boundary')
    for i, row in enumerate(rows):
        require(row['before'] == frames[i]['fingerprint'] and row['after'] == frames[i+1]['fingerprint']
                and row['diagnostics'] == frames[i+1]['diagnostics'], f'Frame/action alignment {i}')
        d = row['diagnostics']
        require(d['native_success'] == (d['predicates'][IN] and d['predicates'][CLOSE]),
                'Native conjunction differs from logged predicates')
    verify_frames(directory, frames)
    return dict(completed=True, actions=len(rows), termination=status['termination'])


def audit_cell(root, candidate, paths, replay_paths, snap, context, pinned_sources):
    cell = dict(root=root, candidate=f'C{candidate}', state='missing', valid=False,
                paths=[str(p) for p in paths], checks={}, labels=None, actor_queries=0, actions=0)
    if not paths:
        return cell
    if len(paths) != 1:
        cell.update(state='duplicate_identity', checks=dict(identity=dict(passed=False,
                    error='Multiple runs occupy one preallocated identity; no outcome-based choice')))
        return cell
    directory = paths[0]
    status = load(directory/'status.json')
    cell.update(state='incomplete', directory=str(directory), termination=status.get('termination'),
                actor_queries=status.get('actor_queries', 0))
    rows = check(cell, 'read_tape', lambda: load_rows(directory/'steps.jsonl'))
    # Do not duplicate full tapes in the report.
    if rows is not None:
        cell['checks']['read_tape']['detail'] = dict(rows=len(rows))
        cell['actions'] = len(rows)
    if context is None:
        cell['checks']['baseline'] = dict(passed=False, error='Frozen baseline unavailable or inconsistent')
        return cell
    if rows is not None:
        cell['labels'] = check(cell, 'labels', lambda: labels(context['initial'], rows, context['first_grasp_z']))
        check(cell, 'completion', lambda: completed_tape(directory, status, snap, context, rows))
        check(cell, 'queries', lambda: validate_queries(directory, status, rows, root, candidate))
        if rows:
            cell['boundary'] = {key: rows[0].get(key) for key in
                                ('tokens', 'action', 'native_proposal', 'policy_png_sha256', 'policy_rgb_sha256')}
    check(cell, 'reconstruction', lambda: reconstruction(directory, snap))
    check(cell, 'provenance', lambda: provenance(directory, status, snap, context, pinned_sources))
    if candidate == 0 and rows is not None:
        check(cell, 'nominal_identity', lambda: nominal_identity(directory, snap, context, rows, cell['labels']))
    candidates = list(dict.fromkeys(replay_paths.get(str(directory.resolve()), [])))
    cell['replay_paths'] = [str(p) for p in candidates]
    def verify_replay():
        require(len(candidates) == 1, 'Exactly one fresh replay required; missing or ambiguous evidence')
        require(rows is not None and cell['labels'] is not None, 'Cannot compare unreadable source tape')
        return replay_check(candidates[0], directory, snap, context, status, rows, cell['labels'])
    check(cell, 'fresh_replay', verify_replay)
    cell['valid'] = all(c['passed'] for c in cell['checks'].values())
    cell['state'] = ('valid' if cell['valid'] else 'censored' if status.get('termination') in
                     ('wall_budget', 'query_timeout', 'outer_timeout') else 'invalid_generation'
                     if status.get('termination') == 'invalid_generation' else 'infrastructure_failed'
                     if status.get('status') == 'infrastructure_failed' else 'nominal_mismatch'
                     if status.get('status') == 'nominal_mismatch' else 'integrity_unverified')
    return cell


def continuation_gate(cells, integrity):
    indexed = {(c['root'], c['candidate']): c for c in cells}
    full = len(cells) == 18 and set(indexed) == {(r, f'C{i}') for r in range(6) for i in range(3)}
    required = ('completion', 'queries', 'reconstruction', 'provenance', 'fresh_replay', 'labels')
    full = full and all(c['valid'] and all(c.get('checks', {}).get(key, {}).get('passed') is True
                                         for key in required) for c in cells)
    greedy = all(indexed.get((r, 'C0'), {}).get('checks', {}).get('nominal_identity', {}).get('passed') is True
                 for r in range(6))
    gains = []
    for root in (0, 4):
        control = indexed.get((root, 'C0'), {})
        if (control.get('labels') or {}).get('32', {}).get('useful_proxy') is not False:
            continue
        for candidate in ('C1', 'C2'):
            cell = indexed.get((root, candidate), {})
            if (cell.get('labels') or {}).get('32', {}).get('useful_proxy') is True:
                gains.append(dict(root=root, candidate=candidate))
    return dict(pass_L2B=bool(integrity and full and greedy and gains), all_18_valid=full,
                all_6_greedy_exact=greedy, global_integrity=integrity,
                sampled_failure_root_gains=gains, automatic_launch=False)


def source_pins(plan, run, snapshots_file, manifest):
    """Read file/hash entries from a launcher freeze; never execute archived code."""
    pins = {}
    def visit(value, key=None):
        if not isinstance(value, dict):
            return
        for name, entry in value.items():
            if isinstance(entry, str) and len(entry) == 64 and name.endswith('.py'):
                pins[Path(name).name] = entry
            elif isinstance(entry, dict):
                path = entry.get('path', name)
                if str(path).endswith('.py') and isinstance(entry.get('sha256'), str):
                    pins[Path(path).name] = entry['sha256']
                visit(entry, name)
    require(plan['protocol'] == manifest['protocol'] and plan['scope'] == 'L2A_only' and
            plan['snapshots'] == manifest['snapshots'] and plan['phase_order'] == ['C0', 'C1', 'C2'],
            'Launch plan protocol/snapshots/order differ')
    require(plan['snapshot_manifest_sha256'] == sha(snapshots_file) and
            plan['design_freeze_sha256'] == sha(snapshots_file.parent/'design_freeze.json'),
            'Launch plan design pins differ')
    require(plan['limits']['calls'] == 576 and plan['limits']['cells'] == 18 and
            plan['limits']['local_horizon'] == 32 and plan['limits']['continuation_steps'] == 0 and
            plan['limits']['retries'] == 0, 'Launch plan budget differs')
    for key in ('sampling', 'instruction', 'checkpoint_revision', 'base_code_revision'):
        require(plan[key] == manifest[key], f'Launch plan {key} differs')
    require(len(plan['cells']) == 18, 'Launch plan identity count')
    planned = set()
    for cell in plan['cells']:
        root, candidate = cell['root'], int(cell['candidate'][1:])
        require(root in range(6) and candidate in range(3), 'Unplanned identity')
        expected = 0 if candidate == 0 else 20261019+10000*root+100*candidate
        require(cell['seed'] == expected and cell['seeds'] ==
                [0 if candidate == 0 else expected+k for k in range(32)], 'Launch plan seed schedule')
        planned.add((root, candidate))
    require(len(planned) == 18, 'Duplicate planned identities')
    # Read source/config artifacts only, never model weights.
    for name, expected in plan['frozen_files_sha256'].items():
        path = Path(name)
        require(path.suffix not in ('.safetensors', '.bin'), 'Unexpected model-weight content pin')
        require(sha(path) == expected, f'Frozen launch input drift: {path}')
    for name, entry in plan.get('checkpoint_weight_metadata', {}).items():
        path = Path(name)
        require(str(path.resolve()) == entry['resolved_path'] and path.stat().st_size == entry['bytes']
                and path.stat().st_mtime_ns == entry['stat_mtime_ns'], 'Checkpoint metadata drift')
    visit(plan['archive_sources'])
    require({'run_native_long_branch.py', 'native_long_candidate_worker.py'} <= pins.keys(),
            'Launch plan exists but lacks branch/worker implementation pins')
    for name, expected in plan['archive_sources'].items():
        require(plan['frozen_files_sha256'].get(str(Path(plan['repo'])/name)) == expected,
                'Launch archive/source pins disagree')
    launch = run/'launch.json'
    if launch.exists():
        recorded = load(launch)
        require(recorded['plan'] == plan and recorded['plan_sha256'] == sha(run/'plan.json'),
                'Launch plan attribution differs')
    return pins


def audit(run, snapshots_file):
    manifest = load(snapshots_file)
    snapshots = {s['root']: s for s in manifest['snapshots']}
    require(set(snapshots) == set(range(6)) and len(manifest['snapshots']) == 6, 'Exactly six frozen roots required')
    report = dict(scope='offline_model_free_no_simulator_no_GPU', protocol=manifest['protocol'],
        run=str(run), snapshots=str(snapshots_file), checks={}, cells=[], originals=[],
        interpretation='Offline oracle availability only; not framework efficacy or a deployable selector.',
        selector='none', model_queries_performed=0, simulator_actions_performed=0,
        auditor_sha256=sha(Path(__file__).resolve()))
    def design_check():
        require(manifest['protocol'] == 'native_microwave_l2_v1_20261019' and
                manifest['local_horizon'] == 32 and manifest['nested_measurements'] == [12, 32]
                and manifest['instruction'] == LANGUAGE and manifest['checkpoint_revision'] == REVISION,
                'Frozen protocol configuration differs')
        require(sha(snapshots_file.parent/'design_freeze.json') == DESIGN_FREEZE_SHA,
                'Frozen design anchor differs')
        freeze = load(snapshots_file.parent/'design_freeze.json')
        for name, expected in freeze['files'].items():
            require(sha(Path(name)) == expected, f'Design freeze differs: {name}')
        require(freeze['files'].get(str(snapshots_file)) == sha(snapshots_file), 'Manifest not bound by design freeze')
        return dict(sha256=sha(snapshots_file), freeze_verified=True)
    check(report, 'design_freeze', design_check)
    pinned = None
    plan = None
    plans = [run/name for name in ('plan.json', 'launch_plan.json', 'execution_plan.json') if (run/name).exists()]
    if plans:
        def plan_check():
            require(len(plans) == 1, 'Ambiguous execution plan')
            return source_pins(load(plans[0]), run, snapshots_file, manifest)
        pinned = check(report, 'launch_plan', plan_check)
        if pinned is None:
            pinned = {}
        else:
            plan = load(plans[0])
    branches, replays, statuses, unknown = discover(run, snapshots)
    report['unassigned_evidence'] = unknown
    report['checks']['discovery'] = dict(passed=not unknown, detail=unknown)
    for root in range(6):
        snap = snapshots[root]
        context = check(report, f'baseline_root{root}', lambda: baseline_context(snap))
        if context is not None:
            report['checks'][f'baseline_root{root}']['detail'] = dict(frozen=True)
            report['originals'].append(dict(root=root, source=snap['source'], labels=context['original_labels']))
        for candidate in range(3):
            report['cells'].append(audit_cell(root, candidate, branches[(root, candidate)], replays,
                                              snap, context, pinned))
    total_calls, total_actions = 0, 0
    accounting_ok = True
    query_directories = {str(p.parent.resolve()) for p in run.rglob('queries.jsonl')
                         if not any(part in ('source', 'branch_source', 'reconstruction')
                                    for part in p.relative_to(run).parts[:-1])}
    for directory in sorted(set(statuses) | query_directories):
        status = statuses.get(directory, {})
        queries_path = Path(directory)/'queries.jsonl'
        try:
            requested, exact = requested_count(queries_path)
            accounting_ok &= exact and bool(status)
            if not status:
                report['unassigned_evidence'].append(dict(path=directory, error='Query tape without branch status'))
            count = status.get('actor_queries', 0)
            total_calls += requested
            require(type(count) is int and count >= 0, 'Invalid query count')
            total_calls += max(count-requested, 0)
            accounting_ok &= count == requested
            if status.get('args', {}).get('replay'):
                accounting_ok &= count == 0
            else:
                steps_path = Path(directory)/'steps.jsonl'
                actions = len(load_rows(steps_path)) if steps_path.exists() else 0
                total_actions += actions
                accounting_ok &= actions <= max(count, requested) <= 32
                accounting_ok &= status.get('steps_executed', 0) == actions
        except (OSError, ValueError, TypeError):
            accounting_ok = False
    report['budget'] = dict(requested_calls_conservative=total_calls, cap=576,
                           actions=total_actions, accounting_exact=accounting_ok)
    report['checks']['discovery']['passed'] = not report['unassigned_evidence']
    report['checks']['budget'] = dict(passed=accounting_ok and total_calls <= 576)
    report['boundary_diversity'] = []
    for root in range(6):
        cells = [c for c in report['cells'] if c['root'] == root]
        available = [c for c in cells if c.get('boundary')]
        pngs = [c['boundary']['policy_png_sha256'] for c in available]
        common = len(available) == 3 and None not in pngs and len(set(pngs)) == 1
        if plan is not None:
            boundary = next((b for b in plan['boundary_inputs'] if b['root'] == root), {})
            common = common and pngs[0] == boundary.get('policy_png_sha256') and boundary.get('prompt') == PROMPT
            common = common and boundary.get('prompt_sha256') == hashlib.sha256(PROMPT.encode()).hexdigest()
        diversity = dict(root=root, observed_candidates=len(available), common_boundary_policy_png=common,
                         duplicates_retained=True)
        for field in ('tokens', 'action'):
            groups = defaultdict(list)
            for cell in available:
                groups[json.dumps(cell['boundary'][field], separators=(',', ':'))].append(cell['candidate'])
            diversity[field] = dict(unique_count=len(groups), duplicate_groups=[v for v in groups.values() if len(v)>1])
        report['boundary_diversity'].append(diversity)
        report['checks'][f'boundary_root{root}'] = dict(passed=common)
    report['oracle_availability'] = []
    for root in range(6):
        cells = [c for c in report['cells'] if c['root'] == root]
        for horizon in ('12', '32'):
            values = {c['candidate']: (c['labels'] or {}).get(horizon, {}).get('useful_proxy')
                      if c['valid'] else None for c in cells}
            oracle = True if True in values.values() else False if all(v is False for v in values.values()) else None
            report['oracle_availability'].append(dict(root=root, horizon=int(horizon), useful_by_candidate=values,
                any_useful=oracle, sampled_gains_over_C0=[key for key in ('C1', 'C2')
                    if values[key] is True and values['C0'] is False]))
    report['valid_candidate_count'] = sum(c['valid'] for c in report['cells'])
    report['cell_states'] = dict(Counter(c['state'] for c in report['cells']))
    report['gate'] = continuation_gate(report['cells'], all(c['passed'] for c in report['checks'].values()))
    return report


def markdown(report):
    text = ['# Native microwave L2A offline audit', '', report['interpretation'], '',
            f"Valid identities: **{report['valid_candidate_count']}/18**. "
            f"Requested model calls: **{report['budget']['requested_calls_conservative']}/576**. "
            f"L2B gate: **{report['gate']['pass_L2B']}**.", '',
            'Missing, invalid and censored cells are not native task failures. '
            'All labels below are tape measurements; only valid cells enter oracle summaries.', '',
            '| Root | Candidate | State | Actions | h | p_h | U_h | Native by h | Predicate loss |',
            '|---|---|---|---:|---:|---:|---|---|---|']
    def display(value):
        return 'unknown' if value is None else str(value).lower()
    for cell in report['cells']:
        for h in ('12', '32'):
            label = (cell['labels'] or {}).get(h, {})
            text.append(f"| {cell['root']} | {cell['candidate']} | {cell['state']} | {cell['actions']} | {h} | "
                        f"{display(label.get('terminal_level'))} | {display(label.get('useful_proxy'))} | "
                        f"{display(label.get('native_success'))} | {display(label.get('snapshot_predicate_lost'))} |")
    text += ['', '## Integrity findings', '']
    for name, item in report['checks'].items():
        if not item['passed']:
            text.append(f"- {name}: {item.get('error', 'not established')}")
    for cell in report['cells']:
        for name, item in cell['checks'].items():
            if not item['passed']:
                text.append(f"- root {cell['root']} {cell['candidate']} / {name}: {item.get('error', 'not established')}")
    text += ['', 'Boundary uniqueness, duplicate groups, original/fresh nominal agreement, '
             'replay checks, predicate losses and per-root sampled gains are in `summary.json`. '
             'No runtime selector or automatic continuation is implemented.', '']
    return '\n'.join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New report directory; never overwrite')
    args = parser.parse_args(argv)
    require(args.run.is_dir(), '--run must be an existing directory')
    require(not args.out.exists(), '--out must be new')
    report = audit(args.run.resolve(), args.snapshots.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.out/'results.md').write_text(markdown(report))
    print(json.dumps(dict(out=str(args.out.resolve()), valid=report['valid_candidate_count'],
                         planned=18, pass_L2B=report['gate']['pass_L2B'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
