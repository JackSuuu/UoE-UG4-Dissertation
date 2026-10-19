"""Read native screening artifacts without importing the runner or simulator.

Usage: python scripts/summarize_native_long_cohort.py --run COHORT --out REPORT
The input tree is never modified. Missing/partial logs remain visible, including
while a cohort is running. Only status.json and diagnostic JSON tapes are read;
this script does not perform replay or infer success from sequential subgoals.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean


TASKS = {
    'microwave': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it',
    'two_pots': 'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove',
    'two_mugs': 'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate',
    'drawer': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it',
}
ROOTS = range(6)
ALIASES = {'moka': 'two_pots', 'mugs': 'two_mugs'}
COMPLETE = {'completed', 'complete', 'success', 'finished'}
INFRA = {'failed', 'error', 'infrastructure_failure', 'infrastructure_failed', 'timeout', 'killed', 'cancelled'}


def task_key(value):
    if not isinstance(value, str):
        return None
    value = ALIASES.get(value, value)
    return next((key for key, name in TASKS.items() if value in (key, name)), None)


def mapping(value):
    return value if isinstance(value, dict) else {}


def boolean(value):
    # Do not turn null, error strings or nonempty diagnostic dictionaries true.
    return value if isinstance(value, bool) else None


def integer(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value) if str(value).lstrip('-').isdigit() else None
    except (ValueError, TypeError):
        return None


def read_json(path, issues):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value
    except (OSError, ValueError) as error:
        issues.append(dict(kind='unreadable_json', path=str(path), detail=str(error)))
        return {}


def layers(record):
    result = [mapping(record)]
    for key in ('diagnostics', 'diagnostic', 'measured', 'observation', 'after_diagnostics'):
        if isinstance(record.get(key), dict):
            result.append(record[key])
    return result


def bool_map(value):
    if isinstance(value, list):
        value = {str(item.get('name', item.get('predicate', index))): item
                 for index, item in enumerate(value) if isinstance(item, dict)}
    result = {}
    for name, item in mapping(value).items():
        if isinstance(item, dict):
            item = next((item[k] for k in ('value', 'satisfied', 'attained', 'grasped')
                         if k in item), None)
        if boolean(item) is not None:
            result[str(name)] = item
    return result


def diagnostics(record):
    predicates, grasps, strict = {}, {}, {}
    native = None
    for layer in layers(record):
        for key in ('native_success', 'native_full_success', 'full_goal_success'):
            if boolean(layer.get(key)) is not None:
                native = layer[key]
                break
        for key in ('predicates', 'goal_predicates', 'constituent_predicates', 'native_predicates'):
            predicates.update(bool_map(layer.get(key)))
        for key in ('grasps', 'grasped', 'target_grasps', 'grasp_by_target', 'grasped_by_target'):
            grasps.update(bool_map(layer.get(key)))
        for name, target in mapping(layer.get('targets')).items():
            if boolean(mapping(target).get('grasped')) is not None:
                grasps[name] = target['grasped']
        for key in ('strict', 'strict_diagnostics', 'strict_success', 'strict_complete', 'completion'):
            if key in layer:
                strict[key] = layer[key]
    return dict(native_success=native, predicates=predicates, grasps=grasps, strict=strict)


def identify(path, status, run):
    args = mapping(status.get('args'))
    task = next((task_key(value) for value in
                 (status.get('task'), status.get('task_key'), status.get('scene'),
                  status.get('task_name'), args.get('task'), args.get('scene'))
                 if task_key(value) is not None), None)
    relative = str(path.parent.relative_to(run)) if path.parent.is_relative_to(run) else path.parent.name
    if task is None:
        task = next((task_key(key) for key in (*TASKS, *ALIASES) if re.search(
            r'(?:^|/)' + re.escape(key) + r'(?:/|_root[_-]?\d|[_-]\d|$)', relative)), None)
    root = next((integer(value) for value in
                 (status.get('init_index'), status.get('root_index'), status.get('index'),
                  args.get('init_index'), args.get('root_index'), args.get('index'))
                 if integer(value) is not None), None)
    if root is None:
        match = re.search(r'(?:^|[/_])root[_-]?(\d+)(?:/|$)', relative)
        if match:
            root = int(match.group(1))
    return task, root


def discover(run):
    found, issues, ignored = {}, [], []
    paths = set(run.rglob('status.json'))
    launch = read_json(run / 'launch.json', issues) if (run / 'launch.json').exists() else {}
    expected = {}
    for job in launch.get('jobs', []):
        directory = job.get('episode_dir')
        if not directory:
            continue
        path = Path(directory)
        path = (path if path.is_absolute() else run / path) / 'status.json'
        expected[path] = (task_key(job.get('task')), integer(job.get('init_index')))
        if path.exists():
            paths.add(path)
    for path in sorted(paths):
        relative = path.relative_to(run) if path.is_relative_to(run) else Path(path.parent.name) / path.name
        if any(part in ('source', 'sources', 'replay', 'report') or part.startswith('replay_')
               for part in relative.parts[:-1]):
            continue
        local = []
        status = read_json(path, local)
        task, root = identify(path, status, run)
        if path in expected and (task, root) != expected[path]:
            local.append(f'Status identity {(task, root)} disagrees with planned {expected[path]}: {path}')
        if task in TASKS and root in ROOTS:
            found.setdefault((task, root), []).append((path.parent, status, local))
        else:
            ignored.append(str(path))
            issues.extend(local)
    return found, issues, ignored


def read_steps(path, issues):
    if not path.exists():
        return []
    rows = []
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected JSON object')
                    step = integer(row.get('step', len(rows)))
                    if step is None or step != len(rows):
                        issues.append(dict(kind='nonsequential_step', line=line_number,
                                           expected=len(rows), observed=step))
                    rows.append(dict(step=step, line=line_number, **diagnostics(row)))
                except ValueError as error:
                    issues.append(dict(kind='malformed_step', line=line_number, detail=str(error)))
    except OSError as error:
        issues.append(dict(kind='unreadable_steps', path=str(path), detail=str(error)))
    return rows


def series(samples, field):
    names = sorted({name for sample in samples for name in sample[field]})
    result = {}
    for name in names:
        previous = None
        first_true = None
        attained, lost = [], []
        observed = 0
        for sample in samples:
            current = sample[field].get(name)
            observed += current is not None
            if current is True and first_true is None:
                first_true = sample['step']
            if previous is False and current is True:
                attained.append(sample['step'])
            if previous is True and current is False:
                lost.append(sample['step'])
            # Unknown observations break continuity rather than proving a transition.
            previous = current
        result[name] = dict(initial=samples[0][field].get(name),
            ever_true=first_true is not None if observed else None,
            first_true_step=first_true, attainment_steps=attained, loss_steps=lost,
            final=samples[-1][field].get(name), observed_samples=observed)
    return result


def replay_info(directory, status, run, task, root, job):
    paths = [directory/'verification.json', directory/'replay.json',
             directory/'replay'/'verification.json',
             run/f'replay_{directory.name}'/'verification.json',
             run/'replay'/task/f'root_{root}'/'verification.json']
    records, issues = [], []
    if job.get('replay_dir'):
        paths.append(Path(job['replay_dir'])/'verification.json')
    command = job.get('replay_command', [])
    if isinstance(command, list) and '--out' in command:
        index = command.index('--out') + 1
        if index < len(command):
            paths.append(Path(command[index])/'verification.json')
    embedded = status.get('replay', status.get('verification'))
    if embedded is not None:
        records.append(dict(path=str(directory/'status.json'), evidence=embedded))
    for path in dict.fromkeys(paths):
        if path.is_file():
            records.append(dict(path=str(path), evidence=read_json(path, issues)))
    states = []
    for record in records:
        value = record['evidence']
        item = mapping(value)
        label = str(item.get('status', value if isinstance(value, str) else '')).lower()
        failed = (item.get('passed') is False or item.get('verified') is False
                  or bool(item.get('mismatches'))
                  or any(token in label for token in ('fail', 'error', 'mismatch')))
        for key in ('max_state_gap', 'max_integration_gap'):
            if isinstance(item.get(key), (int, float)):
                failed |= item[key] != 0
        passed = (item.get('passed') is True or item.get('verified') is True
                  or label in ('passed', 'verified', 'exact_replay_passed', 'complete_verified'))
        states.append('issues' if failed else 'passed' if passed else 'unresolved')
    if job.get('replay_returncode') not in (None, 0):
        states.append('issues')
    elif job.get('status') == 'exact_verified':
        states.append('passed')
    elif job.get('replay_pid') and job.get('status') == 'infrastructure_or_replay_issue':
        states.append('issues')
    state = ('issues' if issues or 'issues' in states else
             'unresolved' if 'unresolved' in states else 'passed' if states else 'not_reported')
    return dict(state=state, records=records, read_issues=issues)


def opportunities(case, rows, budget):
    if (case['outcome'] != 'native_failure' or not case['meaningful_partial_progress']
            or case['data_issues']):
        return []
    result = []
    row_by_step = {row['step']: row for row in rows}
    for field, kind in (('predicates', 'constituent_predicate_loss'),
                        ('grasps', 'target_grasp_ended')):
        for name, item in case[field].items():
            for step in item['loss_steps']:
                row = row_by_step.get(step)
                if row is None or row['native_success'] is True:
                    continue
                result.append(dict(kind=kind, target_or_predicate=name, step=step,
                    phase='early' if step < budget/3 else 'mid' if step < 2*budget/3 else 'late',
                    phase_reference=f'zero-based action index / configured {budget}-action budget',
                    evidence=dict(path=case['artifacts']['steps'], line=row['line'],
                                  current=row[field].get(name), previous=True),
                    achieved_predicates=sorted(k for k, v in row['predicates'].items() if v),
                    unmet_predicates=sorted(k for k, v in row['predicates'].items() if not v),
                    branch_state_index=step+1,
                    remaining_action_budget=max(0, budget-step-1),
                    interpretation='Post-action diagnostic review candidate; recoverability and repair benefit untested.'))
    return sorted(result, key=lambda item: (item['step'], item['kind'], item['target_or_predicate']))


def episode(run, task, root, entries, job):
    expected = Path(job.get('episode_dir', run/task/f'root_{root}'))
    case = dict(task=task, root=root, lifecycle='unstarted', outcome='unobserved',
        native_success=None, meaningful_partial_progress=False, progress_evidence=[],
        steps=None, logged_steps=0, termination=None, predicates={}, grasps={},
        failure_observations=[], diagnostic_opportunities=[], data_issues=[],
        artifacts=dict(directory=str(expected), status=str(expected/'status.json'),
                       steps=str(expected/'steps.jsonl')), replay=dict(state='not_reported', records=[]))
    case['supervisor'] = {key: job[key] for key in ('name', 'status', 'blocked_by', 'error',
        'actor_returncode', 'replay_returncode', 'episode_dir') if key in job}
    if not entries:
        # A tape without a status is an incomplete artifact, not an unstarted run.
        if expected.exists():
            case['lifecycle'] = 'incomplete_artifacts'
            case['data_issues'].append(dict(kind='missing_status'))
        if job.get('status') == 'infrastructure_or_replay_issue':
            case['lifecycle'] = 'infrastructure_failure'
        elif job.get('status') in ('running', 'replaying'):
            case['lifecycle'] = 'incomplete'
        return case
    if len(entries) != 1:
        case.update(lifecycle='ambiguous_artifacts', data_issues=[dict(
            kind='multiple_episode_directories', paths=[str(entry[0]) for entry in entries])])
        return case
    directory, status, issues = entries[0]
    case['data_issues'] = list(issues)
    issues = case['data_issues']
    case['artifacts'] = dict(directory=str(directory), status=str(directory/'status.json'),
        steps=str(directory/'steps.jsonl'), files=[str(p) for p in sorted(directory.iterdir()) if p.is_file()])
    case['timing'] = {key: status[key] for key in ('model_load_s', 'scene_setup_s', 'episode_wall_s',
                                                'total_wall_s', 'started', 'ended') if key in status}
    case['actor_queries'] = status.get('actor_queries')
    case['initialization_actions'] = status.get('initialization_actions')
    rows = read_steps(directory/'steps.jsonl', issues)
    initial = status.get('initial_diagnostics', status.get('initial_observation', {}))
    samples = [dict(step=-1, **diagnostics(mapping(initial)))] + rows
    case['predicates'], case['grasps'] = series(samples, 'predicates'), series(samples, 'grasps')
    label = str(status.get('status', 'unknown')).lower()
    case['lifecycle'] = ('completed' if label in COMPLETE else
                         'infrastructure_failure' if label in INFRA else
                         'incomplete_artifacts' if label == 'unknown' else 'incomplete')
    case['reported_status'] = label
    if case['lifecycle'] != 'completed' and job.get('status') == 'infrastructure_or_replay_issue':
        case['lifecycle'] = 'infrastructure_failure'
    case['termination'] = status.get('termination', status.get('stop_reason'))
    case['error'] = status.get('error')
    case['logged_steps'] = len(rows)
    reported_steps = integer(status.get('steps_executed', status.get('steps')))
    case['steps'] = reported_steps if reported_steps is not None else len(rows)
    if reported_steps is not None and reported_steps != len(rows):
        issues.append(dict(kind='step_count_mismatch', reported=reported_steps, logged=len(rows)))
    if case['lifecycle'] == 'completed' and (not (directory/'steps.jsonl').exists()
            or not rows and samples[0]['native_success'] is None):
        issues.append(dict(kind='completed_without_step_tape'))
    native_values = [sample['native_success'] for sample in samples]
    reported_native = next((status[k] for k in
                            ('any_native_success', 'native_success', 'native_ever', 'full_native_success')
                            if boolean(status.get(k)) is not None), None)
    observed_native = (True if True in native_values else
                       False if all(value is False for value in native_values) else
                       False if rows and all(row['native_success'] is False for row in rows) else None)
    case['native_success'] = reported_native if reported_native is not None else observed_native
    case['native_success_sources'] = dict(reported=reported_native, tape=observed_native)
    case['first_native_success_step'] = next((sample['step'] for sample in samples
                                             if sample['native_success'] is True), None)
    case['final_native_success'] = samples[-1]['native_success']
    if reported_native is not None and observed_native is not None and reported_native != observed_native:
        issues.append(dict(kind='native_status_tape_disagreement'))
        case['native_success'] = None
    goal_names = set(samples[0]['predicates'])
    for sample in samples:
        if (goal_names and set(sample['predicates']) == goal_names
                and sample['native_success'] is not None
                and sample['native_success'] != all(sample['predicates'].values())):
            issues.append(dict(kind='native_constituent_conjunction_disagreement', step=sample['step']))
    if case['lifecycle'] == 'completed':
        case['outcome'] = ('native_success' if case['native_success'] is True else
                           'native_failure' if case['native_success'] is False else 'unknown_native_outcome')
    for name, item in case['grasps'].items():
        step = item['first_true_step']
        if step is not None and step >= 0:
            case['progress_evidence'].append(dict(kind='target_grasp_observed', target=name, step=step))
    for name, item in case['predicates'].items():
        tokens = re.findall(r'[a-z_]+', name.lower())
        if tokens and tokens[0] in ('in', 'on') and item['attainment_steps']:
            case['progress_evidence'].append(dict(kind='object_placement_predicate_became_true', predicate=name,
                                                 step=item['attainment_steps'][0]))
    case['meaningful_partial_progress'] = (case['outcome'] == 'native_failure'
                                           and bool(case['progress_evidence']))
    if case['outcome'] == 'native_failure':
        observations = case['failure_observations']
        observations.append('native_full_conjunction_not_observed')
        if not case['progress_evidence']:
            observations.append('no_meaningful_progress_evidence')
        if any(item['loss_steps'] for item in case['predicates'].values()):
            observations.append('constituent_predicate_attained_then_lost')
        if any(item['loss_steps'] for item in case['grasps'].values()):
            observations.append('target_grasp_ended')
        if any(item['final'] is True for item in case['predicates'].values()):
            observations.append('some_constituent_predicates_true_at_end')
        if any(item['final'] is False for item in case['predicates'].values()):
            observations.append('constituent_predicates_false_at_end')
        if case['termination']:
            observations.append('termination:' + str(case['termination']))
    case['strict_diagnostics'] = dict(
        initial=samples[0]['strict'], final=rows[-1]['strict'] if rows else {},
        reported={key: status[key] for key in ('strict_success', 'strict_complete', 'any_complete',
                                              'strict_diagnostics') if key in status},
        interpretation='Reported physical diagnostics only; separate from native primary endpoint.')
    case['replay'] = replay_info(directory, status, run, task, root, job)
    budget = integer(mapping(status.get('args')).get('steps', status.get('action_budget'))) or 520
    if budget <= 0:
        issues.append(dict(kind='invalid_action_budget', value=budget))
        budget = 520
    case['action_budget'] = budget
    case['diagnostic_opportunities'] = opportunities(case, rows, budget)
    # Replay is recorded separately: it neither changes native outcome nor proves benefit.
    case['selection_evidence_usable'] = (case['lifecycle'] == 'completed' and not issues
                                        and case['replay']['state'] == 'passed')
    return case


def aggregate(task, episodes):
    completed = [case for case in episodes if case['lifecycle'] == 'completed']
    successes = [case for case in completed if case['outcome'] == 'native_success']
    failures = [case for case in completed if case['outcome'] == 'native_failure']
    usable = [case for case in completed if case.get('selection_evidence_usable')]
    eligible_success = sum(case['outcome'] == 'native_success' for case in usable)
    eligible_partial = sum(case['meaningful_partial_progress'] for case in usable)
    diagnostic_counts = {}
    for field in ('grasps', 'predicates'):
        diagnostic_counts[field] = {}
        for name in sorted({name for case in completed for name in case[field]}):
            items = [case[field][name] for case in completed if name in case[field]]
            diagnostic_counts[field][name] = dict(observed_completed=len(items), planned=6,
                ever_true=sum(item['ever_true'] is True for item in items),
                attained=sum(bool(item['attainment_steps']) for item in items),
                lost=sum(bool(item['loss_steps']) for item in items),
                final_true=sum(item['final'] is True for item in items))
    return dict(task=task, native_task_name=TASKS[task], planned=6, observed_completed=len(completed),
        native_successes=len(successes), native_failures=len(failures),
        native_success_rate_planned=len(successes)/6,
        native_success_rate_completed=len(successes)/len(completed) if completed else None,
        completed_unknown_native=sum(case['outcome'] == 'unknown_native_outcome' for case in completed),
        meaningful_partial_progress_failures=sum(case['meaningful_partial_progress'] for case in completed),
        lifecycle_counts=dict(Counter(case['lifecycle'] for case in episodes)),
        replay_counts=dict(Counter(case['replay']['state'] for case in episodes)),
        cases_with_data_issues=sum(bool(case['data_issues']) for case in episodes),
        diagnostics_completed=diagnostic_counts,
        completed_steps=dict(total=sum(case['steps'] for case in completed),
            mean=mean(case['steps'] for case in completed) if completed else None,
            by_root={str(case['root']): case['steps'] for case in completed}),
        selection_usable_successes=eligible_success, selection_usable_partial_failures=eligible_partial,
        eligible=len(usable) == 6 and eligible_success >= 2 and eligible_partial >= 2)


def markdown(summary):
    lines = ['# Native LIBERO-Long screening', '',
        'Development cohort: four tasks × six planned roots (0–5). Primary outcome is the native '
        'simultaneous full-goal conjunction; separately attained predicates are not full success.', '',
        '| Task | Native successes / planned | Observed completed | Partial-progress failures | Eligible |',
        '|---|---:|---:|---:|---|']
    for task in summary['tasks']:
        lines.append(f"| {task['task']} | {task['native_successes']}/6 | {task['observed_completed']}/6 | "
                     f"{task['meaningful_partial_progress_failures']} | {task['eligible']} |")
    selection = summary['selection']
    lines += ['', f"**Priority-qualified task:** {selection['selected_task'] or 'none; criteria not relaxed'}.",
        f"Selection provisional: {selection['provisional']}. Rule: at least two full successes and two "
        'meaningful partial-progress failures; priority microwave → two_pots → two_mugs → drawer. '
        'No method-gain information is used.', '',
        'Meaningful progress requires an observed target grasp or a logged false→true object-placement '
        '(In/On) goal transition. Appliance-only changes and initially satisfied predicates do not qualify. Grasp endings and predicate losses '
        'are observations, not causal diagnoses. No mug order is imposed.', '',
        'Execution failures, unstarted/incomplete episodes, data problems and replay reports remain separate. '
        'The planned-denominator fraction is coverage-aware, not a completed-episode success estimate. '
        'Task selection requires all six cases completed, consistent and replay-verified.', '',
        '## Per-episode failure map and artifacts', '']
    for case in summary['episodes']:
        lines += [f"### {case['task']} / root {case['root']}", '',
            f"- Execution: `{case['lifecycle']}`; outcome: `{case['outcome']}`; "
            f"steps: {case['steps']} (logged {case['logged_steps']}); replay: `{case['replay']['state']}`.",
            '- Failure observations: ' + (', '.join(case['failure_observations']) or 'none recorded') + '.',
            f"- Artifacts: `{case['artifacts']['directory']}`; "
            f"status: `{case['artifacts']['status']}`; tape: `{case['artifacts']['steps']}`."]
        for field, label in (('grasps', 'Target grasp'), ('predicates', 'Constituent')):
            for name, item in case[field].items():
                lines.append(f"- {label} `{name}`: first true={item['first_true_step']}; "
                             f"attainments={item['attainment_steps']}; losses={item['loss_steps']}; final={item['final']}.")
        if case['data_issues']:
            lines.append('- Data issues: ' + ', '.join(item['kind'] for item in case['data_issues']) + '.')
        candidates = case['diagnostic_opportunities']
        lines.append(f'- Evidence-backed diagnostic candidates: {len(candidates)}; ' +
                     (', '.join(f"{key}={value}" for key, value in
                                sorted(Counter(item['phase'] for item in candidates).items())) or 'none') + '.')
        lines.append('')
    lines += ['## Interpretation', '',
        'Machine-readable constituent/grasp histories, early/mid/late candidate evidence and replay records '
        'are in `summary.json`. Step indices are zero-based; −1 is the recorded initial observation. '
        'Candidate observations are post-action, with branch state index step+1. Temporal thirds refer to '
        'the configured action budget. Candidates do not establish recoverability or intervention benefit. '
        'Strict physical diagnostics are retained separately and do not redefine native success.', '']
    return '\n'.join(lines)


def main(args):
    run, out = args.run.resolve(), args.out.resolve()
    if not run.is_dir():
        raise ValueError(f'Cohort directory does not exist: {run}')
    found, issues, ignored = discover(run)
    supervisor_path = run/'launch.json' if (run/'launch.json').exists() else run/'plan.json'
    supervisor = read_json(supervisor_path, issues) if supervisor_path.exists() else {}
    jobs = {}
    for job in supervisor.get('jobs', []):
        if not isinstance(job, dict):
            continue
        key = (task_key(job.get('task')), integer(job.get('init_index')))
        if key[0] in TASKS and key[1] in ROOTS:
            jobs[key] = job
    episodes = [episode(run, task, root, found.get((task, root), []), jobs.get((task, root), {}))
                for task in TASKS for root in ROOTS]
    tasks = [aggregate(task, [case for case in episodes if case['task'] == task]) for task in TASKS]
    selected = next((task['task'] for task in tasks if task['eligible']), None)
    summary = dict(schema_version=1, scope='native_actor_development_screening', run=str(run),
        planned_episodes=24, planned_roots=list(ROOTS), tasks=tasks, episodes=episodes,
        selection=dict(selected_task=selected, priority=list(TASKS), minimum_successes=2,
            minimum_meaningful_partial_failures=2, rules_relaxed=False, method_gains_used=False,
             provisional=any(task['observed_completed'] != 6 or task['cases_with_data_issues']
                             or task['completed_unknown_native']
                             or task['replay_counts'].get('passed', 0) != 6 for task in tasks)),
        discovery_issues=issues, unassigned_status_artifacts=ignored,
        supervisor_artifact=str(supervisor_path) if supervisor else None,
        semantics=dict(primary='Explicit native simultaneous full-goal success, never union-over-time predicates.',
            progress='Target grasp observed after an action or object-placement In/On goal false-to-true transition.',
            ordering='Constituent progress retained independently; no mandatory object order.',
            diagnostic_candidates='Observed losses in completed partial-progress failures; no causal or efficacy claim.',
            replay='Reported verification only; summarizer does not execute replay.',
            strict='Physical completion diagnostics are separate from primary native success.'))
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (out/'results.md').write_text(markdown(summary))
    print(json.dumps(dict(summary=str(out/'summary.json'), results=str(out/'results.md'),
                          selection=summary['selection'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Read-only cohort directory')
    parser.add_argument('--out', type=Path, required=True, help='Report directory (only summary.json/results.md are written)')
    main(parser.parse_args())
