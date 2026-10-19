"""Offline H96 audit: 36 unchanged nested labels plus 18 H96 labels.

No model, simulator or GPU imports. All 18 identities are retained; exact v2
prefixes and independently replayed H96 tapes are necessary for the U96 gate.
"""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import run_native_long_branch_v3 as version

SCRIPT = Path(__file__).resolve()


def configure():
    version.verify_old_sources()
    old_boundary = version.load_frozen('audit_native_microwave_l2_v2.py').configure().baseline_context
    auditor = version.load_frozen('audit_native_microwave_l2.py')
    original_provenance, original_replay = auditor.provenance, auditor.replay_check
    original_cell = auditor.audit_cell

    def labels(initial, rows, z):
        return {str(h): auditor.local_labels(initial, [r['diagnostics'] for r in rows], z, h)
                for h in (12, 32, 96)}

    def baseline_context(snap):
        context = old_boundary(snap)
        tape = auditor.load_rows(Path(snap['source'])/'steps.jsonl')[snap['state_index']:snap['state_index']+96]
        success = next((i+1 for i, row in enumerate(tape) if row['diagnostics']['native_success']), None)
        if success is not None:
            tape = tape[:success]
        context.update(original_labels=labels(context['initial'], tape, context['first_grasp_z']),
                       original_count=len(tape))
        return context

    def completed_tape(directory, status, snap, context, rows):
        version.version_status(status, directory)
        n = len(rows)
        version.require(status['status'] == 'completed' and 32 < n <= 96
            and n == status['steps_executed'] == status['actor_queries']
            and status['initial'] == snap['fingerprint'] and status['initial_diagnostics'] == context['initial'],
            'Incomplete H96 tape/accounting/boundary')
        version.require([r['step'] for r in rows] == list(range(snap['state_index'], snap['state_index']+n)),
                        'Nonsequential H96 actions')
        native = [i for i, row in enumerate(rows) if row['diagnostics']['native_success']]
        version.require((status['termination'] == 'native_success' and native == [n-1]) or
                        (status['termination'] == 'action_budget' and n == 96 and not native), 'Invalid H96 stopping boundary')
        version.require(not status['partial_outputs'] and status['handoff'] is None, 'Partial tape or handoff')
        frames = auditor.load_rows(directory/'frames.jsonl')
        version.require(len(frames) == n+1 and frames[0]['fingerprint'] == snap['fingerprint']
                        and frames[0]['diagnostics'] == context['initial'], 'Frame tape/boundary mismatch')
        for i, row in enumerate(rows):
            version.require(row['before'] == frames[i]['fingerprint'] and row['after'] == frames[i+1]['fingerprint']
                            and row['diagnostics'] == frames[i+1]['diagnostics'], f'Frame/action alignment {i}')
            d = row['diagnostics']
            version.require(d['native_success'] == (d['predicates'][auditor.IN] and d['predicates'][auditor.CLOSE]),
                            'Native conjunction differs')
        auditor.verify_frames(directory, frames)
        return dict(completed=True, actions=n, termination=status['termination'])

    def validate_queries(directory, status, rows, root, candidate):
        version.require(version.cell_identity(version.argparse.Namespace(**status['args'])) == (root, candidate),
                        'Query identity mismatch')
        return version.validate_tape(directory, status)

    def provenance(directory, status, snap, context, pinned_sources):
        version.require(version.version_status(status, directory)[0] == snap['root'], 'Foreign branch root')
        version.require(pinned_sources is not None and all(pinned_sources.get(name) == expected
                        for name, expected in version.branch_hashes().items()), 'Missing v3 implementation pins')
        return original_provenance(directory, status, snap, context, pinned_sources)

    def replay_check(directory, source, snap, context, status, rows, computed_labels):
        version.require(version.version_status(version.load(directory/'status.json'), directory)
                        == version.version_status(status, source), 'Foreign v3 replay')
        version.control_replay(source)
        return original_replay(directory, source, snap, context, status, rows, computed_labels)

    def source_pins(plan, run, snapshots_file, manifest):
        version.require(version.validate_plan_input(run/'plan.json') == plan, 'Audit plan input differs')
        version.require(plan['snapshots'] == manifest['snapshots']
                        and plan['snapshot_manifest_sha256'] == version.sha(snapshots_file), 'Foreign audit plan')
        launch = version.load(run/'launch.json')
        version.require(launch['plan'] == plan and launch['plan_sha256'] == version.sha(run/'plan.json'),
                        'Launch attribution differs')
        return {Path(name).name: expected for name, expected in plan['archive_sources'].items()}

    def audit_cell(root, candidate, paths, replay_paths, snap, context, pinned_sources, plan=None):
        cell = original_cell(root, candidate, paths, replay_paths, snap, context, pinned_sources)
        if len(paths) == 1:
            directory = paths[0]

            def prefix_check():
                version.require(plan is not None, 'No validated v3 prefix inventory')
                return version.validate_prefix_marker(directory, plan['prefix_sources'][f'C{candidate}_root{root}'],
                                                      version.load(directory/'status.json'))
            auditor.check(cell, 'exact_v2_prefix_before_query33', prefix_check)
            cell['valid'] = cell['valid'] and all(item['passed'] for item in cell['checks'].values())
            if not cell['valid'] and cell['state'] == 'valid':
                cell['state'] = 'integrity_unverified'
            if version.load(directory/'status.json').get('status') == 'invalid_identity':
                cell['state'] = 'invalid_identity'
        return cell

    def continuation_gate(cells, integrity):
        indexed = {(c['root'], c['candidate']): c for c in cells}
        full = len(cells) == 18 and set(indexed) == {(r, f'C{i}') for r in range(6) for i in range(3)}
        required = ('completion', 'queries', 'reconstruction', 'provenance', 'fresh_replay', 'labels',
                    'exact_v2_prefix_before_query33')
        full = full and all(c['valid'] and all(c['checks'].get(key, {}).get('passed') is True for key in required)
                            for c in cells)
        greedy = all(indexed.get((r, 'C0'), {}).get('checks', {}).get('nominal_identity', {}).get('passed') is True
                     for r in range(6))
        gains = []
        for root in (0, 4):
            control = indexed.get((root, 'C0'), {})
            if (control.get('labels') or {}).get('96', {}).get('useful_proxy') is not False:
                continue
            for candidate in ('C1', 'C2'):
                cell = indexed.get((root, candidate), {})
                if (cell.get('labels') or {}).get('96', {}).get('useful_proxy') is True:
                    gains.append(dict(root=root, candidate=candidate))
        passed = bool(integrity and full and greedy and gains)
        return dict(pass_L2B=passed, horizon=96, all_18_valid=full, all_6_greedy_exact=greedy,
            global_integrity=integrity, sampled_failure_root_gains=gains, automatic_launch=False,
            further_duration_tuning_allowed=False)

    def audit(run, snapshots_file):
        version.require(snapshots_file.resolve() == version.DESIGN/'snapshot_manifest.json', 'Foreign v3 manifest')
        manifest = version.require_design()
        snapshots = {s['root']: s for s in manifest['snapshots']}
        version.require(set(snapshots) == set(range(6)) and len(manifest['snapshots']) == 6, 'Wrong root bank')
        report = dict(scope='offline_model_free_no_simulator_no_GPU', protocol=version.PROTOCOL_ID,
            run=str(run), snapshots=str(snapshots_file), checks={}, cells=[], originals=[], selector='none',
            interpretation='Offline oracle availability only; not framework efficacy or a deployable selector.',
            model_queries_performed=0, simulator_actions_performed=0, auditor_sha256=version.sha(SCRIPT),
            frozen_v1_auditor_sha256=version.OLD_HASHES['audit_native_microwave_l2.py'])
        report['checks']['design_freeze'] = dict(passed=True, detail=version.FREEZE_SHA)
        plan = None

        def plan_check():
            nonlocal plan
            candidate = version.load(run/'plan.json')
            pins = source_pins(candidate, run, snapshots_file, manifest)
            plan = candidate
            return pins
        pinned = auditor.check(report, 'launch_plan', plan_check)
        branches, replays, statuses, unknown = auditor.discover(run, snapshots)
        report['unassigned_evidence'] = unknown
        for root in range(6):
            snap = snapshots[root]
            context = auditor.check(report, f'baseline_root{root}', lambda: baseline_context(snap))
            if context is not None:
                report['checks'][f'baseline_root{root}']['detail'] = dict(frozen=True)
                report['originals'].append(dict(root=root, source=snap['source'], labels=context['original_labels']))
            for candidate in range(3):
                report['cells'].append(audit_cell(root, candidate, branches[(root, candidate)], replays,
                                                 snap, context, pinned, plan))
        total_calls = total_actions = 0
        accounting = True
        directories = {str(p.parent.resolve()) for p in run.rglob('queries.jsonl')
                       if not any(x in ('source', 'branch_source', 'reconstruction')
                                  for x in p.relative_to(run).parts[:-1])}
        for directory in sorted(set(statuses) | directories):
            status = statuses.get(directory, {})
            try:
                requested, exact = auditor.requested_count(Path(directory)/'queries.jsonl')
                total_calls += requested
                reported = status.get('actor_queries', 0)
                version.require(type(reported) is int and reported >= 0, 'Invalid query count')
                total_calls += max(reported-requested, 0)
                accounting &= exact and bool(status) and reported == requested
                if not status:
                    unknown.append(dict(path=directory, error='Query tape without status'))
                if status.get('args', {}).get('replay'):
                    accounting &= reported == 0
                else:
                    path = Path(directory)/'steps.jsonl'
                    actions = len(auditor.load_rows(path)) if path.exists() else 0
                    total_actions += actions
                    accounting &= actions <= max(reported, requested) <= 96
                    accounting &= status.get('steps_executed', 0) == actions
            except (OSError, ValueError, TypeError):
                accounting = False
        report['budget'] = dict(requested_calls_conservative=total_calls, cap=1728,
                                actions=total_actions, accounting_exact=accounting, includes_fresh_prefix_calls=True)
        report['checks']['budget'] = dict(passed=accounting and total_calls <= 1728)
        report['checks']['discovery'] = dict(passed=not unknown, detail=unknown)
        report['boundary_diversity'], report['oracle_availability'] = [], []
        for root in range(6):
            cells = [c for c in report['cells'] if c['root'] == root]
            available = [c for c in cells if c.get('boundary')]
            pngs = [c['boundary']['policy_png_sha256'] for c in available]
            common = len(available) == 3 and None not in pngs and len(set(pngs)) == 1
            if plan is not None:
                boundary = plan['boundary_inputs'][root]
                common = common and pngs[0] == boundary['policy_png_sha256'] and boundary['prompt'] == auditor.PROMPT
                common = common and boundary['prompt_sha256'] == version.hashlib.sha256(auditor.PROMPT.encode()).hexdigest()
            diversity = dict(root=root, observed_candidates=len(available), common_boundary_policy_png=common,
                             duplicates_retained=True)
            for field in ('tokens', 'action'):
                groups = defaultdict(list)
                for cell in available:
                    groups[version.json.dumps(cell['boundary'][field])].append(cell['candidate'])
                diversity[field] = dict(unique_count=len(groups), duplicate_groups=[v for v in groups.values() if len(v)>1])
            report['boundary_diversity'].append(diversity)
            report['checks'][f'boundary_root{root}'] = dict(passed=common)
            for h in ('12', '32', '96'):
                values = {c['candidate']: (c['labels'] or {}).get(h, {}).get('useful_proxy') if c['valid'] else None
                          for c in cells}
                oracle = True if True in values.values() else False if all(v is False for v in values.values()) else None
                report['oracle_availability'].append(dict(root=root, horizon=int(h), useful_by_candidate=values,
                    any_useful=oracle, sampled_gains_over_C0=[c for c in ('C1', 'C2') if values[c] is True and values['C0'] is False]))
        report['valid_candidate_count'] = sum(c['valid'] for c in report['cells'])
        report['cell_states'] = dict(Counter(c['state'] for c in report['cells']))
        report['gate'] = continuation_gate(report['cells'], all(c['passed'] for c in report['checks'].values()))
        return report

    def markdown(report):
        text = ['# Native microwave H96 offline audit', '', report['interpretation'], '',
            f"Valid: **{report['valid_candidate_count']}/18**. Calls: **{report['budget']['requested_calls_conservative']}/1728**. "
            f"U96 gate: **{report['gate']['pass_L2B']}**.", '',
            '| Root | Candidate | State | h | p_h | U_h | Native by h | Predicate loss |',
            '|---|---|---|---:|---|---|---|---|']
        for cell in report['cells']:
            for h in ('12', '32', '96'):
                label = (cell['labels'] or {}).get(h, {})
                values = [label.get(k) for k in ('terminal_level', 'useful_proxy', 'native_success', 'snapshot_predicate_lost')]
                text.append(f"| {cell['root']} | {cell['candidate']} | {cell['state']} | {h} | "
                            + ' | '.join('unknown' if v is None else str(v).lower() for v in values)+' |')
        text += ['', 'No automatic continuation. No further duration tuning. Missing/censored cells are unknown.',
                 'Complete integrity, prefix, replay and per-root findings are in `summary.json`.', '']
        return '\n'.join(text)

    auditor.labels, auditor.baseline_context = labels, baseline_context
    auditor.completed_tape, auditor.validate_queries = completed_tape, validate_queries
    auditor.provenance, auditor.replay_check = provenance, replay_check
    auditor.source_pins, auditor.audit_cell = source_pins, audit_cell
    auditor.continuation_gate, auditor.audit, auditor.markdown = continuation_gate, audit, markdown
    return auditor


def main(argv=None):
    return configure().main(argv)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'v3 offline audit refused: {error}', file=sys.stderr)
        raise SystemExit(1)
