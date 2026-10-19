"""Offline g+120 audit adapter: unchanged v1 labels, tape checks and all-bank gate.

The v1 audit embeds a literal protocol ID in its local design-check callback.
Replace only that callback with the independently anchored v2 design check;
never rewrite/evaluate its source or substitute labels or candidate outcomes.
All other audit machinery is imported from its exact frozen implementation.
"""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import run_native_long_branch_v2 as version

SCRIPT = Path(__file__).resolve()


def configure():
    auditor = version.load_frozen('audit_native_microwave_l2.py')
    auditor.DESIGN_FREEZE_SHA = version.FREEZE_SHA
    original_check = auditor.check
    original_provenance = auditor.provenance
    original_replay = auditor.replay_check
    original_source_pins = auditor.source_pins
    original_audit = auditor.audit

    def design_check(snapshots_file):
        version.require(snapshots_file.resolve() == version.DESIGN/'snapshot_manifest.json',
                        'v2 audit requires the independently frozen v2 manifest')
        manifest = version.require_design()
        version.require(manifest['local_horizon'] == 32
                        and manifest['nested_measurements'] == [12, 32]
                        and manifest['instruction'] == auditor.LANGUAGE
                        and manifest['checkpoint_revision'] == auditor.REVISION
                        and manifest['base_code_revision'] == auditor.BASE_REVISION,
                        'v2 changed fixed actor/horizon configuration')
        return dict(sha256=version.sha(snapshots_file), freeze_verified=True,
                    protocol=version.PROTOCOL_ID, timer='first_grasp_plus_120')

    def check(result, name, operation):
        if name == 'design_freeze':
            operation = lambda: design_check(Path(result['snapshots']))
        return original_check(result, name, operation)

    def source_pins(plan, run, snapshots_file, manifest):
        version.require(plan['protocol'] == version.PROTOCOL_ID
                        and plan['protocol_sha256'] == version.sha(version.PROTOCOL)
                        and plan['version_adapter']['frozen_v1_sources'] == version.V1_HASHES,
                        'Foreign v2 execution plan')
        version.require(set(plan['archive_sources']) == set(version.ALL_SOURCES),
                        'Incomplete v2 adapter/implementation source inventory')
        for name in version.ALL_SOURCES:
            version.require(plan['archive_sources'][name] == version.sha(version.REPO/name),
                            f'v2 implementation source drift: {name}')
        return original_source_pins(plan, run, snapshots_file, manifest)

    def provenance(directory, status, snap, context, pinned_sources):
        version.require(version.version_status(status, directory) == snap['root'],
                        'Foreign branch root')
        version.require(pinned_sources is not None and all(
            pinned_sources.get(name) == expected for name, expected in version.branch_hashes().items()),
            'v2 branch lacks prepared implementation pins')
        return original_provenance(directory, status, snap, context, pinned_sources)

    def replay_check(directory, source, snap, context, status, rows, computed_labels):
        version.version_status(status, source)
        version.require(version.version_status(auditor.load(directory/'status.json'), directory)
                        == snap['root'], 'Foreign replay root')
        return original_replay(directory, source, snap, context, status, rows, computed_labels)

    def baseline_context(snap):
        # The other embedded v1 version literal is g+180 in baseline_context.
        # Restate this small boundary validator with g+120; label functions are
        # still the exact imported v1 functions, using the new initial reference.
        source = Path(snap['source'])
        for name, expected in snap['source_sha256'].items():
            version.require(auditor.sha(source/name) == expected, f'Frozen baseline hash: {name}')
        rows = auditor.load_rows(source/'steps.jsonl')
        frames = auditor.load_rows(source/'frames.jsonl')
        status = auditor.load(source/'status.json')
        g, t, s = snap['first_grasp_step'], snap['post_action_step'], snap['state_index']
        version.require([r['step'] for r in rows] == list(range(len(rows))), 'Baseline step sequence')
        version.require(next(i for i, row in enumerate(rows) if row['diagnostics']['grasped'][auditor.OBJ]) == g,
                        'Original first grasp differs')
        version.require(t == g+120 == version.SNAPSHOT_STEPS[snap['root']] and s == t+1
                        and not any(row['diagnostics']['native_success'] for row in rows[:s]),
                        'Noncausal, foreign or terminal v2 snapshot')
        version.require(frames[s]['fingerprint'] == rows[t]['after'] == rows[s]['before'] == snap['fingerprint'],
                        'v2 snapshot fingerprint differs')
        initial = rows[t]['diagnostics']
        version.require(frames[s]['diagnostics'] == initial, 'v2 snapshot diagnostics differ')
        version.require(status['status'] == 'completed' and status['args']['seed'] == 0
                        and status['args']['init_index'] == snap['root']
                        and status['language'] == auditor.LANGUAGE, 'Baseline scene/instruction contract')
        z = rows[g]['diagnostics']['object_observations'][auditor.OBJ+'_pos'][2]
        return dict(initial=initial, first_grasp_z=z,
                    original_labels=auditor.labels(initial, rows[s:s+32], z),
                    original_count=min(32, len(rows)-s), manifest=auditor.load(source/'source_manifest.json'))

    def audit(run, snapshots_file):
        # Refuse a foreign design before reading candidate outcomes. Individual
        # invalid/foreign branch artifacts remain in the unchanged 18-cell audit.
        design_check(snapshots_file)
        report = original_audit(run, snapshots_file)
        report.update(auditor_sha256=version.sha(SCRIPT),
                      frozen_v1_auditor_sha256=version.V1_HASHES['audit_native_microwave_l2.py'])
        report['checks']['versioned_launch_plan'] = dict(
            passed=report['checks'].get('launch_plan', {}).get('passed') is True)
        report['gate'] = auditor.continuation_gate(report['cells'],
            all(check['passed'] for check in report['checks'].values()))
        return report

    auditor.check = check
    auditor.source_pins = source_pins
    auditor.provenance = provenance
    auditor.replay_check = replay_check
    auditor.baseline_context = baseline_context
    auditor.audit = audit
    return auditor


def main(argv=None):
    return configure().main(argv)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'v2 offline audit refused: {error}', file=sys.stderr)
        raise SystemExit(1)
