"""g+120 L2A supervisor adapter; --prepare, --validate-only, explicit --launch.

The unchanged, hash-pinned v1 scheduler owns every cap, phase barrier, cleanup
operation and 18-cell status record. Its read-only inventory builder is reused
on a separate unmodified module instance to bind the original L1/v1 provenance;
only the independently verified v2 boundary/design records are then substituted.
This avoids copying the scheduler or weakening its hard-coded v1 checks.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import run_native_long_branch_v2 as version

SCRIPT = Path(__file__).resolve()


def configure():
    # These are distinct module instances. The lineage instance stays entirely
    # v1-configured, including its hard-coded boundary checks and design anchor.
    lineage = version.load_frozen('run_native_microwave_l2.py')
    engine = version.load_frozen('run_native_microwave_l2.py')
    original_commands = engine.commands
    original_check_branch = engine.check_branch
    original_check_replay = engine.check_replay
    engine.SCRIPT = SCRIPT
    engine.DESIGN = version.DESIGN
    engine.PROTOCOL = version.PROTOCOL
    engine.PROTOCOL_ID = version.PROTOCOL_ID
    engine.FREEZE_SHA = version.FREEZE_SHA
    # In the reused check_branch, SOURCES minus the v1 supervisor is exactly the
    # branch-source inventory. Full preparation inventory is ALL_SOURCES below.
    engine.SOURCES = ('scripts/run_native_microwave_l2.py',
                      *('scripts/'+name for name in version.BRANCH_SOURCES))

    def build_plan():
        manifest = version.require_design()
        plan = lineage.build_plan()
        previous_manifest = lineage.load(lineage.DESIGN/'snapshot_manifest.json')
        previous_snapshots = plan['snapshots']
        for key in ('sampling', 'sources', 'seed_formula', 'local_horizon', 'nested_measurements',
                    'checkpoint_revision', 'base_code_revision', 'instruction',
                    'candidate_count_per_snapshot', 'selector', 'retries'):
            version.require(manifest[key] == previous_manifest[key], f'v2 changed fixed setting: {key}')
        for key in ('local_model_calls', 'local_branch_wall_seconds', 'worker_readiness_seconds',
                    'generation_seconds', 'replay_seconds', 'phase_job_wall_seconds'):
            version.require(manifest['budgets'][key] == previous_manifest['budgets'][key],
                            f'v2 changed fixed budget: {key}')
        for key, value in previous_manifest['label_contract'].items():
            if key != 'reference_levels':
                version.require(manifest['label_contract'][key] == value, f'v2 changed label rule: {key}')
        frozen = plan['frozen_files_sha256']

        def pin(path, expected=None):
            path = Path(path).absolute()
            value = version.sha(path)
            version.require(expected is None or value == expected, f'v2 input drift: {path}')
            version.require(str(path) not in frozen or frozen[str(path)] == value,
                            f'v1 lineage input drift: {path}')
            frozen[str(path)] = value

        for snapshot, previous in zip(manifest['snapshots'], previous_snapshots):
            for key in ('root', 'source', 'first_grasp_step', 'source_sha256', 'initial_local_seeds'):
                version.require(snapshot[key] == previous[key], f'v2 changed snapshot source: {key}')
            base = Path(snapshot['source'])
            s = snapshot['state_index']
            version.require(s == previous['state_index']-60, 'v2 is not exactly g+120')
            frame = engine.row_at(base/'frames.jsonl', s)
            version.require(frame['fingerprint'] == snapshot['fingerprint']
                            == engine.row_at(base/'steps.jsonl', s-1)['after']
                            == engine.row_at(base/'steps.jsonl', s)['before'],
                            'v2 original boundary fingerprint mismatch')
            version.require(not frame['diagnostics']['native_success'], 'v2 boundary is already terminal')
            pin(base/'frames'/f'{s:06d}.npz')
        freeze_path = version.DESIGN/'design_freeze.json'
        pin(freeze_path, version.FREEZE_SHA)
        for path, expected in version.load(freeze_path)['files'].items():
            pin(path, expected)
        for name in version.ALL_SOURCES:
            pin(version.REPO/name, version.V1_HASHES.get(Path(name).name))
        inputs = engine.boundary_hashes(manifest['snapshots'], plan['python'])
        prompt = plan['boundary_inputs'][0]['prompt']
        prompt_sha = plan['boundary_inputs'][0]['prompt_sha256']
        for item in inputs:
            item.update(prompt=prompt, prompt_sha256=prompt_sha)
        plan.update(protocol=version.PROTOCOL_ID, snapshots=deepcopy(manifest['snapshots']),
            cells=engine.identities(manifest['snapshots']), boundary_inputs=inputs,
            archive_sources={name: frozen[str(version.REPO/name)] for name in version.ALL_SOURCES},
            design_freeze_sha256=frozen[str(freeze_path)],
            snapshot_manifest_sha256=frozen[str(version.DESIGN/'snapshot_manifest.json')],
            protocol_sha256=frozen[str(version.PROTOCOL)],
            version_adapter=dict(timer='first_grasp_plus_120', predecessor_design_sha256=lineage.FREEZE_SHA,
                                 frozen_v1_sources=dict(version.V1_HASHES)))
        engine.check_pins(plan)
        return plan

    def validate_plan(plan, plan_path):
        version.require_design()
        version.require(plan['protocol'] == version.PROTOCOL_ID
                        and plan['design_freeze_sha256'] == version.FREEZE_SHA, 'Foreign v2 launch plan')
        engine.check_pins(plan, plan_path)
        # Exact complete comparison rejects omitted pins, altered devices,
        # budgets, sources, identities, and all other unapproved changes.
        version.require(plan == build_plan(), 'v2 plan differs from frozen design/implementation')

    def commands(plan, cell, out):
        command, replay = original_commands(plan, cell, out)
        for argv in (command, replay):
            version.require(argv[3] == str(version.REPO/'scripts/run_native_long_branch.py'),
                            'Unexpected frozen v1 command entrypoint')
            argv[3] = str(version.SCRIPT)
        return command, replay

    def check_branch(plan, cell, directory):
        version.version_status(engine.load(directory/'status.json'), directory)
        return original_check_branch(plan, cell, directory)

    def check_replay(cell, branch, replay):
        version.version_status(engine.load(branch/'status.json'), branch)
        replay_status = engine.load(replay/'status.json')
        version.version_status(replay_status, replay)
        branch_status = engine.load(branch/'status.json')
        version.require(Path(replay_status['args']['replay']).resolve() == branch.resolve()
                        and replay_status['pid'] != branch_status['pid'],
                        'v2 replay is not a fresh process for this branch')
        return original_check_replay(cell, branch, replay)

    engine.build_plan = build_plan
    engine.validate_plan = validate_plan
    engine.commands = commands
    engine.check_branch = check_branch
    engine.check_replay = check_replay
    return engine


def main(argv=None):
    return configure().main(argv)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'v2 L2A supervisor refused: {error}', file=sys.stderr)
        raise SystemExit(1)
