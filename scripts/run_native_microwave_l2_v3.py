"""Bounded H96 supervisor: prepare, read-only validate, then explicit launch.

The pinned scheduler retains phase ordering and process-tree cleanup. Its
hard-coded pair watchdog and H32 accounting are replaced by explicit H96 code.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import run_native_long_branch_v3 as version

SCRIPT = Path(__file__).resolve()


def configure():
    version.verify_old_sources()
    lineage = version.load_frozen('run_native_microwave_l2_v2.py').configure()
    engine = version.load_frozen('run_native_microwave_l2.py')
    engine.SCRIPT, engine.DESIGN = SCRIPT, version.DESIGN
    engine.PROTOCOL, engine.PROTOCOL_ID = version.PROTOCOL, version.PROTOCOL_ID
    engine.FREEZE_SHA, engine.LIMITS = version.FREEZE_SHA, deepcopy(version.LIMITS)

    def identities(snapshots):
        cells = lineage.identities(snapshots)
        for cell in cells:
            cell.update(local_horizon=96, seeds=[0 if cell['candidate'] == 'C0' else cell['seed']+i
                                                for i in range(96)])
        return cells

    def build_plan():
        manifest = version.require_design()
        plan = lineage.build_plan()
        frozen = plan['frozen_files_sha256']

        def pin(path, expected=None):
            path = str(Path(path).resolve())
            value = version.sha(path)
            version.require(expected is None or expected == value, f'Input drift: {path}')
            version.require(path not in frozen or frozen[path] == value, f'Lineage drift: {path}')
            frozen[path] = value

        pin(version.DESIGN/'design_freeze.json', version.FREEZE_SHA)
        for path, expected in version.load(version.DESIGN/'design_freeze.json')['files'].items():
            pin(path, expected)
        for name in version.ALL_SOURCES:
            pin(version.REPO/name, version.OLD_HASHES.get(Path(name).name))
        prefixes = {}
        for cell in identities(manifest['snapshots']):
            record = version.prefix_record(cell['root'], int(cell['candidate'][1:]))
            for path, expected in record['sha256'].items():
                pin(path, expected)
            prefixes[cell['name']] = record
        plan.update(protocol=version.PROTOCOL_ID, limits=deepcopy(version.LIMITS),
            cells=identities(manifest['snapshots']), snapshots=deepcopy(manifest['snapshots']),
            prefix_sources=prefixes,
            archive_sources={name: frozen[str(version.REPO/name)] for name in version.ALL_SOURCES},
            design_freeze_sha256=version.FREEZE_SHA,
            snapshot_manifest_sha256=version.sha(version.DESIGN/'snapshot_manifest.json'),
            protocol_sha256=version.sha(version.PROTOCOL),
            version_adapter=dict(scientific_change='local_horizon_32_to_96_only',
                                 frozen_predecessor_sources=dict(version.OLD_HASHES)))
        engine.check_pins(plan)
        return plan

    def validate_plan(plan, path):
        version.validate_plan_input(path)
        engine.check_pins(plan, path)
        version.require(plan == build_plan(), 'Prepared v3 plan differs from complete frozen inventory')

    def commands(plan, cell, out):
        prefix = [plan['python'], '-B', '-u', str(version.SCRIPT)]
        command = prefix + ['--plan', str(out/'plan.json'), '--baseline', cell['baseline'],
            '--snapshot-step', str(cell['snapshot_step']), '--candidate', cell['source'],
            '--seed', str(cell['seed']), '--local-horizon', '96', '--continuation-steps', '0',
            '--gpu', str(cell['lane']['gpu']), '--egl-device', str(cell['lane']['egl_device']),
            '--wall-seconds', '300', '--readiness-seconds', '180', '--query-seconds', '30',
            '--out', str(out/cell['name'])]
        if cell['candidate'] != 'C0':
            command += ['--nominal-controls', *[str(out/f'C0_root{root}') for root in range(6)]]
        replay = prefix + ['--replay', str(out/cell['name']), '--out', str(out/('replay_'+cell['name'])),
                           '--wall-seconds', '180']
        return command, replay

    def budget_available(cells, out, new_jobs=0):
        total = reserved = 0
        for cell in cells:
            used = engine.counts(out/cell['name'])
            version.require(used <= 96, f'Per-cell H96 call cap: {cell["name"]}')
            total += used
            if cell['status'] == 'branch_running':
                reserved += 96-used
        version.require(total+reserved+96*new_jobs <= 1728, 'Aggregate H96 call reservation exceeds 1728')
        return total

    def check_branch(plan, cell, directory):
        status = version.load(directory/'status.json')
        version.require(version.version_status(status, directory) == (cell['root'], int(cell['candidate'][1:])),
                        'Branch root/candidate differs')
        version.require(status['status'] == 'completed' and not status['partial_outputs'], 'Incomplete branch')
        n = engine.counts(directory, final=True)
        version.require((status['termination'] == 'action_budget' and n == 96)
                        or (status['termination'] == 'native_success' and 32 < n <= 96), 'Invalid H96 stop')
        expected = dict(baseline=cell['baseline'], snapshot_step=cell['snapshot_step'],
            candidate=cell['source'], seed=cell['seed'], local_horizon=96, continuation_steps=0,
            gpu=cell['lane']['gpu'], egl_device=cell['lane']['egl_device'], wall_seconds=300,
            readiness_seconds=180, query_seconds=30, out=str(directory))
        version.require(all(status['args'].get(k) == v for k, v in expected.items()), 'Branch CLI drift')
        identity = version.load(directory/'device_identity.json')
        version.require(identity['physical_gpu'] == cell['lane']['gpu']
            and identity['physical_uuid'] == identity['cuda_visible_devices'] == cell['lane']['physical_uuid']
            and identity['egl_device'] == cell['lane']['egl_device'], 'Branch GPU/EGL drift')
        build = status['actor_build']
        version.require(build['H'] == build['batch'] == status['batch_size'] == 1
            and build['transformers'] == '4.40.1' and build['frozen'] and build['official_predict_action']
            and build['unnorm_key'] == 'libero_10'
            and build['device_identity']['physical_uuid'] == cell['lane']['physical_uuid'], 'Actor drift')
        version.require(status['initial'] == cell['fingerprint'] and status['branch_sources'] == {
            name: plan['archive_sources']['scripts/'+name] for name in version.BRANCH_SOURCES}, 'Boundary/source drift')
        engine.exact_report(version.load(directory/'reconstruction.json'), 'exact_reconstruction_passed')
        version.require(version.sha(directory/'policy_inputs/000000.png') ==
                        plan['boundary_inputs'][cell['root']]['policy_png_sha256'], 'Boundary PNG drift')
        version.validate_tape(directory, status)
        version.validate_prefix_marker(directory, plan['prefix_sources'][cell['name']], status)
        if cell['candidate'] == 'C0':
            engine.exact_report(version.load(directory/'nominal_identity.json'), 'exact_nominal_identity_passed')
        return n

    old_replay, old_pin = engine.check_replay, engine.pin_result

    def check_replay(cell, branch, replay):
        version.control_replay(branch)
        old_replay(cell, branch, replay)

    def pin_result(cell, out):
        old_pin(cell, out)
        directory = out/cell['name']
        paths = [directory/'prefix_identity.json', directory/'fresh_device_preflight.json',
                 directory/'worker_responses.jsonl', out/('replay_'+cell['name'])/'fresh_device_preflight.json']
        paths += list((directory/'frames').glob('*.npz')) + list((directory/'policy_inputs').glob('*.png'))
        cell['artifact_sha256'].update({str(path): version.sha(path) for path in paths})

    class Supervisor(engine.Supervisor):
        def run(self):
            # Branches use the run copy of the plan; its source archive must be
            # equally complete before the inherited scheduler starts any child.
            original_save = engine.save

            def save(path, value):
                original_save(path, value)
                if path == self.out/'plan.json':
                    for name, expected in self.plan['archive_sources'].items():
                        destination = self.out/'source'/name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes((version.REPO/name).read_bytes())
                        version.require(version.sha(destination) == expected, 'Run source archive drift')
            engine.save = save
            try:
                return super().run()
            finally:
                engine.save = original_save

        def pair(self, cells, replay=False):
            self.children = engine.Children()
            jobs = []
            try:
                engine.check_pins(self.plan, self.plan_path)
                engine.check_result_pins(self.state['cells'])
                version.require(version.sha(self.plan_path) == self.state['plan_sha256'], 'Plan changed')
                budget_available(self.state['cells'], self.out, new_jobs=0 if replay else len(cells))
                for cell in cells:
                    version.require(time.monotonic() < self.deadline, 'Phase watchdog expired')
                    if not replay and cell['candidate'] != 'C0':
                        version.require(engine.controls_passed(self.state['cells']), 'All six exact C0 replays required')
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
                cap = 210 if replay else 540
                while pending:
                    self.children.scan()
                    version.require(time.monotonic() < self.deadline, 'Phase watchdog expired')
                    self.write()
                    for cell, process, started in list(pending):
                        version.require(time.monotonic()-started < cap-10, f'Outer watchdog: {cell["name"]}')
                        code = process.poll()
                        if code is None:
                            continue
                        kind = 'replay' if replay else 'branch'
                        cell[kind+'_returncode'], cell[kind+'_ended'] = code, time.time()
                        version.require(code == 0, f'{kind} failed: {cell["name"]} ({code})')
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

    engine.identities, engine.build_plan, engine.validate_plan = identities, build_plan, validate_plan
    engine.commands, engine.budget_available = commands, budget_available
    engine.check_branch, engine.check_replay, engine.pin_result = check_branch, check_replay, pin_result
    engine.Supervisor = Supervisor
    return engine


def main(argv=None):
    return configure().main(argv)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'v3 supervisor refused: {error}', file=sys.stderr)
        raise SystemExit(1)
