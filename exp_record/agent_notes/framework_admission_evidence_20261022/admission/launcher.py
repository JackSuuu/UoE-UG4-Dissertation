"""Matched development diagnostics using existing canonical runner modes.

This supervisor changes no controller or action-generation behavior. Two GPU
lanes run serially within each lane. Existing episodes are read, never replaced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import run_framework_comparison as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--existing', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('controls', 'measured', 'admission'), required=True)
    args = parser.parse_args()
    plan = comparison.load(args.plan)
    comparison.check_plan(plan)
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for root in range(6):
        arms = ('baseline', 'joint') if args.phase == 'controls' else (
            ('joint_admission',) if args.phase == 'admission' else ('measured_fusion',))
        for arm in arms:
            cell_name = f"{'baseline' if arm == 'baseline' else 'v2'}_root{root}"
            cell = next(c for c in plan['cells'] if c['name'] == cell_name)
            existing = args.existing/cell_name
            reuse = args.phase == 'controls' and existing.exists()
            if reuse:
                status = comparison.load(existing/'status.json')
                comparison.require(status.get('valid_episode') is True and
                    status.get('status') == 'completed' and
                    status.get('plan_sha256') == comparison.sha(args.plan) and
                    status.get('joint_mode') == (arm == 'joint') and
                    status.get('fusion_mode') == ('off' if arm == 'baseline' else 'gate') and
                    status.get('recovery_options') is False and
                    status.get('action_protocol') == 'unchanged_vla', 'Incompatible existing episode')
            source = existing if reuse else args.out/'episodes'/f'{arm}_root{root}'
            jobs.append(dict(name=f'{arm}_root{root}', arm=arm, root=root, cell=cell,
                source=str(source.resolve()), reused=reuse, status='queued'))
    protocol = dict(
        hypothesis='Joint package may cause unnecessary or ineffective intervention; inspect paired complete-task harms and recovery traces.',
        development_only=True, roots=list(range(6)),
        primary='Complete native In AND Close success under matched 520 actions/queries',
        arms=['baseline', 'joint', 'measured_fusion'],
        contrast='measured_fusion removes joint stage inference, memory, prediction and risk together; NOT a risk-only ablation',
        stopping='All declared cases; no efficacy early stopping; invalid execution retained separately',
        plan=str(args.plan.resolve()), plan_sha256=comparison.sha(args.plan),
        launcher_sha256=comparison.sha(Path(__file__)), jobs=jobs)
    if args.phase == 'admission':
        protocol.update(hypothesis='Joint-only anomaly entry may cause harm; isolate admission from stage selection and handoff.',
            arms=['joint_admission'], contrast='Only joint-only repair ENTRY is disabled; joint stage, memory, risk/prediction and handoff veto remain active.',
            joint_admission='measured_only')
    comparison.save(args.out/'protocol.json', protocol)
    (args.out/'launcher.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'logs').mkdir()
    (args.out/'episodes').mkdir()
    (args.out/'replays').mkdir()

    def run(job, label, command, timeout):
        log = args.out/'logs'/f"{job['name']}.{label}.log"
        record = dict(command=command, started=time.time(), log=str(log))
        with log.open('xb') as stream:
            process = subprocess.Popen(command, cwd=comparison.REPO,
                env=comparison.environment(job['cell']['lane']), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = process.pid
            comparison.save(args.out/f"{job['name']}.{label}.json", record)
            try:
                record['returncode'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                comparison.terminate_group(process)
                record.update(returncode=-15, timeout=True)
        record['ended'] = time.time()
        comparison.save(args.out/f"{job['name']}.{label}.json", record)
        return record['returncode']

    def lane(gpu):
        results = []
        for job in (j for j in jobs if j['cell']['lane']['gpu'] == gpu):
            source = Path(job['source'])
            base = [comparison.PYTHON, '-B', '-u', str(comparison.SCRIPT)]
            print(f"GPU{gpu} {job['name']} reused={job['reused']}", flush=True)
            if not job['reused']:
                command = base+['--worker', '--plan', str(args.plan.resolve()),
                    '--cell', job['cell']['name'], '--out', str(source)]
                if job['arm'] != 'baseline':
                    command += ['--fusion-mode', 'gate', '--mlp-path', plan['fusion']['model']]
                if job['arm'] in ('joint', 'joint_admission'):
                    command += ['--joint-components', plan['joint']['artifact']]
                if job['arm'] == 'joint_admission':
                    command += ['--joint-admission', 'measured_only']
                job['actor_returncode'] = run(job, 'actor', command, comparison.LIMITS['outer_seconds'])
            status = comparison.load(source/'status.json') if (source/'status.json').exists() else {}
            job.update(episode_status=status.get('status'), valid_episode=status.get('valid_episode', False),
                any_native_success=status.get('any_native_success'), steps=status.get('steps_executed'),
                termination=status.get('termination'))
            if job['valid_episode']:
                job['replay_returncode'] = run(job, 'replay', base+['--replay', str(source),
                    '--out', str((args.out/'replays'/job['name']).resolve())], 360)
                if job['arm'] in ('joint', 'joint_admission'):
                    # The stock auditor writes one new report inside the source.
                    comparison.require(not (source/'joint_recomputation.json').exists(),
                        'Refusing to overwrite an existing joint audit')
                    job['audit_returncode'] = run(job, 'audit', [comparison.PYTHON, '-B', '-u',
                        str(comparison.REPO/'scripts/run_joint_framework.py'), '--audit', str(source)], 600)
                elif job['arm'] == 'measured_fusion':
                    if status['controller_summary']['repair_actions']:
                        comparison.require(not (source/'fusion_recomputation.json').exists(),
                            'Refusing to overwrite an existing fusion audit')
                        job['audit_returncode'] = run(job, 'audit', [comparison.PYTHON, '-B', '-u',
                            str(comparison.REPO/'scripts/run_fusion_handoff_ablation.py'),
                            '--audit', str(source)], 600)
                    else:
                        job.update(audit_returncode=0, audit_note='No repair/handoff scores to recompute')
            job['status'] = 'finished'
            comparison.save(args.out/f"{job['name']}.result.json", job)
            print(json.dumps({k: job.get(k) for k in ('name', 'episode_status', 'any_native_success',
                'steps', 'replay_returncode', 'audit_returncode')}), flush=True)
            results.append(job)
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [job for result in pool.map(lane, (2, 3)) for job in result]
    comparison.save(args.out/'results.json', results)
    return 0 if all(j['valid_episode'] and j.get('replay_returncode') == 0 and
        (j['arm'] == 'baseline' or j.get('audit_returncode') == 0) for j in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
