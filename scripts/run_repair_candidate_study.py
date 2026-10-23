"""Frozen first-repair candidate coverage; canonical episodes, replay and audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

import run_framework_comparison as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = comparison.load(args.plan)
    comparison.check_plan(plan)
    comparison.require(plan.get('repair_candidates') is not None, 'Candidate protocol not pinned')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'episodes', 'replays'):
        (out/folder).mkdir()
    jobs = []
    for root in (0, 1, 4):
        for slot in (range(2) if root == 1 else range(4)):
            cell = next(c for c in plan['cells'] if c['name'] == f'v2_root{root}')
            name = f'candidate{slot}_root{root}'
            jobs.append(dict(name=name, root=root, slot=slot, cell=cell,
                source=str(out/'episodes'/name), status='queued'))
    protocol = dict(schema='repair_candidate_coverage_study_v1',
        mainline='verification -> bounded model-generated repair -> reverification -> verified resumption',
        hypothesis='At first measured repair admission, policy samples may contain a useful alternative to greedy',
        development_only=True, plan=str(args.plan.resolve()), plan_sha256=comparison.sha(args.plan),
        generator=plan['repair_candidates'], jobs=jobs,
        cases='roots0/4 four fixed slots each; successful root1 two inactive-path sentinels',
        selection='fixed_slot_for_data_collection; no learned-ranking efficacy claim',
        primary='Complete native task success; paired rescues and harms against slot0',
        labels=['local recovery', 'verified exit', 'actual handoff', '8-step stable nominal continuation',
            'completion after handoff', 'complete native task success'],
        matching='Full original reset and prefix; every slot queries the same first-repair pool before selection',
        budgets='520 executed actions; 520 nominal requests plus at most3candidate queries =523 total',
        scope='One H1 candidate only; all later repair requests, actions and handoff checks use existing controller',
        training='Retain failed/unknown outcomes; split by root group, not frame or candidate; no training on evaluated outcomes',
        stopping='Complete all ten declared episodes; retain infrastructure failures; no efficacy stopping or automatic retries')
    comparison.save(out/'protocol.json', protocol)

    def run(job, phase, command, timeout):
        record = dict(command=command, started=time.time())
        with (out/'logs'/f"{job['name']}.{phase}.log").open('xb') as stream:
            process = subprocess.Popen(command, cwd=comparison.REPO,
                env=comparison.environment(job['cell']['lane']), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = process.pid
            comparison.save(out/f"{job['name']}.{phase}.json", record)
            try:
                record['returncode'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                comparison.terminate_group(process)
                record.update(returncode=-15, timeout=True)
        record['ended'] = time.time()
        comparison.save(out/f"{job['name']}.{phase}.json", record)
        return record['returncode']

    def lane(gpu):
        results = []
        for job in (j for j in jobs if j['cell']['lane']['gpu'] == gpu):
            print(f"GPU{gpu}: {job['name']}", flush=True)
            source = Path(job['source'])
            base = [comparison.PYTHON, '-B', '-u', str(comparison.SCRIPT)]
            command = base+['--worker', '--plan', str(args.plan.resolve()),
                '--cell', job['cell']['name'], '--out', str(source), '--fusion-mode', 'gate',
                '--mlp-path', plan['fusion']['model'], '--joint-components', plan['joint']['artifact'],
                '--joint-admission', 'measured_only', '--candidate-slot', str(job['slot'])]
            job['actor_returncode'] = run(job, 'actor', command, comparison.LIMITS['outer_seconds'])
            status = comparison.load(source/'status.json') if (source/'status.json').exists() else {}
            job.update(valid_episode=status.get('valid_episode', False), episode_status=status.get('status'),
                success=status.get('any_native_success'), actions=status.get('steps_executed'),
                queries=status.get('actor_queries'))
            if job['valid_episode']:
                job['replay_returncode'] = run(job, 'replay', base+['--replay', str(source),
                    '--out', str(out/'replays'/job['name'])], 360)
                job['audit_returncode'] = run(job, 'audit', [comparison.PYTHON, '-B', '-u',
                    str(comparison.REPO/'scripts/run_joint_framework.py'), '--audit', str(source)], 600)
            job['status'] = 'finished'
            comparison.save(out/f"{job['name']}.result.json", job)
            print({k: job.get(k) for k in ('name', 'valid_episode', 'success', 'actions',
                'queries', 'replay_returncode', 'audit_returncode')}, flush=True)
            results.append(job)
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [j for lane_result in pool.map(lane, (2, 3)) for j in lane_result]
    comparison.save(out/'results.json', results)
    return 0 if all(j['valid_episode'] and j.get('replay_returncode') == 0 and
        j.get('audit_returncode') == 0 for j in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
