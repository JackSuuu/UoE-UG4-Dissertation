"""Training roots10..19: all nominal/framework pairs and repair candidate siblings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

import run_framework_comparison as comparison


def sibling_rule(status, replay_code, audit_code):
    """Selection depends on repair availability, never on success or failure."""
    if not status.get('valid_episode') or replay_code != 0 or audit_code != 0:
        return 'blocked_invalid_control'
    if status.get('candidate_pools') == 0:
        return 'not_applicable_no_admitted_repair'
    if status.get('candidate_pools') != 1:
        raise ValueError('Unexpected number of repair probes')
    return 'collect_all_three'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = comparison.load(args.plan)
    comparison.check_plan(plan)
    comparison.require(plan.get('training_data'), 'Requires reserved-training-root plan')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'episodes', 'replays'):
        (out/folder).mkdir()
    jobs = []
    for root in comparison.TRAINING_ROOTS:
        for slot in (None, 0, 1, 2, 3):
            cohort = 'baseline' if slot is None else 'v2'
            cell = next(c for c in plan['cells'] if c['name'] == f'{cohort}_root{root}')
            name = f"{'baseline' if slot is None else 'candidate'+str(slot)}_root{root}"
            jobs.append(dict(name=name, root=root, slot=slot, cell=cell,
                source=str(out/'episodes'/name), status='queued'))
    protocol = dict(schema='repair_training_expansion_v1', data_split='training',
        mainline='verification -> bounded repair -> reverification -> measured nominal continuation',
        plan=str(args.plan.resolve()), plan_sha256=comparison.sha(args.plan), jobs=jobs,
        roots=list(comparison.TRAINING_ROOTS), always_run=20, maximum_episodes=50,
        maximum_actor_queries=50*plan['repair_candidates']['total_query_cap'],
        collection='all bare nominal and greedy-framework pairs; slots1..3 only if valid slot0 admitted a repair',
        selection_bias='No filtering on task success; all triggered successes and failures receive all three siblings',
        skip_rule='No-probe siblings are not executed or counted as independent trials; invalid controls are reported separately',
        splits='10..19training only; candidate siblings grouped by root;6..9calibration and20..49confirmation unopened',
        generator=plan['repair_candidates'], scoring='fixed-slot collection, no trained scorer or oracle ranking',
        primary='Complete native task outcome; paired rescues and harms vs nominal and slot0',
        resources='520actions;523maximum actor queries, nominal uses no auxiliary queries; all slots share actual pool cost',
        stopping='All declared root pairs and all applicable siblings, no efficacy stopping or automatic retries',
        interpretation='Training data and same-task initial-state diversity; not cross-task or held-out method efficacy')
    comparison.save(out/'protocol.json', protocol)

    def run(job, phase, command, timeout):
        result = dict(command=command, started=time.time())
        with (out/'logs'/f"{job['name']}.{phase}.log").open('xb') as stream:
            p = subprocess.Popen(command, cwd=comparison.REPO,
                env=comparison.environment(job['cell']['lane']), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            result['pid'] = p.pid
            comparison.save(out/f"{job['name']}.{phase}.json", result)
            try:
                result['returncode'] = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                comparison.terminate_group(p)
                result.update(returncode=-15, timeout=True)
        result['ended'] = time.time()
        comparison.save(out/f"{job['name']}.{phase}.json", result)
        return result['returncode']

    def execute(job):
        source = Path(job['source'])
        print(f"GPU{job['cell']['lane']['gpu']}: {job['name']}", flush=True)
        base = [comparison.PYTHON, '-B', '-u', str(comparison.SCRIPT)]
        command = base+['--worker', '--plan', str(args.plan.resolve()),
            '--cell', job['cell']['name'], '--out', str(source)]
        if job['slot'] is not None:
            command += ['--fusion-mode', 'gate', '--mlp-path', plan['fusion']['model'],
                '--joint-components', plan['joint']['artifact'], '--joint-admission', 'measured_only',
                '--candidate-slot', str(job['slot'])]
        job['actor_returncode'] = run(job, 'actor', command, comparison.LIMITS['outer_seconds'])
        status = comparison.load(source/'status.json') if (source/'status.json').exists() else {}
        job.update(valid_episode=status.get('valid_episode', False), episode_status=status.get('status'),
            success=status.get('any_native_success'), actions=status.get('steps_executed'),
            queries=status.get('actor_queries'), pools=status.get('candidate_pools', 0))
        if job['valid_episode']:
            job['replay_returncode'] = run(job, 'replay', base+['--replay', str(source),
                '--out', str(out/'replays'/job['name'])], 360)
            if job['slot'] is not None:
                job['audit_returncode'] = run(job, 'audit', [comparison.PYTHON, '-B', '-u',
                    str(comparison.REPO/'scripts/run_joint_framework.py'), '--audit', str(source)], 600)
            else:
                job['audit_returncode'] = 0
                job['audit_note'] = 'Bare nominal has no joint model to recompute'
        job['status'] = 'finished'
        comparison.save(out/f"{job['name']}.result.json", job)
        print({k: job.get(k) for k in ('name', 'success', 'actions', 'pools',
            'replay_returncode', 'audit_returncode')}, flush=True)
        return status

    def lane(gpu):
        results, skips = [], []
        for root in comparison.TRAINING_ROOTS:
            root_jobs = [j for j in jobs if j['root'] == root and j['cell']['lane']['gpu'] == gpu]
            if not root_jobs:
                continue
            execute(root_jobs[0])
            results.append(root_jobs[0])
            slot0 = root_jobs[1]
            status = execute(slot0)
            results.append(slot0)
            rule = sibling_rule(status, slot0.get('replay_returncode'), slot0.get('audit_returncode'))
            for job in root_jobs[2:]:
                if rule == 'collect_all_three':
                    execute(job)
                    results.append(job)
                else:
                    job.update(status='not_executed', reason=rule)
                    comparison.save(out/f"{job['name']}.result.json", job)
                    skips.append(job)
        return results, skips

    with ThreadPoolExecutor(max_workers=2) as pool:
        lanes = list(pool.map(lane, (2, 3)))
    results = [j for result, _ in lanes for j in result]
    skips = [j for _, skipped in lanes for j in skipped]
    comparison.save(out/'results.json', results)
    comparison.save(out/'skipped_siblings.json', skips)
    return 0 if all(j['valid_episode'] and j.get('replay_returncode') == 0 and
        j.get('audit_returncode') == 0 for j in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
