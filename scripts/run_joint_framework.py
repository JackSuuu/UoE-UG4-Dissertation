"""One joint root0 controller, then tape replay AND independent neural audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_framework_comparison as comparison


def audit(source):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(comparison.REPO))
    from src.verification.joint_runtime import JointRuntime
    status = comparison.load(source/'status.json')
    runtime = JointRuntime(status['fusion_model']['path'], status['joint_components']['artifact'], device='cuda:0')
    rows = comparison.rows(source/'controller.jsonl')
    initial = next(r for r in rows if r['event'] == 'before')
    report = dict(status='running', actor_queries=0, observations_recomputed=0,
        handoff_scores_recomputed=0, mismatches=[], numeric_absolute_tolerance=1e-5,
        source=str(source), source_plan_sha256=status['plan_sha256'])

    def compare(expected, actual, path):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(expected) != set(actual):
                report['mismatches'].append(path+'.keys')
                return
            for key in expected:
                compare(expected[key], actual[key], f'{path}.{key}')
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                report['mismatches'].append(path+'.length')
                return
            for i, (e, a) in enumerate(zip(expected, actual)):
                compare(e, a, f'{path}[{i}]')
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not isinstance(actual, (int, float)) or abs(expected-actual) > 1e-5:
                report['mismatches'].append(path)
        elif expected != actual:
            report['mismatches'].append(path)

    def observe(index, state, expected):
        rgb = np.asarray(Image.open(source/'joint_inputs'/f'{index:06d}.png').convert('RGB'))
        value = runtime.observe(rgb, state, index, expected['physical']['official_region'])
        compare({k: v for k, v in expected.items() if k != 'recovery_admission'}, value, f'observation{index}')
        report['observations_recomputed'] += 1

    observe(0, initial['decision']['evidence']['state'], initial['decision']['evidence']['joint'])
    for row in rows:
        if row['event'] != 'after':
            continue
        index = row['step']+1
        state = row['verification']['evidence']['state']
        observe(index, state, row['verification']['joint_after'])
        fusion = row['verification']['verification'].get('fusion')
        if fusion is not None:
            fresh = runtime.score_handoff(state, row['decision']['recovery_goal'])
            compare(fusion['evidence'], fresh, f'handoff{index}')
            report['handoff_scores_recomputed'] += 1
    compare(status['joint_counters'], runtime.counters, 'counters')
    report['status'] = 'joint_recomputation_passed' if not report['mismatches'] else 'joint_recomputation_mismatch'
    comparison.save(source/'joint_recomputation.json', report)
    return 0 if not report['mismatches'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.audit:
        return audit(args.audit.resolve())
    comparison.require(args.plan is not None and args.out is not None, 'Plan and output required')
    plan = comparison.load(args.plan)
    comparison.check_plan(plan)
    comparison.require(plan.get('joint') and plan.get('fusion'), 'Pinned joint artifacts required')
    cell = next(c for c in plan['cells'] if c['name'] == 'v2_root0')
    comparison.require(cell['lane']['gpu'] == 3, 'Expected root0 GPU3 lane')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    environment = comparison.environment(cell['lane'])
    status = dict(status='started', started=time.time(), pid=os.getpid(), actors=1,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=comparison.REPO, text=True).strip(),
        plan=str(args.plan.resolve()), plan_sha256=comparison.sha(args.plan), lane=cell['lane'],
        cell=cell['name'], joint_mode=True, use_legacy_v5=False,
        enabled=['real_visual_stage', 'physical_contracts', 'short_term_prediction', 'trained_outcome_risk',
                 'episodic_retrieval', 'frozen_policy_h1_repair', 'joint_handoff_veto'], commands=[])
    save = lambda: comparison.save(out/'supervisor_status.json', status)

    def run(command, logfile, timeout):
        with (out/logfile).open('xb') as log:
            process = subprocess.Popen(command, cwd=comparison.REPO, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status['child_pid'] = process.pid
            status['commands'].append(dict(command=command, pid=process.pid, log=logfile))
            save()
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                comparison.terminate_group(process)
                return -15

    save()
    try:
        base = [comparison.PYTHON, '-B', str(comparison.SCRIPT)]
        status['phase'] = 'joint_episode'
        save()
        status['episode_returncode'] = run(base + ['--worker', '--plan', str(args.plan.resolve()),
            '--cell', 'v2_root0', '--out', str(out/'joint_root0'), '--fusion-mode', 'gate',
            '--mlp-path', plan['fusion']['model'], '--joint-components', plan['joint']['artifact']],
            'episode.log', comparison.LIMITS['outer_seconds'])
        if status['episode_returncode'] == 0:
            status['phase'] = 'tape_replay'
            save()
            status['replay_returncode'] = run(base + ['--replay', str(out/'joint_root0'),
                '--out', str(out/'replay_joint_root0')], 'replay.log', 360)
            status['phase'] = 'independent_joint_recomputation'
            save()
            status['audit_returncode'] = run([comparison.PYTHON, '-B', str(Path(__file__).resolve()),
                '--audit', str(out/'joint_root0')], 'audit.log', 600)
        status['status'] = 'completed' if all(v == 0 for k, v in status.items() if k.endswith('_returncode')) else 'completed_with_failures'
    except BaseException:
        status.update(status='supervisor_failed', error=traceback.format_exc())
    finally:
        status.update(phase='finished', ended=time.time())
        save()
    return 0 if status['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
