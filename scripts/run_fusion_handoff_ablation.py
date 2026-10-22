"""Sequential matched root0 off/log/gate workers; retain every outcome and replay.

Prepare a fresh canonical plan with --mlp-path first. Start this supervisor in
a detached process session; it uses the plan's GPU3 UUID, never a default GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import run_framework_comparison as comparison


def audit(source):
    """Recompute scores from retained causal policy PNGs, separate from tape replay."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(comparison.REPO))
    from src.verification.handoff_verifier import HandoffVerifier
    status = comparison.load(source/'status.json')
    verifier = HandoffVerifier(status['fusion_model']['path'], device='cuda:0')
    controls = comparison.rows(source/'controller.jsonl')
    before = {r['step']: r for r in controls if r['event'] == 'before'}
    result = dict(actor_queries=0, scores_recomputed=True, compared=0, skipped=[], mismatches=[])
    for row in controls:
        if row['event'] != 'after':
            continue
        stored = row['verification']['verification'].get('fusion', {}).get('evidence')
        if stored is None:
            continue
        step = row['step']
        image = source/'model_inputs'/f'{step+1:06d}.png'
        if not image.exists():
            result['skipped'].append(dict(step=step, reason='no_next_policy_png_at_episode_end'))
            continue
        rgb = np.asarray(Image.open(image).convert('RGB'))
        state = row['verification']['evidence']['state']
        expected_features = before[step+1]['decision']['evidence']['state']
        for key in ('in_goal', 'close_goal', 'grasp', 'held_clear', 'table', 'position', 'rise_m'):
            comparison.require(state[key] == expected_features[key], 'Adjacent endpoint evidence mismatch')
        recomputed = verifier.score(rgb, state, stored['stage'])
        fields = [key for key in ('accepted', 'threshold', 'model_sha256', 'feature_schema',
            'policy_rgb_sha256', 'encoder_sha256', 'physics_features') if recomputed.get(key) != stored.get(key)]
        if abs(recomputed['probability']-stored['probability']) > 1e-5:
            fields.append('probability_tolerance_1e-5')
        if fields:
            result['mismatches'].append(dict(step=step, fields=fields))
        result['compared'] += 1
    result['status'] = 'passed' if result['compared'] and not result['mismatches'] else 'failed'
    comparison.save(source/'fusion_recomputation.json', result)
    return 0 if result['status'] == 'passed' else 1


def summarize(out):
    results = {}
    actions = {}
    for mode in ('off', 'log', 'gate'):
        source = out/mode
        if not (source/'status.json').exists():
            continue
        status = comparison.load(source/'status.json')
        steps = comparison.rows(source/'steps.jsonl')
        controls = comparison.rows(source/'controller.jsonl')
        fusion = [r['verification']['verification'].get('fusion', {}) for r in controls if r['event'] == 'after']
        summary = status.get('controller_summary', {})
        actions[mode] = [r['action'] for r in steps]
        results[mode] = dict(status=status['status'], termination=status.get('termination'),
            native_success=status.get('any_native_success'), queries=status.get('actor_queries'),
            actions=status.get('steps_executed'), numeric_corrections=status.get('numeric_correction_actions'),
            repair_actions=summary.get('repair_actions'), repairs=summary.get('repairs'),
            blocked_verified_exits=sum(bool(r.get('blocked_verified_exit')) for r in fusion),
            initial_state_sha256=comparison.sha(source/'selected_initial_state.npy')
                if (source/'selected_initial_state.npy').exists() else None,
            action_tape_sha256=hashlib.sha256(json.dumps(actions[mode], separators=(',', ':')).encode()).hexdigest(),
            replay=comparison.load(out/f'replay_{mode}'/'verification.json')
                if (out/f'replay_{mode}'/'verification.json').exists() else None,
            fusion_recomputation=comparison.load(source/'fusion_recomputation.json')
                if (source/'fusion_recomputation.json').exists() else None)
    report = dict(results=results, off_log_actions_identical=actions.get('off') == actions.get('log')
        if 'off' in actions and 'log' in actions else None,
        root=0, threshold_reselected=False, complete_task_endpoint='native In AND Close',
        scope='matched development root0 only; no cross-task efficacy claim')
    comparison.save(out/'comparison.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--audit', type=Path)
    args = p.parse_args()
    if args.audit:
        return audit(args.audit.resolve())
    comparison.require(args.plan is not None and args.out is not None, 'Plan and output required')
    plan = comparison.load(args.plan)
    comparison.check_plan(plan)
    comparison.require(plan.get('fusion') is not None, 'Pinned fusion model required')
    cell = next(c for c in plan['cells'] if c['name'] == 'v2_root0')
    comparison.require(cell['lane']['gpu'] == 3, 'Matched root0 GPU3 lane required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    environment = comparison.environment(cell['lane'])
    state = dict(status='started', pid=os.getpid(), started=time.time(), actors=1,
        plan=str(args.plan.resolve()), plan_sha256=comparison.sha(args.plan), lane=cell['lane'],
        modes=['off', 'log', 'gate'], use_v5=True, model=plan['fusion'], commands=[])
    save = lambda: comparison.save(out/'supervisor_status.json', state)

    def run(command, logfile, timeout):
        with (out/logfile).open('xb') as log:
            process = subprocess.Popen(command, cwd=comparison.REPO, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = process.pid
            state['commands'].append(dict(command=command, log=logfile, pid=process.pid))
            save()
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                comparison.terminate_group(process)
                return -signal.SIGTERM

    save()
    try:
        base = [comparison.PYTHON, '-B', str(comparison.SCRIPT)]
        for mode in state['modes']:
            state.update(mode=mode, phase='episode')
            save()
            rc = run(base + ['--worker', '--plan', str(args.plan.resolve()), '--cell', cell['name'],
                '--out', str(out/mode), '--use-v5', '--fusion-mode', mode,
                '--mlp-path', plan['fusion']['model']], f'episode_{mode}.log', comparison.LIMITS['outer_seconds'])
            state[f'{mode}_episode_returncode'] = rc
            if rc == 0:
                state['phase'] = 'replay'
                save()
                state[f'{mode}_replay_returncode'] = run(base + ['--replay', str(out/mode),
                    '--out', str(out/f'replay_{mode}')], f'replay_{mode}.log', 360)
                if mode != 'off':
                    state['phase'] = 'fusion_recomputation'
                    save()
                    state[f'{mode}_audit_returncode'] = run([comparison.PYTHON, '-B', str(Path(__file__).resolve()),
                        '--audit', str(out/mode)], f'audit_{mode}.log', 360)
            summarize(out)
        state['status'] = 'completed' if all(value == 0 for key, value in state.items()
            if key.endswith('_returncode')) else 'completed_with_failures'
    except BaseException:
        state.update(status='supervisor_failed', error=traceback.format_exc())
    finally:
        state.update(phase='finished', ended=time.time())
        save()
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
