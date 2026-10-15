"""Post-hoc A1 diagnosis. Read-only originals; exact-prefix physics counterfactuals.

Counterfactual pulse/settle outcomes are local diagnostics, not online repairs or
evidence of remaining-task recovery. Never compare a predicted three-step pulse
with three subsequent *different* actor actions as though they were the same plan.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np

NAMES = ['nominal', 'hold_grasp', 'retract', 'x_plus', 'x_minus', 'y_plus', 'y_minus', 'insert']


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def load(root, seed, case, arm):
    path = root/f'live_s{seed}_c{case}_{arm}'
    status = json.loads((path/'status.json').read_text())
    proof = json.loads((path/'replay.json').read_text())
    digest = hashlib.sha256((path/'trace.jsonl').read_bytes()).hexdigest()
    assert digest == proof['trace_sha256'] == status['trace_sha256']
    assert proof['status'] == 'exact_replay_passed'
    rows = [json.loads(x) for x in (path/'trace.jsonl').read_text().splitlines()]
    return status, rows


def clearance(obs):
    rot = np.array(obs[6:15]).reshape(3, 3)
    local = rot.T @ -np.array(obs[:3])
    extent = .016*np.abs(rot[:2, :2]).sum(axis=0)
    return (.02275-np.abs(local[:2])-extent)*1000


def metrics(rows):
    result = []
    for r in rows:
        sel = NAMES.index(r['selection'])
        predicted = np.array(r['forecast_mean'])
        obs, after = np.array(r['obs']), np.array(r['after'])
        delta = predicted[sel, 0]-after
        means, std = np.array(r['risk_mean']), np.array(r['risk_std'])
        lateral = np.linalg.norm(predicted[:, -1, :2], axis=1)/.02
        vertical = np.abs(predicted[:, -1, 2]+.02)/.12
        risk_term = .5*np.array(r['forecast_scores'])
        np.testing.assert_allclose(lateral+vertical+risk_term, r['objective'], atol=2e-5, rtol=2e-5)
        best = int(np.argmin(r['objective']))
        result.append(dict(step=r['step'], changed=r['changed'], selection=r['selection'],
            xyz_error_mm=float(np.linalg.norm(delta[:3])*1000),
            xyz_error_components_mm=(delta[:3]*1000).tolist(),
            actual_load_N=r['peak_nut_peg_load_N'], predicted_load_N=float(means[sel, 0, 0]*10),
            predicted_load_std_N=float(std[sel, 0, 0]*10),
            actual_tilt_deg=float(np.rad2deg(r['endpoint']['tilt_rad'])),
            predicted_tilt_deg=float(means[sel, 0, 1]*15),
            pre_xy_error_mm=float(np.linalg.norm(obs[:2])*1000),
            post_xy_error_mm=float(np.linalg.norm(after[:2])*1000),
            expected_xy_change_mm=float((np.linalg.norm(predicted[sel, 0, :2])-np.linalg.norm(obs[:2]))*1000),
            observed_xy_change_mm=float((np.linalg.norm(after[:2])-np.linalg.norm(obs[:2]))*1000),
            budget_used=r['corrections_used'], trigger=r['trigger'],
            would_change=bool(r['trigger'] and best != 0 and r['objective'][best] < r['objective'][0]-.01),
            preferred=NAMES[best], detached=r['endpoint']['detached'],
            seated=r['endpoint']['seated'], contained=r['endpoint']['contained'],
            clearance_mm=clearance(after).tolist(),
            objective_lateral=lateral.tolist(), objective_vertical=vertical.tolist(),
            objective_risk=risk_term.tolist(),
            effective_rotation_norm=float(np.linalg.norm(np.array(r['effective'])[3:6])),
            nominal_rotation_norm=float(np.linalg.norm(np.array(r['nominal'])[3:6]))))
    return result


def describe(root, out):
    out.mkdir(parents=True, exist_ok=False)
    report = dict(interpretation='post-hoc development diagnosis; no controller changes', episodes=[], failures=[])
    for seed in (3000, 3001):
        for case in range(5):
            for arm in ('vla', 'framework', 'feedback'):
                status, rows = load(root, seed, case, arm)
                points = metrics(rows)
                errors = np.array([r['xyz_error_mm'] for r in points])
                interventions = [r for r in points if r['changed']]
                exhausted = next((r['step'] for r in points if r['budget_used'] == 40), None)
                suffix = points[exhausted+1:] if exhausted is not None else []
                e = dict(seed=seed, case=case, arm=arm, success=status['complete'],
                    steps=len(rows), changed=len(interventions),
                    selections=dict(Counter(r['selection'] for r in interventions)),
                    error_mean_mm=float(errors.mean()), error_p95_mm=float(np.quantile(errors, .95)),
                    repair_error_mean_mm=float(np.mean([r['xyz_error_mm'] for r in interventions])) if interventions else None,
                    budget_exhausted_step=exhausted,
                    suppressed_after_budget=sum(r['would_change'] for r in suffix),
                    later_triggers=sum(r['trigger'] for r in suffix),
                    post_budget_peak_load_N=max((r['actual_load_N'] for r in suffix), default=None),
                    actual_repair_xy_increase=sum(r['observed_xy_change_mm'] > .1 for r in interventions),
                    repair_predicted_improve_but_worsened=sum(r['expected_xy_change_mm'] < -.1 and r['observed_xy_change_mm'] > .1 for r in interventions),
                    repair_while_detached=sum(rows[r['step']-1]['endpoint']['detached'] for r in interventions if r['step'] > 0),
                    first_detached=next((r['step'] for r in points if r['detached']), None),
                    first_native_success=next((r['step'] for r in rows if r['endpoint']['native_success']), None),
                    first_seated=next((r['step'] for r in points if r['seated']), None),
                    any_complete=any(r['endpoint']['complete'] for r in rows),
                    terminal=rows[-1]['endpoint'], terminal_obs_mm=(np.array(rows[-1]['after'][:3])*1000).tolist(),
                    terminal_clearance_mm=points[-1]['clearance_mm'],
                    nonzero_rotation_effective_steps=sum(r['effective_rotation_norm'] > 1e-6 for r in points),
                    trace_sha256=status['trace_sha256'])
                report['episodes'].append(e)
                write(out/f'points_s{seed}_c{case}_{arm}.json', points)
                if arm == 'framework' and not status['complete']:
                    worst = max(interventions, key=lambda r: r['xyz_error_mm'])['step']
                    first = interventions[0]['step']
                    blocked = next((r['step'] for r in suffix if r['would_change']), None)
                    events = {}
                    for name, step in [('first_repair', first), ('worst_repair_prediction', worst),
                                       ('budget_last_repair', exhausted), ('first_budget_blocked', blocked)]:
                        if step is not None:
                            events.setdefault(str(step), []).append(name)
                    report['failures'].append(dict(**e, events=events,
                        event_details=[dict(labels=labels, **points[int(step)]) for step, labels in events.items()]))
    write(out/'descriptive.json', report)
    return report


def branches(root, out, seed, case):
    from adapters.assembly_runtime import make_assembly, prepare_insertion, step_measured, features, endpoint
    from run_assembly_development import OFFSETS
    status, rows = load(root, seed, case, 'framework')
    specification = json.loads((out/'descriptive.json').read_text())
    spec = next(r for r in specification['failures'] if r['seed'] == seed and r['case'] == case)
    results = []
    for text_step, labels in spec['events'].items():
        step = int(text_step)
        row = rows[step]
        selected = NAMES.index(row['selection'])
        values = []
        for index, actions in enumerate(row['candidates']):
            env = make_assembly(seed, camera=False)
            try:
                _, setup = prepare_insertion(env, OFFSETS[case])
                assert setup == json.loads((root/f'live_s{seed}_c{case}_framework'/'setup.json').read_text())
                for old in rows[:step]:
                    _, load_N = step_measured(env, old['effective'])
                    ep = endpoint(env)
                    np.testing.assert_array_equal(env.sim.get_state().flatten(), old['after_state'])
                    assert load_N == old['peak_nut_peg_load_N'] and ep == old['endpoint']
                np.testing.assert_array_equal(features(env), np.array(row['obs'], dtype=np.float32))
                states, risks, endpoints = [], [], []
                for action in actions:
                    _, load_N = step_measured(env, action)
                    state = features(env)
                    ep = endpoint(env)
                    states.append(state.tolist())
                    risks.append([load_N/10., ep['tilt_rad']/np.deg2rad(15)])
                    endpoints.append(ep)
                actual = np.array(states)
                predicted = np.array(row['forecast_mean'][index])
                actual_risk = np.array(risks)
                true_objective = (np.linalg.norm(actual[-1, :2])/.02 +
                    abs(actual[-1, 2]+.02)/.12 + .5*actual_risk.max())
                values.append(dict(index=index, name=NAMES[index], states=states, risk=risks,
                    endpoints=endpoints, true_objective=float(true_objective),
                    predicted_objective=row['objective'][index],
                    xyz_error_by_horizon_mm=(np.linalg.norm(predicted[:, :3]-actual[:, :3], axis=1)*1000).tolist(),
                    true_score=float(actual_risk.max()), predicted_score=row['forecast_scores'][index],
                    predicted_mean_only_score=float(np.max(row['risk_mean'][index]))))
            finally:
                env.close()
        best_pred = int(np.argmin(row['objective']))
        best_true = min(range(8), key=lambda i: values[i]['true_objective'])
        result = dict(seed=seed, case=case, step=step, labels=labels,
            trace_sha256=status['trace_sha256'], prefix_physics_exact=True,
            selected=NAMES[selected], preferred=NAMES[best_pred], true_best=NAMES[best_true],
            preferred_local_regret=values[best_pred]['true_objective']-values[best_true]['true_objective'],
            selected_local_regret=values[selected]['true_objective']-values[best_true]['true_objective'],
            preferred_true_advantage_vs_nominal=values[0]['true_objective']-values[best_pred]['true_objective'],
            values=values,
            caveat='Three-step offline pulse/settle only. GT score has no ensemble-uncertainty term; no full-task rescue claim.')
        write(out/f'branch_s{seed}_c{case}_t{step}.json', result)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'values'}), flush=True)
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['describe', 'branches'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--case', type=int)
    a = p.parse_args()
    if a.mode == 'describe': describe(a.root, a.out)
    else: branches(a.root, a.out, a.seed, a.case)
