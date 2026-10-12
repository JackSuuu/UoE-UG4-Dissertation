"""Validation-only skip-one-intervention trajectory counterfactual.

Diagnostic selection uses saved paired harm/rescue labels, never deployment.
For the primary bisection arm, skip its first execution-changing intervention
on every paired harm/rescue episode; thereafter retain the frozen verifier.
Keep all original 32 episode rows and microbatch/cadence, including controls.
This is not a deployable skip rule or a method efficacy/threshold sweep.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from _common import env_for_cell, load_policy, setup
from _audit import verify_replay
from analyze_verifier_audit import analyze, sha256
from audit_verifier import atomic_json
from common import load_json
from rq2_eval import build_controller
import numpy as np
import torch


def skip_plan(controller, action, proposal, rows):
    """Called after act consumes step 0; preserve pointers and observation history."""
    controller.plan[rows] = proposal[rows]
    action[rows] = proposal[rows, 0]
    controller.a_prev[rows] = action[rows]


@torch.no_grad()
def run_branch(env, sim, controller, params, seed, reference, skip_at):
    obs = env.reset(params, seed)
    controller.reset(env, obs)
    B, dev = obs.shape[0], obs.device
    first_viol = torch.full((B,), -1, dtype=torch.long, device=dev)
    first_trig, first_success = first_viol.clone(), first_viol.clone()
    violations = torch.zeros(B, dtype=torch.bool, device=dev)
    interventions = torch.zeros(B, device=dev)
    trace = {k: [] for k in ('state', 'proposal', 'applied', 'trigger', 'skipped', 'live_risk')}
    seen = np.zeros(B, bool)
    di = 0
    for t in range(sim.T):
        decision = bool((controller.plan_left <= 0).all())
        if not decision and bool((controller.plan_left <= 0).any()):
            raise RuntimeError('asynchronous decision cadence')
        if decision:
            state = env.get_state().clone()
        action, info = controller.act(env, obs)
        effective_trigger = info['trig'].clone()
        if decision:
            rows = np.flatnonzero(skip_at == t)
            # Before a selected row branches, all its state/proposal/plan must
            # match the reference. Unselected rows must remain unchanged always.
            unchanged = ~seen
            for key, actual in (('state', state), ('proposal', info['chunk']), ('applied', controller.plan)):
                if not np.allclose(actual.cpu().numpy()[unchanged], reference[key][di, unchanged], rtol=1e-5, atol=1e-6):
                    raise RuntimeError(f'pre-branch/unselected {key} mismatch at t={t}')
            skipped = torch.zeros(B, dtype=torch.bool, device=dev)
            if rows.size:
                idx = torch.as_tensor(rows, device=dev)
                if not bool(info['trig'][idx].all()):
                    raise RuntimeError('selected branch is not a triggered intervention')
                if not bool((torch.abs(controller.plan[idx, :5] - info['chunk'][idx, :5]).amax((1, 2)) > 1e-7).all()):
                    raise RuntimeError('selected branch does not change execution')
                skip_plan(controller, action, info['chunk'], idx)
                skipped[idx] = True
                effective_trigger[idx] = False
                seen[rows] = True
            for key, value in (('state', state), ('proposal', info['chunk']),
                               ('applied', controller.plan), ('trigger', info['trig']), ('skipped', skipped)):
                trace[key].append(value.cpu().numpy().copy())
            di += 1
        obs, risk, success = env.step(action)
        live = risk.amax(1)
        trace['live_risk'].append(live.cpu().numpy())
        v = live > 1
        first_viol = torch.where(v & (first_viol < 0), t, first_viol)
        first_trig = torch.where(effective_trigger & (first_trig < 0), t, first_trig)
        first_success = torch.where(success & (first_success < 0), t, first_success)
        violations |= v
        interventions += effective_trigger.float()
    if not np.array_equal(seen, skip_at >= 0):
        raise RuntimeError('scheduled skip not applied exactly once')
    if controller.n_policy_calls != 16 or controller.n_policy_rows != B * 16:
        raise RuntimeError('branch changed H10/k5 policy cadence')
    data = {k: np.stack(v) for k, v in trace.items()}
    data.update(success=success.cpu().numpy(), violation=violations.cpu().numpy(),
                first_viol=first_viol.cpu().numpy(), first_trig=first_trig.cpu().numpy(),
                first_success=first_success.cpu().numpy(), n_interventions=interventions.cpu().numpy(),
                final_task_cost=sim.task_cost(env.get_state()).cpu().numpy(),
                decision_t=reference['decision_t'], skip_at=skip_at)
    unchanged = skip_at < 0
    verify_replay({k: v[unchanged] for k, v in data.items()
                   if k in ('success', 'violation', 'first_viol', 'first_trig', 'first_success', 'n_interventions', 'final_task_cost')},
                  {k: v[unchanged] for k, v in reference.items()
                   if k in ('success', 'violation', 'first_viol', 'first_trig', 'first_success', 'n_interventions', 'final_task_cost')})
    return data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit_summary', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--device', default='auto')
    cli = parser.parse_args()
    audit_path = Path(cli.audit_summary).resolve()
    attribution = analyze(audit_path)  # requires complete full-grid validation
    source = load_json(attribution['source_result'])
    audit = load_json(audit_path)
    args = SimpleNamespace(**source['metadata']['args'])
    if args.repair_mode != 'bisection' or args.hard_prefix or args.use_grad or args.n_envs != 32:
        raise ValueError('requires original frozen bisection validation protocol')
    for name, artifact in source['metadata']['artifacts'].items():
        if sha256(artifact['path']) != artifact['sha256']:
            raise RuntimeError(f'changed source artifact: {name}')
    args.device = cli.device
    dev, sim, od = setup(args)
    output = Path(od) / ('branch_' + cli.tag)
    output.mkdir(exist_ok=False)
    arm = 'checkvla_orbisim'
    selected = [r for r in attribution['ood_episode_witnesses']
                if r['arm'] == arm and (r['safe_harm'] or r['safe_rescue'])]
    if any(not r['first_divergence'] or not r['first_divergence']['state_matches_baseline'] or
           not r['first_divergence']['proposal_matches_baseline'] for r in selected):
        raise ValueError('selected witnesses lack matched initial divergence')
    summary = {'complete': False, 'arm': arm, 'selected_episodes': len(selected), 'per_cell': [],
               'metadata': {'audit_sha256': sha256(audit_path), 'source_sha256': attribution['source_sha256'],
                            'branch_code_sha256': sha256(__file__), 'raw_sha256': attribution['raw_sha256'],
                            'args': vars(args), 'scope': 'diagnostic skip first divergent intervention, then same verifier',
                            'selection': 'all pooled OOD paired harms and rescues; hindsight labels, NOT runtime policy',
                            'execution_horizon': 5, 'proposal_horizon': 10}}
    atomic_json(summary, output / 'summary.partial.json')
    policy = load_policy(args, od, dev, sim)
    taus = load_json(args.calibration_path)['tau']
    controller = build_controller(arm, policy, sim, od, dev, taus, args)
    indices = source['metadata']['evaluated_cell_indices']
    for ci in sorted({r['grid_index'] for r in selected}):
        entry = source['per_cell'][indices.index(ci)]
        audit_cell = next(c for c in audit['per_cell'] if c['grid_index'] == ci)
        with np.load(audit_cell['arms'][arm]['raw_path'], allow_pickle=False) as z:
            reference = {k: z[k] for k in z.files}
        env, params = env_for_cell(args, args.n_envs, dev, entry['cell'])
        # Reproduce the frozen whole batch immediately before each branch run.
        control = run_branch(env, sim, controller, params, entry['episode_seed'], reference,
                             np.full(args.n_envs, -1))
        verify_replay(control, entry['episodes'][arm])
        if not np.allclose(control['live_risk'], reference['live_risk'], rtol=1e-5, atol=1e-6):
            raise RuntimeError('control risk trace differs from saved replay')
        skip_at = np.full(args.n_envs, -1)
        witnesses = [r for r in selected if r['grid_index'] == ci]
        for r in witnesses:
            skip_at[r['episode_row']] = r['first_divergence']['t']
        branch = run_branch(env, sim, controller, params, entry['episode_seed'], reference, skip_at)
        raw_path = output / f'cell_{ci:02d}.npz'
        with open(str(raw_path) + '.tmp', 'wb') as f:
            np.savez_compressed(f, **branch)
        os.replace(str(raw_path) + '.tmp', raw_path)
        rows = []
        for r in witnesses:
            i = r['episode_row']
            rows.append({'episode_row': i, 'original_safe_harm': r['safe_harm'],
                         'original_safe_rescue': r['safe_rescue'], 'skip_t': int(skip_at[i]),
                         'original_success': r['success'], 'original_violation': r['violation'],
                         'branch_success': bool(branch['success'][i]),
                         'branch_violation': bool(branch['violation'][i]),
                         'branch_safe_success': bool(branch['success'][i] and not branch['violation'][i]),
                         'branch_first_viol': int(branch['first_viol'][i])})
        cell = {'grid_index': ci, 'cell': entry['cell'], 'raw_path': str(raw_path),
                'raw_sha256': sha256(raw_path), 'control_replay_matches': True,
                'unselected_rows_match': True, 'rows': rows}
        summary['per_cell'].append(cell)
        atomic_json(cell, output / f'cell_{ci:02d}.json')
        atomic_json(summary, output / 'summary.partial.json')
        print(f'[branch] cell={ci} control=matched selected={len(rows)} saved={raw_path}', flush=True)
    rows = [r for c in summary['per_cell'] for r in c['rows']]
    summary['counts'] = {
        'original_harms': sum(r['original_safe_harm'] for r in rows),
        'harms_reverted_to_safe_success': sum(r['original_safe_harm'] and r['branch_safe_success'] for r in rows),
        'original_rescues': sum(r['original_safe_rescue'] for r in rows),
        'rescues_lost': sum(r['original_safe_rescue'] and not r['branch_safe_success'] for r in rows)}
    summary['complete'] = True
    atomic_json(summary, output / 'summary.json')
    print('[branch] complete', summary['counts'], flush=True)


if __name__ == '__main__':
    main()
