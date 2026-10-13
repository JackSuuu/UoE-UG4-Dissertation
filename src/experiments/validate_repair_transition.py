"""Matched validation of continuity-aware proposals after an actual repair.

No collector pulses. A deterministic proposal transform is applied before the
unchanged gate/repair, using only the last executed action and a repair flag.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from _common import env_for_cell, load_policy, load_predictor, setup
from _audit import label_chunk, verify_replay
from analyze_sequence_risk import ExactRisk
from audit_verifier import atomic_json
from checkvla.verifier import Controller
from common import load_json
from diagnose_risk_controller import GateRepairVerifier
from validate_sequence_risk import digest
import numpy as np
import torch


MODES = ('direct', 'ramp', 'flat', 'pause')
ARMS = ('none',) + tuple(f'{gate}_{mode}' for gate in ('learned', 'gt') for mode in MODES)


def transition_proposal(raw, previous_action, mode, k=5):
    """Transform only the committed prefix; the gate sees this exact candidate."""
    if mode == 'direct':
        return raw.clone()
    out = raw.clone()
    if mode == 'pause':
        out[:, :k] = 0
    elif mode in ('ramp', 'flat'):
        w = (torch.arange(1, k + 1, device=raw.device, dtype=raw.dtype) / k
             if mode == 'ramp' else raw.new_full((k,), .6))
        out[:, :k] = previous_action[:, None] * (1 - w[None, :, None]) + raw[:, :k] * w[None, :, None]
    else:
        raise ValueError(mode)
    return out


class TransitionPolicy:
    """One actor call; deterministic shaping before Controller checks the chunk."""
    def __init__(self, base, owner):
        self.base, self.owner = base, owner

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, obs, image=None, instruction=None):
        raw = self.base(obs, image, instruction)
        owner, rows = self.owner, self.owner.free_rows
        owner.raw_proposal[rows] = raw
        use = owner.postrepair[rows]
        out = raw.clone()
        if use.any():
            out[use] = transition_proposal(raw[use], owner.a_prev[rows][use], owner.transition)
        return out


class TransitionController(Controller):
    def __init__(self, policy, sim, verifier=None, transition='direct'):
        if transition not in MODES:
            raise ValueError(transition)
        super().__init__(policy, sim, 'checkvla' if verifier else 'none', verifier=verifier, chunk_k=5)
        self.transition = transition
        self.policy = TransitionPolicy(policy, self)

    def reset(self, env, obs):
        super().reset(env, obs)
        self.previous_repair = torch.zeros(len(obs), dtype=torch.bool, device=obs.device)
        self.postrepair = self.previous_repair.clone()
        self.raw_proposal = torch.zeros_like(self.chunk)
        self.free_rows = torch.empty(0, dtype=torch.long, device=obs.device)

    @torch.no_grad()
    def act(self, env, obs):
        free = self.plan_left <= 0
        self.free_rows = free.nonzero().squeeze(1)
        self.postrepair = free & self.previous_repair
        action, info = super().act(env, obs)
        # A triggered abstention/no-op does not arm a bridge. A bridge by itself
        # does not re-arm another bridge. A new real repair does re-arm it.
        repair_changed = free & info['trig'] & ((self.plan[:, :5] - info['chunk'][:, :5]).abs().amax((1, 2)) > 1e-7)
        self.previous_repair[free] = repair_changed[free]
        info.update(raw_proposal=self.raw_proposal, postrepair=self.postrepair.clone(),
                    repair_changed=repair_changed,
                    transition_changed=free & ((info['chunk'][:, :5] - self.raw_proposal[:, :5]).abs().amax((1, 2)) > 1e-7))
        return action, info


class CountRisk:
    def __init__(self, base):
        self.base, self.calls, self.rows = base, 0, 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def risk(self, ctx, actions):
        self.calls += 1
        self.rows += len(actions)
        return self.base.risk(ctx, actions)


@torch.no_grad()
def rollout(env, sim, ctrl, params, seed):
    obs = env.reset(params, seed)
    ctrl.reset(env, obs)
    n, device = len(obs), obs.device
    first_viol = torch.full((n,), -1, dtype=torch.long, device=device)
    first_trig, first_success = first_viol.clone(), first_viol.clone()
    violations = torch.zeros(n, dtype=torch.bool, device=device)
    interventions = torch.zeros(n, device=device)
    trace, live, actions = {}, [], []

    def add(key, value):
        trace.setdefault(key, []).append(value.detach().cpu().numpy().copy())

    for t in range(sim.T):
        if t % 5 == 0:
            state = env.get_state().clone()
            ctx = {k: v.clone() for k, v in {'obs': obs, 'obs_prev': ctrl.obs_prev, 'a_prev': ctrl.a_prev}.items()}
            if t:
                torch.testing.assert_close(ctx['a_prev'], actions[-1], rtol=0, atol=0)
        action, info = ctrl.act(env, obs)
        if t % 5 == 0:
            for k, v in ctx.items():
                add(k, v)
            add('state', state)
            add('prior_violation', violations)
            for key in ('postrepair', 'repair_changed', 'transition_changed', 'raw_proposal'):
                add(key, info[key])
            add('proposal', info['chunk'])
            add('applied', ctrl.plan)
            add('trigger', info['trig'])
            add('score', info['score'])
            # Fixed offline audit queries; never influence control decisions.
            for key, chunk in (('raw_risk', info['raw_proposal']), ('prepared_risk', info['chunk']), ('applied_risk', ctrl.plan)):
                r, _ = label_chunk(sim, state, params, chunk, 5)
                add(key, r)
            torch.testing.assert_close(state, env.get_state(), rtol=0, atol=0)
        actions.append(action.clone())
        obs, risk, success = env.step(action)
        r = risk.amax(-1)
        live.append(r.cpu().numpy())
        first_viol = torch.where((r > 1) & (first_viol < 0), t, first_viol)
        first_trig = torch.where(info['trig'] & (first_trig < 0), t, first_trig)
        first_success = torch.where(success & (first_success < 0), t, first_success)
        violations |= r > 1
        interventions += info['trig'].float()
    assert ctrl.n_policy_calls == 16 and ctrl.n_policy_rows == n * 16
    if ctrl.needs_rgb:
        assert ctrl.n_render_calls == 16 and ctrl.n_render_rows == n * 16
    data = {k: np.stack(v) for k, v in trace.items()}
    data.update(live_risk=np.stack(live), live_actions=torch.stack(actions).cpu().numpy(),
                decision_t=np.arange(0, sim.T, 5), success=success.cpu().numpy(),
                violation=violations.cpu().numpy(), first_viol=first_viol.cpu().numpy(),
                first_trig=first_trig.cpu().numpy(), first_success=first_success.cpu().numpy(),
                n_interventions=interventions.cpu().numpy(), final_task_cost=sim.task_cost(env.get_state()).cpu().numpy())
    expected = data['live_risk'].reshape(16, 5, n).transpose(0, 2, 1)
    np.testing.assert_allclose(data['applied_risk'][:, :, :5], expected, rtol=1e-5, atol=1e-6)
    if ctrl.verifier:
        assert ctrl.verifier.gate.rows == n * 16
        assert ctrl.verifier.predictor.rows == 8 * int(interventions.sum())
    assert all(np.isfinite(v).all() for v in data.values())
    return data


def prebranch_parity(data, reference):
    """Every row must replay until its first actual transition modification."""
    changed = data['transition_changed']
    before = np.concatenate([np.zeros_like(changed[:1]), np.maximum.accumulate(changed, axis=0)[:-1]]) == 0
    for key in ('state', 'obs', 'obs_prev', 'a_prev', 'raw_proposal'):
        np.testing.assert_allclose(data[key][before], reference[key][before], rtol=1e-5, atol=1e-6)
    untouched = before & ~changed
    for key in ('proposal', 'applied', 'score', 'trigger'):
        np.testing.assert_allclose(data[key][untouched], reference[key][untouched], rtol=1e-5, atol=1e-6)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source_run', required=True)
    p.add_argument('--factorial_run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n_shards', type=int, default=3)
    p.add_argument('--device', default='auto')
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    assert 0 <= cli.shard < cli.n_shards
    source, factorial, root = Path(cli.source_run).resolve(), Path(cli.factorial_run).resolve(), Path(cli.output).resolve()
    status, old = [load_json(source / name) for name in ('status.json', 'evaluation.json')]
    fact = load_json(factorial / 'summary.json')
    assert status['complete'] and old['complete'] and fact['complete'] and not status['smoke']
    assert [r['episode_seed'] for r in old['per_cell']] == list(range(26000, 26020))
    for artifact in status['source_artifacts'].values():
        assert digest(artifact['path']) == artifact['sha256']
    args = SimpleNamespace(**status['effective_policy_args'])
    assert args.policy == 'openvla' and args.n_envs == 32 and args.chunk_k == 5 and args.vla_batch_size == 8
    args.device = cli.device
    if cli.smoke:
        assert cli.n_shards == 1
        args.policy, args.n_envs = 'bc', 4
    device, sim, od = setup(args)
    output = root / f'shard_{cli.shard}'
    output.mkdir(parents=True, exist_ok=False)
    premise = root / 'actual_repair_audit.json'
    if not cli.smoke:
        assert premise.exists() and load_json(premise)['complete']
        assert load_json(premise)['factorial_summary_sha256'] == digest(factorial / 'summary.json')
    indices = [0] if cli.smoke else list(range(cli.shard, len(old['per_cell']), cli.n_shards))
    summary = {'complete': False, 'per_cell': [], 'metadata': {
        'smoke': cli.smoke, 'shard': cli.shard, 'n_shards': cli.n_shards, 'cell_indices': indices,
        'arms': ARMS, 'args': vars(args), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_status_sha256': digest(source / 'status.json'), 'source_evaluation_sha256': digest(source / 'evaluation.json'),
        'actual_repair_audit_sha256': digest(premise) if premise.exists() else None,
        'factorial_summary_sha256': digest(factorial / 'summary.json'), 'source_artifacts': status['source_artifacts'],
        'protocol_sha256': digest(Path(__file__).parents[2] / 'exp_record/repair_transition_20261013.md'),
        'code_hashes': {str(f): digest(f) for f in (Path(__file__), Path(__file__).with_name('analyze_repair_transition.py'),
                         Path(__file__).with_name('diagnose_risk_controller.py'), Path(__file__).with_name('analyze_sequence_risk.py'))},
        'scope': 'validation design; primary learned ramp; GT gates privileged diagnostic only',
        'H': 10, 'k': 5, 'gate_queries_per_decision': 1, 'repair_risk_queries_per_trigger': 8}}
    atomic_json(summary, output / 'summary.partial.json')
    policy = load_policy(args, od, device, sim)
    assert policy.H == 10
    anchor = load_predictor(args, od, 'orbisim', device, sim)
    tau = load_json(args.calibration_path)['tau']['orbisim']
    for ci in indices:
        entry = old['per_cell'][ci]
        source_fact = fact['per_cell'][ci]
        assert entry['cell'] == source_fact['cell'] and entry['episode_seed'] == source_fact['episode_seed']
        env, params = env_for_cell(args, args.n_envs, device, entry['cell'])
        seed = 98300 + ci if cli.smoke else entry['episode_seed']
        row = {'grid_index': ci, 'cell': entry['cell'], 'ood': entry['ood'], 'episode_seed': seed, 'episodes': {}, 'raw': {}}
        references = {}
        for arm in ARMS:
            if arm == 'none':
                verifier, mode = None, 'direct'
            else:
                gate, mode = arm.split('_', 1)
                gate_model = anchor if gate == 'learned' else ExactRisk(sim, entry['cell'])
                verifier = GateRepairVerifier(CountRisk(gate_model), CountRisk(anchor), tau if gate == 'learned' else 1., sim.max_vel)
            ctrl = TransitionController(policy, sim, verifier, mode)
            data = rollout(env, sim, ctrl, params, seed)
            if arm in ('none', 'learned_direct', 'gt_direct'):
                references[arm] = data
                if not cli.smoke:
                    source_arm = {'none': 'none', 'learned_direct': 'frozen_ll', 'gt_direct': 'frozen_gl'}[arm]
                    verify_replay(data, source_fact['episodes'][source_arm])
                    raw = source_fact['raw'][source_arm]
                    assert digest(raw['path']) == raw['sha256']
                    with np.load(raw['path'], allow_pickle=False) as z:
                        np.testing.assert_allclose(data['live_risk'], z['risk'].max(-1).T, rtol=1e-5, atol=1e-6)
            else:
                prebranch_parity(data, references[arm.split('_')[0] + '_direct'])
            # Every arm begins on the same paired root/context/proposal.
            for key in ('state', 'raw_proposal'):
                np.testing.assert_array_equal(data[key][0], references['none'][key][0])
            path = output / f'cell_{ci:02d}_{arm}.npz'
            with open(str(path) + '.tmp', 'wb') as f:
                np.savez_compressed(f, **data)
            os.replace(str(path) + '.tmp', path)
            row['episodes'][arm] = {k: data[k] for k in ('success', 'violation', 'first_viol', 'first_trig', 'first_success', 'n_interventions', 'final_task_cost')}
            row['raw'][arm] = {'path': str(path), 'sha256': digest(path), 'prefix_parity': True,
                               'prebranch_or_source_replay': True, 'n_policy_calls': ctrl.n_policy_calls,
                               'n_policy_rows': ctrl.n_policy_rows,
                               'gate_risk_rows': verifier.gate.rows if verifier else 0,
                               'repair_risk_rows': verifier.predictor.rows if verifier else 0}
            row[arm] = {'SR': float(data['success'].mean()), 'CVR': float(data['violation'].mean()),
                        'safe_success': float((data['success'] & ~data['violation']).mean())}
            atomic_json(row, output / f'cell_{ci:02d}.partial.json')
            print(f'[transition] cell={ci} arm={arm} {row[arm]}', flush=True)
        summary['per_cell'].append(row)
        atomic_json(summary, output / 'summary.partial.json')
    summary['complete'] = True
    atomic_json(summary, output / 'summary.json')
    from analyze_repair_transition import finalize
    finalize(root, cli.n_shards)
    print('[transition] shard complete', cli.shard, flush=True)


if __name__ == '__main__':
    main()
