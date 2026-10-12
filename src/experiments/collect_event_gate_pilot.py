"""Reachable-state coverage pilot before learning a separate event gate.

Paired root episodes: policy-alone, frozen repair, one bounded drive pulse,
one bounded brake pulse. Pulses depend only on current observed goal gap and
pre-sampled constants, never future risk labels or hidden physical parameters.
All branches of a root stay together. No training or validation-grid rollout.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from _common import env_for_cell, load_policy, load_predictor, setup
from _audit import SCALES, label_chunk
from audit_verifier import atomic_json
from checkvla.verifier import Controller
from common import PREDICTOR_TRAIN_RANGES, load_json
from validate_sequence_risk import (CONTEXT_KEYS, atomic_torch, candidate_labels,
                                    controller, digest)
import torch


BASES = {'train': 74000, 'dev': 75000}
COLLECTORS = ('none', 'frozen', 'drive', 'brake')
# Pilot screening thresholds, not statistical guarantees or permission to deploy.
MIN_ROOTS = {'train': {'immediate': 32, 'anticipation': 24, 'safe_contact': 64},
             'dev': {'immediate': 16, 'anticipation': 12, 'safe_contact': 32}}
MIN_PER_BIN = {'train': 2, 'dev': 1}


class PulseController(Controller):
    """One pre-specified five-step intervention; subsequent policy is unchanged."""
    def __init__(self, policy, sim, kind, seed):
        if kind not in ('drive', 'brake'):
            raise ValueError(kind)
        super().__init__(policy, sim, 'none', chunk_k=5)
        self.kind, self.seed = kind, seed

    def reset(self, env, obs):
        super().reset(env, obs)
        g = torch.Generator().manual_seed(self.seed)
        u = torch.rand(len(obs), 5, generator=g).to(obs.device)
        self.onset_gap = .04 + .12 * u[:, 0]
        self.gain = 1.25 + .75 * u[:, 1]
        self.forward_bias = .03 + .09 * u[:, 2]
        self.lateral_bias = .02 * (2 * u[:, 3] - 1)
        self.brake_scale = obs.new_tensor([.05, .25, .55])[(u[:, 4] * 3).long()]
        self.pulsed = torch.zeros(len(obs), dtype=torch.bool, device=obs.device)

    @torch.no_grad()
    def act(self, env, obs):
        free = self.plan_left <= 0
        action, info = super().act(env, obs)
        # obs[:,0] is object x minus target x, in metres; no GT future access.
        gap = -obs[:, 0]
        mask = free & ~self.pulsed & (gap > 0) & (gap <= self.onset_gap)
        if mask.any():
            original = info['chunk'][mask]
            if self.kind == 'drive':
                modified = original * self.gain[mask, None, None]
                modified[:, :, 0] += self.forward_bias[mask, None]
                modified[:, :, 1] += self.lateral_bias[mask, None]
                modified.clamp_(-self.sim.max_vel, self.sim.max_vel)
            else:
                modified = original * self.brake_scale[mask, None, None]
            # super().act has already consumed step zero: retain ptr=1/left=4.
            self.plan[mask] = modified
            action[mask] = modified[:, 0]
            self.a_prev = action.clone()
            self.pulsed |= mask
        info['collector_pulse'] = mask
        return action, info


def event_masks(data):
    """Original proposals only, before any prior executed constraint violation."""
    r = data['risk'][:, 0].amax(-1)
    r5, r10 = r[:, :5].amax(-1), r.amax(-1)
    pre = ~data['prior_violation'].bool()
    return {'immediate': pre & (r5 > 1),
            'anticipation': pre & (r5 <= 1) & (r10 > 1),
            'safe_contact': pre & (r10 <= 1) & (r10 >= .1)}


def coverage(data):
    masks = event_masks(data)
    result = {'root_episodes': int(data['episode_id'].unique().numel()),
              'decisions': len(data['obs']),
              'pre_violation_decisions': int((~data['prior_violation'].bool()).sum()),
              'events': {}}
    # Four fixed, in-range friction/mass strata. Physics is used for offline
    # composition reporting only, never as learned-model input.
    bins = (data['physical_mults'][:, 0] >= 1.).long() * 2 + (data['physical_mults'][:, 1] >= 1.1).long()
    for name, mask in masks.items():
        result['events'][name] = {
            'decisions': int(mask.sum()),
            'root_episodes': int(data['episode_id'][mask].unique().numel()),
            'root_episodes_per_physics_bin': [int(data['episode_id'][mask & (bins == b)].unique().numel())
                                            for b in range(4)]}
    return result


def passes_pilot(summary, split):
    return all(summary['events'][name]['root_episodes'] >= minimum
               and min(summary['events'][name]['root_episodes_per_physics_bin']) >= MIN_PER_BIN[split]
               for name, minimum in MIN_ROOTS[split].items())


@torch.no_grad()
def collect_episode_batch(args, sim, policy, anchor, tau, split, batch, kind, output, smoke):
    seed = (98000 + (0 if split == 'train' else 100) if smoke else BASES[split]) + batch
    env, _ = env_for_cell(args, args.n_envs, sim.device, {})
    params = sim.sample_params(args.n_envs, PREDICTOR_TRAIN_RANGES['push'],
                               torch.Generator().manual_seed(seed + 1000000))
    nominal = sim.nominal_params()
    mults = torch.stack([params[k] / nominal[k] for k in ('friction', 'mass')], -1)
    c = (PulseController(policy, sim, kind, seed + 2000000) if kind in ('drive', 'brake')
         else controller(policy, sim, anchor if kind == 'frozen' else None, tau))
    obs = env.reset(params, seed)
    c.reset(env, obs)
    prior_violation = torch.zeros(args.n_envs, dtype=torch.bool, device=sim.device)
    rows, live_risk, live_actions, pulse_count = {}, [], [], 0

    def append(key, value):
        rows.setdefault(key, []).append(value.detach().cpu().clone())

    for t in range(sim.T):
        if t % 5 == 0:
            state = env.get_state().clone()
            ctx = {'obs': obs.clone(), 'obs_prev': c.obs_prev.clone(), 'a_prev': c.a_prev.clone()}
            if t:
                torch.testing.assert_close(ctx['a_prev'], live_actions[-1], rtol=0, atol=0)
        action, info = c.act(env, obs)
        if t % 5 == 0:
            actions, risks = candidate_labels(sim, state, params, info['chunk'])
            applied_risk, _ = label_chunk(sim, state, params, c.plan, 5)
            for k, v in ctx.items():
                append(k, v)
            append('state', state)
            append('actions', actions)
            append('risk', risks)
            append('applied', c.plan)
            append('applied_risk', applied_risk)
            append('prior_violation', prior_violation)
            append('physical_mults', mults)
            append('episode_id', torch.arange(args.n_envs) + seed * args.n_envs)
            append('decision_t', torch.full((args.n_envs,), t))
            append('collector', torch.full((args.n_envs,), COLLECTORS.index(kind)))
            pulse = info.get('collector_pulse', torch.zeros_like(prior_violation))
            append('collector_pulse', pulse)
            pulse_count += int(pulse.sum())
            if not torch.equal(state, env.get_state()):
                raise RuntimeError('offline labels changed live state')
        live_actions.append(action.clone())
        obs, risk, success = env.step(action)
        live_risk.append(risk.amax(-1).cpu())
        prior_violation |= risk.amax(-1) > 1
    if c.n_policy_calls != 16 or c.n_policy_rows != args.n_envs * 16:
        raise RuntimeError('collection changed H10/k5 cadence')
    data = {k: torch.cat(v) for k, v in rows.items()}
    expected = torch.stack(live_risk).reshape(16, 5, args.n_envs).permute(0, 2, 1).reshape(-1, 5)
    torch.testing.assert_close(data['applied_risk'][:, :5], expected, rtol=1e-5, atol=1e-6)
    if not all(torch.isfinite(v).all() for v in data.values()):
        raise RuntimeError('nonfinite collection')
    path = output / f'{split}_{batch:02d}_{kind}.pt'
    atomic_torch(data, path)
    outcomes = {'success': success.cpu(), 'violation': prior_violation.cpu(),
                'live_risk': torch.stack(live_risk), 'live_actions': torch.stack(live_actions).cpu()}
    outcome_path = output / f'{split}_{batch:02d}_{kind}_outcomes.pt'
    atomic_torch(outcomes, outcome_path)
    return data, {'split': split, 'batch': batch, 'seed': seed, 'collector': kind,
                  'path': str(path), 'sha256': digest(path), 'outcomes_path': str(outcome_path),
                  'outcomes_sha256': digest(outcome_path), 'pulse_episodes': pulse_count,
                  'n_policy_calls': c.n_policy_calls, 'n_policy_rows': c.n_policy_rows,
                  'prefix_label_parity': True, 'coverage': coverage(data)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source_run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    source = Path(cli.source_run).resolve()
    status = load_json(source / 'status.json')
    if not status['complete'] or status['smoke']:
        raise ValueError('requires completed production source')
    for a in status['source_artifacts'].values():
        if digest(a['path']) != a['sha256']:
            raise RuntimeError('frozen artifact changed')
    helper = Path(__file__).with_name('validate_sequence_risk.py')
    if digest(helper) != status['new_code_hashes'][helper.name]:
        raise RuntimeError('source collection/label helper changed')
    args = SimpleNamespace(**status['effective_policy_args'])
    if args.policy != 'openvla' or args.n_envs != 32 or args.chunk_k != 5 or args.vla_batch_size != 8:
        raise ValueError('requires full OpenVLA matched H10/k5/microbatch8 source')
    args.device = cli.device
    counts = {'train': 4, 'dev': 2}
    if cli.smoke:
        args.policy, args.n_envs = 'bc', 4
        counts = {'train': 1, 'dev': 1}
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dev, sim, od = setup(args)
    report = {'complete': False, 'smoke': cli.smoke, 'stage': 'collection', 'batches': [],
              'protocol': {'source_status_sha256': digest(source / 'status.json'),
                  'source_artifacts': status['source_artifacts'], 'args': vars(args),
                  'pilot_code_sha256': digest(__file__), 'split_seed_bases': BASES,
                  'label_helper_sha256': digest(helper),
                  'batch_counts': counts, 'collectors': COLLECTORS, 'scales': SCALES,
                  'pulse': {'max_pulses_per_episode': 1, 'duration_steps': 5,
                            'onset_goal_gap_m': [.04, .16], 'drive_gain': [1.25, 2.],
                            'forward_bias_m_per_s': [.03, .12], 'lateral_bias_m_per_s': [-.02, .02],
                            'brake_scales': [.05, .25, .55], 'velocity_clamp': sim.max_vel,
                            'access': 'current observation and reset-seeded random constants only'},
                  'physics_ranges': PREDICTOR_TRAIN_RANGES['push'],
                  'min_root_episodes': MIN_ROOTS, 'min_per_physics_bin': MIN_PER_BIN,
                  'gate_scope': 'original proposals, before prior violation, all stages',
                  'calibration': 'not collected in this pilot; fresh policy-only split required after collector selection',
                  'purpose': 'coverage feasibility, not efficacy or automatic training authorization'}}
    # Persist screening criteria before any model loading, rollout or new labels.
    atomic_json(report, output / 'summary.partial.json')
    policy = load_policy(args, od, dev, sim)
    if policy.H != 10:
        raise ValueError('requires H10')
    anchor = load_predictor(args, od, 'orbisim', dev, sim)
    tau = load_json(args.calibration_path)['tau']['orbisim']
    summaries, split_ids = {}, {}
    for split, n_batches in counts.items():
        all_data = []
        for batch in range(n_batches):
            initial = None
            for kind in COLLECTORS:
                data, meta = collect_episode_batch(args, sim, policy, anchor, tau, split, batch, kind, output, cli.smoke)
                snapshot = {k: data[k][:args.n_envs] for k in ('state', 'obs', 'actions', 'physical_mults', 'episode_id')}
                if initial is None:
                    initial = snapshot
                else:
                    for key, value in snapshot.items():
                        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
                meta['paired_initial_state_proposal_params_match'] = True
                all_data.append(data)
                report['batches'].append(meta)
                atomic_json(report, output / 'summary.partial.json')
                print(f'[coverage] {split} batch={batch} collector={kind} events={meta["coverage"]["events"]}', flush=True)
        merged = {k: torch.cat([d[k] for d in all_data]) for k in all_data[0]}
        split_ids[split] = set(merged['episode_id'].tolist())
        summaries[split] = {'all_collectors': coverage(merged), 'by_collector': {}}
        for i, kind in enumerate(COLLECTORS):
            summaries[split]['by_collector'][kind] = coverage(
                {k: v[merged['collector'] == i] for k, v in merged.items()})
        summaries[split]['passes_pilot'] = passes_pilot(summaries[split]['all_collectors'], split)
        report['coverage'] = summaries
        atomic_json(report, output / 'summary.partial.json')
    if split_ids['train'] & split_ids['dev']:
        raise RuntimeError('root episode split leakage')
    passed = not cli.smoke and all(x['passes_pilot'] for x in summaries.values())
    report.update(complete=True, stage='complete', passes_pilot=passed,
                  next_action='design_frozen_repair_gate_ablation' if passed else 'revise_coverage_protocol_before_training',
                  training_started=False)
    atomic_json(report, output / 'summary.json')
    print(f'[coverage] complete passes_pilot={passed}; no automatic training', flush=True)


if __name__ == '__main__':
    main()
