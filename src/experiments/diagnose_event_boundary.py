"""Training-root boundary reachability and frozen-gate diagnostics.

Three onset shards share the same roots and seven collectors. Labels and frozen
gate scores are recorded offline; neither selects the collector's actions.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from _common import env_for_cell, load_policy, load_predictor, setup
from _audit import label_chunk
from audit_verifier import atomic_json
from checkvla.verifier import Controller
from collect_event_gate_pilot import coverage
from common import PREDICTOR_TRAIN_RANGES, load_json
from models.sequence_risk import SequenceRisk
from validate_sequence_risk import atomic_torch, candidate_labels, digest
import torch


ONSETS = {'early': .20, 'middle': .12, 'late': .06}
ARMS = ('none', 'hard_drive', 'hard_brake', 'gain_110', 'ramp_030', 'ramp_060', 'brake_ramp')


def perturb(chunk, kind, gain, bias, lateral, brake, max_vel):
    """Fixed five-step ramps; no states, risks or learned scores are consulted."""
    out = chunk.clone()
    ramp = torch.arange(1, chunk.shape[1] + 1, device=chunk.device, dtype=chunk.dtype).clamp(max=5) / 5
    if kind == 'hard_drive':
        out *= gain[:, None, None]
        out[:, :, 0] += bias[:, None]
        out[:, :, 1] += lateral[:, None]
    elif kind == 'hard_brake':
        out *= brake[:, None, None]
    elif kind == 'gain_110':
        out *= 1.10
    elif kind in ('ramp_030', 'ramp_060'):
        out[:, :, 0] += (.03 if kind == 'ramp_030' else .06) * ramp
    elif kind == 'brake_ramp':
        out *= (1 - .75 * ramp)[None, :, None]
    elif kind != 'none':
        raise ValueError(kind)
    return out.clamp(-max_vel, max_vel)


class BoundaryController(Controller):
    def __init__(self, policy, sim, kind, onset, seed):
        super().__init__(policy, sim, 'none', chunk_k=5)
        self.kind, self.onset, self.seed = kind, onset, seed

    def reset(self, env, obs):
        super().reset(env, obs)
        u = torch.rand(len(obs), 5, generator=torch.Generator().manual_seed(self.seed)).to(obs.device)
        self.gain, self.bias, self.lateral = 1.25 + .75 * u[:, 1], .03 + .09 * u[:, 2], .02 * (2 * u[:, 3] - 1)
        self.brake = obs.new_tensor([.05, .25, .55])[(u[:, 4] * 3).long()]
        self.pulsed = torch.zeros(len(obs), dtype=torch.bool, device=obs.device)

    @torch.no_grad()
    def act(self, env, obs):
        free = self.plan_left <= 0
        action, info = super().act(env, obs)
        gap = -obs[:, 0]
        mask = free & ~self.pulsed & (gap > 0) & (gap <= self.onset)
        if self.kind == 'none':
            mask[:] = False
        if mask.any():
            changed = perturb(info['chunk'][mask], self.kind, self.gain[mask], self.bias[mask],
                              self.lateral[mask], self.brake[mask], self.sim.max_vel)
            self.plan[mask] = changed
            action[mask] = changed[:, 0]
            self.a_prev = action.clone()
            self.pulsed |= mask
        info['collector_pulse'] = mask
        return action, info


@torch.no_grad()
def collect(args, sim, policy, models, seed, kind, onset, output, source=None):
    env, _ = env_for_cell(args, args.n_envs, sim.device, {})
    params = sim.sample_params(args.n_envs, PREDICTOR_TRAIN_RANGES['push'],
                               torch.Generator().manual_seed(seed + 1000000))
    mults = torch.stack([params[k] / sim.nominal_params()[k] for k in ('friction', 'mass')], -1)
    obs = env.reset(params, seed)
    c = BoundaryController(policy, sim, kind, onset, seed + 2000000)
    c.reset(env, obs)
    prior = torch.zeros(args.n_envs, dtype=torch.bool, device=sim.device)
    pulse_t = torch.full((args.n_envs,), -1, device=sim.device, dtype=torch.long)
    rows, live_risk, live_actions = {}, [], []

    def add(key, value):
        rows.setdefault(key, []).append(value.detach().cpu().clone())

    for t in range(sim.T):
        if t % 5 == 0:
            state = env.get_state().clone()
            ctx = {'obs': obs.clone(), 'obs_prev': c.obs_prev.clone(), 'a_prev': c.a_prev.clone()}
            if t:
                torch.testing.assert_close(ctx['a_prev'], live_actions[-1], rtol=0, atol=0)
        action, info = c.act(env, obs)
        if t % 5 == 0:
            proposal = info['chunk']
            actions, risk = candidate_labels(sim, state, params, proposal)
            applied_risk, _ = label_chunk(sim, state, params, c.plan, 5)
            reverse = proposal.flip(1)
            reverse_risk, _ = label_chunk(sim, state, params, reverse, 5)
            scores, reverse_scores = [], []
            for model in models.values():
                mean, std = model.risk(ctx, proposal)
                scores.append((mean + std).amax(-1))
                mean, std = model.risk(ctx, reverse)
                reverse_scores.append((mean + std).amax(-1))
            pulse = info['collector_pulse']
            pulse_t[pulse] = t
            for k, v in ctx.items():
                add(k, v)
            for k, v in {'state': state, 'actions': actions, 'risk': risk, 'applied': c.plan,
                         'applied_risk': applied_risk, 'reverse_risk': reverse_risk,
                         'scores': torch.stack(scores, 1), 'reverse_scores': torch.stack(reverse_scores, 1),
                         'prior_violation': prior, 'collector_pulse': pulse, 'pulse_t': pulse_t,
                         'physical_mults': mults, 'episode_id': torch.arange(args.n_envs) + seed * args.n_envs,
                         'decision_t': torch.full((args.n_envs,), t)}.items():
                add(k, v)
            torch.testing.assert_close(env.get_state(), state, rtol=0, atol=0)
        live_actions.append(action.clone())
        obs, risk, success = env.step(action)
        live_risk.append(risk.amax(-1).cpu())
        prior |= risk.amax(-1) > 1
    assert c.n_policy_calls == 16 and c.n_policy_rows == 16 * args.n_envs
    if c.needs_rgb:
        assert c.n_render_calls == 16 and c.n_render_rows == 16 * args.n_envs
    data = {k: torch.cat(v) for k, v in rows.items()}
    actual = torch.stack(live_risk).reshape(16, 5, args.n_envs).permute(0, 2, 1).reshape(-1, 5)
    torch.testing.assert_close(data['applied_risk'][:, :5], actual, rtol=1e-5, atol=1e-6)
    assert all(torch.isfinite(v).all() for v in data.values())
    if source is not None:
        # Strict replay on the original production device family; stop rather
        # than weaken tolerance if another device changes the experiment.
        for key in ('state', 'obs', 'obs_prev', 'a_prev', 'actions', 'risk', 'physical_mults',
                    'episode_id', 'decision_t', 'prior_violation'):
            torch.testing.assert_close(data[key], source[key], rtol=1e-5, atol=1e-6)
    path = output / f'{seed}_{kind}.pt'
    atomic_torch(data, path)
    opath = output / f'{seed}_{kind}_outcomes.pt'
    atomic_torch({'success': success.cpu(), 'violation': prior.cpu(),
                  'live_risk': torch.stack(live_risk), 'live_actions': torch.stack(live_actions).cpu()}, opath)
    pulse = data['collector_pulse'] & ~data['prior_violation']
    meta = {'seed': seed, 'arm': kind, 'path': str(path), 'sha256': digest(path),
            'outcomes_path': str(opath), 'outcomes_sha256': digest(opath), 'coverage': coverage(data),
            'pre_violation_pulses': int(pulse.sum()),
            'first_violation_during_pulse': int((pulse & (data['applied_risk'][:, :5].amax(-1) > 1)).sum()),
            'cadence_and_prefix_parity': True, 'source_replay_checked': source is not None}
    return data, meta


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source_run', required=True)
    p.add_argument('--pilot_run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--onset', choices=ONSETS, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    source, pilot, output = Path(cli.source_run).resolve(), Path(cli.pilot_run).resolve(), Path(cli.output).resolve()
    status, training = [load_json(source / f) for f in ('status.json', 'training.json')]
    old = load_json(pilot / 'summary.json')
    assert status['complete'] and training['complete'] and old['complete']
    assert not status['smoke'] and not old['smoke']
    for artifact in status['source_artifacts'].values():
        assert digest(artifact['path']) == artifact['sha256']
    for file in (Path(__file__).with_name('validate_sequence_risk.py'),
                 Path(__file__).parents[1] / 'models/sequence_risk.py'):
        assert digest(file) == status['new_code_hashes'][file.name]
    args = SimpleNamespace(**status['effective_policy_args'])
    assert args.policy == 'openvla' and args.n_envs == 32 and args.chunk_k == 5 and args.vla_batch_size == 8
    args.device = cli.device
    if cli.smoke:
        args.policy, args.n_envs = 'bc', 4
    dev, sim, od = setup(args)
    output.mkdir(parents=True, exist_ok=False)
    seeds = [98200] if cli.smoke else list(range(74000, 74004))
    summary = {'complete': False, 'smoke': cli.smoke, 'batches': [], 'protocol': {
        'onset': cli.onset, 'onset_gap_m': ONSETS[cli.onset], 'arms': ARMS, 'seeds': seeds,
        'scope': 'reused training roots only; diagnosis, not independent evaluation',
        'source_status_sha256': digest(source / 'status.json'), 'pilot_summary_sha256': digest(pilot / 'summary.json'),
        'training_sha256': digest(source / 'training.json'), 'source_artifacts': status['source_artifacts'],
        'model_artifacts': training['models'], 'args': vars(args),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'registered_protocol_sha256': digest(Path(__file__).parents[2] / 'exp_record/event_boundary_20261013.md'),
        'code_hashes': {str(f): digest(f) for f in (Path(__file__), Path(__file__).with_name('validate_sequence_risk.py'),
                         Path(__file__).with_name('analyze_event_boundary.py'),
                         Path(__file__).parents[1] / 'models/sequence_risk.py')},
        'reverse_probe': 'offline original action reversal only; never executed or offered to repair',
        'candidate_budget': 8, 'H': 10, 'k': 5, 'max_pulses_per_episode': 1}}
    atomic_json(summary, output / 'summary.partial.json')
    policy = load_policy(args, od, dev, sim)
    assert policy.H == 10
    models = {'frozen': load_predictor(args, od, 'orbisim', dev, sim)}
    taus = {'frozen': load_json(args.calibration_path)['tau']['orbisim']}
    for name, meta in training['models'].items():
        assert digest(meta['checkpoint']) == meta['sha256']
        ck = torch.load(meta['checkpoint'], map_location='cpu', weights_only=True)
        model = SequenceRisk(ck['kind'], ck['hidden'], ck['E']).to(dev).eval()
        model.load_state_dict(ck['state'])
        models[name], taus[name] = model, meta['tau']
    summary['protocol'].update(model_order=list(models), frozen_taus=taus)
    atomic_json(summary, output / 'summary.partial.json')
    for batch, seed in enumerate(seeds):
        original = None
        if not cli.smoke:
            meta = next(m for m in old['batches'] if m['split'] == 'train' and m['batch'] == batch and m['collector'] == 'none')
            assert digest(meta['path']) == meta['sha256']
            original = torch.load(meta['path'], map_location='cpu', weights_only=True)
        initial = None
        for kind in ARMS:
            data, meta = collect(args, sim, policy, models, seed, kind, ONSETS[cli.onset], output,
                                 original if kind == 'none' else None)
            snapshot = {k: data[k][:args.n_envs] for k in ('state', 'actions', 'physical_mults', 'episode_id')}
            if initial is None:
                initial = snapshot
            else:
                for key in snapshot:
                    torch.testing.assert_close(snapshot[key], initial[key], rtol=0, atol=0)
            meta['paired_initialization'] = True
            summary['batches'].append(meta)
            atomic_json(summary, output / 'summary.partial.json')
            print(f'[boundary] {cli.onset} seed={seed} arm={kind} events={meta["coverage"]["events"]} pulse_viol={meta["first_violation_during_pulse"]}/{meta["pre_violation_pulses"]}', flush=True)
    summary.update(complete=True, training_started=False)
    atomic_json(summary, output / 'summary.json')
    print('[boundary] complete', output, flush=True)


if __name__ == '__main__':
    main()
