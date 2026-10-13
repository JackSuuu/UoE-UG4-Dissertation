"""Registered, bounded R5 objective experiment; original H10 repair stays frozen.

Stages are invoked by the durable coordinator. Collection uses fresh roots;
training/evaluation are prohibited until the common support screen passes.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from _common import env_for_cell, load_policy, load_predictor, setup
from _audit import label_chunk, verify_replay
from _paired import pooled_paired
from audit_verifier import atomic_json
from checkvla.reference import calibrate_tau, chunk_score
from checkvla.verifier import Controller, run_episodes, summarize
from collect_event_gate_pilot import coverage
from common import PREDICTOR_TRAIN_RANGES, load_json, seed_all
from diagnose_event_boundary import BoundaryController
from diagnose_risk_controller import GateRepairVerifier
from models.immediate_gate import ImmediateGate, PrefixRisk
from validate_repair_transition import CountRisk
from validate_sequence_risk import (CONTEXT_KEYS, assert_split_disjoint, atomic_torch,
                                    candidate_labels, controller, digest)
import numpy as np
import torch


BASES = {'train': 76000, 'dev': 77000, 'calibration': 78000}
BATCHES = {'train': 4, 'dev': 2, 'calibration': 2}
COLLECTORS = ('none', 'frozen', 'brake_ramp')
SEEDS = (0, 1, 2)
MINIMUM = {'train': (64, 64, 8), 'dev': (32, 32, 4), 'calibration': (32, 32, 4)}
CODE = (Path(__file__), Path(__file__).parents[1] / 'models/immediate_gate.py',
        Path(__file__).with_name('diagnose_event_boundary.py'),
        Path(__file__).with_name('validate_sequence_risk.py'),
        Path(__file__).with_name('diagnose_risk_controller.py'),
        Path(__file__).with_name('validate_repair_transition.py'),
        Path(__file__).with_name('collect_event_gate_pilot.py'),
        Path(__file__).with_name('_audit.py'), Path(__file__).with_name('_paired.py'),
        Path(__file__).parents[1] / 'models/nets.py',
        Path(__file__).parents[1] / 'common.py',
        Path(__file__).parents[2] / 'scripts/run_immediate_gate.py')
PROTOCOL = Path(__file__).parents[2] / 'exp_record/immediate_gate_20261013.md'


def load_split(root, split, eligible=True):
    manifest = load_json(root / f'collection/{split}/summary.json')
    assert manifest['complete']
    parts = []
    for m in manifest['batches']:
        for path, key in ((m['path'], 'sha256'), (m['outcomes_path'], 'outcomes_sha256')):
            assert digest(path) == m[key]
        parts.append(torch.load(m['path'], map_location='cpu', weights_only=True))
    data = {k: torch.cat([d[k] for d in parts]) for k in parts[0]}
    if eligible:
        mask = ~data['prior_violation'].bool()
        data = {k: v[mask] for k, v in data.items()}
    data['split'] = split
    return data


def subset(data, mask):
    return {k: v[mask] if torch.is_tensor(v) else v for k, v in data.items()}


def r5_labels(data):
    return data['risk'][:, 0, :5].amax((1, 2)) > 1


def support(data):
    positive_min, contact_min, bin_min = MINIMUM[data['split']]
    cov = coverage(data)
    natural = subset(data, data['collector'] == 0)
    safe_natural_roots = int(natural['episode_id'][~r5_labels(natural)].unique().numel())
    passed = (cov['events']['immediate']['root_episodes'] >= positive_min
              and cov['events']['safe_contact']['root_episodes'] >= contact_min
              and min(cov['events']['immediate']['root_episodes_per_physics_bin']) >= bin_min
              and min(cov['events']['safe_contact']['root_episodes_per_physics_bin']) >= bin_min
              and safe_natural_roots >= contact_min)
    return {'passes': passed, 'coverage': cov, 'natural_safe_R5_roots': safe_natural_roots,
            'by_collector': {kind: coverage(subset(data, data['collector'] == i))
                             for i, kind in enumerate(COLLECTORS)}}


@torch.no_grad()
def collect_batch(args, sim, policy, anchor, tau, split, batch, kind, output, smoke):
    seed = (BASES[split] if not smoke else BASES[split] + 10000) + batch
    env, _ = env_for_cell(args, args.n_envs, sim.device, {})
    params = sim.sample_params(args.n_envs, PREDICTOR_TRAIN_RANGES['push'],
                               torch.Generator().manual_seed(seed + 1000000))
    mults = torch.stack([params[k] / sim.nominal_params()[k] for k in ('friction', 'mass')], -1)
    c = (BoundaryController(policy, sim, kind, .20, seed + 2000000) if kind == 'brake_ramp'
         else controller(policy, sim, anchor if kind == 'frozen' else None, tau))
    obs = env.reset(params, seed)
    c.reset(env, obs)
    prior = torch.zeros(args.n_envs, device=sim.device, dtype=torch.bool)
    rows, live, actions = {}, [], []

    def add(k, v):
        rows.setdefault(k, []).append(v.detach().cpu().clone())

    for t in range(sim.T):
        if t % 5 == 0:
            state = env.get_state().clone()
            ctx = {'obs': obs.clone(), 'obs_prev': c.obs_prev.clone(), 'a_prev': c.a_prev.clone()}
            if t:
                torch.testing.assert_close(ctx['a_prev'], actions[-1], rtol=0, atol=0)
        a, info = c.act(env, obs)
        if t % 5 == 0:
            candidates, risk = candidate_labels(sim, state, params, info['chunk'])
            applied_risk, _ = label_chunk(sim, state, params, c.plan, 5)
            for k, v in ctx.items():
                add(k, v)
            for k, v in {'state': state, 'actions': candidates, 'risk': risk,
                         'applied': c.plan, 'applied_risk': applied_risk,
                         'prior_violation': prior, 'physical_mults': mults,
                         'collector_pulse': info.get('collector_pulse', torch.zeros_like(prior)),
                         'episode_id': torch.arange(args.n_envs) + seed * args.n_envs,
                         'decision_t': torch.full((args.n_envs,), t),
                         'collector': torch.full((args.n_envs,), COLLECTORS.index(kind))}.items():
                add(k, v)
            torch.testing.assert_close(state, env.get_state(), rtol=0, atol=0)
        actions.append(a.clone())
        obs, risk, success = env.step(a)
        live.append(risk.amax(-1).cpu())
        prior |= risk.amax(-1) > 1
    assert c.n_policy_calls == 16 and c.n_policy_rows == args.n_envs * 16
    if c.needs_rgb:
        assert c.n_render_calls == 16 and c.n_render_rows == args.n_envs * 16
    data = {k: torch.cat(v) for k, v in rows.items()}
    actual = torch.stack(live).reshape(16, 5, args.n_envs).permute(0, 2, 1).reshape(-1, 5)
    torch.testing.assert_close(data['applied_risk'][:, :5], actual, rtol=1e-5, atol=1e-6)
    reconstructed_prior = torch.cat([torch.zeros(1, args.n_envs, dtype=torch.bool),
                                    (torch.stack(live) > 1).cummax(0).values[:-1]])[::5].reshape(-1)
    torch.testing.assert_close(data['prior_violation'], reconstructed_prior, rtol=0, atol=0)
    assert all(torch.isfinite(v).all() for v in data.values()) and not data['risk'][..., 1].any()
    path, opath = output / f'{batch:02d}_{kind}.pt', output / f'{batch:02d}_{kind}_outcomes.pt'
    atomic_torch(data, path)
    atomic_torch({'success': success.cpu(), 'violation': prior.cpu(), 'live_risk': torch.stack(live),
                  'live_actions': torch.stack(actions).cpu()}, opath)
    return data, {'seed': seed, 'collector': kind, 'path': str(path), 'sha256': digest(path),
                  'outcomes_path': str(opath), 'outcomes_sha256': digest(opath),
                  'coverage': coverage(data), 'cadence_and_prefix_parity': True}


def collect(cli, args, sim, od, root):
    split = cli.split
    output = root / 'collection' / split
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'complete': False, 'split': split, 'smoke': cli.smoke, 'batches': [],
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    atomic_json(manifest, output / 'summary.partial.json')
    policy = load_policy(args, od, sim.device, sim)
    assert policy.H == 10
    anchor = load_predictor(args, od, 'orbisim', sim.device, sim)
    tau = load_json(args.calibration_path)['tau']['orbisim']
    for batch in range(1 if cli.smoke else BATCHES[split]):
        initial = None
        for kind in COLLECTORS:
            data, meta = collect_batch(args, sim, policy, anchor, tau, split, batch, kind, output, cli.smoke)
            snapshot = {k: data[k][:args.n_envs] for k in ('state', 'obs', 'actions', 'physical_mults', 'episode_id')}
            if initial is None:
                initial = snapshot
            else:
                for k in initial:
                    torch.testing.assert_close(initial[k], snapshot[k], rtol=0, atol=0)
            manifest['batches'].append(meta)
            atomic_json(manifest, output / 'summary.partial.json')
            print('[R5 collect]', split, batch, kind, meta['coverage']['events'], flush=True)
    manifest['complete'] = True
    atomic_json(manifest, output / 'summary.json')


def screen(root, smoke):
    data = {split: load_split(root, split) for split in BASES}
    assert_split_disjoint(list(data.values()))
    summaries = {split: support(d) for split, d in data.items()}
    passed = not smoke and all(s['passes'] for s in summaries.values())
    result = {'complete': True, 'smoke': smoke, 'passes_support': passed,
              'smoke_only_bypass': smoke, 'splits': summaries,
              'training_authorized': passed, 'H10_training_authorized': False,
              'collection_manifest_hashes': {split: digest(root / f'collection/{split}/summary.json') for split in BASES}}
    atomic_json(result, root / 'support.json')
    print('[R5 support]', result, flush=True)
    return passed or smoke


class RootBalancedSampler:
    """Same 50/50 event/non-event original-decision batches for BOTH objectives."""
    def __init__(self, data, seed):
        self.gen = np.random.default_rng(seed + 82000)
        y, roots = r5_labels(data).numpy(), data['episode_id'].numpy()
        self.groups = []
        for value in (False, True):
            ids = np.flatnonzero(y == value)
            self.groups.append([ids[roots[ids] == r] for r in np.unique(roots[ids])])
        if not all(self.groups):
            raise ValueError('both classes required; no silent training fallback')

    def sample(self, n):
        if n % 2:
            raise ValueError('balanced batch size must be even')
        ids = []
        for groups in self.groups:
            for j in self.gen.integers(len(groups), size=n // 2):
                group = groups[j]
                ids.append(group[self.gen.integers(len(group))])
        return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def scores_on(model, data, device):
    parts = []
    for start in range(0, len(data['obs']), 256):
        ctx = {k: data[k][start:start+256].to(device) for k in CONTEXT_KEYS}
        a = data['actions'][start:start+256, 0].to(device)
        parts.append(chunk_score(model, ctx, a)[0].cpu())
    result = torch.cat(parts)
    assert torch.isfinite(result).all()
    return result


def detection(scores, data, tau):
    y, pred = r5_labels(data).numpy(), scores.numpy() > tau
    roots = data['episode_id'].numpy()
    unique = np.unique(roots)
    counts = np.array([[int((y[roots == r] & pred[roots == r]).sum()), int(y[roots == r].sum()),
                        int((~y[roots == r] & pred[roots == r]).sum()), int((~y[roots == r]).sum())]
                       for r in unique])
    rng = np.random.default_rng(3300)
    boot = counts[rng.integers(len(counts), size=(4000, len(counts)))].sum(1)
    def ratio(num, den):
        valid = den > 0
        return np.quantile(num[valid] / den[valid], [.025, .975]).tolist() if valid.any() else None
    h10only = (~y) & (data['risk'][:, 0].amax((1, 2)).numpy() > 1)
    return {'decisions': len(y), 'roots': len(unique), 'positive_roots': int(len(np.unique(roots[y]))),
            'unsafe_R5': int(y.sum()), 'true_trigger': int((y & pred).sum()),
            'safe_R5': int((~y).sum()), 'false_trigger': int((~y & pred).sum()),
            'recall': float(pred[y].mean()) if y.any() else None,
            'safe_trigger_rate': float(pred[~y].mean()) if (~y).any() else None,
            'recall_root_ci95': ratio(boot[:, 0], boot[:, 1]),
            'safe_trigger_root_ci95': ratio(boot[:, 2], boot[:, 3]),
            'H10_only_events': int(h10only.sum()), 'H10_only_triggers': int((h10only & pred).sum()),
            'uncertainty_scope': '4000 root bootstrap draws; paired collectors stay grouped'}


def dev_report(model, data, tau, device):
    scores = scores_on(model, data, device)
    return {'all': detection(scores, data, tau), 'by_collector': {
        kind: detection(scores[data['collector'] == i], subset(data, data['collector'] == i), tau)
        for i, kind in enumerate(COLLECTORS)}}


def train(cli, args, sim, od, root):
    check = load_json(root / 'support.json')
    assert check['passes_support'] or (cli.smoke and check['smoke_only_bypass'])
    data = {s: load_split(root, s) for s in BASES}
    assert_split_disjoint(list(data.values()))
    if cli.smoke:
        # Explicit synthetic training fixture, only for exercising the sampler
        # and optimiser when the tiny CPU/BC rollout has no unsafe decisions.
        # Persisted collection labels remain untouched; never production data.
        data['train']['risk'] = data['train']['risk'].clone()
        data['train']['risk'][:, 0, :5] = 0
        data['train']['risk'][::2, 0, 4, 0] = 1.1
    cal = subset(data['calibration'], (data['calibration']['collector'] == 0) & ~r5_labels(data['calibration']))
    assert len(cal['obs']) > 0
    output = root / f'training/seed_{cli.seed}'
    output.mkdir(parents=True, exist_ok=False)
    seed_all(cli.seed)
    base = ImmediateGate('regression', hidden=16 if cli.smoke else 256).to(sim.device)
    base.fit_normalizers(data['train'])
    models = {'regression': base, 'event': copy.deepcopy(base)}
    models['event'].objective = 'event'
    for k in base.state_dict():
        torch.testing.assert_close(models['event'].state_dict()[k], base.state_dict()[k], rtol=0, atol=0)
    init_path = output / 'shared_initialization.pt'
    atomic_torch(base.state_dict(), init_path)
    optimizers = {k: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5) for k, m in models.items()}
    sampler = RootBalancedSampler(data['train'], cli.seed)
    iters, bs = (3, 16) if cli.smoke else (3000, 256)
    draw_hash = hashlib.sha256()
    for step in range(iters):
        ids = sampler.sample(bs)
        draw_hash.update(ids.numpy().tobytes())
        ctx = {k: data['train'][k][ids].to(sim.device) for k in CONTEXT_KEYS}
        actions = data['train']['actions'][ids, 0].to(sim.device)
        target = data['train']['risk'][ids, 0].to(sim.device)
        losses = {}
        for kind, model in models.items():
            model.train()
            optimizer = optimizers[kind]
            for group in optimizer.param_groups:
                group['lr'] = 1e-3 * (.5 + .5 * np.cos(np.pi * step / iters))
            loss = model.loss(ctx, actions, target)
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            losses[kind] = float(loss.detach())
        if step % 500 == 0 or step + 1 == iters:
            print('[R5 train]', cli.seed, step + 1, losses, flush=True)
    report = {'complete': False, 'seed': cli.seed, 'models': {}, 'shared_initialization_sha256': digest(init_path),
              'shared_minibatch_sha256': draw_hash.hexdigest(), 'support_sha256': digest(root / 'support.json'),
              'iters': iters, 'smoke': cli.smoke, 'synthetic_training_fixture': cli.smoke,
              'calibration': 'R5-safe, policy-alone actual decisions; nominal alpha .05',
              'calibration_decisions': len(cal['obs']), 'calibration_roots': int(cal['episode_id'].unique().numel())}
    for kind, model in models.items():
        model.eval()
        path = output / f'{kind}.pt'
        atomic_torch({'state': model.state_dict(), 'objective': kind, 'hidden': model.hidden, 'E': model.E}, path)
        tau = calibrate_tau(scores_on(model, cal, sim.device).numpy(), .05)
        report['models'][kind] = {'path': str(path), 'sha256': digest(path), 'tau': tau,
                                  'parameters': sum(p.numel() for p in model.parameters()),
                                  'dev': dev_report(model, data['dev'], tau, sim.device)}
    if cli.seed == 0:
        anchor = load_predictor(args, od, 'orbisim', sim.device, sim)
        prefix = PrefixRisk(anchor)
        tau = calibrate_tau(scores_on(prefix, cal, sim.device).numpy(), .05)
        report['frozen_r5'] = {'tau': tau, 'dev': dev_report(prefix, data['dev'], tau, sim.device)}
        oldtau = load_json(args.calibration_path)['tau']['orbisim']
        report['frozen_h10'] = {'tau': oldtau, 'dev': dev_report(anchor, data['dev'], oldtau, sim.device)}
    report['complete'] = True
    atomic_json(report, output / 'summary.json')
    print('[R5 train complete]', cli.seed, flush=True)


def evaluate(cli, args, sim, od, root, source):
    output = root / f'evaluation/shard_{cli.shard}'
    output.mkdir(parents=True, exist_ok=False)
    reports = {seed: load_json(root / f'training/seed_{seed}/summary.json') for seed in ((0,) if cli.smoke else SEEDS)}
    assert all(r['complete'] for r in reports.values())
    old = load_json(source / 'evaluation.json')
    policy = load_policy(args, od, sim.device, sim)
    assert policy.H == 10
    anchor = load_predictor(args, od, 'orbisim', sim.device, sim)
    gates = {'frozen_h10': (anchor, load_json(args.calibration_path)['tau']['orbisim']),
             'frozen_r5': (PrefixRisk(anchor), reports[0]['frozen_r5']['tau'])}
    for seed, report in reports.items():
        for kind, meta in report['models'].items():
            assert digest(meta['path']) == meta['sha256']
            ck = torch.load(meta['path'], map_location='cpu', weights_only=True)
            model = ImmediateGate(ck['objective'], ck['hidden'], ck['E']).to(sim.device).eval()
            model.load_state_dict(ck['state'])
            gates[f'{kind}_s{seed}'] = (model, meta['tau'])
    arms = ('none', *gates)
    indices = [0] if cli.smoke else list(range(cli.shard, 20, 3))
    report = {'complete': False, 'shard': cli.shard, 'arms': arms, 'per_cell': [],
              'training_summary_hashes': {str(s): digest(root / f'training/seed_{s}/summary.json') for s in reports}}
    for ci in indices:
        oldrow = old['per_cell'][ci]
        env, params = env_for_cell(args, args.n_envs, sim.device, oldrow['cell'])
        seed = 98400 + ci if cli.smoke else oldrow['episode_seed']
        assert cli.smoke or seed == 26000 + ci
        row = {'grid_index': ci, 'cell': oldrow['cell'], 'ood': oldrow['ood'],
               'episode_seed': seed, 'episodes': {}, 'raw': {}}
        for name in arms:
            verifier = None
            if name != 'none':
                gate, tau = gates[name]
                verifier = GateRepairVerifier(CountRisk(gate), CountRisk(anchor), tau, sim.max_vel)
            ctrl = Controller(policy, sim, 'checkvla' if verifier else 'none', verifier=verifier, chunk_k=5)
            result = run_episodes(env, sim, ctrl, params, seed=seed, record=True)
            assert ctrl.n_policy_calls == 16 and ctrl.n_policy_rows == args.n_envs * 16
            if ctrl.needs_rgb:
                assert ctrl.n_render_calls == 16 and ctrl.n_render_rows == args.n_envs * 16
            if verifier:
                assert verifier.gate.rows == args.n_envs * 16
                assert verifier.predictor.rows == 8 * int(np.asarray(result['n_interventions']).sum())
            if not cli.smoke and name in ('none', 'frozen_h10'):
                verify_replay(result, oldrow['episodes']['none' if name == 'none' else 'frozen'])
                # Stronger live-risk replay against the completed factorial.
                fact = load_json(source.parent / 'risk_controller_diagnosis_20261012_a/summary.json')
                raw = fact['per_cell'][ci]['raw']['none' if name == 'none' else 'frozen_ll']
                assert digest(raw['path']) == raw['sha256']
                with np.load(raw['path']) as z:
                    np.testing.assert_allclose(result['trace']['risk'], z['risk'], rtol=1e-5, atol=1e-6)
            row[name] = summarize(result)
            row['episodes'][name] = {k: result[k] for k in ('success', 'violation', 'first_viol', 'first_trig',
                'first_success', 'n_interventions', 'final_task_cost', 'stall_decisions', 'n_decisions')}
            path = output / f'cell_{ci:02d}_{name}.npz'
            with open(str(path) + '.tmp', 'wb') as f:
                np.savez_compressed(f, **row['episodes'][name], **result['trace'])
            os.replace(str(path) + '.tmp', path)
            row['raw'][name] = {'path': str(path), 'sha256': digest(path),
                'n_policy_calls': ctrl.n_policy_calls, 'n_policy_rows': ctrl.n_policy_rows,
                'gate_rows': verifier.gate.rows if verifier else 0,
                'repair_rows': verifier.predictor.rows if verifier else 0}
            atomic_json(row, output / f'cell_{ci:02d}.partial.json')
            print('[R5 eval]', ci, name, {k: row[name][k] for k in ('SR', 'CVR', 'safe_success')}, flush=True)
        report['per_cell'].append(row)
        atomic_json(report, output / 'summary.partial.json')
    report['complete'] = True
    atomic_json(report, output / 'summary.json')


def aggregate(root, smoke):
    summaries = [load_json(root / f'evaluation/shard_{i}/summary.json') for i in range(1 if smoke else 3)]
    assert all(s['complete'] for s in summaries)
    assert all(s['arms'] == summaries[0]['arms'] and s['training_summary_hashes'] == summaries[0]['training_summary_hashes'] for s in summaries)
    rows = sorted([r for s in summaries for r in s['per_cell']], key=lambda r: r['grid_index'])
    assert [r['grid_index'] for r in rows] == list(range(1 if smoke else 20))
    verified = 0
    for r in rows:
        for name, meta in r['raw'].items():
            assert digest(meta['path']) == meta['sha256']
            with np.load(meta['path']) as d:
                for k, v in r['episodes'][name].items():
                    np.testing.assert_array_equal(d[k], v)
                risk = d['risk'].max(-1)
                np.testing.assert_array_equal(d['violation'], risk.max(1) > 1)
                first = np.where((risk > 1).any(1), (risk > 1).argmax(1), -1)
                np.testing.assert_array_equal(d['first_viol'], first)
            verified += 1
    arms, seeds = summaries[0]['arms'], (0,) if smoke else SEEDS
    result = {'complete': True, 'smoke': smoke, 'per_cell': rows, 'verified_raw_hashes': verified,
              'metadata_sha256': digest(root / 'metadata.json'), 'support_sha256': digest(root / 'support.json'),
              'training_summary_hashes': summaries[0]['training_summary_hashes'],
              'paired_vs_none': {a: pooled_paired(rows, a, 3400) for a in arms if a != 'none'},
              'event_vs_regression': {}, 'event_vs_frozen': {}, 'pooled': {}}
    for domain in ('ood', 'id'):
        selected = [r for r in rows if r['ood'] == (domain == 'ood')]
        result['pooled'][domain] = {a: {k: float(np.mean([r[a][k] for r in selected]))
                    for k in ('SR', 'CVR', 'safe_success')} for a in arms} if selected else {}
    for seed in seeds:
        arm = f'event_s{seed}'
        for base, key in ((f'regression_s{seed}', 'event_vs_regression'), ('frozen_h10', 'event_vs_frozen')):
            entries = [{'ood': r['ood'], 'episodes': {**r['episodes'], 'none': r['episodes'][base]}} for r in rows]
            result[key][str(seed)] = pooled_paired(entries, arm, 3500)
    result['decision'] = {
        'all_seed_event_vs_regression_positive_point_delta': not smoke and all(v['delta_safe_success'] > 0 for v in result['event_vs_regression'].values()),
        'all_seed_event_vs_regression_positive_CI_and_SR_loss_le_2pp': not smoke and all(
            v['paired_intervals']['safe_success']['ci95'][0] > 0 and v['delta_SR'] >= -.02
            for v in result['event_vs_regression'].values()),
        'all_seed_policy_relative_candidate_screen': not smoke and all(
            result['paired_vs_none'][f'event_s{s}']['delta_safe_success'] >= .03
            and result['paired_vs_none'][f'event_s{s}']['paired_intervals']['safe_success']['ci95'][0] > 0
            and result['paired_vs_none'][f'event_s{s}']['delta_SR'] >= -.02 for s in seeds),
        'independent_confirmation_required': True, 'H10_anticipation_learned': False}
    result['stress_review'] = {f'event_s{s}': [r['grid_index'] for r in rows if r['ood'] and (
        r[f'event_s{s}']['safe_success'] < r['none']['safe_success'] or r[f'event_s{s}']['SR'] < r['none']['SR'] - .02)] for s in seeds}
    atomic_json(result, root / 'summary.json')
    print('[R5 complete]', result['pooled'], result['decision'], flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source_run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=('init', 'collect', 'screen', 'train', 'evaluate', 'aggregate'), required=True)
    p.add_argument('--split', choices=BASES)
    p.add_argument('--seed', type=int, choices=SEEDS, default=0)
    p.add_argument('--shard', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    source, root = Path(cli.source_run).resolve(), Path(cli.output).resolve()
    status = load_json(source / 'status.json')
    assert status['complete'] and not status['smoke']
    for meta in status['source_artifacts'].values():
        assert digest(meta['path']) == meta['sha256']
    args = SimpleNamespace(**status['effective_policy_args'])
    assert args.policy == 'openvla' and args.n_envs == 32 and args.chunk_k == 5 and args.vla_batch_size == 8
    args.device = cli.device
    if cli.smoke:
        args.policy, args.n_envs = 'bc', 4
    if cli.stage == 'init':
        root.mkdir(parents=True, exist_ok=False)
        atomic_json({'smoke': cli.smoke, 'args': vars(args), 'source_artifacts': status['source_artifacts'],
            'source_status_sha256': digest(source / 'status.json'), 'source_evaluation_sha256': digest(source / 'evaluation.json'),
            'protocol_sha256': digest(PROTOCOL), 'code_hashes': {str(f): digest(f) for f in CODE},
            'split_bases': BASES, 'batch_counts': BATCHES, 'collectors': COLLECTORS,
            'minimum_support': MINIMUM, 'training_seeds': SEEDS,
            'scope': 'R5 objectives only; original repair frozen; no H10 learning claim'}, root / 'metadata.json')
        return
    meta = load_json(root / 'metadata.json')
    assert meta['smoke'] == cli.smoke and meta['protocol_sha256'] == digest(PROTOCOL)
    assert meta['source_status_sha256'] == digest(source / 'status.json')
    assert meta['source_evaluation_sha256'] == digest(source / 'evaluation.json')
    for f, sha in meta['code_hashes'].items():
        assert digest(f) == sha, f'code changed during run: {f}'
    if cli.stage == 'screen':
        raise SystemExit(0 if screen(root, cli.smoke) else 2)
    if cli.stage == 'aggregate':
        aggregate(root, cli.smoke)
        return
    _, sim, od = setup(args)
    if cli.stage == 'collect':
        assert cli.split is not None
        collect(cli, args, sim, od, root)
    elif cli.stage == 'train':
        train(cli, args, sim, od, root)
    else:
        evaluate(cli, args, sim, od, root, source)


if __name__ == '__main__':
    main()
