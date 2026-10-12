"""Matched-data temporal representation experiment, isolated from legacy files.

Collect fresh training/development/calibration episodes, train static and GRU
risk ensembles, then evaluate all seeds on the existing validation grid. No
confirmation or final-test episodes are used. All labels are offline. Runtime
controllers retain frozen bisection semantics, H10 and five-step commitment.
"""
import argparse
import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace

from _common import env_for_cell, load_policy, load_predictor, setup
from _audit import SCALES, verify_replay
from _paired import pooled_paired
from audit_verifier import atomic_json
from checkvla.reference import RefCheckVLA, calibrate_tau
from checkvla.verifier import Controller, run_episodes, summarize
from common import PREDICTOR_TRAIN_RANGES, grid_cells, is_ood, load_json, seed_all
from models.sequence_risk import SequenceRisk, RiskWithFrozenDynamics, matched_static_width
import numpy as np
import torch


SPLIT_BASES = {'train': 71000, 'dev': 72000, 'calibration': 73000}
CONTEXT_KEYS = ('obs', 'obs_prev', 'a_prev')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_torch(data, path):
    torch.save(data, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def controller(policy, sim, predictor=None, tau=None):
    if predictor is None:
        return Controller(policy, sim, 'none', chunk_k=5)
    verifier = RefCheckVLA(predictor, tau=tau, beta=1, commit=5,
                          max_vel=sim.max_vel, hard_prefix=False, use_grad=False,
                          abstain=True, repair_mode='bisection', execution_horizon=5)
    return Controller(policy, sim, 'checkvla', verifier=verifier, chunk_k=5)


@torch.no_grad()
def candidate_labels(sim, state, params, proposal):
    """Vectorize the same eight scales; no labels enter the live controller."""
    B, H, A = proposal.shape
    C = len(SCALES)
    actions = proposal[:, None] * proposal.new_tensor(SCALES)[None, :, None, None]
    s = state[:, None].expand(-1, C, -1).reshape(B * C, -1).clone()
    p = {k: v[:, None].expand(-1, C).reshape(-1) for k, v in params.items()}
    risks = []
    for j in range(H):
        s, risk = sim.step(s, actions[:, :, j].reshape(B * C, A), p)
        risks.append(risk.reshape(B, C, 2))
    return actions, torch.stack(risks, 2)


@torch.no_grad()
def collect_split(split, batches, args, sim, policy, anchor, tau, dev, output):
    ctrl = controller(policy, sim, anchor, tau)
    plain = controller(policy, sim)
    keys = (*CONTEXT_KEYS, 'actions', 'risk', 'episode_id', 'decision_t', 'collector')
    all_batches = {key: [] for key in keys}
    meta = []
    for batch in range(batches):
        seed = SPLIT_BASES[split] + batch
        env, _ = env_for_cell(args, args.n_envs, dev, {})
        gen = torch.Generator().manual_seed(seed + 1000000)
        params = sim.sample_params(args.n_envs, PREDICTOR_TRAIN_RANGES['push'], gen)
        # Training/development contain both policy and repaired reachable states.
        # Calibration is only uncorrected policy decisions, without action noise.
        c = plain if split == 'calibration' or batch % 2 == 0 else ctrl
        obs = env.reset(params, seed)
        c.reset(env, obs)
        rows = {key: [] for key in keys}
        for t in range(sim.T):
            if t % 5 == 0:
                state = env.get_state().clone()
                ctx = {'obs': obs.clone(), 'obs_prev': c.obs_prev.clone(), 'a_prev': c.a_prev.clone()}
            action, info = c.act(env, obs)
            if t % 5 == 0:
                candidates, risk = candidate_labels(sim, state, params, info['chunk'])
                if not torch.equal(env.get_state(), state):
                    raise RuntimeError('labels mutated live state')
                for key in CONTEXT_KEYS:
                    rows[key].append(ctx[key].cpu())
                rows['actions'].append(candidates.cpu())
                rows['risk'].append(risk.cpu())
                rows['episode_id'].append(torch.arange(args.n_envs) + seed * args.n_envs)
                rows['decision_t'].append(torch.full((args.n_envs,), t))
                rows['collector'].append(torch.full((args.n_envs,), int(c is ctrl)))
            obs, _, _ = env.step(action)
        if c.n_policy_calls != 16 or c.n_policy_rows != 16 * args.n_envs:
            raise RuntimeError('collection changed policy cadence')
        data = {key: torch.cat(values) for key, values in rows.items()}
        path = output / f'{split}_batch_{batch:02d}.pt'
        atomic_torch(data, path)
        for key in keys:
            all_batches[key].append(data[key])
        meta.append({'seed': seed, 'collector': c.mode, 'path': str(path), 'sha256': digest(path),
                     'physical_multiplier_ranges': PREDICTOR_TRAIN_RANGES['push']})
        print(f'[collect] {split} {batch+1}/{batches} decisions={len(data["obs"])} '
              f'unsafe_candidates={(data["risk"].amax((-1,-2)) > 1).float().mean():.4f}', flush=True)
        atomic_json({'complete': batch + 1 == batches, 'batches': meta}, output / f'{split}_manifest.json')
    data = {key: torch.cat(values) for key, values in all_batches.items()}
    data['split'] = split
    return data


def assert_split_disjoint(datasets):
    sets = [set(d['episode_id'].tolist()) for d in datasets]
    for i, ids in enumerate(sets):
        for other in sets[:i]:
            if ids & other:
                raise ValueError('episode overlap across splits')


def gather(data, decisions, candidates, dev):
    ctx = {key: data[key][decisions].to(dev) for key in CONTEXT_KEYS}
    return ctx, data['actions'][decisions, candidates].to(dev), data['risk'][decisions, candidates].to(dev)


@torch.no_grad()
def predict_dataset(model, data, dev):
    scores, means = [], []
    C = data['actions'].shape[1]
    for start in range(0, len(data['obs']), 128):
        stop = min(start + 128, len(data['obs']))
        ids = torch.arange(start, stop).repeat_interleave(C)
        candidates = torch.arange(C).repeat(stop - start)
        ctx, actions, _ = gather(data, ids, candidates, dev)
        mean, std = model.risk(ctx, actions)
        scores.append((mean + std).amax(-1).cpu().reshape(stop-start, C, 10))
        means.append(mean.cpu().reshape(stop-start, C, 10, 2))
    return torch.cat(scores), torch.cat(means)


def diagnostic_metrics(scores, mean, data, tau):
    true = data['risk'].amax(-1)
    result = {'per_step_MAE': float((mean - data['risk']).abs().mean()), 'horizons': {}}
    # Same full-H trigger in both diagnostic horizon scopes, as at deployment.
    triggers = scores[:, 0].amax(-1) > tau
    for name, h in (('prefix', 5), ('lookahead', 10)):
        gt, pred = true[..., :h].amax(-1), scores[..., :h].amax(-1)
        unsafe = gt[:, 0] > 1
        inv = comparable = 0
        for i in range(len(SCALES)):
            for j in range(i + 1, len(SCALES)):
                dg, dp = gt[:, i] - gt[:, j], pred[:, i] - pred[:, j]
                ok = (dg.abs() > 1e-4) & (dp.abs() > 1e-4)
                inv += int((ok & (dg * dp < 0)).sum()); comparable += int(ok.sum())
        result['horizons'][name] = {
            'unsafe_proposals': int(unsafe.sum()), 'true_trigger': int((unsafe & triggers).sum()),
            'missed': int((unsafe & ~triggers).sum()),
            'safe_proposal_trigger': int((~unsafe & triggers).sum()),
            'safe_proposals': int((~unsafe).sum()),
            'recall': float(triggers[unsafe].float().mean()) if unsafe.any() else None,
            'safe_proposal_trigger_rate': float(triggers[~unsafe].float().mean()) if (~unsafe).any() else None,
            'ranking_inversions': inv, 'ranking_comparable_pairs': comparable,
            'ranking_inversion_rate': inv / max(comparable, 1)}
    result['scope'] = 'all collected decisions, including post-violation and seated; descriptive correlated samples'
    return result


@torch.no_grad()
def audit_validation(learned, taus, anchor, anchor_tau, audit_path, source_path, dev, output):
    """Identical held-out states/candidates for every head, before rollout drift.

    Source audit decisions are ONLY evaluated, never used for fitting or tau.
    Primary fixed trajectory: none. Frozen bisection is a secondary view.
    """
    audit = load_json(audit_path)
    if not audit['complete'] or audit['metadata']['source_result_sha256'] != digest(source_path):
        raise ValueError('offline audit must be complete and match validation source')
    predictions = {'frozen': anchor, **learned}
    thresholds = {'frozen': anchor_tau, **taus}
    report = {'complete': False, 'audit_sha256': digest(audit_path), 'per_cell': [],
              'scope': 'validation-only, pre-first-violation decisions; no fitting or recalibration'}
    pools = {}
    for cell in audit['per_cell']:
        for trajectory in ('none', 'checkvla_orbisim'):
            entry = cell['arms'][trajectory]
            if not entry['replay_matches_source'] or not entry['executed_prefix_matches_offline_labels']:
                raise ValueError('invalid audit replay')
            with np.load(entry['raw_path'], allow_pickle=False) as z:
                scope = ((z['first_viol'][None] < 0) | (z['decision_t'][:, None] <= z['first_viol'][None]))
                data = {key: torch.from_numpy(z[key][scope]) for key in CONTEXT_KEYS}
                proposal = torch.from_numpy(z['proposal'][scope])
                data['actions'] = proposal[:, None] * proposal.new_tensor(SCALES)[None, :, None, None]
                true = torch.from_numpy(z['candidate_gt_risk'][scope])
                # Push's second risk channel is identically zero, as in the source simulator.
                data['risk'] = torch.stack([true, torch.zeros_like(true)], -1)
            row = {'grid_index': cell['grid_index'], 'cell': cell['cell'], 'ood': cell['ood'],
                   'trajectory': trajectory, 'raw_sha256': digest(entry['raw_path']), 'models': {}}
            for name, model in predictions.items():
                scores, mean = predict_dataset(model, data, dev)
                metrics = diagnostic_metrics(scores, mean, data, thresholds[name])
                metrics['scope'] = 'pre-first-violation, all stages, same saved trajectory across models'
                row['models'][name] = metrics
                if cell['ood']:
                    target = pools.setdefault(trajectory, {}).setdefault(name, {})
                    for h, values in metrics['horizons'].items():
                        agg = target.setdefault(h, {})
                        for k, v in values.items():
                            if type(v) is int:
                                agg[k] = agg.get(k, 0) + v
            report['per_cell'].append(row)
        atomic_json(report, output / 'offline_validation.partial.json')
        print(f'[offline-validation] cell={cell["grid_index"]} same-state predictions saved', flush=True)
    for models in pools.values():
        for hs in models.values():
            for values in hs.values():
                values['recall'] = values['true_trigger'] / max(values['unsafe_proposals'], 1)
                values['safe_proposal_trigger_rate'] = values['safe_proposal_trigger'] / max(values['safe_proposals'], 1)
                values['ranking_inversion_rate'] = values['ranking_inversions'] / max(values['ranking_comparable_pairs'], 1)
    report.update(complete=True, pooled_ood=pools)
    atomic_json(report, output / 'offline_validation.json')


def train_pair(train, dev_data, calibration, cli, dev, output, anchor):
    learned, thresholds, reports = {}, {}, {}
    if any(d['split'] != name for d, name in ((train, 'train'), (dev_data, 'dev'), (calibration, 'calibration'))):
        raise ValueError('incorrect data roles')
    assert_split_disjoint([train, dev_data, calibration])
    # All candidates from a decision remain together in the split. Both models
    # consume identical batches and loss targets, with independent initializations.
    hot = torch.nonzero(train['risk'].amax((1, 2, 3)) > 0.8).flatten()
    for seed in cli.train_seeds:
        models, optimizers = {}, {}
        training_seconds = {'static': 0.0, 'sequence': 0.0}
        for kind in ('static', 'sequence'):
            seed_all(seed)
            width = matched_static_width(cli.hidden) if kind == 'static' else cli.hidden
            model = SequenceRisk(kind, width, E=5).to(dev)
            model.fit_normalizers(train)
            models[kind] = model
            optimizers[kind] = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        pc = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
        if abs(pc['static'] / pc['sequence'] - 1) > 0.05:
            raise RuntimeError('parameter budget mismatch exceeds 5%')
        gen = torch.Generator().manual_seed(seed + 81000)
        for step in range(cli.iters):
            ids = torch.randint(len(train['obs']), (cli.batch_size,), generator=gen)
            nh = int(cli.batch_size * 0.3) if len(hot) else 0
            if nh:
                ids[:nh] = hot[torch.randint(len(hot), (nh,), generator=gen)]
            cs = torch.randint(len(SCALES), (cli.batch_size,), generator=gen)
            ctx, actions, target = gather(train, ids, cs, dev)
            losses = {}
            for kind, model in models.items():
                started = time.perf_counter()
                model.train()
                optimizer = optimizers[kind]
                lr = 1e-3 * (0.5 + 0.5 * np.cos(np.pi * step / cli.iters))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                predicted = model(ctx, actions)
                loss = (predicted - target.clamp(max=3)[None]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'nonfinite training loss: {kind}')
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step(); losses[kind] = float(loss.detach())
                training_seconds[kind] += time.perf_counter() - started
            if step % 500 == 0 or step + 1 == cli.iters:
                print(f'[train] seed={seed} step={step+1}/{cli.iters} loss={losses}', flush=True)
        # Fixed final checkpoint: no selecting a lucky training seed or dev point.
        for kind, model in models.items():
            model.eval()
            name = f'{kind}_s{seed}'
            path = output / f'{name}.pt'
            atomic_torch({'state': model.state_dict(), 'kind': kind, 'hidden': model.hidden,
                          'E': model.E, 'seed': seed, 'iterations': cli.iters,
                          'train_only_normalizers': True}, path)
            scores, _ = predict_dataset(model, calibration, dev)
            safe = calibration['risk'][:, 0].amax((1, 2)) <= 1
            if not safe.any() or not torch.isfinite(scores).all():
                raise RuntimeError('invalid calibration scores')
            tau = calibrate_tau(scores[safe, 0].amax(-1).numpy(), 0.05)
            ds, dm = predict_dataset(model, dev_data, dev)
            reports[name] = {'parameters': pc[kind], 'checkpoint': str(path), 'sha256': digest(path),
                             'training_seconds': training_seconds[kind],
                             'tau': tau, 'n_safe_calibration_decisions': int(safe.sum()),
                             'dev': diagnostic_metrics(ds, dm, dev_data, tau)}
            thresholds[name] = tau
            learned[name] = RiskWithFrozenDynamics(model, anchor)
            atomic_json({'complete': False, 'models': reports}, output / 'training.partial.json')
            print(f'[model] {name} params={pc[kind]} tau={tau:.6f} dev={reports[name]["dev"]}', flush=True)
    atomic_json({'complete': True, 'models': reports,
                 'calibration_scope': 'in-range safe policy-decision H10 chunks; no OOD guarantee'},
                output / 'training.json')
    return learned, thresholds, reports


def evaluate(args, sim, policy, anchor, anchor_tau, learned, taus, reports, source, cli, dev, output):
    controllers = {'none': controller(policy, sim), 'frozen': controller(policy, sim, anchor, anchor_tau)}
    controllers.update({k: controller(policy, sim, m, taus[k]) for k, m in learned.items()})
    cells = grid_cells('push')
    indices = [0] if cli.smoke else list(range(len(cells)))
    summary = {'complete': False, 'arms': list(controllers), 'per_cell': [],
               'metadata': {'split': 'implementation_smoke' if cli.smoke else 'validation',
                            'execution_horizon': 5, 'proposal_horizon': 10,
                            'evaluated_cell_indices': indices, 'models': reports,
                            'source_result_sha256': digest(cli.source_result),
                            'repair': 'frozen bisection, gradient/hard-prefix off',
                            'episode_seed_base': 99000 if cli.smoke else 26000,
                            'no_new_method_claim_before_final_test': True}}
    for ci in indices:
        env, params = env_for_cell(args, args.n_envs, dev, cells[ci])
        entry = {'cell': cells[ci], 'ood': is_ood('push', cells[ci]), 'episodes': {},
                 'episode_seed': (99000 if cli.smoke else 26000) + ci}
        for name, ctrl in controllers.items():
            res = run_episodes(env, sim, ctrl, params, seed=entry['episode_seed'])
            if ctrl.n_policy_calls != 16 or ctrl.n_policy_rows != args.n_envs * 16:
                raise RuntimeError('unmatched evaluation cadence')
            if not cli.smoke and name in ('none', 'frozen'):
                expected = source['per_cell'][source['metadata']['evaluated_cell_indices'].index(ci)]
                key = 'none' if name == 'none' else 'checkvla_orbisim'
                verify_replay(res, expected['episodes'][key])
            entry[name] = summarize(res)
            entry['episodes'][name] = {k: res[k] for k in (
                'success', 'violation', 'first_viol', 'first_trig', 'first_success',
                'n_interventions', 'final_task_cost', 'stall_decisions', 'n_decisions')}
            # Preserve each arm before the next expensive VLA rollout.
            atomic_json(entry, output / f'eval_cell_{ci:02d}.partial.json')
            print(f'[eval] cell={ci} arm={name} SR={entry[name]["SR"]:.4f} '
                  f'CVR={entry[name]["CVR"]:.4f} safe={entry[name]["safe_success"]:.4f}', flush=True)
        summary['per_cell'].append(entry)
        atomic_json(summary, output / 'evaluation.partial.json')
    summary['paired_vs_none'] = {name: pooled_paired(summary['per_cell'], name, 1700)
                                 for name in controllers if name != 'none'}
    # Direct paired architecture contrast for each training seed, not only baseline deltas.
    summary['sequence_vs_static'] = {}
    for seed in cli.train_seeds:
        pair_entries = [{'ood': e['ood'], 'episodes': {'none': e['episodes'][f'static_s{seed}'],
                                                      'sequence': e['episodes'][f'sequence_s{seed}']}}
                        for e in summary['per_cell']]
        summary['sequence_vs_static'][str(seed)] = pooled_paired(pair_entries, 'sequence', 1700)
    ood = [e for e in summary['per_cell'] if e['ood']] or summary['per_cell']
    summary['pooled_ood'] = {name: {metric: float(np.mean([e[name][metric] for e in ood]))
                                    for metric in ('SR', 'CVR', 'safe_success')}
                           for name in controllers}
    summary['complete'] = True
    atomic_json(summary, output / 'evaluation.json')
    print('[eval] complete', summary['pooled_ood'], flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source_result', required=True)
    p.add_argument('--audit_summary')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--train_batches', type=int, default=12)
    p.add_argument('--dev_batches', type=int, default=4)
    p.add_argument('--calibration_batches', type=int, default=4)
    p.add_argument('--train_seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    if not cli.smoke and not cli.audit_summary:
        raise ValueError('full temporal ablation requires the frozen same-state validation audit')
    source = load_json(cli.source_result)
    if not source['complete'] or source['metadata']['args']['split'] != 'validation':
        raise ValueError('requires completed validation provenance')
    args = SimpleNamespace(**source['metadata']['args'])
    if args.task != 'push' or args.backend != 'torch' or args.chunk_k != 5 or args.vla_batch_size != 8:
        raise ValueError('requires original push H10/k5/microbatch8 protocol')
    for name, artifact in source['metadata']['artifacts'].items():
        if digest(artifact['path']) != artifact['sha256']:
            raise RuntimeError(f'frozen artifact changed: {name}')
    if cli.smoke:
        cli.train_batches = cli.dev_batches = cli.calibration_batches = 1
        cli.train_seeds = [0]; cli.hidden = 16; cli.iters = 3; cli.batch_size = 16
        args.policy = 'bc'; args.n_envs = 4
    if any(n < 1 or n >= 1000 for n in (cli.train_batches, cli.dev_batches, cli.calibration_batches)):
        raise ValueError('split batches must fit nonoverlapping reserved seed blocks')
    if cli.iters < 1 or not cli.train_seeds or len(set(cli.train_seeds)) != len(cli.train_seeds):
        raise ValueError('invalid training configuration')
    output = Path(cli.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.device = cli.device
    dev, sim, od = setup(args)
    metadata = {'complete': False, 'stage': 'collection', 'smoke': cli.smoke,
                'config': vars(cli), 'effective_policy_args': vars(args),
                'source_result_sha256': digest(cli.source_result), 'source_artifacts': source['metadata']['artifacts'],
                'new_code_hashes': {f.name: digest(f) for f in (
                    Path(__file__), Path(__file__).parents[1] / 'models/sequence_risk.py')},
                'split_seed_bases': SPLIT_BASES, 'candidate_scales': SCALES,
                'physics_training_ranges': PREDICTOR_TRAIN_RANGES['push'],
                'loss': 'same per-step risk MSE, targets clipped at 3, 30% hot decisions',
                'normalization': 'training decisions only, shared across heads',
                'selection': 'fixed final checkpoint; all three training seeds reported; no test tuning'}
    atomic_json(metadata, output / 'status.json')
    policy = load_policy(args, od, dev, sim)
    if policy.H != 10:
        raise ValueError('requires H10 policy')
    anchor = load_predictor(args, od, 'orbisim', dev, sim)
    tau = load_json(args.calibration_path)['tau']['orbisim']
    datasets = {}
    for split, batches in (('train', cli.train_batches), ('dev', cli.dev_batches),
                           ('calibration', cli.calibration_batches)):
        datasets[split] = collect_split(split, batches, args, sim, policy, anchor, tau, dev, output)
    assert_split_disjoint(list(datasets.values()))
    metadata['stage'] = 'training'; atomic_json(metadata, output / 'status.json')
    learned, taus, reports = train_pair(datasets['train'], datasets['dev'], datasets['calibration'],
                                       cli, dev, output, anchor)
    if not cli.smoke:
        metadata['stage'] = 'offline_validation'; atomic_json(metadata, output / 'status.json')
        audit_validation(learned, taus, anchor, tau, cli.audit_summary, cli.source_result, dev, output)
    metadata['stage'] = 'validation'; atomic_json(metadata, output / 'status.json')
    evaluate(args, sim, policy, anchor, tau, learned, taus, reports, source, cli, dev, output)
    metadata.update(stage='complete', complete=True)
    atomic_json(metadata, output / 'status.json')
    print('[sequence-risk] complete', output, flush=True)


if __name__ == '__main__':
    main()
