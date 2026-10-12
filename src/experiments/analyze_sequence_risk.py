"""Post-run diagnosis, without fitting, threshold selection, or policy rollouts.

The four gate/repair combinations are same-state, open-loop counterfactuals on
the saved policy-alone trajectory. They do NOT estimate closed-loop efficacy.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from _common import load_predictor
from _audit import label_chunk, verify_replay
from audit_verifier import atomic_json
from common import load_json
from checkvla.reference import chunk_score, scale_repair
from models.sequence_risk import SequenceRisk
from sims.torch_push import PushSim
from validate_sequence_risk import (CONTEXT_KEYS, SCALES, assert_split_disjoint,
                                    digest, predict_dataset)
import numpy as np
import torch


class ExactRisk:
    """Privileged diagnostic only: homogeneous cell, exact state and parameters."""
    needs_img = False
    differentiable = False

    def __init__(self, sim, cell):
        self.sim, self.cell = sim, cell

    @torch.no_grad()
    def risk(self, ctx, actions):
        s = ctx['state'].clone()
        params = self.sim.make_params(len(s), self.cell)
        risks = []
        for a in actions.unbind(1):
            s, r = self.sim.step(s, a, params)
            risks.append(r)
        mean = torch.stack(risks, 1)
        return mean, torch.zeros_like(mean)


def dataset_counts(data):
    r = data['risk'].amax(-1)
    unsafe = r.amax(-1) > 1
    hot = r.amax((1, 2)) > 0.8
    any_unsafe = unsafe.any(-1)
    return {
        'episodes': int(data['episode_id'].unique().numel()),
        'decisions': len(r), 'candidates': int(unsafe.numel()),
        'unsafe_candidates': int(unsafe.sum()),
        'unsafe_original_H10': int(unsafe[:, 0].sum()),
        'unsafe_original_R5': int((r[:, 0, :5].amax(-1) > 1).sum()),
        'decisions_with_unsafe_candidate': int(any_unsafe.sum()),
        'episodes_with_unsafe_candidate': int(data['episode_id'][any_unsafe].unique().numel()),
        'hot_decisions': int(hot.sum()),
        'hot_episodes': int(data['episode_id'][hot].unique().numel()),
        'unsafe_steps': int((r > 1).sum()), 'candidate_steps': int(r.numel()),
        'risk_peak_quantiles': torch.quantile(r.amax(-1).flatten(),
                                             torch.tensor([0., .5, .9, .95, .99, 1.])).tolist(),
    }


def pair_counts(gt, pred):
    """Fixed GT denominator; distinguish prediction ties from correct ordering."""
    out = dict(comparable=0, inverted=0, tied=0, boundary_pairs=0,
               boundary_inverted=0, boundary_tied=0)
    for i in range(gt.shape[1]):
        for j in range(i + 1, gt.shape[1]):
            dg, dp = gt[:, i] - gt[:, j], pred[:, i] - pred[:, j]
            valid = dg.abs() > 1e-4
            boundary = valid & ((gt[:, i] > 1) != (gt[:, j] > 1))
            tied = dp.abs() <= 1e-4
            inverted = ~tied & (dg * dp < 0)
            for prefix, scope in (('', valid), ('boundary_', boundary)):
                out['boundary_pairs' if prefix else 'comparable'] += int(scope.sum())
                out[prefix + 'inverted'] += int((scope & inverted).sum())
                out[prefix + 'tied'] += int((scope & tied).sum())
    return out


def local_counts(trigger, proposal, applied, original_gt, applied_gt):
    changed = (applied[:, :5] - proposal[:, :5]).abs().amax((1, 2)) > 1e-7
    counts = {'decisions': len(proposal), 'triggers': int(trigger.sum()),
              'execution_changes': int((trigger & changed).sum())}
    for h in (5, 10):
        g, a = original_gt[:, :h].amax(-1), applied_gt[:, :h].amax(-1)
        counts.update({
            f'R{h}_unsafe_original': int((g > 1).sum()),
            f'R{h}_true_triggers': int((trigger & (g > 1)).sum()),
            f'R{h}_safe_triggers': int((trigger & (g <= 1)).sum()),
            f'R{h}_local_rescues': int((trigger & (g > 1) & (a <= 1)).sum()),
            f'R{h}_local_harms': int((trigger & (g <= 1) & (a > 1)).sum()),
            f'R{h}_risk_increase_gt_0_1': int((trigger & (a - g > .1)).sum()),
        })
    return counts


def add_counts(target, values):
    for k, v in values.items():
        target[k] = target.get(k, 0) + v


@torch.no_grad()
def diagnose(run, output, device='cpu'):
    status, training, evaluation, offline = [load_json(run / n) for n in
        ('status.json', 'training.json', 'evaluation.json', 'offline_validation.json')]
    if not all(d['complete'] for d in (status, training, evaluation, offline)) or status['smoke']:
        raise ValueError('requires completed production run')
    for name, sha in status['new_code_hashes'].items():
        path = (Path(__file__).parent / name if name == 'validate_sequence_risk.py'
                else Path(__file__).parents[1] / 'models' / name)
        if digest(path) != sha:
            raise RuntimeError(f'production source changed: {name}')
    source = load_json(status['config']['source_result'])
    if digest(status['config']['source_result']) != status['source_result_sha256']:
        raise RuntimeError('source result changed')
    for artifact in status['source_artifacts'].values():
        if digest(artifact['path']) != artifact['sha256']:
            raise RuntimeError('source artifact changed')
    audit_path = status['config']['audit_summary']
    if digest(audit_path) != offline['audit_sha256']:
        raise RuntimeError('audit changed')
    audit = load_json(audit_path)
    for e, ci in zip(evaluation['per_cell'], evaluation['metadata']['evaluated_cell_indices']):
        old = source['per_cell'][source['metadata']['evaluated_cell_indices'].index(ci)]
        for new_name, old_name in (('none', 'none'), ('frozen', 'checkvla_orbisim')):
            verify_replay(e['episodes'][new_name], old['episodes'][old_name])
        for name in evaluation['arms']:
            ep = e['episodes'][name]
            s, v = np.asarray(ep['success'], bool), np.asarray(ep['violation'], bool)
            for key, value in (('SR', s.mean()), ('CVR', v.mean()), ('safe_success', (s & ~v).mean())):
                if abs(e[name][key] - value) > 1e-9:
                    raise RuntimeError('saved aggregate/outcome mismatch')
            if e[name]['n_policy_calls'] != 16 or e[name]['n_policy_rows'] != 512:
                raise RuntimeError('cadence mismatch')
    data, composition = {}, {}
    for split in ('train', 'dev', 'calibration'):
        manifest = load_json(run / f'{split}_manifest.json')
        if not manifest['complete']:
            raise RuntimeError('incomplete collection')
        batches = []
        for b in manifest['batches']:
            if digest(b['path']) != b['sha256']:
                raise RuntimeError('dataset hash mismatch')
            batches.append(torch.load(b['path'], map_location='cpu', weights_only=True))
        d = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
        data[split] = d
        composition[split] = {'all': dataset_counts(d), 'collectors': {}}
        for kind in d['collector'].unique().tolist():
            composition[split]['collectors'][str(kind)] = dataset_counts(
                {k: v[d['collector'] == kind] for k, v in d.items()})
    assert_split_disjoint(list(data.values()))
    args = SimpleNamespace(**status['effective_policy_args'])
    sim = PushSim(device)
    models = {'frozen': load_predictor(args, str(run.parent), 'orbisim', device, sim)}
    taus = {'frozen': load_json(args.calibration_path)['tau']['orbisim']}
    for name, info in training['models'].items():
        if digest(info['checkpoint']) != info['sha256']:
            raise RuntimeError('checkpoint hash mismatch')
        ck = torch.load(info['checkpoint'], map_location='cpu', weights_only=True)
        m = SequenceRisk(ck['kind'], ck['hidden'], ck['E']).eval()
        m.load_state_dict(ck['state'])
        reference = SequenceRisk('static', 16, E=1)
        reference.fit_normalizers({'split': 'train', **data['train']})
        for key in ('obs_mean', 'obs_std', 'delta_std'):
            torch.testing.assert_close(getattr(m, key), getattr(reference, key), rtol=0, atol=0)
        models[name], taus[name] = m.to(device), info['tau']
    report = {'complete': False, 'scope': 'post-hoc validation diagnosis; no fitting or threshold search',
              'source_hashes': {n: digest(run / n) for n in
                               ('status.json', 'training.json', 'evaluation.json', 'offline_validation.json')},
              'analysis_code_sha256': digest(__file__), 'device': str(sim.device),
              'checks': 'source/artifact/data/model hashes, split isolation, train-only normalizers, baseline replay, cadence and outcome aggregates passed',
              'data_composition': composition, 'thresholds': taus, 'per_cell': [],
              'counterfactual_scope': 'policy-alone pre-first-violation OOD states; H10 risk, k5 local labels; GT gate uses physical limit 1, GT repair retains the same bisection/relative target/responsiveness rule; not closed-loop results',
              'pooled_local': {}, 'score_distributions': {}}
    score_pools = {name: [] for name in models}
    for cell in audit['per_cell']:
        if not cell['ood']:
            continue
        entry = cell['arms']['none']
        provenance = next(r for r in offline['per_cell'] if r['grid_index'] == cell['grid_index'] and r['trajectory'] == 'none')
        if digest(entry['raw_path']) != provenance['raw_sha256']:
            raise RuntimeError('raw audit changed')
        with np.load(entry['raw_path'], allow_pickle=False) as z:
            scope = (z['first_viol'][None] < 0) | (z['decision_t'][:, None] <= z['first_viol'][None])
            ctx = {k: torch.from_numpy(z[k][scope]).to(device) for k in (*CONTEXT_KEYS, 'state')}
            proposal = torch.from_numpy(z['proposal'][scope]).to(device)
            true = torch.from_numpy(z['candidate_gt_risk'][scope]).to(device)
        params = sim.make_params(len(proposal), cell['cell'])
        original_gt, _ = label_chunk(sim, ctx['state'], params, proposal, 5)
        torch.testing.assert_close(original_gt, true[:, 0], rtol=1e-5, atol=1e-6)
        exact = ExactRisk(sim, cell['cell'])
        gt_trigger = original_gt.amax(-1) > 1
        gt_repaired = scale_repair(exact, ctx, proposal, hard_prefix=False, use_grad=False, abstain=True)
        gt_repaired_risk, _ = label_chunk(sim, ctx['state'], params, gt_repaired, 5)
        row = {'grid_index': cell['grid_index'], 'cell': cell['cell'], 'models': {},
               'GT_gate_GT_repair': local_counts(gt_trigger, proposal, gt_repaired, original_gt, gt_repaired_risk)}
        add_counts(report['pooled_local'].setdefault('GT_gate_GT_repair', {}), row['GT_gate_GT_repair'])
        candidate_data = {**ctx, 'actions': proposal[:, None] * proposal.new_tensor(SCALES)[None, :, None, None],
                          'risk': torch.stack([true, torch.zeros_like(true)], -1)}
        for name, model in models.items():
            scores, _ = predict_dataset(model, candidate_data, device)
            scores = scores.to(device)
            s0 = scores[:, 0].amax(-1)
            trigger = s0 > taus[name]
            repaired = scale_repair(model, ctx, proposal, hard_prefix=False, use_grad=False, abstain=True)
            repaired_risk, _ = label_chunk(sim, ctx['state'], params, repaired, 5)
            applied_score = chunk_score(model, ctx, repaired)[0]
            target = torch.minimum(.75 * s0, s0.new_tensor(.8))
            changed = (repaired[:, :5] - proposal[:, :5]).abs().amax((1, 2)) > 1e-7
            floor = ((repaired - .05 * proposal).abs().amax((1, 2)) < 1e-7) & changed
            counts = local_counts(trigger, proposal, repaired, original_gt, repaired_risk)
            counts.update(changed_above_target=int((trigger & changed & (applied_score > target + 1e-6)).sum()),
                          changed_at_floor=int((trigger & floor).sum()))
            m = {'learned_gate_learned_repair': counts,
                 'GT_gate_learned_repair': local_counts(gt_trigger, proposal, repaired, original_gt, repaired_risk),
                 'learned_gate_GT_repair': local_counts(trigger, proposal, gt_repaired, original_gt, gt_repaired_risk),
                 'R5_ranking': pair_counts(true[..., :5].amax(-1), scores[..., :5].amax(-1)),
                 'R10_ranking': pair_counts(true.amax(-1), scores.amax(-1))}
            row['models'][name] = m
            target_pool = report['pooled_local'].setdefault(name, {})
            for kind, values in m.items():
                add_counts(target_pool.setdefault(kind, {}), values)
            score_pools[name].append(torch.stack([s0, original_gt[:, :5].amax(-1), original_gt.amax(-1)], -1))
        report['per_cell'].append(row)
        atomic_json(report, output / 'analysis.partial.json')
        print(f'[diagnosis] cell={cell["grid_index"]} local substitutions complete', flush=True)
    for name, rows in score_pools.items():
        values = torch.cat(rows)
        report['score_distributions'][name] = {}
        for h, col in ((5, 1), (10, 2)):
            for unsafe in (False, True):
                v = values[(values[:, col] > 1) == unsafe, 0]
                report['score_distributions'][name][f'R{h}_{"unsafe" if unsafe else "safe"}'] = {
                    'n': len(v), 'quantiles_0_25_50_75_100': torch.quantile(v, v.new_tensor([0., .25, .5, .75, 1.])).tolist()}
    report['complete'] = True
    atomic_json(report, output / 'analysis.json')
    print('[diagnosis] complete', output, flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu', help='match original simulator device; parity check is strict')
    cli = p.parse_args()
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    diagnose(Path(cli.run).resolve(), output, cli.device)


if __name__ == '__main__':
    main()
