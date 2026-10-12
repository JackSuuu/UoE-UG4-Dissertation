"""Privileged factorial gate/repair diagnosis; not a deployable method.

L/L replays a frozen model and threshold. G/L replaces only its gate with the
exact H10 event (risk > 1); L/G replaces only repair scoring with exact risk;
G/G replaces both. Repair search/target/abstention and H10/k5 stay unchanged.
GT has a physical threshold, not the learned score quantile. This distinction
is intentional and explicit. No training, calibration or parameter sweep.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from _common import env_for_cell, load_policy, load_predictor, setup
from _audit import verify_replay
from _paired import pooled_paired
from analyze_sequence_risk import ExactRisk
from audit_verifier import atomic_json
from checkvla.reference import RefCheckVLA, chunk_score
from checkvla.verifier import Controller, run_episodes, summarize
from common import load_json
from models.sequence_risk import SequenceRisk
from validate_sequence_risk import digest
import numpy as np
import torch


class GateRepairVerifier(RefCheckVLA):
    def __init__(self, gate, repair, tau, max_vel):
        super().__init__(repair, tau=tau, beta=1, commit=5, max_vel=max_vel,
                         hard_prefix=False, use_grad=False, abstain=True,
                         repair_mode='bisection', execution_horizon=5)
        self.gate = gate

    def score(self, ctx, chunk):
        return chunk_score(self.gate, ctx, chunk, self.beta)[0]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True, help='completed sequence-risk experiment directory')
    p.add_argument('--output', required=True)
    p.add_argument('--models', nargs='+', default=['frozen', 'static_s0', 'sequence_s0'],
                   help='default seed 0 is a diagnostic convention, not best-seed selection')
    p.add_argument('--device', default='auto')
    p.add_argument('--smoke', action='store_true')
    cli = p.parse_args()
    run = Path(cli.run).resolve()
    status, training, source = [load_json(run / n) for n in
                               ('status.json', 'training.json', 'evaluation.json')]
    if not all(x['complete'] for x in (status, training, source)) or status['smoke']:
        raise ValueError('requires completed production temporal-risk experiment')
    if len(set(cli.models)) != len(cli.models) or not set(cli.models) <= {'frozen', *training['models']}:
        raise ValueError('invalid model selection')
    for a in status['source_artifacts'].values():
        if digest(a['path']) != a['sha256']:
            raise RuntimeError('source artifact changed')
    args = SimpleNamespace(**status['effective_policy_args'])
    if args.policy != 'openvla' or args.chunk_k != 5 or args.n_envs != 32 or args.vla_batch_size != 8:
        raise ValueError('requires frozen full OpenVLA H10/k5/microbatch8 protocol')
    args.device = cli.device
    if cli.smoke:
        args.policy, args.n_envs = 'bc', 4
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dev, sim, od = setup(args)
    summary = {'complete': False, 'per_cell': [], 'metadata': {
        'diagnosis_only': True, 'smoke': cli.smoke, 'config': vars(cli),
        'source_result_sha256': digest(run / 'evaluation.json'),
        'source_status_sha256': digest(run / 'status.json'),
        'source_training_sha256': digest(run / 'training.json'),
        'source_artifacts': status['source_artifacts'],
        'source_code_sha256': {str(f): digest(f) for f in (
            Path(__file__), Path(__file__).with_name('analyze_sequence_risk.py'),
            Path(__file__).parents[1] / 'checkvla/reference.py',
            Path(__file__).parents[1] / 'checkvla/verifier.py',
            Path(__file__).parents[1] / 'sims/torch_push.py',
            Path(__file__).parents[1] / 'models/sequence_risk.py')},
        'GT_gate': 'exact H10 risk > physical limit 1; NOT learned calibration tau',
        'GT_repair': 'same six-step continuous bisection, floor .05, relative target min(.8,.75*s0), responsiveness .02',
        'H': 10, 'k': 5, 'policy_microbatch': 8,
        'scope': 'validation seeds only; privileged causal-component ablation, not learned-method efficacy',
    }}
    atomic_json(summary, output / 'summary.partial.json')
    policy = load_policy(args, od, dev, sim)
    if policy.H != 10:
        raise ValueError('requires H10')
    models, taus = {}, {}
    for name in cli.models:
        if name == 'frozen':
            models[name] = load_predictor(args, od, 'orbisim', dev, sim)
            taus[name] = load_json(args.calibration_path)['tau']['orbisim']
        else:
            info = training['models'][name]
            if digest(info['checkpoint']) != info['sha256']:
                raise RuntimeError('model hash mismatch')
            ck = torch.load(info['checkpoint'], map_location='cpu', weights_only=True)
            model = SequenceRisk(ck['kind'], ck['hidden'], ck['E']).to(dev).eval()
            model.load_state_dict(ck['state'])
            models[name], taus[name] = model, info['tau']
    entries = source['per_cell'][:1] if cli.smoke else source['per_cell']
    for ci, old in enumerate(entries):
        cell = old['cell']
        env, params = env_for_cell(args, args.n_envs, dev, cell)
        exact = ExactRisk(sim, cell)
        verifiers = {'gt_gt': GateRepairVerifier(exact, exact, 1., sim.max_vel)}
        for name, model in models.items():
            verifiers.update({
                f'{name}_ll': GateRepairVerifier(model, model, taus[name], sim.max_vel),
                f'{name}_gl': GateRepairVerifier(exact, model, 1., sim.max_vel),
                f'{name}_lg': GateRepairVerifier(model, exact, taus[name], sim.max_vel),
            })
        controllers = {'none': Controller(policy, sim, 'none', chunk_k=5)}
        controllers.update({n: Controller(policy, sim, 'checkvla', verifier=v, chunk_k=5)
                            for n, v in verifiers.items()})
        seed = 99000 + ci if cli.smoke else old['episode_seed']
        row = {'cell': cell, 'ood': old['ood'], 'episode_seed': seed, 'episodes': {}, 'raw': {}}
        for name, ctrl in controllers.items():
            result = run_episodes(env, sim, ctrl, params, seed=seed, record=True)
            if ctrl.n_policy_calls != 16 or ctrl.n_policy_rows != 16 * args.n_envs:
                raise RuntimeError('cadence changed')
            if not cli.smoke and (name == 'none' or name.endswith('_ll')):
                expected_name = 'none' if name == 'none' else name[:-3]
                verify_replay(result, old['episodes'][expected_name])
            row[name] = summarize(result)
            row['episodes'][name] = {k: result[k] for k in (
                'success', 'violation', 'first_viol', 'first_trig', 'first_success',
                'n_interventions', 'final_task_cost', 'stall_decisions', 'n_decisions')}
            raw_path = output / f'cell_{ci:02d}_{name}.npz'
            with open(str(raw_path) + '.tmp', 'wb') as f:
                np.savez_compressed(f, **row['episodes'][name], **result['trace'])
            os.replace(str(raw_path) + '.tmp', raw_path)
            row['raw'][name] = {'path': str(raw_path), 'sha256': digest(raw_path)}
            atomic_json(row, output / f'cell_{ci:02d}.partial.json')
            print(f'[factorial] cell={ci} arm={name} SR={row[name]["SR"]:.4f} CVR={row[name]["CVR"]:.4f} safe={row[name]["safe_success"]:.4f}', flush=True)
        summary['per_cell'].append(row)
        atomic_json(summary, output / 'summary.partial.json')
    summary['paired_vs_none'] = {name: pooled_paired(summary['per_cell'], name, 1700)
                                 for name in verifiers}
    summary['paired_vs_learned'] = {}
    for name in cli.models:
        rows = [{'ood': e['ood'], 'episodes': {'none': e['episodes'][name + '_ll'],
                 **{key: e['episodes'][key] for key in (name + '_gl', name + '_lg', 'gt_gt')}}}
                for e in summary['per_cell']]
        summary['paired_vs_learned'][name] = {key: pooled_paired(rows, key, 1700)
                                              for key in (name + '_gl', name + '_lg', 'gt_gt')}
    ood = [e for e in summary['per_cell'] if e['ood']]
    summary['pooled_ood'] = {name: {k: float(np.mean([e[name][k] for e in ood]))
                                    for k in ('SR', 'CVR', 'safe_success')} for name in controllers}
    summary['complete'] = True
    atomic_json(summary, output / 'summary.json')
    print('[factorial] complete', output, flush=True)


if __name__ == '__main__':
    main()
