"""Fit joint risk and train-only episodic stage bank without opening reserved data."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..legacy.microwave_controller import NativeFramework
from ..perception.frozen_visual_encoder import FrozenVisualEncoder, file_sha
from ..verification.calibrated_risk_head import RiskHeadMLP
from ..verification.fusion_mlp import extract_physics_features, STAGES
from ..verification.outcome_risk import temporal_features, RISK_SCHEMA
from .train_fusion_mlp import metrics


def risk_label(current, history, future, horizon=24):
    """Measured regression or sustained acquisition stagnation in nominal rollout."""
    states = []
    for row in future[:horizon]:
        if row['owner'] != 'nominal':
            return None, 'repair_censored'
        s = row['state']
        if any(s.get(k) is None for k in ('held_clear', 'in_goal', 'grasp', 'close_goal')):
            return None, 'missing_evidence'
        if any(current.get(k) is True and s.get(k) is False for k in ('in_goal', 'close_goal')):
            return 1, 'measured_preserved_goal_loss'
        if current.get('held_clear') is True and s['held_clear'] is False and s['in_goal'] is False:
            return 1, 'measured_lift_loss'
        states.append(s)
        if s.get('native_complete') is True:
            return 0, 'native_completion_without_regression'
    if len(states) < horizon:
        return None, 'right_censored'
    prior_grasp = 0
    for s in reversed(history):
        if s.get('grasp') is not True or s.get('held_clear') is not False:
            break
        prior_grasp += 1
    if prior_grasp+len(states) >= 30 and prior_grasp and all(
            s['grasp'] is True and s['held_clear'] is False and s['in_goal'] is False for s in states):
        return 1, 'observed_sustained_grasp_without_lift'
    positions = [s.get('position') for s in [current]+states]
    if all(p is not None for p in positions) and not current.get('in_goal') and all(
            s['grasp'] is False for s in [current]+states) and max(
            np.linalg.norm(np.asarray(p)-positions[0]) for p in positions) < .003:
        return 1, 'observed_acquisition_no_motion'
    return 0, 'observed_nominal_window_without_defined_violation'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fusion-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=60)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Output exists; do not overwrite experiments')
    args.out.mkdir(parents=True)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    manifest = json.loads((args.fusion_dir/'dataset_manifest.json').read_text())
    report = json.loads((args.fusion_dir/'report.json').read_text())
    encoder_meta = report['metadata']['encoder']
    encoder = FrozenVisualEncoder(encoder_meta['checkpoint'], encoder_meta['source'],
        device=args.device, expected_sha=encoder_meta['sha256'])
    if encoder.provenance != encoder_meta:
        raise ValueError('Visual source/preprocessing drift')
    cache_file = args.fusion_dir/'visual_features.npz'
    if not cache_file.exists():
        raise ValueError('Pass training output with saved real visual features')
    with np.load(cache_file, allow_pickle=False) as cache:
        visual_cache = dict(zip(cache['paths'].tolist(), cache['features']))
    records, bank = [], []
    for episode in manifest['episodes']:
        if episode['split'] == 'evaluation':
            continue  # No evaluation feature inspection/fitting/calibration here.
        directory = Path(episode['episode'])
        if file_sha(directory/'controller.jsonl') != episode['controller_sha256']:
            raise ValueError('Controller data changed')
        rows = [json.loads(line) for line in (directory/'controller.jsonl').read_text().splitlines()]
        before = {r['step']: r['decision'] for r in rows if r['event'] == 'before'}
        after = {r['step']: dict(owner=r['decision']['owner'], state=r['verification']['evidence']['state'])
                 for r in rows if r['event'] == 'after'}
        success = json.loads((directory/'status.json').read_text()).get('any_native_success') is True
        history = []
        for step, decision in before.items():
            current = decision['evidence']['state']
            history.append(current)
            if step % 8:
                continue
            stage = NativeFramework._subgoal(current)
            image = str(directory/'model_inputs'/f'{step:06d}.png')
            if image not in visual_cache:
                visual_cache[image] = encoder.encode(np.asarray(Image.open(image).convert('RGB')))
            v = visual_cache[image]
            if episode['split'] == 'train' and success:
                bank.append(dict(feature=v, stage=stage, metadata=dict(episode=str(directory), step=step,
                    root=episode['root'], source='completed_successful_train_trajectory',
                    image_sha256=file_sha(image))))
            if decision['owner'] != 'nominal':
                continue
            future = [after[i] for i in range(step, min(step+24, len(after)))]
            label, reason = risk_label(current, history, future)
            if label is None:
                continue
            records.append(dict(episode=str(directory), split=episode['split'], step=step, root=episode['root'],
                stage=stage, visual=v, physics=extract_physics_features(current),
                temporal=temporal_features(history, stage), label=label, reason=reason))
    train = [s for s in records if s['split'] == 'train']
    validation = [s for s in records if s['split'] == 'validation']
    if len({s['label'] for s in train}) != 2 or not bank or not validation:
        raise ValueError('Need both measured risk outcomes, episodic bank and isolated validation')
    raw = np.stack([np.concatenate([s['physics'][:12], s['temporal'][:8]]) for s in train])
    mean, std = raw.mean(0), np.maximum(raw.std(0), .01)
    for i in range(12):
        observed = [s['physics'][i] for s in train if s['physics'][i+12] == 1]
        mean[i] = np.mean(observed) if observed else 0.
        std[i] = max(float(np.std(observed)), .01) if observed else 1.

    def tensors(rows):
        visual = np.stack([s['visual'] for s in rows])
        physics, temporal = [], []
        for s in rows:
            x = (np.concatenate([s['physics'][:12], s['temporal'][:8]])-mean)/std
            physics.append(np.concatenate([x[:12]*s['physics'][12:], s['physics'][12:]]))
            temporal.append(np.concatenate([x[12:], s['temporal'][8:]]))
        return [torch.tensor(np.asarray(x), dtype=torch.float32, device=args.device)
            for x in (visual, physics, temporal, [[s['label']] for s in rows])]

    model = RiskHeadMLP(vision_dim=384, physics_dim=24, stats_dim=12).to(args.device)
    v, f, t, y = tensors(train)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(1-y).sum()/y.sum())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    model.train()
    for _ in range(args.epochs):
        for ids in torch.randperm(len(train), device=args.device).split(64):
            optimizer.zero_grad()
            loss = criterion(model(v[ids], f[ids], t[ids]), y[ids])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        vv, ff, tt, yy = tensors(validation)
        scores = model(vv, ff, tt).sigmoid()[:, 0].cpu().numpy()
    negative_scores = scores[np.array([s['label'] == 0 for s in validation])]
    if len(negative_scores) == 0:
        raise ValueError('No observed negative-risk validation outcomes')
    threshold = float(np.quantile(negative_scores, .95, method='higher'))
    artifact = dict(risk_schema=RISK_SCHEMA, risk_state_dict=model.cpu().state_dict(),
        input_mean=torch.from_numpy(mean), input_std=torch.from_numpy(std), risk_threshold=threshold,
        encoder=encoder_meta, bank_features=torch.from_numpy(np.stack([b['feature'] for b in bank])),
        bank_stages=[b['stage'] for b in bank], bank_metadata=[b['metadata'] for b in bank],
        manifest_sha256=file_sha(args.fusion_dir/'dataset_manifest.json'),
        risk_horizon=24, risk_persistence=3, no_lift_steps=30, stage_dwell=3,
        calibration='validation_negative_sample_95percent_quantile_not_conformal_coverage',
        train_roots=[1, 2, 5], validation_roots=[3, 4], evaluation_roots_excluded=[0])
    torch.save(artifact, args.out/'joint_components.pt')
    summary = dict(encoder=encoder_meta, risk_schema=RISK_SCHEMA, threshold=threshold,
        calibration=artifact['calibration'], train_labels=dict(Counter(str(s['label']) for s in train)),
        validation_labels=dict(Counter(str(s['label']) for s in validation)),
        validation=metrics([s['label'] for s in validation], scores > threshold),
        bank_entries=len(bank), bank_stages=dict(Counter(b['stage'] for b in bank)),
        roots=dict(train=[1, 2, 5], validation=[3, 4], excluded_evaluation=[0]),
        definition='24 nominal actions: preserved predicate/lift loss or sustained acquisition stagnation',
        risk_artifact_sha256=file_sha(args.out/'joint_components.pt'))
    with (args.out/'risk_labels.jsonl').open('w') as stream:
        for s in records:
            stream.write(json.dumps({k: value for k, value in s.items() if k not in ('visual','physics','temporal')})+'\n')
    (args.out/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
