"""Train real visual/physical fusion, isolate roots, select a validation threshold.

python -m src.training.train_fusion_mlp --run-dir RUN --out NEW_DIR \
    --encoder-checkpoint WEIGHTS --encoder-source DINOV2_SOURCE --device cuda:0

Output is a NEW experiment directory (old weights are never overwritten).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.perception.frozen_visual_encoder import FrozenVisualEncoder, file_sha
from src.training.handoff_dataset import collect_samples, contract, LABEL_SPEC
from src.verification.fusion_mlp import (VerificationFusionMLP, extract_physics_features,
    train_on_pairs, save_model, STAGE_TO_IDX, FEATURE_SCHEMA)


def metrics(labels, accepted):
    y, a = np.asarray(labels, dtype=bool), np.asarray(accepted, dtype=bool)
    tp, fp, tn, fn = [int(x.sum()) for x in (a & y, a & ~y, ~a & ~y, ~a & y)]
    return dict(n=len(y), positives=int(y.sum()), negatives=int((~y).sum()),
        tp=tp, fp=fp, tn=tn, fn=fn,
        false_accept_rate=fp/(fp+tn) if fp+tn else None,
        false_reject_rate=fn/(fn+tp) if fn+tp else None,
        unsafe_among_accepted=fp/(tp+fp) if tp+fp else None,
        accuracy=(tp+tn)/len(y) if len(y) else None)


def select_threshold(labels, scores, max_false_accept=.05):
    if len(set(labels)) < 2:
        return None
    feasible = []
    for threshold in sorted(set([0., 1.] + list(map(float, scores)))):
        m = metrics(labels, np.asarray(scores) >= threshold)
        if m['false_accept_rate'] <= max_false_accept and m['tp'] > 0:
            feasible.append((m['false_reject_rate'], m['false_accept_rate'], threshold))
    return min(feasible)[2] if feasible else None


def predict(model, samples):
    if not samples:
        return np.empty(0)
    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), 256):
            batch = samples[start:start+256]
            v = torch.from_numpy(np.stack([s['vision_feat'] for s in batch])).to(device)
            p = torch.from_numpy(np.stack([s['physics_feat'] for s in batch])).to(device)
            stage = torch.eye(4, device=device)[[STAGE_TO_IDX[s['stage']] for s in batch]]
            result.extend(model(v, p, stage).softmax(-1)[:, 1].cpu().tolist())
    return np.asarray(result)


def evaluate(samples, scores, threshold):
    y = [s['label'] for s in samples]
    # A measured-current-contract baseline cannot see future regression either.
    rule = [contract(s['state'], s['stage']) is True for s in samples]
    report = dict(rule_current_contract=metrics(y, rule), always_accept=metrics(y, [True]*len(y)),
        always_reject=metrics(y, [False]*len(y)),
        mlp=metrics(y, scores >= threshold) if threshold is not None else None)
    selected = [i for i, s in enumerate(samples) if s['actual_handoff']]
    report['actual_handoffs'] = dict(n=len(selected),
        mlp=metrics([y[i] for i in selected], scores[selected] >= threshold) if threshold is not None else None,
        rule_current_contract=metrics([y[i] for i in selected], [rule[i] for i in selected]))
    eligible = np.asarray(rule, dtype=bool)
    report['current_contract_true'] = dict(n=int(eligible.sum()),
        mlp=metrics(np.asarray(y)[eligible], scores[eligible] >= threshold) if threshold is not None else None,
        rule_current_contract=metrics(np.asarray(y)[eligible], np.ones(int(eligible.sum()), dtype=bool)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, action='append')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--encoder-checkpoint', type=Path, required=True)
    parser.add_argument('--encoder-source', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--encoder-batch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('Refusing to overwrite a previous training experiment')
    args.out.mkdir(parents=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    samples, manifest = collect_samples(args.run_dir, stride=args.stride)
    (args.out/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('trajectory counts', Counter(s['split'] for s in manifest['episodes']), flush=True)
    encoder = FrozenVisualEncoder(args.encoder_checkpoint, args.encoder_source, device=args.device)
    (args.out/'encoder.json').write_text(json.dumps(encoder.provenance, indent=2)+'\n')
    images = sorted({s['image'] for s in samples})
    cache = {}
    for start in range(0, len(images), args.encoder_batch_size):
        paths = images[start:start+args.encoder_batch_size]
        rgb = [np.asarray(Image.open(p).convert('RGB')) for p in paths]
        features = encoder.encode_batch(rgb)
        for path, feature in zip(paths, features):
            cache[path] = feature
        if start % 256 == 0:
            print('encoded', min(start+len(paths), len(images)), '/', len(images), flush=True)
    np.savez_compressed(args.out/'visual_features.npz', paths=np.asarray(images), features=np.stack([cache[p] for p in images]))
    for sample in samples:
        sample['vision_feat'] = cache[sample['image']]
        sample['physics_feat'] = extract_physics_features(sample['state'])
        sample['image_sha256'] = file_sha(sample['image'])
    splits = {name: [s for s in samples if s['split'] == name] for name in ('train', 'validation', 'evaluation')}
    if any(not rows for rows in splits.values()):
        raise ValueError('Empty trajectory split')
    # No validation/evaluation statistics are used in normalization or fitting.
    model = VerificationFusionMLP()
    losses = train_on_pairs(model, splits['train'], epochs=args.epochs,
        batch_size=args.batch_size, device=args.device, seed=args.seed)
    validation_scores = predict(model, splits['validation'])
    # The causal gate is consulted only after current-contract checks pass.
    # Selecting on all easy current-false negatives would dilute unsafe handoffs.
    eligible = np.asarray([contract(s['state'], s['stage']) is True for s in splits['validation']])
    threshold = select_threshold(np.asarray([s['label'] for s in splits['validation']])[eligible].tolist(),
        validation_scores[eligible])
    model.metadata = dict(feature_schema=FEATURE_SCHEMA, label_spec=LABEL_SPEC,
        encoder=encoder.provenance, threshold=threshold, threshold_split='validation',
        threshold_population='current_contract_true', supported_stages=['acquire', 'transfer', 'contain_release'],
        max_empirical_false_accept=.05, seed=args.seed,
        dataset_manifest_sha256=file_sha(args.out/'dataset_manifest.json'),
        roots=manifest['roots'], horizon=manifest['horizon'],
        scope='reused-development nominal continuation; not unseen-task generalization',
        control_ready=threshold is not None, efficacy_validated=False, training_losses=losses)
    save_model(model, args.out/'fusion_mlp_v2.pt')
    report = dict(metadata=model.metadata, splits={})
    with (args.out/'predictions.jsonl').open('w') as stream:
        for name, data in splits.items():
            scores = validation_scores if name == 'validation' else predict(model, data)
            report['splits'][name] = evaluate(data, scores, threshold)
            report['splits'][name]['labels'] = dict(Counter(str(s['label']) for s in data))
            for sample, score in zip(data, scores):
                row = {k: v for k, v in sample.items() if k not in ('vision_feat', 'physics_feat')}
                stream.write(json.dumps(dict(row, probability=float(score)), allow_nan=False)+'\n')
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
