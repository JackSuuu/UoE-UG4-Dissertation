"""Fit a generic risk/A/B head on disjoint training roots; no old stage weights.

Dataset creation and experiment bookkeeping live in src.experiment, not here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import math

from ..verification.features import FEATURE_DIM, FEATURE_SCHEMA
from .handoff_dataset import LABEL_SCHEMA


def validate_rows(rows, role):
    signatures, ids = [], set()
    for row in rows:
        if (row.get('schema') != FEATURE_SCHEMA or row.get('label_schema') != LABEL_SCHEMA
                or row.get('role') != role or type(row.get('root')) is not int
                or row['root'] not in range(10, 20) or row.get('split') != 'training'
                or type(row.get('label')) is not int or row['label'] not in (0, 1)):
            raise ValueError('Only versioned, labelled training-root records are accepted')
        if row['sample_id'] in ids:
            raise ValueError('Duplicate sample')
        ids.add(row['sample_id'])
        vector = row['features']
        if len(vector) != FEATURE_DIM or not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
            raise ValueError('Feature dimension/values mismatch')
        signature = row.get('encoder_signature')
        if (not isinstance(signature, dict) or set(signature) != {'sha256', 'source_sha256', 'transform', 'feature_dim'}
                or signature['feature_dim'] != 384):
            raise ValueError('Real encoder provenance is required')
        signatures.append(signature)
    if not rows or any(signature != signatures[0] for signature in signatures):
        raise ValueError('Dataset empty or mixes incompatible encoders')
    return signatures[0]


def threshold_metrics(scores, labels):
    if set(labels) != {0, 1}:
        raise ValueError('Both classes are required for validation/calibration')
    best = None
    for threshold in sorted({0., 1., *scores}):
        pred = [v > threshold for v in scores]
        tpr = sum(p and y == 1 for p, y in zip(pred, labels))/labels.count(1)
        fpr = sum(p and y == 0 for p, y in zip(pred, labels))/labels.count(0)
        candidate = (tpr-fpr, -fpr, threshold)
        if best is None or candidate > best[0]:
            best = (candidate, dict(threshold=threshold, tpr=tpr, fpr=fpr,
                                   calibration='empirical_validation_not_conformal_coverage'))
    return best[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--role', choices=('risk', 'recovery', 'handoff'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-roots', nargs='+', type=int, default=list(range(10, 18)))
    parser.add_argument('--validation-roots', nargs='+', type=int, default=[18, 19])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if (not args.train_roots or not args.validation_roots or args.epochs < 1
            or set(args.train_roots) & set(args.validation_roots)
            or any(r not in range(10, 20) for r in [*args.train_roots, *args.validation_roots])):
        parser.error('Need positive epochs and disjoint training/validation roots within 10–19')
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    signature = validate_rows(rows, args.role)
    train = [r for r in rows if r['root'] in args.train_roots]
    validation = [r for r in rows if r['root'] in args.validation_roots]
    if set(r['label'] for r in train) != {0, 1} or set(r['label'] for r in validation) != {0, 1}:
        parser.error('Both train and validation need positive and negative observed labels; collect more data')
    import torch
    from ..verification.fusion_mlp import FusionMLP
    torch.manual_seed(args.seed)
    head = FusionMLP(device=args.device)
    x = torch.tensor([r['features'] for r in train], dtype=torch.float32, device=args.device)
    y = torch.tensor([[r['label']] for r in train], dtype=torch.float32, device=args.device)
    optimizer = torch.optim.Adam(head.net.parameters(), lr=1e-3)
    head.net.train()
    for _ in range(args.epochs):
        for indices in torch.randperm(len(train), device=args.device).split(64):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head.net(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    head.net.eval().requires_grad_(False)
    head.ready = True
    scores = [head.probability(r['features']) for r in validation]
    report = threshold_metrics(scores, [r['label'] for r in validation])
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(dict(schema=FEATURE_SCHEMA, feature_dim=FEATURE_DIM, role=args.role, trained=True,
        hidden=head.hidden, state_dict={k: v.cpu() for k, v in head.net.state_dict().items()},
        encoder_signature=signature, training_roots=sorted({r['root'] for r in train}),
        validation_roots=sorted({r['root'] for r in validation}), threshold=report['threshold'],
        label_schema=LABEL_SCHEMA), args.out/'model.pt')
    report.update(role=args.role, train_samples=len(train), validation_samples=len(validation), seed=args.seed,
                  dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                  training_roots=sorted({r['root'] for r in train}),
                  validation_roots=sorted({r['root'] for r in validation}))
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
