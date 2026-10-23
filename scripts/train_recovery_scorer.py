"""Train module A: lightweight action-conditioned recovery scorer.

Reads audited repair traces from study/ directories, encodes the policy PNG
at the first-repair probe step, extracts masked physics features, and trains
a small MLP to predict whether the *executed* candidate action led to native
task success.  Unexecuted candidates receive no label (no counterfactual
label leakage).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repair.recovery_scorer import RecoveryScorerMLP, FEATURE_SCHEMA
from src.perception.frozen_visual_encoder import FrozenVisualEncoder
from src.verification.fusion_mlp import extract_physics_features


def collect(run_dirs, encoder, device):
    X_v, X_p, X_a, y = [], [], [], []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        for ep_dir in sorted((run_dir / 'episodes').iterdir()):
            status_path = ep_dir / 'status.json'
            if not status_path.exists():
                continue
            status = json.loads(status_path.read_text())
            if not status.get('valid_episode'):
                continue
            success = bool(status.get('any_native_success'))
            steps_path = ep_dir / 'steps.jsonl'
            controls_path = ep_dir / 'controller.jsonl'
            if not steps_path.exists() or not controls_path.exists():
                continue
            steps = [json.loads(l) for l in steps_path.read_text().splitlines()]
            probes = [r for r in steps if 'candidate_selection' in r]
            if len(probes) != 1:
                continue
            probe = probes[0]
            step = probe['step']
            sel = probe['candidate_selection']
            candidates = sel['candidates']
            selected_id = sel['selected_id']
            img_path = ep_dir / 'model_inputs' / f'{step:06d}.png'
            if not img_path.exists():
                continue
            img = np.asarray(Image.open(img_path).convert('RGB'))
            vf = encoder.encode(img)
            controls = [json.loads(l) for l in controls_path.read_text().splitlines()]
            before = next((r for r in controls if r['event'] == 'before' and r['step'] == step), None)
            if before is None:
                continue
            state = before['decision']['evidence']['state']
            pf = extract_physics_features(state)
            for c in candidates:
                ac = np.asarray(c['decode']['action'], dtype=np.float32)
                if c['candidate_id'] == selected_id:
                    label = 1.0 if success else 0.0
                else:
                    continue  # unexecuted: no label
                X_v.append(vf); X_p.append(pf); X_a.append(ac); y.append(label)
    return X_v, X_p, X_a, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True,
                        help='study directory containing episodes/')
    parser.add_argument('--encoder-checkpoint', type=Path, required=True)
    parser.add_argument('--encoder-source', type=Path, required=True)
    parser.add_argument('--encoder-sha', type=str, default=None)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    encoder = FrozenVisualEncoder(args.encoder_checkpoint, args.encoder_source,
        device=args.device, expected_sha=args.encoder_sha)
    X_v, X_p, X_a, y = collect(args.run_dir, encoder, args.device)
    if not X_v:
        raise ValueError('No training labels found in provided run dirs')

    Xv = torch.tensor(np.stack(X_v), dtype=torch.float32)
    Xp = torch.tensor(np.stack(X_p), dtype=torch.float32)
    Xa = torch.tensor(np.stack(X_a), dtype=torch.float32)
    Xe = torch.zeros(len(y), 4, dtype=torch.float32)  # physics evidence placeholder
    Y = torch.tensor(y, dtype=torch.float32)

    model = RecoveryScorerMLP().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    ds = TensorDataset(Xv, Xp, Xa, Xe, Y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for vv, pp, aa, ee, yy in dl:
            vv, pp, aa, ee, yy = vv.to(args.device), pp.to(args.device), aa.to(args.device), ee.to(args.device), yy.to(args.device)
            opt.zero_grad()
            pred = model(vv, pp, aa, ee)
            loss = criterion(pred, yy)
            loss.backward()
            opt.step()
            total += loss.item() * len(yy)
        print(f'epoch {epoch+1}/{args.epochs} loss={total/len(ds):.4f}', flush=True)

    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = dict(
        state_dict=model.state_dict(),
        metadata=dict(
            feature_schema=FEATURE_SCHEMA, threshold=0.5,
            trained_from='audited_repair_traces', n_samples=len(y),
            epochs=args.epochs, lr=args.lr,
        ),
    )
    torch.save(ckpt, args.out / 'recovery_scorer.pt')
    print('saved', args.out / 'recovery_scorer.pt', flush=True)


if __name__ == '__main__':
    main()
