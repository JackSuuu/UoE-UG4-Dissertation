"""Train VerificationFusionMLP on completed framework trajectories.

Usage:
    python src/adapters/train_fusion_mlp.py \
        --run-dir /home/ami/scratch/framework_prompt_refine_run_20261021 \
        --out src/adapters/fusion_mlp.pt \
        --epochs 50

Reads steps.jsonl + controller.jsonl from completed cells, labels each step
by verification outcome, extracts physics features from measured diagnostics,
and trains the fusion MLP.  Vision features are zero placeholders until the
frozen encoder (DINOv2/V-JEPA) is wired and used to encode saved PNGs.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapters.fusion_mlp import (
    VerificationFusionMLP, extract_physics_features, train_on_pairs,
    save_model, STAGES, STAGE_TO_IDX,
)


def load_cell(cell_dir: Path):
    steps_path = cell_dir / 'steps.jsonl'
    ctrl_path = cell_dir / 'controller.jsonl'
    if not steps_path.exists() or not ctrl_path.exists():
        return None, None
    steps = [json.loads(l) for l in steps_path.read_text().splitlines() if l.strip()]
    ctrl = [json.loads(l) for l in ctrl_path.read_text().splitlines() if l.strip()]
    return steps, ctrl


def build_training_pairs(run_dir: Path, max_cells: int = 20):
    pairs = []
    cells = sorted([d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith('replay_')])
    used = 0
    for cell_dir in cells:
        if used >= max_cells:
            break
        steps, ctrl = load_cell(cell_dir)
        if steps is None:
            continue
        used += 1

        ctrl_by_step = {}
        for row in ctrl:
            dec = row.get('decision', {})
            ver = row.get('verification', {})
            step = dec.get('step')
            if step is not None:
                ctrl_by_step[step] = {
                    'verified': ver.get('repair_verified', False),
                    'owner': dec.get('owner'),
                    'subgoal': dec.get('subgoal'),
                }

        for i, step_row in enumerate(steps):
            step_num = step_row.get('step', i)
            info = ctrl_by_step.get(step_num, {})
            measured = step_row.get('diagnostics', {})
            if not isinstance(measured, dict):
                continue

            label = 1.0 if info.get('verified') is True else 0.0
            subgoal = info.get('subgoal') or 'transfer'

            vision_feat = np.zeros(384, dtype=np.float32)  # placeholder until encoder wired

            pairs.append({
                'vision_feat': vision_feat,
                'measured': measured,
                'stage': subgoal,
                'label': label,
            })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-cells', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print(f'[train] Reading cells from {args.run_dir}')
    pairs = build_training_pairs(args.run_dir, max_cells=args.max_cells)
    print(f'[train] Built {len(pairs)} training pairs')

    if len(pairs) == 0:
        print('[train] No pairs found; check run-dir path')
        return

    pos = sum(1 for p in pairs if p['label'] > 0.5)
    neg = len(pairs) - pos
    print(f'[train] Positive={pos} Negative={neg}')

    model = VerificationFusionMLP()
    losses = train_on_pairs(
        model, pairs, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, device=args.device,
    )
    print(f'[train] Final loss: {losses[-1]:.4f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(args.out))
    print(f'[train] Saved model to {args.out}')


if __name__ == '__main__':
    main()
