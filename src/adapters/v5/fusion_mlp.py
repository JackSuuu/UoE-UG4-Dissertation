"""Small MLP for multi-modal fusion in the verification loop.

Fuses vision features (frozen encoder), physical measurements (position,
contact, grasp force), and the current subgoal into a single verification
probability.  This is the learned component that sits *between* the raw
evidence and the framework's accept / handoff decision.

Design principles:
- The MLP does NOT generate robot actions.  It only scores whether the
  current state satisfies the repair acceptance contract and whether it is
  safe to hand control back to the nominal VLA policy.
- Inputs are low-dimensional and interpretable; the model is tiny (~30K
  parameters) so it can be trained on a handful of paired trajectories.
- At inference time it runs in eval() mode with no gradients.
"""
from typing import Optional, Dict, Any, List
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch optional at import time
    HAS_TORCH = False


STAGES = ('acquire', 'transfer', 'contain_release', 'close')
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}


class VerificationFusionMLP(nn.Module if HAS_TORCH else object):
    """Fusion MLP: vision + physics + subgoal -> verification probability."""

    def __init__(
        self,
        vision_dim: int = 256,
        physics_dim: int = 12,
        num_stages: int = 4,
        hidden1: int = 128,
        hidden2: int = 64,
    ):
        if not HAS_TORCH:
            raise RuntimeError('PyTorch is required for VerificationFusionMLP')
        super().__init__()
        self.vision_dim = vision_dim
        self.physics_dim = physics_dim
        self.num_stages = num_stages
        input_dim = vision_dim + physics_dim + num_stages

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, 2),  # [verify_pass, verify_fail] logits
        )

    def forward(
        self,
        vision_feat: 'torch.Tensor',        # [B, vision_dim]
        physics_feat: 'torch.Tensor',       # [B, physics_dim]
        stage_onehot: 'torch.Tensor',       # [B, num_stages]
    ) -> 'torch.Tensor':
        x = torch.cat([vision_feat, physics_feat, stage_onehot], dim=-1)
        logits = self.net(x)
        return logits  # [B, 2]

    def predict_verify_probability(
        self,
        vision_feat: np.ndarray,
        physics_feat: np.ndarray,
        stage: str,
    ) -> float:
        """Return P(verification passes) for a single state."""
        if not HAS_TORCH:
            raise RuntimeError('PyTorch required for inference')
        self.eval()
        with torch.no_grad():
            vf = torch.from_numpy(vision_feat.astype(np.float32)).unsqueeze(0)
            pf = torch.from_numpy(physics_feat.astype(np.float32)).unsqueeze(0)
            soh = torch.zeros(1, self.num_stages, dtype=torch.float32)
            if stage in STAGE_TO_IDX:
                soh[0, STAGE_TO_IDX[stage]] = 1.0
            logits = self.forward(vf, pf, soh)
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0, 0].item())


def extract_physics_features(
    measured: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Convert measured diagnostics into a fixed-length physics feature vector.

    Features (all physical, no learned embeddings):
      [0]  in_goal (0/1)
      [1]  close_goal (0/1)
      [2]  grasp (0/1)
      [3]  held_clear (0/1)
      [4]  table contact (0/1)
      [5]  mug_microwave contact (0/1)
      [6]  gripper_microwave contact (0/1)
      [7]  mug_gripper contact (0/1)
      [8]  object x position (m, clamped)
      [9]  object y position (m, clamped)
      [10] object z position (m, clamped)
      [11] rise_m (m, clamp at 0)
    """
    def b(val):
        return 1.0 if val is True else 0.0

    pos = measured.get('position') or [0.0, 0.0, 0.0]
    return np.array([
        b(measured.get('in_goal')),
        b(measured.get('close_goal')),
        b(measured.get('grasp')),
        b(measured.get('held_clear')),
        b(measured.get('table')),
        b(measured.get('mug_microwave')),
        b(measured.get('gripper_microwave')),
        b(measured.get('mug_gripper')),
        float(np.clip(pos[0], -2, 2)),
        float(np.clip(pos[1], -2, 2)),
        float(np.clip(pos[2], -2, 2)),
        float(max(0.0, measured.get('rise_m') or 0.0)),
    ], dtype=np.float32)


def save_model(model: VerificationFusionMLP, path: str) -> None:
    torch.save({
        'state_dict': model.state_dict(),
        'vision_dim': model.vision_dim,
        'physics_dim': model.physics_dim,
        'num_stages': model.num_stages,
    }, path)


def load_model(path: str, device='cpu') -> VerificationFusionMLP:
    checkpoint = torch.load(path, map_location=device)
    model = VerificationFusionMLP(
        vision_dim=checkpoint['vision_dim'],
        physics_dim=checkpoint['physics_dim'],
        num_stages=checkpoint['num_stages'],
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------------
# Minimal training loop (supervised on paired trajectories)
# ------------------------------------------------------------------
def train_on_pairs(
    model: VerificationFusionMLP,
    pairs: List[Dict[str, Any]],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device='cpu',
) -> List[float]:
    """Train the MLP on (state, label) pairs.

    Each pair dict must contain:
      - 'vision_feat': np.ndarray [vision_dim]
      - 'measured': dict with measured diagnostics
      - 'stage': str in STAGES
      - 'label': float in {0.0, 1.0}  (1 = verification should pass)
    """
    if not HAS_TORCH:
        raise RuntimeError('PyTorch required for training')

    import torch.optim as optim

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Build tensors
    v_list, p_list, s_list, y_list = [], [], [], []
    for pair in pairs:
        v_list.append(pair['vision_feat'].astype(np.float32))
        p_list.append(extract_physics_features(pair['measured']))
        s_list.append(STAGE_TO_IDX.get(pair['stage'], 0))
        y_list.append(float(pair['label']))

    X_v = torch.from_numpy(np.stack(v_list)).to(device)
    X_p = torch.from_numpy(np.stack(p_list)).to(device)
    X_s = torch.zeros(len(pairs), len(STAGES), dtype=torch.float32).to(device)
    for i, si in enumerate(s_list):
        X_s[i, si] = 1.0
    Y = torch.from_numpy(np.array(y_list, dtype=np.float32)).unsqueeze(1).to(device)

    n = len(pairs)
    losses: List[float] = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X_v[idx], X_p[idx], X_s[idx])
            loss = criterion(logits, Y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / n)
    return losses


def build_pairs_from_trajectory(
    steps: List[Dict[str, Any]],
    vision_feats: List[np.ndarray],
    positive_phase: str = 'post_handoff_stable',
) -> List[Dict[str, Any]]:
    """Construct training pairs from a single trajectory.

    Positive examples: steps after a verified handoff where the state remains
    stable (goal preserved).
    Negative examples: steps where a trigger fired but the repair was later
    unverified (budget exit), or where post-handoff goal loss occurred.
    """
    pairs = []
    for i, step in enumerate(steps):
        if i >= len(vision_feats):
            break
        measured = step.get('measured', {})
        stage = step.get('stage', 'transfer')
        label = 0.0

        # Heuristic labeling: replace with real verification outcomes
        if step.get('verified') is True:
            label = 1.0
        elif step.get('exit_reason') == 'local_budget_exit_unverified':
            label = 0.0
        elif step.get('post_handoff_goal_loss') is True:
            label = 0.0

        pairs.append({
            'vision_feat': vision_feats[i],
            'measured': measured,
            'stage': stage,
            'label': label,
        })
    return pairs
