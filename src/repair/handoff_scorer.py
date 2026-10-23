"""Lightweight handoff-readiness scorer.

Answers: if control is returned to the nominal VLA right now, is the task
likely to be preserved / completed over the next few steps?

Input: current visual feature (384) + masked physics state (24).  No candidate
action is needed; this scores the *state*, not a proposed action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class HandoffScorerMLP(nn.Module):
    def __init__(self, vision_dim: int = 384, physics_dim: int = 24, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim + physics_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.metadata = {}

    def forward(self, vision_feat, physics_feat):
        x = torch.cat([vision_feat, physics_feat], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class HandoffScorer:
    def __init__(self, checkpoint: Optional[Path] = None, device: str = 'cpu',
                 vision_dim: int = 384, physics_dim: int = 24,
                 hidden: int = 64, threshold: float = 0.5):
        self.device = device
        self.threshold = threshold
        self.model = HandoffScorerMLP(vision_dim, physics_dim, hidden)
        self.metadata = {}
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.model.load_state_dict(ckpt['state_dict'], strict=True)
            self.metadata = ckpt.get('metadata', {})
            self.threshold = float(self.metadata.get('threshold', threshold))
        self.model.to(device).eval()

    def score(self, vision_feat, physics_feat) -> dict:
        vf = np.asarray(vision_feat, dtype=np.float32).reshape(-1)
        pf = np.asarray(physics_feat, dtype=np.float32).reshape(-1)
        with torch.inference_mode():
            v = torch.from_numpy(vf[None]).to(self.device)
            p = torch.from_numpy(pf[None]).to(self.device)
            prob = float(self.model(v, p)[0].cpu().numpy())
        return dict(
            probability=prob, threshold=self.threshold,
            accept=prob >= self.threshold, schema='handoff_scorer_v1',
        )
