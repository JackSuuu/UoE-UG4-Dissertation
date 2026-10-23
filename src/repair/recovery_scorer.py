"""Lightweight action-conditioned recovery scorer for frozen OpenVLA repair.

Only scores candidate actions produced by the frozen VLA.  Does not generate
or modify actions.  Input: DINOv2 visual feature (384) + masked physics state
(24) + candidate action (7) + physics evidence (4).  Output: probability the
action leads to reliable recovery and eventual task completion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


FEATURE_SCHEMA = 'recovery_scorer_v1'


class RecoveryScorerMLP(nn.Module):
    def __init__(self, vision_dim: int = 384, physics_dim: int = 24,
                 action_dim: int = 7, evidence_dim: int = 4,
                 hidden: int = 64):
        super().__init__()
        self.vision_dim = vision_dim
        self.physics_dim = physics_dim
        self.action_dim = action_dim
        self.evidence_dim = evidence_dim
        self.net = nn.Sequential(
            nn.Linear(vision_dim + physics_dim + action_dim + evidence_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.metadata = {}

    def forward(self, vision_feat, physics_feat, action, evidence):
        x = torch.cat([vision_feat, physics_feat, action, evidence], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class RecoveryScorer:
    def __init__(self, checkpoint: Optional[Path] = None, device: str = 'cpu',
                 vision_dim: int = 384, physics_dim: int = 24,
                 action_dim: int = 7, evidence_dim: int = 4,
                 hidden: int = 64,
                 threshold: float = 0.5):
        self.device = device
        self.threshold = threshold
        self.vision_dim = vision_dim
        self.physics_dim = physics_dim
        self.action_dim = action_dim
        self.evidence_dim = evidence_dim
        self.model = RecoveryScorerMLP(vision_dim, physics_dim, action_dim,
                                       evidence_dim, hidden)
        self.metadata = {}
        self.provenance = dict(schema=FEATURE_SCHEMA, threshold=threshold)
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.model.load_state_dict(ckpt['state_dict'], strict=True)
            self.metadata = ckpt.get('metadata', {})
            self.threshold = float(self.metadata.get('threshold', threshold))
            self.provenance.update(self.metadata)
        self.model.to(device).eval()

    def score(self, vision_feat, physics_feat, action,
              evidence: Optional[np.ndarray] = None) -> dict:
        vf = np.asarray(vision_feat, dtype=np.float32).reshape(-1)
        pf = np.asarray(physics_feat, dtype=np.float32).reshape(-1)
        ac = np.asarray(action, dtype=np.float32).reshape(-1)
        if evidence is None:
            evidence = np.zeros(self.evidence_dim, dtype=np.float32)
        ev = np.asarray(evidence, dtype=np.float32).reshape(-1)
        if vf.shape[0] != self.vision_dim:
            raise ValueError(f'vision_feat dim {vf.shape[0]} != {self.vision_dim}')
        if pf.shape[0] != self.physics_dim:
            raise ValueError(f'physics_feat dim {pf.shape[0]} != {self.physics_dim}')
        if ac.shape[0] != self.action_dim:
            raise ValueError(f'action dim {ac.shape[0]} != {self.action_dim}')
        if ev.shape[0] != self.evidence_dim:
            raise ValueError(f'evidence dim {ev.shape[0]} != {self.evidence_dim}')
        with torch.inference_mode():
            v = torch.from_numpy(vf[None]).to(self.device)
            p = torch.from_numpy(pf[None]).to(self.device)
            a = torch.from_numpy(ac[None]).to(self.device)
            e = torch.from_numpy(ev[None]).to(self.device)
            prob = float(self.model(v, p, a, e)[0].cpu().numpy())
        return dict(
            probability=prob, threshold=self.threshold,
            accept=prob >= self.threshold, schema=FEATURE_SCHEMA,
        )

    def score_candidates(self, vision_feat, physics_feat, candidates,
                         evidence_list: Optional[list] = None):
        """Return one finite score per candidate (list of floats).

        ``evidence_list`` is an optional per-candidate evidence vector list
        (aligned with candidates).  When omitted, all candidates receive a
        zero evidence vector.
        """
        scores = []
        for i, c in enumerate(candidates):
            evidence = evidence_list[i] if evidence_list is not None else None
            res = self.score(vision_feat, physics_feat, c['decode']['action'],
                             evidence)
            scores.append(res['probability'])
        return scores
