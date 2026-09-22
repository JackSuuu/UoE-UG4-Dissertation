"""
mlp_policy.py
-------------
A lightweight MLP policy used as the BC baseline.

Architecture: Linear(obs_dim, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, act_dim) → Tanh
The Tanh output is scaled by MAX_VEL so actions are in [-MAX_VEL, MAX_VEL].
"""

import torch
import torch.nn as nn
from env.push_env import MAX_VEL


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 7, act_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )
        self.scale = MAX_VEL

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.scale

    @torch.no_grad()
    def predict(self, obs_np):
        """NumPy in → NumPy out. Works with single observation or batch."""
        import numpy as np
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        act_t = self.forward(obs_t)
        return act_t.numpy()
