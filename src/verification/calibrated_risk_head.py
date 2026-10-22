"""Calibrated risk head for repair triggering (CheckVLA Module 2).

Trains a small MLP risk head on success/failure trajectories to output a risk
score r_t ∈ [0,1] for the current step.  Then applies *functional conformal
calibration* on held-out success trajectories to fit a dynamic threshold
δ_t such that normal (non-intervention-required) trajectories have false
intervention rate < α (default 5%).

Not a world model — just a risk scorer over current + recent trajectory stats.
"""
from typing import Optional, Dict, Any, List
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RiskHeadMLP(nn.Module if HAS_TORCH else object):
    """Small MLP: vision + physics + trajectory stats -> risk logit."""

    def __init__(
        self,
        vision_dim: int = 384,
        physics_dim: int = 12,
        stats_dim: int = 8,
        hidden: int = 64,
    ):
        if not HAS_TORCH:
            raise RuntimeError('PyTorch is required for RiskHeadMLP')
        super().__init__()
        input_dim = vision_dim + physics_dim + stats_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),  # single risk logit
        )

    def forward(self, vision_feat, physics_feat, traj_stats):
        x = torch.cat([vision_feat, physics_feat, traj_stats], dim=-1)
        return self.net(x)


def extract_trajectory_stats(history: List[Dict[str, Any]]) -> np.ndarray:
    """Compute recent trajectory statistics for risk input.

    Features:
      [0] mean |velocity| over last window (m/step)
      [1] max |velocity| over last window
      [2] fraction of last N steps with in_goal=True
      [3] fraction of last N steps with grasp=True
      [4] steps since last grasp (normalized)
      [5] steps since last in_goal (normalized)
      [6] position variance over last window (m^2)
      [7] contact count over last window
    """
    if not history:
        return np.zeros(8, dtype=np.float32)

    window = min(12, len(history))
    recent = history[-window:]

    # Velocities
    velocities = []
    positions = []
    for h in recent:
        pos = h.get('position')
        if pos is not None and len(pos) == 3:
            positions.append(np.array(pos, dtype=np.float32))
    if len(positions) >= 2:
        diffs = [np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions))]
        mean_vel = float(np.mean(diffs))
        max_vel = float(np.max(diffs))
        pos_var = float(np.var([p[0] for p in positions]) + np.var([p[1] for p in positions]) + np.var([p[2] for p in positions]))
    else:
        mean_vel = 0.0
        max_vel = 0.0
        pos_var = 0.0

    # Predicate fractions
    in_goal_frac = float(np.mean([1.0 if h.get('in_goal') is True else 0.0 for h in recent]))
    grasp_frac = float(np.mean([1.0 if h.get('grasp') is True else 0.0 for h in recent]))

    # Steps since last events (normalized by window)
    steps_since_grasp = window
    steps_since_in_goal = window
    for i in range(len(recent) - 1, -1, -1):
        if recent[i].get('grasp') is True:
            steps_since_grasp = window - 1 - i
            break
    for i in range(len(recent) - 1, -1, -1):
        if recent[i].get('in_goal') is True:
            steps_since_in_goal = window - 1 - i
            break

    # Contact count
    contact_count = 0.0
    for h in recent:
        contacts = h.get('contacts_endpoint')
        if isinstance(contacts, list):
            contact_count += len(contacts)

    return np.array([
        mean_vel,
        max_vel,
        in_goal_frac,
        grasp_frac,
        float(steps_since_grasp) / float(window),
        float(steps_since_in_goal) / float(window),
        pos_var,
        float(contact_count),
    ], dtype=np.float32)


class CalibratedRiskHead:
    """Risk head with conformal calibration.

    Workflow:
      1. Train RiskHeadMLP on labeled (state -> risk) pairs from trajectories.
      2. Calibrate threshold δ on held-out success trajectories using conformal
         method: choose δ such that fraction of success steps with risk > δ
         is ≤ α.
      3. At inference: compute r_t, trigger repair if r_t > δ_t.
    """

    def __init__(
        self,
        vision_dim: int = 384,
        physics_dim: int = 12,
        stats_dim: int = 8,
        hidden: int = 64,
        alpha: float = 0.05,
    ):
        if not HAS_TORCH:
            raise RuntimeError('PyTorch is required for CalibratedRiskHead')
        self.alpha = alpha
        self.threshold: Optional[float] = None
        self.calibrated = False

        self.model = RiskHeadMLP(
            vision_dim=vision_dim,
            physics_dim=physics_dim,
            stats_dim=stats_dim,
            hidden=hidden,
        )

    # ------------------------------------------------------------------
    # Training (supervised on labeled pairs)
    # ------------------------------------------------------------------
    def train_on_pairs(
        self,
        pairs: List[Dict[str, Any]],
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        device='cpu',
    ) -> List[float]:
        """Train on (state, label) pairs where label=1 means 'needs repair'.

        Each pair dict must contain:
          - 'vision_feat': np.ndarray [vision_dim]
          - 'physics_feat': np.ndarray [physics_dim]
          - 'traj_stats': np.ndarray [stats_dim]
          - 'label': float in {0.0, 1.0}
        """
        import torch.optim as optim

        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        X_v = torch.from_numpy(np.stack([p['vision_feat'] for p in pairs])).to(device)
        X_p = torch.from_numpy(np.stack([p['physics_feat'] for p in pairs])).to(device)
        X_s = torch.from_numpy(np.stack([p['traj_stats'] for p in pairs])).to(device)
        Y = torch.from_numpy(np.array([p['label'] for p in pairs], dtype=np.float32)).unsqueeze(1).to(device)

        n = len(pairs)
        losses: List[float] = []
        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                logits = self.model(X_v[idx], X_p[idx], X_s[idx])
                loss = criterion(logits, Y[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            losses.append(epoch_loss / n)
        return losses

    # ------------------------------------------------------------------
    # Conformal calibration on success trajectories
    # ------------------------------------------------------------------
    def calibrate(
        self,
        success_trajectory_risks: List[float],
        alpha: Optional[float] = None,
    ) -> float:
        """Fit threshold δ using conformal calibration on success risks.

        Given risk scores from steps in *successful* trajectories (where no
        repair was needed), choose δ as the (1-α)-quantile of the risk scores
        (plus the standard conformal slack term).  This guarantees that, on
        exchangeable success data, the false intervention rate is ≤ α.

        The conformal quantile: sort risks, take the ceil((1-α)(n+1))-th
        smallest value.
        """
        if alpha is not None:
            self.alpha = alpha

        risks = np.array(success_trajectory_risks, dtype=np.float64)
        if len(risks) == 0:
            self.threshold = 1.0
            self.calibrated = True
            return self.threshold

        risks_sorted = np.sort(risks)
        n = len(risks_sorted)
        # Conformal quantile level
        q_level = min(1.0, (1.0 - self.alpha) * (n + 1) / n)
        idx = int(np.ceil(q_level * n)) - 1
        idx = max(0, min(idx, n - 1))
        self.threshold = float(risks_sorted[idx])
        self.calibrated = True
        return self.threshold

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def compute_risk(
        self,
        vision_feat: np.ndarray,
        physics_feat: np.ndarray,
        traj_stats: np.ndarray,
        device='cpu',
    ) -> float:
        """Return risk score r_t ∈ [0,1] for the current step."""
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            vf = torch.from_numpy(vision_feat.astype(np.float32)).unsqueeze(0).to(device)
            pf = torch.from_numpy(physics_feat.astype(np.float32)).unsqueeze(0).to(device)
            ts = torch.from_numpy(traj_stats.astype(np.float32)).unsqueeze(0).to(device)
            logit = self.model(vf, pf, ts)
            risk = torch.sigmoid(logit).item()
        return float(risk)

    def should_trigger(self, risk: float) -> bool:
        """Return True if risk exceeds the calibrated threshold."""
        if self.threshold is None:
            return False
        return bool(risk > self.threshold)

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'alpha': self.alpha,
            'calibrated': self.calibrated,
            'vision_dim': self.model.vision_dim if hasattr(self.model, 'vision_dim') else 384,
            'physics_dim': self.model.physics_dim if hasattr(self.model, 'physics_dim') else 12,
            'stats_dim': 8,
            'hidden': 64,
        }, path)

    @classmethod
    def load(cls, path: str, device='cpu') -> 'CalibratedRiskHead':
        checkpoint = torch.load(path, map_location=device)
        instance = cls(
            vision_dim=checkpoint.get('vision_dim', 384),
            physics_dim=checkpoint.get('physics_dim', 12),
            stats_dim=checkpoint.get('stats_dim', 8),
            hidden=checkpoint.get('hidden', 64),
            alpha=checkpoint.get('alpha', 0.05),
        )
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.to(device)
        instance.model.eval()
        instance.threshold = checkpoint.get('threshold')
        instance.calibrated = checkpoint.get('calibrated', False)
        return instance


def make_risk_head(
    vision_dim: int = 384,
    physics_dim: int = 12,
    stats_dim: int = 8,
    hidden: int = 64,
    alpha: float = 0.05,
) -> CalibratedRiskHead:
    return CalibratedRiskHead(
        vision_dim=vision_dim,
        physics_dim=physics_dim,
        stats_dim=stats_dim,
        hidden=hidden,
        alpha=alpha,
    )
