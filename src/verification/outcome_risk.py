"""Versioned outcome-trained risk model; empirical, not conformal coverage."""
import numpy as np
import torch

from .calibrated_risk_head import RiskHeadMLP, extract_trajectory_stats
from .fusion_mlp import STAGE_TO_IDX, extract_physics_features

RISK_SCHEMA = 'masked_state24_history8_stage4_observed_nominal24_v1'


def temporal_features(history, stage):
    # History consists of parsed controller states, not raw diagnostic dicts.
    enriched = [dict(s, contacts_endpoint=[None] * sum(s.get(k) is True for k in
        ('table', 'mug_microwave', 'mug_gripper', 'gripper_microwave'))) for s in history[-12:]]
    return np.concatenate([extract_trajectory_stats(enriched), np.eye(4, dtype=np.float32)[STAGE_TO_IDX[stage]]])


class OutcomeRisk:
    def __init__(self, checkpoint, device='cpu'):
        artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if artifact.get('risk_schema') != RISK_SCHEMA:
            raise ValueError('Untrained/unversioned joint risk artifact')
        self.artifact = artifact
        self.model = RiskHeadMLP(vision_dim=384, physics_dim=24, stats_dim=12).to(device)
        self.model.load_state_dict(artifact['risk_state_dict'], strict=True)
        self.model.eval()
        self.device = device
        self.threshold = float(artifact['risk_threshold'])
        self.mean = artifact['input_mean'].numpy()
        self.std = artifact['input_std'].numpy()

    def features(self, state, history, stage):
        p = extract_physics_features(state)
        t = temporal_features(history, stage)
        x = np.concatenate([p[:12], t[:8]])
        x = (x-self.mean)/self.std
        return np.concatenate([x[:12]*p[12:], p[12:]]).astype(np.float32), np.concatenate([x[12:], t[8:]]).astype(np.float32)

    def score(self, visual, state, history, stage):
        p, t = self.features(state, history, stage)
        with torch.inference_mode():
            v, p, t = [torch.from_numpy(np.asarray(x, dtype=np.float32)[None]).to(self.device) for x in (visual, p, t)]
            value = float(self.model(v, p, t).sigmoid()[0, 0])
        return dict(score=value, threshold=self.threshold, exceeds=value > self.threshold,
            schema=RISK_SCHEMA, calibration='empirical_validation_negative_sample_quantile_not_coverage')
