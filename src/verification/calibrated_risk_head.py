"""Learned observed-regression risk with an empirical validation threshold.

No random/untrained fallback and no episode-level/conformal guarantee is claimed.
"""
from .fusion_mlp import FusionMLP


class CalibratedRiskHead:
    def __init__(self, checkpoint, *, encoder_signature, device='cpu'):
        self.head = FusionMLP.load(checkpoint, role='risk', encoder_signature=encoder_signature, device=device)
        self.provenance = self.head.provenance

    def score(self, features):
        value = self.head.probability(features)
        return dict(score=value, threshold=self.head.threshold, exceeds=value > self.head.threshold,
                    calibration='empirical_validation_not_conformal_coverage')
