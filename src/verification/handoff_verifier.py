"""Causal frozen-encoder + masked-MLP scoring shared by logging and control arms."""
import hashlib
import numpy as np

from ..perception.frozen_visual_encoder import FrozenVisualEncoder, file_sha
from .fusion_mlp import FEATURE_SCHEMA, extract_physics_features, load_model


class HandoffVerifier:
    def __init__(self, checkpoint, device='cpu'):
        self.model = load_model(checkpoint, device=device)
        metadata = self.model.metadata
        threshold = metadata.get('threshold')
        if (not metadata.get('control_ready') or threshold is None or
                not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('No usable validation-selected threshold in fusion artifact')
        encoder = metadata['encoder']
        if not encoder.get('pretrained') or encoder.get('transform') != FrozenVisualEncoder.transform:
            raise ValueError('Real policy-RGB encoder provenance required')
        self.encoder = FrozenVisualEncoder(encoder['checkpoint'], encoder['source'],
            device=device, expected_sha=encoder['sha256'])
        if self.encoder.provenance != encoder:
            raise ValueError('Visual preprocessing/source provenance differs from training')
        self.threshold = float(threshold)
        self.model_sha256 = file_sha(checkpoint)

    def score(self, policy_rgb, state, stage):
        if stage not in self.model.metadata.get('supported_stages', []):
            return dict(probability=0., threshold=self.threshold, accepted=False,
                stage=stage, supported=False, reason='stage_has_no_outcome_training_support',
                feature_schema=FEATURE_SCHEMA, model_sha256=self.model_sha256)
        rgb = np.asarray(policy_rgb)
        visual = self.encoder.encode(rgb)
        physics = extract_physics_features(state)
        probability = self.model.predict_verify_probability(visual, physics, stage)
        return dict(probability=probability, threshold=self.threshold,
            accepted=probability >= self.threshold, stage=stage,
            feature_schema=FEATURE_SCHEMA, model_sha256=self.model_sha256,
            encoder_sha256=self.encoder.provenance['sha256'],
            policy_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            physics_features=physics.tolist(), visual_feature_norm=float(np.linalg.norm(visual)))
