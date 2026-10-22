"""Vision grounding module: frozen visual encoder for state inference.

Encodes the current camera image and goal-prototype images using a frozen
vision model (V-JEPA / DINOv2), and outputs candidate predicates for the
framework monitor (e.g. in_goal_vision, held_clear_vision).

The encoder is frozen — no gradients, no training at inference time.  A small
MLP can be trained offline (see fusion_mlp.py) to fuse vision + physics scores
into a single verification probability.
"""
from typing import Optional, Dict, Any, List
import numpy as np


class VisualGroundingModule:
    def __init__(
        self,
        encoder_name: str = 'vjepa_2_ac',
        device: str = 'cuda:2',
        feature_dim: int = 256,
    ):
        self.encoder_name = encoder_name
        self.device = device
        self.feature_dim = feature_dim
        self._encoder = None        # lazy-loaded frozen encoder
        self._goal_prototypes: Dict[str, np.ndarray] = {}
        self._stage_prototypes: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Encoder (frozen)
    # ------------------------------------------------------------------
    def _load_encoder(self):
        if self._encoder is None:
            self._encoder = self._make_frozen_encoder()
        return self._encoder

    def _make_frozen_encoder(self):
        """Instantiate the frozen vision encoder.

        Substitute the actual V-JEPA 2-AC / DINOv2 loader here.  The encoder
        must be in eval() mode with grad disabled.
        """
        raise NotImplementedError(
            'Wire the frozen vision encoder (V-JEPA/DINOv2) here. '
            'Return a callable that takes an HxWxC uint8 image and returns '
            'an L2-normalized float32 feature vector of shape [feature_dim].')

    def _encode(self, image: np.ndarray) -> np.ndarray:
        """Encode a single image to a normalized feature vector."""
        enc = self._load_encoder()
        with np.errstate(all='ignore'):
            feat = enc(image)                      # [feature_dim]
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

    # ------------------------------------------------------------------
    # Goal prototypes (for "is the subgoal achieved?" checks)
    # ------------------------------------------------------------------
    def set_goal_prototype(self, subgoal: str, image: np.ndarray) -> None:
        """Store the visual prototype for a subgoal's 'achieved' state."""
        self._goal_prototypes[subgoal] = self._encode(image)

    def set_stage_prototype(self, stage: str, image: np.ndarray) -> None:
        """Store the visual prototype for a task stage (acquire/transfer/...)."""
        self._stage_prototypes[stage] = self._encode(image)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer_in_goal(
        self,
        current_image: np.ndarray,
        goal_prototype_key: str = 'contained',
        threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """Return visual-side inference of whether the goal state is achieved."""
        if goal_prototype_key not in self._goal_prototypes:
            return {'in_goal_vision': None, 'score': None,
                    'reason': f'no_prototype_for_{goal_prototype_key}'}

        curr_feat = self._encode(current_image)
        proto = self._goal_prototypes[goal_prototype_key]
        score = float(np.dot(curr_feat, proto))

        return {
            'in_goal_vision': bool(score > threshold),
            'score': score,
            'threshold': threshold,
            'prototype_key': goal_prototype_key,
        }

    def infer_stage(
        self,
        current_image: np.ndarray,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Infer the current task stage from visual features.

        Compares the current image against per-stage prototypes and returns the
        best-matching stage with a confidence score.  This replaces the
        hard-coded _subgoal() predicate chain.
        """
        if stages is None:
            stages = list(self._stage_prototypes.keys())
        if not stages:
            return {'stage': None, 'confidence': None, 'reason': 'no_stage_prototypes'}

        curr_feat = self._encode(current_image)
        best_stage, best_score = None, -1.0
        scores: Dict[str, float] = {}
        for stage in stages:
            if stage not in self._stage_prototypes:
                continue
            s = float(np.dot(curr_feat, self._stage_prototypes[stage]))
            scores[stage] = s
            if s > best_score:
                best_score, best_stage = s, stage

        return {
            'stage': best_stage,
            'confidence': best_score,
            'all_scores': scores,
        }


def make_grounding(encoder_name='vjepa_2_ac', device='cuda:2') -> VisualGroundingModule:
    return VisualGroundingModule(encoder_name=encoder_name, device=device)
