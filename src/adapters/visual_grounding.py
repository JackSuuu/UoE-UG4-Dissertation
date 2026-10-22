"""Vision grounding: frozen encoder for state inference.

Task-agnostic: goal/stage prototypes are registered per-task at experiment
setup.  No object/receptacle names are hard-coded.
"""
from typing import Optional, Dict, Any, List
import warnings
import numpy as np


class VisualGroundingModule:
    def __init__(
        self,
        encoder_name: str = 'dinov2_vits14',
        device: str = 'cuda:2',
        feature_dim: int = 384,
    ):
        self.encoder_name = encoder_name
        self.device = device
        self.feature_dim = feature_dim
        self._encoder_fn = None
        self._goal_prototypes: Dict[str, np.ndarray] = {}
        self._stage_prototypes: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Encoder (frozen)
    # ------------------------------------------------------------------
    def _load_encoder(self):
        if self._encoder_fn is not None:
            return self._encoder_fn
        self._encoder_fn = self._make_frozen_encoder()
        return self._encoder_fn

    def _make_frozen_encoder(self):
        try:
            import torch
            import torchvision
            import torch.nn.functional as F

            model = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14',
                pretrained=True, force_reload=False,
            )
            model.to(self.device)
            model.eval()

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

            def encode(image: np.ndarray) -> np.ndarray:
                img = image.astype(np.float32) / 255.0
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
                t = (t - mean) / std
                with torch.no_grad():
                    feat = model(t)
                feat = feat.cpu().numpy().reshape(-1).astype(np.float32)
                norm = np.linalg.norm(feat) + 1e-8
                return feat / norm

            self.feature_dim = 384
            return encode

        except Exception:
            warnings.warn(
                'VisualGroundingModule: DINOv2 unavailable. '
                'Using deterministic random projection (smoke test only). '
                'Wire a real encoder before experiments.',
                stacklevel=2,
            )
            rng = np.random.RandomState(42)
            proj = rng.randn(224 * 224 * 3, self.feature_dim).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

            def encode(image: np.ndarray) -> np.ndarray:
                small = image[::8, ::8, :].astype(np.float32).reshape(-1)
                target = 224 * 224 * 3
                if len(small) < target:
                    small = np.pad(small, (0, target - len(small)))
                else:
                    small = small[:target]
                feat = (proj * small[:, None]).sum(axis=0)
                norm = np.linalg.norm(feat) + 1e-8
                return (feat / norm).astype(np.float32)

            return encode

    def _encode(self, image: np.ndarray) -> np.ndarray:
        enc = self._load_encoder()
        feat = enc(image)
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

    # ------------------------------------------------------------------
    # Prototypes
    # ------------------------------------------------------------------
    def set_goal_prototype(self, subgoal: str, image: np.ndarray) -> None:
        self._goal_prototypes[subgoal] = self._encode(image)

    def set_stage_prototype(self, stage: str, image: np.ndarray) -> None:
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
        }

    def infer_stage(
        self,
        current_image: np.ndarray,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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


def make_grounding(
    encoder_name: str = 'dinov2_vits14',
    device: str = 'cuda:2',
) -> VisualGroundingModule:
    return VisualGroundingModule(encoder_name=encoder_name, device=device)
