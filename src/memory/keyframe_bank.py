"""Keyframe bank for episodic context (CheckVLA Module 4, lightweight).

Stores visual features + stage labels from successful trajectory steps.
When entering a similar stage, reads back the stage's "typical success"
visual prototype to augment VisualGroundingModule's stage inference.

Task-agnostic: no object/receptacle names hard-coded.
"""
from typing import Optional, Dict, Any, List
import numpy as np


class KeyframeBank:
    """Fixed-capacity bank of (visual_feature, stage_label, success_flag)."""

    def __init__(self, capacity: int = 200, feature_dim: int = 384):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self._entries: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write: add a keyframe when a stage is successfully achieved
    # ------------------------------------------------------------------
    def write(
        self,
        visual_feature: np.ndarray,
        stage: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a keyframe entry.  FIFO eviction when at capacity."""
        if visual_feature is None:
            return
        feat = np.asarray(visual_feature, dtype=np.float32).reshape(-1)
        if feat.shape[0] != self.feature_dim:
            # Pad or truncate to match expected dim
            if feat.shape[0] < self.feature_dim:
                feat = np.pad(feat, (0, self.feature_dim - feat.shape[0]))
            else:
                feat = feat[:self.feature_dim]

        self._entries.append({
            'feature': feat / (np.linalg.norm(feat) + 1e-8),  # L2-normalize
            'stage': stage,
            'success': bool(success),
            'metadata': metadata or {},
        })

        if len(self._entries) > self.capacity:
            self._entries.pop(0)  # FIFO

    # ------------------------------------------------------------------
    # Read: get prototype feature for a stage (from successful entries)
    # ------------------------------------------------------------------
    def read_prototype(self, stage: str) -> Optional[np.ndarray]:
        """Return the mean normalized feature of successful entries for a stage."""
        stage_feats = [
            e['feature'] for e in self._entries
            if e['stage'] == stage and e['success']
        ]
        if not stage_feats:
            return None
        proto = np.mean(np.stack(stage_feats), axis=0)
        norm = np.linalg.norm(proto) + 1e-8
        return (proto / norm).astype(np.float32)

    # ------------------------------------------------------------------
    # Read: nearest neighbor prototypes for a stage (top-k)
    # ------------------------------------------------------------------
    def read_topk(
        self,
        query_feature: np.ndarray,
        stage: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return top-k most similar successful entries for a stage."""
        if query_feature is None:
            return []
        q = np.asarray(query_feature, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.feature_dim:
            if q.shape[0] < self.feature_dim:
                q = np.pad(q, (0, self.feature_dim - q.shape[0]))
            else:
                q = q[:self.feature_dim]
        q = q / (np.linalg.norm(q) + 1e-8)

        candidates = [
            e for e in self._entries
            if e['stage'] == stage and e['success']
        ]
        if not candidates:
            return []

        scored = []
        for e in candidates:
            sim = float(np.dot(q, e['feature']))
            scored.append((sim, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                'similarity': sim,
                'stage': e['stage'],
                'metadata': e['metadata'],
            }
            for sim, e in scored[:k]
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stage_counts(self) -> Dict[str, int]:
        """Count of successful entries per stage."""
        counts: Dict[str, int] = {}
        for e in self._entries:
            if e['success']:
                counts[e['stage']] = counts.get(e['stage'], 0) + 1
        return counts

    def size(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        self._entries = []


def make_keyframe_bank(
    capacity: int = 200,
    feature_dim: int = 384,
) -> KeyframeBank:
    return KeyframeBank(capacity=capacity, feature_dim=feature_dim)
