"""Lightweight action suffix correction (CheckVLA Module 3, lightweight).

When the VLA's proposed action contradicts physical evidence (e.g. pushes
outward while the object should move inward), apply a minimal corrective
offset to the translational components.  This is NOT full trajectory
rewriting — it is a bounded, interpretable safety nudge that preserves the
VLA's grasp channel and rotational intent.

Design constraints:
- Only the first 3 dims (translation) are adjusted.
- Gripper channel (last dim) is never touched.
- Correction magnitude is bounded (max 0.1 m per step).
- Applied only when action-physics consistency is low.
"""
from typing import Optional, List
import math
import numpy as np


class ActionSuffixCorrector:
    """Corrects VLA action proposals when they contradict physical evidence."""

    def __init__(
        self,
        max_correction_m: float = 0.10,
        consistency_threshold: float = 0.25,
        correction_blend: float = 0.5,
    ):
        self.max_correction_m = max_correction_m
        self.consistency_threshold = consistency_threshold
        self.correction_blend = correction_blend

    def correct(
        self,
        proposed_action: List[float],
        action_physics_consistent: Optional[bool],
        action_physics_score: Optional[float],
        intent_vs_observed_cosine: Optional[float] = None,
    ) -> List[float]:
        """Return corrected action (copy, never mutates input).

        If action-physics consistency is below threshold, nudge the
        translational components away from the contradictory direction.
        """
        corrected = list(proposed_action)

        if action_physics_consistent is True:
            return corrected

        # Only correct if we have evidence of contradiction
        need_correction = False
        if action_physics_consistent is False:
            need_correction = True
        elif action_physics_score is not None and action_physics_score < self.consistency_threshold:
            need_correction = True
        elif intent_vs_observed_cosine is not None and intent_vs_observed_cosine < 0.0:
            need_correction = True

        if not need_correction:
            return corrected

        # Compute correction: reverse the translational intent partially
        # Scale by how contradictory the signal is
        contradiction = 1.0
        if intent_vs_observed_cosine is not None:
            contradiction = max(0.0, -intent_vs_observed_cosine)  # 0 = aligned, 1 = opposite
        elif action_physics_score is not None:
            contradiction = max(0.0, 1.0 - action_physics_score)

        correction_magnitude = self.max_correction_m * self.correction_blend * contradiction

        # Apply opposite correction to translation (first 3 dims)
        for i in range(3):
            original = corrected[i]
            # Push opposite to the proposed direction
            corrected[i] = original - np.sign(original) * correction_magnitude if abs(original) > 1e-6 else 0.0

        # Clamp to valid action range (typical VLA action bounds)
        for i in range(3):
            corrected[i] = float(np.clip(corrected[i], -1.0, 1.0))

        # Gripper channel (dim 6) and rotation (dims 3-5) remain unchanged
        return corrected

    def correct_with_physics_state(
        self,
        proposed_action: List[float],
        prev_pos: Optional[List[float]],
        curr_pos: Optional[List[float]],
        receptacle_bounds: Optional[dict],
    ) -> List[float]:
        """Correct using observed motion vs action intent.

        If the object moved opposite to the action's translational intent
        (e.g. VLA pushes in but object moved out), apply corrective offset.
        """
        corrected = list(proposed_action)

        if prev_pos is None or curr_pos is None:
            return corrected

        # Observed displacement
        displacement = np.array(curr_pos, dtype=np.float32) - np.array(prev_pos, dtype=np.float32)
        disp_norm = np.linalg.norm(displacement)
        if disp_norm < 1e-6:
            return corrected

        obs_dir = displacement / disp_norm

        # Action intent direction (translation only)
        intent = np.array(proposed_action[:3], dtype=np.float32)
        intent_norm = np.linalg.norm(intent)
        if intent_norm < 1e-6:
            return corrected
        intent_dir = intent / intent_norm

        # Cosine similarity
        cosine = float(np.dot(intent_dir, obs_dir))

        # If cosine < 0, object moved opposite to intent -> correct
        if cosine >= 0.0:
            return corrected

        contradiction = -cosine  # 0 = aligned, 1 = fully opposite
        correction_magnitude = self.max_correction_m * self.correction_blend * contradiction

        # Nudge translation opposite to proposed intent
        for i in range(3):
            corrected[i] = float(np.clip(
                corrected[i] - np.sign(corrected[i]) * correction_magnitude if abs(corrected[i]) > 1e-6 else 0.0,
                -1.0, 1.0,
            ))

        return corrected


def make_action_suffix_corrector(
    max_correction_m: float = 0.10,
    consistency_threshold: float = 0.25,
    correction_blend: float = 0.5,
) -> ActionSuffixCorrector:
    return ActionSuffixCorrector(
        max_correction_m=max_correction_m,
        consistency_threshold=consistency_threshold,
        correction_blend=correction_blend,
    )
