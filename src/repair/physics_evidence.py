"""Physics evidence extraction for scorer inputs.

Wraps ``PhysicalVerifier`` to produce a fixed-length numeric evidence vector
that can be appended to scorer features, giving the lightweight MLP direct
access to rule-based physical consistency signals (in-goal support, grasp
reality, action intent vs observed motion) without keeping the --use-v5
independent branch.

Layout (4 dims):
  0: in_goal_phys_score        [0,1]  physical in-goal + support (verifier score)
  1: grasp_phys_score          [0,1]  physical grasp reality (verifier score)
  2: executed_action_alignment [0,1]  last executed action vs observed motion
  3: contact_count_norm        [0,1]  normalized contact point count
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
import math

import numpy as np

from ..verification.physical_verifier import PhysicalVerifier, extract_physics_state


def _unit(vec):
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def extract_physics_evidence(
    diagnostics: Dict[str, Any],
    target: str,
    proposed_action: Optional[List[float]] = None,
    prev_pos: Optional[List[float]] = None,
    curr_pos: Optional[List[float]] = None,
    verifier: Optional[PhysicalVerifier] = None,
    *, executed_action: Optional[List[float]] = None,
) -> np.ndarray:
    """Return a fixed-length physics evidence vector (4 dims, float32).

    Historical scorer feature layout. ``proposed_action`` is retained for old
    callers but deliberately does not affect physical evidence: a candidate's
    consequence has not been observed. Alignment requires an executed action
    and its corresponding endpoints; otherwise the legacy neutral value is
    used. New runtime code uses PhysicalVerifier.evidence with validity flags.
    """
    if verifier is None:
        verifier = PhysicalVerifier()
    state = extract_physics_state(diagnostics, target=target)

    in_goal_res = verifier.verify_in_goal(
        state['object_pos'], state['receptacle_bounds'], state['contact_info']
    )
    grasp_res = verifier.verify_grasp(state['gripper_force'])

    # This is retrospective evidence, not a candidate outcome prediction.
    action_dir = None
    if executed_action is not None and len(executed_action) >= 3:
        action_dir = _unit(executed_action[:3])

    obs_dir = None
    if prev_pos is not None and curr_pos is not None:
        disp = math.dist(prev_pos, curr_pos)
        if disp > 1e-6:
            obs_dir = [(curr_pos[i] - prev_pos[i]) / disp for i in range(3)]

    if action_dir is not None and obs_dir is not None:
        dot = sum(a * b for a, b in zip(action_dir, obs_dir))
        alignment = (dot + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    else:
        alignment = 0.5

    contacts = state['contact_info'] or []
    contact_count_norm = min(len(contacts) / 10.0, 1.0)

    return np.asarray([
        float(in_goal_res.score),
        float(grasp_res.score),
        float(alignment),
        float(contact_count_norm),
    ], dtype=np.float32)
