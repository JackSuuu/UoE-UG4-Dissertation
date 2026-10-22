"""Physical simulation verification for the VLA repair framework.

This module adds *real physics-grounded verification* on top of the existing
privileged-predicate monitor.  Instead of only reading boolean flags from the
simulator diagnostics, it queries the simulator's physical state (object
position, contact normals, grasp forces, receptacle bounds) and checks whether
a claimed predicate (e.g. "mug is inside microwave") holds under the
simulator's own collision / kinematic model.

On real hardware the privileged physics reads are unavailable; the same checks
degrade to vision-only consistency scores (see VisualGroundingModule).
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import math


@dataclass
class PhysicalVerificationResult:
    consistent: bool
    score: float                      # 0..1, higher = more consistent with physics
    details: Dict[str, Any] = field(default_factory=dict)


class PhysicalVerifier:
    """Verifies state inferences against simulator physics.

    This is NOT a learned predictor.  It reads the current privileged physics
    state (object position, contact geometry, grasp force) and checks whether
    the claimed predicate holds under the simulator's own model.  All thresholds
    are physical constants (meters, Newtons), not learned weights.
    """

    def __init__(
        self,
        target: str = 'white_yellow_mug_1',
        receptacle: str = 'microwave_1',
        region: str = 'microwave_1_heating_region',
        contact_threshold_n: float = 0.5,
        grasp_force_threshold_n: float = 1.0,
        support_margin_m: float = 0.02,
    ):
        self.target = target
        self.receptacle = receptacle
        self.region = region
        self.contact_threshold_n = contact_threshold_n
        self.grasp_force_threshold_n = grasp_force_threshold_n
        self.support_margin_m = support_margin_m

    # ------------------------------------------------------------------
    # 1. Is the object physically inside the target region?
    # ------------------------------------------------------------------
    def verify_in_goal(
        self,
        object_pos: Optional[List[float]],
        receptacle_bounds: Optional[Dict[str, List[float]]],
        contact_info: Optional[List[Dict[str, Any]]],
    ) -> PhysicalVerificationResult:
        """Check object position against receptacle bounding volume + contact support.

        Returns consistent=True only if the object is inside the receptacle bounds
        AND physically supported by it (not just overlapping the bounding box while
        floating above it).
        """
        if object_pos is None or receptacle_bounds is None:
            return PhysicalVerificationResult(
                consistent=False, score=0.0,
                details={'reason': 'missing_object_or_receptacle_state'})

        ox, oy, oz = object_pos
        minx, miny, minz = receptacle_bounds['min']
        maxx, maxy, maxz = receptacle_bounds['max']

        inside = (
            minx <= ox <= maxx and
            miny <= oy <= maxy and
            minz <= oz <= maxz + self.support_margin_m
        )

        supported = self._is_supported_by_receptacle(contact_info, object_pos)

        # Score: 1.0 = inside + supported, 0.5 = inside only, 0.0 = outside
        if inside and supported:
            score = 1.0
        elif inside:
            score = 0.5
        else:
            score = 0.0

        consistent = inside and supported
        return PhysicalVerificationResult(
            consistent=consistent, score=score,
            details={
                'object_pos': object_pos,
                'inside_bounds': inside,
                'supported_by_receptacle': supported,
                'contact_points': len(contact_info or []),
                'receptacle_bounds': receptacle_bounds,
            })

    # ------------------------------------------------------------------
    # 2. Is the grasp physically real?
    # ------------------------------------------------------------------
    def verify_grasp(
        self,
        gripper_force: Optional[float],
        object_contact_normal: Optional[List[float]],
    ) -> PhysicalVerificationResult:
        """Check whether the gripper is actually exerting force on the object."""
        if gripper_force is None:
            return PhysicalVerificationResult(
                consistent=False, score=0.0,
                details={'reason': 'missing_grasp_force'})
        real_grasp = gripper_force >= self.grasp_force_threshold_n
        score = min(1.0, gripper_force / 5.0)
        return PhysicalVerificationResult(
            consistent=real_grasp, score=score,
            details={'gripper_force_n': gripper_force})

    # ------------------------------------------------------------------
    # 3. Does observed object motion match the proposed action intent?
    # ------------------------------------------------------------------
    def verify_action_physics_consistency(
        self,
        proposed_action: Optional[List[float]],
        prev_pos: Optional[List[float]],
        curr_pos: Optional[List[float]],
        dt: float = 0.05,
    ) -> PhysicalVerificationResult:
        """Detect "VLA pushes in but object moves out" regression.

        Compares the first 3 dims of the proposed action (translational intent)
        with the observed object displacement direction.  A low cosine similarity
        means the physics contradicts the action intent — a strong signal that
        the repair is not producing the desired physical effect.
        """
        if proposed_action is None or prev_pos is None or curr_pos is None:
            return PhysicalVerificationResult(
                consistent=True, score=1.0,
                details={'reason': 'insufficient_position_history'})

        displacement = math.dist(prev_pos, curr_pos)
        intent = proposed_action[:3]
        intent_norm = math.sqrt(sum(v * v for v in intent)) or 1.0
        intent_dir = [v / intent_norm for v in intent]

        if displacement < 1e-6:
            obs_dir = [0.0, 0.0, 0.0]
        else:
            obs_dir = [(curr_pos[i] - prev_pos[i]) / displacement for i in range(3)]

        dot = sum(intent_dir[i] * obs_dir[i] for i in range(3))
        consistency_score = max(0.0, min(1.0, (dot + 1.0) / 2.0))

        # Tolerant threshold: only flag strong contradictions (outward motion when
        # inward intent, etc.)
        consistent = consistency_score > 0.25
        return PhysicalVerificationResult(
            consistent=consistent, score=consistency_score,
            details={
                'displacement_m': displacement,
                'intent_dir': intent_dir,
                'observed_dir': obs_dir,
                'cosine_similarity': dot,
            })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_supported_by_receptacle(
        self,
        contact_info: Optional[List[Dict[str, Any]]],
        object_pos: Optional[List[float]],
    ) -> bool:
        if not contact_info or object_pos is None:
            return False
        oz = object_pos[2]
        for contact in contact_info:
            geoms = {contact.get('geom1', ''), contact.get('geom2', '')}
            if any(g.startswith(self.receptacle + '_') for g in geoms):
                # Contact exists with receptacle geometry.  Check that the contact
                # point is at or below the object's top surface (i.e. the object is
                # resting on the receptacle, not merely touching its side while held).
                cp = contact.get('contact_point')
                if cp is not None and len(cp) >= 3:
                    if cp[2] <= oz + self.support_margin_m:
                        return True
                else:
                    # No contact point available; conservatively count receptacle
                    # contact as support.
                    return True
        return False


# Singleton-style helper for the runner to instantiate once per episode.
def make_verifier(target='white_yellow_mug_1', receptacle='microwave_1',
                  region='microwave_1_heating_region') -> PhysicalVerifier:
    return PhysicalVerifier(target=target, receptacle=receptacle, region=region)
