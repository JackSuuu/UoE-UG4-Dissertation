"""Physics-grounded verification for the VLA repair framework.

Task-agnostic: no object/receptacle names are hard-coded.  Object position,
receptacle bounds, contact info and grasp force are read from the simulator
diagnostics at runtime.  The verifier checks physical consistency (object in
region, grasp real, motion matches action intent) without generating actions.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import math


@dataclass
class PhysicalVerificationResult:
    consistent: bool
    score: float                      # 0..1
    details: Dict[str, Any] = field(default_factory=dict)


class PhysicalVerifier:
    """Verifies state inferences against simulator physics.

    Not a learned predictor.  Reads object position, contact geometry, grasp
    force from diagnostics and checks whether a claimed predicate holds under
    the simulator's own model.  All thresholds are physical constants.
    """

    def __init__(
        self,
        contact_threshold_n: float = 0.5,
        grasp_force_threshold_n: float = 1.0,
        support_margin_m: float = 0.02,
    ):
        self.contact_threshold_n = contact_threshold_n
        self.grasp_force_threshold_n = grasp_force_threshold_n
        self.support_margin_m = support_margin_m

    # ------------------------------------------------------------------
    # 1. Is the object physically inside the receptacle?
    # ------------------------------------------------------------------
    def verify_in_goal(
        self,
        object_pos: Optional[List[float]],
        receptacle_bounds: Optional[Dict[str, List[float]]],
        contact_info: Optional[List[Dict[str, Any]]],
    ) -> PhysicalVerificationResult:
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
        supported = self._is_supported(contact_info, object_pos)

        if inside and supported:
            score = 1.0
        elif inside:
            score = 0.5
        else:
            score = 0.0

        return PhysicalVerificationResult(
            consistent=(inside and supported), score=score,
            details={
                'object_pos': object_pos,
                'inside_bounds': inside,
                'supported_by_receptacle': supported,
                'contact_points': len(contact_info or []),
            })

    # ------------------------------------------------------------------
    # 2. Is the grasp physically real?
    # ------------------------------------------------------------------
    def verify_grasp(
        self,
        gripper_force: Optional[float],
    ) -> PhysicalVerificationResult:
        if gripper_force is None:
            return PhysicalVerificationResult(
                consistent=False, score=0.0,
                details={'reason': 'missing_grasp_force'})
        real_grasp = gripper_force >= self.grasp_force_threshold_n
        return PhysicalVerificationResult(
            consistent=real_grasp, score=min(1.0, gripper_force / 5.0),
            details={'gripper_force_n': gripper_force})

    # ------------------------------------------------------------------
    # 3. Does observed motion match action intent?
    # ------------------------------------------------------------------
    def verify_action_physics_consistency(
        self,
        proposed_action: Optional[List[float]],
        prev_pos: Optional[List[float]],
        curr_pos: Optional[List[float]],
    ) -> PhysicalVerificationResult:
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

        return PhysicalVerificationResult(
            consistent=(consistency_score > 0.25), score=consistency_score,
            details={
                'displacement_m': displacement,
                'cosine_similarity': dot,
            })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _is_supported(
        self,
        contact_info: Optional[List[Dict[str, Any]]],
        object_pos: Optional[List[float]],
    ) -> bool:
        if not contact_info or object_pos is None:
            return False
        oz = object_pos[2]
        for contact in contact_info:
            cp = contact.get('contact_point')
            if cp is not None and len(cp) >= 3:
                if cp[2] <= oz + self.support_margin_m:
                    return True
            elif contact.get('normal') is not None:
                return True
        return False


def extract_physics_state(diagnostics: Dict[str, Any], target: Optional[str] = None) -> Dict[str, Any]:
    """Extract a bound object's absolute position; ambiguous evidence stays missing."""
    obj_obs = diagnostics.get('object_observations', {}) or {}
    object_pos = None
    candidates = [key for key in obj_obs if key.endswith('_pos') and '_to_' not in key]
    selected = f'{target}_pos' if target else candidates[0] if len(candidates) == 1 else None
    for key, val in obj_obs.items():
        if key != selected:
            continue
        if isinstance(val, dict) and 'pos' in val:
            try:
                object_pos = [float(v) for v in val['pos']]
                if len(object_pos) == 3 and all(math.isfinite(v) for v in object_pos):
                    break
            except (TypeError, ValueError):
                object_pos = None
        elif isinstance(val, (list, tuple)) and len(val) == 3:
            try:
                object_pos = [float(v) for v in val]
                if all(math.isfinite(v) for v in object_pos):
                    break
            except (TypeError, ValueError):
                object_pos = None

    if object_pos is not None and (len(object_pos) != 3 or not all(math.isfinite(v) for v in object_pos)):
        object_pos = None
    grasped = diagnostics.get('grasped', {})
    grasp = grasped.get(target) if isinstance(grasped, dict) and target else None

    receptacle_bounds = diagnostics.get('receptacle_bounds')
    contact_info = diagnostics.get('contacts_endpoint')
    if not isinstance(contact_info, list):
        contact_info = None

    grasp_force = None
    gf = diagnostics.get('grasp_force')
    if gf is not None:
        try:
            grasp_force = float(gf)
        except (TypeError, ValueError):
            grasp_force = None

    return {
        'object_pos': object_pos,
        'receptacle_bounds': receptacle_bounds,
        'contact_info': contact_info,
        'gripper_force': grasp_force,
        'grasp': grasp if isinstance(grasp, bool) else None,
    }
