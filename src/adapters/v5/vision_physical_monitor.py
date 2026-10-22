"""Vision + physical joint monitor.

Replaces the predicate-only _monitor() in the frozen native_framework.  A
repair trigger fires only when the visual anomaly AND the physical verifier
agree, which suppresses false interventions (e.g. root3's release-transition
false alarm) and prevents "visually in but physically out" handoffs.

Additionally exposes an action-physics consistency check that catches the
root4 failure mode: VLA proposes inward motion but the object physically
moves outward.
"""
from typing import Optional, Dict, Any, List
from .physical_verifier import PhysicalVerifier, PhysicalVerificationResult
from .visual_grounding import VisualGroundingModule


class VisionPhysicalMonitor:
    def __init__(
        self,
        verifier: PhysicalVerifier,
        grounding: VisualGroundingModule,
        vision_threshold: float = 0.75,
        action_physics_threshold: float = 0.25,
    ):
        self.verifier = verifier
        self.grounding = grounding
        self.vision_threshold = vision_threshold
        self.action_physics_threshold = action_physics_threshold
        self._prev_object_pos: Optional[List[float]] = None
        self._prev_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Main evaluation: called once per step, before/after the VLA query
    # ------------------------------------------------------------------
    def evaluate(
        self,
        current_image,
        diagnostics: Dict[str, Any],
        proposed_action: Optional[List[float]] = None,
        curr_object_pos: Optional[List[float]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return joint vision + physical evidence bundle.

        Keys:
          trigger: None or one of the named repair events
          vision_evidence: dict from visual grounding
          physical_evidence: dict from physical verifier
          action_physics_consistent: whether observed motion matches action intent
          action_physics_score: cosine-based consistency score [0,1]
          consistent: whether vision and physics agree on the main predicate
        """
        # --- Visual inference ---
        vision = self.grounding.infer_in_goal(
            current_image, goal_prototype_key='contained',
            threshold=self.vision_threshold)
        in_goal_vision = vision.get('in_goal_vision')
        vision_score = vision.get('score')

        # --- Physical verification ---
        receptacle_bounds = diagnostics.get('receptacle_bounds')
        contact_info = diagnostics.get('contacts_endpoint')
        phys_in_goal = self.verifier.verify_in_goal(
            curr_object_pos, receptacle_bounds, contact_info)

        # --- Action-physics consistency (new: catches outward-motion regression) ---
        action_phys = None
        if (proposed_action is not None and
                self._prev_object_pos is not None and
                curr_object_pos is not None):
            action_phys = self.verifier.verify_action_physics_consistency(
                proposed_action, self._prev_object_pos, curr_object_pos)

        # Update position history for next step
        if curr_object_pos is not None:
            self._prev_object_pos = curr_object_pos
            self._prev_step = step

        # --- Joint decision ---
        # containment_lost: vision says NOT in goal AND physics confirms NOT inside
        trigger = None
        if (in_goal_vision is False and not phys_in_goal.consistent):
            trigger = 'containment_lost'

        # action_physics_regression: vision+physics say in-goal, but action pushes outward
        elif (in_goal_vision is True and phys_in_goal.consistent
              and action_phys is not None and not action_phys.consistent):
            trigger = 'action_physics_regression'

        return {
            'trigger': trigger,
            'vision_evidence': vision,
            'physical_evidence': {
                'in_goal': phys_in_goal.consistent,
                'score': phys_in_goal.score,
                'details': phys_in_goal.details,
            },
            'action_physics_consistent': (
                action_phys.consistent if action_phys else None),
            'action_physics_score': (
                action_phys.score if action_phys else None),
            'consistent': (
                (in_goal_vision == phys_in_goal.consistent)
                if in_goal_vision is not None else None),
        }

    # ------------------------------------------------------------------
    # Grasp verification (for acquire-phase monitoring)
    # ------------------------------------------------------------------
    def verify_grasp(
        self,
        gripper_force: Optional[float],
        object_contact_normal: Optional[List[float]],
    ) -> PhysicalVerificationResult:
        return self.verifier.verify_grasp(gripper_force, object_contact_normal)

    def reset_history(self) -> None:
        self._prev_object_pos = None
        self._prev_step = None


def make_monitor(
    target='white_yellow_mug_1', receptacle='microwave_1',
    region='microwave_1_heating_region', vision_threshold=0.75,
) -> VisionPhysicalMonitor:
    verifier = PhysicalVerifier(
        target=target, receptacle=receptacle, region=region)
    grounding = VisualGroundingModule()
    return VisionPhysicalMonitor(
        verifier=verifier, grounding=grounding,
        vision_threshold=vision_threshold)
