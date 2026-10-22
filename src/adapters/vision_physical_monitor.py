"""Vision + physical joint monitor.

A repair trigger fires only when visual anomaly AND physical verifier agree,
suppressing false interventions.  Also exposes action-physics consistency to
catch "push in but object moves out" regressions.
"""
from typing import Optional, Dict, Any, List
from .physical_verifier import PhysicalVerifier, extract_physics_state
from .visual_grounding import VisualGroundingModule


class VisionPhysicalMonitor:
    def __init__(
        self,
        verifier: PhysicalVerifier,
        grounding: VisualGroundingModule,
        vision_threshold: float = 0.75,
    ):
        self.verifier = verifier
        self.grounding = grounding
        self.vision_threshold = vision_threshold
        self._prev_object_pos: Optional[List[float]] = None

    def evaluate(
        self,
        current_image,
        diagnostics: Dict[str, Any],
        proposed_action: Optional[List[float]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return joint vision + physical evidence bundle."""
        # --- Vision ---
        vision = self.grounding.infer_in_goal(
            current_image, goal_prototype_key='contained',
            threshold=self.vision_threshold)
        in_goal_vision = vision.get('in_goal_vision')
        vision_score = vision.get('score')

        # --- Physical ---
        phys_state = extract_physics_state(diagnostics)
        phys_in_goal = self.verifier.verify_in_goal(
            phys_state['object_pos'],
            phys_state['receptacle_bounds'],
            phys_state['contact_info'],
        )

        # --- Action-physics consistency ---
        action_phys = None
        if (proposed_action is not None and
                self._prev_object_pos is not None and
                phys_state['object_pos'] is not None):
            action_phys = self.verifier.verify_action_physics_consistency(
                proposed_action, self._prev_object_pos, phys_state['object_pos'])

        if phys_state['object_pos'] is not None:
            self._prev_object_pos = phys_state['object_pos']

        # --- Joint trigger ---
        trigger = None
        if (in_goal_vision is False and not phys_in_goal.consistent):
            trigger = 'containment_lost'
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
            'physics_state': phys_state,
        }

    def verify_grasp(self, gripper_force: Optional[float]):
        return self.verifier.verify_grasp(gripper_force)

    def reset(self):
        self._prev_object_pos = None


def make_monitor(
    vision_threshold: float = 0.75,
) -> VisionPhysicalMonitor:
    verifier = PhysicalVerifier()
    grounding = VisualGroundingModule()
    return VisionPhysicalMonitor(
        verifier=verifier, grounding=grounding,
        vision_threshold=vision_threshold)
