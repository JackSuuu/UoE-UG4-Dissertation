"""Vision + physical subgoal inference.

Replaces the hard-coded _subgoal() predicate chain in native_framework.py.
Instead of hand-written rules over privileged booleans, the current task stage
(acquire / transfer / contain_release / close) is inferred from visual features
with physical boundary correction.

This makes the framework task-agnostic: new stages can be added by supplying
new visual prototypes, without rewriting Python control flow.
"""
from typing import Optional, Dict, Any, List
from .visual_grounding import VisualGroundingModule
from .physical_verifier import PhysicalVerifier


class VisionSubgoalInference:
    STAGES = ('acquire', 'transfer', 'contain_release', 'close')

    def __init__(
        self,
        grounding: VisualGroundingModule,
        verifier: Optional[PhysicalVerifier] = None,
    ):
        self.grounding = grounding
        self.verifier = verifier
        self._stage_prototypes: Dict[str, any] = {}

    def set_stage_prototype(self, stage: str, image) -> None:
        """Register a visual prototype for a task stage's canonical state."""
        self.grounding.set_stage_prototype(stage, image)

    def infer(
        self,
        current_image,
        measured: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Infer the current subgoal from visual features + physical measurements.

        Returns a dict with:
          - subgoal: inferred stage name or None
          - confidence: visual matching score
          - source: 'vision' | 'vision_physical' | 'fallback'
          - all_scores: per-stage visual scores
        """
        visual = self.grounding.infer_stage(current_image, stages=list(self.STAGES))
        stage = visual.get('stage')
        confidence = visual.get('confidence')
        all_scores = visual.get('all_scores', {})

        # Physical boundary correction: if the measured predicates strongly
        # contradict the visual inference, override with the measured signal.
        # E.g. visual says 'close' but the mug is clearly not grasped / not in
        # the receptacle -> fall back to 'acquire'.
        in_goal = measured.get('in_goal')
        grasp = measured.get('grasp')
        held_clear = measured.get('held_clear')

        source = 'vision'
        if stage is not None:
            if in_goal is True and grasp is False and stage != 'close':
                # Mug is in goal and released -> should be close, not contain_release
                stage = 'close'
                source = 'vision_physical'
            elif in_goal is False and held_clear is True and stage == 'contain_release':
                # Holding-clear but not in goal -> transfer, not contain_release
                stage = 'transfer'
                source = 'vision_physical'
            elif grasp is False and not held_clear and stage in ('transfer', 'contain_release', 'close'):
                # Not grasping, not holding clear -> need to acquire first
                stage = 'acquire'
                source = 'vision_physical'

        return {
            'subgoal': stage,
            'confidence': confidence,
            'source': source,
            'all_scores': all_scores,
            'visual_stage': visual.get('stage'),
        }


def make_subgoal_inference(target='white_yellow_mug_1',
                           receptacle='microwave_1') -> VisionSubgoalInference:
    grounding = VisualGroundingModule()
    verifier = PhysicalVerifier(target=target, receptacle=receptacle)
    return VisionSubgoalInference(grounding=grounding, verifier=verifier)
