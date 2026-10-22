"""Vision + physical subgoal inference.

Replaces hard-coded _subgoal() rules.  Current stage is inferred from visual
features with physical boundary correction.  Task-agnostic: new stages added
by registering new visual prototypes, not by rewriting Python control flow.
"""
from typing import Optional, Dict, Any, List
from .visual_grounding import VisualGroundingModule


class VisionSubgoalInference:
    STAGES = ('acquire', 'transfer', 'contain_release', 'close')

    def __init__(
        self,
        grounding: VisualGroundingModule,
    ):
        self.grounding = grounding

    def set_stage_prototype(self, stage: str, image) -> None:
        self.grounding.set_stage_prototype(stage, image)

    def infer(
        self,
        current_image,
        measured: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Infer current subgoal from visual features + physical measurements."""
        visual = self.grounding.infer_stage(
            current_image, stages=list(self.STAGES))
        stage = visual.get('stage')
        confidence = visual.get('confidence')
        all_scores = visual.get('all_scores', {})

        source = 'vision'
        if stage is not None:
            in_goal = measured.get('in_goal')
            grasp = measured.get('grasp')
            held_clear = measured.get('held_clear')

            if in_goal is True and grasp is False and stage != 'close':
                stage = 'close'
                source = 'vision_physical'
            elif in_goal is False and held_clear is True and stage == 'contain_release':
                stage = 'transfer'
                source = 'vision_physical'
            elif grasp is False and not held_clear and stage in ('transfer', 'contain_release', 'close'):
                stage = 'acquire'
                source = 'vision_physical'

        return {
            'subgoal': stage,
            'confidence': confidence,
            'source': source,
            'all_scores': all_scores,
            'visual_stage': visual.get('stage'),
        }


def make_subgoal_inference() -> VisionSubgoalInference:
    grounding = VisualGroundingModule()
    return VisionSubgoalInference(grounding=grounding)
