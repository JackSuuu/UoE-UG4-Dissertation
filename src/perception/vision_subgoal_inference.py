"""Visual prototype stage inference with explicit physical admissibility."""
from typing import Optional, Dict, Any, List
from .visual_grounding import VisualGroundingModule
import numpy as np


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

    @staticmethod
    def admissible_stages(measured):
        if measured.get('in_goal') is True:
            if measured.get('grasp') is False and measured.get('mug_gripper') is False:
                return ['close']
            return ['transfer', 'contain_release']
        if measured.get('grasp') is False:
            return ['acquire']
        if measured.get('grasp') is True:
            # Acquisition can still be incomplete while transfer is beginning.
            # Vision/memory resolve this ambiguity, rather than a fixed switch.
            return ['acquire', 'transfer']
        return ['acquire']  # Unknown grasp never authorizes release or closing.

    def infer_features(self, visual_feature, measured, prototypes):
        q = np.asarray(visual_feature, dtype=np.float32)
        if q.shape != (384,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
            raise ValueError('Missing real visual features for joint stage inference')
        q = q/np.linalg.norm(q)
        scores = {stage: float(np.dot(q, feature)) for stage, feature in prototypes.items()
                  if feature is not None and stage in self.STAGES}
        allowed = self.admissible_stages(measured)
        candidates = [stage for stage in allowed if stage in scores]
        if not candidates:
            raise ValueError('No train-supported prototype for physically admissible stages')
        selected = max(candidates, key=lambda stage: scores[stage])
        return dict(subgoal=selected, source='pretrained_visual_prototypes_physical_admissibility',
            all_scores=scores, admissible=allowed, confidence=scores[selected])


def make_subgoal_inference(offline: bool = True, device: str = 'cuda:0') -> VisionSubgoalInference:
    grounding = VisualGroundingModule(offline=offline, device=device)
    return VisionSubgoalInference(grounding=grounding)
