"""Generated-milestone visual context; no microwave stages or success override."""
from .visual_grounding import VisualGroundingModule
from ..verification.physical_verifier import PhysicalVerifier


class VisionSubgoalInference:
    def __init__(self, feature_dim=384):
        self.grounding = VisualGroundingModule(feature_dim)
        self.verifier = PhysicalVerifier()

    def infer_features(self, feature, task, observation, history, memory):
        records = []
        for predicate in dict.fromkeys((*task.goals, *task.milestones)):
            records.append(dict(predicate=predicate.to_dict(),
                measured=self.verifier.evaluate(predicate, observation, history),
                similarity=self.grounding.compare(feature, memory.prototype(predicate))))
        return dict(method='generated_predicate_prototypes', predicates=records,
                    visual_is_verification=False)
