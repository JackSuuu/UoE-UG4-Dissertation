"""Visual similarity is context, not proof of a generated physical predicate."""
from ..memory.keyframe_bank import unit


class VisualGroundingModule:
    def __init__(self, feature_dim=384):
        self.feature_dim = feature_dim

    def compare(self, feature, prototype):
        if prototype is None:
            return None
        return sum(a*b for a, b in zip(unit(feature, self.feature_dim), unit(prototype, self.feature_dim)))
