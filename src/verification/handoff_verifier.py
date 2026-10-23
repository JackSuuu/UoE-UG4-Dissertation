"""Module B: learned acceptance of the exact fresh original-task action."""
from .fusion_mlp import FusionMLP


class HandoffVerifier:
    def __init__(self, checkpoint, monitor, *, device='cpu'):
        if monitor.encoder is None:
            raise ValueError('Learned handoff verification requires visual context')
        self.monitor = monitor
        self.head = FusionMLP.load(checkpoint, role='handoff', encoder_signature=monitor.signature, device=device)
        self.provenance = self.head.provenance

    def assess(self, observation, task, plan, nominal_action, physics):
        features = self.monitor.action_features(observation, task, plan, nominal_action, physics)
        score = self.head.probability(features)
        return dict(accept=score > self.head.threshold, score=score, threshold=self.head.threshold,
                    method='learned_exact_nominal_action', features=features)
