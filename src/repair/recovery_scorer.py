"""Module A: learned scoring of unchanged H1 recovery proposals."""
from ..verification.fusion_mlp import FusionMLP


class RecoveryScorer:
    def __init__(self, checkpoint, monitor, *, device='cpu'):
        if monitor.encoder is None:
            raise ValueError('Learned recovery ranking requires visual context')
        self.monitor = monitor
        self.head = FusionMLP.load(checkpoint, role='recovery', encoder_signature=monitor.signature, device=device)
        self.provenance = self.head.provenance

    def score(self, observation, task, plan, actions, physics):
        return [self.head.probability(self.monitor.action_features(observation, task, plan, action, physics))
                for action in actions]
