"""Tri-state measured contracts and four-dimensional causal physical evidence."""
import json
import math


class PhysicalVerifier:
    def evaluate(self, predicate, observation, history=(), *, window=3, motion_m=.003):
        """Unknown evidence never verifies, including a negated predicate.

        Stability uses consecutive executed endpoints, not repeated reads of
        one frame. The initial frame is not an executed dwell observation.
        """
        if predicate.op == 'stable':
            recent = list(history)[-window:]
            positions = [o.positions.get(predicate.args[0]) for o in recent]
            indices = [o.index for o in recent]
            if (len(recent) != window or any(p is None for p in positions)
                    or indices[0] < 1 or indices[-1] != observation.index
                    or indices != list(range(indices[0], indices[0] + window))):
                value = None
            else:
                value = max((math.dist(a, b) for a in positions for b in positions), default=0.) <= motion_m
        else:
            value = observation.facts.get(predicate.key)
        return None if value is None else value == predicate.expected

    def check_all(self, predicates, observation, history=()):
        values = [self.evaluate(p, observation, history) for p in predicates]
        if False in values:
            return False
        return None if None in values else True

    def evidence(self, observation, *, target=None, previous=None, executed_action=None):
        """Measured containment/grasp, last executed EEF alignment and contacts.

        Validity/provenance are metadata, not extra feature dimensions. Alignment
        is retrospective: never a forecast of an unexecuted candidate's effects.
        Containment uses the bridge's measured relation, not an invented AABB.
        """
        inside = []
        for key, value in observation.facts.items():
            clause = json.loads(key)
            if len(clause) == 3 and clause[0] == 'in' and clause[1] == target:
                inside.append(value)
        containment = (True if True in inside else False
                       if inside and all(v is False for v in inside) else None)
        grasp_key = json.dumps(['grasped', target], separators=(',', ':'))
        grasp = observation.facts.get(grasp_key) if target else None
        alignment = None
        if (executed_action is not None and previous is not None
                and previous.index + 1 == observation.index
                and previous.eef_position is not None and observation.eef_position is not None):
            delta = [b-a for a, b in zip(previous.eef_position, observation.eef_position)]
            norm = math.sqrt(sum(x*x for x in delta))
            intent = executed_action[:3]
            intent_norm = math.sqrt(sum(x*x for x in intent))
            if norm > 1e-6 and intent_norm > 1e-6:
                alignment = max(0., min(1., (sum(a*b for a, b in zip(intent, delta)) /
                                           (norm*intent_norm) + 1.) / 2.))
        contacts = (None if observation.contacts is None or target is None else
                    min(sum(target in pair for pair in observation.contacts)/10., 1.))
        values = [containment, grasp, alignment, contacts]
        return dict(schema='measured_physics4_v2', values=[0. if v is None else float(v) for v in values],
                    valid=[v is not None for v in values], target=target,
                    observation_index=observation.index,
                    semantics=['measured_containment', 'measured_grasp',
                               'last_executed_eef_alignment', 'target_contact_count'],
                    prospective=False)
