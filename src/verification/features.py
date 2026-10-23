"""Versioned generic fusion features; physical evidence remains four-dimensional.

Visual embedding + physics values/masks + generated predicate grammar/truth
summaries + acceptance alternatives + causal history + optional unchanged action.
No fixed task stages, object-name slots, benchmark goals or future observations.
This is grammar pooling, not a language entailment model.
"""
import math

from ..memory.keyframe_bank import unit
from ..planning.schema import ARITY
from .physical_verifier import PhysicalVerifier


FEATURE_SCHEMA = 'generated_vision384_physics4mask4_grammar180_or4_history8_action8_v1'
FEATURE_DIM = 384 + 8 + 180 + 4 + 8 + 8


def fusion_features(visual, observation, task, physics, *, plan=None, action=None, history=()):
    if physics.get('schema') != 'measured_physics4_v2' or len(physics['values']) != 4 or len(physics['valid']) != 4:
        raise ValueError('Expected four-dimensional measured physical evidence and validity')
    result = list(unit(visual, 384)) + list(physics['values']) + [float(v) for v in physics['valid']]
    verifier = PhysicalVerifier()
    history = list(history)
    if not history or history[-1].index != observation.index:
        history.append(observation)
    groups = (task.goals, task.milestones,
              tuple(p for group in plan.alternatives for p in group) if plan else (),
              plan.preserve if plan else ())
    for group in groups:
        for op in sorted(ARITY):
            predicates = [p for p in group if p.op == op]
            values = [verifier.evaluate(p, observation, history) for p in predicates]
            n = max(1, len(group))
            result += [len(predicates)/256., sum(p.expected for p in predicates)/n,
                       sum(v is True for v in values)/n, sum(v is False for v in values)/n,
                       sum(v is None for v in values)/n]
    alternatives = [verifier.check_all(g, observation, history) for g in plan.alternatives] if plan else []
    n = max(1, len(alternatives))
    result += [len(alternatives)/8., sum(v is True for v in alternatives)/n,
               sum(v is False for v in alternatives)/n, sum(v is None for v in alternatives)/n]
    recent = history[-12:]
    speeds = [math.dist(a.eef_position, b.eef_position) for a, b in zip(recent, recent[1:])
              if a.index+1 == b.index and a.eef_position is not None and b.eef_position is not None]
    current_values = [verifier.evaluate(p, observation, history) for p in task.goals]
    known = [v for v in current_values if v is not None]
    result += [len(recent)/12., sum(speeds)/len(speeds) if speeds else 0., max(speeds, default=0.),
               sum(v is True for v in known)/len(known) if known else 0.,
               1., float(bool(speeds)), float(bool(speeds)), float(bool(known))]
    if action is not None:
        from ..interfaces import Proposal
        Proposal(tuple(action))
    result += list(action) + [1.] if action is not None else [0.]*8
    if len(result) != FEATURE_DIM or not all(math.isfinite(float(x)) for x in result):
        raise ValueError('Invalid generic fusion vector')
    return tuple(float(x) for x in result)


def encoder_signature(provenance):
    return {k: provenance[k] for k in ('sha256', 'source_sha256', 'transform', 'feature_dim')}
