"""Validated model outputs. These are runtime records, not per-task configs.

All clauses bind to environment-provided entity IDs. Recovery acceptance is
OR-of-ANDs; task completion and preservation are conjunctions. No generated
Python, numeric actions, thresholds or budget overrides are accepted.
"""
from dataclasses import dataclass


class PlanningError(ValueError):
    pass


ARITY = {'in': 2, 'on': 2, 'contact': 2, 'grasped': 1,
         'open': 1, 'close': 1, 'turnon': 1, 'turnoff': 1, 'stable': 1}


def fields(data, required, optional=()):
    if not isinstance(data, dict) or set(data) - set(required) - set(optional) or set(required) - set(data):
        raise PlanningError(f'Expected fields {required}, optional {optional}')


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise PlanningError('Expected nonempty bounded text')
    return value


@dataclass(frozen=True)
class Predicate:
    op: str
    args: tuple[str, ...]
    expected: bool = True

    @property
    def key(self):
        import json
        return json.dumps([self.op, *self.args], separators=(',', ':'))

    def to_dict(self):
        return dict(op=self.op, args=list(self.args), expected=self.expected)

    @classmethod
    def parse(cls, data, entities, capabilities):
        fields(data, ('op', 'args'), ('expected',))
        op, args, expected = data['op'], data['args'], data.get('expected', True)
        if not isinstance(op, str) or op not in ARITY or op not in capabilities:
            raise PlanningError(f'Unsupported predicate: {op}')
        if (not isinstance(args, list) or len(args) != ARITY[op]
                or any(not isinstance(a, str) or a not in entities for a in args)
                or type(expected) is not bool):
            raise PlanningError('Unbound entity or invalid predicate arguments')
        return cls(op, tuple(args), expected)


def clauses(data, entities, capabilities, *, empty=False):
    if not isinstance(data, list) or len(data) > 32 or (not data and not empty):
        raise PlanningError('Expected a bounded list of predicates')
    result = tuple(Predicate.parse(p, entities, capabilities) for p in data)
    values = {}
    for p in result:
        if p.key in values:
            raise PlanningError('Duplicate or contradictory predicate')
        values[p.key] = p.expected
    return result


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    goals: tuple[Predicate, ...]
    milestones: tuple[Predicate, ...]

    @classmethod
    def parse(cls, data, instruction, entities, capabilities):
        fields(data, ('goals', 'milestones'))
        return cls(text(instruction), clauses(data['goals'], entities, capabilities),
                   clauses(data['milestones'], entities, capabilities, empty=True))

    def to_dict(self):
        return dict(instruction=self.instruction, goals=[p.to_dict() for p in self.goals],
                    milestones=[p.to_dict() for p in self.milestones])


@dataclass(frozen=True)
class RecoveryPlan:
    instruction: str
    alternatives: tuple[tuple[Predicate, ...], ...]
    preserve: tuple[Predicate, ...]
    reason: str

    @classmethod
    def parse(cls, data, entities, capabilities):
        fields(data, ('instruction', 'acceptance', 'preserve', 'reason'))
        alternatives = data['acceptance']
        if not isinstance(alternatives, list) or not 1 <= len(alternatives) <= 8:
            raise PlanningError('Recovery needs 1..8 nonempty acceptance alternatives')
        alternatives = tuple(clauses(c, entities, capabilities) for c in alternatives)
        preserve = clauses(data['preserve'], entities, capabilities, empty=True)
        for alternative in alternatives:
            for p in alternative:
                if any(p.key == q.key and p.expected != q.expected for q in preserve):
                    raise PlanningError('Recovery contradicts its preservation contract')
        return cls(text(data['instruction']), alternatives, preserve, text(data['reason']))

    def to_dict(self):
        return dict(instruction=self.instruction,
                    acceptance=[[p.to_dict() for p in c] for c in self.alternatives],
                    preserve=[p.to_dict() for p in self.preserve], reason=self.reason)
