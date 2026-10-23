"""Dependency-free backend contracts; no model or simulator imports.

Backends implement these structural protocols; inheritance is not required.
Session owns policy queries and steps. The caller owns backend lifetimes.
"""
from dataclasses import dataclass, field
import math
from typing import Protocol, TypedDict

from .observation import Observation
from .planning.schema import Predicate, RecoveryPlan, TaskSpec


@dataclass(frozen=True)
class Proposal:
    action: tuple[float, ...]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if (not isinstance(self.action, tuple) or len(self.action) != 7
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.action)):
            raise ValueError('A frozen policy must return one finite numeric seven-coordinate proposal')
        if not isinstance(self.provenance, dict):
            raise ValueError('Proposal provenance must be a mapping')


class Environment(Protocol):
    capabilities: frozenset[str]
    done: bool

    def observe(self, predicates: tuple[Predicate, ...] = ()) -> Observation: ...
    def step(self, action: tuple[float, ...]) -> None: ...


class Policy(Protocol):
    def propose(self, image_png: bytes, instruction: str, *, sample: bool = False,
                seed: int = 0) -> Proposal: ...


class Planner(Protocol):
    def task(self, instruction: str, observation: Observation,
             capabilities: frozenset[str]) -> TaskSpec: ...
    def recover(self, task: TaskSpec, observation: Observation, capabilities: frozenset[str],
                event: dict, history: list[dict]) -> RecoveryPlan | None: ...


class RecoveryScorer(Protocol):
    def score(self, observation: Observation, task: TaskSpec, plan: RecoveryPlan,
              actions: tuple[tuple[float, ...], ...], physics: dict) -> list[float] | tuple[float, ...]: ...


class HandoffAssessment(TypedDict):
    accept: bool


class HandoffScorer(Protocol):
    def assess(self, observation: Observation, task: TaskSpec, plan: RecoveryPlan,
               nominal_action: tuple[float, ...], physics: dict) -> HandoffAssessment: ...
