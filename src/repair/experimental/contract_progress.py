"""Measured contract progress for bounded H1 request control.

This component never predicts action outcomes or synthesizes numeric actions.
Task adapters supply measured acceptance checks and preservation predicates.
"""
from collections import deque
from copy import deepcopy
import math


class ContractProgress:
    SCHEMA = 'measured_contract_progress_v1'

    def __init__(self, objective, window, motion_m):
        if window < 2 or motion_m <= 0:
            raise ValueError('Progress monitoring needs a positive motion scale and a multi-step window')
        self.objective = objective
        self.window = window
        self.motion_m = motion_m
        self.states = deque(maxlen=window)
        self.checks = deque(maxlen=window)

    def observe(self, state, acceptance_checks, preserved):
        self.states.append(deepcopy(state))
        self.checks.append(deepcopy(acceptance_checks))
        report = dict(schema=self.SCHEMA, objective=self.objective,
            acceptance_checks=deepcopy(acceptance_checks), preserved=preserved,
            observed_endpoints=len(self.states), required_endpoints=self.window,
            motion_threshold_m=self.motion_m, diameter_m=None, semantic_change=None,
            stop_unverified=False)
        if not preserved:
            return dict(report, diagnosis='preservation_failed_or_unknown', stop_unverified=True)
        if any(acceptance_checks.values()):
            return dict(report, diagnosis='local_condition_observed_not_handoff_verified')
        if len(self.states) < self.window:
            return dict(report, diagnosis='insufficient_local_history')
        keys = ('grasp', 'held_clear', 'in_goal', 'close_goal')
        if any(s.get('position') is None or any(s.get(k) is None for k in keys) for s in self.states):
            return dict(report, diagnosis='missing_progress_measurements')
        if any(any(check.values()) for check in self.checks):
            return dict(report, diagnosis='recent_acceptance_progress')
        first = self.states[0]
        changed = any(any(s[k] != first[k] for s in self.states) for k in keys)
        diameter = max(math.dist(a['position'], b['position']) for a in self.states for b in self.states)
        stalled = not changed and diameter <= self.motion_m
        return dict(report, diameter_m=diameter, semantic_change=changed,
            diagnosis='stationary_unmet_contract' if stalled else 'changing_unmet_contract',
            stop_unverified=stalled)
