"""Development failure mining: causal transition repair and literature-inspired audit.

Not official SEAL/CheckVLA. Geometry and candidate simulation are privileged.
No fault schedule, future labels or candidate outcomes enter the event monitor.
"""
import random
import time

import numpy as np

from adapters.libero_drawer import (DrawerMonitor, MAX_REPAIR, TRANSLATION,
                                    decision_trace, settled_drawer)
from adapters.libero_completion import ContactRecorder
from adapters.libero_preservation import fingerprint
from adapters.libero_stress import ReleaseDelay, release_opportunity

ARMS = ('vla', 'gate_hold', 'timed_repair', 'phase_repair', 'verified_repair')
FORECAST_HORIZON = 3


def apply_fault(selected, step, case):
    """Evaluator-only post-routing actuator defect; never passed to a controller."""
    action = np.asarray(selected, dtype=float).copy()
    assert action.shape == (7,) and np.isfinite(action).all()
    if case['fault'] == 'release_delay':
        return ReleaseDelay(case['anchor_step'], case['duration']).apply(action, step)
    active = case['anchor_step'] <= step < case['anchor_step'] + case['duration']
    effective = action.copy()
    if active:
        effective[case['axis']] += case['requested_m'] / (.05 * case['duration'])
        effective[:6] = np.clip(effective[:6], -1, 1)
    return effective, dict(kind=case['fault'], active=active, step=step,
        selected_command=action.tolist(), effective_command=effective.tolist(),
        changed=not np.array_equal(action, effective),
        provenance='external actuator fault; evaluator log only')


class EvidenceMemory:
    """CheckVLA-inspired command/consequence audit and event evidence bank.

    Symbolic expectation, NOT a trained world-model residual or conformal score.
    Bank remains outside frozen actor; no claim of learned keyframe cross-attention.
    """
    def __init__(self):
        self.open_contact_streak = 0
        self.bank = []
        self.signature = None

    def observe(self, previous_selected, measured, tracking, step):
        open_requested = bool(previous_selected is not None and previous_selected[-1] < 0)
        self.open_contact_streak = self.open_contact_streak + 1 if (
            open_requested and measured['bowl_gripper'] and measured['native_in']) else 0
        signature = (tracking['phase'], measured['retained'], tracking['released'], tracking['clear'])
        if signature != self.signature:
            self.bank.append(dict(evidence_step=step-1, phase=tracking['phase'],
                retained=measured['retained'], released=tracking['released'], clear=tracking['clear']))
            self.bank = self.bank[-16:]
            self.signature = signature
        return dict(open_requested_previous=open_requested,
            contact_persists_after_open=self.open_contact_streak,
            expected_effect='opening should eventually detach; transient contact is not failure',
            keyframes=list(self.bank), prediction_type='symbolic action-consequence audit')


class ReleaseGate:
    """One transaction, early release entry; same detector across repair arms."""
    def __init__(self, arm):
        assert arm in ARMS
        self.arm, self.trigger, self.handoff = arm, None, None
        self.used, self.outside_streak = 0, 0
        self.forecast_events = set()

    def decide(self, proposal, measured, tracking, step):
        action = np.asarray(proposal).copy()
        entry = release_opportunity(measured, proposal)
        if self.trigger is None and entry:
            self.trigger = step
        self.outside_streak = self.outside_streak + 1 if not measured['native_in'] else 0
        active = self.trigger is not None and self.handoff is None
        event = None
        if active and 'entry' not in self.forecast_events:
            event = 'entry'
        elif active and tracking['released'] and 'released' not in self.forecast_events:
            event = 'released'
        if event is not None:
            self.forecast_events.add(event)
        reason, route, mode = 'waiting_release_opportunity', 'continue', 'nominal'
        hold = np.zeros(7)
        hold[-1] = -1
        phase_action = hold.copy()
        if tracking['released'] and not measured['clear']:
            phase_action[:3] = np.asarray(measured['outward_world']) * (TRANSLATION/.05)
        if self.arm == 'vla':
            reason = 'baseline_passthrough'
        elif self.handoff is not None:
            reason = 'actor_after_handoff'
        elif active:
            exit_reason = None
            if self.outside_streak >= 3:
                exit_reason = 'release_envelope_lost'
            elif self.used >= MAX_REPAIR:
                exit_reason = 'budget_exhausted'
            elif self.arm == 'gate_hold' and tracking['released']:
                exit_reason = 'stable_release'
            elif self.arm in ('phase_repair','verified_repair') and tracking['clear']:
                exit_reason = 'stable_clearance'
            if exit_reason:
                reason = exit_reason
                self.handoff = dict(step=step, reason=reason, stable_clear=tracking['clear'],
                                    stable_release=tracking['released'])
            else:
                action, reason, route, mode = hold.copy(), 'release_hold', 'hold', 'contact_transition'
                if self.arm == 'timed_repair' and self.used >= 10:
                    action[:3] = np.asarray(measured['outward_world']) * (TRANSLATION/.05)
                elif self.arm in ('phase_repair','verified_repair'):
                    action = phase_action.copy()
                if np.any(action[:6]):
                    reason, route = 'retract_outward', 'repair'
                self.used += 1
        return action, dict(mode=mode, route=route, reason=reason, trigger=self.trigger,
            used=self.used, handoff=self.handoff, release_opportunity=entry,
            outside_streak=self.outside_streak, forecast_event=event), phase_action


def prefix_decision():
    return dict(mode='nominal', route='continue', reason='common_recorded_prefix',
        trigger=None, used=0, handoff=None, release_opportunity=False,
        outside_streak=0, forecast_event=None)


def trace_record(step, arm, measured, tracking, proposal, selected, decision, memory, forecast):
    trace = decision_trace(step, arm, measured, tracking, proposal, selected, decision)
    trace['schema'] = 2
    trace['selected_action'] = trace.pop('executed_action')
    trace['predicates']['release_opportunity'] = dict(value=decision['release_opportunity'],
        rule='previous native_in AND bowl_gripper AND NOT closed AND nominal open')
    trace['predicates']['supported_entry']['diagnostic_only'] = True
    trace['predicates']['envelope_loss'] = dict(consecutive=decision['outside_streak'], required=3)
    trace['action_consequence_memory'] = memory
    trace['candidate_verification'] = forecast
    trace['interpretation'] = 'causal physical execution proof; inspired components, not official SEAL/CheckVLA'
    return trace


def observed_outcome(measured, tracking):
    return dict(retained=measured['retained'], bowl_gripper=measured['bowl_gripper'],
        cabinet_gripper=measured['cabinet_gripper'], clear=measured['clear'],
        drawer_q=measured['drawer_q'], native=measured['native_success'],
        full_completion=tracking['complete'], phase=tracking['phase'])


def candidate_plans(proposal, phase_action):
    first = np.zeros((3,7))
    first[0], first[1] = proposal, phase_action
    first[2,-1] = -1
    plans = np.zeros((3,FORECAST_HORIZON,7))
    plans[:,0] = first
    plans[:,1:,-1] = first[:,-1,None]
    return plans


def forecast_key(rows, first, proposal, index):
    # Lexicographic physical subgoal checks; no calibrated collision/safety claim.
    return [sum(not m['native_in'] for m in rows),
            sum(not m['retained'] for m in rows),
            sum(m['max_bowl_cabinet_penetration'] > .001 for m in rows),
            sum(m['bowl_gripper'] for m in rows), sum(not m['clear'] for m in rows),
            float(np.linalg.norm(first-proposal)), index]


def forecast_candidates(root, index, executed_prefix, reference, proposal, phase_action):
    """SEAL-inspired propose/predict/verify: three local pulse-and-settle plans.

    No future fault injection, no future VLA action invention. All arms get same
    opportunity/candidate cap; only verified_repair uses the selected candidate.
    """
    plans = candidate_plans(proposal, phase_action)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    start, predictions = time.monotonic(), []
    try:
        for k, plan in enumerate(plans):
            env = rec = None
            try:
                env, obs = settled_drawer(root, index)
                for action in executed_prefix:
                    obs, _, _, _ = env.step(action)
                assert fingerprint(env, obs) == reference, ('forecast prefix', k)
                mon = DrawerMonitor(env)
                rec = ContactRecorder(env.sim, keywords=('akita_black_bowl','white_cabinet'))
                rows = []
                for action in plan:
                    rec.reset()
                    obs, _, _, _ = env.step(action.tolist())
                    assert len(rec.samples) == 25
                    rows.append(mon.measure(rec.samples))
                predictions.append(dict(index=k, plan=plan.tolist(), rows=rows,
                    key=forecast_key(rows, plan[0], proposal, k), replay_identity=True))
            finally:
                if rec is not None:
                    rec.close()
                if env is not None:
                    env.close()
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    chosen = min(range(3), key=lambda k: predictions[k]['key'])
    return dict(candidates=predictions, selected=chosen, first_action=plans[chosen,0].tolist(),
        names=['nominal_pulse','phase_controller_pulse','hold_open'],
        forecast_steps=3*FORECAST_HORIZON, prefix_replay_steps=3*len(executed_prefix),
        wall_seconds=time.monotonic()-start, future_fault_known=False,
        privileged=True, learned=False, horizon_semantics='local controller primitive, not VLA chunk')


def apply_verification(arm, selected, decision, forecast):
    if arm == 'verified_repair' and decision['mode'] == 'contact_transition' and forecast is not None:
        selected = np.asarray(forecast['first_action'])
        decision = dict(decision, reason='verified_candidate_'+str(forecast['selected']), route='repair')
    return selected, decision
