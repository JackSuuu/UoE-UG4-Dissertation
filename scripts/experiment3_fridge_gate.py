"""Fridge v1: measured stage observer plus model-candidate-only decisions."""
from collections import deque
import numpy as np
from experiment3_candidate_gate import ARMS, LOCAL_CAP, COOLDOWN, WINDOW, executed_events, observable_trace


def command_checks(action, previous, state):
    a = np.asarray(action)
    valid = bool(a.shape == (7,) and np.isfinite(a).all() and np.all(np.abs(a) <= 1))
    delta = float(np.linalg.norm(a[:6]-np.asarray(previous)[:6])) if valid and previous is not None else 0.
    support = state.get('local_release_support')
    release_ok = (not state['grasp'] or support is True) if valid and a[-1] < 0 else True
    return dict(command_contract=valid, command_continuity=delta <= .35,
        continuity_margin=.35-delta, local_release_consistency=bool(release_ok),
        local_release_support='unknown' if support is None else 'satisfied' if support else 'violated',
        release_rule='when currently grasped, opening needs measured target-table support and placement envelope; never full native success',
        future_collision='unknown', future_support='unknown', future_task_progress='unknown')


class FridgeCandidateGate:
    def __init__(self, arm):
        assert arm in ARMS
        self.arm = arm
        self.history = deque(maxlen=WINDOW+1)
        self.actions = deque(maxlen=5)
        self.active_phase = None
        self.last_phase = None
        self.used = self.stable = self.transaction = self.cooldown_until = 0

    @property
    def active(self):
        return self.active_phase is not None

    def begin(self, step, state, stages):
        phase = stages['phase']
        if phase != self.last_phase:
            self.history.clear()
            self.last_phase = phase
        self.history.append(state)
        old = self.history[0]
        dm = float(np.linalg.norm(np.asarray(state['item'])-old['item']))
        de = float(np.linalg.norm(np.asarray(state['eef'])-old['eef']))
        da = abs(state['angle']-old['angle'])
        stalled = len(self.history) == WINDOW+1 and dm < .002 and de < .005 and da < .01
        trigger = bool(step >= 40 and stalled and stages['predicates'][phase]['status'] != 'satisfied')
        events = []
        if self.active:
            current = stages['predicates'][self.active_phase]['status']
            self.stable = self.stable+1 if current == 'satisfied' else 0
            reason = None
            if self.used >= LOCAL_CAP:
                reason = 'local_action_cap_unverified'
            elif self.stable >= (3 if self.arm == 'verification' else 1):
                reason = 'three_current_stage_rechecks' if self.arm == 'verification' else 'single_current_stage_check'
            if reason:
                events.append(dict(type='local_exit', step=step, phase=self.active_phase,
                    transaction=self.transaction, reason=reason, task_completion_claim=False))
                self.active_phase = None
                self.cooldown_until = step+COOLDOWN
        if self.arm != 'nominal' and not self.active and step >= self.cooldown_until and trigger:
            self.active_phase, self.used, self.stable = phase, 0, 0
            self.transaction += 1
            events.append(dict(type='local_entry', step=step, phase=phase,
                transaction=self.transaction, reason='same_phase_observed_stagnation'))
        return dict(trigger=trigger, stalled=bool(stalled), active=self.active,
            active_phase=self.active_phase, phase=phase, transaction=self.transaction,
            item_motion_margin_m=dm-.002, eef_motion_margin_m=de-.005,
            door_motion_margin_rad=da-.01, observations=len(self.history),
            stage_recheck_streak=self.stable, used_before=self.used, events=events)

    def choose(self, candidates, state, monitor):
        audits = []
        for i, c in enumerate(candidates):
            audits.append(dict(index=i, source=c['source'], query=c['query'], action=c['action'],
                checks=command_checks(c['action'], self.actions[-1] if self.actions else None, state),
                novelty=min((float(np.linalg.norm(np.asarray(c['action'])[:6]-np.asarray(a)[:6]))
                    for a in self.actions), default=0.)))
        index, owner, reason = 0, 'nominal', 'shared_phase_frozen_greedy'
        if self.active:
            owner = self.arm+'_candidate'
            if self.arm == 'simple_feedback':
                eligible = [c for c in audits[1:] if c['checks']['command_contract']]
                if eligible:
                    index, reason = eligible[0]['index'], 'first_contract_valid_model_sample'
                else:
                    reason = 'sample_budget_or_contract_fallback_nominal'
            else:
                eligible = [c for c in audits if all(c['checks'][k] for k in
                    ('command_contract', 'command_continuity', 'local_release_consistency'))]
                if eligible:
                    index = max(eligible, key=lambda c:(c['novelty'], -c['index']))['index']
                    reason = 'highest_command_novelty_among_current_checks_passed'
                else:
                    reason = 'all_rejected_nominal_fallback_unverified'
            self.used += 1
        if not audits[index]['checks']['command_contract']:
            raise RuntimeError('invalid_selected_model_action_no_scripted_fallback')
        action = candidates[index]['action']
        self.actions.append(action)
        return action, dict(owner=owner, chosen=index, reason=reason, candidates=audits,
            used=self.used, local_cap=LOCAL_CAP, remaining_local=max(0, LOCAL_CAP-self.used),
            events=list(monitor['events']))


def trace(step, arm, state, stages, candidates, action, monitor, decision, native):
    before = dict(state=state, stages=stages, native_success=native)
    t = observable_trace(step, arm, before, candidates, action, monitor, decision)
    t.update(schema='fridge_candidate_v1', subgoal=stages['phase'],
        milestones_before=stages['milestone_history'],
        stage_evidence_role='read-only privileged observer, synthetic semantics tested; not real-robot validation')
    return t
