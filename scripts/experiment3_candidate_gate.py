"""Observable book-task candidate experiment; no motion synthesis or lookahead."""
from collections import deque
import numpy as np

ARMS = ('nominal', 'simple_feedback', 'verification')
CAP = 600
WALL_SECONDS = 1200
LOCAL_CAP = 12
WINDOW = 20
COOLDOWN = 20


class Budget:
    def __init__(self, cap=CAP):
        self.cap, self.queries, self.actions = cap, 0, 0

    def query(self):
        if self.queries >= self.cap:
            raise RuntimeError('query_budget')
        self.queries += 1

    def action(self):
        if self.actions >= self.cap:
            raise RuntimeError('action_budget')
        self.actions += 1


def checks(action, previous, evidence):
    a = np.asarray(action)
    contract = bool(a.shape == (7,) and np.isfinite(a).all() and np.all(np.abs(a) <= 1))
    delta = float(np.linalg.norm(a[:6]-np.asarray(previous)[:6])) if contract and previous is not None else 0.
    premature_open = bool(contract and evidence['grasped'] and not evidence['native_success'] and a[-1] < 0)
    return dict(command_contract=contract, command_continuity=delta <= .35,
        continuity_margin=.35-delta, grasp_release_consistency=not premature_open,
        future_collision='unknown', future_support='unknown', future_task_progress='unknown',
        scope='current command constraints only; no uncertainty or safety certificate')


class CandidateGate:
    def __init__(self, arm):
        assert arm in ARMS
        self.arm = arm
        self.history = deque(maxlen=WINDOW+1)
        self.actions = deque(maxlen=5)
        self.active = False
        self.used = 0
        self.cooldown_until = 0
        self.anchor = None
        self.progress_streak = 0
        self.transaction = 0

    def begin(self, step, evidence):
        self.history.append(evidence)
        old = self.history[0]
        item_delta = float(np.linalg.norm(np.array(evidence['item_position'])-old['item_position']))
        eef_delta = float(np.linalg.norm(np.array(evidence['eef_position'])-old['eef_position']))
        stalled = len(self.history) == WINDOW+1 and item_delta < .002 and eef_delta < .005
        unfinished_native = evidence['native_success'] and not evidence.get('completion', {}).get('complete', False)
        trigger = step >= 40 and (stalled or unfinished_native)
        events = []
        local_progress = False
        if self.active:
            anchor_delta = float(np.linalg.norm(np.array(evidence['item_position'])-self.anchor['item_position']))
            local_progress = anchor_delta >= .003 or evidence.get('completion', {}).get('candidate', False)
            # Reverification concerns measured progress and current contact, not predicted success.
            contact_ok = evidence.get('completion', {}).get('max_book_caddy_penetration', 0.) <= .001
            self.progress_streak = self.progress_streak+1 if local_progress and contact_ok else 0
            exit_reason = None
            if self.used >= LOCAL_CAP:
                exit_reason = 'local_action_cap'
            elif self.arm == 'simple_feedback' and local_progress:
                exit_reason = 'single_measured_progress'
            elif self.arm == 'verification' and self.progress_streak >= 3:
                exit_reason = 'three_measured_progress_contact_rechecks'
            if exit_reason:
                events.append(dict(type='local_exit', step=step, reason=exit_reason,
                    transaction=self.transaction, task_completion_claim=False))
                self.active = False
                self.cooldown_until = step+COOLDOWN
        if self.arm != 'nominal' and not self.active and step >= self.cooldown_until and trigger:
            self.active, self.used, self.progress_streak = True, 0, 0
            self.anchor = evidence
            self.transaction += 1
            events.append(dict(type='local_entry', step=step, transaction=self.transaction,
                reason='unfinished_native' if unfinished_native else 'observed_stagnation'))
        return dict(trigger=bool(trigger), stalled=bool(stalled), unfinished_native=bool(unfinished_native),
            item_motion_margin_m=item_delta-.002, eef_motion_margin_m=eef_delta-.005,
            window_observations=len(self.history), local_progress=bool(local_progress),
            progress_streak=self.progress_streak, active=self.active, used_before=self.used,
            transaction=self.transaction, events=events)

    def choose(self, candidates, evidence, monitor):
        previous = self.actions[-1] if self.actions else None
        audited = []
        for i, c in enumerate(candidates):
            audit = checks(c['action'], previous, evidence)
            novelty = min((float(np.linalg.norm(np.asarray(c['action'])[:6]-np.asarray(a)[:6]))
                           for a in self.actions), default=0.)
            audited.append(dict(index=i, source=c['source'], query=c['query'], action=c['action'],
                checks=audit, novelty_against_last_five_commands=novelty))
        chosen, reason, owner = 0, 'frozen_nominal_greedy', 'nominal'
        if self.active:
            owner = self.arm+'_candidate'
            eligible = [c for c in audited[1:] if c['checks']['command_contract']]
            if self.arm == 'verification':
                eligible = [c for c in audited if all(c['checks'][k] for k in
                    ('command_contract', 'command_continuity', 'grasp_release_consistency'))]
                reason = 'highest_command_novelty_among_current_checks_passed'
                if eligible:
                    chosen = max(eligible, key=lambda c: (c['novelty_against_last_five_commands'], -c['index']))['index']
                else:
                    reason = 'all_rejected_nominal_fallback_unverified'
            elif eligible:
                chosen, reason = eligible[0]['index'], 'first_contract_valid_model_sample'
            else:
                reason = 'sampling_budget_or_contract_fallback_nominal'
            self.used += 1
        if not audited[chosen]['checks']['command_contract']:
            raise RuntimeError('invalid_nominal_command_no_scripted_fallback')
        action = candidates[chosen]['action']
        self.actions.append(action)
        return action, dict(owner=owner, chosen=chosen, reason=reason, candidates=audited,
            used=self.used, local_cap=LOCAL_CAP, remaining_local=max(0, LOCAL_CAP-self.used),
            events=list(monitor['events']))


def executed_events(decision, step, action, nominal, transaction):
    """Call only after env.step returns, never when merely planning a handoff."""
    events = list(decision['events'])
    if any(e['type'] == 'local_exit' for e in events):
        assert decision['owner'] == 'nominal' and action == nominal
        events.append(dict(type='handoff_executed', step=step, transaction=transaction,
            owner='nominal', actual_action=action, reason='same_step_frozen_greedy_executed_after_local_exit'))
    return events


def observable_trace(step, arm, before, candidates, action, monitor, decision):
    """Reuse drawer decision_trace attribution; replace all task-specific predicates.

    Drawer-only placeholders never enter output or control. This preserves its actual
    nominal/executed action delta implementation, with honest book evidence semantics.
    """
    from adapters.libero_drawer import decision_trace
    fields = ('native_in retained bowl_supported bowl_gripper cabinet_gripper bowl_min bowl_max '
              'drawer_lower drawer_upper relative_speed angular_speed gripper_front_margin '
              'gripper_above_margin clear drawer_q alignment_error').split()
    measured = dict.fromkeys(fields, None)
    tracking = dict(phase='book_placement', milestones={}, entry=False, entry_streak=0,
        released=False, release_streak=0, clear=False, clear_streak=0, stagnation=monitor['stalled'])
    trace = decision_trace(step, arm, measured, tracking, candidates[0]['action'], action,
        dict(used=decision['used']))
    trace.update(schema=3, provenance='current/past privileged measured geometry and recorded contact; RGB+language actor',
        evidence=before, predicates=monitor, gate=decision,
        milestones_before=dict(native=before['native_success'], strict_complete=before.get('completion', {}).get('complete', False)),
        interpretation='observable execution evidence, not internal chain-of-thought or causal proof',
        candidate_screening='no future simulator rollout; command heuristics, not validated repair or uncertainty verifier')
    return trace
