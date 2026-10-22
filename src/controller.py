"""Causal, privileged measured-verification wrapper for native microwave H1.

Call before(step, diagnostics), query the frozen VLA ONCE with its language,
source and seed, execute that unchanged mapped action, then call after(...).
Only after() records execution/handoff. This module has no model/simulator access
and generates no robot actions. Semantic subgoals are declared task definitions,
not model-generated plans. Default actions come from the external VLA; explicit
v7 recovery mode instead delegates oracle skills to the runner and records their
feedback. This controller never produces numeric actions itself.
Measured endpoint checks are not learned prediction or a safety certificate.
"""
from collections import deque
from copy import deepcopy
import json
import math


class NativeFramework:
    VERSION = 'native_microwave_measured_framework_v4_prompt_refine'

    def __init__(self, mode='baseline', language='put the yellow and white mug in the microwave and close it',
                  variant='v2', fusion_mode='off', joint_mode=False, recovery_options=False):
        if mode not in ('baseline', 'framework') or not isinstance(language, str) or not language.strip():
            raise ValueError('Expected baseline/framework and nonempty native language')
        self.mode, self.language = mode, language
        if fusion_mode not in ('off', 'log', 'gate') or (mode == 'baseline' and fusion_mode == 'gate'):
            raise ValueError('Expected off/log/gate fusion; baseline cannot use a gate')
        self.fusion_mode = fusion_mode
        self.joint_mode = joint_mode
        self.recovery_options = recovery_options
        if recovery_options and not joint_mode:
            raise ValueError('Recovery options require the joint runtime')
        if joint_mode and (mode != 'framework' or fusion_mode != 'gate'):
            raise ValueError('Joint integration requires framework arm and causal fusion gating')
        if joint_mode:
            self.VERSION = 'joint_visual_physical_framework_v6_h1'
        if recovery_options:
            self.VERSION = 'joint_framework_v7_oracle_recovery_options'
        if variant not in ('v2', 'prompt_only', 'gate_only', 'combined', 'prompt_place', 'prompt_deepen', 'prompt_settle'):
            raise ValueError('Unknown frozen factorial variant')
        self._config = dict(version=self.VERSION, variant=variant,
            prompt_factor=variant in ('prompt_only', 'combined', 'prompt_place', 'prompt_deepen', 'prompt_settle'),
            resumption_factor=variant in ('gate_only', 'combined'),
            resumption_motion_m=.003, resumption_endpoints=3,
            target='white_yellow_mug_1',
            receptacle='microwave_1', region='microwave_1_heating_region',
            action_cap=520, query_cap=520, local_cap=32, repair_cap=160,
            rechecks=3, cooldown=8, lift_m=.020, motion_m=.003,
            contact_window=12, acquisition_window=180,
            release_radius_m=.050, release_flight_steps=3, retry_motion_m=.020,
            recovery_contracts=dict(acquire=['held_clear', 'contained'],
                transfer=['contained'], contain_release=['contained_detached'],
                close=['native_complete']),
            verification='privileged_current_past_endpoint_measurements',
            subgoal_source='declared_task_semantics_not_model_generated',
            learned_predictor=False, future_simulation=False,
            prompts=dict(acquire='pick up the yellow and white mug',
                transfer='put the yellow and white mug in the microwave',
                contain_release='release the yellow and white mug in the microwave',
                close='close the microwave'))
        if variant == 'prompt_place':
            self._config['prompts']['contain_release'] = 'place the yellow and white mug fully inside the microwave'
        elif variant == 'prompt_deepen':
            self._config['prompts']['contain_release'] = 'push the yellow and white mug to the back of the microwave and place it inside'
        elif variant == 'prompt_settle':
            self._config['prompts']['contain_release'] = 'place the yellow and white mug fully inside the microwave and leave it there'
        elif self._config['prompt_factor']:
            # Preserve the placement objective while contact remains. This is
            # still one unchanged model proposal, never a gripper override.
            self._config['prompts']['contain_release'] = self._config['prompts']['transfer']
        if fusion_mode != 'off':
            self._config['fusion_mode'] = fusion_mode
        if joint_mode:
            self._config.update(joint_mode=True, subgoal_source='pretrained_visual_episodic_memory_physical_constraints',
                learned_risk=True, repair_generation='frozen_policy_h1_remaining_execution_v1')
        if recovery_options:
            from src.repair.recovery_options import CONFIG
            self._config.update(recovery_options=deepcopy(CONFIG),
                repair_generation='oracle_closed_loop_options_with_matched_unused_vla_queries')
        self._history = deque(maxlen=181)
        self._current = None
        self._pending = None
        self._queries = self._actions = self._repair_actions = 0
        self._first_grasp = self._grasp_z = None
        self._ever_lifted = False
        self._in_streak = 0
        self._in_achieved = self._in_lost = self._elevated_lost = False
        self._containment_anchor = None
        self._last_in_step = self._release_started = None
        self._failed_contexts = {}
        self._milestones = {}
        self._active = self._handoff_pending = None
        self._transactions = []
        self._cooldown_until = 0
        self._native_completed = False

    @property
    def config(self):
        return deepcopy(self._config)

    @property
    def summary(self):
        return dict(version=self.VERSION, arm=self.mode, config=self.config,
            queries=self._queries, actions=self._actions,
            repair_actions=self._repair_actions, native_complete=self._native_completed,
            milestones=deepcopy(self._milestones),
            repairs=deepcopy(self._transactions))

    def _measure(self, diagnostics):
        cfg = self._config
        goals = {}
        for key, value in diagnostics.get('predicates', {}).items():
            clause = json.loads(key)
            if clause == ['in', cfg['target'], cfg['region']]:
                goals['in'] = value if isinstance(value, bool) else None
            elif clause == ['close', cfg['receptacle']]:
                goals['close'] = value if isinstance(value, bool) else None
        inside, closed = goals.get('in'), goals.get('close')
        complete = inside and closed if inside is not None and closed is not None else None
        native = diagnostics.get('native_success')
        if complete is not None and isinstance(native, bool) and native != complete:
            raise ValueError('Native success disagrees with microwave In AND Close')
        grasp = diagnostics.get('grasped', {}).get(cfg['target'])
        grasp = grasp if isinstance(grasp, bool) else None
        pos = diagnostics.get('object_observations', {}).get(cfg['target']+'_pos')
        if pos is not None:
            try:
                pos = [float(v) for v in pos]
                if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
                    pos = None
            except (TypeError, ValueError):
                pos = None
        contacts = diagnostics.get('contacts_endpoint')
        flags = dict(table=None, mug_microwave=None, gripper_microwave=None, mug_gripper=None)
        if isinstance(contacts, list):
            flags = dict.fromkeys(flags, False)
            for contact in contacts:
                a, b = contact.get('geom1'), contact.get('geom2')
                if not isinstance(a, str) or not isinstance(b, str):
                    flags = dict.fromkeys(flags, None)
                    break
                for left, right in ((a, b), (b, a)):
                    mug = left.startswith(cfg['target']+'_')
                    flags['table'] |= mug and right.startswith('table')
                    flags['mug_microwave'] |= mug and right.startswith(cfg['receptacle']+'_')
                    flags['mug_gripper'] |= mug and right.startswith('gripper')
                    flags['gripper_microwave'] |= left.startswith('gripper') and right.startswith(cfg['receptacle']+'_')
        return dict(in_goal=inside, close_goal=closed, native_complete=complete,
            grasp=grasp, position=pos, **flags)

    def _observe(self, step, measured):
        previous = self._current
        state = dict(measured, step=step)
        if self._first_grasp is None and state['grasp'] is True:
            self._first_grasp = step
            # A missing first-grasp height cannot be silently replaced later.
            self._grasp_z = None if state['position'] is None else state['position'][2]
            self._milestones['first_grasp'] = step
        height = None if state['position'] is None or self._grasp_z is None else state['position'][2]-self._grasp_z
        state['rise_m'] = height
        self._ever_lifted |= height is not None and height > self._config['lift_m']
        state['held_clear'] = (state['grasp'] and height > self._config['lift_m'] and not state['table']
            if state['grasp'] is not None and height is not None and state['table'] is not None else None)
        if state['held_clear']:
            self._milestones.setdefault('held_clear', step)
        # Initial snapshot is not an executed-action dwell observation.
        self._in_streak = self._in_streak+1 if step >= 0 and state['in_goal'] is True else 0
        if self._in_streak >= self._config['rechecks']:
            self._in_achieved = True
            self._in_lost = False
            self._milestones.setdefault('durable_in', step)
            self._containment_anchor = state['position']
        elif self._in_achieved and state['in_goal'] is False:
            self._in_lost = True
        if (previous and previous['held_clear'] is True and state['grasp'] is False
                and state['in_goal'] is False):
            self._elevated_lost = True
        if state['grasp'] is True or state['in_goal'] is True:
            self._elevated_lost = False
        if state['in_goal'] is True:
            self._last_in_step = step
        # A release from recently attained containment is a transition, not an
        # automatic reacquisition request. Flight grace applies only to this
        # event; losses before any containment still trigger immediately.
        if (previous and previous['grasp'] is True and state['grasp'] is False
                and self._in_achieved and self._last_in_step is not None
                and step-self._last_in_step <= self._config['rechecks']):
            self._release_started = step
        anchor = self._containment_anchor
        distance = (math.dist(anchor, state['position'])
            if anchor is not None and state['position'] is not None else None)
        near = distance is not None and distance <= self._config['release_radius_m']
        flight = (self._release_started is not None
            and step-self._release_started < self._config['release_flight_steps'])
        release_context = (self._release_started is not None and near
            and state['grasp'] is False and state['table'] is False
            and (flight or state['mug_microwave'] is True))
        state['release_transition'] = release_context
        state['containment_anchor_distance_m'] = distance
        # Once contact/near evidence is lost, a later unrelated contact cannot
        # resurrect the exemption without a fresh release event.
        if self._release_started is not None and not release_context:
            self._release_started = None
        if state['native_complete']:
            self._milestones.setdefault('native_complete', step)
        self._history.append(state)
        self._current = state

    @staticmethod
    def _diameter(states):
        positions = [s['position'] for s in states]
        if any(p is None for p in positions):
            return None
        return max((math.dist(a, b) for i, a in enumerate(positions) for b in positions[i+1:]), default=0.)

    def _monitor(self):
        cfg, s = self._config, self._current
        history = list(self._history)
        short = history[-(cfg['contact_window']+1):]
        short_motion = self._diameter(short)
        contact_stall = (len(short) == cfg['contact_window']+1
            and all(x['held_clear'] is True and x['in_goal'] is False
                    and (x['mug_microwave'] is True or x['gripper_microwave'] is True) for x in short)
            and short_motion is not None and short_motion < cfg['motion_m'])
        long_motion = None
        acquisition_stall = False
        if len(history) == cfg['acquisition_window']+1 and s['grasp'] is False:
            long_motion = self._diameter(history)
            acquisition_stall = (all(x['grasp'] is False and x['in_goal'] is False for x in history)
                and long_motion is not None and long_motion < cfg['motion_m'])
        low_grasp = (s['grasp'] is True and s['in_goal'] is False and self._grasp_z is not None
            and s['rise_m'] is not None and not self._ever_lifted
            and s['step']-self._first_grasp >= cfg['acquisition_window'])
        entry_loss = (self._active is not None and any(
            s[key] is False for key in self._active['preserve']))
        flags = dict(containment_lost=self._in_lost and not s['release_transition'], entry_goal_lost=entry_loss,
            elevated_grasp_lost=self._elevated_lost and not s['release_transition'], contact_constrained_holding=contact_stall,
            persistent_low_grasp=low_grasp, acquisition_no_progress=acquisition_stall)
        missing = [key for key, value in s.items() if value is None]
        return dict(flags=flags, trigger=next((key for key, value in flags.items() if value), None),
            evidence_through_step=s['step'], state=deepcopy(s),
            first_grasp_step=self._first_grasp, first_grasp_z=self._grasp_z,
            ever_lifted=self._ever_lifted, in_streak=self._in_streak,
            raw_containment_lost=self._in_lost, raw_elevated_grasp_lost=self._elevated_lost,
            release_transition=s['release_transition'], release_started=self._release_started,
            short_motion_diameter_m=short_motion, acquisition_motion_diameter_m=long_motion,
            missing=missing, uncertainty='missing_measurements' if missing else 'uncalibrated_measured_rules',
            future_task_progress='unknown', future_collision='unknown',
            provenance=cfg['verification'])

    @staticmethod
    def _subgoal(state):
        if state['in_goal'] is True:
            detached = state['grasp'] is False and state['mug_gripper'] is False
            return 'close' if detached else 'contain_release'
        return 'transfer' if state['held_clear'] is True else 'acquire'

    def _retry_allowed(self, trigger):
        """Require durable changed evidence since the last failure of this event.

        One-frame grasp/contact flicker and the passage of a cooldown alone do
        not justify spending another repair transaction. This only gates entry;
        all original events and their measurements remain visible in the trace.
        """
        failed = self._failed_contexts.get(trigger)
        if failed is None:
            return True
        recent = list(self._history)[-self._config['rechecks']:]
        if len(recent) < self._config['rechecks'] or recent[0]['step'] <= failed['step']:
            return False
        for key in ('grasp', 'held_clear', 'in_goal', 'close_goal', 'table', 'mug_microwave'):
            value = recent[0][key]
            if (value is not None and failed[key] is not None and value != failed[key]
                    and all(s[key] == value for s in recent)):
                return True
        return (failed['position'] is not None and all(s['position'] is not None
            and math.dist(s['position'], failed['position']) >= self._config['retry_motion_m']
            for s in recent))

    def _budget(self):
        return dict(actions=self._actions, queries=self._queries,
            repair_actions=self._repair_actions, remaining_actions=520-self._actions,
            remaining_queries=520-self._queries, remaining_repair=160-self._repair_actions)

    def before(self, step, diagnostics, joint_evidence=None):
        """Return one greedy H1 request; call after only once that action executes."""
        if self._pending is not None or step != self._actions:
            raise ValueError('Expected sequential before/after pairs starting at step zero')
        if self._queries >= self._config['query_cap'] or step >= self._config['action_cap']:
            raise RuntimeError('episode_budget_exhausted')
        measured = self._measure(diagnostics)
        if self._current is None:
            self._observe(-1, measured)
        elif any(self._current[k] != v for k, v in measured.items()):
            raise ValueError('before diagnostics differ from last executed endpoint')
        if measured['native_complete']:
            raise RuntimeError('native_complete_no_further_action')
        monitor = self._monitor()
        shadow_goal = self._subgoal(self._current)
        native_goal = shadow_goal
        if self.joint_mode:
            if (not isinstance(joint_evidence, dict) or
                    joint_evidence.get('schema') != 'joint_visual_physical_prediction_memory_risk_h1_v1' or
                    joint_evidence.get('observation_index') != step or
                    joint_evidence.get('selected_stage') not in self._config['recovery_contracts']):
                raise ValueError('Fresh joint visual/predictive/memory/risk evidence is required')
            monitor['joint'] = deepcopy(joint_evidence)
            monitor['native_trigger'] = monitor['trigger']
            if monitor['trigger'] is None:
                monitor['trigger'] = joint_evidence['trigger']
            shadow_goal = joint_evidence['selected_stage']
            if self.recovery_options and joint_evidence.get('recovery_admission', {}).get('schema') != 'oracle_recovery_options_v1':
                raise ValueError('Fresh recovery admission evidence required')
        events = []
        blocked = None
        if self.mode == 'framework' and self._active is None and monitor['trigger']:
            if self._handoff_pending is not None:
                blocked = 'pending_actual_handoff'
            elif step < self._cooldown_until:
                blocked = 'unverified_exit_cooldown'
            elif self._repair_actions >= self._config['repair_cap']:
                blocked = 'total_repair_budget'
            elif self.recovery_options and not joint_evidence['recovery_admission']['allowed']:
                blocked = 'same_context_all_options_failed'
            elif not self.recovery_options and not self._retry_allowed(monitor['trigger']):
                blocked = 'failed_repair_requires_changed_evidence'
            else:
                repair = dict(repair_id=len(self._transactions)+1, entry_step=step,
                    trigger=monitor['trigger'], subgoal=shadow_goal, used=0, streak=0,
                    recovery_goal=native_goal,
                    acceptance=list(self._config['recovery_contracts'][native_goal]),
                     acceptance_streaks={k: 0 for k in self._config['recovery_contracts'][native_goal]},
                    local_recovery_step=None, resumption_by=None,
                    verified_by=None, completed_during_repair=False,
                    preserve=[k for k in ('in_goal', 'close_goal') if self._current[k] is True],
                    lost_entry_goals=[], preservation_unknown=False,
                    verified=False, exit_step=None, exit_reason=None,
                    handoff_step=None, recurrence_step=None, post_handoff_goal_loss_step=None,
                    completed_after_handoff=False)
                self._transactions.append(repair)
                self._active = repair
                events.append(dict(type='local_entry', step=step, repair_id=repair['repair_id'],
                    trigger=monitor['trigger'], subgoal=shadow_goal))
        if self._active is not None:
            # The action prompt may follow current evidence, but acceptance is
            # frozen at entry. In particular, attaining In cannot move an
            # acquisition contract to a three-native-success closing contract.
            goal = self._active['subgoal']
            if ((self.joint_mode and goal != shadow_goal)
                    or (goal in ('contain_release', 'close') and self._current['in_goal'] is not True)
                    or (goal == 'transfer' and self._current['grasp'] is False
                        and self._current['in_goal'] is not True)
                    or (self._current['in_goal'] is True and goal != shadow_goal)):
                self._active['subgoal'] = shadow_goal
                events.append(dict(type='local_goal_refeed', step=step,
                    repair_id=self._active['repair_id'], previous=goal, subgoal=shadow_goal))
        active = self._active
        goal = active['subgoal'] if active else shadow_goal
        owner = 'repair' if active else 'nominal'
        self._queries += 1
        decision = dict(version=self.VERSION, arm=self.mode, step=step,
            language=self._config['prompts'][goal] if active else self.language,
            source='greedy', seed=0, owner=owner,
            mode='contact_transition' if active else 'nominal_tracking', subgoal=goal,
            trigger=monitor['trigger'], evidence=monitor, milestones=deepcopy(self._milestones),
            repair_id=active['repair_id'] if active else None,
            recovery_goal=active['recovery_goal'] if active else None,
            recovery_acceptance=list(active['acceptance']) if active else [],
            handoff_repair_id=self._handoff_pending,
            local_used=active['used'] if active else 0, local_cap=self._config['local_cap'],
            shadow_only=self.mode == 'baseline', shadow_subgoal=shadow_goal,
            intervention_blocked=blocked, budget=self._budget(), events=events,
            subgoal_source=self._config['subgoal_source'], verification=self._config['verification'])
        self._pending = deepcopy(decision)
        return decision

    def _progress(self, criterion):
        s = self._current
        if criterion == 'held_clear':
            return s['held_clear'] is True
        if criterion == 'contained':
            return s['in_goal'] is True
        if criterion == 'contained_detached':
            return s['in_goal'] is True and s['grasp'] is False and s['mug_gripper'] is False
        if criterion == 'native_complete':
            return s['native_complete'] is True
        raise ValueError('Unknown recovery criterion: '+criterion)

    def _resumption(self, criterion):
        """Measured candidate readiness, not a forecast of nominal success.

        Each alternative must hold over the same three post-entry endpoints;
        mixing held and released endpoints cannot create a passing window.
        Contact alone is not a support or settling certificate.
        """
        cfg = self._config
        recent = list(self._history)[-cfg['resumption_endpoints']:]
        window = (len(recent) == cfg['resumption_endpoints']
            and recent[0]['step'] >= self._active['entry_step'])
        motion = self._diameter(recent) if window else None
        stable = motion is not None and motion <= cfg['resumption_motion_m']
        alternatives = {}
        if criterion == 'held_clear':
            alternatives['free_held'] = window and all(
                s['held_clear'] is True and s['in_goal'] is False
                and s['mug_microwave'] is False and s['gripper_microwave'] is False
                for s in recent)
        if criterion in ('contained', 'contained_detached'):
            if criterion == 'contained':
                alternatives['contained_held'] = window and all(
                    s['in_goal'] is True and s['held_clear'] is True
                    and s['mug_gripper'] is True for s in recent)
            alternatives['contained_released_contact'] = window and all(
                s['in_goal'] is True and s['grasp'] is False
                and s['mug_gripper'] is False and s['mug_microwave'] is True
                and s['table'] is False for s in recent)
        if criterion == 'native_complete':
            alternatives['native_complete'] = window and all(s['native_complete'] is True for s in recent)
        ready_by = next((name for name, passed in alternatives.items() if passed and stable), None)
        return dict(ready=ready_by is not None, ready_by=ready_by,
            motion_diameter_m=motion, stable=stable, complete_window=window,
            alternatives=alternatives)

    def preview_fusion_state(self, diagnostics):
        """Current endpoint features without mutating history or using future data."""
        state = self._measure(diagnostics)
        grasp_z = self._grasp_z
        if self._first_grasp is None and state['grasp'] is True:
            grasp_z = None if state['position'] is None else state['position'][2]
        height = None if state['position'] is None or grasp_z is None else state['position'][2]-grasp_z
        state['rise_m'] = height
        state['held_clear'] = (state['grasp'] and height > self._config['lift_m'] and not state['table']
            if state['grasp'] is not None and height is not None and state['table'] is not None else None)
        return state

    def after(self, step, diagnostics, action, decision, handoff_evidence=None, recovery_feedback=None):
        """Record an already executed VLA action and its fresh measured endpoint.

        The runner owns token/action provenance and env.step; this pure module
        validates only the seven-coordinate finite action contract. It does not
        infer execution merely from a decision, or regenerate/modify the action.
        """
        expected = self._pending
        if expected is None or step != self._actions or any(decision.get(k) != v for k, v in expected.items()):
            raise ValueError('after requires the unchanged pending decision and executed step')
        if len(action) != 7 or not all(math.isfinite(float(v)) for v in action):
            raise ValueError('Expected a finite seven-coordinate executed VLA action')
        measured = self._measure(diagnostics)
        self._observe(step, measured)
        self._actions += 1
        events = deepcopy(expected['events'])
        verification = dict(scope=self._config['verification'], repair_id=expected['repair_id'],
            progress=False, consecutive=0, required=self._config['rechecks'],
            repair_verified=False, native_complete=measured['native_complete'] is True)
        if recovery_feedback is not None:
            if not self.recovery_options or recovery_feedback.get('schema') != 'oracle_recovery_options_v1':
                raise ValueError('Unexpected recovery feedback')
            verification['recovery_feedback'] = deepcopy(recovery_feedback)
        if expected['owner'] == 'repair':
            repair = self._active
            repair['used'] += 1
            self._repair_actions += 1
            if self._in_streak >= self._config['rechecks'] and 'in_goal' not in repair['preserve']:
                repair['preserve'].append('in_goal')
                events.append(dict(type='preservation_promoted', step=step,
                    repair_id=repair['repair_id'], predicate='in_goal'))
            for key in repair['preserve']:
                if measured[key] is False and key not in repair['lost_entry_goals']:
                    repair['lost_entry_goals'].append(key)
                if measured[key] is None:
                    repair['preservation_unknown'] = True
            checks = {key: self._progress(key) for key in repair['acceptance']}
            progress = any(checks.values())
            preserved = not repair['lost_entry_goals'] and not repair['preservation_unknown']
            for key, passed in checks.items():
                repair['acceptance_streaks'][key] = (repair['acceptance_streaks'][key]+1
                    if passed and preserved else 0)
            repair['streak'] = max(repair['acceptance_streaks'].values())
            verification.update(progress=progress, consecutive=repair['streak'],
                recovery_goal=repair['recovery_goal'], acceptance_checks=checks,
                acceptance_streaks=deepcopy(repair['acceptance_streaks']),
                preserved=preserved, lost_entry_goals=list(repair['lost_entry_goals']),
                preservation_unknown=repair['preservation_unknown'])
            local_candidates = [k for k, n in repair['acceptance_streaks'].items()
                if n >= self._config['rechecks']]
            if local_candidates and repair['local_recovery_step'] is None:
                repair['local_recovery_step'] = step
                events.append(dict(type='local_condition_met', step=step,
                    repair_id=repair['repair_id'], criteria=local_candidates))
            readiness = {key: self._resumption(key) for key in repair['acceptance']}
            eligible = [k for k in local_candidates if not self._config['resumption_factor']
                or readiness[k]['ready']]
            verification.update(resumption_required=self._config['resumption_factor'],
                resumption_checks=readiness, local_condition_met=bool(local_candidates),
                exit_condition_met=bool(eligible))
            fusion_pass = True
            if self.fusion_mode != 'off':
                evidence = deepcopy(handoff_evidence)
                valid = isinstance(evidence, dict) and evidence.get('stage') == repair['recovery_goal']
                if valid:
                    probability, threshold = evidence.get('probability'), evidence.get('threshold')
                    valid = (isinstance(probability, (int, float)) and math.isfinite(probability)
                        and 0 <= probability <= 1 and isinstance(threshold, (int, float))
                        and math.isfinite(threshold) and 0 <= threshold <= 1
                        and evidence.get('feature_schema') == 'controller_state_values12_validity12_v2'
                        and isinstance(evidence.get('model_sha256'), str) and len(evidence['model_sha256']) == 64)
                accepted = bool(valid and evidence['probability'] >= evidence['threshold'])
                if self.joint_mode:
                    accepted = accepted and evidence.get('joint_prediction_veto') is False and evidence.get('joint_risk_veto') is False
                fusion_pass = self.fusion_mode != 'gate' or accepted
                verification['fusion'] = dict(mode=self.fusion_mode, evidence=evidence,
                    evidence_valid=valid, accepted=accepted,
                    blocked_verified_exit=bool(eligible) and not fusion_pass)
            reason = None
            # A capped exit never becomes a verified repair, even on a tied check.
            if measured['native_complete']:
                reason = 'native_completion_during_repair'
                repair['completed_during_repair'] = True
                events.append(dict(type='native_completion_during_repair', step=step,
                    repair_id=repair['repair_id']))
            elif repair['used'] >= self._config['local_cap']:
                reason = 'local_budget_exit_unverified'
            elif self._repair_actions >= self._config['repair_cap']:
                reason = 'total_repair_budget_exit_unverified'
            elif eligible and fusion_pass:
                reason = 'measured_progress_verified'
                repair['verified'] = True
                repair['verified_by'] = eligible[0]
                repair['resumption_by'] = (readiness[eligible[0]]['ready_by']
                    if self._config['resumption_factor'] else 'not_required_v2_contract')
                verification['repair_verified'] = True
                verification['verified_by'] = repair['verified_by']
                events.append(dict(type='repair_verified', step=step, repair_id=repair['repair_id']))
            elif self._actions >= self._config['action_cap']:
                reason = 'episode_budget_exit_unverified'
            elif self.recovery_options and recovery_feedback and recovery_feedback['exhausted']:
                reason = 'recovery_options_exhausted_unverified'
            if reason:
                repair.update(exit_step=step, exit_reason=reason)
                events.append(dict(type='local_exit', step=step, repair_id=repair['repair_id'],
                    reason=reason, verified=repair['verified']))
                self._active = None
                if not measured['native_complete'] and self._actions < self._config['action_cap']:
                    self._handoff_pending = repair['repair_id']
                if not repair['verified']:
                    self._cooldown_until = step+1+self._config['cooldown']
                    self._failed_contexts[repair['trigger']] = deepcopy(self._current)
        elif expected['handoff_repair_id'] is not None:
            repair = self._transactions[expected['handoff_repair_id']-1]
            repair['handoff_step'] = step
            self._handoff_pending = None
            events.append(dict(type='handoff_executed', step=step, repair_id=repair['repair_id'],
                verified_repair=repair['verified'], owner='nominal',
                language=expected['language'], source='greedy', seed=0))
        monitor = self._monitor()
        for repair in self._transactions:
            if repair['handoff_step'] is None:
                continue
            goal_lost = (self._current['held_clear'] is False and measured['in_goal'] is False
                if repair['verified_by'] == 'held_clear' else measured['in_goal'] is False)
            if repair['verified'] and goal_lost and repair['post_handoff_goal_loss_step'] is None:
                repair['post_handoff_goal_loss_step'] = step
                events.append(dict(type='post_handoff_goal_loss', step=step,
                    repair_id=repair['repair_id'], subgoal=repair['subgoal']))
            if monitor['trigger'] and repair['recurrence_step'] is None:
                repair['recurrence_step'] = step
                events.append(dict(type='post_handoff_recurrence', step=step,
                    repair_id=repair['repair_id'], trigger=monitor['trigger']))
            if measured['native_complete'] and not repair['completed_after_handoff']:
                repair['completed_after_handoff'] = True
                events.append(dict(type='native_completion_after_handoff', step=step,
                    repair_id=repair['repair_id'], verified_repair=repair['verified']))
        if measured['native_complete'] and not self._native_completed:
            self._native_completed = True
            events.append(dict(type='native_completion', step=step))
        self._pending = None
        return dict(version=self.VERSION, step=step, events=events, verification=verification,
            evidence=monitor, milestones=deepcopy(self._milestones), budget=self._budget(),
            native_complete=self._native_completed, handoff_pending=self._handoff_pending)
