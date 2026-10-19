"""Causal, privileged measured-verification wrapper for native microwave H1.

Call before(step, diagnostics), query the frozen VLA ONCE with its language,
source and seed, execute that unchanged mapped action, then call after(...).
Only after() records execution/handoff. This module has no model/simulator access
and generates no robot actions. Semantic subgoals are declared task definitions,
not model-generated plans; every task/repair action must come from external VLA.
Measured endpoint checks are not learned prediction or a safety certificate.
"""
from collections import deque
from copy import deepcopy
import json
import math


class NativeFramework:
    VERSION = 'native_microwave_measured_framework_v1'

    def __init__(self, mode='baseline', language='put the yellow and white mug in the microwave and close it'):
        if mode not in ('baseline', 'framework') or not isinstance(language, str) or not language.strip():
            raise ValueError('Expected baseline/framework and nonempty native language')
        self.mode, self.language = mode, language
        self._config = dict(version=self.VERSION, target='white_yellow_mug_1',
            receptacle='microwave_1', region='microwave_1_heating_region',
            action_cap=520, query_cap=520, local_cap=32, repair_cap=160,
            rechecks=3, cooldown=8, lift_m=.020, motion_m=.003,
            contact_window=12, acquisition_window=180,
            verification='privileged_current_past_endpoint_measurements',
            subgoal_source='declared_task_semantics_not_model_generated',
            learned_predictor=False, future_simulation=False,
            prompts=dict(acquire='pick up the yellow and white mug',
                transfer='put the yellow and white mug in the microwave',
                contain_release='release the yellow and white mug in the microwave',
                close='close the microwave'))
        self._history = deque(maxlen=181)
        self._current = None
        self._pending = None
        self._queries = self._actions = self._repair_actions = 0
        self._first_grasp = self._grasp_z = None
        self._ever_lifted = False
        self._in_streak = 0
        self._in_achieved = self._in_lost = self._elevated_lost = False
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
        elif self._in_achieved and state['in_goal'] is False:
            self._in_lost = True
        if (previous and previous['held_clear'] is True and state['grasp'] is False
                and state['in_goal'] is False):
            self._elevated_lost = True
        if state['grasp'] is True or state['in_goal'] is True:
            self._elevated_lost = False
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
        flags = dict(containment_lost=self._in_lost, entry_goal_lost=entry_loss,
            elevated_grasp_lost=self._elevated_lost, contact_constrained_holding=contact_stall,
            persistent_low_grasp=low_grasp, acquisition_no_progress=acquisition_stall)
        missing = [key for key, value in s.items() if value is None]
        return dict(flags=flags, trigger=next((key for key, value in flags.items() if value), None),
            evidence_through_step=s['step'], state=deepcopy(s),
            first_grasp_step=self._first_grasp, first_grasp_z=self._grasp_z,
            ever_lifted=self._ever_lifted, in_streak=self._in_streak,
            short_motion_diameter_m=short_motion, acquisition_motion_diameter_m=long_motion,
            missing=missing, uncertainty='missing_measurements' if missing else 'uncalibrated_measured_rules',
            future_task_progress='unknown', future_collision='unknown',
            provenance=cfg['verification'])

    @staticmethod
    def _subgoal(state):
        if state['in_goal'] is True:
            return 'contain_release' if state['grasp'] is True else 'close'
        return 'transfer' if state['held_clear'] is True else 'acquire'

    def _budget(self):
        return dict(actions=self._actions, queries=self._queries,
            repair_actions=self._repair_actions, remaining_actions=520-self._actions,
            remaining_queries=520-self._queries, remaining_repair=160-self._repair_actions)

    def before(self, step, diagnostics):
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
        events = []
        blocked = None
        if self.mode == 'framework' and self._active is None and monitor['trigger']:
            if self._handoff_pending is not None:
                blocked = 'pending_actual_handoff'
            elif step < self._cooldown_until:
                blocked = 'unverified_exit_cooldown'
            elif self._repair_actions >= self._config['repair_cap']:
                blocked = 'total_repair_budget'
            else:
                repair = dict(repair_id=len(self._transactions)+1, entry_step=step,
                    trigger=monitor['trigger'], subgoal=shadow_goal, used=0, streak=0,
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
            # Refeed after observed loss; never release/close an uncontained mug.
            goal = self._active['subgoal']
            if ((goal in ('contain_release', 'close') and self._current['in_goal'] is not True)
                    or (goal == 'transfer' and self._current['grasp'] is False
                        and self._current['in_goal'] is not True)
                    or (goal == 'acquire' and self._current['in_goal'] is True)):
                self._active['subgoal'] = shadow_goal
                self._active['streak'] = 0
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
            handoff_repair_id=self._handoff_pending,
            local_used=active['used'] if active else 0, local_cap=self._config['local_cap'],
            shadow_only=self.mode == 'baseline', shadow_subgoal=shadow_goal,
            intervention_blocked=blocked, budget=self._budget(), events=events,
            subgoal_source=self._config['subgoal_source'], verification=self._config['verification'])
        self._pending = deepcopy(decision)
        return decision

    def _progress(self, goal):
        s = self._current
        if goal == 'acquire':
            return s['held_clear'] is True
        if goal == 'transfer':
            return s['in_goal'] is True
        if goal == 'contain_release':
            return s['in_goal'] is True and s['grasp'] is False and s['mug_gripper'] is False
        return s['native_complete'] is True

    def after(self, step, diagnostics, action, decision):
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
        if expected['owner'] == 'repair':
            repair = self._active
            repair['used'] += 1
            self._repair_actions += 1
            for key in repair['preserve']:
                if measured[key] is False and key not in repair['lost_entry_goals']:
                    repair['lost_entry_goals'].append(key)
                if measured[key] is None:
                    repair['preservation_unknown'] = True
            progress = self._progress(repair['subgoal'])
            preserved = not repair['lost_entry_goals'] and not repair['preservation_unknown']
            repair['streak'] = repair['streak']+1 if progress and preserved else 0
            verification.update(progress=progress, consecutive=repair['streak'],
                preserved=preserved, lost_entry_goals=list(repair['lost_entry_goals']),
                preservation_unknown=repair['preservation_unknown'])
            reason = None
            # A capped exit never becomes a verified repair, even on a tied check.
            if repair['used'] >= self._config['local_cap']:
                reason = 'local_budget_exit_unverified'
            elif self._repair_actions >= self._config['repair_cap']:
                reason = 'total_repair_budget_exit_unverified'
            elif repair['streak'] >= self._config['rechecks']:
                reason = 'measured_progress_verified'
                repair['verified'] = True
                verification['repair_verified'] = True
                events.append(dict(type='repair_verified', step=step, repair_id=repair['repair_id']))
            elif measured['native_complete']:
                reason = 'native_completion_before_three_rechecks'
            elif self._actions >= self._config['action_cap']:
                reason = 'episode_budget_exit_unverified'
            if reason:
                repair.update(exit_step=step, exit_reason=reason)
                events.append(dict(type='local_exit', step=step, repair_id=repair['repair_id'],
                    reason=reason, verified=repair['verified']))
                self._active = None
                if not measured['native_complete'] and self._actions < self._config['action_cap']:
                    self._handoff_pending = repair['repair_id']
                if not repair['verified']:
                    self._cooldown_until = step+1+self._config['cooldown']
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
                if repair['subgoal'] == 'acquire' else measured['in_goal'] is False)
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
