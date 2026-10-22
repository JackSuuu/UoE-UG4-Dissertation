"""Oracle-state recovery skills, NOT generated actions or learned physics.

Two acquisition options share one bounded transaction: probe a grasp by lifting,
or release/retract/reapproach/reclose/lift. Absolute Cartesian targets are
converted using the live OSC scale; rotations are held, not overwritten with
policy deltas. A counterfactual VLA proposal is still queried each executed step
for matched query costs, but is explicitly not the source of skill actions.
"""
from copy import deepcopy
import math
import numpy as np

SCHEMA = 'oracle_recovery_options_v1'
CONFIG = dict(schema=SCHEMA, window=6, min_progress_m=.003, max_step_m=.008,
    lift_m=.040, retreat_m=.040, lift_cap=8, regrasp_cap=24,
    context_reset_m=.020, reset_dwell=3, reach_tolerance_m=.006)


class RecoveryOptions:
    def __init__(self, config, controller):
        self.target = config['target']
        self.scale = np.asarray(controller['action_scale'][:3], dtype=float)
        if (self.scale.shape != (3,) or not np.isfinite(self.scale).all() or
                np.any(self.scale <= 0) or controller.get('use_delta') is not True or
                any(np.any(np.asarray(controller[k]) != 0) for k in
                    ('action_input_transform', 'action_output_transform'))):
            raise ValueError('Recovery needs explicit positive zero-offset delta OSC scaling')
        self.failed, self.anchor = set(), None
        self.changed_streak = 0
        self.last_admission = -1
        self.transaction = self.option = None
        self.used = self.phase_used = 0
        self.phase = None
        self.samples = []
        self.events = []
        self.pending = None
        self.finished = False

    def _available(self, state):
        return [n for n in ('lift_probe', 'regrasp') if n not in self.failed and
            (n != 'lift_probe' or state.get('grasp') is True)]

    def _state(self, diagnostics):
        p = diagnostics.get('object_observations', {}).get(self.target+'_pos')
        e = diagnostics.get('eef_position')
        if p is None or e is None:
            raise ValueError('Missing oracle object/EEF positions; no silent skill fallback')
        p, e = np.asarray(p, dtype=float), np.asarray(e, dtype=float)
        if p.shape != (3,) or e.shape != (3,) or not np.isfinite([p, e]).all():
            raise ValueError('Invalid oracle recovery positions')
        return p, e

    def admission(self, state, step):
        if step != self.last_admission+1:
            raise ValueError('Recovery admission must be sequential')
        self.last_admission = step
        p = state.get('position')
        if self.anchor is not None and p is not None:
            moved = math.dist(p, self.anchor) >= CONFIG['context_reset_m']
            # Grasp/contact flicker and alternate trigger names do not reset.
            self.changed_streak = self.changed_streak+1 if moved else 0
            if self.changed_streak >= CONFIG['reset_dwell']:
                self.failed.clear()
                self.anchor = list(p)
                self.changed_streak = 0
                self.events.append(dict(type='durable_physical_context_change', step=step))
        supported = state.get('in_goal') is False and state.get('held_clear') is not True
        available = self._available(state)
        return dict(schema=SCHEMA, allowed=not supported or bool(available),
            available=available, failed=sorted(self.failed), acquisition_supported=supported,
            reason=None if not supported or available else 'same_context_all_options_failed')

    def _start(self, name, p, e, step):
        self.option, self.used, self.phase_used = name, 0, 0
        self.phase = 'lift' if name == 'lift_probe' else 'release'
        self.goal = e + np.array([0., 0., CONFIG['lift_m']])
        self.initial_z = float(p[2])
        self.last_coupling = None
        self.grasp_diagnosis = 'unresolved'
        self.offset = e-p
        # Keep the measured grasp approach, not an object-name-specific pose.
        self.origin_eef = e.copy()
        self.samples = []
        self.finished = False
        self.events.append(dict(type='option_started', option=name, step=step))

    def command(self, decision, diagnostics, proposal):
        if self.pending is not None:
            raise ValueError('Unobserved recovery action')
        p, e = self._state(diagnostics)
        state = decision['evidence']['state']
        if decision['owner'] != 'repair':
            self.transaction = None
            return list(proposal), None
        if self.transaction != decision['repair_id']:
            self.transaction = decision['repair_id']
            self.option = None
            self.finished = False
            if self.anchor is None:
                self.anchor = p.tolist()
        if state.get('in_goal') is True:
            return list(proposal), None
        if state.get('held_clear') is True:
            # Never release an achieved containment or a raised object. Hold
            # acquire recovery while existing measured+fusion verification runs.
            if decision['recovery_goal'] != 'acquire':
                return list(proposal), None
            action = [0., 0., 0., 0., 0., 0., 1.]
            record = dict(schema=SCHEMA, option='hold_recovered', phase='hold',
                step=decision['step'], target_eef=e.tolist(), eef=e.tolist(),
                object_position=p.tolist(), scale=self.scale.tolist(), max_step_m=CONFIG['max_step_m'])
            self.pending = record
            return action, deepcopy(record)
        if decision['recovery_goal'] != 'acquire':
            return list(proposal), None
        if self.option is None:
            available = self._available(state)
            if not available:
                raise ValueError('Exhausted recovery should have blocked entry or exited')
            name = 'lift_probe' if state.get('grasp') is True and 'lift_probe' in available else available[-1]
            self._start(name, p, e, decision['step'])
        if self.phase == 'release':
            target, grip = self.origin_eef, -1.
        elif self.phase == 'retreat':
            target, grip = self.origin_eef+np.array([0., 0., CONFIG['retreat_m']]), -1.
        elif self.phase == 'approach':
            target, grip = p+self.offset+np.array([0., 0., CONFIG['retreat_m']]), -1.
        elif self.phase == 'descend':
            target, grip = p+self.offset, -1.
        elif self.phase == 'close':
            target, grip = e, 1.
        else:
            target, grip = self.goal, 1.
        delta = target-e
        norm = float(np.linalg.norm(delta))
        delta *= min(1., CONFIG['max_step_m']/max(norm, 1e-12))
        xyz = np.clip(delta/self.scale, -1., 1.)
        action = xyz.tolist()+[0., 0., 0., grip]
        record = dict(schema=SCHEMA, option=self.option, phase=self.phase,
            step=decision['step'], target_eef=target.tolist(), eef=e.tolist(),
            object_position=p.tolist(), scale=self.scale.tolist(), max_step_m=CONFIG['max_step_m'])
        self.pending = record
        return action, deepcopy(record)

    def observe(self, diagnostics, state):
        if self.pending is None:
            return None
        record, self.pending = self.pending, None
        p, e = self._state(diagnostics)
        if record['option'] == 'hold_recovered':
            return dict(schema=SCHEMA, exhausted=False, option='hold_recovered', outcome='holding_for_verification')
        self.used += 1
        self.phase_used += 1
        self.samples.append(dict(z=float(p[2]), eef=e.tolist(), object_position=p.tolist()))
        phase = self.phase
        distance = float(np.linalg.norm(e-np.asarray(record['target_eef'])))
        reached = distance < CONFIG['reach_tolerance_m']
        fail = None
        if phase == 'release' and self.phase_used >= 2:
            self.phase = 'retreat'
        elif phase == 'retreat' and (reached or self.phase_used >= 6):
            self.phase = 'approach'
        elif phase == 'approach' and (reached or self.phase_used >= 3):
            self.phase = 'descend'
        elif phase == 'descend' and (reached or self.phase_used >= 6):
            self.phase = 'close'
        elif phase == 'close' and self.phase_used >= 3:
            self.phase = 'lift'
            self.goal = e+np.array([0., 0., CONFIG['lift_m']])
            self.samples = []
        elif phase == 'lift':
            n = CONFIG['window']
            if len(self.samples) >= n:
                window = self.samples[-n:]
                rise = window[-1]['z']-window[0]['z']
                eef_delta = np.asarray(window[-1]['eef'])-window[0]['eef']
                object_delta = np.asarray(window[-1]['object_position'])-window[0]['object_position']
                self.last_coupling = float(np.linalg.norm(object_delta-eef_delta))
                self.grasp_diagnosis = ('object_follows_eef' if
                    np.linalg.norm(eef_delta) >= CONFIG['min_progress_m'] and
                    self.last_coupling <= CONFIG['min_progress_m'] else
                    'contact_without_object_motion' if
                    np.linalg.norm(eef_delta) >= CONFIG['min_progress_m'] and
                    np.linalg.norm(object_delta) < CONFIG['min_progress_m'] else 'unresolved')
                if rise < CONFIG['min_progress_m'] and state.get('held_clear') is not True:
                    fail = 'lift_no_object_progress'
        cap = CONFIG['lift_cap'] if self.option == 'lift_probe' else CONFIG['regrasp_cap']
        if self.used >= cap and state.get('held_clear') is not True:
            fail = fail or 'option_budget_no_recovery'
        if phase in ('retreat', 'approach', 'descend') and len(self.samples) >= CONFIG['window']:
            movement = math.dist(self.samples[-1]['eef'], self.samples[-CONFIG['window']]['eef'])
            if movement < CONFIG['min_progress_m'] and not reached:
                fail = 'eef_no_progress'
        if self.phase != phase:
            self.phase_used = 0
            self.samples = []
        feedback = dict(schema=SCHEMA, option=record['option'], phase=phase,
            used=self.used, phase_next=self.phase, reached=reached, target_error_m=distance,
            outcome='running', exhausted=False, object_rise_m=float(p[2])-self.initial_z,
            coupling_error_m=getattr(self, 'last_coupling', None),
            grasp_diagnosis=getattr(self, 'grasp_diagnosis', 'unresolved'))
        if fail:
            self.failed.add(self.option)
            self.events.append(dict(type='option_failed', option=self.option, reason=fail, step=record['step']))
            self.option = None
            feedback.update(outcome=fail, exhausted=not self._available(state))
        return feedback


def validate_skill(row):
    """Independent structural/action reconstruction; full state replay also reruns skills."""
    r = row['recovery_action']
    if (r.get('schema') != SCHEMA or row['decision']['owner'] != 'repair' or
            row['action_protocol'] != SCHEMA or r['step'] != row['step'] or
            r.get('phase') not in ('release', 'retreat', 'approach', 'descend', 'close', 'lift', 'hold') or
            r.get('option') not in ('lift_probe', 'regrasp', 'hold_recovered')):
        raise ValueError('Unauthorized skill action')
    delta = np.asarray(r['target_eef'])-r['eef']
    delta *= min(1., CONFIG['max_step_m']/max(float(np.linalg.norm(delta)), 1e-12))
    grip = -1. if r['phase'] in ('release', 'retreat', 'approach', 'descend') else 1.
    expected = np.clip(delta/np.asarray(r['scale']), -1., 1.).tolist()+[0., 0., 0., grip]
    np.testing.assert_array_equal(row['action'], expected)
