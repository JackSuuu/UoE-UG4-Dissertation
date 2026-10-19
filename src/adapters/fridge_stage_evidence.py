"""Read-only stage evidence for the full fridge sequence; never generates actions.

The input adapter is currently privileged R2 simulator measurement. Boolean geometry
predicates inherit that adapter's scene-specific definitions; this module makes no
visual-grounding, generic-geometry or formal-correctness claim.
"""
import math
from numbers import Real


PHASES = ('open', 'grasp', 'extract', 'place', 'close', 'withdraw')


def predicate(value, margins, evidence_age=0, missing=()):
    unknown = bool(missing) or evidence_age != 0
    return dict(status='unknown' if unknown else 'satisfied' if value else 'violated',
                margins=margins, evidence_age_intervals=evidence_age, missing=list(missing))


def predicates(s, interval_released, evidence_age=0):
    """Three-valued measured predicates, not predicted candidate-action outcomes."""
    def check(keys, fn):
        missing = [k for k in keys if k not in s or s[k] is None
                   or isinstance(s[k], Real) and not math.isfinite(s[k])]
        if missing:
            return predicate(False, {}, evidence_age, missing)
        value, margins = fn()
        return predicate(value, margins, evidence_age)

    result = {
        'open': check(['angle'], lambda: (s['angle'] >= 1.8, {'door_open_rad': s['angle']-1.8})),
        'grasp': check(['grasp'], lambda: (bool(s['grasp']), {})),
        'extract': check(['grasp', 'extracted', 'shelf'],
            lambda: (bool(s['grasp'] and s['extracted'] and not s['shelf']), {})),
        'place': check(['placed', 'table', 'hand_contact', 'item_speed', 'item_angular_speed'],
            lambda: (bool(s['placed'] and s['table'] and not s['hand_contact']
                          and s['item_speed'] < .02 and s['item_angular_speed'] < .1
                          and interval_released is True),
                     {'linear_speed_m_s': .02-s['item_speed'],
                      'angular_speed_rad_s': .1-s['item_angular_speed']})),
        'close': check(['angle', 'door_speed'],
            lambda: (abs(s['angle']) <= math.radians(5) and abs(s['door_speed']) < .05,
                     {'closed_angle_rad': math.radians(5)-abs(s['angle']),
                      'door_speed_rad_s': .05-abs(s['door_speed'])})),
        'withdraw': check(['clear', 'door_contact', 'hand_contact'],
            lambda: (bool(s['clear'] and not s['door_contact'] and not s['hand_contact']), {})),
    }
    if interval_released is None:
        result['place'] = predicate(False, result['place']['margins'], evidence_age,
            [*result['place']['missing'], 'whole_interval_release_contact_evidence'])
    return result


class FridgeStageEvidence:
    """Ordered historical milestones plus current predicates and continuous dwell.

    This observer cannot choose a robot action or authorize a repair/handoff. External
    controllers must log those decisions separately, alongside the actual executed
    source. Missing/stale evidence resets consecutive verification streaks.
    """
    def __init__(self, initial_door):
        if initial_door not in ('open', 'closed'):
            raise ValueError(initial_door)
        self.initial_door = initial_door
        self.streaks = dict.fromkeys(PHASES, 0)
        self.milestones = dict.fromkeys(PHASES, False)
        # Open initialization is an experimental condition, not an achieved action.
        self.milestones['open'] = initial_door == 'open'
        self.full_streak = 0
        self.last_step = -1

    def observe(self, step, s, interval_released, evidence_age=0):
        if step != self.last_step+1:
            raise ValueError('Evidence must represent consecutive executed intervals')
        self.last_step = step
        checks = predicates(s, interval_released, evidence_age)
        events = []
        for phase in PHASES:
            self.streaks[phase] = self.streaks[phase]+1 if checks[phase]['status']=='satisfied' else 0
        for i, phase in enumerate(PHASES):
            required = 10 if phase == 'place' else 3
            earlier = all(self.milestones[p] for p in PHASES[:i])
            if not self.milestones[phase] and earlier and self.streaks[phase] >= required:
                self.milestones[phase] = True
                events.append(dict(type='measured_stage_milestone', phase=phase, step=step,
                    streak=self.streaks[phase], required=required, control_handoff_claim=False))
        complete_now = (all(self.milestones.values()) and
            all(checks[p]['status']=='satisfied' for p in ('place', 'close', 'withdraw')))
        self.full_streak = self.full_streak+1 if complete_now else 0
        if checks['place']['status']=='satisfied' and self.milestones['extract']:
            phase = 'place' if self.streaks['place']<10 else (
                'close' if checks['close']['status']!='satisfied' else 'withdraw')
        elif checks['grasp']['status']=='satisfied' and self.milestones['open']:
            phase = 'place' if checks['extract']['status']=='satisfied' else 'extract'
        elif checks['open']['status']=='satisfied':
            phase = 'grasp'
        else:
            phase = 'open'
        unknown = [p for p in PHASES if checks[p]['status']=='unknown']
        return dict(schema='fridge_stage_evidence_v1', step=step, phase=phase,
            source='privileged_measured_state_and_interval_contacts',
            predicates=checks, unknown=unknown, milestone_history=dict(self.milestones),
            consecutive_intervals=dict(self.streaks), complete_dwell=self.full_streak,
            remaining_sequence_complete=self.full_streak>=20,
            full_closed_start_completion=self.full_streak>=20 and self.initial_door=='closed',
            events=events, action_generation=False, repair_selection=False,
            interpretation='measured stage evidence only; causal contribution requires matched controls')
