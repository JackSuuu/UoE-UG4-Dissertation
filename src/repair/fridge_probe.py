"""One bounded frozen-policy repair transaction for the fridge transfer pilot.

Measured predicates/prompts follow archived R2; no geometric action generator.
This deliberately has no microwave-trained visual/risk/fusion thresholds.
"""
from collections import deque
from copy import deepcopy
import math


CONFIG = dict(window=30, stagnation_range=.005, local_cap=32, total_cap=32,
              verification_streak=3, continuation_window=8, max_repairs=1)


def achieved(s, phase):
    return bool({'open': s['angle'] >= 1.8, 'grasp': s['grasp'],
        'extract': s['grasp'] and s['extracted'], 'place': s['placed'],
        'close': abs(s['angle']) <= math.radians(5) and abs(s['door_speed']) < .05,
        'withdraw': s['clear']}[phase])


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def metric(s, phase):
    return {'open': s['angle'], 'grasp': -distance(s['eef'], s['item']),
        'extract': -s['item'][0],
        'place': -distance(s['item'], [-.025, -.15, s['floor']+.02]),
        'close': -abs(s['angle']), 'withdraw': distance(s['eef'], s['handle'])}[phase]


class Probe:
    def __init__(self, language, prompts, enabled):
        self.language, self.prompts, self.enabled = language, prompts, enabled
        self.history = deque(maxlen=CONFIG['window'])
        self.last_phase = self.previous = self.repair = None
        self.active = False
        self.streak = self.continuation = 0

    def before(self, step, state, task):
        phase = task['phase']
        if self.last_phase != phase:
            self.history.clear()
            self.last_phase = phase
        self.history.append(metric(state, phase))
        stalled = (len(self.history) == CONFIG['window'] and
            max(self.history)-min(self.history) < CONFIG['stagnation_range'] and not achieved(state, phase))
        lost = bool(self.previous and self.previous['grasp'] and not state['grasp'] and not state['placed'])
        self.previous = deepcopy(state)
        trigger = 'grasp_lost' if lost else 'phase_stagnation' if stalled else None
        if self.enabled and self.repair is None and trigger:
            self.active = True
            # Preserve placement when closing/withdrawing; door-open during grasp
            # or extraction. Achieved goals can be intentionally relinquished by
            # later stages; the transaction freezes only its own contract.
            preserve = (['placed'] if state['placed'] and phase in ('close', 'withdraw') else [])
            if phase in ('grasp', 'extract') and state['angle'] >= 1.5:
                preserve.append('door_open')
            self.repair = dict(entry_step=step, phase=phase, trigger=trigger, used=0,
                preserve=preserve, verified=False, local_recovery_step=None,
                exit_step=None, exit_reason=None, handoff_step=None,
                stable_nominal_continuation=None, post_handoff_loss_step=None)
        goal = self.repair['phase'] if self.active else phase
        return dict(owner='repair' if self.active else 'nominal', language=self.prompts[goal] if self.active else self.language,
            recovery_goal=goal, repair_id=1 if self.active else None, trigger=trigger,
            source='greedy', seed=0)

    def preserved(self, state):
        return all(state['angle'] >= 1.5 if key == 'door_open' else state[key]
                   for key in self.repair['preserve'])

    def after(self, step, state, decision):
        if self.repair:
            r = self.repair
            if decision['owner'] == 'repair':
                r['used'] += 1
                self.streak = self.streak+1 if achieved(state, r['phase']) and self.preserved(state) else 0
                if self.streak >= CONFIG['verification_streak']:
                    r.update(local_recovery_step=step, verified=True, exit_step=step, exit_reason='measured_verified')
                    self.active = False
                elif r['used'] >= CONFIG['local_cap']:
                    r.update(exit_step=step, exit_reason='local_budget_unverified')
                    self.active = False
            elif r['exit_step'] is not None:
                if r['handoff_step'] is None:
                    r['handoff_step'] = step
                if r['verified'] and self.continuation < CONFIG['continuation_window']:
                    self.continuation += 1
                    if not achieved(state, r['phase']) or not self.preserved(state):
                        r['stable_nominal_continuation'] = False
                        if r['post_handoff_loss_step'] is None:
                            r['post_handoff_loss_step'] = step
                    elif self.continuation == CONFIG['continuation_window'] and r['stable_nominal_continuation'] is None:
                        r['stable_nominal_continuation'] = True
        return deepcopy(self.repair)
