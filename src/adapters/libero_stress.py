"""External actuator faults for paired recovery experiments, never monitor inputs.

Apply AFTER policy/gate/repair routing. A frozen schedule is shared across arms.
This module does not select cases using method outcomes or modify simulator state.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ReleaseDelay:
    anchor_step: int
    intervals: int

    def __post_init__(self):
        if self.anchor_step < 0 or self.intervals < 0:
            raise ValueError('Nonnegative anchor and duration required')

    def apply(self, selected_action, step):
        """Hold the physical close command during a one-shot release delay.

        The selected action remains logged separately from the command sent to OSC.
        Duration0 is an exact sham. No timing/active flag may enter the framework.
        """
        command = np.asarray(selected_action, dtype=float)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError('Require finite native 7D action')
        effective = command.copy()
        active = self.anchor_step <= step < self.anchor_step + self.intervals
        if active:
            effective[-1] = 1.  # Native robosuite close command, independent of arm.
        return effective, dict(kind='release_delay', **asdict(self), step=step,
            active=active, selected_command=command.tolist(), effective_command=effective.tolist(),
            changed=not np.array_equal(command, effective),
            provenance='external actuator fault; evaluator log only')


def release_opportunity(previous, nominal_action):
    """Causal injection anchor, NOT success or a framework risk/safety label.

    The point-based native region is deliberately only an opportunity envelope.
    Physical completion still uses the full registered collision/contact metric.
    No stable-placement dwell: that missed the actual release on D1root2.
    """
    return bool(previous['native_in'] and previous['bowl_gripper']
                and not previous['closed'] and np.asarray(nominal_action)[-1] < 0)
