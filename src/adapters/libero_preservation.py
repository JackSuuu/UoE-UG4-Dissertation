"""Privileged, bounded placement-preservation diagnostic, not a learned verifier.

Native actor cadence is H1/k1. Forecasts are five-interval pulse-and-settle
controller primitives, NOT future VLA proposals. Only their first action executes.
"""
import hashlib
import json
import random
import time

import numpy as np

from adapters.libero_completion import CompletionMonitor, ContactRecorder
from adapters.libero_runtime import make_env

NAMES = ('original', 'half_pose', 'hold_grip', 'hold_open', 'up_open', 'up_grip')
ARMS = ('vla', 'hold', 'retreat', 'physics')
HORIZON = 5
MAX_OPPORTUNITIES = 20
TAIL_STEPS = 40
MAX_STEPS = 520
SETTLE = [0, 0, 0, 0, 0, 0, -1]


def candidates(proposal, previous_action):
    proposal = np.asarray(proposal, dtype=float)
    assert proposal.shape == (7,) and np.isfinite(proposal).all()
    grip = float(previous_action[-1])
    first = np.zeros((6, 7))
    first[0] = proposal
    first[1] = proposal
    first[1, :6] *= .5
    first[2, -1] = grip
    first[3, -1] = -1
    first[4, 2], first[4, -1] = .1, -1  # +5mm WORLD-z desired displacement
    first[5, 2], first[5, -1] = .1, grip
    plans = np.zeros((6, HORIZON, 7))
    plans[:, 0] = first
    plans[:, 1:, -1] = first[:, -1, None]
    return plans


class PreservationWindow:
    """One causal window, triggered by the first measured valid placement."""
    def __init__(self):
        self.trigger = None
        self.used = 0

    def active(self, step, previous, first_complete):
        if self.trigger is None and previous is not None and previous['candidate']:
            self.trigger = step
        if self.trigger is None or first_complete is not None:
            return False
        if step >= self.trigger + MAX_OPPORTUNITIES:
            return False
        self.used += 1
        return True


def validate_controller(env):
    c = env.robots[0].controller
    assert c.use_delta and c.use_ori and c.impedance_mode == 'fixed'
    np.testing.assert_array_equal(c.input_max, np.ones(6))
    np.testing.assert_array_equal(c.input_min, -np.ones(6))
    np.testing.assert_array_equal(c.output_max, [.05, .05, .05, .5, .5, .5])
    np.testing.assert_array_equal(c.output_min, [-.05, -.05, -.05, -.5, -.5, -.5])
    assert c.interpolator_pos is None and c.interpolator_ori is None
    assert env.env.control_freq == 20 and env.sim.model.opt.timestep == .002


def settled_env(root, index):
    env, obs = make_env(root, index)
    bind_renderer_context()
    validate_controller(env)
    for _ in range(10):
        obs, _, _, _ = env.step(SETTLE)
    return env, obs


def bind_renderer_context():
    """Robosuite1.4.1 assumes one current EGL context per render thread.

Branch creation/closure switches that thread-local context. Its render method
does not rebind it, so subsequent LIVE images can be invalid even when physics
is identical. Bind the renderer owning each draw, inside MjSim's render lock.
This is process-local and does not modify the frozen runtime or installed package.
"""
    from robosuite.utils.binding_utils import MjRenderContext
    original = MjRenderContext.render
    if getattr(original, '_libero_context_bound', False):
        return

    def render(self, *args, **kwargs):
        self.gl_ctx.make_current()
        return original(self, *args, **kwargs)

    render._libero_context_bound = True
    MjRenderContext.render = render


def fingerprint(env, obs):
    """Integration state + controller numeric fields + gripper/counters/images.

Replay reconstructs the environment rather than installing qpos/qvel. Comparing
these extra fields catches warm-start, controller-history and sensor divergence.
"""
    import mujoco
    m, d = env.sim.model._model, env.sim.data._data
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, state, spec)
    numeric = {}
    for key, value in vars(env.robots[0].controller).items():
        if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
            numeric[key] = value.tolist()
        elif isinstance(value, (bool, int, float, str)) or value is None:
            numeric[key] = value
    numeric['gripper_current_action'] = env.robots[0].gripper.current_action.tolist()
    numeric['timestep'] = env.env.timestep
    numeric['cur_time'] = env.env.cur_time
    numeric['done'] = bool(env.env.done)
    control = hashlib.sha256(json.dumps(numeric, sort_keys=True).encode()).hexdigest()
    images = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
              for k, v in obs.items() if k.endswith('_image')}
    return dict(integration=hashlib.sha256(state.tobytes()).hexdigest(),
                controller=control, images=images)


def rank_key(rows, first_action, proposal, index):
    # No force-based "safety" claims. This ranks frozen placement conditions.
    invalid = sum(not row['candidate'] for row in rows)
    recontact = sum(not row['conditions']['released'] for row in rows)
    geometry = sum(not row['conditions']['footprint_inside'] for row in rows)
    geometry += sum(not row['conditions']['bottom_seated'] for row in rows)
    deviation = float(np.linalg.norm(first_action[:6] - proposal[:6]))
    deviation += float(first_action[-1] != proposal[-1])
    return [invalid, recontact, geometry, deviation, index]


def forecast(root, index, actions, plans, reference, previous):
    """Independently replay the ACTUALLY executed prefix for every candidate."""
    saved_numpy, saved_python = np.random.get_state(), random.getstate()
    start = time.monotonic()
    forecasts = []
    try:
        for k, plan in enumerate(plans):
            env = recorder = None
            try:
                env, obs = settled_env(root, index)
                for action in actions:
                    obs, _, _, _ = env.step(np.asarray(action).tolist())
                observed = fingerprint(env, obs)
                if observed != reference:
                    raise AssertionError(f'Non-identical branch {k}: {observed} != {reference}')
                monitor = CompletionMonitor(env)
                monitor.streak = previous['streak']
                monitor.max_streak = previous['streak']
                recorder = ContactRecorder(env.sim)
                rows = []
                for offset, action in enumerate(plan):
                    recorder.reset()
                    env.step(action.tolist())
                    assert len(recorder.samples) == 25
                    rows.append(monitor.measure(len(actions) + offset, recorder.samples))
                forecasts.append(dict(name=NAMES[k], rows=rows, replay_identity=True,
                    key=rank_key(rows, plan[0], plans[0, 0], k)))
            finally:
                if recorder is not None:
                    recorder.close()
                if env is not None:
                    env.close()
    finally:
        np.random.set_state(saved_numpy)
        random.setstate(saved_python)
    chosen = min(range(len(plans)), key=lambda k: forecasts[k]['key'])
    return chosen, dict(candidates=forecasts, selected=chosen,
        replay_control_steps=len(plans) * (10 + len(actions)),
        forecast_control_steps=len(plans) * HORIZON, wall_seconds=time.monotonic()-start)
