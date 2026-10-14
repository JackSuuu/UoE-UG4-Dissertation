"""Privileged five-layer H1 release-to-clearance development diagnostic.

Simulator geometry/contact are explicit measured-state inputs, not learned vision.
No future simulation, hidden dynamics, changed language prompt or actor training.
"""
from collections import deque

import numpy as np

from adapters.libero_completion import box_corners
from adapters.libero_preservation import SETTLE, bind_renderer_context, validate_controller
from adapters.libero_runtime import make_env

TASK = 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it'
ARMS = ('vla', 'gate_hold', 'timed_repair', 'phase_repair')
MAX_STEPS = 600
MAX_REPAIR = 40
ENTRY_DWELL = 3
EXIT_DWELL = 3
FINAL_DWELL = 20
CLEARANCE = .01
TRANSLATION = .005


def settled_drawer(root, index):
    env, obs = make_env(root, index, TASK)
    bind_renderer_context()
    validate_controller(env)
    for _ in range(10):
        obs, _, _, _ = env.step(SETTLE)
    return env, obs


def collision_vertices(model, geom):
    """Vertices in compiled geom coordinates (mesh/box only; fail on new types)."""
    kind = int(model.geom_type[geom])
    if kind == 6:
        return box_corners(np.zeros(3), np.eye(3), model.geom_size[geom])
    assert kind == 7, (model.geom_id2name(geom), kind)
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return model.mesh_vert[start:start + count].copy()


def contact_flags(samples, drawer_geoms):
    flags = dict(bowl_gripper=False, cabinet_gripper=False, bowl_supported=False,
                 max_bowl_cabinet_penetration=0.)
    for si, sample in enumerate(samples):
        for c in sample['contacts']:
            pair = (c['geom1'], c['geom2'])
            bowl = any(n.startswith('akita_black_bowl_1_') for n in pair)
            grip = any(n.startswith('gripper0_') for n in pair)
            cabinet = any(n.startswith('white_cabinet_1_') for n in pair)
            flags['bowl_gripper'] |= bowl and grip
            flags['cabinet_gripper'] |= cabinet and grip
            if bowl and cabinet:
                flags['max_bowl_cabinet_penetration'] = max(
                    flags['max_bowl_cabinet_penetration'], -c['distance'])
            if si == len(samples) - 1:
                # Floor contact, not arbitrary side contact, is the support proxy.
                flags['bowl_supported'] |= bowl and drawer_geoms['floor'] in pair
    return flags


class DrawerMonitor:
    """Read-only geometry, raw contacts and task milestone observations."""
    def __init__(self, env):
        self.env = env
        m = env.sim.model
        self.drawer = m.body_name2id('white_cabinet_1_cabinet_bottom')
        self.bowl = m.body_name2id('akita_black_bowl_1_main')
        self.joint = m.get_joint_qpos_addr('white_cabinet_1_bottom_level')
        self.dof = m.get_joint_qvel_addr('white_cabinet_1_bottom_level')
        self.bowl_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == self.bowl
                           and (m.geom_contype[i] or m.geom_conaffinity[i])]
        self.gripper_geoms = [i for i in range(m.ngeom)
                              if (m.geom_id2name(i) or '').startswith('gripper0_')
                              and (m.geom_contype[i] or m.geom_conaffinity[i])]
        assert len(self.bowl_geoms) == 40 and len(self.gripper_geoms) == 5
        self.vertices = {g: collision_vertices(m, g) for g in self.bowl_geoms + self.gripper_geoms}
        # Inner faces from the compiled collision boxes, in drawer body frame.
        import mujoco
        def extent(suffix):
            g = m.geom_name2id('white_cabinet_1_' + suffix)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, m.geom_quat[g])
            ext = np.abs(rotation.reshape(3, 3)) @ m.geom_size[g]
            return m.geom_pos[g] - ext, m.geom_pos[g] + ext
        left, right = extent('g37'), extent('g36')
        front, back, floor, ceiling = extent('g35'), extent('g34'), extent('g38'), extent('g8')
        self.lower = np.array([left[1][0], front[1][1], floor[1][2]])
        self.upper = np.array([right[0][0], back[0][1], ceiling[0][2]])
        self.front_outside = float(extent('g33')[0][1])
        self.drawer_geoms = dict(floor='white_cabinet_1_g38')
        assert np.all(self.upper > self.lower)

    def local_vertices(self, geoms, rotation, origin):
        d = self.env.sim.data
        return np.concatenate([(self.vertices[g] @ d.geom_xmat[g].reshape(3, 3).T
                                + d.geom_xpos[g] - origin) @ rotation for g in geoms])

    def measure(self, samples):
        e, d = self.env, self.env.sim.data
        r, origin = d.body_xmat[self.drawer].reshape(3, 3), d.body_xpos[self.drawer]
        bowl = self.local_vertices(self.bowl_geoms, r, origin)
        grip = self.local_vertices(self.gripper_geoms, r, origin)
        lo, hi = bowl.min(0), bowl.max(0)
        flags = contact_flags(samples, self.drawer_geoms)
        native_in = bool(e.env.object_states_dict['white_cabinet_1_bottom_region'].check_contain(
            e.env.object_states_dict['akita_black_bowl_1']))
        closed = bool(e.env.object_states_dict['white_cabinet_1_bottom_region'].is_close())
        v = d.get_body_xvelp('akita_black_bowl_1_main')
        dv = d.get_body_xvelp('white_cabinet_1_cabinet_bottom')
        linear = float(np.linalg.norm(v))
        angular = float(np.linalg.norm(d.get_body_xvelr('akita_black_bowl_1_main')))
        contained = bool(np.all(lo[:2] >= self.lower[:2]) and np.all(hi[:2] <= self.upper[:2]))
        seated = bool(abs(lo[2] - self.lower[2]) <= .005)
        retained = bool(contained and seated and flags['bowl_supported'])
        # Front-of-drawer OR entirely above the bowl: both are measured clearance,
        # never an instruction to move vertically into the cabinet ceiling.
        front_margin = float(self.front_outside - grip[:, 1].max())
        above_margin = float(grip[:, 2].min() - hi[2])
        clear = bool(max(front_margin, above_margin) >= CLEARANCE
                     and not flags['bowl_gripper'] and not flags['cabinet_gripper'])
        checks = dict(native=bool(e.check_success()), full_footprint=contained,
            seated=seated, supported=flags['bowl_supported'],
            vertical_fit=bool(lo[2] >= self.lower[2] - .001 and hi[2] <= self.upper[2] + .001),
            released=not flags['bowl_gripper'] and not flags['cabinet_gripper'],
            shallow=flags['max_bowl_cabinet_penetration'] <= .001,
            slow=linear <= .02 and angular <= .1 and abs(float(d.qvel[self.dof])) <= .01)
        return dict(**flags, native_in=native_in, closed=closed, native_success=checks['native'],
            bowl_min=lo.tolist(), bowl_max=hi.tolist(), drawer_lower=self.lower.tolist(),
            drawer_upper=self.upper.tolist(), drawer_q=float(d.qpos[self.joint]),
            drawer_speed=float(d.qvel[self.dof]), linear_speed=linear, angular_speed=angular,
            relative_speed=float(np.linalg.norm(v-dv)), retained=retained,
            gripper_front_margin=front_margin, gripper_above_margin=above_margin,
            clear=clear, outward_world=(-r[:, 1]).tolist(),
            conditions=checks, candidate=all(checks.values()),
            alignment_error=float(np.linalg.norm((lo[:2]+hi[:2]-self.lower[:2]-self.upper[:2])/2)))


class SubgoalTracker:
    """Shared causal progress monitor. Milestones do not terminate the episode."""
    def __init__(self):
        self.milestones = {}
        self.history = deque(maxlen=20)
        self.release_streak = self.clear_streak = self.entry_streak = self.final_streak = 0

    def update(self, measured, step):
        m = measured
        self.entry_streak = self.entry_streak + 1 if (
            m['retained'] and m['relative_speed'] <= .02 and m['angular_speed'] <= .1) else 0
        self.release_streak = self.release_streak + 1 if m['retained'] and not m['bowl_gripper'] else 0
        self.clear_streak = self.clear_streak + 1 if m['retained'] and m['clear'] else 0
        self.final_streak = self.final_streak + 1 if m['candidate'] else 0
        events = {'grasp_contact': m['bowl_gripper'], 'inside': m['native_in'],
                  'supported_placement': self.entry_streak >= ENTRY_DWELL,
                  'released': self.release_streak >= EXIT_DWELL,
                  'cleared': self.clear_streak >= EXIT_DWELL,
                  'closed_with_bowl': m['closed'] and m['retained'],
                  'complete': self.final_streak >= FINAL_DWELL}
        for name, occurred in events.items():
            if occurred:
                self.milestones.setdefault(name, step)
        self.history.append((m['alignment_error'], m['drawer_q']))
        phase = 'acquire'
        for milestone, next_phase in [('grasp_contact','transport_align'), ('inside','insert_release'),
                                     ('released','clear'), ('cleared','close_drawer'),
                                     ('closed_with_bowl','verify'), ('complete','complete')]:
            if milestone in self.milestones:
                phase = next_phase
        # Logged diagnosis only: no unrelated alignment/stagnation repair this study.
        stagnation = len(self.history) == 20 and abs(self.history[-1][0]-self.history[0][0]) < .002
        return dict(phase=phase, entry=self.entry_streak >= ENTRY_DWELL,
                    released=self.release_streak >= EXIT_DWELL, clear=self.clear_streak >= EXIT_DWELL,
                    entry_streak=self.entry_streak, release_streak=self.release_streak,
                    clear_streak=self.clear_streak,
                    final_streak=self.final_streak, milestones=dict(self.milestones),
                    stagnation=bool(stagnation), complete=self.final_streak >= FINAL_DWELL)


class TransitionGate:
    """One bounded release/clearance transaction; all modes hand back for closing.

    gate_hold is a hold/open veto control, not H1 same-frame 'replanning'.
    Timed control has the same primitives but a blind 10+30 interval schedule.
    """
    def __init__(self, arm):
        assert arm in ARMS
        self.arm = arm
        self.trigger = None
        self.used = 0
        self.done = False
        self.handoff = None

    def decide(self, proposal, measured, tracking, step):
        action = np.asarray(proposal).copy()
        reason = 'baseline_passthrough' if self.arm == 'vla' else (
            'actor_after_handoff' if self.done else 'waiting_supported_entry')
        route, mode = 'continue', 'nominal'
        if self.trigger is None and tracking['entry']:
            self.trigger = step
        if self.trigger is not None and not self.done and self.arm != 'vla':
            # Releasing a dropped/misaligned bowl further is not a recovery.
            if not measured['retained']:
                reason = 'placement_lost'
                self.done = True
            elif self.used >= MAX_REPAIR:
                reason = 'budget_exhausted'
                self.done = True
            elif self.arm != 'timed_repair' and tracking['clear']:
                reason = 'stable_clearance'
                self.done = True
            if self.done:
                self.handoff = dict(step=step, reason=reason, stable_clear=tracking['clear'])
            else:
                mode, route = 'contact_transition', 'hold'
                action = np.zeros(7)
                action[-1] = -1
                reason = 'release_hold'
                move = (self.arm == 'timed_repair' and self.used >= 10) or (
                    self.arm == 'phase_repair' and tracking['released'] and not measured['clear'])
                if move:
                    action[:3] = np.asarray(measured['outward_world']) * (TRANSLATION / .05)
                    reason, route = 'retract_outward', 'repair'
                self.used += 1
        return action, dict(mode=mode, route=route, reason=reason, trigger=self.trigger,
                            used=self.used, handoff=self.handoff)


def decision_trace(step, arm, measured, tracking, proposal, action, decision):
    """Observable execution proof, derived from the actual causal gate inputs.

    This is not a language-model rationale or an unexecuted candidate forecast.
    Keep pre-decision evidence distinct from subsequently appended outcomes.
    """
    delta = np.asarray(action) - np.asarray(proposal)
    return dict(schema=1, step=step, arm=arm, evidence_through_step=step-1,
        provenance='privileged current/past simulator geometry and substep contact',
        subgoal=tracking['phase'], milestones_before=tracking['milestones'],
        evidence={k: measured[k] for k in [
            'native_in', 'retained', 'bowl_supported', 'bowl_gripper', 'cabinet_gripper',
            'bowl_min', 'bowl_max', 'drawer_lower', 'drawer_upper', 'relative_speed',
            'angular_speed', 'gripper_front_margin', 'gripper_above_margin', 'clear',
            'drawer_q', 'alignment_error']},
        predicates=dict(supported_entry=dict(value=tracking['entry'],
                            consecutive=tracking['entry_streak'], required=ENTRY_DWELL,
                            relative_speed_limit_m_s=.02, angular_speed_limit_rad_s=.1,
                            seating_tolerance_m=.005),
                        detached=dict(value=tracking['released'],
                            consecutive=tracking['release_streak'], required=EXIT_DWELL),
                        stable_clearance=dict(value=tracking['clear'],
                            consecutive=tracking['clear_streak'], required=EXIT_DWELL,
                            geometric_margin_m=CLEARANCE),
                        stagnation=dict(value=tracking['stagnation'], diagnostic_only=True)),
        gate=dict(**decision, budget_cap=MAX_REPAIR, remaining_after=MAX_REPAIR-decision['used']),
        nominal_action=np.asarray(proposal).tolist(), executed_action=np.asarray(action).tolist(),
        correction=dict(changed=bool(np.any(delta != 0)), pose_delta=delta[:6].tolist(),
                        gripper_delta=float(delta[-1])),
        interpretation='deterministic rule trace; no learned reasoning or future-outcome access')
