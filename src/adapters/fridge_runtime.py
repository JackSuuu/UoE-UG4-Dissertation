"""Development five-layer fridge wrapper; privileged feedback, no learned predictor.

The frozen VLA remains the nominal controller. Repairs are contact-driven, bounded
phase-local primitives. No object/robot state is installed after reset.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from adapters.libero_completion import box_corners

SIDE = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
PROMPTS = {
    'open': 'open the fridge',
    'grasp': 'pick up the cream cheese box inside the fridge',
    'extract': 'take the cream cheese box out of the fridge',
    'place': 'place the cream cheese box on the table in front of the fridge',
    'close': 'close the fridge',
    'withdraw': 'move the empty gripper away from the fridge',
}


def sense(env, obs):
    m, d = env.sim.model, env.sim.data
    jid = m.joint_name2id('short_fridge_1_fridge_door_joint_0')
    g = m.geom_name2id('cream_cheese_1_g1')
    world = box_corners(d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g])
    contacts = [(m.geom_id2name(c.geom1) or '', m.geom_id2name(c.geom2) or '', float(c.dist))
                for c in d.contact[:d.ncon]]
    item_contacts = [c for c in contacts if 'cream_cheese' in str(c)]
    table = any('table_collision' in str(c) for c in item_contacts)
    hand = any('gripper' in str(c) for c in item_contacts)
    shelf = any('short_fridge' in str(c) for c in item_contacts)
    item = env.env.objects_dict['cream_cheese_1']
    grasp = bool(env.env._check_grasp(env.env.robots[0].gripper, item.contact_geoms))
    fridge = d.body_xpos[m.body_name2id('short_fridge_1_main')].copy()
    front = float(fridge[0]-.095)
    table_id = m.geom_name2id('table_collision')
    floor = float(d.geom_xpos[table_id][2]+m.geom_size[table_id][2])
    speed = float(np.linalg.norm(d.get_body_xvelp('cream_cheese_1_main')))
    angular = float(np.linalg.norm(d.get_body_xvelr('cream_cheese_1_main')))
    footprint = bool(np.all(world[:, :2] >= [-.15, -.25]) and np.all(world[:, :2] <= [.10, -.05]))
    placed = bool(footprint and table and not hand and speed < .02 and angular < .1
                  and abs(world[:, 2].min()-floor) < .005)
    handle = d.geom_xpos[m.geom_name2id('short_fridge_1_g17')].copy()
    door_contact = any('gripper' in str(c) and 'short_fridge_1_g1' in str(c)
                       and any(f'short_fridge_1_g{k}' in str(c) for k in range(15, 20))
                       for c in contacts)
    return dict(item=obs['cream_cheese_1_pos'].copy(), eef=obs['robot0_eef_pos'].copy(),
        quat=obs['robot0_eef_quat'].copy(), angle=float(d.qpos[m.jnt_qposadr[jid]]),
        door_speed=float(d.qvel[m.jnt_dofadr[jid]]), handle=handle,
        hinge=d.xanchor[jid].copy(), fridge=fridge, front=front, floor=floor,
        grasp=grasp, shelf=shelf, table=table, hand_contact=hand, placed=placed,
        extracted=bool(world[:, 0].max() < front-.015 and not shelf),
        lifted=bool(grasp and not shelf), door_contact=door_contact,
        clear=bool(np.linalg.norm(obs['robot0_eef_pos']-handle) > .16 and not door_contact and not hand),
        item_speed=speed, item_angular_speed=angular,
        max_penetration=max([max(0., -c[2]) for c in item_contacts]+[0.]))


class TaskMonitor:
    """Same task tracker and prompts in both arms; completion requires sequence+dwell."""
    def __init__(self, initial_door):
        self.initial_door = initial_door
        self.opened = initial_door == 'open'
        self.grasp_seen = self.extracted_seen = False
        self.placement_streak = self.completion_streak = 0
        self.phase = 'open' if initial_door == 'closed' else 'grasp'

    def observe(self, s, interval_released=True):
        self.opened |= s['angle'] >= 1.8
        self.grasp_seen |= s['grasp']
        self.extracted_seen |= bool(s['extracted'] and s['grasp'])
        self.placement_streak = self.placement_streak+1 if s['placed'] and interval_released else 0
        closed = abs(s['angle']) <= np.deg2rad(5) and abs(s['door_speed']) < .05
        if self.placement_streak >= 10:
            self.phase = 'withdraw' if closed else 'close'
        elif s['placed'] and self.extracted_seen:
            self.phase = 'place'
        elif s['grasp']:
            self.phase = 'place' if s['extracted'] else 'extract'
        elif self.opened and s['angle'] >= 1.5:
            self.phase = 'grasp'
        else:
            self.phase = 'open'
        complete = bool(self.opened and self.grasp_seen and self.extracted_seen
                        and s['placed'] and interval_released and closed and s['clear'])
        self.completion_streak = self.completion_streak+1 if complete else 0
        return dict(phase=self.phase, opened=self.opened, grasp_seen=self.grasp_seen,
            extracted_seen=self.extracted_seen, placement_streak=self.placement_streak,
            completion_streak=self.completion_streak,
            remaining_sequence_complete=self.completion_streak >= 20,
            full_completion=self.completion_streak >= 20 and self.initial_door == 'closed')


def pose_action(s, target, grip=-1., orientation=None):
    action = np.zeros(7)
    action[:3] = np.clip((np.asarray(target)-s['eef'])*12, -.5, .5)
    action[-1] = grip
    if orientation is not None:
        error = (Rotation.from_matrix(orientation)*Rotation.from_quat(s['quat']).inv()).as_rotvec()
        action[3:6] = np.clip(error*2, -.5, .5)
    return action


class FridgeRepair:
    def __init__(self, max_total=420, max_local=200):
        self.max_total, self.max_local = max_total, max_local
        self.used = self.local = self.stable = 0
        self.active = None
        self.stage = 0
        self.history = deque(maxlen=30)
        self.last_phase = None
        self.cooldown = 0
        self.entry = None

    def metric(self, s, phase):
        if phase == 'open':
            return s['angle']
        if phase == 'grasp':
            return -float(np.linalg.norm(s['eef']-(s['item']+[-.025, 0, .008])))
        if phase == 'extract':
            return -float(s['item'][0])
        if phase == 'place':
            return -float(np.linalg.norm(s['item']-[-.025, -.15, s['floor']+.02]))
        if phase == 'close':
            return -abs(s['angle'])
        return float(np.linalg.norm(s['eef']-s['handle']))

    def verified(self, s, phase):
        return {'open': s['angle'] >= 1.8,
                'grasp': s['grasp'], 'extract': s['grasp'] and s['extracted'],
                'place': s['placed'],
                'close': abs(s['angle']) <= np.deg2rad(5) and abs(s['door_speed']) < .05,
                'withdraw': s['clear']}[phase]

    def primitive(self, s):
        phase = self.active
        e, obj = s['eef'], s['item']
        orientation, grip = SIDE, -1.
        if phase == 'grasp':
            # Event-conditioned approach; target derives from current object state.
            targets = [[-.25, e[1], 1.23], [-.25, obj[1], 1.23],
                       [-.25, obj[1], obj[2]+.036], [-.25, obj[1], obj[2]+.008],
                       (obj+[-.025, 0, .008]).tolist()]
            target = targets[min(self.stage, 4)]
            orientation = None if self.stage < 3 else SIDE
            if self.stage >= 5:
                grip = 1.
            close_pose = np.linalg.norm(np.asarray(target)-e) < .012
            aligned = orientation is None or np.linalg.norm((Rotation.from_matrix(SIDE)*Rotation.from_quat(s['quat']).inv()).as_rotvec()) < .12
            if close_pose and aligned and self.stage < 5:
                self.stage += 1
        elif phase == 'extract':
            grip = 1.
            target = self.entry['eef'].copy()
            target[2] += .025
            if self.stage == 0 and np.linalg.norm(target-e) < .012 and not s['shelf']:
                self.stage = 1
            if self.stage:
                target[0] = -.25
        elif phase == 'place':
            grip = 1.
            targets = [[-.25, e[1], 1.18], [-.025, -.15, 1.18],
                       [-.025, -.15, s['floor']+.025]]
            target = targets[min(self.stage, 2)]
            if self.stage < 2 and np.linalg.norm(np.asarray(target)-e) < .015:
                self.stage += 1
            if self.stage == 2 and (s['table'] or abs(e[2]-target[2]) < .012):
                self.stage = 3
            if self.stage == 3:
                grip = -1.
        elif phase in ('open', 'close'):
            h = s['handle']
            if self.stage == 0:
                target = [-.25, e[1], 1.28]
                orientation = None
            elif self.stage == 1:
                target = [h[0]-.12, h[1], 1.28]
                orientation = None
            elif self.stage == 2:
                target = [h[0]-.12, h[1], h[2]]
            elif self.stage == 3:
                target = h+[-.025, 0, 0]
            else:
                grip = 1.
                delta = .10 if phase == 'open' else -.10
                rot = Rotation.from_rotvec([0, 0, delta]).as_matrix()
                target = s['hinge']+rot@(h-s['hinge'])+[-.025, 0, 0]
            if self.stage < 4 and np.linalg.norm(np.asarray(target)-e) < .015:
                self.stage += 1
        else:
            target, orientation = [-.25, -.10, 1.23], None
        return pose_action(s, target, grip, orientation)

    def decide(self, s, phase, proposal):
        event = None
        if phase != self.last_phase:
            self.history.clear()
            self.last_phase = phase
        self.history.append(self.metric(s, phase))
        stalled = len(self.history) == self.history.maxlen and max(self.history)-min(self.history) < .005
        unsafe_release = s['grasp'] and phase in ('extract', 'place') and not s['table'] and proposal[-1] < 0
        checks = dict(stalled=bool(stalled), premature_release=bool(unsafe_release),
                      repair_budget_available=self.used < self.max_total,
                      state_source='privileged_simulator_feedback')
        if self.active:
            self.stable = self.stable+1 if self.verified(s, self.active) else 0
            if self.stable >= 3:
                event = 'verified_handoff'
                self.active = None
                self.cooldown = 20
                self.history.clear()
            elif self.local >= self.max_local or self.used >= self.max_total:
                event = 'repair_budget_exit_unverified'
                self.active = None
                self.cooldown = 30
                self.history.clear()
        if self.cooldown:
            self.cooldown -= 1
        elif not self.active and stalled and self.used < self.max_total:
            self.active, self.local, self.stable, self.stage = phase, 0, 0, 0
            self.entry = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in s.items()}
            event = 'stall_trigger'
        if self.active:
            action = self.primitive(s)
            self.used += 1
            self.local += 1
            owner = 'repair'
        elif unsafe_release and self.used < self.max_total:
            action = proposal.copy()
            action[-1] = 1.
            self.used += 1
            owner, event = 'repair', 'retain_grip'
        else:
            action, owner = proposal.copy(), 'vla'
        return action, dict(owner=owner, event=event, checks=checks, repair_phase=self.active,
            repair_stage=self.stage if self.active else None, correction_count=self.used,
            local_corrections=self.local if self.active else 0)
