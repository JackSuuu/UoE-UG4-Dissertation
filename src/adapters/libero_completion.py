"""Development-only released-placement diagnostics; never policy inputs."""
import itertools
import numpy as np

DWELL_STEPS = 20  # 1 second at 20Hz


def box_corners(center, rotation, half_size):
    signs = np.asarray(list(itertools.product([-1., 1.], repeat=3)))
    return np.asarray(center) + (signs * half_size) @ np.asarray(rotation).T


def placement_conditions(corners, bounds, floor, linear_speed, angular_speed,
                         gripper_contact, caddy_contact, penetration):
    """Full collision-box footprint, seated bottom, release and slow motion.

    Tolerances are preregistered development definitions, not certified precision
    or force safety limits. Open-top caddy permits the book to extend above walls.
    """
    corners = np.asarray(corners)
    return dict(
        footprint_inside=bool(np.all(corners[:, :2] >= bounds[0]) and
                              np.all(corners[:, :2] <= bounds[1])),
        bottom_seated=bool(abs(corners[:, 2].min() - floor) <= 0.005),
        released=not bool(gripper_contact),
        supported=bool(caddy_contact),
        shallow_contact=bool(penetration <= 0.001),
        slow=bool(linear_speed <= 0.02 and angular_speed <= 0.1))


class CompletionMonitor:
    def __init__(self, env):
        self.env = env
        self.streak = 0
        self.max_streak = 0
        self.first_native = None
        self.first_complete = None
        m = env.sim.model
        self.book = m.geom_name2id('black_book_1_g1')
        self.caddy = m.body_name2id('desk_caddy_1_main')
        self.book_body = m.body_name2id('black_book_1_main')
        # Compile collision-box faces into caddy-body coordinates. These four
        # axis-aligned wall faces bound the lower rear compartment.
        import mujoco
        def extent(name):
            g = m.geom_name2id(name)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, m.geom_quat[g])
            return m.geom_pos[g], np.abs(rotation.reshape(3, 3)) @ m.geom_size[g]
        front, f = extent('desk_caddy_1_g5')
        back, b = extent('desk_caddy_1_g4')
        left, l = extent('desk_caddy_1_g2')
        right, r = extent('desk_caddy_1_g7')
        bottom, h = extent('desk_caddy_1_g1')
        self.bounds = np.array([[front[0] + f[0], left[1] + l[1]],
                                [back[0] - b[0], right[1] - r[1]]])
        self.floor = float(bottom[2] + h[2])
        assert np.all(self.bounds[1] > self.bounds[0])

    def measure(self, step, substeps):
        sim = self.env.sim
        d, m = sim.data, sim.model
        rotation = d.body_xmat[self.caddy].reshape(3, 3)
        world = box_corners(d.geom_xpos[self.book], d.geom_xmat[self.book].reshape(3, 3),
                            m.geom_size[self.book])
        corners = (world - d.body_xpos[self.caddy]) @ rotation
        linear = float(np.linalg.norm(d.get_body_xvelp('black_book_1_main')))
        angular = float(np.linalg.norm(d.get_body_xvelr('black_book_1_main')))
        contacts = [c for s in substeps for c in s['contacts'] if 'black_book' in str(c)]
        gripper = any('gripper' in str(c) for c in contacts)
        # Support is evaluated at the last solver sample; release is required
        # throughout the control interval, not just its final endpoint.
        supported = any('black_book' in str(c) and 'desk_caddy' in str(c)
                        for c in substeps[-1]['contacts'])
        penetration = max([max(0., -c['distance']) for c in contacts
                           if 'desk_caddy' in str(c)] or [0.])
        checks = placement_conditions(corners, self.bounds, self.floor, linear, angular,
                                      gripper, supported, penetration)
        native = bool(self.env.check_success())
        if native and self.first_native is None:
            self.first_native = step
        candidate = all(checks.values())
        self.streak = self.streak + 1 if candidate else 0
        self.max_streak = max(self.max_streak, self.streak)
        complete = self.streak >= DWELL_STEPS
        if complete and self.first_complete is None:
            self.first_complete = step
        return dict(native_success=native, conditions=checks, candidate=candidate,
                    streak=self.streak, complete=complete, book_corners_caddy=corners.tolist(),
                    linear_speed=linear, angular_speed=angular,
                    max_book_caddy_penetration=penetration)


class ContactRecorder:
    """Observe existing mj_step calls without extra forward/step/control calls."""
    def __init__(self, sim):
        self.sim = sim
        self.original_step = sim.step
        self.samples = []
        sim.step = self.step

    def step(self, *args, **kwargs):
        import mujoco
        result = self.original_step(*args, **kwargs)
        contacts = []
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            a = self.sim.model.geom_id2name(c.geom1) or ''
            b = self.sim.model.geom_id2name(c.geom2) or ''
            if not any(k in a + b for k in ('black_book', 'desk_caddy')):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, i, wrench)
            contacts.append(dict(geom1=a, geom2=b, wrench=wrench.tolist(), distance=float(c.dist)))
        self.samples.append(dict(time=float(self.sim.data.time), contacts=contacts))
        return result

    def reset(self):
        self.samples = []

    def close(self):
        self.sim.step = self.original_step
