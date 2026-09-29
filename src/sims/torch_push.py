"""
Task A — planar push-insertion (rigid control task), differentiable PyTorch GT.

A square "peg" (half-width R_b, so side 2*R_b) is pushed by a kinematic
two-fingertip pusher (two points 2*finger_dy apart along y, like a closed Franka
gripper; p is their midpoint) towards a seat at (target_x, ty). Behind the seat is a rigid end-stop wall
at x_w = target_x + R_b + gap. Arriving too fast -> large impact force against
the end-stop (the "contact-force overload" constraint). Low friction makes the
peg coast; heavy pegs carry more momentum — both are OOD failure modes for a
policy trained at nominal physics.

The peg is a box, not a disk, so that the same geometry can be simulated with
gradients in Genesis 1.4 (cylinder-sphere contacts are not detected in its
differentiable mode). A box pushed by a point also rotates, so the peg has a
yaw angle and yaw rate. Pushing a box with a single point is unstable (any
offset spins it; tested: nominal expert SR 0.11-0.12), so the pusher has two
fingertips: a two-point contact on a face turns the box back into alignment.
Each fingertip contact has Coulomb friction mu_p.
The peg-wall contact is frictionless; the floor carries the OOD friction mu.

State (B, 10): [bx, by, vx, vy, px, py, ty, t, th, w]
               (th, w appended at the end so older column indices stay valid)
Action (B, 2): pusher velocity (m/s), clipped to ±max_vel
Obs (B, 10):   [bx-tx, by-ty, vx, vy, px-bx, py-by, t/T, sin4th, cos4th, w/10]
               (4*th because a square is symmetric under 90° rotations)

``smooth=True`` swaps every non-smooth primitive (relu contact, stick/slip
friction, action clamp) for a smooth relaxation. It is used only as the
"relaxation gradient" in the adaptive gradient stabilizer (RQ3); the GT
dynamics are the hard (non-smooth) version, mirroring Genesis's contact model
where gradients can be zero/undefined.
"""
import torch
import torch.nn.functional as F

from sims.base import FunctionalSim

# local corner offsets of the unit square
_CORNERS = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])


class PushSim(FunctionalSim):
    name = "push"
    state_dim = 10
    obs_dim = 10
    act_dim = 2
    T = 80
    max_vel = 0.5
    param_names = ("friction", "mass")
    regime_names = ("free", "pusher_contact", "wall_contact", "stuck")

    # geometry / numerics
    dt = 0.05
    substeps = 10
    R_b = 0.03              # peg half-width
    r_p = 0.01
    k_c = 3000.0
    c_c = 15.0
    k_w = 8000.0
    c_w = 20.0
    g = 9.81
    target_x = 0.30
    gap = 0.006             # clearance; a box yawed by 0.2 rad is ~5 mm wider in x than an aligned one
    success_tol = 0.02
    success_vel = 0.05
    force_limit = 12.0      # N  (nominal expert ~0% violations)
    deform_limit = 1.0      # unused for rigid task
    beta = 400.0            # smoothing sharpness
    th0_range = 0.15        # initial yaw is uniform in ±th0_range (rad)
    mu_p = 0.5              # fingertip (pusher-peg) friction; not an OOD axis (1.0 made gradients ill-conditioned)
    v_slip = 0.005          # m/s, regularisation of the fingertip friction
    k_y = 15.0              # expert lateral-correction gain (1/m)
    finger_dy = 0.015       # fingertips at p ± (0, finger_dy): 3 cm apart, side by side along y

    def __init__(self, device):
        super().__init__(device)
        self.x_w = self.target_x + self.R_b + self.gap
        self.c_fric = 0.765 * self.R_b                 # friction lever arm of the square
        self.rot_fric_scale = self.c_fric ** 2 / ((2.0 / 3.0) * self.R_b ** 2)   # m*c^2/I
        self.corners = _CORNERS.to(self.device) * self.R_b
        self.fingers = [torch.tensor([0.0, self.finger_dy], device=self.device),
                        torch.tensor([0.0, -self.finger_dy], device=self.device)]

    def nominal_params(self):
        d = self.device
        return {"friction": torch.tensor(0.25, device=d),
                "mass": torch.tensor(0.5, device=d)}

    # ------------------------------------------------------------------
    def _surface_dist(self, dirn, th):
        """Centre-to-surface distance of the box along unit direction dirn (B,2)."""
        c, s = torch.cos(th), torch.sin(th)
        d1 = (dirn[:, 0] * c + dirn[:, 1] * s).abs()
        d2 = (-dirn[:, 0] * s + dirn[:, 1] * c).abs()
        return self.R_b / torch.maximum(d1, d2).clamp(min=1e-3)

    def init_state(self, n, gen):
        u = torch.rand(n, 4, generator=gen, device=gen.device).to(self.device)
        bx = -0.02 + 0.04 * u[:, 0]
        by = -0.04 + 0.08 * u[:, 1]
        ty = -0.08 + 0.16 * u[:, 2]
        th = self.th0_range * (2 * u[:, 3] - 1)
        # pusher placed behind the peg along the peg->target line
        dx, dy = self.target_x - bx, ty - by
        nrm = torch.sqrt(dx ** 2 + dy ** 2)
        dirn = torch.stack([dx / nrm, dy / nrm], 1)
        # extra margin: an off-axis fingertip meets a yawed face earlier
        off = self._surface_dist(dirn, th) + self.r_p + 0.0005 + self.finger_dy * th.abs().sin()
        px, py = bx - dirn[:, 0] * off, by - dirn[:, 1] * off
        z = torch.zeros_like(bx)
        return torch.stack([bx, by, z, z, px, py, ty, z, th, z], 1)

    # ------------------------------------------------------------------
    def _relu(self, x, smooth):
        return F.softplus(x * self.beta) / self.beta if smooth else F.relu(x)

    def _pusher_contact(self, b, th, p):
        """Signed distance from the pusher centre to the box, and the outward
        world-frame normal at the closest box point (pointing peg -> pusher)."""
        c, s = torch.cos(th), torch.sin(th)
        d = p - b
        q = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
        qa = q.abs() - self.R_b
        out_vec = F.relu(qa)
        out_norm = torch.sqrt((out_vec ** 2).sum(-1) + 1e-12)
        inside = torch.clamp(qa.max(-1).values, max=0.0)
        sdf = torch.where(out_norm > 1e-6, out_norm, torch.zeros_like(out_norm)) + inside
        sgn = torch.where(q >= 0, torch.ones_like(q), -torch.ones_like(q))
        n_out = out_vec * sgn / out_norm[:, None]
        n_in = F.one_hot(qa.argmax(-1), 2).float() * sgn
        n_loc = torch.where((out_norm > 1e-6)[:, None], n_out, n_in)
        n = torch.stack([c * n_loc[:, 0] - s * n_loc[:, 1], s * n_loc[:, 0] + c * n_loc[:, 1]], 1)
        return sdf, n

    def step(self, s, a, params, smooth=False):
        mu, m = params["friction"], params["mass"]
        if smooth:
            a = self.max_vel * torch.tanh(a / self.max_vel)
        else:
            a = a.clamp(-self.max_vel, self.max_vel)
        b, v, p = s[:, 0:2], s[:, 2:4], s[:, 4:6]
        th, w = s[:, 8], s[:, 9]
        h = self.dt / self.substeps
        I = (2.0 / 3.0) * m * self.R_b ** 2
        fmax = torch.zeros_like(mu)
        for _ in range(self.substeps):
            p = p + a * h
            # --- two fingertips -> peg (normal + regularised Coulomb friction) ---
            Fp = torch.zeros_like(v)
            tau = torch.zeros_like(w)
            fc_sum = torch.zeros_like(w)
            for off in self.fingers:
                pf = p + off
                sdf, n_out = self._pusher_contact(b, th, pf)
                n = -n_out                                # fingertip -> peg
                pen = self._relu(self.r_p - sdf, smooth)
                in_c = torch.sigmoid((self.r_p - sdf) * self.beta) if smooth else (pen > 0).float()
                rc = pf - n_out * self.r_p - b            # contact point rel. peg centre
                vc = v + w[:, None] * torch.stack([-rc[:, 1], rc[:, 0]], 1)
                vrel = ((vc - a) * n).sum(-1)
                fc = self._relu(self.k_c * pen - self.c_c * vrel * in_c, smooth)
                tvec = torch.stack([-n[:, 1], n[:, 0]], 1)
                vt = ((vc - a) * tvec).sum(-1)
                ft = -self.mu_p * fc * torch.tanh(vt / self.v_slip)
                Fk = fc[:, None] * n + ft[:, None] * tvec
                Fp = Fp + Fk
                tau = tau + rc[:, 0] * Fk[:, 1] - rc[:, 1] * Fk[:, 0]
                fc_sum = fc_sum + fc
            # --- end-stop wall on every corner (frictionless) ------------
            cth, sth = torch.cos(th), torch.sin(th)
            kx = cth[:, None] * self.corners[:, 0] - sth[:, None] * self.corners[:, 1]   # (B,4)
            ky = sth[:, None] * self.corners[:, 0] + cth[:, None] * self.corners[:, 1]
            wd = b[:, 0:1] + kx - self.x_w
            wpen = self._relu(wd, smooth)
            in_w = torch.sigmoid(wd * self.beta) if smooth else (wpen > 0).float()
            vcx = v[:, 0:1] - w[:, None] * ky
            fwk = self._relu(self.k_w * wpen + self.c_w * vcx * in_w, smooth)          # (B,4)
            fw = fwk.sum(-1)
            tau = tau + (ky * fwk).sum(-1)
            # --- integrate --------------------------------------------------
            force = Fp + torch.stack([-fw, torch.zeros_like(fw)], 1)
            v = v + force / m[:, None] * h
            w = w + tau / I * h
            # Coulomb floor friction with coupled translation/rotation (ellipsoidal
            # limit surface): a sliding box offers almost no resistance to turning.
            # u = [v, c*w] with c = 0.765*R_b (mean lever arm of a uniformly loaded
            # square); friction decelerates |u| by mu*g*h (rotation scaled by m*c^2/I).
            # Treating the two independently (as before) over-damped rotation while
            # sliding and did not match Genesis for off-centre pushes.
            cw = self.c_fric * w
            rho = torch.sqrt((v ** 2).sum(-1) + cw ** 2 + 1e-12)
            dec = mu * self.g * h
            v = v * (self._relu(rho - dec, smooth) / rho)[:, None]
            w = w * self._relu(rho - self.rot_fric_scale * dec, smooth) / rho
            b = b + v * h
            th = th + w * h
            fmax = torch.maximum(fmax, torch.maximum(fc_sum, fw))
        t = s[:, 7:8] + 1.0
        s2 = torch.cat([b, v, p, s[:, 6:7], t, th[:, None], w[:, None]], 1)
        risk = torch.stack([fmax / self.force_limit, torch.zeros_like(fmax)], 1)
        return s2, risk

    def obs(self, s):
        return torch.stack([
            s[:, 0] - self.target_x, s[:, 1] - s[:, 6], s[:, 2], s[:, 3],
            s[:, 4] - s[:, 0], s[:, 5] - s[:, 1], s[:, 7] / self.T,
            torch.sin(4 * s[:, 8]), torch.cos(4 * s[:, 8]), s[:, 9] / 10.0], 1)

    def success(self, s):
        d = torch.sqrt((s[:, 0] - self.target_x) ** 2 + (s[:, 1] - s[:, 6]) ** 2)
        sp = torch.sqrt(s[:, 2] ** 2 + s[:, 3] ** 2)
        return (d < self.success_tol) & (sp < self.success_vel)

    def task_cost(self, s):
        return (s[:, 0] - self.target_x) ** 2 + (s[:, 1] - s[:, 6]) ** 2

    def contact_regime(self, s):
        b, p, th = s[:, 0:2], s[:, 4:6], s[:, 8]
        sdf = torch.minimum(*[self._pusher_contact(b, th, p + off)[0] for off in self.fingers])
        cth, sth = torch.cos(th), torch.sin(th)
        kx = cth[:, None] * self.corners[:, 0] - sth[:, None] * self.corners[:, 1]
        sp = torch.sqrt(s[:, 2] ** 2 + s[:, 3] ** 2)
        reg = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
        reg[sdf < self.r_p + 0.003] = 1
        reg[(s[:, 0:1] + kx).max(-1).values > self.x_w - 0.003] = 2
        reg[(reg == 0) & (sp < 1e-4)] = 3
        return reg

    # ------------------------------------------------------------------
    def expert(self, s, params=None, gain=None):
        """Scripted expert tuned for *nominal* physics: align, then insert straight.

        The push heading is mostly +x with a lateral correction atan(k_y * e_y), so
        the lateral error decays along the way and the final approach to the
        end-stop is straight. (Steering straight at the seat made the pusher swing
        sideways near the end, push off-centre, yaw the box and wedge a corner
        against the wall.) The pusher stays on the heading line through the peg
        centre, at the box surface; speed ramps down with the remaining depth."""
        b, p, ty, th = s[:, 0:2], s[:, 4:6], s[:, 6], s[:, 8]
        ey = ty - b[:, 1]
        depth = self.target_x - b[:, 0]
        head = torch.stack([torch.ones_like(ey), (self.k_y * ey).clamp(-0.5, 0.5)], 1)
        dirn = head / head.norm(dim=1, keepdim=True)
        # Stay centred on the box's back face (box -x axis) so both fingertips keep
        # contact; the lateral correction comes from moving diagonally and letting
        # fingertip friction carry the box, not from moving the pusher off-centre.
        back = torch.stack([torch.cos(th), torch.sin(th)], 1)
        behind = b - back * (self.R_b + self.r_p - 0.001)
        g = 1.0 if gain is None else gain[:, None]
        ramp = ((s[:, 7] + 1.0) / 6.0).clamp(max=1.0)[:, None]
        v_des = (5.0 * (depth + 0.002)).clamp(0.0, 0.4)[:, None] * g * ramp
        a = 8.0 * (behind - p) + dirn * v_des
        a = a * (depth > 0.004).float()[:, None]           # at insertion depth -> hold still
        return a.clamp(-self.max_vel, self.max_vel)

    # ------------------------------------------------------------------
    def render(self, s, res=32):
        """3-channel top-down image: peg (rotated square), pusher, target+end-stop."""
        B = s.shape[0]
        xs = torch.linspace(-0.08, 0.40, res, device=s.device)
        ys = torch.linspace(-0.24, 0.24, res, device=s.device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        X, Y = X[None], Y[None]

        def blob(cx, cy, r):
            return torch.exp(-((X - cx[:, None, None]) ** 2 + (Y - cy[:, None, None]) ** 2)
                             / (2 * r ** 2))

        c, sn = torch.cos(s[:, 8])[:, None, None], torch.sin(s[:, 8])[:, None, None]
        dx, dy = X - s[:, 0, None, None], Y - s[:, 1, None, None]
        qx, qy = c * dx + sn * dy, -sn * dx + c * dy
        c0 = torch.sigmoid((self.R_b - torch.maximum(qx.abs(), qy.abs())) / 0.006)
        c1 = blob(s[:, 4], s[:, 5], 0.012)
        c2 = blob(torch.full((B,), self.target_x, device=s.device), s[:, 6], 0.015)
        c2 = torch.maximum(c2, torch.exp(-((X - self.x_w) ** 2) / (2 * 0.006 ** 2)).expand(B, -1, -1))
        return torch.stack([c0, c1, c2], 1)
