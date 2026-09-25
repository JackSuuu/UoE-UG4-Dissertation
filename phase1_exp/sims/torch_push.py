"""
Task A — planar push-insertion (rigid control task), differentiable PyTorch GT.

A disk "peg" (radius R_b) is pushed by a kinematic point pusher towards a seat
at (target_x, ty). Behind the seat is a rigid end-stop wall at
x_w = target_x + R_b + gap. Arriving too fast -> large impact force against the
end-stop (the "contact-force overload" constraint). Low friction makes the peg
coast; heavy pegs carry more momentum — both are OOD failure modes for a policy
trained at nominal physics.

State (B, 8): [bx, by, vx, vy, px, py, ty, t]
Action (B, 2): pusher velocity (m/s), clipped to ±max_vel
Obs (B, 7):   [bx-tx, by-ty, vx, vy, px-bx, py-by, t/T]   (object-centric)

``smooth=True`` swaps every non-smooth primitive (relu contact, stick/slip
friction, action clamp) for a smooth relaxation. It is used only as the
"relaxation gradient" in the adaptive gradient stabilizer (RQ3); the GT
dynamics are the hard (non-smooth) version, mirroring Genesis's contact model
where gradients can be zero/undefined.
"""
import math

import torch
import torch.nn.functional as F

from sims.base import FunctionalSim


class PushSim(FunctionalSim):
    name = "push"
    state_dim = 8
    obs_dim = 7
    act_dim = 2
    T = 80
    max_vel = 0.5
    param_names = ("friction", "mass")
    regime_names = ("free", "pusher_contact", "wall_contact", "stuck")

    # geometry / numerics
    dt = 0.05
    substeps = 10
    R_b = 0.03
    r_p = 0.01
    k_c = 3000.0
    c_c = 15.0
    k_w = 8000.0
    c_w = 20.0
    g = 9.81
    target_x = 0.30
    gap = 0.002
    success_tol = 0.02
    success_vel = 0.05
    force_limit = 12.0      # N  (nominal expert ~0% violations)
    deform_limit = 1.0      # unused for rigid task
    beta = 400.0            # smoothing sharpness

    def __init__(self, device):
        super().__init__(device)
        self.x_w = self.target_x + self.R_b + self.gap

    def nominal_params(self):
        d = self.device
        return {"friction": torch.tensor(0.25, device=d),
                "mass": torch.tensor(0.5, device=d)}

    def init_state(self, n, gen):
        u = torch.rand(n, 3, generator=gen).to(self.device)
        bx = -0.02 + 0.04 * u[:, 0]
        by = -0.04 + 0.08 * u[:, 1]
        ty = -0.08 + 0.16 * u[:, 2]
        # pusher placed behind the peg along the peg->target line
        dx, dy = self.target_x - bx, ty - by
        nrm = torch.sqrt(dx ** 2 + dy ** 2)
        off = self.R_b + self.r_p + 0.0005
        px, py = bx - dx / nrm * off, by - dy / nrm * off
        z = torch.zeros_like(bx)
        return torch.stack([bx, by, z, z, px, py, ty, z], 1)

    # ------------------------------------------------------------------
    def _relu(self, x, smooth):
        return F.softplus(x * self.beta) / self.beta if smooth else F.relu(x)

    def step(self, s, a, params, smooth=False):
        mu, m = params["friction"], params["mass"]
        if smooth:
            a = self.max_vel * torch.tanh(a / self.max_vel)
        else:
            a = a.clamp(-self.max_vel, self.max_vel)
        b, v, p = s[:, 0:2], s[:, 2:4], s[:, 4:6]
        h = self.dt / self.substeps
        fmax = torch.zeros_like(mu)
        R = self.R_b + self.r_p
        for _ in range(self.substeps):
            p = p + a * h
            d = b - p
            dist = torch.sqrt((d ** 2).sum(-1) + 1e-10)
            n = d / dist[:, None]
            pen = self._relu(R - dist, smooth)
            in_c = torch.sigmoid((R - dist) * self.beta) if smooth else (pen > 0).float()
            vrel = ((v - a) * n).sum(-1)
            fc = self._relu(self.k_c * pen - self.c_c * vrel * in_c, smooth)
            wpen = self._relu(b[:, 0] + self.R_b - self.x_w, smooth)
            in_w = (torch.sigmoid((b[:, 0] + self.R_b - self.x_w) * self.beta)
                    if smooth else (wpen > 0).float())
            fw = self._relu(self.k_w * wpen + self.c_w * v[:, 0] * in_w, smooth)
            force = fc[:, None] * n
            force = force + torch.stack([-fw, torch.zeros_like(fw)], 1)
            v = v + force / m[:, None] * h
            # Coulomb friction as a velocity-magnitude reduction (stick when slow)
            speed = torch.sqrt((v ** 2).sum(-1) + 1e-12)
            new_speed = self._relu(speed - mu * self.g * h, smooth)
            v = v * (new_speed / speed)[:, None]
            b = b + v * h
            fmax = torch.maximum(fmax, torch.maximum(fc, fw))
        t = s[:, 7:8] + 1.0
        s2 = torch.cat([b, v, p, s[:, 6:7], t], 1)
        risk = torch.stack([fmax / self.force_limit, torch.zeros_like(fmax)], 1)
        return s2, risk

    def obs(self, s):
        return torch.stack([
            s[:, 0] - self.target_x, s[:, 1] - s[:, 6], s[:, 2], s[:, 3],
            s[:, 4] - s[:, 0], s[:, 5] - s[:, 1], s[:, 7] / self.T], 1)

    def success(self, s):
        d = torch.sqrt((s[:, 0] - self.target_x) ** 2 + (s[:, 1] - s[:, 6]) ** 2)
        sp = torch.sqrt(s[:, 2] ** 2 + s[:, 3] ** 2)
        return (d < self.success_tol) & (sp < self.success_vel)

    def task_cost(self, s):
        return (s[:, 0] - self.target_x) ** 2 + (s[:, 1] - s[:, 6]) ** 2

    def contact_regime(self, s):
        b, p = s[:, 0:2], s[:, 4:6]
        dist = torch.sqrt(((b - p) ** 2).sum(-1))
        sp = torch.sqrt(s[:, 2] ** 2 + s[:, 3] ** 2)
        reg = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
        reg[dist < self.R_b + self.r_p + 0.003] = 1
        reg[s[:, 0] + self.R_b > self.x_w - 0.003] = 2
        reg[(reg == 0) & (sp < 1e-4)] = 3
        return reg

    # ------------------------------------------------------------------
    def expert(self, s, params=None, gain=None):
        """Scripted expert tuned for *nominal* physics (decelerates near the seat)."""
        b, p, ty = s[:, 0:2], s[:, 4:6], s[:, 6]
        tgt = torch.stack([torch.full_like(ty, self.target_x), ty], 1)
        e = tgt - b
        dist = torch.sqrt((e ** 2).sum(-1) + 1e-10)
        dirn = e / dist[:, None]
        behind = b - dirn * (self.R_b + self.r_p - 0.001)
        g = 1.0 if gain is None else gain[:, None]
        ramp = ((s[:, 7] + 1.0) / 6.0).clamp(max=1.0)[:, None]
        v_des = (5.0 * (dist - 0.004)).clamp(0.0, 0.4)[:, None] * g * ramp
        a = 8.0 * (behind - p) + dirn * v_des
        a = a * (dist > 0.006).float()[:, None]          # seated -> hold still
        return a.clamp(-self.max_vel, self.max_vel)

    # ------------------------------------------------------------------
    def render(self, s, res=32):
        """3-channel top-down image: peg, pusher, target+end-stop."""
        B = s.shape[0]
        xs = torch.linspace(-0.08, 0.40, res, device=s.device)
        ys = torch.linspace(-0.24, 0.24, res, device=s.device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        X, Y = X[None], Y[None]

        def blob(cx, cy, r):
            return torch.exp(-((X - cx[:, None, None]) ** 2 + (Y - cy[:, None, None]) ** 2)
                             / (2 * r ** 2))

        c0 = blob(s[:, 0], s[:, 1], self.R_b)
        c1 = blob(s[:, 4], s[:, 5], 0.012)
        c2 = blob(torch.full((B,), self.target_x, device=s.device), s[:, 6], 0.015)
        c2 = torch.maximum(c2, torch.exp(-((X - self.x_w) ** 2) / (2 * 0.006 ** 2)).expand(B, -1, -1))
        return torch.stack([c0, c1, c2], 1)
