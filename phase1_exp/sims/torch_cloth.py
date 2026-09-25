"""
Task B — cloth corner-fold (deformable headline task), differentiable PyTorch GT.

An N x N mass-spring cloth (structural + shear + bend springs) lies on a table.
A kinematic gripper holds corner particle 0 and must fold it onto the opposite
corner's *initial* world position (diagonal fold).

Constraints (plan §3.3):
  * deformation  = max structural-spring strain  (tear/damage proxy)
  * force        = |net spring force on the grasped particle| (grasp overload)

State (B, 6*P + 7): [x (P*3), v (P*3), tgt (3), grasp_start(3), t]
Action (B, 3): gripper velocity (m/s)
Obs (B, 13): [g-tgt(3), centroid-tgt(3), centroid_vel(3), 10*mean_z,
              strain_max/limit, 10*grasp_z, t/T]

No self-collision (documented limitation); the fold follows an arc so the
grasped corner lands on top of the cloth region supported by the table.
"""
import math

import torch
import torch.nn.functional as F

from sims.base import FunctionalSim


class ClothSim(FunctionalSim):
    name = "cloth"
    act_dim = 3
    obs_dim = 13
    T = 60
    max_vel = 0.3
    param_names = ("stiffness", "mass", "friction")
    regime_names = ("on_table", "lifting", "carrying", "landing")

    N = 8
    L = 0.20
    dt = 0.05
    substeps = 100
    g = 9.81
    k_floor = 2000.0
    c_spring = 0.02
    air_damp = 0.3
    success_tol = 0.03
    force_limit = 1.3       # N   (nominal expert ~0% violations)
    deform_limit = 0.05     # 5 % strain
    beta = 2000.0
    fold_steps = 40
    arc_h = 0.06

    def __init__(self, device):
        super().__init__(device)
        N, d = self.N, self.device
        self.P = N * N
        self.state_dim = 6 * self.P + 7
        ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        idx = (ii * N + jj)
        rest = self.L / (N - 1)
        pairs, rl, kind = [], [], []

        def add(a, b, r, k):
            pairs.append(torch.stack([a.flatten(), b.flatten()], 1))
            rl.append(torch.full((a.numel(),), r))
            kind.append(torch.full((a.numel(),), k))

        add(idx[:-1, :], idx[1:, :], rest, 0)
        add(idx[:, :-1], idx[:, 1:], rest, 0)
        add(idx[:-1, :-1], idx[1:, 1:], rest * math.sqrt(2), 1)
        add(idx[1:, :-1], idx[:-1, 1:], rest * math.sqrt(2), 1)
        add(idx[:-2, :], idx[2:, :], 2 * rest, 2)
        add(idx[:, :-2], idx[:, 2:], 2 * rest, 2)
        self.pairs = torch.cat(pairs).to(d)
        self.rest = torch.cat(rl).to(d)
        kind = torch.cat(kind).to(d)
        self.kscale = torch.tensor([1.0, 0.5, 0.1], device=d)[kind]
        self.structural = kind == 0
        x0 = torch.stack([ii.flatten() * rest - self.L / 2,
                          jj.flatten() * rest - self.L / 2,
                          torch.full((self.P,), 0.001)], 1)
        self.x0 = x0.to(d)
        self.not_grasped = torch.ones(self.P, 1, device=d)
        self.not_grasped[0] = 0.0

    def nominal_params(self):
        d = self.device
        return {"stiffness": torch.tensor(400.0, device=d),
                "mass": torch.tensor(0.1, device=d),
                "friction": torch.tensor(0.5, device=d)}

    # --- packing ----------------------------------------------------------
    def unpack(self, s):
        P = self.P
        x = s[:, : 3 * P].view(-1, P, 3)
        v = s[:, 3 * P: 6 * P].view(-1, P, 3)
        tgt = s[:, 6 * P: 6 * P + 3]
        t = s[:, -1:]
        return x, v, tgt, t

    def start_pos(self, s):
        return s[:, 6 * self.P + 3: 6 * self.P + 6]

    def pack(self, x, v, tgt, t, start):
        B = x.shape[0]
        return torch.cat([x.reshape(B, -1), v.reshape(B, -1), tgt, start, t], 1)

    def init_state(self, n, gen):
        u = torch.rand(n, 3, generator=gen).to(self.device)
        shift = torch.cat([(u[:, :2] - 0.5) * 0.02, torch.zeros(n, 1, device=self.device)], 1)
        x = self.x0[None] + shift[:, None]
        v = torch.zeros_like(x)
        tgt = x[:, -1].clone()
        tgt[:, 2] = 0.006
        tgt[:, :2] += (u[:, 2:3] - 0.5) * 0.02
        return self.pack(x, v, tgt, torch.zeros(n, 1, device=self.device), x[:, 0].clone())

    # --- physics ----------------------------------------------------------
    def _spring_forces(self, x, v, k):
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = x[:, j] - x[:, i]
        l = torch.sqrt((d ** 2).sum(-1) + 1e-12)
        n = d / l[..., None]
        strain = (l - self.rest) / self.rest
        vrel = ((v[:, j] - v[:, i]) * n).sum(-1)
        fmag = k[:, None] * self.kscale * (l - self.rest) + self.c_spring * vrel
        f = fmag[..., None] * n
        F_ = torch.zeros_like(x)
        F_ = F_.index_add(1, i, f).index_add(1, j, -f)
        return F_, strain

    def step(self, s, a, params, smooth=False):
        k, M, mu = params["stiffness"], params["mass"], params["friction"]
        mp = (M / self.P)[:, None, None]
        if smooth:
            a = self.max_vel * torch.tanh(a / self.max_vel)
        else:
            a = a.clamp(-self.max_vel, self.max_vel)
        x, v, tgt, t = self.unpack(s)
        h = self.dt / self.substeps
        fmax = torch.zeros_like(k)
        smax = torch.zeros_like(k)
        grav = torch.tensor([0.0, 0.0, -self.g], device=s.device)
        for _ in range(self.substeps):
            Fs, strain = self._spring_forces(x, v, k)
            gf = torch.sqrt((Fs[:, 0] ** 2).sum(-1) + 1e-12)
            pen = -x[..., 2]
            pen = F.softplus(pen * self.beta) / self.beta if smooth else F.relu(pen)
            touch = torch.sigmoid(pen * self.beta * 5) if smooth else (pen > 0).float()
            az = self.k_floor * pen / 0.001 - 50.0 * v[..., 2] * touch
            acc = Fs / mp + grav - self.air_damp * v
            acc = torch.cat([acc[..., :2], acc[..., 2:] + az[..., None]], -1)
            v = v + acc * h
            # table friction on horizontal velocity of touching particles
            vh = v[..., :2]
            sp = torch.sqrt((vh ** 2).sum(-1) + 1e-12)
            red = mu[:, None] * self.g * h * touch
            nsp = F.softplus((sp - red) * 200) / 200 if smooth else F.relu(sp - red)
            vh = vh * (nsp / sp)[..., None]
            v = torch.cat([vh, v[..., 2:]], -1)
            # kinematic grasp
            v = v * self.not_grasped + (1 - self.not_grasped) * a[:, None]
            x = x + v * h
            fmax = torch.maximum(fmax, gf)
            smax = torch.maximum(smax, strain[:, self.structural].amax(1))
        s2 = self.pack(x, v, tgt, t + 1.0, self.start_pos(s))
        risk = torch.stack([fmax / self.force_limit, smax / self.deform_limit], 1)
        return s2, risk

    # --- observation ------------------------------------------------------
    def _strain_force(self, x):
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = x[:, j] - x[:, i]
        l = torch.sqrt((d ** 2).sum(-1) + 1e-12)
        strain = (l - self.rest) / self.rest
        return strain[:, self.structural].amax(1)

    def obs(self, s):
        x, v, tgt, t = self.unpack(s)
        g = x[:, 0]
        c = x.mean(1)
        cv = v.mean(1)
        st = self._strain_force(x)
        # grasp force proxy (static part only — obs must not depend on params)
        return torch.cat([g - tgt, c - tgt, cv, x[..., 2].mean(1, keepdim=True) * 10,
                          st[:, None] / self.deform_limit,
                          (g[:, 2:3] * 10), t / self.T], 1)

    def success(self, s):
        x, _, tgt, _ = self.unpack(s)
        return torch.sqrt(((x[:, 0] - tgt) ** 2).sum(-1)) < self.success_tol

    def task_cost(self, s):
        x, _, tgt, _ = self.unpack(s)
        return ((x[:, 0] - tgt) ** 2).sum(-1)

    def contact_regime(self, s):
        x, v, tgt, t = self.unpack(s)
        z = x[:, 0, 2]
        prog = (t[:, 0] / self.fold_steps)
        reg = torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
        reg[(z > 0.005) & (prog < 0.3)] = 1
        reg[(z > 0.005) & (prog >= 0.3) & (prog < 0.8)] = 2
        reg[(prog >= 0.8) & (z > 0.003)] = 3
        return reg

    # --- expert -----------------------------------------------------------
    def expert(self, s, params=None, gain=None):
        """Time-parametrised arc fold tuned for nominal physics.
        ``gain`` > 1 speeds the fold up (used for violation-rich data)."""
        x, _, tgt, t = self.unpack(s)
        g = x[:, 0]
        start = self.start_pos(s)
        gn = 1.0 if gain is None else gain[:, None]
        sph = ((t + 1.0) * gn / self.fold_steps).clamp(max=1.0)
        des = start + (tgt - start) * sph
        des = des.clone()
        des[:, 2:3] = des[:, 2:3] + self.arc_h * torch.sin(math.pi * sph)
        a = (des - g) / self.dt
        return a.clamp(-self.max_vel, self.max_vel)

    # --- render -----------------------------------------------------------
    def render(self, s, res=32):
        x, _, tgt, _ = self.unpack(s)
        B = x.shape[0]
        lin = torch.linspace(-0.16, 0.16, res, device=s.device)
        X, Y = torch.meshgrid(lin, lin, indexing="ij")
        X, Y = X[None, None], Y[None, None]
        sig = 0.012
        w = torch.exp(-((X - x[..., 0, None, None]) ** 2 + (Y - x[..., 1, None, None]) ** 2)
                      / (2 * sig ** 2))                       # (B,P,res,res)
        c0 = w.sum(1).clamp(max=3.0) / 3.0
        c1 = (w * (x[..., 2, None, None] * 10)).sum(1).clamp(max=3.0) / 3.0
        gx = torch.exp(-((X[:, 0] - x[:, 0, 0, None, None]) ** 2 + (Y[:, 0] - x[:, 0, 1, None, None]) ** 2)
                       / (2 * sig ** 2))
        tx = torch.exp(-((X[:, 0] - tgt[:, 0, None, None]) ** 2 + (Y[:, 0] - tgt[:, 1, None, None]) ** 2)
                       / (2 * sig ** 2))
        c2 = torch.maximum(gx * (1 + 10 * x[:, 0, 2, None, None]).clamp(max=2) / 2, 0.5 * tx)
        return torch.stack([c0, c1, c2], 1)
