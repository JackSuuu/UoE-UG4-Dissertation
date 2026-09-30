"""
An RGB camera for the torch ground-truth scene (Task A, planar push).

Why this exists
---------------
``PushSim.render`` is a 32x32 three-channel *diagram*: a top-down orthographic
blit of the peg, the pusher and the target, built for the vision world model
baseline. It is not a camera and a VLA cannot use it -- OpenVLA and friends
expect a perspective RGB image, and ``Env.render_rgb`` raised
``NotImplementedError``, which blocked every real-VLA experiment. Genesis has a
camera, but Genesis is the *optional* cross-check backend (it is worse here:
lateral error 3.4 cm against the torch GT's 0.4 cm, cause unknown), so it cannot
be the thing a VLA depends on. This module is the missing prerequisite for P2a.

It is a **pure renderer**. It reads the state tensor and returns pixels; it never
writes ``self.state``, never calls ``sim.step``, and touches no physical
parameter. A camera cannot change the physics, so this cannot invalidate any
result computed with ``--backend torch``; the dynamics, the risk definition and
the ground truth are untouched. The Genesis parity question is a separate
problem and is not addressed here.

Scene
-----
The physics is 2-D in ``(x, y)`` with the peg sliding on a table, so the render
gives it a third dimension ``z`` (up) purely for the camera. Everything is a
yaw-oriented box or the table plane, which means one slab test handles the lot:

    table      plane z = 0
    peg        box, half-width ``sim.R_b``, height ``peg_h``
    fingers    two boxes at ``p +/- (0, sim.finger_dy)``, height ``finger_h``
    wrist      one box above the fingers, so the end-effector is visible
    wall       vertical plate at ``sim.x_w`` -- the object the constraint is about
    seat       a printed target marker on the table at ``(target_x, ty)``

The wall is included on purpose. The whole thesis is about the contact-force
constraint at the end-stop, so the camera must make the wall and the gap to it
legible; a VLA that cannot see the end-stop cannot learn to avoid it.

Shading
-------
Lambert + Blinn-Phong, one directional light per env, plus a cheap contact
darkening on the table near the peg and the pusher. Two honest caveats, both of
which belong in the paper rather than in a footnote:

  * The contact darkening is a *rendered* cue, not simulated lighting. It is
    what a real camera would show (contact shadows are visible), and it does
    encode proximity -- physically the right variable -- but it makes the visual
    task easier than the physical one. State it as a limitation.
  * Shading is not calibrated to any radiometric model. It is plausible-looking
    and domain-randomised, which is what the job needs; it is not a simulator.

Domain randomisation
--------------------
Appearance is randomised per env (light azimuth/elevation/intensity/tint, and
the colour of table, peg, gripper, wall, marker and background) so a policy
cannot key on a fixed lighting condition. Camera *pose* is fixed, which is the
normal setup for a cell-mounted camera; per-env pose jitter is not supported
because it would make the ray set per-env and roughly triple renderer memory.

Determinism: appearance is drawn from a ``torch.Generator`` owned by the camera
and reseeded by :meth:`reset_appearance`, so a given seed always gives the same
pixels. Nothing here uses global RNG.

Cost, measured on one A5000 for 64 envs at 224x224:

    ss=1     42 ms      ss=2  146 ms      32x32 ss=1   8 ms

``ss`` defaults to 1 for that reason. Supersampling is a 3.5x cost for smoother
edges, and at 146 ms a render is 29 ms/step even amortised over a 5-step chunk --
more than half the 50 ms budget this project runs under, spent on antialiasing.
The Controller also caches the frame across a chunk (see ``Controller.act``), so
the per-step cost is a fifth of the per-render cost. Raise ``ss`` only if frame
quality is shown to matter; do not assume it is free.

Rendered in row blocks, so peak memory is set by ``block`` rather than by
``res``: 64 envs at ss=2 is 12.8M rays, which unsupersampled per-element would
need several GB of temporaries. Measured peak 1.12 GiB.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# (r, g, b, specular, shininess) as linear radiance before gamma encoding.
_TABLE = (0.55, 0.54, 0.51, 0.04, 12.0)
_PEG = (0.80, 0.36, 0.20, 0.35, 48.0)
_GRIP = (0.86, 0.87, 0.89, 0.60, 90.0)
_WALL = (0.30, 0.32, 0.36, 0.20, 30.0)
_MARK = (0.16, 0.55, 0.28, 0.10, 16.0)

MAT_FLOOR, MAT_PEG, MAT_GRIP, MAT_WALL, MAT_MARK, MAT_BG = 0, 1, 2, 3, 4, 5

# camera-only geometry: the 2-D dynamics never look at z
PEG_H, FIN_H, WRIST_H = 0.024, 0.040, 0.014
WALL_HALF = (0.008, 0.16, 0.050)
# The workspace is a finite region of the table, wider than the reachable state
# range (bx in [-0.02, 0.02], x_w = 0.336) so the pusher and the wall are framed.
# It bounds both the table plane and the box primitives: a box whose hit lands
# outside it is culled, so a state that somehow left the workspace shows empty
# table rather than an object floating in mid-air.
TABLE_X, TABLE_Y = (-0.13, 0.40), (-0.20, 0.20)
MARK_HALF, MARK_RIM = 0.048, 0.040
SHADOW_OFF, SHADOW_SIG, SHADOW_K = 0.022, 0.020, 0.50


class PushCamera:
    """Fixed-perspective RGB camera for :class:`~sims.torch_push.PushSim`.

    ``render(state)`` -> float32 ``(B, 3, res, res)`` in ``[0, 1]``,
    gamma-encoded (sRGB-like), which is what a pretrained VLA expects.
    """

    def __init__(self, sim, device, res=224, ss=1, block=65536,
                 pos=(0.13, -0.42, 0.40), look=(0.15, 0.0, 0.02),
                 fov_deg=48.0, seed=0, randomize=True, n=1):
        self.sim, self.dev = sim, torch.device(device)
        self.res, self.ss, self.block = int(res), int(ss), int(block)
        self.randomize = bool(randomize)
        self.gen = torch.Generator().manual_seed(int(seed))
        d = self.dev
        self.pos = torch.tensor(pos, dtype=torch.float32, device=d)
        self.look = torch.tensor(look, dtype=torch.float32, device=d)
        self.fov = math.radians(fov_deg)
        # Appearance is provisional until the first Env.reset, which redraws it
        # from the episode generator so one seed fixes the pixels too.
        self.appearance = None
        self.appearance_rows = n
        self.reset_appearance(n)

    # ------------------------------------------------------------------ setup
    def reset_appearance(self, n: int, gen: torch.Generator | None = None):
        """Draw a fresh appearance per env. ``Env.reset`` calls this with the
        episode's own generator, so a seed fixes the pixels just as it fixes
        the initial states."""
        if not self.randomize:
            self.appearance = None
            self.appearance_rows = n
            return
        g = gen if gen is not None else self.gen
        dev = self.dev
        self.appearance_rows = n
        r = lambda *s: torch.rand(*s, generator=g, device="cpu").to(dev)  # noqa: E731

        # light: azimuth swung +/-100 deg around a base direction, elevation 25-65
        az = (-1.75 + 3.5 * r(n)) * math.pi
        el = (25.0 + 40.0 * r(n)) * math.pi / 180.0
        L = torch.stack([torch.cos(el) * torch.cos(az),
                         torch.cos(el) * torch.sin(az),
                         torch.sin(el)], 1)
        self.appearance = {
            "L": L / L.norm(dim=1, keepdim=True).clamp(min=1e-6),
            # scalar knobs are kept 1-D (B,) so the renderer's `[:, None]`
            # promotes them to (B,1) and they broadcast against (B,P)
            "amb": 0.18 + 0.22 * r(n),
            "diff": 0.75 + 0.35 * r(n),
            "tint": (1.0 + 0.25 * (r(n, 3) - 0.5) * 2.0).clamp(min=0.3),
            "spec": 0.7 + 0.6 * r(n),
            "col_table": self._jitter(_TABLE, r(n, 3)),
            "col_peg": self._jitter(_PEG, r(n, 3)),
            "col_grip": self._jitter(_GRIP, r(n, 3)),
            "col_wall": self._jitter(_WALL, r(n, 3)),
            "col_mark": self._jitter(_MARK, r(n, 3)),
            "col_bg": self._jitter((0.20, 0.21, 0.24, 0.0, 1.0), r(n, 3)),
        }

    @staticmethod
    def _in_table(p):
        """(...,3) -> (...,) bool: is this world point on the workspace table?"""
        (x0, x1), (y0, y1) = TABLE_X, TABLE_Y
        return (p[..., 0] > x0) & (p[..., 0] < x1) & (p[..., 1] > y0) & (p[..., 1] < y1)

    @staticmethod
    def _jitter(material, u):
        """Per-env colour jitter around a base material, ``u`` uniform (B,3).

        +/-15% overall value (one scalar per env, so the hue shift does not also
        change brightness) and +/-12% per channel.
        """
        base = torch.tensor(material[:3], device=u.device)
        scale = (0.85 + 0.30 * u[:, :1])                            # (B,1)
        hue = 1.0 + 0.12 * (u - 0.5) * 2.0                          # (B,3)
        return (base.unsqueeze(0) * scale * hue).clamp(0.02, 0.98)

    def _flat_appearance(self, B):
        """Fixed appearance, used when domain randomisation is off."""
        dev = self.dev
        out = {f"col_{k}": torch.tensor(v[:3], device=dev).expand(B, 3).contiguous()
               for k, v in (("table", _TABLE), ("peg", _PEG), ("grip", _GRIP),
                            ("wall", _WALL), ("mark", _MARK))}
        out["col_bg"] = torch.tensor([0.20, 0.21, 0.24], device=dev).expand(B, 3).contiguous()
        out.update({"L": torch.tensor([[0.4, -0.5, 0.77]], device=dev).expand(B, 3).contiguous(),
                    "amb": torch.full((B,), 0.28, device=dev),
                    "diff": torch.full((B,), 0.90, device=dev),
                    "spec": torch.full((B,), 1.0, device=dev),
                    "tint": torch.ones(B, 3, device=dev)})
        return out

    # ----------------------------------------------------------------- rays
    def _rays(self, n):
        """Ray directions for an ``n``-pixel square image -> (dir, view) each
        ``(n*n, 3)``.

        Pinhole: the image plane sits at unit distance from the eye, so the
        perspective divide is already folded into the directions.
        """
        f = self.look - self.pos
        f = f / f.norm()
        up = torch.tensor([0.0, 0.0, 1.0], device=f.device)
        r = torch.cross(f, up, dim=0)
        r = r / r.norm().clamp(min=1e-6)
        u = torch.cross(r, f, dim=0)
        hx = math.tan(self.fov / 2.0)
        j = torch.linspace(-1.0, 1.0, n, device=f.device)
        gy, gx = torch.meshgrid(j, j, indexing="ij")
        d = (f[None, None, :] + r[None, None, :] * (gx[..., None] * hx)
             + u[None, None, :] * (gy[..., None] * hx))
        d = d.reshape(n * n, 3)
        return d, -d

    # ------------------------------------------------------------ primitives
    def _slab(self, o, d, centre, half, yaw):
        """Ray vs yaw-oriented box. ``o`` (3,), ``d`` (P,3), ``centre``/``half``
        (B,3), ``yaw`` (B,) -> ``(t, n_local)``, both (B,P).

        ``t`` is ``inf`` on a miss. ``n_local`` is the outward face normal in
        the box's own frame: the entry axis, signed away from the eye, which
        follows from the slab test -- the entry face is at ``-half`` when the
        ray runs along ``+axis`` and at ``+half`` when it runs along ``-axis``.
        """
        B, P = centre.shape[0], d.shape[0]
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]        # (B,1)

        rel = (o.view(1, 1, 3) - centre[:, None, :])                  # (B,1,3)
        lo = torch.stack([c * rel[..., 0] + s * rel[..., 1],
                          -s * rel[..., 0] + c * rel[..., 1],
                          rel[..., 2]], -1)                            # (B,1,3)
        dl = torch.stack([c * d[None, :, 0] + s * d[None, :, 1],
                          -s * d[None, :, 0] + c * d[None, :, 1],
                          d[None, :, 2].expand(B, P)], -1)             # (B,P,3)

        safe = torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
        t1 = (-half[:, None, :] - lo) / safe
        t2 = (half[:, None, :] - lo) / safe
        tmin, axis = torch.minimum(t1, t2).max(-1)                     # (B,P)
        tmax = torch.maximum(t1, t2).min(-1).values
        hit = (tmax > tmin.clamp(min=1e-4)) & (tmax > 1e-4)
        t = torch.where(hit, tmin.clamp(min=1e-4), torch.full_like(tmin, float("inf")))

        idx = axis.clamp(min=0)[..., None]
        sgn = -torch.sign(torch.gather(dl, -1, idx).squeeze(-1))       # (B,P)
        n_loc = torch.zeros(B, P, 3, device=o.device)
        n_loc.scatter_(-1, idx, sgn[..., None])
        return t, n_loc, c, s

    def _boxes(self, s):
        """Scene boxes as ``(centre, half, yaw, material)``. See module docstring."""
        sim, B, dev = self.sim, s.shape[0], s.device
        bx, by, px, py, th = s[:, 0], s[:, 1], s[:, 4], s[:, 5], s[:, 8]
        z = torch.zeros(B, device=dev)
        ones = torch.ones(B, device=dev)
        out = [
            # peg
            (torch.stack([bx, by, z + PEG_H / 2], 1),
             torch.stack([ones * sim.R_b, ones * sim.R_b, ones * (PEG_H / 2)], 1),
             th, MAT_PEG),
            # two fingertips
            (torch.stack([px, py + sim.finger_dy, z + FIN_H / 2], 1),
             torch.stack([ones * sim.r_p, ones * sim.r_p, ones * (FIN_H / 2)], 1),
             z, MAT_GRIP),
            (torch.stack([px, py - sim.finger_dy, z + FIN_H / 2], 1),
             torch.stack([ones * sim.r_p, ones * sim.r_p, ones * (FIN_H / 2)], 1),
             z, MAT_GRIP),
            # gripper body above the fingers
            (torch.stack([px, py, z + FIN_H + WRIST_H / 2], 1),
             torch.tensor([0.024, sim.finger_dy + 0.012, WRIST_H / 2], device=dev)
             .expand(B, 3).contiguous(),
             z, MAT_GRIP),
            # end-stop wall
            (torch.stack([ones * sim.x_w, z + WALL_HALF[2], z + WALL_HALF[2]], 1),
             torch.tensor(WALL_HALF, device=dev).expand(B, 3).contiguous(),
             z, MAT_WALL),
        ]
        return out

    # ------------------------------------------------------------- rendering
    @torch.no_grad()
    def render(self, s, res=None, ss=None, return_mat=False, rows=None):
        """state (B,10) -> (B,3,res,res) float in [0,1], gamma-encoded.

        With ``return_mat`` also return the (B,res,res) material map, taking the
        top-left subpixel of each supersample block. The point is that tests
        then inspect *this* renderer's own classification instead of
        re-implementing the intersection pass -- a duplicated copy of that
        geometry is exactly the kind of thing that silently diverges and makes a
        renderer test pass while the render is wrong.

        ``rows`` indexes the *appearance* and must be the indices of ``s`` within
        the full batch, so the Controller can render only the rows about to call
        the policy. Slicing the appearance alongside the states is what keeps row
        i's pixels a function of row i alone; without it a subset render would
        either mismatch the batch or, worse, be silently wrong.
        """
        sim = self.sim
        res = int(res or self.res)
        ss = int(ss if ss is not None else self.ss)
        R, dev, B = res * ss, s.device, s.shape[0]
        ap = self.appearance if self.appearance is not None else self._flat_appearance(
            self.appearance_rows if self.appearance is not None else B)
        if rows is not None:
            if rows.numel() != B:
                raise ValueError(
                    f"render(rows=...) got {B} states for {rows.numel()} indices")
            ap = {k: v[rows] for k, v in ap.items()}
        if ap["L"].shape[0] != B:
            # Appearance is drawn per env in reset_appearance and belongs to the
            # episode. Rendering a different batch size would either broadcast
            # env 0's look across everything or silently mix looks, so refuse.
            raise ValueError(
                f"camera appearance is built for batch {ap['L'].shape[0]} but got "
                f"state batch {B}; call PushCamera.reset_appearance({B}) first "
                "(Env.reset does this for you).")
        d, vdir = self._rays(R)
        o = self.pos

        cols = torch.stack([ap["col_table"], ap["col_peg"], ap["col_grip"],
                            ap["col_wall"], ap["col_mark"], ap["col_bg"]], 1)  # (B,6,3)
        ks = torch.tensor([_TABLE[3], _PEG[3], _GRIP[3], _WALL[3], _MARK[3], 0.0],
                          device=dev)
        sh = torch.tensor([_TABLE[4], _PEG[4], _GRIP[4], _WALL[4], _MARK[4], 1.0],
                          device=dev)
        L = ap["L"][:, None, :]                                    # (B,1,3)
        amb0, diff = ap["amb"][:, None], ap["diff"][:, None]      # (B,1)
        tint, spec_amp = ap["tint"][:, None, :], ap["spec"][:, None]

        boxes = self._boxes(s)
        img = torch.empty(B, 3, res, res, device=dev)

        # Iterate over *rows* of the supersampled image, not over a flat pixel
        # range: a block has to be row-aligned for the reshape below, and `block`
        # is not generally a multiple of R (at 224x224 ss=2, R=448 and the default
        # block of 65536 is 146.28 rows). Slicing rows keeps the invariant exact
        # for any R.
        # rpb is a multiple of ss so that the supersample groups (ss x ss) never
        # straddle a block boundary.
        rpb = max(ss, (self.block // R // ss) * ss)
        mats = torch.empty(B, res, res, dtype=torch.long, device=dev) if return_mat else None
        # A `while` rather than `range(0, R, rpb)`: when rpb does not divide R the
        # last range block stops *short* of R (rpb=146, R=448 -> 0,146,292 and
        # 292+146=438 < 448), which leaves the bottom rows of `img` as whatever
        # torch.empty put there. Advancing row0 to the clamped row1 covers them.
        row0 = 0
        while row0 < R:
            row1 = min(row0 + rpb, R)
            p0, p1, P = row0 * R, row1 * R, (row1 - row0) * R
            dp, vp = d[p0:p1], vdir[p0:p1]

            depth = torch.full((B, P), float("inf"), device=dev)
            mat = torch.full((B, P), MAT_BG, dtype=torch.long, device=dev)
            nrm = torch.zeros(B, P, 3, device=dev)
            hitp = torch.zeros(B, P, 3, device=dev)

            # ---- table plane z = 0
            dz = dp[:, 2].clamp(max=-1e-6)                          # (P,)
            t = (-o[2] / dz)[None].expand(B, P)
            hp = o.view(1, 1, 3) + t[..., None] * dp[None]
            on_tab = self._in_table(hp) & (t < float("inf"))
            floor = on_tab
            depth = torch.where(floor, t, depth)
            mat = torch.where(floor, torch.full_like(mat, MAT_FLOOR), mat)
            nrm = torch.where(floor[..., None],
                              torch.tensor([0.0, 0.0, 1.0], device=dev), nrm)
            hitp = torch.where(floor[..., None], hp, hitp)

            # ---- oriented boxes, nearest hit wins
            for centre, half, yaw, m in boxes:
                tb, n_loc, c, sn = self._slab(o, dp, centre, half, yaw)
                hb = o.view(1, 1, 3) + tb[..., None] * dp[None]
                # cull a hit that lands outside the workspace (see TABLE_X/Y):
                # a box out there is not part of the cell, and drawing it would
                # put an object in mid-air over empty background
                closer = (tb < depth) & self._in_table(torch.nan_to_num(hb, nan=1e9))
                if not bool(closer.any()):
                    continue
                nw = torch.stack([c * n_loc[..., 0] - sn * n_loc[..., 1],
                                  sn * n_loc[..., 0] + c * n_loc[..., 1],
                                  n_loc[..., 2]], -1)
                depth = torch.where(closer, tb, depth)
                mat = torch.where(closer, torch.full_like(mat, m), mat)
                nrm = torch.where(closer[..., None], nw, nrm)
                hitp = torch.where(closer[..., None], hb, hitp)

            # ---- printed target marker, drawn onto the table
            q = torch.maximum((hitp[..., 0] - sim.target_x).abs(),
                              (hitp[..., 1] - s[:, 6, None]).abs())
            inside = (q < MARK_HALF) & floor
            rim = (q >= MARK_RIM) & (q < MARK_HALF) & floor
            mat = torch.where(inside, torch.full_like(mat, MAT_MARK), mat)
            amb = amb0 * torch.where(rim, torch.full((B, 1), 0.55, device=dev),
                                     torch.ones((B, 1), device=dev))

            # ---- contact darkening under the peg and the pusher.
            # A rendered cue, not simulated light; see the module docstring.
            sh_fac = torch.ones(B, P, device=dev)
            if bool(floor.any()):
                Lxy = L[..., :2]
                Lxy = Lxy / Lxy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                for cx, cy, hw, yaw in (
                        (s[:, 0], s[:, 1], torch.full_like(s[:, 0], sim.R_b * 1.25), s[:, 8]),
                        (s[:, 4], s[:, 5],
                         torch.full_like(s[:, 0], sim.finger_dy + sim.r_p),
                         torch.zeros_like(s[:, 0]))):
                    c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
                    ddx = (hitp[..., 0] - cx[:, None]) - SHADOW_OFF * Lxy[..., 0]
                    ddy = (hitp[..., 1] - cy[:, None]) - SHADOW_OFF * Lxy[..., 1]
                    lx = (c * ddx + sn * ddy).abs() - hw[:, None]
                    ly = (-sn * ddx + c * ddy).abs() - hw[:, None]
                    qq = torch.maximum(lx, ly).clamp(min=0.0)
                    sh_fac = sh_fac * (1.0 - SHADOW_K
                                       * torch.exp(-(qq ** 2) / (2 * SHADOW_SIG ** 2)))
            sh_fac = torch.where(floor, sh_fac, torch.ones_like(sh_fac))

            # ---- shade
            ndl = (nrm * L).sum(-1).clamp(min=0.0)                   # (B,P)
            half_v = F.normalize(L + vp[None], dim=-1, eps=1e-6)   # Blinn halfway
            ndh = (nrm * half_v).sum(-1).clamp(min=0.0)
            base = torch.gather(cols, 1, mat[..., None].expand(B, P, 3))
            rgb = base * ((amb + diff * ndl)[..., None] * tint)
            rgb = rgb + (spec_amp * ks[mat] * ndh ** sh[mat])[..., None] * base
            rgb = rgb * sh_fac[..., None]
            rgb = torch.where((mat == MAT_BG)[..., None], ap["col_bg"][:, None, :], rgb)

            blk = rgb.clamp(0.0, 1.0) ** (1.0 / 2.2)                 # sRGB-ish
            blk = blk.reshape(B, row1 - row0, R, 3)                  # (B,rows,R,3)
            if ss > 1:
                blk = blk.reshape(B, (row1 - row0) // ss, ss, res, ss, 3).mean((2, 4))
            img[:, :, row0 // ss:row1 // ss, :] = blk.permute(0, 3, 1, 2)   # -> (B,3,H,W)
            if return_mat:
                mats[:, row0 // ss:row1 // ss, :] = mat.reshape(
                    B, (row1 - row0) // ss, ss, res, ss)[:, :, 0, :, 0]
            row0 = row1
        return (img, mats) if return_mat else img
