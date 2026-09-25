"""
Train the VLA proxy (BC policy) and the two verifier predictors.

  bc            BCPolicy on nominal expert demos (obs -> action chunk)
  orbisim       OrbiSimDynamics ensemble, multi-step (H) rollout loss on
                object-centric obs + per-step risk
  vision        VisionWM (action-conditioned) risk world model on rendered frames
  vision_noact  VisionWM observation-only ablation

Usage: python experiments/train.py --task push --what all
"""
import os
import time

import torch
import torch.nn.functional as F

from _common import CHUNK_H, base_parser, setup
from models.nets import BCPolicy, OrbiSimDynamics, VisionWM, save_model

RISK_CLIP = 3.0


def obs_all(sim, S, dev, bs=4096):
    N, T1, D = S.shape
    flat = S.reshape(-1, D)
    out = [sim.obs(flat[i:i + bs].to(dev)).cpu() for i in range(0, flat.shape[0], bs)]
    return torch.cat(out).view(N, T1, -1)


def split(N, frac=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    nv = max(1, int(N * frac))
    return perm[nv:], perm[:nv]


# ---------------------------------------------------------------------------
def train_bc(sim, od, dev, iters, H):
    d = torch.load(os.path.join(od, "bc_data.pt"))
    S, A = d["states"], d["actions"]
    O = obs_all(sim, S, dev)[:, :-1]                          # (N,T,O)
    N, T, _ = O.shape
    Apad = torch.cat([A, torch.zeros(N, H, A.shape[-1])], 1)
    idx = torch.arange(T)[:, None] + torch.arange(H)[None]    # (T,H)
    Y = Apad[:, idx]                                          # (N,T,H,A)
    X, Y = O.reshape(-1, O.shape[-1]).to(dev), Y.reshape(-1, H, A.shape[-1]).to(dev)
    pol = BCPolicy(sim.obs_dim, sim.act_dim, H, max_vel=sim.max_vel).to(dev)
    pol.obs_n.fit(X)
    opt = torch.optim.AdamW(pol.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        b = torch.randint(0, X.shape[0], (512,), device=dev)
        loss = F.mse_loss(pol(X[b]), Y[b]) / sim.max_vel ** 2
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0 or it == iters - 1:
            print(f"[bc] it {it} loss {loss.item():.5f}")
    save_model(pol, os.path.join(od, "bc_policy.pt"), obs_dim=sim.obs_dim,
               act_dim=sim.act_dim, H=H, max_vel=sim.max_vel)


# ---------------------------------------------------------------------------
class DynWindows:
    """(episode, t) windows for predictor training."""

    def __init__(self, sim, od, dev, H):
        d = torch.load(os.path.join(od, "dyn_data.pt"))
        self.S = d["states"].to(dev)
        self.A = d["actions"].to(dev)
        self.R = d["risk"].clamp(max=RISK_CLIP).to(dev)
        self.O = obs_all(sim, d["states"], dev).to(dev)
        self.N, self.T = self.A.shape[:2]
        self.H, self.sim, self.dev = H, sim, dev
        self.tr, self.va = split(self.N)
        # "hot" windows (a violation-level risk within the chunk) are rare
        # (impacts last a single step) -> oversample them for both predictors.
        nt = self.T - self.H + 1
        wmax = torch.stack([self.R[:, t:t + self.H].amax((1, 2)) for t in range(nt)], 1)
        is_tr = torch.zeros(self.N, dtype=torch.bool, device=dev)
        is_tr[self.tr.to(dev)] = True
        self.hot = ((wmax > 0.8) & is_tr[:, None]).nonzero()
        print(f"[data] {self.N} eps, hot-window fraction "
              f"{(wmax > 0.8).float().mean():.3f} ({len(self.hot)} train windows)")

    def batch(self, B, val=False, img=False, hot_frac=0.3):
        pool = (self.va if val else self.tr).to(self.dev)
        e = pool[torch.randint(0, len(pool), (B,), device=self.dev)]
        t = torch.randint(0, self.T - self.H + 1, (B,), device=self.dev)
        nh = int(B * hot_frac) if (not val and len(self.hot)) else 0
        if nh:
            h = self.hot[torch.randint(0, len(self.hot), (nh,), device=self.dev)]
            e, t = e.clone(), t.clone()
            e[:nh], t[:nh] = h[:, 0], h[:, 1]
        tp = (t - 1).clamp(min=0)
        hs = t[:, None] + torch.arange(self.H, device=self.dev)[None]
        b = {
            "obs": self.O[e, t], "obs_prev": self.O[e, tp],
            "a_prev": self.A[e, tp] * (t > 0).float()[:, None],
            "actions": self.A[e[:, None], hs],
            "obs_next": self.O[e[:, None], hs + 1],
            "risk": self.R[e[:, None], hs],
        }
        if img:
            b["img"] = self.sim.render(self.S[e, t])
            b["img_prev"] = self.sim.render(self.S[e, tp])
        return b


def train_orbisim(sim, od, dev, iters, H, data=None):
    data = data or DynWindows(sim, od, dev, H)
    m = OrbiSimDynamics(sim.obs_dim, sim.act_dim, max_vel=sim.max_vel).to(dev)
    m.obs_n.fit(data.O.reshape(-1, sim.obs_dim))
    m.dobs_n.fit((data.O[:, 1:] - data.O[:, :-1]).reshape(-1, sim.obs_dim))
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        b = data.batch(512)
        o, r = m.rollout(b["obs"], b["obs_prev"], b["a_prev"], b["actions"])
        lo = (((o - b["obs_next"][None]) / m.dobs_n.std) ** 2).mean()
        lr_ = F.mse_loss(r, b["risk"][None].expand_as(r))
        loss = 0.05 * lo + lr_
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step(); sched.step()
        if it % 1000 == 0 or it == iters - 1:
            with torch.no_grad():
                v = data.batch(1024, val=True)
                ov, rv = m.rollout(v["obs"], v["obs_prev"], v["a_prev"], v["actions"])
                vo = (((ov.mean(0) - v["obs_next"]) / m.dobs_n.std) ** 2).mean()
                vr = (rv.mean(0) - v["risk"]).abs().mean()
            print(f"[orbisim] it {it} loss {loss.item():.4f} | val obs {vo:.4f} risk-MAE {vr:.4f}")
    save_model(m, os.path.join(od, "orbisim.pt"), obs_dim=sim.obs_dim,
               act_dim=sim.act_dim, max_vel=sim.max_vel)
    return data


def train_vision(sim, od, dev, iters, H, action_conditioned=True, data=None):
    name = "vision" if action_conditioned else "vision_noact"
    data = data or DynWindows(sim, od, dev, H)
    m = VisionWM(sim.act_dim, action_conditioned=action_conditioned,
                 max_vel=sim.max_vel).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        b = data.batch(256, img=True)
        r = m(b["img"], b["img_prev"], b["actions"])
        loss = F.mse_loss(r, b["risk"][None].expand_as(r))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step(); sched.step()
        if it % 1000 == 0 or it == iters - 1:
            with torch.no_grad():
                v = data.batch(512, val=True, img=True)
                vr = (m(v["img"], v["img_prev"], v["actions"]).mean(0) - v["risk"]).abs().mean()
            print(f"[{name}] it {it} loss {loss.item():.4f} | val risk-MAE {vr:.4f}")
    save_model(m, os.path.join(od, f"{name}.pt"), act_dim=sim.act_dim,
               action_conditioned=action_conditioned, max_vel=sim.max_vel)
    return data


def main():
    p = base_parser(__doc__)
    p.add_argument("--what", default="all",
                   choices=["all", "bc", "orbisim", "vision", "vision_noact"])
    p.add_argument("--iters_bc", type=int, default=10000)
    p.add_argument("--iters_dyn", type=int, default=15000)
    p.add_argument("--iters_vis", type=int, default=8000)
    args = p.parse_args()
    dev, sim, od = setup(args)
    if args.quick:
        args.iters_bc = args.iters_dyn = args.iters_vis = 50
    H = CHUNK_H
    t0 = time.time()
    data = None
    if args.what in ("all", "bc"):
        train_bc(sim, od, dev, args.iters_bc, H)
    if args.what in ("all", "orbisim"):
        data = train_orbisim(sim, od, dev, args.iters_dyn, H, data)
    if args.what in ("all", "vision"):
        data = train_vision(sim, od, dev, args.iters_vis, H, True, data)
    if args.what in ("all", "vision_noact"):
        data = train_vision(sim, od, dev, args.iters_vis, H, False, data)
    print(f"[train] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
