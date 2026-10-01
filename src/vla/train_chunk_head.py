"""Train the 2-D planar chunk head on cached OpenVLA features.

Input: ~/scratch/openvla_push_features.h5 from extract_features.py.
Output: ~/scratch/openvla_chunk_head.pt, holding the head weights plus the
feature normalisation and the name of the feature it was trained on. The policy
must compute exactly that feature at run time.

A falling training loss says almost nothing here. The question is whether the
frozen backbone's features carry the task state well enough to predict the
expert's next H actions on *unseen episodes*. So the report is held-out R^2
per chunk step, split by episode (frames from one episode are strongly
correlated, so a frame-level split would leak), against two baselines:
  mean   predict the training-set mean chunk (R^2 = 0 by construction)
  ridge  closed-form linear readout; if the MLP cannot beat it, keep it simple

    cd src && python -u vla/train_chunk_head.py --feature llm
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)


class ChunkHead(nn.Module):
    """Normalised feature -> (H, 2) world-frame pusher velocity in m/s."""

    def __init__(self, d_in=4096, H=5, hidden=1024, dropout=0.1):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, H * 2))

    def forward(self, x):
        return self.net(x).view(-1, self.H, 2)


def r2(pred, y):
    """R^2 per (step, axis), averaged over axes -> one number per chunk step."""
    ss_res = ((pred - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return (1 - ss_res / np.maximum(ss_tot, 1e-12)).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="~/scratch/openvla_push_features.h5")
    ap.add_argument("--feature", choices=["llm", "vis"], default="llm")
    ap.add_argument("--out", default="~/scratch/openvla_chunk_head.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with h5py.File(os.path.expanduser(args.features), "r") as f:
        X = f[args.feature][:].astype(np.float32)
        Y = f["target"][:]
        ep = f["episode"][:]
        H = int(f.attrs["H"])
    rng = np.random.default_rng(args.seed)
    eps = np.unique(ep)
    val_eps = rng.choice(eps, max(1, int(len(eps) * args.val_frac)), replace=False)
    va = np.isin(ep, val_eps)
    tr = ~va
    print(f"feature={args.feature}  train {tr.sum()} frames / {len(eps) - len(val_eps)} eps"
          f"  val {va.sum()} frames / {len(val_eps)} eps", flush=True)

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd

    # mean baseline and ridge readout
    print(f"  mean   val R2 per step: {np.round(r2(np.broadcast_to(Y[tr].mean(0), Y[va].shape), Y[va]), 3)}")
    A = np.c_[Xn[tr], np.ones(tr.sum())]
    reg = np.eye(A.shape[1]) * 10.0
    reg[-1, -1] = 0.0                       # never penalise the intercept
    W = np.linalg.solve(A.T @ A + reg, A.T @ Y[tr].reshape(tr.sum(), -1))
    ridge = (np.c_[Xn[va], np.ones(va.sum())] @ W).reshape(Y[va].shape)
    print(f"  ridge  val R2 per step: {np.round(r2(ridge, Y[va]), 3)}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = ChunkHead(X.shape[1], H).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xt, Yt = torch.tensor(Xn[tr], device=dev), torch.tensor(Y[tr], device=dev)
    Xv = torch.tensor(Xn[va], device=dev)
    best, best_state = -np.inf, None
    for e in range(args.epochs):
        head.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = ((head(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xv).cpu().numpy()
        score = r2(pv, Y[va])
        if score.mean() > best:
            best, best_state = score.mean(), {k: v.clone() for k, v in head.state_dict().items()}
        if e % 10 == 0 or e == args.epochs - 1:
            print(f"  epoch {e:3d}  train mse {loss.item():.5f}  val R2 per step "
                  f"{np.round(score, 3)}", flush=True)

    head.load_state_dict(best_state)
    print(f"best mean val R2 = {best:.3f}  (ridge {r2(ridge, Y[va]).mean():.3f})")
    torch.save({"head_state": head.state_dict(), "H": H, "feature": args.feature,
                "feat_mu": torch.tensor(mu), "feat_sd": torch.tensor(sd),
                "val_r2": float(best)}, os.path.expanduser(args.out))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
