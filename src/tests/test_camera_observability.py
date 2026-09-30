"""Is the camera *sufficient* for the task? An observability check.

A picture is not evidence: it can look plausible while being useless. The
decisive question for this camera is measurable -- does the rendered image
carry the variables the thesis is about?

    bx, by, th, ty     where the box is, and where the seat is
    gap   = x_w - bx   the clearance to the end-stop, i.e. how much room is
                       left before the contact-force constraint can bite

A ridge regression from pixels to each target answers this without a VLA. High
held-out R^2 means the variable is linearly decodable, so a policy has access
to it; R^2 near zero means the camera discards the signal no matter how good the
policy is.

Three things this had to get right, each of which produced a convincing wrong
number first:

  * Fit on train, score on held-out. Refitting on the test split and reporting
    in-sample R^2 flatters every target -- it "proved" friction was visible from
    the camera at R^2 0.97.
  * Do not penalise the intercept. It has squared norm n, so lambda shrinks it by
    n/(n+lambda); for a target whose mean is 30 sigma that is a systematic bias
    of 2 sigma and R^2 goes negative while the fit looks healthy.
  * Seed the parameter and appearance generators differently. The appearance is
    drawn by the generator Env.reset makes from the episode seed, so reusing the
    integer makes the appearance a deterministic function of the friction draw.

The opposite check runs too: the privileged parameters. Friction and mass are
not in the state tensor, so they cannot appear in the image. That is correct,
and it is exactly why a physics-derived risk signal can add something a visual
one cannot. Making the claim falsifiable is the point.

Run: cd src && CUDA_VISIBLE_DEVICES=0 python -u tests/test_camera_observability.py
"""
import os
import sys
import time

import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)  # noqa: E402

from sims.base import make_env, make_sim  # noqa: E402

RES, NTRAIN, NTEST, NPCA, LAM = 64, 2048, 1024, 256, 3.0
TARGETS = ("bx", "by", "th", "ty", "gap", "mu", "mass")
NEED, LEAK = 0.60, 0.20
READING = {
    "bx": "where the box is",
    "by": "lateral position",
    "th": "peg yaw (the box is symmetric under 90 deg: use sin/cos)",
    "ty": "target row (the seat is printed on the table)",
    "gap": "MUST be decodable: the constraint the thesis is about",
    "mu": "MUST be ~0: friction is privileged and cannot be seen",
    "mass": "MUST be ~0: mass is privileged and cannot be seen",
}


def build(n, seed, res=RES, randomize=True):
    """Render n states drawn the way episodes draw them, plus their targets.

    The parameter generator is seeded ``seed + 10_000``, NOT ``seed``: see the
    module docstring.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = make_sim("push", dev)
    env = make_env("push", "torch", n, dev, camera=True, cam_res=res, cam_ss=1)
    env.cam.randomize = randomize
    gp = torch.Generator().manual_seed(seed + 10_000)
    mults = {"friction": 0.15 + 1.7 * torch.rand(n, generator=gp),
             "mass": 0.5 + 1.0 * torch.rand(n, generator=gp)}
    env.reset(sim.make_params(n, mults), seed=seed)
    st = env.get_state()
    img = env.render_rgb()
    t = {"bx": st[:, 0], "by": st[:, 1], "th": st[:, 8], "ty": st[:, 6],
         "gap": sim.x_w - st[:, 0],
         "mu": env.params["friction"], "mass": env.params["mass"]}
    return img, t, env, sim


def scale_cols_per_m(res, fov_deg=48.0, dist=0.58):
    """Analytic image columns per world metre at the look-at distance.

    Perspective, so the scale is not constant across the image; this is the value
    at the centre. An earlier draft measured it by displacing the peg 1 cm and
    reading the pixel centroid, which gave 112 +/- 14 col/m against a true 124 --
    a 4 px displacement is below the centroid's own noise. Worth stating: do not
    calibrate a camera with a probe smaller than its own resolution.
    """
    import math
    return res / (2.0 * math.tan(math.radians(fov_deg) / 2.0) * dist)


def fit_pca(X, k):
    """Centre + whiten-free PCA, components fitted on X (train) only."""
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    # economy SVD on (n, d) with d >> n: use the Gram matrix, it is n x n
    G = Xc @ Xc.T
    evals, evecs = torch.linalg.eigh(G.double())
    idx = torch.argsort(evals, descending=True)[:k]
    V = (Xc.T.double() @ (evecs[:, idx] / evals[idx].sqrt().clamp(min=1e-12))).float()
    return mu, V


def fit_ridge(X, Y, lam=LAM):
    """Ridge on already-reduced features. Intercept solved exactly, not penalised."""
    d = X.shape[1]
    ybar = Y.mean(0, keepdim=True)
    A = X.T @ X + lam * torch.eye(d, device=X.device)
    return ybar, torch.linalg.solve(A, X.T @ (Y - ybar))


def r2(Y, P):
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(0)
    return (1.0 - ss_res / ss_tot.clamp(min=1e-12)).cpu().tolist()


def evaluate(label, randomize, NTRAIN=NTRAIN, NTEST=NTEST, ncomp=NPCA):
    t0 = time.perf_counter()
    Itr, Ttr, _, _ = build(NTRAIN, seed=11, randomize=randomize)
    Ite, Tte, _, _ = build(NTEST, seed=12, randomize=randomize)
    ftr, fte = Itr.reshape(NTRAIN, -1), Ite.reshape(NTEST, -1)
    Ytr = torch.stack([Ttr[k] for k in TARGETS], 1)
    Yte = torch.stack([Tte[k] for k in TARGETS], 1)

    mu, V = fit_pca(ftr, ncomp)              # PCA fitted on TRAIN only
    Ptr, Pte = (ftr - mu) @ V, (fte - mu) @ V
    ybar, W = fit_ridge(Ptr, Ytr)            # ridge fitted on TRAIN only
    return (label, randomize, ftr.shape[1], r2(Ytr, Ptr @ W + ybar),
            r2(Yte, Pte @ W + ybar), time.perf_counter() - t0)


def main():
    print(f"analytic scale: {scale_cols_per_m(RES):.0f} image columns per metre at "
          f"res={RES} (centre of frame; perspective, so not constant)")
    rows = [evaluate("appearance randomised", True),
            evaluate("fixed appearance      ", False)]
    for label, _, d, r2tr, r2te, dt in rows:
        print(f"\n{label}   {d} px -> {NPCA} PCA components, fit on {NTRAIN}, "
              f"scored on {NTEST} unseen   [{dt:.1f}s]")
        print(f"  {'target':>8s}  {'train':>8s}  {'held out':>9s}   reading")
        for i, k in enumerate(TARGETS):
            print(f"  {k:>8s}  {r2tr[i]:8.4f}  {r2te[i]:9.4f}   {READING[k]}")

    print("\nverdict on the harder of the two (randomised appearance):")
    r2te = rows[0][4]
    ok = True
    for k in ("bx", "by", "ty", "gap"):
        i = TARGETS.index(k)
        good = r2te[i] >= NEED
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {k:>4s} held-out R^2 {r2te[i]:6.3f} "
              f"{'>= %.2f' % NEED if good else '< %.2f' % NEED}")
    for k in ("mu", "mass"):
        i = TARGETS.index(k)
        clean = r2te[i] <= LEAK
        ok &= clean
        print(f"  {'PASS' if clean else 'FAIL'}  {k:>4s} held-out R^2 {r2te[i]:6.3f} "
              f"{'<= %.2f: not visible' % LEAK if clean else '> %.2f: LEAKS' % LEAK}")
    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
