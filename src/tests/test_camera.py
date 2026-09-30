"""Sanity + visual check for the torch GT's new RGB camera.

Writes a PNG contact sheet so the frames can be *looked at*, which is the only
way to catch a broken projection, a wrong normal, or an occluded wall. The
numeric assertions cover what a picture cannot: purity (the renderer must not
mutate the state), shape/range, sensitivity, and -- through the renderer's own
``return_mat`` map -- that the constraint-relevant geometry is actually visible.

Deliberately no private-geometry re-implementation here: an earlier draft of
this file duplicated the intersection pass, which then reported "marker missing"
while the render was fine. Ask the renderer.

Run: cd src && python -u tests/test_camera.py
"""
import os
import sys
import time

import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)  # noqa: E402

from sims.base import make_env, make_sim                              # noqa: E402
from sims.camera import (MAT_BG, MAT_FLOOR, MAT_GRIP, MAT_MARK,  # noqa: E402
                         MAT_PEG, MAT_WALL)

PR = 128          # probe resolution for the material-map checks


def centroid(mask, res, axis):
    """Mean row (axis=0) or column (axis=1) of a (B,res,res) bool mask."""
    flat = mask.reshape(mask.shape[0], -1)
    idx = torch.arange(res * res, device=mask.device, dtype=torch.float32)
    g = ((idx // res) if axis == 0 else (idx % res))[None, :]
    n = flat.sum(1).clamp(min=1)
    return (flat * g).sum(1) / n


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = make_sim("push", dev)
    env = make_env("push", "torch", 8, dev, camera=True, cam_res=224, cam_ss=2)
    obs = env.reset(sim.nominal_params(), seed=3)

    # ---- purity: a renderer must not touch the state or the params
    s0 = env.get_state().clone()
    p0 = {k: v.clone() for k, v in env.params.items()}
    img = env.render_rgb()
    assert torch.equal(s0, env.get_state()), "renderer mutated the state!"
    assert all(torch.equal(p0[k], env.params[k]) for k in p0), "params mutated!"
    print(f"purity      ok    state {tuple(s0.shape)} unchanged, params unchanged")

    # ---- shape, range, dtype, variation across envs
    assert img.shape == (8, 3, 224, 224), img.shape
    assert img.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    var = img.flatten(1).std(1)
    print(f"shape       ok    {tuple(img.shape)}  range [{img.min():.3f}, {img.max():.3f}]"
          f"  per-env pixel std {var.min():.3f}-{var.max():.3f}")
    assert var.min() > 0.02, "a frame is nearly constant -- projection is broken"
    d = (img[0] - img[1]).abs().mean()
    print(f"dr          ok    mean |frame0-frame1| = {d:.4f} (appearance is randomised)")
    assert d > 0.01, "domain randomisation is not doing anything"

    # ---- coverage: is the constraint-relevant geometry actually on screen?
    _, mat = env.render_rgb(res=PR, ss=1, return_mat=True)
    st = env.get_state()
    fr = {m: float((mat == m).float().mean()) for m in range(6)}
    print(f"coverage    floor {fr[MAT_FLOOR]:.3f}  peg {fr[MAT_PEG]:.3f}  "
          f"gripper {fr[2]:.3f}  wall {fr[MAT_WALL]:.3f}  "
          f"marker {fr[MAT_MARK]:.3f}  bg {fr[MAT_BG]:.3f}")
    assert fr[MAT_PEG] > 0.002, "the peg is not visible -- check camera placement"
    assert fr[MAT_FLOOR] > 0.20, "the table is not visible"
    assert fr[MAT_WALL] > 0.002, ("the end-stop wall is not visible; a VLA cannot "
                                  "avoid a constraint it cannot see")
    assert fr[MAT_MARK] > 0.002, "the target seat is not visible"
    for m, nm in ((MAT_PEG, "peg"), (MAT_WALL, "wall"), (MAT_MARK, "marker")):
        cnt = (mat == m).sum((1, 2))
        assert bool((cnt > 0).all()), f"{nm} missing in some frames: {cnt.tolist()}"

    # ---- projection: the peg's pixel centroid must follow the peg in the world
    def peg_c(state):
        _, m = env.render_rgb(res=PR, ss=1, return_mat=True) if state is None else \
            env.cam.render(state, res=PR, ss=1, return_mat=True)
        px = m == MAT_PEG
        return (centroid(px, PR, 0), centroid(px, PR, 1)), px.sum((1, 2))

    (r0, c0), n0 = peg_c(None)
    st2 = st.clone()
    st2[:, 0] += 0.05
    (r1, c1), n1 = peg_c(st2)
    dr, dc = (r1 - r0).abs().mean().item(), (c1 - c0).abs().mean().item()
    print(f"projection  5 cm world +x -> centroid d(row) {dr:.2f} px, "
          f"d(col) {dc:.2f} px   ({n0[0].item():.0f} -> {n1[0].item():.0f} peg px)")
    assert dc > 1.0, "the peg's pixel centroid did not follow the peg"
    assert dc > 3 * dr, "x maps to the image's row, not its column"

    # ---- legibility of the constraint: marker left of the wall, gap in between
    wall, mark = (mat == MAT_WALL), (mat == MAT_MARK)
    gap = (centroid(wall, PR, 1) - centroid(mark, PR, 1)).mean().item()
    print(f"legibility  marker-to-wall column gap {gap:.1f} px")
    assert gap > 0, "the wall renders on the wrong side of the target"

    # ---- geometry teleported outside the workspace must be culled, not drawn
    # floating in mid-air. The wall is at a fixed x_w and is deliberately *not*
    # moved, so only the peg and the gripper can be checked here.
    far = st.clone()
    far[:, 0] = far[:, 4] = -5.0
    far[:, 1] = far[:, 5] = -5.0
    imf, mf = env.cam.render(far, res=64, ss=1, return_mat=True)
    for m, nm in ((MAT_PEG, "peg"), (MAT_GRIP, "gripper")):
        assert not bool((mf == m).any()), f"{nm} drawn outside the workspace"
    assert bool((mf == MAT_WALL).any()), "the wall vanished with the rest"
    print("culling     ok    peg+gripper culled outside the workspace, wall kept")
    # background pixels must carry exactly the gamma-encoded background colour.
    # Locate them from the renderer's own material map rather than guessing a
    # coordinate: the table fills ~45% of the frame, so (32,32) is table, not bg.
    bgm = mf == MAT_BG
    assert bool(bgm.any()), "no background pixel to check"
    bg = env.cam.appearance["col_bg"]
    exp = (bg[0].clamp(0, 1) ** (1 / 2.2))
    got = imf[0].permute(1, 2, 0)[bgm[0]]
    err = (got - exp).abs().max().item()
    print(f"bg colour   ok    {bgm[0].sum().item()} bg px, max |pixel - expected| "
          f"= {err:.4f} (got {['%.3f' % v for v in got[0].tolist()]}, "
          f"expected {['%.3f' % v for v in exp.tolist()]})")
    assert err < 1e-5, "background pixels are not the background colour"

    # ---- determinism: same seed -> same pixels
    env.reset(sim.nominal_params(), seed=3)
    again = env.render_rgb()
    assert torch.equal(img, again), "the camera is not reproducible from the seed"
    env.reset(sim.nominal_params(), seed=4)
    diff = env.render_rgb()
    print(f"determinism ok    same seed bit-identical, seed 4 differs by "
          f"{(again - diff).abs().mean():.4f}")

    # ---- cost at the resolution a VLA wants, and the batch-mismatch guard
    try:
        env.cam.render(env.get_state().repeat(8, 1), res=64, ss=1)
        raise AssertionError("a batch-size mismatch was accepted silently")
    except ValueError as e:
        print(f"batch guard ok    {str(e).split(';')[0]}")
    big = make_env("push", "torch", 64, dev, camera=True, cam_res=224, cam_ss=2)
    big.reset(sim.nominal_params(), seed=3)
    for r, ss in ((224, 2), (224, 1), (32, 1)):
        for _ in range(2):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = big.render_rgb(res=r, ss=ss)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
        print(f"cost        B={big.n:3d} res={r:3d} ss={ss}  {dt:7.1f} ms  "
              f"-> {tuple(out.shape)}")
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
    print(f"peak GPU    {peak:.2f} GiB")

    _sheet(img, os.environ.get("CAMERA_SHEET", "camera_frames.png"))
    print("\nwrote camera_frames.png  (2x4 contact sheet)")


def _sheet(img, path):
    """Write a nrow-column contact sheet with PIL (no torchvision in this env)."""
    from PIL import Image
    B, _, h, w = img.shape
    nrow, pad = 4, 2
    nrow = min(nrow, B)
    ncol = (B + nrow - 1) // nrow
    H, W = ncol * (h + pad) + pad, nrow * (w + pad) + pad
    canvas = Image.new("RGB", (W, H), (16, 16, 20))
    a = (img.clamp(0, 1) * 255).round().byte().cpu().numpy()
    for i in range(B):
        r, c = divmod(i, nrow)
        im = Image.fromarray(a[i].transpose(1, 2, 0))
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    out = os.path.expanduser(path)
    canvas.save(out)
    # also a zoomed crop of one frame, so the constraint detail is inspectable
    Image.fromarray(a[0].transpose(1, 2, 0)).resize((w * 2, h * 2), Image.NEAREST)\
        .save(out.replace(".png", "_zoom.png"))
    return out


if __name__ == "__main__":
    main()
