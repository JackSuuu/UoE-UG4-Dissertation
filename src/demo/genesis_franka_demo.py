"""
Genesis visual demo — Franka Panda arm doing Task A (push-insertion).

Why this exists: the experiment backends (sims/torch_push.py, sims/genesis_push.py)
use a free-floating point pusher so that rollouts are cheap and batched.
For a dissertation demo you want to *see a robot arm*. This script puts a Franka
Panda in the same scene geometry: the closed fingertips play the pusher, and the
policy's pusher-velocity actions are turned into end-effector targets through
Genesis inverse kinematics.

    policy (expert | BC stand-in | --policy openvla ...)  ->  pusher velocity
      -> [optional CheckVLA check / repair]  ->  EE target (x, y, z_fixed)
      -> Franka IK -> joint position control -> Genesis physics

STATUS: written against the genesis-world 0.2/0.3 API (Franka tutorial pattern:
MJCF 'xml/franka_emika_panda/panda.xml', link 'hand', inverse_kinematics,
control_dofs_position, add_camera + start_recording). NOT RUN YET — expect to
fix small API details on the server.

Usage (from src/):
    # interactive window (needs a display; on macOS Genesis wants the viewer on the main thread)
    python demo/genesis_franka_demo.py --viewer
    # headless server: record an mp4 (set PYOPENGL_PLATFORM=egl if rendering fails)
    python demo/genesis_franka_demo.py --record demo_push.mp4
    # OOD physics + learned policy + CheckVLA intervention (needs results/push_torch/ from run_all.sh)
    python demo/genesis_franka_demo.py --record ood.mp4 --policy bc --checkvla --friction 0.2 --mass 2.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sims.torch_push import PushSim  # noqa: E402

X0 = 0.35            # task-frame x=0  ->  world x=0.35 in front of the Franka base
FINGER_Z = 0.015     # fingertip height above the table (peg is 3 cm tall)
HAND_TO_TIP = 0.105  # 'hand' link origin -> closed fingertip distance (Panda)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # gripper pointing down (Genesis wxyz)


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", action="store_true", help="open the interactive Genesis viewer")
    ap.add_argument("--record", default=None, help="save an mp4 to this path")
    ap.add_argument("--policy", default="expert", choices=["expert", "bc", "openvla"])
    ap.add_argument("--checkvla", action="store_true", help="wrap the policy with the CheckVLA verifier")
    ap.add_argument("--orbisim_impl", default="standin", choices=["standin", "official"])
    ap.add_argument("--verifier", default="ref", choices=["ref", "official"])
    ap.add_argument("--friction", type=float, default=1.0, help="friction multiplier (OOD)")
    ap.add_argument("--mass", type=float, default=1.0, help="peg mass multiplier (OOD)")
    ap.add_argument("--steps", type=int, default=None, help="control steps (default sim.T)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    import genesis as gs
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = PushSim(dev)
    nom = sim.nominal_params()
    mu = float(nom["friction"]) * args.friction
    mass = float(nom["mass"]) * args.mass
    dt_sub = 0.01
    n_sub = int(round(sim.dt / dt_sub))
    T = args.steps or sim.T

    # ------------------------------------------------------------------ scene
    cam_pos, cam_look = (1.25, -0.95, 0.75), (X0 + 0.15, 0.0, 0.05)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt_sub, substeps=4),
        viewer_options=gs.options.ViewerOptions(camera_pos=cam_pos, camera_lookat=cam_look,
                                                camera_fov=40, max_FPS=60),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=mu))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    h = 0.03
    vol = np.pi * sim.R_b ** 2 * h
    g = torch.Generator().manual_seed(args.seed)
    s0 = sim.init_state(1, g)[0].cpu().numpy()          # [bx,by,vx,vy,px,py,ty,t] task frame
    peg = scene.add_entity(
        gs.morphs.Cylinder(radius=sim.R_b, height=h, pos=(X0 + s0[0], s0[1], h / 2)),
        material=gs.materials.Rigid(rho=mass / vol, friction=mu),
        surface=gs.surfaces.Default(color=(0.9, 0.5, 0.1)))
    scene.add_entity(
        gs.morphs.Box(size=(0.02, 0.4, 0.06), pos=(X0 + sim.x_w + 0.01, 0.0, 0.03), fixed=True),
        material=gs.materials.Rigid(friction=mu),
        surface=gs.surfaces.Default(color=(0.4, 0.4, 0.45)))
    cam = None
    if args.record:
        cam = scene.add_camera(res=(1280, 720), pos=cam_pos, lookat=cam_look, fov=40, GUI=False)
    scene.build()

    motors = np.arange(7)
    fingers = np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    hand = franka.get_link("hand")

    def ik(px, py):
        q = franka.inverse_kinematics(link=hand, pos=np.array([X0 + px, py, FINGER_Z + HAND_TO_TIP]),
                                      quat=DOWN_QUAT)
        return q

    # teleport arm to the start pose behind the peg, fingers closed
    q0 = ik(s0[4], s0[5])
    q0 = to_np(q0).copy()
    q0[-2:] = 0.0
    franka.set_dofs_position(q0)
    franka.control_dofs_position(q0[:-2], motors)
    franka.control_dofs_position(np.zeros(2), fingers)
    for _ in range(20):
        scene.step()

    # ------------------------------------------------------------------ policy
    policy, verifier = None, None
    if args.policy != "expert" or args.checkvla:
        sys.path.insert(0, os.path.join(ROOT, "experiments"))
        from common import out_dir, load_json
        from registry import build_policy, build_predictor, build_verifier
        ns = argparse.Namespace(policy=args.policy, orbisim_impl=args.orbisim_impl,
                                verifier=args.verifier, latency=1, commit=5,
                                vla_model="openvla/openvla-7b")
        od = out_dir("push", "torch")
        if args.policy != "expert":
            policy = build_policy(ns, od, dev, sim)
        if args.checkvla:
            tau = load_json(os.path.join(od, "taus.json"))["tau"]["orbisim"]
            verifier = build_verifier(ns, build_predictor(ns, od, dev, sim, "orbisim"), sim, tau)
    if args.policy == "openvla":
        raise SystemExit("OpenVLA in the demo needs cam frames passed to the policy; "
                         "wire cam.render(rgb=True) into policy(obs, img) here first.")

    def read_state(t, p_cmd):
        pp, pv = to_np(peg.get_pos()).reshape(-1), to_np(peg.get_vel()).reshape(-1)
        hp = to_np(hand.get_pos()).reshape(-1)
        s = np.array([pp[0] - X0, pp[1], pv[0], pv[1], hp[0] - X0, hp[1], s0[6], t], np.float32)
        return torch.as_tensor(s, device=dev)[None]

    # ------------------------------------------------------------------ loop
    if cam is not None:
        cam.start_recording()
    render_every = max(1, int(round(1.0 / (args.fps * dt_sub))))
    p_cmd = np.array([s0[4], s0[5]])
    obs_prev, a_prev = None, torch.zeros(1, 2, device=dev)
    plan, plan_left, plan_ptr = None, 0, 0
    fmax_ep, n_trig = 0.0, 0
    k = 0
    for t in range(T):
        s = read_state(t, p_cmd)
        obs = sim.obs(s)
        obs_prev = obs if obs_prev is None else obs_prev
        with torch.no_grad():
            chunk = (sim.expert(s)[:, None].repeat(1, 10, 1) if policy is None else policy(obs))
        if plan_left > 0:
            a = plan[:, plan_ptr]
            plan_ptr += 1
            plan_left -= 1
        else:
            a = chunk[:, 0]
            if verifier is not None:
                ctx = {"obs": obs, "obs_prev": obs_prev, "a_prev": a_prev, "state": s}
                trig, score = verifier.check(ctx, chunk)
                if bool(trig[0]):
                    n_trig += 1
                    plan = verifier.repair(ctx, chunk)
                    plan_left = min(verifier.commit, plan.shape[1]) - 1
                    plan_ptr = 1
                    a = plan[:, 0]
                    print(f"[t={t:3d}] CheckVLA TRIGGER score={float(score[0]):.2f} -> suffix repaired")
        obs_prev, a_prev = obs, a
        a_np = to_np(a[0]).clip(-sim.max_vel, sim.max_vel)

        # integrate the commanded pusher position, convert to joint targets
        p_cmd = p_cmd + a_np * sim.dt
        q = to_np(ik(*p_cmd))
        franka.control_dofs_position(q[:-2], motors)
        franka.control_dofs_position(np.zeros(2), fingers)
        for _ in range(n_sub):
            scene.step()
            k += 1
            try:
                f = to_np(peg.get_links_net_contact_force()).reshape(-1, 3).sum(0)
                fmax_ep = max(fmax_ep, float(np.hypot(f[0], f[1])))
            except Exception:
                pass
            if cam is not None and k % render_every == 0:
                cam.render()
        if t % 10 == 0:
            print(f"[t={t:3d}] peg=({s[0,0]:.3f},{s[0,1]:.3f}) target=({sim.target_x:.3f},{s0[6]:.3f}) "
                  f"a=({a_np[0]:+.2f},{a_np[1]:+.2f}) max contact force so far {fmax_ep:.1f} N")

    s = read_state(T, p_cmd)
    ok = bool(sim.success(s)[0])
    print(f"done: success={ok}  max peg contact force={fmax_ep:.1f} N "
          f"(limit {sim.force_limit} N, violation={fmax_ep > sim.force_limit})  CheckVLA triggers={n_trig}")
    if cam is not None:
        cam.stop_recording(save_to_filename=args.record, fps=args.fps)
        print(f"saved {args.record}")


if __name__ == "__main__":
    main()
