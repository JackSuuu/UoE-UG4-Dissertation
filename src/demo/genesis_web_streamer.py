"""
Genesis Web Streamer — view the simulation in your Mac's browser, no VNC needed.

Uses Genesis's offscreen renderer to capture frames, then serves them via HTTP
with MJPEG streaming. The simulation runs on the server; the browser on your
Mac just displays the video stream.

Usage:
    # On the server:
    python demo/genesis_web_streamer.py --port 8080

    # On your Mac, open:
    http://<server-ip>:8080

    # Or use SSH tunnel (more secure):
    ssh -L 8080:localhost:8080 user@server
    # then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sims.torch_push import PushSim  # noqa: E402

X0 = 0.35
FINGER_Z = 0.015
HAND_TO_TIP = 0.105
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def peg_yaw_w(peg):
    """Yaw angle and yaw rate of the peg (state columns 8, 9)."""
    q = to_np(peg.get_quat()).reshape(-1)                 # wxyz
    yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return yaw, float(to_np(peg.get_ang()).reshape(-1)[2])


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves MJPEG stream and control page."""

    streamer = None  # set by main()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(self._html_page().encode())

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last = None
                while True:
                    frame = self.streamer.get_frame()
                    if frame is None or frame is last:
                        time.sleep(0.03)
                        continue
                    last = frame
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/status":
            import json
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = self.streamer.get_status() if self.streamer else {}
            self.wfile.write(json.dumps(status).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs

    def _html_page(self):
        return """<!DOCTYPE html>
<html>
<head>
    <title>Genesis Franka Demo</title>
    <style>
        body { margin: 0; background: #1a1a2e; display: flex; flex-direction: column;
               align-items: center; justify-content: center; height: 100vh;
               font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #eee; }
        h1 { font-size: 1.2em; margin-bottom: 8px; }
        .info { font-size: 0.85em; color: #aaa; margin-bottom: 12px; }
        img { border-radius: 8px; box-shadow: 0 4px 24px rgba(0,0,0,0.5); }
        .controls { margin-top: 12px; display: flex; gap: 8px; }
        button { padding: 6px 16px; border: none; border-radius: 6px; cursor: pointer;
                 background: #4a9eff; color: #fff; font-size: 0.9em; }
        button:hover { background: #3a8eef; }
        #status { font-size: 0.8em; color: #888; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>Genesis Franka Arm — Push Insertion</h1>
    <div class="info">Real-time simulation stream from server</div>
    <img src="/stream" width="960" height="540">
    <div class="controls">
        <button onclick="fetch('/status').then(r=>r.json()).then(s=>document.getElementById('status').textContent=JSON.stringify(s))">Status</button>
    </div>
    <div id="status"></div>
</body>
</html>"""


class GenesisStreamer:
    """Captures Genesis frames and serves them to web clients."""

    def __init__(self, sim, scene, cam, peg, franka, hand, ik, n_sub, T, policy_fn):
        self.sim = sim
        self.scene = scene
        self.cam = cam
        self.peg = peg
        self.franka = franka
        self.hand = hand
        self.ik = ik
        self.n_sub = n_sub
        self.T = T
        self.policy_fn = policy_fn

        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._stats = {"step": 0, "success": False, "fmax": 0.0, "n_trig": 0}

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_frame(self):
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame

    def get_status(self):
        return self._stats.copy()

    def _run(self):
        """Loop episodes forever so the browser always has a live stream."""
        ep = 0
        while self._running:
            if ep > 0:
                self.scene.reset()
            print(f"[streamer] episode {ep}")
            self._episode()
            ep += 1

    def _episode(self):
        """One simulation episode."""
        dev = self.sim.device
        motors = np.arange(7)
        fingers = np.arange(7, 9)
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))

        g = torch.Generator(device=dev).manual_seed(0)
        s0 = self.sim.init_state(1, g)[0].cpu().numpy()
        p_cmd = np.array([s0[4], s0[5]])
        q0 = to_np(self.ik(s0[4], s0[5])).copy()
        q0[-2:] = 0.0
        self.franka.set_dofs_position(q0)
        self.franka.control_dofs_position(q0[:-2], motors)
        self.franka.control_dofs_position(np.zeros(2), fingers)
        for _ in range(20):
            self.scene.step()

        obs_prev, a_prev = None, torch.zeros(1, 2, device=dev)
        fmax_ep = 0.0

        for t in range(self.T):
            if not self._running:
                break
            pp = to_np(self.peg.get_pos()).reshape(-1)
            pv = to_np(self.peg.get_vel()).reshape(-1)
            hp = to_np(self.hand.get_pos()).reshape(-1)
            yaw, wz = peg_yaw_w(self.peg)
            s = np.array([pp[0] - X0, pp[1], pv[0], pv[1],
                          hp[0] - X0, hp[1], s0[6], t, yaw, wz], np.float32)
            s_t = torch.as_tensor(s, device=dev)[None]
            obs = self.sim.obs(s_t)
            obs_prev = obs if obs_prev is None else obs_prev

            with torch.no_grad():
                a = self.policy_fn(s_t, t)
            obs_prev, a_prev = obs, a
            a_np = to_np(a[0]).clip(-self.sim.max_vel, self.sim.max_vel)

            p_cmd = p_cmd + a_np * self.sim.dt
            q = to_np(self.ik(*p_cmd))
            self.franka.control_dofs_position(q[:-2], motors)
            self.franka.control_dofs_position(np.zeros(2), fingers)
            for _ in range(self.n_sub):
                self.scene.step()

            try:
                f = to_np(self.peg.get_links_net_contact_force()).reshape(-1, 3).sum(0)
                fmax_ep = max(fmax_ep, float(np.hypot(f[0], f[1])))
            except Exception:
                pass

            # Capture frame
            try:
                rgb = self.cam.render(rgb=True)[0]
                img = np.asarray(rgb)
                # Convert to JPEG
                from PIL import Image
                pil_img = Image.fromarray(img)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                with self._frame_lock:
                    self._frame = buf.getvalue()
            except Exception as e:
                print(f"[streamer] render error: {e}")

            self._stats = {"step": t, "success": False,
                           "fmax": fmax_ep, "n_trig": 0}
            if t % 10 == 0:
                print(f"[streamer] t={t:3d} peg=({pp[0]:.3f},{pp[1]:.3f}) "
                      f"a=({a_np[0]:+.2f},{a_np[1]:+.2f}) fmax={fmax_ep:.1f}N")

        pp = to_np(self.peg.get_pos()).reshape(-1)
        pv = to_np(self.peg.get_vel()).reshape(-1)
        yaw, wz = peg_yaw_w(self.peg)
        s = np.array([pp[0] - X0, pp[1], pv[0], pv[1], 0, 0, s0[6], self.T, yaw, wz], np.float32)
        s_t = torch.as_tensor(s, device=dev)[None]
        ok = bool(self.sim.success(s_t)[0])
        self._stats = {"step": self.T, "success": ok,
                       "fmax": fmax_ep, "n_trig": 0}
        print(f"[streamer] done: success={ok} fmax={fmax_ep:.1f}N")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import genesis as gs
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
            logging_level="warning")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = PushSim(dev)
    nom = sim.nominal_params()
    mu = float(nom["friction"]) * args.friction
    mass = float(nom["mass"]) * args.mass
    dt_sub = 0.01
    n_sub = int(round(sim.dt / dt_sub))
    T = args.steps or sim.T

    cam_pos, cam_look = (1.25, -0.95, 0.75), (X0 + 0.15, 0.0, 0.05)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt_sub, substeps=4),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=cam_pos, camera_lookat=cam_look,
            camera_fov=40, refresh_rate=60),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(),
                     material=gs.materials.Rigid(friction=mu))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    h = 0.03
    vol = (2 * sim.R_b) ** 2 * h
    g = torch.Generator().manual_seed(args.seed)
    s0 = sim.init_state(1, g)[0].cpu().numpy()
    peg = scene.add_entity(
        gs.morphs.Box(size=(2 * sim.R_b, 2 * sim.R_b, h), pos=(X0 + s0[0], s0[1], h / 2),
                      euler=(0, 0, float(np.degrees(s0[8])))),
        material=gs.materials.Rigid(rho=mass / vol, friction=mu),
        surface=gs.surfaces.Default(color=(0.9, 0.5, 0.1)))
    scene.add_entity(
        gs.morphs.Box(size=(0.02, 0.4, 0.06),
                      pos=(X0 + sim.x_w + 0.01, 0.0, 0.03), fixed=True),
        material=gs.materials.Rigid(friction=mu),
        surface=gs.surfaces.Default(color=(0.4, 0.4, 0.45)))
    cam = scene.add_camera(res=(1280, 720), pos=cam_pos, lookat=cam_look,
                           fov=40, GUI=False)
    scene.build()

    hand = franka.get_link("hand")

    def ik(px, py):
        return franka.inverse_kinematics(
            link=hand, pos=np.array([X0 + px, py, FINGER_Z + HAND_TO_TIP]),
            quat=DOWN_QUAT)

    def expert_policy(s, t):
        return sim.expert(s)

    streamer = GenesisStreamer(sim, scene, cam, peg, franka, hand, ik,
                               n_sub, T, expert_policy)

    StreamingHandler.streamer = streamer
    server = ThreadingHTTPServer(("0.0.0.0", args.port), StreamingHandler)

    print(f"[web] Genesis streamer running at http://0.0.0.0:{args.port}")
    print(f"[web] Open http://<server-ip>:{args.port} in your Mac's browser")
    print(f"[web] Or: ssh -L {args.port}:localhost:{args.port} user@server, then http://localhost:{args.port}")

    streamer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        server.shutdown()
        print("[web] stopped")


if __name__ == "__main__":
    main()
