"""
push_env.py
-----------
A MuJoCo-based block-pushing environment for the Physical CoT experiment.

Key design choices:
- The floor and block friction can be changed at reset time (set_friction)
  → this is the OOD variable that the baseline BC policy never saw during training
- State: [block_x, block_y, target_x, target_y, dx, dy, dist]  (7-dim)
- Action: [vx, vy]  velocity command for the pusher  (2-dim, clipped to ±MAX_VEL)
- Episode ends when block reaches target (success) or max_steps exceeded
"""

import os
import copy
import numpy as np
import mujoco

XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/push_scene.xml")

MAX_VEL   = 0.6          # m/s — maximum pusher velocity
DT        = 0.005        # simulation timestep (matches XML)
CTRL_FREQ = 10           # control steps per action (50ms control period)
MAX_STEPS = 300          # max control steps per episode
SUCCESS_DIST = 0.05      # block within 5cm of target → success
TRAIN_FRICTION = 0.5     # friction used during training / expert data generation


class BlockPushEnv:
    """
    Minimal MuJoCo block-pushing environment.

    Parameters
    ----------
    friction : float
        Sliding friction coefficient of the floor and block geoms.
        Default = TRAIN_FRICTION (the value used for BC training).
    seed : int
        RNG seed for reproducible resets.
    """

    def __init__(self, friction: float = TRAIN_FRICTION, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self.rng   = np.random.default_rng(seed)
        self.friction = friction
        self._apply_friction(friction)

        # joint / actuator indices (cached for speed)
        self._jnt = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("block_x", "block_y", "pusher_x", "pusher_y")
        }
        self._act = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("push_x", "push_y")
        }
        self._body_block  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self._body_target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        self._body_pusher = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")

        self.target_pos = np.array([0.4, 0.0])
        self.step_count = 0

        # observation / action dims
        self.obs_dim = 7
        self.act_dim = 2

    # ------------------------------------------------------------------
    # Friction control
    # ------------------------------------------------------------------
    def _apply_friction(self, mu: float):
        """Overwrite friction for floor and block geoms."""
        for geom_name in ("floor", "block_geom", "pusher_geom"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if gid >= 0:
                self.model.geom_friction[gid, 0] = mu   # sliding
                self.model.geom_friction[gid, 1] = 0.005
                self.model.geom_friction[gid, 2] = 0.0001
        self.friction = mu

    def set_friction(self, mu: float):
        self._apply_friction(mu)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, block_pos=None, target_pos=None):
        """Reset simulation; randomise block/target within a safe range."""
        mujoco.mj_resetData(self.model, self.data)

        # Block start: near origin with small jitter
        bx = self.rng.uniform(-0.05, 0.05) if block_pos is None else block_pos[0]
        by = self.rng.uniform(-0.05, 0.05) if block_pos is None else block_pos[1]

        # Target: somewhere in front of the block
        tx = self.rng.uniform(0.25, 0.45) if target_pos is None else target_pos[0]
        ty = self.rng.uniform(-0.15, 0.15) if target_pos is None else target_pos[1]

        # Set block qpos (slides only — no rotation joint)
        self.data.qpos[self._jnt["block_x"]]   = bx
        self.data.qpos[self._jnt["block_y"]]   = by

        # Pusher starts directly behind the block along X axis
        self.data.qpos[self._jnt["pusher_x"]] = bx - 0.06
        self.data.qpos[self._jnt["pusher_y"]] = by

        # Move target marker body
        self.model.body_pos[self._body_target, 0] = tx
        self.model.body_pos[self._body_target, 1] = ty
        self.target_pos = np.array([tx, ty])

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        """
        Apply a 2D velocity command to the pusher for CTRL_FREQ sim steps.

        Returns
        -------
        obs, reward, done, info
        """
        action = np.clip(action, -MAX_VEL, MAX_VEL)
        self.data.ctrl[self._act["push_x"]] = action[0]
        self.data.ctrl[self._act["push_y"]] = action[1]

        for _ in range(CTRL_FREQ):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs  = self._get_obs()
        dist = obs[6]
        reward   = -dist                         # dense: negative distance
        success  = dist < SUCCESS_DIST
        truncated= self.step_count >= MAX_STEPS
        done     = success or truncated

        info = {
            "success": success,
            "dist": dist,
            "block_pos": self._block_pos().copy(),
            "target_pos": self.target_pos.copy(),
            "friction": self.friction,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _block_pos(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self._jnt["block_x"]],
            self.data.qpos[self._jnt["block_y"]],
        ])

    def _pusher_pos(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self._jnt["pusher_x"]],
            self.data.qpos[self._jnt["pusher_y"]],
        ])

    def _get_obs(self) -> np.ndarray:
        bp = self._block_pos()
        tp = self.target_pos
        dx, dy = tp - bp
        dist   = np.linalg.norm([dx, dy])
        pp     = self._pusher_pos()
        return np.array([bp[0], bp[1], tp[0], tp[1], dx, dy, dist], dtype=np.float32)

    # ------------------------------------------------------------------
    # Physics snapshot/restore (used by physics correction module)
    # ------------------------------------------------------------------
    def snapshot(self):
        """Return a deep copy of (qpos, qvel, ctrl) for rollback."""
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
        )

    def restore(self, snap):
        """Restore simulation state from snapshot."""
        qpos, qvel, ctrl = snap
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)

    def simulate_action(self, action: np.ndarray, n_steps: int = 5) -> np.ndarray:
        """
        Non-destructive forward rollout: simulate `action` for `n_steps`
        control steps and return the predicted block positions (n_steps × 2).
        Restores state afterwards.
        """
        snap = self.snapshot()
        traj = []
        action = np.clip(action, -MAX_VEL, MAX_VEL)
        for _ in range(n_steps):
            self.data.ctrl[self._act["push_x"]] = action[0]
            self.data.ctrl[self._act["push_y"]] = action[1]
            for _ in range(CTRL_FREQ):
                mujoco.mj_step(self.model, self.data)
            traj.append(self._block_pos().copy())
        self.restore(snap)
        return np.array(traj)   # shape (n_steps, 2)
