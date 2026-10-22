"""Short-term motion predictor for pre-emptive repair triggering.

Implements CheckVLA Module 1 (Action-Conditioned Rolling Prediction) in a
lightweight, model-free form.  Instead of a learned world model, it extrapolates
the object's future position from recent observed motion + the VLA's proposed
action intent.  If the extrapolated trajectory leaves the receptacle bounds
within the prediction horizon, a pre-emptive repair trigger is raised before the
physical violation actually occurs.

Task-agnostic: operates on generic object position + receptacle bounds extracted
from diagnostics at runtime.  No hard-coded geom names.
"""
from typing import Optional, Dict, Any, List
import math
import numpy as np


class ShortTermMotionPredictor:
    """Extrapolates object motion over a short horizon using recent history.

    Prediction is a simple constant-velocity model blended with the proposed
    action's translational intent.  This is NOT a learned world model — it is a
    deterministic physics heuristic that runs in O(horizon) time.
    """

    def __init__(
        self,
        horizon_steps: int = 5,
        dt: float = 0.05,
        velocity_window: int = 3,
        action_blend: float = 0.5,
        lift_threshold_m: float = 0.02,
    ):
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.velocity_window = velocity_window
        self.action_blend = action_blend  # weight on action intent vs observed velocity
        self.lift_threshold_m = lift_threshold_m  # min Z rise to count as lifted
        self._position_history: List[np.ndarray] = []
        self._grasp_start_pos: Optional[np.ndarray] = None
        self._grasp_start_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Update history (call once per step after execution)
    # ------------------------------------------------------------------
    def update(
        self,
        object_pos: Optional[List[float]],
        grasp: Optional[bool] = None,
        step: Optional[int] = None,
    ) -> None:
        if object_pos is not None:
            self._position_history.append(np.array(object_pos, dtype=np.float32))
            if len(self._position_history) > self.velocity_window + 1:
                self._position_history.pop(0)

        # Track grasp start for no-lift detection (acquire phase)
        if grasp is True and self._grasp_start_pos is None and object_pos is not None:
            self._grasp_start_pos = np.array(object_pos, dtype=np.float32)
            self._grasp_start_step = step
        elif grasp is False:
            # Grasp released: reset lift tracking
            self._grasp_start_pos = None
            self._grasp_start_step = None

    # ------------------------------------------------------------------
    # Predict future positions
    # ------------------------------------------------------------------
    def predict(
        self,
        current_pos: Optional[List[float]],
        proposed_action: Optional[List[float]],
    ) -> Optional[List[List[float]]]:
        """Return predicted object positions for steps 1..horizon_steps.

        Returns None if insufficient history.
        """
        if current_pos is None or len(self._position_history) < 2:
            return None

        curr = np.array(current_pos, dtype=np.float32)

        # Observed velocity from recent position history
        if len(self._position_history) >= 2:
            recent = self._position_history[-self.velocity_window - 1:]
            # Simple average velocity over the window
            vel_obs = (recent[-1] - recent[0]) / ((len(recent) - 1) * self.dt)
        else:
            vel_obs = np.zeros(3, dtype=np.float32)

        # Action intent direction (first 3 dims of proposed action)
        if proposed_action is not None:
            action_dir = np.array(proposed_action[:3], dtype=np.float32)
            action_norm = np.linalg.norm(action_dir)
            if action_norm > 1e-6:
                action_dir = action_dir / action_norm
            else:
                action_dir = np.zeros(3, dtype=np.float32)
        else:
            action_dir = np.zeros(3, dtype=np.float32)

        # Blended velocity: observed + action intent
        blended_vel = (1.0 - self.action_blend) * vel_obs + self.action_blend * action_dir

        # Extrapolate
        predictions: List[List[float]] = []
        pos = curr.copy()
        for h in range(self.horizon_steps):
            pos = pos + blended_vel * self.dt
            predictions.append(pos.tolist())

        return predictions

    # ------------------------------------------------------------------
    # Check if any predicted position leaves the receptacle bounds
    # ------------------------------------------------------------------
    def predicts_exit(
        self,
        current_pos: Optional[List[float]],
        proposed_action: Optional[List[float]],
        receptacle_bounds: Optional[Dict[str, List[float]]],
    ) -> Optional[Dict[str, Any]]:
        """Check whether the predicted trajectory exits the receptacle.

        Returns a dict with:
          - 'exit_predicted': bool
          - 'exit_step': int or None (1-indexed within horizon)
          - 'predicted_positions': list of [x,y,z]
          - 'min_distance_to_boundary': float or None
        """
        if receptacle_bounds is None or current_pos is None:
            return None

        predictions = self.predict(current_pos, proposed_action)
        if predictions is None:
            return None

        minx, miny, minz = receptacle_bounds['min']
        maxx, maxy, maxz = receptacle_bounds['max']

        exit_predicted = False
        exit_step = None
        min_dist = float('inf')

        for h, pos in enumerate(predictions, start=1):
            x, y, z = pos
            inside = (minx <= x <= maxx and miny <= y <= maxy and minz <= z <= maxz)
            if not inside:
                exit_predicted = True
                exit_step = h
                break
            # Distance to nearest boundary (positive = inside margin)
            dist = min(
                x - minx, maxx - x,
                y - miny, maxy - y,
                z - minz, maxz - z,
            )
            min_dist = min(min_dist, dist)

        return {
            'exit_predicted': exit_predicted,
            'exit_step': exit_step,
            'predicted_positions': predictions,
            'min_distance_to_boundary': min_dist if not exit_predicted else 0.0,
        }

    # ------------------------------------------------------------------
    # Check if object has not lifted after being grasped (acquire phase)
    # ------------------------------------------------------------------
    def predicts_no_lift(
        self,
        grasp: Optional[bool],
        current_pos: Optional[List[float]],
        step: Optional[int] = None,
        max_steps_since_grasp: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Check whether the object has not been lifted after grasp.

        root0 failure mode: object is grasped but never lifted above
        lift_threshold_m within max_steps_since_grasp steps.

        Returns:
          - 'no_lift_predicted': bool
          - 'steps_since_grasp': int or None
          - 'z_rise_m': float or None
        """
        if self._grasp_start_pos is None or current_pos is None:
            return None
        if grasp is not True:
            return None

        steps_since = None
        if self._grasp_start_step is not None and step is not None:
            steps_since = step - self._grasp_start_step

        z_rise = float(current_pos[2] - self._grasp_start_pos[2])
        no_lift = z_rise < self.lift_threshold_m

        # Only flag if we've waited long enough to conclude it's stuck
        stuck = no_lift and (steps_since is None or steps_since >= max_steps_since_grasp)

        return {
            'no_lift_predicted': stuck,
            'steps_since_grasp': steps_since,
            'z_rise_m': z_rise,
            'grasp_start_pos': self._grasp_start_pos.tolist(),
            'current_pos': current_pos,
        }

    def reset(self) -> None:
        self._position_history = []
        self._grasp_start_pos = None
        self._grasp_start_step = None


def make_short_term_predictor(
    horizon_steps: int = 5,
    dt: float = 0.05,
    velocity_window: int = 3,
    action_blend: float = 0.5,
    lift_threshold_m: float = 0.02,
) -> ShortTermMotionPredictor:
    return ShortTermMotionPredictor(
        horizon_steps=horizon_steps,
        dt=dt,
        velocity_window=velocity_window,
        action_blend=action_blend,
        lift_threshold_m=lift_threshold_m,
    )
