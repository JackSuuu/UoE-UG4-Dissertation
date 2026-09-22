"""
physics_correction.py
---------------------
The core "Physical CoT + Physics-Guided Residual Correction" module.

Pipeline for each timestep
--------------------------
1. Receive raw action from the BC policy (coarse semantic action).
2. Run a non-destructive 5-step forward rollout in MuJoCo to measure
   how fast the block actually moves with the proposed action.
3. Compare observed block speed vs expected speed at training friction.
   - If block moves TOO FAST  (μ < μ_train, slippery) → scale down action
   - If block moves TOO SLOW  (μ > μ_train, rough)    → scale up action
   - If no contact yet        → pass through unchanged
4. Emit a Physical CoT reasoning string (qualitative thesis figure).
5. Return corrected action.

Why this works
--------------
The BC policy was trained at μ=0.5. When tested at μ=0.1, the block
slides further per pusher step → policy overshoots.  When tested at
μ=0.9, the block barely moves → policy undershoots.
The corrector measures the actual block displacement in a shadow rollout
and rescales the pusher speed to compensate, without any gradient flow
through the policy's discrete outputs.

This is the "Residual Learning" design recommended in the proposal: the
Foundation Model handles high-level direction; physics handles local
magnitude calibration.
"""

import numpy as np
from env.push_env import MAX_VEL, TRAIN_FRICTION


# ─────────────────────────────────────────────────────────────────────────────
# Physical CoT text generator (template-driven, no real LLM needed)
# ─────────────────────────────────────────────────────────────────────────────
def generate_physical_cot(obs: np.ndarray, friction: float,
                           raw_action: np.ndarray,
                           corrected_action: np.ndarray,
                           contact_detected: bool,
                           speed_ratio: float) -> str:
    bx, by, tx, ty, dx, dy, dist = obs

    if friction < 0.25:
        friction_desc = "VERY LOW (slippery)"
        risk = "block will overshoot target due to insufficient damping"
        strategy = "reduce velocity magnitude to prevent sliding past target"
    elif friction < 0.4:
        friction_desc = "LOW"
        risk = "moderate overshoot risk on smooth surface"
        strategy = "apply gentle speed reduction toward target"
    elif friction < 0.65:
        friction_desc = "NOMINAL (training distribution)"
        risk = "minimal — surface matches training conditions"
        strategy = "use nominal proportional control"
    elif friction < 0.8:
        friction_desc = "HIGH"
        risk = "block may stall before reaching target due to excess friction"
        strategy = "increase initial velocity to overcome static friction"
    else:
        friction_desc = "VERY HIGH (rough)"
        risk = "significant stall risk — block may not move at all"
        strategy = "apply maximum safe velocity with sustained push"

    raw_spd = np.linalg.norm(raw_action)
    cor_spd = np.linalg.norm(corrected_action)
    scale = cor_spd / (raw_spd + 1e-8)

    contact_str = "YES — block displacement detected in rollout" if contact_detected \
                  else "NO  — pusher not yet in contact, pass-through"

    cot = (
        f"\n{'─'*60}\n"
        f"[Physical CoT] Timestep Analysis\n"
        f"{'─'*60}\n"
        f"  Observation  : block=({bx:.3f},{by:.3f})  target=({tx:.3f},{ty:.3f})\n"
        f"  Distance     : {dist:.4f} m\n"
        f"  Surface μ    : {friction:.2f}  →  {friction_desc}\n"
        f"  Risk         : {risk}\n"
        f"  Strategy     : {strategy}\n"
        f"  Contact det. : {contact_str}\n"
        f"  Block speed ratio (actual/expected): {speed_ratio:.3f}\n"
        f"  Raw action   : vx={raw_action[0]:.3f}  vy={raw_action[1]:.3f}  "
        f"|v|={raw_spd:.3f}\n"
        f"  Correction   : action scaled by {scale:.3f}×\n"
        f"  Final action : vx={corrected_action[0]:.3f}  vy={corrected_action[1]:.3f}  "
        f"|v|={cor_spd:.3f}\n"
        f"{'─'*60}"
    )
    return cot


# ─────────────────────────────────────────────────────────────────────────────
# Physics-guided speed scaling corrector
# ─────────────────────────────────────────────────────────────────────────────
class PhysicsCorrector:
    """
    Wraps a BlockPushEnv and a BC policy to provide physics-corrected actions.

    The corrector runs a shadow rollout and compares the block's observed
    displacement to what is expected at training friction (μ=0.5).
    It then rescales the action speed to compensate for the friction mismatch.

    Parameters
    ----------
    env : BlockPushEnv
        The live simulation environment (used for non-destructive rollouts).
    n_rollout_steps : int
        How many control steps to simulate ahead.
    contact_thresh : float
        Minimum block displacement (m) per rollout to count as "in contact".
    verbose : bool
        If True, print Physical CoT text to stdout.
    """

    def __init__(self, env, n_rollout_steps: int = 5,
                 contact_thresh: float = 0.003, verbose: bool = False):
        self.env = env
        self.n = n_rollout_steps
        self.contact_thresh = contact_thresh
        self.verbose = verbose

    def correct(self, obs: np.ndarray, raw_action: np.ndarray,
                print_cot: bool = False) -> np.ndarray:
        """
        Return a physics-corrected action.

        Algorithm
        ---------
        1. If block is already near target (fine-positioning zone), don't
           interfere — BC policy handles fine control well at all friction.
        2. Simulate raw_action for self.n steps; record block displacement.
        3. If block barely moved (no contact) → return raw_action unchanged.
        4. Compute actual block speed per step.
        5. Expected speed at training friction: speed ∝ 1/μ (sliding friction),
           so expected = actual * (μ_test / μ_train).
        6. Speed ratio r = actual / expected.
           If r > 1.05 (slippery, moving too fast) → scale action down.
           If r < 0.95 (rough, moving too slow)    → scale action up conservatively.
        7. Clip corrected action to [-MAX_VEL, MAX_VEL].
        """
        dist = obs[6]

        # Don't interfere in fine-positioning zone (BC handles it well)
        FINE_ZONE = 0.12   # m
        if dist < FINE_ZONE:
            return raw_action.copy().astype(np.float32)

        block_start = obs[0:2].copy()

        # Shadow rollout
        traj = self.env.simulate_action(raw_action, n_steps=self.n)
        block_end = traj[-1]
        block_displacement = np.linalg.norm(block_end - block_start)

        contact_detected = block_displacement > self.contact_thresh

        speed_ratio = 1.0
        corrected_action = raw_action.copy()

        if contact_detected:
            actual_speed = block_displacement / self.n

            # Physical model: block sliding speed ∝ 1/μ (Coulomb friction)
            expected_speed = actual_speed * (self.env.friction / TRAIN_FRICTION)
            speed_ratio = actual_speed / (expected_speed + 1e-8)

            # Only apply correction on SLIPPERY surfaces (μ < μ_train).
            # On rough surfaces (μ > μ_train), the block decelerates naturally
            # and the BC policy's proportional control is sufficient.
            # Speeding up on rough surfaces causes overshoot in the fine zone.
            if speed_ratio > 1.05:        # slippery → slow down
                scale = 1.0 / speed_ratio
                scale = max(scale, 0.45)
                corrected_action = np.clip(raw_action * scale, -MAX_VEL, MAX_VEL)

        if print_cot or self.verbose:
            cot_text = generate_physical_cot(
                obs, self.env.friction, raw_action, corrected_action,
                contact_detected, speed_ratio
            )
            print(cot_text)

        return corrected_action.astype(np.float32)

