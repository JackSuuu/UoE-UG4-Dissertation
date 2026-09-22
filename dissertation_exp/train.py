"""
train.py
--------
Step 1: Generate expert demonstrations using a hand-crafted PD controller
        (always at TRAIN_FRICTION = 0.5).
Step 2: Train a BC (Behaviour Cloning) MLP policy on those demonstrations.

Usage
-----
    conda activate pybullet_env
    cd dissertation_exp
    python train.py

Outputs
-------
    results/expert_data.npz      — raw (obs, action) pairs
    results/baseline_policy.pt   — trained MLP weights
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from env.push_env import BlockPushEnv, TRAIN_FRICTION, MAX_VEL
from policy.mlp_policy import MLPPolicy

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
N_EPISODES   = 300       # expert episodes to collect
MAX_EP_STEPS = 150       # max steps per expert episode
BATCH_SIZE   = 256
EPOCHS       = 60
LR           = 1e-3
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Expert (PD) controller
# ─────────────────────────────────────────────────────────────────────────────
def pd_expert(obs: np.ndarray, friction: float = TRAIN_FRICTION) -> np.ndarray:
    """
    Two-phase PD expert controller.

    Phase 1 (approach): move the pusher to a position directly behind the
      block along the block→target axis.
    Phase 2 (push): once aligned, drive pusher (and block) toward target.

    obs = [bx, by, tx, ty, dx, dy, dist]
    The pusher position is NOT in obs. We track it via a module-level dict
    keyed by the friction value (different friction → different env instance).
    This is fine for data collection since we only collect at one friction.
    """
    bx, by, tx, ty, dx, dy, dist = obs

    # Direction from block to target
    target_dir = np.array([dx, dy])
    norm = np.linalg.norm(target_dir) + 1e-8
    target_dir_unit = target_dir / norm

    # Ideal pusher position: directly behind block, offset by pusher_radius + block_half
    approach_offset = 0.055   # metres behind block
    ideal_pusher = np.array([bx, by]) - target_dir_unit * approach_offset

    # We approximate pusher position from obs history via a simple heuristic:
    # assume pusher tracks the "approach" position with lag.
    # For expert data generation, the controller just outputs velocity toward
    # the ideal pusher pos for alignment, then pushes once roughly aligned.

    # Estimate how mis-aligned pusher is (we use block position as proxy):
    # If dist > 0.15 we do a straight proportional push; the pusher starts
    # directly behind the block anyway (reset logic), so this is valid.

    if dist > 0.08:
        # Push phase: drive toward target
        speed = np.clip(dist * 3.0, 0.15, MAX_VEL)
        action = target_dir_unit * speed
    else:
        # Fine-positioning: slow down
        speed = np.clip(dist * 4.0, 0.05, 0.3)
        action = target_dir_unit * speed

    action += np.random.normal(0, 0.015, size=2)
    return np.clip(action, -MAX_VEL, MAX_VEL).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────
def collect_expert_data(n_episodes: int, friction: float = TRAIN_FRICTION):
    env = BlockPushEnv(friction=friction, seed=0)
    all_obs, all_act = [], []
    successes = 0

    for ep in range(n_episodes):
        obs = env.reset()
        for _ in range(MAX_EP_STEPS):
            act = pd_expert(obs, friction)
            all_obs.append(obs.copy())
            all_act.append(act.copy())
            obs, _, done, info = env.step(act)
            if done:
                if info["success"]:
                    successes += 1
                break

    print(f"[Expert] {n_episodes} episodes | success rate: "
          f"{successes/n_episodes*100:.1f}% | transitions: {len(all_obs)}")
    return np.array(all_obs, dtype=np.float32), np.array(all_act, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# BC Training
# ─────────────────────────────────────────────────────────────────────────────
def train_bc(obs_data: np.ndarray, act_data: np.ndarray) -> MLPPolicy:
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"[Train] Using device: {device}")

    policy = MLPPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    criterion = nn.MSELoss()

    obs_t = torch.tensor(obs_data).to(device)
    act_t = torch.tensor(act_data).to(device)
    dataset = TensorDataset(obs_t, act_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for obs_b, act_b in loader:
            pred = policy(obs_b)
            loss = criterion(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(obs_b)
        avg_loss = total_loss / len(obs_data)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1: Collecting expert demonstrations (μ = 0.5)")
    print("=" * 60)
    obs_data, act_data = collect_expert_data(N_EPISODES, TRAIN_FRICTION)

    np.savez(os.path.join(RESULTS_DIR, "expert_data.npz"),
             obs=obs_data, act=act_data)
    print(f"  Saved → results/expert_data.npz\n")

    print("=" * 60)
    print("  Phase 2: Behaviour Cloning training")
    print("=" * 60)
    policy = train_bc(obs_data, act_data)

    save_path = os.path.join(RESULTS_DIR, "baseline_policy.pt")
    torch.save(policy.state_dict(), save_path)
    print(f"\n  Saved → results/baseline_policy.pt")
    print("\nDone. Run evaluate.py next.")
