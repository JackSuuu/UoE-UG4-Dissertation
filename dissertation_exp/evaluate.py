"""
evaluate.py
-----------
Compare three policies across a range of friction coefficients:

  1. Baseline BC      — pure MLP, no physics awareness
  2. Physics-Corrected — BC policy + physics engine residual correction
  3. Expert (PD)      — oracle upper bound (uses ground-truth friction)

Metrics
-------
- Success rate (block within SUCCESS_DIST of target at episode end)
- Mean final distance to target
- Mean episode steps to success

Usage
-----
    conda activate pybullet_env
    cd dissertation_exp
    python evaluate.py

Outputs
-------
    results/eval_results.npz   — raw numbers for visualize.py
    results/cot_example.txt    — one Physical CoT trace for the thesis
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from env.push_env import BlockPushEnv, MAX_VEL, SUCCESS_DIST, TRAIN_FRICTION
from policy.mlp_policy import MLPPolicy
from policy.physics_correction import PhysicsCorrector
from train import pd_expert   # reuse the PD oracle

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
FRICTION_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]   # test distribution
N_EVAL_EPS      = 60                                   # episodes per (method, μ)
MAX_EP_STEPS    = 300
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a policy
# ─────────────────────────────────────────────────────────────────────────────
def eval_policy(policy_fn, friction: float, n_eps: int, seed_offset: int = 0):
    """
    Run `n_eps` episodes with a given friction, return dict of metrics.

    policy_fn : callable(obs) → action (np array, shape (2,))
    """
    env = BlockPushEnv(friction=friction, seed=seed_offset)
    successes, dists, steps_list = [], [], []

    for ep in range(n_eps):
        obs = env.reset()
        for t in range(1, MAX_EP_STEPS + 1):
            act = policy_fn(obs)
            obs, _, done, info = env.step(act)
            if done:
                break
        successes.append(float(info["success"]))
        dists.append(info["dist"])
        steps_list.append(t)

    return {
        "success_rate": np.mean(successes),
        "mean_dist":    np.mean(dists),
        "mean_steps":   np.mean(steps_list),
        "std_dist":     np.std(dists),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load trained BC policy
# ─────────────────────────────────────────────────────────────────────────────
def load_baseline() -> MLPPolicy:
    path = os.path.join(RESULTS_DIR, "baseline_policy.pt")
    if not os.path.exists(path):
        raise FileNotFoundError("Run train.py first to generate baseline_policy.pt")
    policy = MLPPolicy()
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading trained BC policy …")
    bc_policy = load_baseline()

    # storage: shape (3 methods, len(FRICTION_VALUES))
    methods = ["Baseline BC", "Physics-Corrected (Ours)", "Expert PD (Oracle)"]
    n_mu = len(FRICTION_VALUES)

    success_rates = np.zeros((3, n_mu))
    mean_dists    = np.zeros((3, n_mu))
    std_dists     = np.zeros((3, n_mu))

    cot_logged = False   # log one CoT trace for the thesis

    for j, mu in enumerate(FRICTION_VALUES):
        print(f"\n{'═'*60}")
        print(f"  Friction μ = {mu:.2f}")
        print(f"{'═'*60}")

        # ── 1. Baseline BC ──────────────────────────────────────────────
        def baseline_fn(obs):
            return bc_policy.predict(obs)

        res = eval_policy(baseline_fn, mu, N_EVAL_EPS, seed_offset=1000)
        success_rates[0, j] = res["success_rate"]
        mean_dists[0, j]    = res["mean_dist"]
        std_dists[0, j]     = res["std_dist"]
        print(f"  [Baseline BC]         success={res['success_rate']*100:5.1f}%  "
              f"mean_dist={res['mean_dist']:.4f}")

        # ── 2. Physics-Corrected ─────────────────────────────────────────
        # IMPORTANT: the corrector must share the SAME live environment so
        # its non-destructive rollouts start from the correct physics state.
        def make_phys_eval(mu_val, n_eps, seed_off):
            live_env  = BlockPushEnv(friction=mu_val, seed=seed_off)
            corrector = PhysicsCorrector(live_env, n_rollout_steps=5, verbose=False)
            successes, dists, steps_list = [], [], []
            for _ in range(n_eps):
                obs = live_env.reset()
                for t in range(1, MAX_EP_STEPS + 1):
                    raw = bc_policy.predict(obs)
                    act = corrector.correct(obs, raw)
                    obs, _, done, info = live_env.step(act)
                    if done:
                        break
                successes.append(float(info["success"]))
                dists.append(info["dist"])
                steps_list.append(t)
            return {
                "success_rate": np.mean(successes),
                "mean_dist":    np.mean(dists),
                "std_dist":     np.std(dists),
            }

        res_p = make_phys_eval(mu, N_EVAL_EPS, 2000)
        success_rates[1, j] = res_p["success_rate"]
        mean_dists[1, j]    = res_p["mean_dist"]
        std_dists[1, j]     = res_p["std_dist"]
        print(f"  [Physics-Corrected]   success={res_p['success_rate']*100:5.1f}%  "
              f"mean_dist={res_p['mean_dist']:.4f}")

        # ── 3. Expert PD oracle ──────────────────────────────────────────
        def expert_fn(obs):
            return pd_expert(obs, mu)

        res_e = eval_policy(expert_fn, mu, N_EVAL_EPS, seed_offset=3000)
        success_rates[2, j] = res_e["success_rate"]
        mean_dists[2, j]    = res_e["mean_dist"]
        std_dists[2, j]     = res_e["std_dist"]
        print(f"  [Expert PD (Oracle)]  success={res_e['success_rate']*100:5.1f}%  "
              f"mean_dist={res_e['mean_dist']:.4f}")

        # ── CoT trace (once, for μ = 0.1 to show OOD correction) ────────
        if not cot_logged and mu == 0.1:
            cot_env = BlockPushEnv(friction=mu, seed=9999)
            cot_cor = PhysicsCorrector(cot_env, n_rollout_steps=5, verbose=False)
            obs_demo = cot_env.reset(block_pos=[0.0, 0.0], target_pos=[0.4, 0.0])
            raw_act  = bc_policy.predict(obs_demo)
            corrected = cot_cor.correct(obs_demo, raw_act, print_cot=True)
            cot_path  = os.path.join(RESULTS_DIR, "cot_example.txt")
            # Capture the CoT text by re-generating it
            from policy.physics_correction import generate_physical_cot
            traj = cot_env.simulate_action(raw_act, n_steps=5)
            block_disp = np.linalg.norm(traj[-1] - obs_demo[0:2])
            contact = block_disp > 0.003
            actual_spd = block_disp / 5
            expected_spd = actual_spd * (mu / TRAIN_FRICTION)
            ratio = actual_spd / (expected_spd + 1e-8)
            cot_text = generate_physical_cot(obs_demo, mu, raw_act, corrected, contact, ratio)
            with open(cot_path, "w") as f:
                f.write(cot_text + "\n")
            cot_logged = True

    # ─────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "eval_results.npz")
    np.savez(
        out_path,
        friction_values=np.array(FRICTION_VALUES),
        methods=np.array(methods),
        success_rates=success_rates,
        mean_dists=mean_dists,
        std_dists=std_dists,
    )
    print(f"\n  Results saved → results/eval_results.npz")
    print("  Run visualize.py to generate thesis plots.")

    # ─────────────────────────────────────────────────────────────────────
    # Quick summary table
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  SUMMARY — Success Rate (%) across friction values")
    print(f"{'═'*60}")
    header = "Method" + "".join(f"  μ={mu:.1f}" for mu in FRICTION_VALUES)
    print(header)
    print("─" * len(header))
    for i, m in enumerate(methods):
        row = f"{m:<28}" + "".join(f"  {success_rates[i,j]*100:5.1f}" for j in range(n_mu))
        print(row)
