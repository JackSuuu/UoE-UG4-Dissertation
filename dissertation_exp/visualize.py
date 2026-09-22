"""
visualize.py
------------
Generate publication-quality figures from evaluation results.

Figures produced
----------------
  results/fig1_success_rate.pdf/.png
      Line plot: success rate vs friction coefficient, three methods.
      Highlights the training friction (μ=0.5) with a vertical dashed line.

  results/fig2_mean_dist.pdf/.png
      Bar chart: mean final distance to target with std error bars.

  results/fig3_trajectory_compare.pdf/.png
      Qualitative trajectory comparison at μ=0.1 (most OOD condition):
      Baseline BC vs Physics-Corrected vs Expert PD.

Usage
-----
    conda activate pybullet_env
    cd dissertation_exp
    python visualize.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless (no display needed on macOS terminal)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from env.push_env import BlockPushEnv, MAX_STEPS as MAX_EP_STEPS, TRAIN_FRICTION
from policy.mlp_policy import MLPPolicy
from policy.physics_correction import PhysicsCorrector
from train import pd_expert

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
COLORS  = {"Baseline BC": "#e74c3c",
           "Physics-Corrected (Ours)": "#2ecc71",
           "Expert PD (Oracle)": "#3498db"}
MARKERS = {"Baseline BC": "o",
           "Physics-Corrected (Ours)": "s",
           "Expert PD (Oracle)": "^"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_results():
    path = os.path.join(RESULTS_DIR, "eval_results.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("Run evaluate.py first.")
    d = np.load(path, allow_pickle=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Success rate vs friction
# ─────────────────────────────────────────────────────────────────────────────
def fig_success_rate(d):
    mu_vals = d["friction_values"]
    methods = [str(m) for m in d["methods"]]
    srates  = d["success_rates"]   # (3, n_mu)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, m in enumerate(methods):
        ax.plot(mu_vals, srates[i] * 100,
                marker=MARKERS[m], color=COLORS[m],
                linewidth=2.0, markersize=7, label=m)

    ax.axvline(TRAIN_FRICTION, color="gray", linestyle="--",
               alpha=0.7, linewidth=1.2, label=f"Training μ = {TRAIN_FRICTION}")
    ax.fill_betweenx([0, 105], 0, TRAIN_FRICTION - 0.05,
                     color="lightyellow", alpha=0.4, zorder=0)
    ax.fill_betweenx([0, 105], TRAIN_FRICTION + 0.05, 1.0,
                     color="lightyellow", alpha=0.4, zorder=0)

    ax.set_xlabel("Floor Friction Coefficient (μ)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Generalisation Under Friction Shift\n"
                 "Baseline BC vs Physics-Corrected Policy", fontsize=13)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 105)
    ax.set_xticks(mu_vals)
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotation showing OOD gap
    ood_mu = 0.1
    ood_idx = list(mu_vals).index(ood_mu)
    gap = srates[1, ood_idx] - srates[0, ood_idx]
    if gap > 0.02:
        ax.annotate(
            f"+{gap*100:.0f}% at μ={ood_mu}",
            xy=(ood_mu, srates[1, ood_idx] * 100),
            xytext=(ood_mu + 0.05, srates[1, ood_idx] * 100 + 8),
            fontsize=9, color=COLORS["Physics-Corrected (Ours)"],
            arrowprops=dict(arrowstyle="->", color=COLORS["Physics-Corrected (Ours)"])
        )

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig1_success_rate.png"))
    fig.savefig(os.path.join(RESULTS_DIR, "fig1_success_rate.pdf"))
    print("  Saved → fig1_success_rate.png/.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Mean distance bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_mean_dist(d):
    mu_vals = d["friction_values"]
    methods = [str(m) for m in d["methods"]]
    mdists  = d["mean_dists"]
    sdists  = d["std_dists"]

    x = np.arange(len(mu_vals))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, m in enumerate(methods):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, mdists[i], width,
                      yerr=sdists[i], capsize=4,
                      color=COLORS[m], alpha=0.85, label=m,
                      error_kw={"elinewidth": 1.2})

    ax.set_xlabel("Floor Friction Coefficient (μ)", fontsize=12)
    ax.set_ylabel("Mean Final Distance to Target (m)", fontsize=12)
    ax.set_title("Final Position Error Across Friction Conditions", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"μ={v:.1f}" for v in mu_vals])
    ax.axhline(0.05, color="gray", linestyle="--", alpha=0.6,
               linewidth=1.2, label="Success threshold (0.05 m)")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig2_mean_dist.png"))
    fig.savefig(os.path.join(RESULTS_DIR, "fig2_mean_dist.pdf"))
    print("  Saved → fig2_mean_dist.png/.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Trajectory comparison at μ = 0.1 (most OOD)
# ─────────────────────────────────────────────────────────────────────────────
def collect_trajectory(policy_fn, friction: float, seed: int = 7777):
    env = BlockPushEnv(friction=friction, seed=seed)
    obs = env.reset(block_pos=[0.0, 0.0], target_pos=[0.4, 0.0])
    traj = [env._block_pos().copy()]
    for _ in range(MAX_EP_STEPS):
        act = policy_fn(obs)
        obs, _, done, info = env.step(act)
        traj.append(env._block_pos().copy())
        if done:
            break
    return np.array(traj), info["success"]


def fig_trajectory(d):
    path = os.path.join(RESULTS_DIR, "baseline_policy.pt")
    if not os.path.exists(path):
        print("  [skip fig3] baseline_policy.pt not found")
        return

    bc = MLPPolicy()
    bc.load_state_dict(torch.load(path, map_location="cpu"))
    bc.eval()

    ood_mu = 0.1

    # Baseline
    traj_base, suc_base = collect_trajectory(bc.predict, ood_mu, seed=7777)

    # Physics-corrected
    phys_env = BlockPushEnv(friction=ood_mu, seed=7777)
    cor = PhysicsCorrector(phys_env, n_rollout_steps=5, verbose=False)
    def phys_fn(obs):
        return cor.correct(obs, bc.predict(obs))
    traj_phys, suc_phys = collect_trajectory(phys_fn, ood_mu, seed=7777)

    # Expert
    def expert_fn(obs):
        return pd_expert(obs, ood_mu)
    traj_exp, suc_exp = collect_trajectory(expert_fn, ood_mu, seed=7777)

    fig, ax = plt.subplots(figsize=(7, 5))

    styles = [
        (traj_base, "Baseline BC",             COLORS["Baseline BC"],            "--", suc_base),
        (traj_phys, "Physics-Corrected (Ours)", COLORS["Physics-Corrected (Ours)"], "-",  suc_phys),
        (traj_exp,  "Expert PD (Oracle)",       COLORS["Expert PD (Oracle)"],       ":",  suc_exp),
    ]
    for traj, label, color, ls, suc in styles:
        suffix = " ✓" if suc else " ✗"
        ax.plot(traj[:, 0], traj[:, 1], linestyle=ls, color=color,
                linewidth=2.0, label=label + suffix)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=6)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color=color, markersize=8, markeredgewidth=2)

    # Target circle
    target = plt.Circle((0.4, 0.0), 0.05, color="green", fill=False,
                         linestyle="--", linewidth=1.5, label="Target zone (r=0.05)")
    ax.add_patch(target)
    ax.plot(0.4, 0.0, "g*", markersize=12)

    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    ax.set_title(f"Block Trajectory Comparison  (μ = {ood_mu}, OOD condition)\n"
                 "○ = start,  × = end", fontsize=12)
    ax.set_xlim(-0.15, 0.55)
    ax.set_ylim(-0.25, 0.25)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_trajectory.png"))
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_trajectory.pdf"))
    print("  Saved → fig3_trajectory.png/.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating thesis figures …\n")
    d = load_results()
    fig_success_rate(d)
    fig_mean_dist(d)
    fig_trajectory(d)
    print("\nAll figures saved to results/")
