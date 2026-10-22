# src — Current five-layer verifier–repair framework

This directory contains only the adapters for the **five-layer verifier–repair–reverification–reliable-handoff framework** around the frozen OpenVLA actor (LIBERO microwave task).

## Layout

```
src/
├── adapters/        # All runtime adapters (see src/adapters/README.md)
│   ├── native_framework.py   # Core 5-layer controller (FROZEN)
│   ├── libero_openvla.py     # OpenVLA actor wrapper (policy_image)
│   ├── tf5_compat.py         # TF compat layer (imported by libero_openvla)
│   ├── physical_verifier.py  # Task-agnostic physics verification
│   ├── visual_grounding.py   # Frozen DINOv2/V-JEPA encoder
│   ├── vision_physical_monitor.py  # Joint vision+physical monitor
│   ├── vision_subgoal_inference.py # Vision-based stage inference
│   ├── fusion_mlp.py         # Verification fusion MLP
│   ├── train_fusion_mlp.py   # Training script for fusion MLP
│   ├── short_term_predictor.py   # Pre-emptive exit prediction
│   ├── calibrated_risk_head.py   # Conformal-calibrated risk scorer
│   ├── action_suffix_corrector.py  # Lightweight action safety nudge
│   ├── keyframe_bank.py      # Episodic context for stage inference
│   └── README.md             # Adapter architecture doc
├── results/          # Experiment outputs (gitignored / not tracked)
└── README.md         # This file
```

## Deleted Phase 1 code

The following Phase 1 directories/files were removed because they are not used by the current LIBERO microwave five-layer framework:

- `src/experiments/`, `src/models/`, `src/sims/`, `src/systems/`, `src/tests/`, `src/vla/`, `src/demo/`
- `src/common.py`, `src/interfaces.py`, `src/registry.py`, `src/requirements.txt`, `src/run_all.sh`
- `src/checkvla/` (CheckVLA stand-in implementation)

All of the above are preserved in git history. To restore them, check out the commit before deletion:

```bash
# See the commit that deleted Phase 1 code
git log --oneline --diff-filter=D -- src/

# Restore Phase 1 tree (example, replace <commit> with actual hash)
git checkout <commit>^ -- src/experiments src/models src/sims src/systems src/tests src/vla src/demo src/common.py src/interfaces.py src/registry.py src/requirements.txt src/run_all.sh src/checkvla
```

The most recent commit containing Phase 1 code is the parent of the commit that removed it. Use `git log -- src/` to find the exact hash.

## Current focus

Root0 failure analysis: root0 fails in the **acquire phase** (persistent_low_grasp), never reaching contain_release. All prompt-only variants had zero effect on root0. The current improvement direction is:

1. Wire a real DINOv2/V-JEPA encoder into `visual_grounding.py` (currently using deterministic random projection fallback).
2. Train `CalibratedRiskHead` on success trajectories so it can fire a `high_risk` trigger for root0's acquire-phase failures.
3. Extend `ShortTermMotionPredictor` to detect "object not lifting after grasp" in the acquire phase.
4. Add a refusal/safe-action option in `ActionSuffixCorrector` when risk is extreme, instead of only nudging translation.