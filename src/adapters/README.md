# src/adapters — VLA Verifier–Repair Framework Adapters

This directory contains the runtime adapters for the **five-layer verifier–repair–reverification–reliable-handoff framework** around the frozen OpenVLA actor.

## Directory layout

```
src/adapters/
├── __init__.py
├── native_framework.py          # Core 5-layer controller (FROZEN — do not edit mid-run)
├── libero_openvla.py            # LIBERO OpenVLA actor wrapper (policy_image, LiberoOpenVLA)
├── fusion_mlp.pt                # Trained verification fusion MLP weights
├── fusion_mlp.py                # VerificationFusionMLP: vision+physics+subgoal -> P(verify passes)
├── train_fusion_mlp.py          # Supervised training script for fusion_mlp
├── physical_verifier.py         # PhysicalVerifier: task-agnostic physics-grounded checks
├── visual_grounding.py          # VisualGroundingModule: frozen DINOv2/V-JEPA encoder
├── vision_physical_monitor.py   # VisionPhysicalMonitor: joint vision+physical monitor
├── vision_subgoal_inference.py  # VisionSubgoalInference: vision-based stage inference
├── short_term_predictor.py      # ShortTermMotionPredictor: pre-emptive exit prediction
├── calibrated_risk_head.py      # CalibratedRiskHead: conformal-calibrated risk scorer
├── action_suffix_corrector.py   # ActionSuffixCorrector: lightweight action safety nudge
├── keyframe_bank.py             # KeyframeBank: episodic context for stage inference
├── tf5_compat.py                # TF compatibility layer (shared with other experiments)
├── libero_*.py                  # LIBERO task infrastructure (shared with other experiments)
└── archive_redundant/           # Adapters removed from the current framework (assembly/fridge/etc.)
```

## Five-layer mapping

| Layer | Component | File | Role |
|-------|-----------|------|------|
| **L1** High-level subgoal | `VisionSubgoalInference` | `vision_subgoal_inference.py` | Infers current stage (`acquire/transfer/contain_release/close`) from visual features + physical boundary correction. Replaces hard-coded `_subgoal()` rules. |
| **L2** Frozen VLA | — | `native_framework.py` / `libero_openvla.py` | Frozen OpenVLA actor. Every action is one unchanged VLA proposal. No numeric edits. |
| **L3** Risk/event monitor | `VisionPhysicalMonitor` | `vision_physical_monitor.py` | Joint vision+physical trigger. Fires only when vision anomaly AND physical verifier agree. Also runs short-term prediction + risk head + action correction. |
| **L4** Dual-mode gate | — | `native_framework.py` | nominal ↔ repair routing, cooldown, budget, retry gating. |
| **L5** Bounded repair + handoff | `VerificationFusionMLP` | `fusion_mlp.py` | Scores `P(verification passes)` as a pre-handoff safety check. Physical consistency prevents unverified handoffs. |

## CheckVLA-inspired modules (new)

| Module | File | CheckVLA analog | What it adds |
|--------|------|-----------------|--------------|
| Short-term motion prediction | `short_term_predictor.py` | Module 1 (rolling prediction) | Extrapolates object position over a short horizon; pre-emptive `preemptive_exit` trigger if exit is predicted. |
| Calibrated risk head | `calibrated_risk_head.py` | Module 2 (risk triggering) | MLP risk scorer `r_t ∈ [0,1]` with conformal-calibrated threshold `δ_t` guaranteeing false-intervention rate < α on success trajectories. |
| Action suffix corrector | `action_suffix_corrector.py` | Module 3 (suffix rewriting) | Lightweight translational nudge when action intent contradicts observed physics. Gripper channel untouched. |
| Keyframe bank | `keyframe_bank.py` | Module 4 (episodic memory) | Stores (visual feature, stage, success) from successful steps; reads back stage prototypes to augment stage inference. |

## Key design principles

1. **Task-agnostic**: No object/receptacle geom names are hard-coded. Object position, receptacle bounds, and contact info are extracted from simulator diagnostics at runtime via `extract_physics_state()`.
2. **No action rewriting as default**: The framework's default behavior is to execute VLA proposals unchanged (`validate_action` asserts `action == proposal`). The action suffix corrector is an optional safety nudge, not a trajectory rewrite.
3. **No learned world model**: Unlike CheckVLA, there is no separately trained dynamics model. Short-term prediction uses a deterministic constant-velocity heuristic.
4. **Frozen core**: `native_framework.py` is frozen during experiments. New verification logic lives in the standalone modules above and is wired into the runner via `--use-v5`.

## Archive

Files moved to `archive_redundant/` are adapters from earlier experiment tracks (assembly, fridge, checkvla official, relational transport) that are not used by the current LIBERO microwave five-layer framework. They are preserved for reference but not imported by any current runner.