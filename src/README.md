# Source layout

Components are grouped by responsibility, not collected in `adapters/`.
The objective is a cross-task verifier–repair framework around frozen OpenVLA;
the measured controller is currently microwave-specific and the visual–physical
extensions remain experimental. See [implementation status](COMPONENT_STATUS.md).

```text
src/
├── controller.py                 # Stage tracking, triggers, budgets, repair/handoff
├── adapters/                     # External model/runtime integration only
│   ├── libero_openvla.py
│   └── tf5_compat.py
├── perception/                   # Visual features and subgoal inference
│   ├── visual_grounding.py
│   ├── frozen_visual_encoder.py    # Strict pretrained DINOv2 path for fusion
│   └── vision_subgoal_inference.py
├── verification/                 # Physical evidence, fusion and risk scoring
│   ├── physical_verifier.py
│   ├── fusion_mlp.py
│   ├── handoff_verifier.py
│   ├── calibrated_risk_head.py
│   └── vision_physical_monitor.py
├── prediction/                   # Short-horizon motion/progress diagnostics
│   └── short_term_predictor.py
├── repair/                       # Bounded action-correction prototype
│   └── action_suffix_corrector.py
├── memory/                       # Episodic keyframes and prototype retrieval
│   └── keyframe_bank.py
├── training/                     # Offline training, not runtime control
│   ├── handoff_dataset.py
│   └── train_fusion_mlp.py
├── checkpoints/                  # Weight artifacts, separated from source
│   ├── fusion_mlp.pt               # Legacy weights; new loader rejects these
│   └── fusion_handoff_v2_20261022/ # Outcome-trained weights + report/manifest
└── results/                      # Existing output directory (left untouched)
```

## Entry points and imports

- Run/prepare/replay: [`../scripts/run_framework_comparison.py`](../scripts/run_framework_comparison.py).
- Controller: `from src.controller import NativeFramework` (class API unchanged).
- Components use `src.perception`, `src.verification`, `src.prediction`,
  `src.repair` and `src.memory`; package initializers do not load model weights.
- CPU checks: `python -B scripts/test_native_framework.py` and
  `python -B scripts/test_action_correction.py`, from the repository root.
- Fusion checks: `python -B scripts/test_fusion_handoff.py`.
- Offline training: `python -m src.training.train_fusion_mlp --help`.
  New weights and their validation report live under
  `src/checkpoints/fusion_handoff_v2_20261022/`. Legacy weights are retained separately.
  See [training evidence](../backups/fusion_handoff_20261022/README.md).

For the matched off/log/gate experiment, prepare a new canonical plan with
`--prepare NEW_PLAN --mlp-path src/checkpoints/fusion_handoff_v2_20261022/fusion_mlp_v2.pt`.
Then run `scripts/run_fusion_handoff_ablation.py --plan NEW_PLAN/plan.json --out NEW_RUN`.
It launches one actor at a time on the recorded root0 GPU3 lane, with identical
520-action/query limits and bounded-correction treatment in all three arms.
`gate` only vetoes verified exits; unchanged local/total caps can still force
an explicitly **unverified** handoff. No unbounded wait or extra action budget.

## Joint framework experiment

The unified v6 path connects **real visual stage inference + physical contracts
+ short-term prediction + trained risk + episodic retrieval + policy-generated
repair + joint handoff** in one run, rather than loading unused interfaces.

```bash
python -B scripts/run_framework_comparison.py --prepare NEW_JOINT_PLAN \
  --mlp-path src/checkpoints/fusion_handoff_v2_20261022/fusion_mlp_v2.pt \
  --joint-components src/checkpoints/joint_components_v1_20261022/joint_components.pt
python -B scripts/run_joint_framework.py --plan NEW_JOINT_PLAN/plan.json --out NEW_JOINT_RUN
```

The supervisor selects only root0, uses one actor on physical GPU3, then runs
tape replay and independent joint-model recomputation. The other plan cells are
not scheduled. The new mode disables the legacy untrained/random monitor and
numeric offsets. H1 repair is generated step by step by frozen OpenVLA, not a
pretended native multi-action chunk. See [artifact limits and measured risk
performance](checkpoints/joint_components_v1_20261022/README.md).

`native_framework.py` was moved to `controller.py` without changing controller
logic during the layout migration. Old import-path aliases are not retained.
Prepare a fresh source-pinned plan after migration; old plans/artifacts retain
their original paths and should use their archived source tree, not be rewritten.

## Deleted Phase 1 code

Browse the exact pre-deletion tree:
**[Phase 1 source at `72bd689`](https://github.com/JackSuuu/UoE-UG4-Dissertation/tree/72bd6892d01d2c148b8953d00bff40d951065dcf/src)**.
Deletion commit: [`d0438e4`](https://github.com/JackSuuu/UoE-UG4-Dissertation/commit/d0438e4).

This includes the old `experiments/`, `models/`, `sims/`, `systems/`, `tests/`,
`vla/`, `demo/`, `checkvla/`, and Phase 1 core files. Historical scripts importing
removed modules require that historical tree; the current refactor does not
restore those obsolete dependencies.

To inspect it without overwriting current work:

```bash
git worktree add --detach ../dissertation-phase1 72bd6892d01d2c148b8953d00bff40d951065dcf
```
