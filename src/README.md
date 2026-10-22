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
│   └── vision_subgoal_inference.py
├── verification/                 # Physical evidence, fusion and risk scoring
│   ├── physical_verifier.py
│   ├── fusion_mlp.py
│   ├── calibrated_risk_head.py
│   └── vision_physical_monitor.py
├── prediction/                   # Short-horizon motion/progress diagnostics
│   └── short_term_predictor.py
├── repair/                       # Bounded action-correction prototype
│   └── action_suffix_corrector.py
├── memory/                       # Episodic keyframes and prototype retrieval
│   └── keyframe_bank.py
├── training/                     # Offline training, not runtime control
│   └── train_fusion_mlp.py
├── checkpoints/                  # Weight artifacts, separated from source
│   └── fusion_mlp.pt
└── results/                      # Existing output directory (left untouched)
```

## Entry points and imports

- Run/prepare/replay: [`../scripts/run_framework_comparison.py`](../scripts/run_framework_comparison.py).
- Controller: `from src.controller import NativeFramework` (class API unchanged).
- Components use `src.perception`, `src.verification`, `src.prediction`,
  `src.repair` and `src.memory`; package initializers do not load model weights.
- CPU checks: `python -B scripts/test_native_framework.py` and
  `python -B scripts/test_action_correction.py`, from the repository root.
- Offline training: `python -m src.training.train_fusion_mlp --help`.
  Weights now live at `src/checkpoints/fusion_mlp.pt`; training methodology
  limitations are recorded in [COMPONENT_STATUS.md](COMPONENT_STATUS.md).

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
