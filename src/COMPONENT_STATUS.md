# Framework components: implementation status

The objective is cross-task visual–physical verification, bounded repair,
reverification and reliable handoff around frozen OpenVLA. Current validation
is limited to development microwave trajectories; generalization is untested.

## Active controller

- `controller.py`: measured task stages, triggers, budgets and handoff.
  This remains the actual controller, not visual stage inference or MLP gating.
- `adapters/libero_openvla.py`, `adapters/tf5_compat.py`: actor/image and compatibility utilities.
- Canonical runner: `../scripts/run_framework_comparison.py`.

## Experimental components (`--use-v5`)

| File | Current status |
| --- | --- |
| `verification/physical_verifier.py` | Diagnostic-based heuristics, not scene reconstruction or forward physics simulation. Positions and grasp evidence are bound to an explicit task target. Missing bounds/force remain missing. |
| `perception/visual_grounding.py` | Encoder interface with random-projection fallback. No verified real DINOv2/V-JEPA checkpoint or registered goal prototypes in current runs. |
| `perception/vision_subgoal_inference.py` | Prototype interface, not used to replace native stage decisions. |
| `prediction/short_term_predictor.py` | Heuristic extrapolation and observed no-lift detection; not a learned world model. Action-direction blending is not physically calibrated. |
| `verification/calibrated_risk_head.py` | Runner creates an untrained head with no threshold. No episode-level false-intervention guarantee. |
| `verification/fusion_mlp.py`, `checkpoints/fusion_mlp.pt`, `training/train_fusion_mlp.py` | Advisory classifier trained with zero visual features and weak owner-based labels; no held-out validation. Not a reliable handoff probability. |
| `repair/action_suffix_corrector.py` | Hand-coded normalized translation offsets, not genuine VLA suffix regeneration. Rotation and gripper remain unchanged. |
| `memory/keyframe_bank.py` | Storage/read APIs; successful-memory retrieval is not causally integrated into control. |
| `verification/vision_physical_monitor.py` | Evidence aggregation; its triggers do not themselves enter native repair mode. No-lift evidence can activate the runner's separate numeric correction. |

Default actions must equal official VLA proposals. The opt-in
`bounded_translation_v1` diagnostic treatment separately records proposal,
executed action, reason and delta. Validation preserves official proposal
mapping, allows only translation changes with L2 delta at most 0.10 normalized
action units, and requires explicit caller permission. Corrections have a
separate 160-action total cap and 32-consecutive-action cap; native repair
budgets remain separate. This is not the old unchanged-action experiment.
Numeric edits count toward the unchanged 520 action/query limits.

Tape replay checks executed dynamics, images and native controller decisions;
it does **not** recompute the experimental monitor/MLP evidence. A replay pass
is therefore not evidence that the learned modules are reproducible or useful.

Phase 1 and obsolete adapters were deleted, not moved into an archive directory.
See the [historical source link](README.md#deleted-phase-1-code).
