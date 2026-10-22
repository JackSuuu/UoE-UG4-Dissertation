# Framework components: implementation status

The objective is cross-task visual–physical verification, bounded repair,
reverification and reliable handoff around frozen OpenVLA. Current validation
is limited to development microwave trajectories; generalization is untested.

## Active controller

- `controller.py`: measured task stages, triggers, budgets and handoff.
  This remains the actual stage/repair controller. Optional `fusion_mode='gate'`
  additionally vetoes measured verified exits; `log` has no decision effect.
- `adapters/libero_openvla.py`, `adapters/tf5_compat.py`: actor/image and compatibility utilities.
- Canonical runner: `../scripts/run_framework_comparison.py`.

## Experimental components (`--use-v5`)

| File | Current status |
| --- | --- |
| `verification/physical_verifier.py` | Diagnostic-based heuristics, not scene reconstruction or forward physics simulation. Positions and grasp evidence are bound to an explicit task target. Missing bounds/force remain missing. |
| `perception/visual_grounding.py` | Encoder interface with random-projection fallback. No verified real DINOv2/V-JEPA checkpoint or registered goal prototypes in current runs. |
| `perception/frozen_visual_encoder.py` | Separate strict pretrained DINOv2 ViT-S/14 path for the trained fusion verifier. Real 384-D policy-RGB features; exact checkpoint/source/preprocessing provenance; no random fallback. Does not yet replace stage prototypes or the legacy monitor visual path. |
| `perception/vision_subgoal_inference.py` | Prototype interface, not used to replace native stage decisions. |
| `prediction/short_term_predictor.py` | Heuristic extrapolation and observed no-lift detection; not a learned world model. Action-direction blending is not physically calibrated. |
| `verification/calibrated_risk_head.py` | Runner creates an untrained head with no threshold. No episode-level false-intervention guarantee. |
| `verification/fusion_mlp.py`, `training/handoff_dataset.py`, `training/train_fusion_mlp.py` | Version-2 real visual + measured state/missing-mask + recovery-contract classifier. Class 1 means pass; labels require current contract and 12 observed nominal actions without regression, or native completion. Complete initial-state roots are isolated across train/validation/evaluation. Legacy `checkpoints/fusion_mlp.pt` is retained but rejected by the new loader. |
| `verification/handoff_verifier.py`, `checkpoints/fusion_handoff_v2_20261022/` | Trained artifact and causal scoring. Threshold selected on current-contract-true validation samples, not easy current-false negatives. Off/log/gate experimental ablation available through canonical runner. Validation has high false rejection and poor actual-handoff acceptance: not enabled by default and not an established efficacy gain. Unsupported close stage rejects rather than fabricating a trained score; native completion and unverified budget exits retain priority. |
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
`scripts/run_fusion_handoff_ablation.py --audit EPISODE` separately recomputes
trained fusion scores from saved causal policy images and measured endpoints;
it reports probability tolerance (1e-5) and any terminal endpoint without a saved
next policy PNG. Neither this audit nor an empirical threshold is a safety guarantee.

Phase 1 and obsolete adapters were deleted, not moved into an archive directory.
See the [historical source link](README.md#deleted-phase-1-code).
