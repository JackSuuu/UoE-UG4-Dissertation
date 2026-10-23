# Preserved architecture: current status

| Component | Implementation / causal connection | Validation boundary |
|---|---|---|
| High-level semantics | MLLM-generated goals/milestones/recovery contracts | Actual grounding accuracy pending |
| Frozen VLA | Official H1 actions for nominal and repair, no numeric edits | Real OpenVLA rollout pending |
| Visual encoding | Strict pretrained DINOv2, one encoding per image | Real checkpoint inference pending |
| Visual subgoals | Similarity to train-supported generated-predicate prototypes enters planner context | Never overrides measured truth; no fixed stages |
| Memory retrieval | Read-only roots10–19 keyframes enter recovery planning | Empty lookup is explicit; real bank pending |
| Short-term prediction | Consecutive observed-motion forecasts enter recovery planning | Heuristic, not action-conditioned/learned physics |
| Learned risk | Loaded trained head can trigger persistent intervention and veto handoff in gate mode | CPU train/save/load checked; real training/calibration pending |
| A / B | Generic trained-artifact loaders score unchanged proposals / exact nominal actions | Actual labelled data and ranking/continuation efficacy pending |
| Verifier and gate | Measured contracts, preservation, continuous rechecks, 32/160/520 budgets | CPU integration tests |
| Experiment tooling | Staged checks, microwave arms, candidate siblings, labels, memory, training, paired summaries | No independent simulator replay |

Code integration is not evidence of efficacy. Base runs without visual/head paths
explicitly record those modules absent. `full` experiment rejects missing heads
or a missing/empty memory bank; it never silently downgrades to measured-only.
Historical stage checkpoints remain preserved and are rejected by new loaders.
