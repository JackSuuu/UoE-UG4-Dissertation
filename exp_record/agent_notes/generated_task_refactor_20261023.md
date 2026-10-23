# Generated-task runtime refactor — 23 October 2026

## Confirmed design

The user supplies natural-language task instructions. An external vision-capable
MLLM generates grounded task goals and context-dependent recovery contracts.
Per-task Python adapters and hand-authored task configurations are superseded.
The model proposes semantics; sensor/benchmark measurements verify them.

## Implemented

- Replaced the active controller with a task-independent transaction state machine.
  Moved the old fixed-stage controller to `src/legacy/microwave_controller.py`
  and redirected historical experiment/training/test imports.
- Added validated `TaskSpec` / `RecoveryPlan`, a real multimodal chat client,
  environment-neutral observations, a shared LIBERO measurement bridge, official
  frozen OpenVLA H1 backend, and a sole query/execution owner (`Session`).
- Added `scripts/run_task.py` and exact-input/action/planner-response journaling.
  The main loop always uses `PhysicalVerifier`; no task constants/stages are
  embedded in the new controller. The discarded task-config stubs were removed.
- Removed `--use-v5` and the live numeric-action-correction path from the old
  comparison runner. Archived replay handling and original experiment artifacts
  remain historical evidence.
- Kept four physical feature values; added validity/provenance as separate
  metadata. Candidate actions are not compared to past motion as if that were
  their predicted effect. Main-loop alignment uses the actual last executed
  action and its corresponding end-effector endpoints.
- A ranks unchanged proposals from one first-repair H1 pool when supplied.
  B consumes the fresh original-task nominal action; a rejected proposal consumes
  a query but never executes. Without learned scorers the runtime explicitly
  reports greedy/measured-only behavior.
- Preserved 32 local / 160 total repair / 520 episode-action defaults. All policy
  queries count within the 520-query cap; planner calls have a separate budget.

## CPU validation

`python -B scripts/test_task_runtime.py`: **30 tests passed** using standard-library
fixtures/mocked model transport. Includes multi-task semantics, missing/negated
evidence, release/regrasp, fixed alternatives, preservation, cap ties, A/B query
accounting, actual handoff, post-handoff regression and a simulated LIBERO API.

Historical suite with `/home/jack/miniconda3/bin/python`, `PYTHONPATH=scripts:.`:
**95 tests passed** across native, joint, fusion, candidate, repair-progress,
data-expansion, fridge, action-validation and oracle-recovery tests. The system
Python lacks NumPy, so historical tests use the existing conda interpreter.

Both CLI help commands and source syntax/import checks pass. No GPU rollout,
model download, checkpoint training or experimental efficacy result was produced.

## Remaining research work

The real MLLM's grounding/semantic accuracy needs validation. Generic monitoring
currently uses generated progress loss and stagnation, not the historical
microwave-specific contact rules. The new runner has no independent simulator
tape replay yet. Old A/B features/labels are incompatible with task/contract and
nominal-action-conditioned scoring; new datasets and model validation are needed.
No claim is made that current local verification predicts downstream success.

See `docs/task_runtime.md` for interfaces, GPU-machine invocation, data splits and
packaging direction. Packaging should expose the existing single implementation.
