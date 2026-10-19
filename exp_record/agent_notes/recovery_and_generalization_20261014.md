# Recovery snapshot and attribution clarification — 14 Oct 2026

## User request and delivered structure

The user requested GitHub-linked recovery of important scratch artifacts/videos and LIBERO, and asked whether worker 2 already demonstrates physics repair/generalization.

The repository now includes a pinned `third_party/LIBERO` git submodule and a `backups/` recovery entry point. Large archives/videos use Git LFS rather than Git history blobs or server-only links. The recovery script records SHA256 and byte counts for each archive and every original member, checks source stability during creation, and restores into a new directory. It preserves all selected scratch/result artifacts, including negative/historical experiments, not only the visually best episode. Full upstream pretrained weights are referenced by immutable HF revisions rather than mirrored; the disposable probe environment and pre-existing uncommitted notes are excluded explicitly.

**Remote verification passed:** commit `f33915a` was pushed with all14LFS objects (2,449,423,402bytes). A fresh SSH GitHub clone, independent LFS download, `git lfs fsck`, full extraction and SHA256 checks recovered all1,674source files,10archives and4standalone videos. The submodule was independently cloned and its revision/task-initial-state hash matched. [Machine-readable verification](../../backups/remote_recovery_verified_20261014.json). This verifies remote artifact recovery, not full pretrained-weight mirroring or binary environment reproduction.

## What actually controls worker 2

`scripts/run_libero_continuation.py` calls `actor.act(...)` then directly `env.step(action.tolist())`. `CompletionMonitor` and `ContactRecorder` measure the resulting execution; neither selects nor changes the action. The monitor can terminate the episode after the registered complete-placement dwell, but it is not a physics-prediction repair controller. All workers use the same frozen checkpoint with different initial roots. Worker 2's visible readjustments are VLA actions, and its transient placement is later disturbed by re-contact. It does not meet the complete acceptance definition.

The long-term framework includes physical prediction and bounded correction, and legacy planar experiments did evaluate such components. That must not be conflated with their deployment in this new 7D robot pilot. Being executed in MuJoCo does not itself mean physical prediction is embedded in the policy loop.

## Generalization and task order

1. **Mechanism first, one task:** on the current development task, compare unchanged VLA with a matched simple phase/geometric controller and an exact-physics lookahead selector. Change one algorithmic component at a time in response to failures; preserve full continuation/completion and correction-budget controls. No new training/intervention run is launched as part of the backup.
2. **Within-task generalization:** preregister untouched initial-state roots and held-out observation/actuation/physical disturbance conditions; separate train/dev/calibration/test by root, including all branches/perturbations from that root. Shared disturbance schedules support paired comparisons. Freeze thresholds and method before test, report rescue/harm and uncertainty. Root 0/1/2 remain development-only. CVR and safe completion await justified physical limits and stable contact measurements.
3. **Cross-task transfer:** after useful effect is established and the method frozen, test a second manipulation task with a capable frozen task policy and the same framework/interface and budget. Retuning task-specific correction rules or training on the new task changes the claim to adaptation. Report zero-shot transfer separately from adaptation. Two tasks support evidence on those tasks, not unrestricted manipulation generalization.

Do not simultaneously expand scene count, model size and training data to chase a positive result. One task is suitable for mechanism development; its three selected development videos are insufficient for either generalization or framework efficacy. Genuine H10/k5 compatibility remains unresolved before formal efficacy under the historical protocol.
