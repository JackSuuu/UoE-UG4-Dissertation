# Unified v6 root0 result — 22 Oct 2026

**Complete-task failure:** 520 queries/actions, no numeric corrections, no In
or Close attainment. Implementation/source freeze `5ab4d0f`; run directory
`/home/ami/scratch/joint_framework_root0_20261022_r1/`. All components ran in
one causal path, rather than a completed set of isolated component ablations.

## What happened

- First grasp step 73; predictive no-lift first enters repair at step 104.
- Five repair transactions enter at 104, 205, 252, 292, 357. Each uses all
  32 local actions and exits **unverified**; nominal handoffs execute at
  136, 237, 284, 324, 389. Total repair budget 160 is exhausted by step 388.
- Held-clear is first attained at 426, after the repair budget is exhausted.
  Neither original task predicate becomes true at any recorded endpoint.
- Every repair request selected acquire (160 actions). Joint stage inference
  selected transfer on 62 nominal observations but **did not change the repair
  stage relative to the measured-stage rule** in this episode. Memory-plus-dwell
  selection differs from centroid-only inference seven times; this is not a
  clean memory-effect ablation or an observed task-level benefit.
- 521 pretrained-visual/risk/prediction evaluations, 935 nonempty stage memory
  retrievals and two delayed online memory writes are recorded. Prediction is
  active on 323 observations. Risk exceeds its threshold on 252 post-action
  observations and reaches its persistence criterion on 240, but never becomes
  the selected primary joint trigger because prediction has priority. The zero
  `risk_trigger_observations` counter is **not** zero risk evaluations/exceedances.
- All 160 repair endpoints receive fusion scores, but **none meets the original
  measured local exit condition**. Thus there are zero MLP-accepted exits and
  zero otherwise-eligible exits blocked by MLP. Risk/prediction vetoes are logged
  on 128/160 repair endpoints respectively, without an eligible exit to affect.
  The failure cannot be attributed to MLP rejecting a successful recovery.

## Verification and preservation

- Executed dynamics/controller/image tape: **exact replay passed**, 520 actions,
  532 checked frames including initialization, zero mismatches, zero queries.
- Independent joint audit: **passed**, all 521 endpoint encodings, stage/memory/
  prediction/risk decisions and 160 handoff scores recomputed, zero mismatches.
  Discrete decisions match exactly; numeric comparison tolerance is 1e-5.
- [`result.json`](result.json) summarizes causal contribution separately from
  constructor/evaluation counts. Full repair records remain in the episode
  status and raw controller trace.
- `joint_root0_evidence.tar.gz` retains 1,237 files: raw state/image/action tapes,
  model and joint input PNGs, controller/query traces, initialization, frozen
  source tree, scene, plans, logs and verification reports. Archive SHA256 and
  size are in [`archive.json`](archive.json). No failed evidence was discarded.
- [`evidence_manifest.json`](evidence_manifest.json) hashes the main raw files.

This result establishes executable/recomputable joint integration, **not** a
root0 rescue, a useful risk detector, reliable learned handoff or cross-task
generalization. The immediate bottleneck is a recovery policy that fails its
measured contract and consumes its budget, not merely an absent fusion gate.
No threshold, prompt or action budget was retuned after observing this result.
