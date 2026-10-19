# R0: first refrigerator manipulation development probe

## Scope and sequence

Design: [two-demo experiment design](two_demo_experiment_design_20261019.md).
Source: `scripts/probe_fridge_task.py` and `configs/tasks/fridge_retrieval_dev.bddl`.
This is a new custom task, not an official LIBERO benchmark task or confirmation
cohort. No hybrid controller exists for this task yet. All attempts remain visible.

1. `/home/ami/scratch/fridge_scene_20261019_a`: seed0, closed door, reset +20settle
   intervals +2no-op intervals. Item supported by the fridge shelf; native placement
   false and item-in-fridge true. Camera saw mainly the side; retained original image.
2. `_b`: same layout, open door. Same item support; opening alone is not task success.
3. `_c`: move fridge y0.25→0.15m and yaw0→pi/2 so the camera sees the interior and
   shelf item. Same open-door settle/2step check. This is development scene design,
   before actor outcomes, not post-hoc selection of successful trajectories.

All three retained the original asset's hinge `limited=false`. This is not a
validated refrigerator joint model. Retain it in this first diagnostic so issues are
observed, then version a physical-limit/closure-predicate correction before comparison.
The native `Close` predicate uses angle<0; it is not the planned5-degree/dwell endpoint.
No native boolean will be reported as complete stable task success in this probe.

## Native-actor launch

Output: `/home/ami/scratch/fridge_actor_20261019_a`.
Two detached `start_new_session=True` workers, each externally bounded to900s:

| Initialization | GPU | Launcher PID | Horizon | Seed |
|---|---:|---:|---:|---:|
|closed|1|3620136|240actor queries|0|
|open (extraction diagnostic)|2|3620137|240actor queries|0|

Same scene/placement/instruction/normalization and frozen full7B LIBERO10 checkpoint
`80970322773f81baa2e22fe495d0487b93a05cfa`. NativeH1, actual images every query, no
scripted grasp or transport. Twenty empty-hand settle intervals precede inference.
All native predicates, physical states, contact endpoints, door angle, action tokens,
integration/controller/image fingerprints and inference time are recorded. Raw
single-arm videos are retained as source evidence; no synthetic hybrid panel.
The world pauses during inference; this is not a real-time experiment.

Source and BDDL copies are in `source/`; per-run `status.json` carries source hashes
and actor build, `steps.jsonl` actions/diagnostics, `states.npz` states, `model.xml`,
initial/policy images and `raw_baseline.mp4`. Source hashes take precedence over the
pre-change Git HEAD reported during this uncommitted development launch.

At launch: no outcome or full success rate available. Final diagnosis and independent
action replay will be appended after completion. Open/closed trials cannot be pooled
into a refrigerator efficacy percentage. These runs consume development exposure.
