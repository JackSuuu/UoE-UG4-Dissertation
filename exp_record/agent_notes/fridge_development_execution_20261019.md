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

## Completion and independent replay (same day)

Both workers completed all240queries. Independent fresh-process action replay passes
exactly on all240steps of each run: integration/controller/image fingerprints,
diagnostics and saved physics states all agree, maximum state difference0. The
earlier `_c` scene smoke also replays exactly. This verifies the recorded simulation,
not correspondence to a real refrigerator.

| Initial door | Observed progress | Item/gripper contact | Native placement ever |
|---|---|---|---|
|closed|EEF displacement up to0.33473m; door angle unchanged|none at sampled control endpoints|false|
|open|EEF displacement up to0.001446m; door angle unchanged|none at sampled control endpoints|false|

Item displacement is numerical noise (<1e-12m), supported on its initial shelf
throughout. Actor median inference0.714/0.719s (closed/open). The open run emits near
zero pose increments. Inspected policy-input and terminal frames: item visible in
open view; closed trial moves above/around the fridge without opening it. Endpoint
contact sampling does not exclude transient substep contact, but no manipulation of
the door or item occurred. No stable-full-completion score was implemented or claimed.

Interpretation: first **native-policy/scene integration failure**, not evidence that
the hybrid succeeds or fails. One seed and a240-query cap cannot distinguish missing
task skill from prompt, view, handle accessibility or action-interface deficiencies.
Before any task training, check reachable handle/food poses and shared phase prompts
against the canonical full instruction; retain both variants as development. If
nominal grasp/open skills remain absent, use declared task training or a task-capable
frozen nominal model shared across all arms. Do not grow a complete scripted solver
inside the bounded repair layer and attribute its task execution to a VLA correction.

First source freeze is now commit`432bfa5`; the launch records also preserve hashes
for the then-uncommitted sources. Both complete failures and all three scene versions
are archived. [Artifact index](../../backups/fridge_r0_20261019/README.md).
