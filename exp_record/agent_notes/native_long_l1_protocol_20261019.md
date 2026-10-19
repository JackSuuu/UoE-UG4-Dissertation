# L1 frozen execution protocol — native OpenVLA long-task failure map

User authorizes execution and parallel GPU use after the consolidated goal/queue.
This cohort establishes actor competence and natural failure opportunities, not
framework benefit. Protocol and executable sources are committed before launch.

## Cohort and fixed interface

Four native LIBERO10 tasks, official initial-state indices0–5each,24episodes:

1. `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`
2. `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`
3. `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
4. `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`

Seed0; task-specific official initial states, native BDDL/geometry/camera/language.
The microwave/drawer start open, the two-pot stove starts on. Goal components are
unordered conjunctions. No new stages or mandatory object order are introduced.
Checkpoint: `openvla/openvla-7b-finetuned-libero-10` revision
`80970322773f81baa2e22fe495d0487b93a05cfa`. Official TF4.40.1 subprocess, BF16,
native greedy H1, batch1; same preprocessing/normalization/gripper action mapping
as the existing audited worker. No candidate calls or framework interventions.

Each episode:10labelled initialization no-ops; at most520model calls and actions;
1200s rollout wall budget excluding model loading;1800s outer worker cap. Stop at
native full-task success or a cap. Report native success separately from physical
release/support/stability diagnostics. Simulator pauses during inference; this
does not claim real-time asynchronous control.

## Evidence and selection rule

Record initial-file/BDDL/source hashes, initial and per-step simulator/controller/
image fingerprints, model tokens and mapped actions, causal before/after native
predicates, per-object grasp and placement-related evidence, joint/contact
diagnostics, elapsed times, states/images and full video. Replay in a fresh process
with no actor queries, checking exact physics, rendered images and diagnostics.
Do not treat replay as evidence of task competence or model re-query identity.

Meaningful partial progress for L2 eligibility means an observed target-object
grasp or newly attained object-placement goal predicate during model execution.
An initially true stove-on/closed predicate alone is not manipulation progress.
Eligibility: at least2native full successes and2native failures with meaningful
partial progress among six completed episodes. Infrastructure/replay issues are
reported separately; do not admit a task before its full six-case integrity review.
Choose eligible tasks in the listed priority order, before any framework outcomes.
Retain all24cells, including ceiling and zero-success tasks. If none qualifies,
report the complete failure map and preregister any subsequent expansion.

Indices0–5are development (drawer0–2historically consumed);6–9calibration,
10–19reserved training (unused),20–49final confirmation remain unopened. Repeated
development runs do not establish held-out gains. No actor training is scheduled.

## Parallel execution and failure handling

Physical GPUs1/2/3, with pinned UUIDs and audited CUDA/EGL selection; GPU0excluded.
Preflight discovered that EGL order is reversed on this host: physicalGPU1→EGL2,
GPU2→EGL1,GPU3→EGL0. EGL3would render on physicalGPU0. Select CUDA by UUID and
rendering by the mapped EGL index, not equal numeric identifiers. The launch
evidence retains CUDA UUID/PCI and EGL_CUDA_DEVICE_NV enumeration proofs.
Allocate roots by `gpu = 1 + init_index % 3`: eight episodes per GPU, covering all
four tasks. Each lane serializes its actor and independent replay. Initial smoke
checks and a one-query/zero-action model preflight are implementation checks outside the
cohort; any consumed development root is disclosed. Replay cap900s accounts for
cached image-tape loading; keep earlier failure artifacts if a retry is justified.

Detached supervisor and workers use persistent logs and process groups. Native
task failure advances the queue. Infrastructure/timeout/replay failure blocks that
lane pending inspection, with untouched later cells marked blocked; no hidden
automatic retries. Shared source hashes are checked before each episode.

The per-run `plan.json` freezes exact command arrays and source hashes;
`launch.json` records planned/running/replaying/verified/blocked states and PIDs.
Automatic reporting retains incomplete denominators. R0read-only RoboCasa365
readiness research proceeds independently without GPU or environment changes.

## Next decision

Use the failure map to choose early-event/candidate diagnostics (L2), then compare
frozen framework revisions with shared-candidate feedback controls (L3). Baseline
outcomes never automatically authorize a new task, actor, root split or synthetic
disturbance. All task-solving actions in this cohort come from the frozen model.
