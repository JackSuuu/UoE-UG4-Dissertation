# Experiment 2: shared frozen actor qualification (frozen before rollout)

Scope: model-only native-book basic competence qualification, not adaptation,
recovery/framework efficacy, or refrigerator success. Source baseline is
`89002dedbf74ac49d3383cd966d014af5b024d9b`. Run exactly initial-state indices
0, 1, 2 from the native LIBERO10 book task, fresh environments, seed 0. Index 0
was used in earlier development; indices 1/2 are fresh in this qualification.
Three cases are descriptive, not a generalization estimate.

## Frozen actor and budgets

Use `run_fridge_r1.SNAPSHOT` (LIBERO10 revision
`80970322773f81baa2e22fe495d0487b93a05cfa`), unmodified
`openvla_tf440_worker.py`, and pinned Python
`/home/ami/scratch/openvla_tf440_audit_20261019/venv/bin/python`.
Official TF4.40.1 cached greedy predict_action, batch 1, H=1; same RGB preprocessing,
native language, native camera, action denormalization and gripper conversion as
the prior audit. CUDA_VISIBLE_DEVICES=3 and MUJOCO_EGL_DEVICE_ID=3 only;
physical GPU UUID must match the prior runner. The pinned actor's cuda:0 is
physical GPU3 under visibility masking. No training, waypoint, corrective or
human motion actions. Existing benchmark initialization and 20 zero-motion
open-gripper settling calls are setup, not model actions, and are declared.
No reset or state restore after setup.

Each episode: at most 600 model queries/executed actions and 1200 seconds of
rollout wall time, checked before each query. Hard process-group timeout 1800 s
including startup; independent replay timeout 360 s. Three episodes sequential
on GPU3; supervisor max 7500 s. No alternate seed retries or dropped failures.
One query yields one action, which must exactly equal the executed proposal.

## Endpoints and continuation semantics

Continue the same model beyond native task `done`, until the existing unmodified
CompletionMonitor reports complete, the action/wall cap, or an explicit error.
Strict success requires 20 consecutive 20-Hz intervals: full book collision-box
footprint inside the rear caddy, bottom within 5 mm of floor, no gripper contact
throughout each interval, caddy support at last solver sample, at most 1 mm
book/caddy penetration, linear speed <=.02 m/s and angular speed <=.1 rad/s.
Monitor measurements never enter model input or action selection; only complete
can terminate. Native and strict endpoints are reported separately.

Inspected LIBERO `env_wrapper.py:87` delegates step without autoreset;
`bddl_base_domain.py:800-809` returns `_check_success()` as local `done` without
assigning internal self.done. Robosuite base step checks internal self.done and
sets it only at horizon. Horizon remains prior-audit 640 (>setup+600), so native
success may be followed by further actions without resetting or modifying done.
Freeze these external sources too. Dynamically assert that internal done is false
before every action; timestep increments exactly once and simulator time advances
one control interval. Log returned/native/internal done, before/after timestep/time,
and whether a model action executed after a prior returned done. Independently
replay these continuation transitions and strict endpoints. If semantics differ,
retain the error and stop that episode; do not patch terminal state.

## Freeze and evidence

Supervisor copies all imported local source dependencies, protocol and worker
before launching any episode. Execute frozen copies, validate hashes, save external
semantic sources, initialization provenance, checkpoint/build identity and commands.
Per-step preserve RGB images, processed policy image hashes, physical states,
controller/image fingerprints, model tokens/native actions/executed actions,
latencies, contact solver samples, strict checks and observable decision evidence.
Decision evidence is RGB+language -> returned tokens/action -> identity execution;
it is not hidden model reasoning. All actions have owner=vla; intervention count=0.
Fresh-process replay re-executes the action tape and recomputes both endpoints,
matching states, images, fingerprints and contacts, without re-querying the model.
Annotated actor-only video retains every source frame at 20-Hz simulation speed,
labels native vs strict endpoint, owner, action count and stop reason; end holds
are explicitly labelled and do not count as additional dwell.

## Adaptation-data audit (separate, no training authorized here)

Inspect local dataset roots, metadata and collection/training utilities. Distinguish
benchmark initialization states, checkpoint weights, scripted simulation datasets,
actor evaluation traces and actual fridge demonstrations. Report exact provenance
and availability. Do not turn handwritten motion solutions or these evaluation
traces into training labels. A feasible learned route must define independently
collected fridge demonstrations, episode-level training/development/evaluation
split, camera/action schema and normalization, then freeze a shared adapted actor
before comparisons. If no valid fridge demos/split exist, adaptation is blocked;
document collection/conversion/learning work remaining, without launching training.
