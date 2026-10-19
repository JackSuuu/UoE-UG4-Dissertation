# L1 parallel execution started — 2026-10-19

**Latest state: paused after6/24episodes.** The initial cohort exited1with
`complete_with_issues`: microwave roots0–2completed with2native successes and
exact replay; two-pot roots0–2completed with1recorded native success, but wrist
image mismatch at1/1/3frames blocked each lane. Physics/integration gaps are zero.
The18remaining cells were not started. Diagnosis and a separately frozen
continuation are pending; original outcomes and failed replay proofs are retained.

**Historical startup state:** All three workers had entered actor
execution. Initial health snapshot records4model actions per worker and no worker
error; it is a launch check, not a task outcome or success-rate estimate.

* Frozen executable/protocol commit: `63fecbdbcd266a0f566949b5cac1f45a0dd5de68`.
* [Protocol](native_long_l1_protocol_20261019.md), [current goal/queue](current_goal_and_experiment_queue_20261019.md).
* Run: `/home/ami/scratch/native_long_l1_20261019_a`.
* Supervisor PID3715906; detached with `setsid --wait`, stdin closed and durable
  `/home/ami/scratch/native_long_l1_20261019_a.supervisor.log`.
*24planned actor-only episodes: four tasks × initial indices0–5. No framework
  intervention, new weights or training in this screening.

| Physical GPU | CUDA selection | EGL device | Initial worker | Lane |
|---|---|---:|---:|---|
|1|UUID ending `abcc6c0330b0`|2|3715914|roots0/3across all four tasks,8episodes|
|2|UUID ending `426672ebc1a8`|1|3715913|roots1/4across all four tasks,8episodes|
|3|UUID ending `5c98029c2a2b`|0|3715915|roots2/5across all four tasks,8episodes|

Initial actor episodes are microwave roots0/1/2. Subsequent jobs are serialized
within each lane, with a fresh-process replay after each episode. Native task
failure continues the queue; an infrastructure or replay issue blocks that lane
and preserves the remaining planned cells for inspection. Successful startup does
not establish a completed rollout or exact actor-action replay yet.

## Prelaunch evidence

* Four native root0scene checks and their independent replays passed exactly:
 10initialization noops per scene, no task-solving actions or actor calls.
  Original initialization40steps; replay initialization40steps, separately counted.
* One official model query on the saved microwave observation passed7token/action
  and gripper mapping checks; zero robot actions. Model load8.67s, first-query wall
 2.46s (model inference0.69s); these are preflight timings, not cohort averages.
*13synthetic summary cases passed, including appliance-only false progress,
  failed/missing replay, malformed/incomplete tapes and non-simultaneous goals.
* Supervisor orchestration was exercised with synthetic successful jobs, a failed
  replay process, false replay proof and report failure. These checks made no
  robot actions and are labelled synthetic; final launch-plan validation checks
  all24identities, command arrays, source hashes and GPU mappings.
* The existing checkpoint's16files (15,085,050,947bytes) were SHA256 inventoried.
  The actor remains the pinned official TF4.40.1 worker with native greedyH1.

### GPU provenance correction

EGL_CUDA_DEVICE_NV enumeration joined to unmasked CUDA UUID/PCI identities shows
the graphics order is reversed on this host. Equal CUDA/EGL numeric identifiers
do **not** select the same physical GPU. PhysicalGPU0is EGL3and is excluded here.
CUDA selection by UUID was separately verified in the pinned actor interpreter.
The initial health check shows onlyGPUs1/2/3allocated (~15.6GiB each), GPU0idle.
This corrects device selection for the current cohort; it does not retrospectively
attribute historical task failures to rendering placement.

[Launch/preflight archive and initial health snapshot](../../backups/native_long_l1_launch_20261019/).
The annual-plan user edit retains SHA256
`d4617d1c65e2ef2ecbad3cb11c19753e0e2b9bbbafafeebeb5d7ccb57d55c83b`.

## Reporting and next experiment

`launch.json` is the durable status board. Episode folders retain raw videos,
query/action logs, states, both camera tapes, contacts and predicate transitions.
After the queue finishes, the supervisor writes `report/summary.json` and
`report/results.md`, retaining all24planned cells and separating incomplete,
infrastructure, replay and native-outcome states. Only six completed/integrity-
checked cases can qualify a task. Freeze L2early-event/model-candidate diagnostics
from that failure map before any new repair comparison; do not hot-edit this
cohort's frozen sources while jobs remain active.

The independent [R0readiness audit](robocasa365_readiness_20261019.md) completed
source research and identified public composite-capable checkpoint candidates.
It made zero GPU/model calls, installations or weight downloads. Local RoboCasa
qualification and a runnable public CheckVLA baseline remain unverified.
