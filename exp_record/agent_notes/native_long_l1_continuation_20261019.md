# L1 continuation after wrist-rendering diagnosis

## Reviewed decision

Continue only the18untouched cells from original cohort
`/home/ami/scratch/native_long_l1_20261019_a`, in new directory
`/home/ami/scratch/native_long_l1_20261019_b`. The six original actor trajectories
and all failed strict replay proofs remain referenced, immutable, and unrerun.
The hash-bound [decision](native_long_l1_continuation_decision_20261019.json)
pins the prior plan/status, continuation script and updated report reader.

The source of the pause has been investigated:11fresh zero-query replays compared
5,752frames. Physical/integration/controller state, contacts, diagnostics and
agentview RGB remain exact. Captured differences affect1–5wrist pixels with
maximum channel delta1. At root0frame482,32repeated renders at identical sensor
time produce three image hashes;3recover the original image. This directly
demonstrates renderer nondeterminism. The NVIDIA/MSAA implementation cause is
not isolated. Sensor samples precede saved endpoint state by2ms; endpoint-only
redraws must not be conflated with sensor-time comparisons.

All diagnostic runs used physicalGPU1/EGL2. Root1/2comparisons are cross-device,
and cannot certify their originalGPU2/3lanes. Two untouched strict root0repeats
both failed. No repeated pass replaces an earlier failure. Original replay
images were not saved, so measured deltas refer to new diagnostic replays.

## Continuation contract

* Original actor, task, roots, instructions, observations, renderer and query/action
  budgets remain frozen. Three lanes use the audited UUID/EGL map, six new cases
  each. First new cases are two-mug roots0/1/2.
* Strict byte-exact replay remains the gate for every new episode. Any new issue
  blocks that lane for inspection. No tolerance or revised selection rule is
  introduced by this continuation.
* The combined report always includes all24planned identities and reads the six
  original episodes through explicit absolute paths. Reporter support for those
  paths is the only amended original source; an actual-data regression confirmed
  identical original episode/task/selection results, and the combined preflight
  confirms6completed,3replay issues,18unstarted.
* Even if all18new cells pass, the combined overall status remains
  `complete_with_issues` until a separately declared integrity protocol resolves
  the original issues. The current frozen eligibility rule is unchanged.
* Detached supervision, unique outputs, bounded workers and source checks remain
  enforced. No task or method is selected from the partial outcomes.

## Result-driven implications

[Partial analysis](native_long_l1_partial_results_20261019.md) finds successful
microwave traces also wait156–158actions from first grasp before a20mm rise.
Therefore early no-lift alarms would misclassify success controls. Two-pot failures
retain first-object placement but fail to acquire/transfer the unfinished object.
These are causal-prefix diagnostic hypotheses for L2, not proven repair methods.

Evidence directory: `/home/ami/scratch/native_long_replay_diagnosis_20261019`.
The source, zero-query proofs, exact comparisons, original image checks and
sensor-time redraws will be archived alongside the continuation decision.
