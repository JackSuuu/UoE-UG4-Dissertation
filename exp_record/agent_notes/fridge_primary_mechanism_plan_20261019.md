# Refrigerator remains the primary rigid-body method-development scene

## User steering

The user explicitly asks to prioritize completing the fridge scenario concurrently
and use it to study framework improvement factors. Native-book experiments are
mechanism/actor controls, not a replacement demonstration. No successful complete
fridge episode or autonomous fridge repair benefit has yet been established.

The active parallel lanes are steered accordingly:

* GPU1 retains the frozen six-episode fridge camera diagnosis; no outcome-driven
  alteration of its calibrated pose or actor.
* GPU3 prioritizes genuinely learned fridge skills, task-aligned demonstrations and
  relevant available checkpoints in its data/actor audit. Any changed actor and
  normalization must be shared by both comparative arms and declared before tests.
* GPU2 retains its frozen native-book candidate pilot, then prepares a separately
  versioned fridge mechanism integration. No reuse of R2's `FridgeRepair.primitive`
  or `pose_action` as an autonomous candidate source. Formal fridge efficacy still
  requires actor capability qualification; exploratory failures remain useful but
  cannot be silently relabeled successful recovery.

## Whole-task evidence and factor separation

| Stage | Measured evidence to expose in video | Failure distinction |
|---|---|---|
| Open | Hinge angle/speed and contact evidence | No useful opening proposal vs rejected useful proposal |
| Grasp | Measured grasp/contact continuity | Target visibility vs proposal vs failed physical grip |
| Extract | Held-object clearance and shelf contact | Grip loss vs failure to clear enclosure |
| Place/release | Goal footprint/support, no hand contact, low motion, interval release | Native predicate vs actually supported stable release |
| Close | Door angle/speed while placement remains valid | Door skill vs disturbed placed object |
| Withdraw/dwell | Hand/object/door separation, closed door, supported release over time | Local exit vs real handoff vs complete downstream task |

Observation correction is a shared capability input, not automatically a framework
gain. Extra candidate-generation ability is separated using a simple-feedback arm
with the same model-generated candidate source. Verification needs to show useful
selection/rechecking beyond that arm, within matched information/query/action/time
caps. Handoff is counted only after an unmodified nominal model action actually
executes, and later completion/harm is reported separately. No extra model queries
or future simulator rollouts may be hidden in a candidate selector.

The final demonstration must start closed and achieve the entire sequence. Open-start
grasp/extraction tests isolate early issues and never count as full-task success.
Each actual fridge efficacy comparison must deliver matched baseline-left/framework-
right MP4s and decision-chain evidence, including failures, to GitHub.

## Read-only stage observer added

`src/adapters/fridge_stage_evidence.py` exposes measured six-stage predicates with
`satisfied / violated / unknown`, evidence age, available margins, historical
milestones and current continuous completion dwell. Missing/stale/nonfinite evidence
cannot pass a predicate. Open initialization is distinct from achieving opening;
an initially complete-looking pose cannot invent a manipulation history. Whole-
interval release evidence is required for placement; a missing interval breaks dwell.
It neither generates actions nor selects repairs nor claims handoff.

The current input adapter is the privileged R2 simulator measurement vocabulary;
some underlying geometric predicates remain scene-specific. This is an evidence
interface, not online visual grounding, a generic geometry model, a validated
candidate predictor or a formal correctness certificate. No current rollout has
been retroactively rescored as success by adding this module.

Four meaningful semantic unit tests pass: missing/stale/nonfinite input handling;
no fabricated history from a favorable initial pose; ordered sequence with open/closed
start distinction and interrupted release evidence; rejection of skipped intervals.
Synthetic fixtures are not robot demonstrations or empirical task successes.

Parallel implementation and video contract: [execution record](parallel_diagnosis_execution_20261019.md).
