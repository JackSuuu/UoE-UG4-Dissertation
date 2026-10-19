# Experiment 3: frozen candidate mechanism, native book/init0 development pilot

This is an autonomous model-action exploration qualification, not full-fridge
efficacy, transfer, a validated repair, an uncertainty-aware verifier, or causal
proof. One seed and initial state, no fault injection and no outcome-based tuning.
The LIBERO10 book task uses its native camera, initial state 0, unchanged language,
and the frozen OpenVLA checkpoint from `run_fridge_r1.SNAPSHOT`. Official
Transformers 4.40.1 predict_action executes batch one on GPU2 only. The shared
actor receives RGB and language, never privileged state or handcrafted routes.

## Frozen arms and equal maximum resources

All arms have at most 600 model calls, 600 executed actions, and 1,200 seconds
episode wall time (excluding scene/model loading). Each greedy or sampled candidate
costs one call. There are no hidden baseline candidate queries or smaller nominal
cap. Extra candidate calls reduce possible actions under the common query cap;
realized exposure is reported, not represented as matched action counts.

* Nominal: one official greedy proposal per executed action.
* Simple feedback: greedy until an event; active steps query greedy plus up to two
  independently seeded samples (temperature .7, top_p .95), all sequential batch1.
  Execute the first contract-valid sample, else greedy. One measured item-progress
  observation or strict-placement candidate exits; 12 local actions also exit.
* Verification/reverification: identical event, samples, information and budgets.
  Rank candidates by novelty versus last five executed pose commands, selecting
  the largest novelty among the current command checks. Greedy is included.
  Three consecutive measured progress observations with caddy penetration <=1mm
  exit, as does the same 12-local-action cap. Exit is not task success. Failure of
  every screen explicitly falls back to unverified greedy; no scripted action.

Trigger after step40: last20-step net book displacement <2mm AND net EEF
displacement <5mm, or current native success without strict completion. This is a
stagnation heuristic (including cyclic motion), not a calibrated failure detector.
Local progress: item net displacement >=3mm from local-entry observation, or
strict-placement candidate. Transaction cooldown20, shared across feedback arms.

Pre-execution checks: finite seven-vector in [-1,1], pose-command change <=.35 L2
from the previous executed action, and no opening while currently grasping outside
native task predicate. All checks are declared command heuristics. They neither
predict collisions/support/progress nor use simulator rollouts or future labels.
No XYZ target, waypoint, motion primitive, synthesized hold, or controller repair
is generated. Every executed action must equal a saved model candidate exactly.
Both feedback arms receive the same current/past privileged positions, grasp,
native predicate, and strict completion/contact measurements. The nominal arm
also records these measurements. No future simulation informs selection.

## Completion and continuation

Reuse `CompletionMonitor`: full collision-box footprint inside caddy compartment,
bottom within5mm of floor, no gripper contact over the whole control interval,
caddy support at final solver sample, penetration<=1mm, linear<=.02m/s and
angular<=.1rad/s for20 consecutive executed20Hz steps. Native `done` is a milestone
only. Inspected LIBERO `BDDLBaseDomain.step` replaces returned done with native
success; robosuite internal `self.done` remains horizon-based. Horizon640 exceeds
20 settling plus600 task actions. Continue the same environment without resetting
or modifying its termination state; assert internal done is false each step.
All arms stop only on strict completion or common action/query/wall caps (technical
errors are retained). Standard20-step initial open/no-motion settling is shared
scene setup, not counted or claimed as model-driven task behavior.

## Evidence and replay

Reuse `libero_drawer.decision_trace` nominal/executed action attribution and delta
implementation, substituting book-specific pre-decision evidence/predicates and
gate fields. `libero_mining` was inspected; its privileged future-candidate rollouts
and manual phase actions are deliberately not used. Trace evidence is through
step-1; pre-observation hashes, threshold margins, source/actions/checks/unknowns,
choice, owner, and local exit are recorded before execution. A distinct
`handoff_executed` event is appended only after a same-step nominal action really
executes following exit. These are observable execution evidence, not model
chain-of-thought or causal proof. Offline replay reconstructs environment actions,
physics/images and gate decisions from frozen inputs without re-querying the model.

Save all query tokens/actions/latencies, all before images and states plus final,
strict outcomes, source snapshots and hashes before runs. Freeze model/code
metadata and library termination source hashes. Preserve failed exploration.
Baseline-left/framework-right and baseline-left/simple-right annotated MP4s use
the true pre-action frame with that synchronous decision, common20fps simulation
clock (simulation pauses during inference). Explicit HELD FINAL FRAME marks
padding and is never counted as physics/dwell. No post-outcome invented narration.

Unit checks cover budgets, reject/unknown semantics, three-observation exit,
local cap, distinct exit versus executed handoff, and exact candidate selection.
No commits/push; GPU2 only; no subagents.
