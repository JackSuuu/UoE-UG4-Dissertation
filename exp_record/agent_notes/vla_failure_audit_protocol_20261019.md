# Actor failure audit: no hand-written repair in the experiment

## User priority and exclusions

The user asks why the original VLA stalls and which part of the framework actually
caused local progress. Human-designed motion sequences are not acceptable evidence
of autonomous correction. R2 remains scene-specific scripted-repair development;
the new relational transport probe is also an engineered route and is paused,
not integrated as a claimed intelligent framework improvement.

This cohort runs **actor only**, with zero intervention actions. It changes model
inference stack or camera under specified controls. It does not tune prompts,
train models, prescribe robot routes, query simulator state for action selection,
or install state after the common benchmark initialization/settling period.

## Prior evidence motivating this audit

The LIBERO10-finetuned checkpoint is not a generally task-capable refrigerator
policy. The ten task definitions include book/caddy and cream-cheese/basket tasks,
but no refrigerator-opening/retrieval task. This is task-suite evidence, not proof
that the base model never saw a refrigerator in pretraining.

Six fixed R2 observations were exported before pinned-model inference: closed actor
steps0/200, open actor0/300, open wrapper245/400. TF4.40.1 vs existing TF5 inputs
match exactly for all six, and actions/tokens match for five. Open-wrapper400differs,
including gripper sign. Official cached and uncached TF4 both produce the same
tokens on all six; thus the difference is not explained by caching alone.
Other libraries differ too (torch/timm/tokenizers/Python), so do not assign cause
solely to transformers. Four baseline observations, including the initial near-zero
open-fridge action, agree with the official prediction path. This six-case audit
does not establish global equivalence or full simulator-version parity.

## Fixed cohort (one development initial state per condition)

| Name | Task | Camera | Inference |
|---|---|---|---|
| book_native_current | LIBERO10 book into caddy, initial index0 | Native | Existing TF5 adapter |
| book_native_pinned | Same | Native | TF4.40.1 official predict_action |
| book_side_pinned | Same | R1 robot-side pose | TF4.40.1 official predict_action |
| cheese_native_pinned | LIBERO10 cream cheese + butter into basket, index0 | Native | TF4.40.1 official predict_action |
| fridge_side_pinned | R2 seed0/open refrigerator | R2 robot-side pose | TF4.40.1 official predict_action |

The existing completed R2 `seed0_open_actor` is the current-stack fridge comparator;
assert exact initial physics/controller/both-camera fingerprint before comparing.
Book native/side conditions assert identical initial physical state/controller;
camera images intentionally differ. Book native current/pinned additionally require
identical initial images. If framing loses the target, report visibility as part
of the camera intervention rather than claiming pure coordinate-frame sensitivity.

Same600action/query cap and1200s episode wall cap excluding load; same20settling
steps with stationary open gripper. No action prefix beyond settling. Benchmark
initial state installation is only at reset. Single-case outcomes are diagnostic,
not reliable success-rate estimates, held-out generalization or repair efficacy.
Worker timeout1800s; separate process-group cleanup includes the pinned actor child.
Healthy GPUs1/2/3 only; batch1. Freeze sources/protocol before actor outcomes.

## Measurements and checks

Record processed first policy image, raw agentview images, action tokens, native and
executed actions, EEF/item positions, grasp, native predicate, action timings, physics
and controller fingerprints. For book, also use the existing released-placement20-
interval completion monitor; for cheese/fridge, native labels are diagnostic only.
Do not equate an intermediate grasp or native predicate with complete fridge success.

Independently replay actions in a fresh process with frozen sources and verify
physics/controller/images, task diagnostics and book completion. No model re-query
is implied by action replay. Publish all cases and negative results. Actor diagnostic
videos must be labeled as actor controls, not baseline/framework efficacy pairs.

Separately audit existing R2 action ownership, near-zero motion proposals, first
grasp/extraction ownership, intervention occupancy and local-exit accounting. This
is descriptive provenance: causal attribution of a verifier versus an added scripted
skill would require shared skills and matched trigger/repair ablations, which R2
does not provide. Autonomous repair claims remain deferred until a proposal source
and selection mechanism that meet the user's requirement are implemented and tested.
