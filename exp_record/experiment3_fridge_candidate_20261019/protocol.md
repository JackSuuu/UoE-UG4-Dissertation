# Fridge candidate integration v1: primary rigid-body sequence, exploratory only

Separate from the frozen failed book mechanism preflight; no book retuning.
Primary run condition is full **closed-start** fridge, seed0, three matched arms
(nominal, simple-feedback candidate sampling, candidate command-check/recheck).
An open-start condition would be a separately declared remaining-sequence run,
never full closed-start completion. No open-start run is planned in this v1 cohort.

This is integration/actor-candidate feasibility evidence, **not formal framework
efficacy**. Full-sequence actor readiness remains independently unresolved. The
same frozen official TF4.40.1 OpenVLA and batch1 GPU2 worker are used. No model
training, manually generated actions, target poses/XYZ routes, FridgeRepair,
pose_action, motion primitives, future simulator candidate rollouts, or modified
gripper fallbacks. Every executed command equals a saved model proposal.

## Shared scene, actor inputs, evidence and costs

Use R1 physically limited hinge geometry and fixed main-session camera config
`/home/ami/scratch/fridge_observation_preflight_20261019/calibration/camera_config.json`,
selected once before outcomes. Reuse `fridge_observation_experiment.make_scene`:
shared20-step settling, then identical nonintegrating camera refresh. No recalibration
or outcome-based camera search. Freeze source/config/model hashes before runs.

All arms use the same `fridge_stage_evidence.FridgeStageEvidence` observer and
`fridge_runtime.PROMPTS` mapping, including phase-conditioned language. Shared
privileged current/past geometry/contact is logged; actor sees RGB plus shared phase
prompt only. The read-only stage observer has satisfied/violated/unknown predicates,
ordered open/grasp/extract/place/close/withdraw milestones, and full sequence dwell.
Its synthetic semantic tests do not validate real-robot completion or generic geometry.

Limits for every arm:600 total model calls,600 task actions,1200s episode wall,
including every extra candidate call. Model/scene startup is outside episode wall.
No hidden baseline candidate calls. Active candidate steps request greedy plus at
most two batch1 samples, temperature.7/top_p.95, deterministic step/index seeds.
Extra candidate calls reduce possible actions. Report actual exposure and costs.

## Declared command checks and transaction mechanism

Shared event after step40: same phase for20 intervals; net item displacement<2mm,
EEF displacement<5mm and door-angle change<.01rad; current phase predicate not
satisfied. These net-motion heuristics can misclassify useful pauses/cycles.
Local cap12 executed actions and cooldown20; each extra candidate consumes a call.
Simple feedback chooses first contract-valid sample and exits on one satisfied
current entry-phase predicate. Verification chooses maximum command novelty versus
last five executed pose commands among candidates passing the declared checks,
and exits on three consecutive satisfied entry-phase predicates. Unknown/violated
resets its recheck streak. Cap exits explicitly remain unverified.

Checks genuinely available before execution: finite7-vector within[-1,1], pose-command
change<=.35 L2, and local release consistency. For a currently grasped item, an
opening proposal passes the latter only if current measured table support, full
collision-box target footprint, seated bottom within5mm, no shelf contact and
penetration<=1mm hold. **This does not require already being released or full-task
native_success**; the latter includes door closure and must not block supported
release. Missing support evidence is unknown, not permission. These are shared
current-state measurements, never generated motion routes. Future collision,
support and progress remain explicitly unknown. If all candidates fail, execute
the original contract-valid greedy with an explicit unverified-fallback label;
never synthesize motion or claim a safety guarantee.

## Completion, attribution, and review

Native done is a milestone only. Continue the same environment without reset;
internal horizon640 exceeds20 settling plus600 task actions. Stop only on the
observer's20-step released/supported/stationary placement plus closed-door/clearance
dwell after ordered milestones, or common budget. Strict stage evidence inherits
scene-specific R2 geometry and is not a certified physical verifier. Log unknowns.

Reuse actual drawer decision_trace attribution through the book preflight's
immutable trace adapter; replace subgoal/evidence/milestones with real fridge
inputs. Record before-image/state fingerprints, active stage, trigger margins,
candidate source/checks/unknown, selection reason, owner, local exit, and a separate
event only after same-step nominal handoff truly executes. These are observable
execution evidence, not internal chain-of-thought or causal proof. Factor comparisons
remain exploratory: nominal vs simple combines sampling/feedback; simple vs checked
combines selection/check/recheck policy and realized exposure, not isolated factors.

Retain all failures, states/images/actions/query costs and independent replay.
Annotated baseline-left/comparison-right videos preserve every actual pre-action
frame at common20fps simulation time with explicit held-final-frame padding; no
outcome-invented narration or held-frame dwell. GPU2 only, no edits to frozen book
files, main-owned stage observer, historical sources, shared documentation or workers.
