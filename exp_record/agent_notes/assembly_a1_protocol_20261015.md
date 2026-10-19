# A1: frozen OpenVLA + learned physical prediction + CheckVLA-style verification

Status: pre-launch development protocol. Frozen by source/protocol hashes before
launch. This supersedes the prior no-new-training instruction following the user's
explicit request to start assembly experiments with our framework. M1 remains stopped
and unchanged, including every sham harm and unresolved replay issue.

## Question and implementation identity

Can our learned physical predictor and verification-based local action selection
help a frozen visual actor finish an insertion after contact misalignment?

The five layers remain: subgoal (approach / align-and-insert / release-and-clear),
frozen full7B OpenVLA, risk/event monitor, nominal/contact-transition modes, bounded
local correction. `continue/hold/repair` are routes, not the two modes.

**Actual code:** new `AssemblyDynamics` subclasses the repository's self-developed
`OrbiSimDynamics` MLP ensemble (3 members, hidden128), with genuinely autoregressive
state AND risk evaluation. Historical push risk instead re-read the initial state;
that source is preserved. `RefCheckVLA.check/score` actually gates and scores all
candidate chunks. Its old push-specific scaling repair is not used: the assembly
repair selects among the eight documented primitives. These are our implementations,
not official OrbiSim or CheckVLA reproductions. Official adapters remain templates.
Author page https://jjleejj85.github.io/projects/orbisim checked 15 Oct 2026: "Code
coming soon." MuJoCo supplies data and evaluation, never online candidate forecasts.

## Task and outcome specification

Stock robosuite1.4.1 NutAssemblySquare/Panda, OSC_POSE, 20Hz simulated control and
2ms physics. A1 rendering disables shadows, reflections and GL dithering uniformly
before the setup and all actor observations. This pre-launch appearance setting
passed exact replay; default-renderer fixtures had bit-exact physical state but
one-channel/one-level image discrepancies. Their cause is not uniquely isolated;
this does not explain or resolve M1's historical mismatch. No image tolerance was
introduced. Real physical grasp of the handle, lift, approach; no weld or teleport.
Initial nut yaw pi keeps its handle facing the robot. Fixed setup320 actions is
evaluator preparation, not OpenVLA success. Setup errors are infrastructure failures.
Task scope starts with a grasped nut above the peg; not full pick-and-insert efficacy.

Nut opening half-width22.75mm, peg half-width16mm => nominal clearance6.75mm.
Development cases: aligned, world-x ±12mm, world-y ±12mm entry displacement.
No outcome-conditioned filtering. Two assembly seeds3000/3001 ×5 offsets ×3 arms
=30 continuations, each at most160 fresh-image actor queries/intervals. New seed
namespace; no LIBERO final roots or M1 continuation consumed.

Completion: square peg cross-section contained in nut opening in local coordinates,
nut center z<0.84m (table0.82m), tilt<10deg, no gripper/nut contact, object linear
speed<0.01m/s, for5 consecutive intervals. Native success, seated-only success,
complete dwell and failures are separately reported. These fixed thresholds are
development task specifications, not certified physical safety limits. No breakage
model or validated collision/force limit exists. Report **nut/peg summed contact
force magnitude**, peak across all internal substeps, not net wrench or CVR. Raw
10N load/15deg tilt normalizations are engineering development scales, not safety
claims. Paired rescue, harm, both failure and both success use the full endpoint.

## Learning and calibration

Train seeds1000–1023, calibration2000–2007, validation2100–2107. Each: independently
sample x/y offset uniform[-18,18]mm,160 transitions; initial45-step insertion then
5-step random pulses (hold/retract/lateral/insert/alignment), final30 steps open and
retract. The distribution and split are fixed before outcome inspection. Save every
setup, transition and hash. No online episode goes into training.

Privileged40D input: nut-minus-peg xyz, eef-minus-nut xyz, nut orientation matrix,
robot joint positions/velocities, finger joint positions/velocities, nut linear and
angular velocity, current nut/peg load. Previous state/action are also inputs.
Targets: next state; interval peak nut/peg load /10N and next tilt /15deg.
Train1800 AdamW steps, lr3e-4, minibatch8, horizon3 autoregressive smooth-L1 state
and risk losses; normalized state deltas, gradient clip5. Seed17, no validation-based
checkpoint selection. Freeze weights before any live comparison. Report horizon-wise
position and risk MAE. Tau is95% empirical calibration quantile on windows whose
observed normalized risk<=1; at least20 eligible windows required. Overlapping
windows are NOT independent conformal units; no finite-sample FAR guarantee claimed.

## Actual control participation and matched arms

Native OpenVLA H1, current LIBERO10 checkpoint (not assembly-finetuned). Its ability
here is unestablished and is part of this development probe. Camera is stock
robosuite agentview: compensate horizontal flip before existing LIBERO180deg image
processing, yielding upright vertical-flipped raw image; save actual actor input.
Same actor, sensing, setup, horizons, candidate and correction budget for all arms.

Every interval forecasts8 candidates ×3 steps. Candidates: nominal, hold-grasp,
retract+z .15, ±x .15, ±y .15, insert-z .10, in normalized OSC units. Each candidate
is a one-step pulse followed by two zero-pose settling actions; nominal keeps actor
gripper value, repair candidates keep grip closed. Execute only the first action;
these are NOT actor chunks. Every subsequent interval queries a fresh actual image.

Contact mode: nut-center height over peg origin<.13m OR measured load>.1N.
CheckVLA score=max over horizon/channels(mean risk + ensemble std). Trigger in
contact mode if score>tau OR three consecutive contact/downward/no-progress
intervals (selected z<-.03, load>.1N, height reduction<.5mm).

* `vla`: no changes; shadow model/monitor still runs.
* `framework`: trigger permits the candidate minimizing predicted terminal lateral
  error/.02 + absolute seated-height error/.12 + .5*CheckVLA score. It changes action
  only if another candidate beats nominal objective by>.01. Maximum40 changed
  intervals. The gate and ranking both consume actual learned predictions.
* `feedback`: same learned trigger, candidates, shadow computations and40-change
  cap; chooses retract if current load>10N, else largest-error-axis alignment if
  lateral error>4mm, else insert. This is a **ranking ablation/shared-trigger
  comparator**, not a completely predictor-free baseline or capable general planner.

Handoff after each pulse is a fresh actor query.16-event memory records prediction
residual, measured load and stalled progress. Residual is diagnostic, not an
additional claimed gating contribution. Log subgoal/mode/evidence/gate, all candidate
predictions/scores/objectives, nominal/selected/effective, budget, observed outcome,
handoff and remaining completion. Report actual changed-action count, not only flags.
Prediction can be inaccurate or ineffective; full framework may harm completion.

## Integrity, compute and reporting

Healthy GPUs1/2/3 only, numerical CUDA/EGL IDs matched. Collection/training on2;
live arms VLA1/framework2/feedback3, process-isolated episodes and process-isolated
exact action/image/state/force/outcome replay. Same shadow forecasts in all arms;
query count/control budget matched, no real-time/equal-wallclock claim. Measure actor
and decision latency separately;20Hz simulated control does not imply20Hz inference.

Supervisor survives SSH termination. Every registered case runs regardless of VLA
success. Integrity/setup/process errors stop scheduling, preserve partial data and
are NOT task failures. A finished worker may complete its current episode when a
peer fails. Incomplete trajectories stay unresolved. No automatic fallback to a
scripted nominal policy. No online MuJoCo lookahead substituted for the learned model.
Annotated2x videos and exact replay proofs accompany completed episodes. Full raw
archive and independent remote recovery follow completion; source/preflight/protocol
are backed up before launch. This is development, not held-out confirmation or proof
of calibrated prediction, universal safety, or superiority of official methods.
