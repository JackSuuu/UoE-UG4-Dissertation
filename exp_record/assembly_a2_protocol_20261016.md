# A2: integrated shared predictive gate + geometry + trusted prediction

Status: frozen-before-launch development protocol. User explicitly prioritizes
best complete-task behavior over forcing a physics-prediction contribution.
The fixed five layers remain subgoal → frozen OpenVLA → risk/event monitor →
nominal/contact-transition gating → bounded local repair. Geometry is part of
this framework. A1, its endpoints, checkpoints, data and results stay unchanged.

## Question, scope, cases

Can integrating measured geometry and phase management with selectively trusted
physical predictions improve full insertion/release compared with the existing
shared-gate geometric feedback? Record the actual action-selection owner every step;
if geometry supplies the gains or predictions rarely override it, report that.

Same physical NutAssemblySquare grasped-entry setup, camera/render configuration,
native H1 full7B frozen OpenVLA, input processing,160-interval horizon and5-interval
complete endpoint as A1. No change to the near-boundary geometric criterion.
Root3000/3001×5 original offsets are **consumed regression cases**, not independent
tests. New assembly roots3100/3101×the same5 offsets are prospective development
checks, not a final benchmark. All20 cases×4 arms=80 live episodes; no success veto,
failure filtering, severity tuning or repeated-root substitution. No LIBERO final
roots consumed. Seed-level sample size is four; do not treat20 cases as20 independent
task families. Full pick-up and long-horizon behavior remain outside this local test.

## Actual-state audit before online comparisons

Replay all30 completed A1 live trajectories (all three arms, successes and failures),
verify their status/replay/trace hashes and exact physical prefixes, and recompute
model forecasts on the actual states/actions. Include rotations, sliding, contact
loss, insertion and release as actually encountered; do not replace these with new
scripted exploration. Report positional/load/tilt residuals, owner/route counts,
grasp loss, premature releases and candidate acceptance. Hybrid proposals are only
per-state **shadow decisions**; the subsequent recorded outcomes belong to the
original executed actions, never to those unexecuted hybrid proposals.

Retain the A1 dynamics checkpoint for this first integration experiment. This
isolates mixing/acceptance from retraining; no new predictor weights are fitted.
The audit thresholds below are fixed before audit/online outcomes and are not
retuned if the model has low coverage. All old seeds are development-exposed.

## Shared observations, candidates and budgets

All arms receive identical privileged state/contact access and shadow computations;
all query the actor on each fresh observed frame.11 candidates×3-step pulse/settle
lookahead, computed in batches8+3 (never above8). These are primitive plans, not
OpenVLA chunks. MuJoCo is the evaluator, not an online forecasting substitute.
Self-developed AssemblyDynamics and RefCheckVLA.score fill the predictive roles;
neither is claimed as an official method release or certified physical model.

Original8 candidates remain. Add:
8=open-hold;9=open-retract(+z .15);10=retain the nominal pose action but close grip.
Each pulse is followed by two zero-pose settling steps with its chosen grip state.
Only the first pulse executes. All arms compute all11 predictions, even when their
controller ignores newer candidates. Every intervention arm has cap80 changed
intervals; the actor query/horizon budget stays160. Cap80 is selected for this new
version using prior development evidence; original A1's40-cap score is not replaced.

## Arms and common decision mechanics

* `vla`: nominal actions, shared shadow forecasts/monitor.
* `feedback80`: A1 feedback rule exactly, except cap80; legacy predicted trigger,
  measured load>10N retract, else largest lateral error>4mm alignment, else insert.
* `phase_geometry80`: shared predictive/physical phase gate plus the geometry
  repair below, with no prediction-based ranking override.
* `hybrid80`: **identical** phase gate, candidates, geometry, budget and fallback as
  phase_geometry80; optionally replace that geometry proposal using trusted ranking.

The first comparison measures complete-system integration. Hybrid vs phase_geometry
isolates the ranking override, since gate/candidates/budget are identical. Neither
geometry arm is a completely predictor-free baseline. No claim that prediction is
necessary if its override gives no benefit.

Legacy gate: in contact mode (nut z relative to peg origin<.13m or load>.1N),
nominal score>frozen A1 tau OR3 downward-contact/no-progress intervals. The phase
gate also reacts to seated release, an actor-open command while grasped/not seated,
or previous interval peak>10N in contact mode. Force scale10N is an engineering
diagnostic threshold, not a safety certificate.

Phase-aware geometry:
1. Seated: open-hold while touching; open-retract when detached, continuing full
   endpoint monitoring. Both completion and intervening regressions remain logged.
2. Three consecutive pre-action observations without bilateral grasp while not
   seated: hand back to the actor; do not simulate useful planar repair without
   coupling. No new scripted regrasp or different actor prompt is substituted.
3. Current/previous peak load>10N: retract.
4. Grasped actor-open before seating: retain the nominal pose command, keep grip.
5. Otherwise align the largest lateral error if>4mm; else insert.
Corrections execute only when the phase gate triggers and budget remains. High-level
labels: approach / align-and-insert / reacquire / release-and-clear; dual modes remain
nominal tracking vs contact/transition management, not four routing modes.

## Hybrid prediction acceptance

Release/clear remains geometry-managed. When actually grasped and before release,
the hybrid may choose among geometry, hold, retract, measured-error-reducing lateral
directions, insert if max lateral error<=4mm, and nominal if it does not open early
or point outward in the world-xy error projection. This projection is a heuristic
admissibility condition, not an exact action-effect guarantee under rotation/contact.

Predictions can override geometry only if:
* Previous executed-action residual: xyz<=2.25mm (one third nominal clearance),
  load<=5N, tilt<=5deg. No history means geometry fallback.
* Candidate xyz ensemble-spread norm across all3 future steps<=2.25mm.
* Candidate risk score<=max(A1 tau, current load/10N, measured tilt/15deg).
* Predicted final lateral error is no more than nominal-clearance/4 worse than
  the geometry proposal's predicted lateral error.
* Its objective improves over geometry by>.05+2×the larger of the two objective
  ensemble standard deviations. Objective retains normalized final lateral/height
  distance plus .5×peak(mean risk+std), making the added acceptance directly auditable.

If any condition fails, use measured geometry, not a disguised "verified" candidate.
These are engineering reliability/acceptance rules; no conformal/FAR/safety guarantee
is claimed. Predictions choosing nominal are logged as prediction-owned vetoes but
not as changed actions. Count actual predictive substitutions separately from gate
uses and shadow forecasts. Geometry may itself fail; do not call it certified safe.

## Integrity, execution and outputs

Before launch: unit invariants for coupling, premature release, actual prediction
override, rejection of outward/high-risk proposals, fallback and budget; four-arm
real-simulator fixtures with recorded nominal actions are labeled fixture-only.
Independent processes verify physical state, image, full forecasts, monitored
history, controller decision, effective actions and endpoint exactly.

Healthy physical GPUs1/2/3 with matched CUDA/EGL IDs. Detached supervisor first
executes the30-trajectory real-state audit, then all80 live cases with independent
process replay. Same shadow compute and query/control budgets, not a claimed common
wall-clock deadline. Microbatch<=8. Any integrity error stops new scheduling and
retains completed and partial artifacts; incomplete cases are unresolved, not failures.

Output: `/home/ami/scratch/assembly_a2_20261016_a`. Save source/protocol hashes,
checkpoint identity, monitor/history, candidates/predictions, owner/reason/acceptance,
nominal→selected→effective, observed outcome, remaining completion, latencies and
annotated2x videos. Separately report old-case regression and new-seed outcomes;
report paired rescue/harm/both-failure, load, correction budget and prediction-owner
coverage. All artifacts archived with hash verification; remote recovery recorded.
