# Two demonstrators and a testable paper contribution

## Decision: demonstrate transfer of a repair architecture across two task families

The user selects **refrigerator retrieval and garment folding** as the two headline
demonstrators. These are reasonable complementary targets: articulated rigid-body
interaction and deformable manipulation. The claim to test is:

> With the same frozen nominal policy within each task, does a shared five-layer
> runtime repair architecture improve full-task completion and recovery across
> articulated retrieval and deformable folding?

Task-specific state estimators, goal predicates, dynamics representations and bounded
repair primitives are allowed and disclosed. Reuse the gate/handoff/budget interface;
do not call two unrelated scripted task solvers a shared controller. This is a
cross-task *architecture evaluation*, not automatically zero-shot family transfer.
Target adaptation does not invalidate this question. If a task needs actor training,
train once on declared development data and give **every arm that same frozen actor**.
Report task-specific data and engineering cost. Additional household tasks are future
extensions, not assumed to need only small fine-tuning.

Assembly remains the completed mechanism study (8/20 VLA vs 18/20 intervention,
including one hybrid harm); it is insertion continuation after a fixed physical
prefix. The new demonstrators must exercise full resets and downstream handoff.
Prediction ranking has not yet shown incremental benefit; useful measured-feedback
repair is a valid outcome and remains the strongest practical starting point.

## Demo R: refrigerator retrieval

**First implementation:** stationary Panda and an articulated tabletop mini-fridge
using the local LIBERO `short_fridge` asset, one packaged food item and a visible
placement region. This tests refrigerator manipulation, not mobile whole-kitchen
navigation or cooling. Initial scene: door closed, object supported on a shelf,
robot empty-handed; no scripted opening/grasp/transport prefix in the full demo.

Sequence: find handle → open → approach/grasp item → extract through opening →
place/release on table → close door → withdraw → verify retained placement.

| Failure opportunity | Monitor evidence | Bounded repair hypothesis |
|---|---|---|
| Handle miss / door stalls | hand-handle offset, contact, hinge progress | back off and realign; bounded regrasp |
| Shelf/door interference during extraction | clearance trend, contact, object/gripper relative motion | unload, retract along free direction, reorient |
| Lost grasp | finger contact and relative motion, falling item | stop transport; regrasp only within the registered reachable recovery region |
| Premature close / recontact | item still inside or hand in doorway | phase-aware hold/withdraw, then resume actor |

**Development endpoint proposal:** all object collision bounds inside the target
region with >=5 mm margin; object supported on table, no finger contact; hinge
within 5 degrees of its physical closed pose; final object speed <0.02 m/s and
angular speed <0.1 rad/s for 20 control intervals (1 s at 20 Hz). Verify hand clearance
and door/object retention throughout that dwell. Necessary handle/grasp/support
contacts are not violations. These are proposed evaluator tolerances, to be checked
against metric fixtures and asset scale **before** the comparison registration.

**Immediate diagnostic:** build/reset/settle scene, inspect initial image and item
support, execute native full OpenVLA from closed-door and open-door initializations.
The open-door trial isolates extraction capability; it is never pooled with the
closed-door full-task score. Record native predicates as diagnostics, not as the
complete endpoint above. Keep failures and simulation defects visible. Asset
inspection found a door range with `limited=false`: validate actual joint mechanics
and define a versioned scene fix before a scientific comparison, rather than treating
unbounded rotation as a meaningful robot failure.

## Demo C: cloth folding, then garment folding

**C0 manipulation integration:** reachable rectangular cloth on a table, physical
parallel-jaw grasp and single fold about a marked midline. A passive cloth drape
does not complete this stage. No particle teleportation; if attachment-based grasp
is needed for an engineering check, declare it and keep it separate from contact grasp.

**C1 first complete demo:** start with an empty gripper and flat/slightly wrinkled
cloth → select corner → grasp → lift and carry over fold line → align → release →
withdraw → verify shape. Single-arm first; no need to introduce bimanual hardware
before contact/shape/actor compatibility is known. Rectangular towel folding must
be named towel folding, not full garment folding.

**C2 headline garment extension:** a T-shirt with known mesh correspondence; fold
sides/sleeves then bottom, release and verify the desired layered shape. Start with
a laid-out garment before evaluating crumpled starts. This is a milestone, not a
capability already supplied by the installed cloth solver or LIBERO10 checkpoint.

| Failure opportunity | Monitor evidence | Bounded repair hypothesis |
|---|---|---|
| Missed corner / lost grasp | jaw contact, cloth/gripper relative motion | bounded corner reacquisition |
| Slip or excessive stretch | material-point separation and strain estimate | reduce motion, unload, regrasp within reach |
| Misaligned fold | edge/corner correspondence to target | small edge alignment before release |
| Snag / failed handoff | local shape stagnation and contact; remaining milestones | retreat/unload, preserve completed fold, hand back |

**Development endpoint proposal for rectangular single-fold:** target footprint IoU
>=0.85, RMS error of corresponding folded material points <=0.10 of cloth width,
correct layer ordering at preregistered interior probes, no gripper attachment/contact,
and p95 material-point speed <0.02 m/s for 1 s. Do not use silhouette alone: a crumpled
cloth can have the right outline. Test metrics on unfurled, correctly folded,
misaligned, crumpled, and still-held fixtures. Mesh/material resolution sensitivity
and grasp stability precede setting a strain/damage threshold. T-shirt metrics need
their own frozen correspondence/target-layer specification.

## Execution ladder: engineering first, comparative scores second

1. **Scene + action integration:** robot workspace, camera convention, actuation,
   support/contact, hinge/cloth constraints; retain reset and motion artifacts.
2. **Native actor capability:** three development seeds (0–2) per task configuration,
   no repair and no success-selection. First refrigerator smoke is bounded at 240
   actor calls per closed/open scene, seed 0; it only diagnoses integration/progress.
   A full episode horizon is selected from engineering needs before comparison.
3. **Measured-feedback repair:** implement only corrections motivated by actual
   failures; test handoff and remaining-task completion. Run every development case
   on every arm, including clean cases and baseline successes.
4. **Freeze comparison:** metric fixtures, source/actor hashes, adapter parameters,
   seeds/assets, observation contract, delays, disturbances, budgets and endpoints.
5. **Confirmation:** disjoint seeds/assets and perturbation schedules. No changes
   after inspecting results; later changes create a new development version.

Initial actor failure is diagnostic: determine whether it is scene/action mismatch,
missing task skill, detection failure, or ineffective local correction. If whole
task skills are absent, declare task training or a task-capable nominal actor rather
than making the repair layer secretly execute the full task. Keep trained and
zero-shot runs separate. The proposal of 3 development seeds is not a powered study.

## Paper: figures and experiments

**Figure 1:** shared five-layer architecture and target dual-rate schedule (README).
**Figure 2:** fridge and cloth phase sequences with one actual repair/handoff each.
**Main table:** full-task completion, time/queries, paired rescues/harms for all tasks;
assembly is clearly labeled continuation, fridge/cloth full-reset manipulation.
**Mechanism table:** event detection on encountered states, local repair success,
post-handoff completion and prediction acceptance/overrides.
**Transfer table:** held-out object/layout/hinge friction and cloth dimensions,
stiffness/friction/initial shape; report each named shift and actor exposure.
**Timing table:** observation age, detection-to-command and detection-to-physical
response p50/p95/p99, missed deadlines and stale proposals; include whole-loop cost.

Comparison arms (same nominal actor, task prompts, state-estimation source and
completion rule):

* A: frozen nominal VLA.
* B: task-compatible simple feedback repair with a predictor-free trigger.
* C: shared five-layer event/phase-aware feedback controller.
* D: C plus trusted physical prediction **once a compatible predictor exists**.

Compare C/B to isolate phase/event management, D/C to isolate prediction, and C/A
for full-system benefit. Declare B/C/D's extra sensor inputs relative to actor-only A;
use identical inputs for B/C/D. An optional competent task-space planning controller
can strengthen the paper but must obey the same observation/budget contract.
H1 actor queries on unchanged observations are not a meaningful replan ablation.

Suggested confirmation scale: per task, 30 independent initializations each under
nominal, one registered physical disturbance, and one held-out-instance condition;
all arms paired on all 90 cases. Final sample size/budgets depend on development
variance and compute and must be registered before confirmation. These counts are
a design proposal, not an already launched cohort. Randomize execution order; pair
exogenous disturbance schedules by simulated or wall time, never by observed
baseline failure. Report event exposure when arms reach different phases. Confidence
intervals resample independent initializations/assets, not action steps. Unstarted
or invalid-environment cases stay visible separately from genuine task failures.

## Real-time milestone

Keep the five layers. Split compute scheduling, not the research architecture:
slow VLA proposals carry observation time, phase, and sequence ID; fast monitor/repair
owns execution during an intervention. Reject stale/phase-incompatible proposals,
reset handoff state deliberately and request fresh inference after major state change.
One executor arbitrates commands; simultaneous actor and repair commands are forbidden.
A nominal 20 Hz feedback loop is an initial target, not a measured guarantee or a
task-independent deadline. Infer task deadlines from tolerated motion during delay,
then freeze them. In clocked evaluation physics continues during actor inference.
Use the same scheduler for actor-only and repaired arms. Compare paused and clocked
evaluation separately; report missed deadlines, not just simulator frequency.

## Demo delivery

For each implemented task publish matched VLA-left/framework-right videos from the
same reset/disturbance, common speed and labeled terminal holds. Show phase, event,
repair interval, handoff and complete outcome. Select representative rescue,
preservation, harm and both-failure cases deterministically from an all-case index.
The strongest demo is not merely a successful animation: show a real disturbance,
local recovery, and **the rest of the task completing**. Raw baselines are retained
as data; a comparison video requires both arms, never a fabricated framework panel.
