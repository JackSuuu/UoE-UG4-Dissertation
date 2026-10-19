# Week 5 (19–25 Oct 2026) — Unstructured transfer, honest OOD and paired evidence

Newest entry at the top. [Weekly index](README.md). Detailed records live in
[`agent_notes/`](agent_notes/README.md).

### Update 19 Oct — Wrist-rendering diagnosis and explicit18-cell continuation

11fresh zero-query replays/5,752frames show exact physical/controller/contact and
agentview evidence; captured wrist differences are1–5pixels with channel delta1.
Fixed sensor-time redraws vary without advancing physics, demonstrating rendering
nondeterminism. Original strict failures remain. Root1/2diagnosis onGPU1is
cross-device, not original-lane certification. Source `53a74b3` freezes continuation
of18untouched cells onGPUs1/2/3; supervisor3727880started after combined-report
preflight. Original6episodes are referenced and all24identities retained.
[Decision/diagnosis](agent_notes/native_long_l1_continuation_20261019.md).

Partial failure analysis cautions against early no-lift alarms: successful microwave
controls also wait156–158actions after grasp. Two-pot failures retain one placement
and fail unfinished-object acquisition. These guide later controlled diagnostics,
not asserted repair gains. Launch/R0source archives independently restored from
GitHub/LFS; the first180s download timeout remains recorded.

### Update 19 Oct — First six L1 episodes completed; replay gate paused remaining18

Microwave roots0/1/2: failure520actions, native success444, native success397;
all three replay exactly. Two-pot roots0/1/2: recorded success500, failure520,
failure520; physics and integration trajectories match exactly, but wrist-image
differences in1/1/3frames caused strict replay failures. Agent-view images have
no reported mismatches. The supervisor correctly stopped all three lanes;
18cells remain unstarted. A rendering diagnostic and an explicit continuation
are being prepared. Preserve the original replay failures and keep task
selection provisional until all six cases per task pass the declared integrity
review. These are actor-only development outcomes, not framework gains.

### Update 19 Oct — L1 parallel native screening launched; R0 readiness research completed

User authorizes execution and parallel GPU use. Frozen implementation/protocol
`63fecbd`;24episodes (four tasks×six development roots), physicalGPUs1/2/3,
eight episodes each. Supervisor3715906is detached; all three initial microwave
workers entered actor execution. Initial health snapshot has4actions each, no
task outcome yet. Four scene+initialization replays and one zero-action official
model query passed;13synthetic result-accounting cases passed.

Preflight discovered reversed EGL enumeration: physicalGPU1/2/3maps toEGL2/1/0.
CUDA UUID selection and the rendering map are separately verified; GPU0is idle.
Preserve all native failures, infrastructure issues and source/replay proofs.
[Actual launch](agent_notes/native_long_l1_launch_20261019.md),
[launch archive](../backups/native_long_l1_launch_20261019/).

Independent [R0audit](agent_notes/robocasa365_readiness_20261019.md) identifies public
native Xiaomi/Awomo checkpoints with author-reported composite-task success and
documents their distinct observation/action/history contracts. Local readiness,
competence and public executable CheckVLA reproduction remain unqualified.
No RoboCasa weight download, installation or actor query occurred. L2/L3will use
L1's failure map to target trigger, candidate, verification or handoff defects.

### Update 19 Oct — Fixed OpenVLA + native LIBERO-Long; alternate setup removed

> Later discussion adds RoboCasa365 as a second-platform replication track while
> retaining LIBERO as immediate mechanism development. The consolidated
> [current goal and queue](agent_notes/current_goal_and_experiment_queue_20261019.md)
> orders24native baselines, early-state candidate diagnostics, paired framework
> revisions and held-out confirmation. Second-platform model readiness can be
> researched alongside baseline execution. This consolidation launches no rollouts
> or installation; the cleanup below remains a historical fact.

User explicitly rejects changing model/platform and asks for harder native tasks.
The existing LIBERO10 checkpoint is already the official LIBERO-Long model. Paper
Table12reports53.7%±1.3%SE suite success; no per-task local rate follows from it.
Read-only audit checks all10native task definitions/runtime hashes and initial-file
availability. Recommended cohort: microwave mug+close, two pots on stove, two mugs
on designated plates, bowl+drawer close. Microwave/drawer start open; stove starts
on in the two-pot task. Do not invent extra native subgoals or mandatory object order.

Proposed next screen:4tasks×6development roots0–5, original task predicates,
520model actions+10settle steps, same frozen weights/official TF4.40.1 inference.
Keep existing calibration/training/confirmation reservations. Seek demonstrated
competence plus natural local failures rather than requiring8/10success and first
adding disturbances. The24-episode screen is **not launched** in this update.
[Selection, prior evidence and attribution plan](agent_notes/openvla_native_long_selection_20261019.md).

RoboCasa preparation had downloaded assets/alternate weights but made zero actor
queries/actions. Installation was terminated; its dedicated source/environment/
assets tree, alternate model caches and exclusively referenced shared blobs are
deleted. Existing OpenVLA/LIBERO preserved. [Deletion record](agent_notes/robocasa_cancelled_20261019.json).

### Update 19 Oct — User confirms capability-first reliability, not new skill learning

The user clarifies that the framework should improve tasks the frozen VLA already
performs. Suspend current unqualified fridge efficacy trials and move new actor
training out of the next main stage. First reproduce a capable native actor/task
pair, then register recoverable perturbations and test matched nominal/simple/
verification policies. Fridge stays preferred; a native CloseFridge skill is only
a component check, not complete retrieval. RoboCasa/LeRobot are absent in the
current runtime; a pinned native-environment readiness check precedes qualification.
No new actor rollout or training is launched by this planning update.
[Admission gates, scope and evaluation ordering](agent_notes/capability_first_reliability_plan_20261019.md).

### Update 19 Oct — Camera cohort finalized: no grasp benefit, narrow initial diversity

All six original/framed actor runs finish600actions without grasp/native success;
the full retrieval endpoint is unscored. Each initial camera pair matches exact
physics/controller/wrist evidence. Seeds differ, but item-position separations are
only3.4–15.5mm. Reframing changes position only and does not establish visibility,
domain compatibility or skill. More varied motion does not imply task progress.

All six separately versioned cached replays pass exactly, preserving original
300s replay timeouts and all242historical files. Finalization makes no new actor
queries. Three original-left/framed-right videos include all600actions, aligned
annotations and labelled holds. [Results and interpretation](agent_notes/fridge_observation_results_20261019.md).

### Update 19 Oct — Failure audit identifies late triggering and non-goal-directed choice

User challenges repeated failures. Parent reads the completed fridge tapes: EEF–
handle distance starts0.4601m and never meaningfully improves; first intervention
at368occurs at1.0192m. Stagnation-only monitoring misses continuous departure;
command-novelty ranking does not evaluate task-directed progress. Q1 samples these
late states, not earlier recovery opportunities. Missing learned skill is not the
sole established explanation; camera/domain compatibility remains unresolved.
Pause repeated prototype efficacy sweeps, qualify the task contract/basic capability,
then redesign early detection and task-directed candidates before Q2/cloth.
[Evidence and accountability](agent_notes/fridge_failure_accountability_20261019.md).

### Update 19 Oct — Q1 complete: no useful candidate in the sampled opportunities

Closed-start fridge nominal/simple/verification finish600/552/504actions under
the shared600-query cap, with no opening/stage attainment. All six actual handoffs
are local-budget exits, not verified recovery. All1656actions replay exactly.

The queued offline Q1 diagnostic then evaluates all9candidates at the first three
eligible entries368/403/443, with exact fresh-prefix reconstruction.288branch
actions/279fresh model calls; all independent replays pass. There are six distinct
state/action pairs, not nine independent observations. At1/12/32actions none
advances opening usefully under the fixed greedy continuation; maximum directional
progress1.33e-15rad. Conditional useful-selection rates are undefined, not equivalent
between selectors. Q2 remains gated; prioritize shared skill/proposal capability
and observation-interface qualification before further attribution or cloth.
[Q1 results](experiment3_fridge_q1_20261019/results.md)
· [Fridge pilot](experiment3_fridge_candidate_20261019/results.md)
· [Book negative preflight](experiment3_candidate_20261019/results.md).

Published four baseline-left paired videos plus the offline all-candidate video
and three complete archives in`0cb42b8`. Fresh GitHub/LFS recovery verifies403/403
archive members and all five MP4s. [Evidence index and remote proof](../backups/autonomous_candidates_20261019/README.md).

### Update 19 Oct — Shared actor lane complete; fridge capability remains unqualified

LIBERO90 OpenVLA fails both600-action fridge transfer tests: no closed-door opening
and no open-door grasp. All1200actions replay exactly with token/named-normalization,
RGB, contact and clock checks. Book controls add1800exact-replayed actions, with
native success3/3and strict completion0/3. This is a negative capability result,
not a framework experiment. The new actor has no in-domain positive control; its
original robot-side camera/constant instructions differ from the framed/phase-prompt
candidate pilot. Observation/domain and checkpoint quality remain unresolved.

Remote fridge-trained policies/data do exist, but inspected embodiments/interfaces
do not match fixed-base Panda OSC. No training or framework-only actor swap took
place. Keep the candidate-usefulness diagnostic next; formal framework gain awaits
a qualified shared actor. [Fridge results](agent_notes/shared_actor_fridge90_results_20261019.md)
· [book results](agent_notes/shared_actor_qualification_results_20261019.md)
· [resource audit](agent_notes/shared_actor_fridge_pretrained_audit_20261019.md).

Both verified videos and the204-member complete artifact/data-audit archive are
[published](../backups/shared_actor_20261019/README.md) in`3de144b`. A fresh
GitHub/LFS download restored and SHA256-verified every member and both videos;
[remote proof](../backups/shared_actor_20261019/remote_verified.json).

### Update 19 Oct — Candidate usefulness and verification attribution before cloth

User approves two explicit next questions: whether model candidates advance the
task, and whether verification improves selection/rechecking/actual handoff over
simple feedback. Native-book3-arm preflight is negative: all strict completions
fail, only nominal attains native success; verification handoffs are budget exits.
Book actor qualification has3/3native milestones but0/3additional strict completion.
Validate the stronger endpoint's feasibility rather than invalidating native success.

Fridge camera diagnosis, closed-start candidate integration and LIBERO90 actor
transfer are still active at this decision check. Assign the next GPU2 experiment
after its current cohort: bounded exact-common-state candidate branches, with
offline-only outcome labels, task-directed progress and fixed nominal continuation.
Then use selection × rechecking controls for attribution, conditional on useful
candidates/shared capability. Cloth experiments follow this rigid-body stage.
[Protocol, costs, decision gates and execution snapshot](agent_notes/candidate_progress_verification_next_20261019.md).

### Update 19 Oct — Keep the fridge as the main rigid-body task

User reiterates full refrigerator completion and framework-factor study as priority.
Native-book controls remain preflights. Parallel actor/data work is steered toward
genuine fridge capability and the candidate-mechanism integration toward fridge,
without modifying frozen camera/book pilot outcomes. Added a read-only six-stage
evidence observer (no action generation), with satisfied/violated/unknown predicates,
ordered milestones, release evidence, continuous dwell and explicit open/closed-start
completion distinction. Four semantic tests pass; no empirical fridge success is
claimed. [Plan and evidence scope](agent_notes/fridge_primary_mechanism_plan_20261019.md).

### Update 19 Oct — Parallel observation, actor and autonomous-proposal experiments

User explicitly authorizes experiments1/2/3in parallel, requesting reuse of the
existing reasoning-chain implementation and synchronized video evidence. Located
`libero_drawer.decision_trace`: pre-action evidence, gate, actual action and separate
outcome, not a fabricated model rationale. GPU1launches six pure-actor fridge camera
controls from`7a158cd`; GPU3is assigned strict shared-actor qualification and genuine
adaptation-data audit; GPU2is assigned autonomous model-candidate/verification pilot.
Fridge efficacy remains conditional on actor competence. Six camera scene preflights
and a one-action exact frozen replay pass. [Execution contract](agent_notes/parallel_diagnosis_execution_20261019.md).

### Update 19 Oct — Actor controls finished: capability and transfer are distinct

All five zero-repair trials completed (1,997 model actions) with exact independent
physics/controller/images/diagnostic replay. Book/native reaches native success
after196actions under TF5 and359under official TF4; cheese+butter/native reaches it
after242. Book/robot-side fails to grasp in600actions, despite identical initial
physics; the camera intervention also removes much of the gripper/caddy visibility.
Official-inference refrigerator still fails to grasp in600actions, with576near-zero
XYZ commands versus504in historical TF5. Thus inference compatibility alone is not
a sufficient explanation or fix. Stacks do diverge on identical observations at
book step3and fridge step6; global parity is false.

Neither book run verifies stable released placement: both stop on native done before
any strict placement candidate/dwell. Native predicates must not be presented as
our stricter complete-task endpoints. No framework gains are claimed. Four annotated
20fps diagnostic videos preserve all frames and explicit terminal holds.
[Detailed results](agent_notes/vla_failure_audit_results_20261019.md) ·
[Artifacts](../backups/vla_failure_audit_20261019/README.md).

Published in`4d67845`; fresh GitHub/LFS download verifies185/185cohort archive
members and all four standalone MP4s by SHA256. No actor experiment remains running.

### Update 19 Oct — Diagnose the VLA; no scripted repair as autonomy evidence

User sharpens the requirement: explain the baseline failure and attributable
improvement, rather than human-writing its motion solution. R2open-wrapper grasp
and extraction occur under scene-specific primitives:382primitive actions plus
38grip-retention actions versus180unmodified VLA actions. Open baseline has504/600
near-zero XYZ proposals. Pre-intervention paired actions/states/images match.

Isolated official TF4.40.1inference matches current inputs on6fixed observations,
but only5/6token/action sequences; one differs including gripper sign. Official
cached and uncached outputs agree with each other. This corrects an intermediate
mistaken6/6statement. No complete inference-parity or root-cause claim is made.

Five actor-only task/camera/decoder diagnostics are prepared with600queries/actions
each and fresh-process replay. Four scene smokes, exact initial pairing and a pinned
one-action book rollout/replay pass. Further manual motion-route tuning is paused.
[Protocol](agent_notes/vla_failure_audit_protocol_20261019.md) ·
[Evidence and execution](agent_notes/vla_failure_audit_execution_20261019.md).

Actor-only audit launched detached from pushed freeze`1477011`, supervisor3664969,
healthy GPU1/2/3. Three book controls start first; fridge/cheese follow. Preflight
archive139/139members independently restored from GitHub/LFS with exact SHA256.

### Update 19 Oct — R2 finished; pause scene-specific repair tuning

All four600-query/action trials finished. Closed-start baseline and wrapper both
failed to open. Open-start wrapper grasped/extracted but did not place or complete;
baseline never grasped. Both wrapper runs used420corrections (70%). Three exact
replays pass; the fourth has exact physics/controller/decisions/agentview, with two
wrist frames differing by three total RGB channels, each by1.

Post-hoc frozen replay corrects an event-accounting bug: `retain_grip` overwrote two
verified local exits (steps245/302). Neither immediately handed an unmodified
proposal to VLA; subsequent sequence completion remains zero. Original records stay
unchanged. Both601-frame/20fps baseline-left/framework-right videos are delivered.

Door diagnostics found a32mm no-contact gap; contact-based branches moved the door
partway but none passed opening. The user rejected per-scene coordinate tuning;
it is paused. Next-version acceptance requires the same frozen repair across held-out
layouts/instances, object-relative inputs, shared-primitives feedback ablations and
a task-capable shared actor. These capabilities are not yet implemented.
[Detailed results](agent_notes/fridge_r2_results_20261019.md) ·
[Videos and complete archive](../backups/fridge_r2_20261019/README.md).

GitHub/LFS upload and independent recovery are complete:200R2members,374R1actor
members and both standalone R2comparison MP4s match their SHA256 manifests from
commit`afd97b0`. Recovery proofs accompany each archive.

### Update 19 Oct — Implement actual VLA + five-layer refrigerator runtime

User requests the integrated comparison directly. Added `fridge_runtime.py` and
`run_fridge_framework.py`: shared phase tracker/prompts, actual frozen OpenVLA in
both arms, progress/grasp checks, bounded repair gate and revalidated handoff.
Six guard tests pass;100live physical steps exercise stall-triggered repair without
claiming task success. Both arms retain privileged state access/shared phase prompts;
visual grounding and learned physics prediction are not implemented.

[R2 protocol](agent_notes/fridge_r2_framework_protocol_20261019.md): seed0 closed/open
× baseline/framework,600actions/queries each,200local/420total correction cap.
Independent replay now verifies gate/phase/endpoint decisions as well as physics.
`run_fridge_r2.py` supervises detached GPU1/3 jobs and exports matched left-VLA/right-
framework videos including ownership, triggers, handoffs and budget failures.
The prior R1 actor cohort has completed12/12with exact independent replay.

**R2 launched:** source freeze`c4afde8`, detached supervisor3653705, GPU1baseline /
GPU3framework;4planned600-step runs. Automatic decision/physics replay and paired
videos follow completion. Three separate physical endpoint fixtures pass, including
positive supported placement and negative out-of-bounds/unsupported placements;
fixture state installation is explicitly excluded from task evidence.
[Execution record](agent_notes/fridge_r2_execution_20261019.md).

### Update 19 Oct — Default paired-video delivery; first refrigerator side-by-side

User reiterates baseline VLA left / our framework right for every future experiment.
This is now explicit in the root plan and README. Added `render_fridge_pair.py`,
which verifies initial state/controller/image equality, streams both sources at a
common simulated rate and labels held end frames. Formal framework mode requires
an actual hybrid run and a shared comparison contract.

[Current refrigerator pair](../backups/fridge_r1_20261019/baseline_left_servo_right.mp4)
shows seed0 open-door canonical VLA (160steps) against our engineering servo
(485steps). The video explicitly labels the scripted right arm and unequal budgets;
it is not an efficacy result. Initial physics/controller/both images match exactly.
Verified486output frames at20fps and inspected a late frame for legible labels.

### Update 19 Oct — Preserve verification theme; define visual grounding and ICLR targets

The user clarifies that the central contribution remains an external runtime
verification–repair loop around VLA. Updated the Chinese plan and README accordingly.
[New evidence plan](agent_notes/verification_repair_iclr_20261019.md) defines proposed
phase-condition checks, uncertainty-aware revalidation/handoff, sensor-versus-oracle
comparisons, 1/3/5-object grounding development, strong matched baselines and cross-task
causal tests. The official ICLR reviewer guide motivates rigor and new knowledge,
not a mandated number of tasks or an acceptance promise.

The successful extraction servo uses fixed waypoints, privileged EEF feedback and
fixed phase durations; images do not select or localize its object. It establishes
physical feasibility, not a visual-physical verifier or refrigerator framework gain.
The existing frozen actor cohort continues unchanged; its observed completed cases
have not yet demonstrated object grasp. Assembly's measured intervention benefit
does not automatically transfer to refrigerator or cabinet tasks.

### Update 19 Oct — Recovery-centered plan; R1 exposes scene and grasp constraints

The revised [mainline and experiment ladder](agent_notes/closed_loop_recovery_r1_20261019.md)
prioritizes event/phase-aware closed-loop recovery beyond simple feedback. Differentiable
prediction is optional and separately ablated. A2's feedback arm shares prediction
gating, and neither phase nor predictive-ranking benefit has been established.
The Chinese root plan now makes this precedence explicit; the user's one-year notes
and frozen R0 protocol remain untouched.

[R1 engineering](agent_notes/fridge_r1_execution_20261019.md) finds that R0 exposes
the fridge back wall to the robot. Versioned limits, approach/camera layout and
gripper-palm clearance diagnostics precede actor comparisons. Five servo variants
retain failures; v5 physically grasps, lifts and extracts the item by0.173744m with
bilateral pad contact. This is a scripted feasibility probe, not a VLA success.
Independent physics/controller replay is exact, but a few image hashes differ;
raw pixel recording is added to diagnose that unresolved discrepancy. A frozen
12-run native-actor prompt comparison is the next executable stage.

**Launched after source freeze `080c9a4`:** detached supervisor3644620, healthy GPU1/3,
all12fresh-reset canonical/skill × closed/open × seeds0–2 trials,160queries each.
No scripted prefix; independent replay after each run. GPU2's pixel-replay diagnostic
continues separately. [Labeled extraction clip and205-file engineering archive](../backups/fridge_r1_20261019/README.md).

**Pixel diagnosis completed:**485-step instrumented servo replay has exact physics,
controller and all actor-facing agentview images. Four wrist frames differ by1in
30RGBchannels total. This remains a nonexact two-camera replay; no tolerance was
relaxed. The archive now includes that report; actor trials continue in the background.

### Update 19 Oct — Commit to refrigerator and folding demonstrators; start robot validation

**Completion addendum:** both240-query fridge trials finished and passed exact
independent action/physics/controller/image replay. Neither retrieved the object;
closed trial moves but never opens the door, open trial is nearly stationary.
No item/gripper contact at sampled endpoints. These expose nominal-policy/scene
integration issues, not a tested hybrid failure. Diagnose workspace/phase prompts
and task skill before comparing local repair. Both failures are retained in the
[R0 archive](../backups/fridge_r0_20261019/README.md).

User clarifies that the priority is a useful, convincing two-task demonstration and
paper evaluation, not a requirement to establish six-family zero-shot capability
before running anything. [Two-demo design](agent_notes/two_demo_experiment_design_20261019.md)
defines complete fridge opening/retrieval/placement/closing and cloth single-fold →
garment-fold milestones. The paper question is whether the same five-layer runtime
architecture improves completion across task-specific adapters. Per-task actor
training is permitted if disclosed and shared/frozen across all comparison arms;
task-specific repair development is not mislabeled unseen-family transfer.

Added the five-layer Mermaid diagram at the top of the root README, including
feedback, phase-aware handoff, task adapters and the target asynchronous actor / fast
feedback scheduler. The figure explicitly distinguishes the tested synchronous
assembly implementation from the not-yet-implemented dual-rate real-time schedule.

Actual scene work: discovered the existing articulated mini-fridge in local LIBERO,
built a Panda/food-item task and ran closed/open-door reset/settle checks. Food is
supported on the shelf; initial camera exposed a poor side view, corrected by moving
and rotating the fixture before actor trials. The asset's door joint has disabled
limits; this is recorded as an environment issue for the next scene version, not a
robot safety metric. Both seed-0, 240-query native-actor probes launched detached on
healthy GPUs1/2. No scripted grasp prefix or intervention. These are development
diagnostics, not a hybrid comparison or final full-task score. [Execution record](agent_notes/fridge_development_execution_20261019.md).

### Update 19 Oct — A2 paired videos and unstructured probe independently recovered

Pushed results, five representative comparison MP4s, U0probe and record reorganization
as`46a5371`. Independently fetched GitHub/LFS bytes and checked697A2archive members,
8U0probe members and all5standalone comparison videos against SHA256. Proof:
[`week5_remote_verified_20261019.json`](../backups/week5_remote_verified_20261019.json).
All20A2pairs are locally rendered; the representative5include negative outcomes.
No six-task robot efficacy cohort is running; U0is complete and remains a physics
backend qualification only.

### Update 19 Oct — Six unstructured task families require explicit capability and OOD tests

User prioritizes cutting vegetables, peeling garlic, raw-meat handling, refrigerator
retrieval, dish washing and garment folding. [U-series design and data/OOD contract](agent_notes/unstructured_ood_design_20261019.md)
defines what the current mixed controller supports, what source/target experience
means, task-specific physical criteria, full-reset trials and paired recovery controls.
The suggested LIBERO multi-object pilot was not launched before this steering.

Current A2 is task-specific: a frozen LIBERO10actor, a learned nut/peg predictor and
handcrafted assembly phases/geometry. Target-specific fitting or threshold/candidate
design is adaptation, even if actor weights stay fixed. New seeds alone are not
unseen-task generalization. Component-relative exposure and held-out families/assets
must be registered before inspecting evaluation outcomes. Unknown foundation-model
pretraining overlap stays unknown.

Installed capability audit: Genesis1.4.2and MuJoCo3.14available; OmniGibson/IsaacSim,
ManiSkill and SoftGym/PyFlex not installed. **U0actually executed** on healthy GPU2:
120steps of cloth draping over an asymmetric obstacle,1089particles,121recorded
states; maximum pair-distance change0.14776m confirms non-rigid evolution rather
than a rigid-cloth proxy. Independent-process repeat is **not bit-exact**: maximum
particle-coordinate difference~0.0000505764m. No post-hoc tolerance is declared a pass.
This establishes executable deformable backend primitives only—not garment folding,
robot manipulation, real-time control, calibrated material physics or OOD efficacy.
No target-task policy training or six-task hybrid evaluation has been performed.

Plan: articulated retrieval and cloth manipulation first; washing needs actual
residue/tool interaction, cutting needs validated fracture or an explicitly named
symbolic benchmark, peeling needs adhesion/separation, meat requires a specified
deformable handling operation and validated material assumptions. Runtime response
must be measured with a moving world and timestamps, separately from paused simulation.

### Update 19 Oct — A2 verified18/20; prediction ranking never overrides geometry

[Complete results](agent_notes/assembly_a2_results_20261019.md): all80live episodes
and30historical actual-state audits completed, with independent exact replay.
VLA8/20; feedback80, phase_geometry80and hybrid80each18/20. Old/new cohorts separately
score3/10+5/10forVLA and9/10+9/10for each intervention arm. Hybrid rescues11baseline
failures but harms1baseline success. Prediction override count0; hybrid and phase
geometry have identical executed trajectories in all20cases. Feedback achieves the
same completion with lower mean peak load27.97vs30.39N and fewer corrections1442vs1600.
These are small development results, not universal/generalization/safety evidence.

User requests paired video as the default deliverable. Rendered20matched VLA-left /
hybrid-right MP4s with synchronized decision index,2x simulated-time labels and held
end-frame labels. Five representative clips include preservation, old/new rescues,
harm and both-failure; all20pairs remain indexed. [Video index and download notes](../backups/assembly_a2_paired_review_20261019/README.md).

### Update 19 Oct — Weekly records reorganized without changing frozen protocols

Moved38non-week detailed records into`exp_record/agent_notes/`; weekly files stay
at the outer level. Relative navigation repaired, provenance/hashes retained in
`agent_notes/relocation_manifest.json`.32relocated files are byte-identical; six
historical result/launch notes change only relative Markdown links. Frozen protocol
bytes and source code are preserved. Historical launchers that require old paths
should be run from their frozen commit/source archive. Original`notes/`proposals
and the user's uncommitted one-year-plan edit are preserved.
