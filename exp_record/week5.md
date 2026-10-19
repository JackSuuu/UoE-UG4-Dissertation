# Week 5 (19–25 Oct 2026) — Unstructured transfer, honest OOD and paired evidence

Newest entry at the top. [Weekly index](README.md). Detailed records live in
[`agent_notes/`](agent_notes/README.md).

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
