# Closed-loop recovery mainline and R1 refrigerator diagnosis

## Decision and evidence

The user authorizes a recovery-centered revision, refrigerator basic skills first,
then cloth/garment folding. Keep the five layers. The proposed contribution is
**event/phase-aware bounded repair and validated handoff to a frozen VLA**, assessed
by complete-task success, retained progress, paired rescue/harm and wall-clock cost.
Differentiable physical prediction is optional and must earn its place by incremental
decision utility. A stronger research question is not an already stronger result.

A2 has 8/20 actor successes and 18/20 in all three intervention arms, zero predictive
ranking overrides and identical phase/hybrid tapes. Simple feedback uses fewer
corrections and has lower mean peak load. Its trigger still uses prediction, so
neither prediction-free parity nor phase-management superiority has been established.
R0's two 240-query trials manipulate neither door nor item. These are development
integration failures, not a repair-controller evaluation.

## Literature-derived experiments, not inherited performance claims

* [CheckVLA](https://arxiv.org/html/2607.26789v1): action-conditioned discrepancy,
  calibrated first-intervention trigger, latency-feasible suffix repair, progress
  memory. Its paired periodic-replan/full comparison is 27.6/36.1%; action-condition
  ablations matter. Our H1 OpenVLA cannot reproduce flow-chunk rewriting unchanged.
* [SARM](https://qianzhong-chen.github.io/sarm.github.io/), ICLR 2026: stage/progress
  estimation and reward-aligned policy training, including real T-shirt folding.
  Transfer the evaluation idea of progress regression, not its training gains.
* [ARMOR](https://arxiv.org/html/2602.12405v2), paper labeled ICLR 2026: failure detection
  and open-ended reasoning; useful diagnosis must additionally be tested in recovery.
* [Ctrl-World](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0ae94013da7cd459402fd77874e09ee3-Abstract-Conference.html),
  ICLR 2026: controllable world modeling for ranking/training policies. Prediction
  fidelity and downstream control benefit are distinct evaluation axes.

## Paper experiment ladder

| Stage | Question | Comparison / criterion |
|---|---|---|
| R1 | Are workspace, geometry, actuation and nominal skills viable? | Collision inventory, physical servo diagnostic, fixed canonical vs skill prompts; all cases retained |
| R2 | Can the nominal actor perform the whole task? | Full reset; shared task-capable frozen actor; valid placement/closure/release/dwell fixtures |
| R3 | Do event/phase/handoff mechanisms add value? | Predictor-free simple feedback vs five-layer controller, identical state source/primitives/budget |
| R4 | Which component changes outcomes? | Shared repair with different triggers; identical trigger snapshots with different repairs/handoff; progress-memory ablation |
| R5 | Is the effect robust and deployable? | Disjoint initializations/assets; natural and registered disturbances; continuing physics under latency; shared scheduler |
| C0–C2 | Does the architecture extend to deformables? | Contact grasp → towel single-fold → multi-stage T-shirt; frozen task-specific adapters disclosed |

Main arms: actor A; truly predictor-free feedback B; phase/event recovery C;
competent matched-information planner P if implemented; C+trusted predictor D only
after task-compatible prediction exists. Include actor with the same phase prompts
to isolate high-level prompting from intervention. H1 repeat queries on the same
image are not a replanning baseline. Any planner with simulator rollouts is an
oracle diagnostic unless the same information contract is available to comparators.

Report full completion, local recovery and completion after handoff separately;
rescue/harm with explicit denominators; completed-stage regression; correction and
query counts; adaptation data; observation age; detection-to-physical-response p95/p99;
deadline misses and stale proposal discards. Match initial state, allowed observations,
external disturbances, termination and action/query/time budgets. Trigger comparisons
use common no-repair shadow tapes; repair comparisons branch from common snapshots.
Later full closed-loop evaluation permits histories to diverge naturally.

Proposed confirmation: 30 independent initializations/task × clean, registered
disturbance, held-out condition; all arms paired. Freeze final counts and values
using development variability, before looking at confirmation. Bootstrap independent
initializations/assets, not frames. Success is a positive paired complete-task gain
over the strong feedback baseline with a predeclared confidence interval, accompanied
by harm and cost; do not promise a positive outcome or use demo selection as evidence.

## R1 bounded development protocol (registered before actor execution)

Historical R0 runner/BDDL/design and simulator asset remain byte-identical. R1 uses
a new runner and local, process-scoped asset construction patch: enable the existing
0–2.7 rad hinge limits; initialize closed at zero rather than a negative angle;
use abs(angle)<=5 degrees as a **diagnostic closure** (full closure includes future
release/clearance/dwell checks). Store the generated compiled XML and source hashes.
Do not change door inertia, contact geometry or force thresholds silently.

1. Inventory collision geometry, robot/gripper pose, shelf/item support and hinge
   anchor/axis. Inspect robot-facing versus camera-facing access. Report bounding
   extents and contact evidence, not visual reachability alone.
2. Bounded empty-gripper XYZ pulses and feedback servo to pregrasp/handle approach
   poses test action direction and collision obstruction. Any scripted manipulation
   is explicitly an engineering diagnostic, never a VLA or framework success.
3. Fixed prompt comparison on seeds 0,1,2, 160 actor queries each: closed door with
   canonical instruction vs `open the fridge`; open door with canonical instruction
   vs `pick up the cream cheese box`. Same initial scene/physics/actor for paired
   prompts, fresh reset, no scripted actor prefix. Run all 12 cases without selection.
   This isolates only the first skills; whole-task completion remains unscored.
4. Freeze the geometry/camera variant after the engineering checks and before these
   actor trials. Later geometry/prompt changes get a new version/cohort. Preserve
   invalid setups, incomplete jobs and failures separately.

R1 measurements: hinge progress, EEF/item displacement, pad-item contacts, actual
grasp predicate, item/hand relative movement, support, action/token distribution,
inference latency and complete action/state/image tapes. Independent action replay
checks physics/controller/image fingerprints; it does not certify real-world physics.
Every job has an external wall-time bound and detached supervisor, healthy CUDA/EGL
device pairing, one actor per GPU, inference batch 1 (all future microbatches <=8).

Decision after the fixed batch: physical obstruction → version and revalidate scene;
servo viable but all nominal prompts fail → task-skill adaptation/training with
declared data and a shared frozen actor, rather than unbounded prompt search;
basic skills viable → full-sequence development and evaluator fixtures, then R3.
No final garment evaluation starts before refrigerator basic-skill qualification.

### R1 actor-scene freeze after geometry diagnostics, before actor queries

Use yaw=-pi/2 (opening toward the robot), fridge center y=0.35 m, agent camera
position (-0.55,0.85,1.55) looking at (0,0.35,1.05). Closed hinge=0, open extraction
diagnostic=130 degrees; original 0–2.7 rad range now enforced. The item region is
shifted +0.04 m in local base x toward the shelf lip and +0.035 m in local base z
away from the hinge side. These are declared development layout changes; the item
remains supported on the shelf. No per-case actor-outcome tuning is allowed.

Geometry trials v1–v4 remain retained: door interference, orientation/IK branch
limitations and unsuccessful grasp. v5's real contact servo establishes bilateral
pad grasp, shelf clearance/lift and 0.1737 m outward transport while retaining the
item, without changing the door angle. This is a **scripted physical feasibility
probe**, not actor success, full task completion or a learned repair. Door opening
skill remains unqualified. R1 actor trials have no servo prefix and retain all 12
cases regardless of success. They diagnose opening and extraction independently.

## Demonstration quality target

Engineering mini-fridge/single-fold probes are milestones. The headline should show
an uncut full task, meaningful contact and phase transitions, an observed failure,
bounded correction, control handoff and remaining-task completion. Expand object,
layout and initial-shape diversity only after basic capability; do not add visual
clutter to disguise weak manipulation. Name tabletop versus mobile navigation and
towel versus T-shirt precisely. Publish same-speed actor-left/framework-right MP4s
with simulated and wall-clock timing, an all-case index, and negative cases. A paper
needs measured cross-case recovery beyond the strongest selected clip.
