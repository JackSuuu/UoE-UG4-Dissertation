# U-series: unstructured manipulation, data provenance and OOD claims

## User decision and current capability

User requests cutting vegetables, peeling garlic, handling raw meat, retrieving
objects from a refrigerator, washing dishes and folding clothes. These are the new
evaluation targets, superseding the suggested three LIBERO tasks as the primary
next suite. A proposed LIBERO long-task pilot was not launched before this steering.
No further square-nut tuning is the active priority.

The five-layer architecture remains subgoals → frozen VLA → risk/event monitoring
→ nominal/contact-transition gating → bounded local repair. Its *structure* can be
reused; its current implementation is not a general household controller:

* Actor: frozen full7B OpenVLA finetuned on LIBERO10, native H1. No validated cutting,
  peeling, tissue, refrigerator, cleaning or garment skill in this local integration.
* A1 predictor: supervised self-developed40D nut/peg state and7D action model,
  trained on6400 simulated transitions (3840 training subset); no rotational actions
  in collection. This is task-distribution learning, not a general physical model.
* A2: same learned checkpoint, no new training; measured geometry is default,
  prediction can override only after residual/uncertainty/risk/progress checks.
  Its phases, clearance, seating, candidate set and thresholds are assembly-specific.
  Its eleven candidates cannot represent peeling, cutting paths or garment folding.
* Ground-truth object/contact access is privileged. RGB-only perception robustness
  and general task-conditioned repair are not implemented.
* A2 has18/20complete outcomes vs VLA8/20, but prediction-ranking overrides are zero.
  Shared-predictive-gate geometry also18/20, at lower mean peak load and repair cost.
  No predictor-free gate control was tested. Neither model-specific improvement,
  task-family generalization nor20Hz real-time operation follows from these results.

## Does the framework require data?

There are three different uses of data:
1. **Evaluation logging:** record frozen-policy states/actions/outcomes to measure
   behavior. This alone is not training, but inspecting target outcomes to choose
   thresholds, prompts, candidates, model versions or tasks consumes development data.
2. **Model fitting/calibration:** a learned dynamics/risk model needs transitions and
   calibration examples, not necessarily demonstrations or expert action labels.
   Self-collected simulation experience is still data. A frozen pretrained model
   inherits its training exposure; "no new training" never means "no data".
3. **Task adaptation:** target trajectories used for fitting, threshold calibration,
   handcrafted rules or candidate design are target adaptation even if VLA weights
   stay frozen. This can be useful but is not zero-shot transfer to that task.

Runtime repair differs from policy fine-tuning in *where* it changes behavior:
intercept and validate the next action, execute a bounded correction, then hand back
to the same frozen actor. This is an architectural difference, not an automatic
generalization guarantee. Report both runtime transfer and any adaptation cost.

## OOD contract: state exactly what is unseen, and to which component

Keep a provenance table for actor pretraining, predictor training, calibration,
controller development and final evaluation. Foundation-model pretraining overlap
may be unknown; then claim only "held out from our repair development/training",
not "never seen by the robot model". LIBERO10 tasks are not unseen task families
for the current LIBERO10-finetuned actor.

| Evaluation | Permitted target exposure before freeze | Defensible claim |
|---|---|---|
| New seeds in familiar task | task-specific development | same-task robustness |
| Held-out objects/layout/material range | source instances, disjoint assets/parameters | explicitly named instance/parameter OOD |
| Held-out task family | no trajectories, tuned rules or target-specific repair primitives from that family | repair-family transfer, conditional on actor provenance |
| Target-trained predictor/controller | declared target episode budget | adaptation, not zero-shot |
| Unknown actor pretraining overlap | must be declared | component-relative OOD only |

Before a manipulation cohort, freeze actor, perception, state interface, predictor,
gate, candidates, thresholds, prompts, budgets and reset rules. Evaluator goal
definitions may be task-specific but must not leak privileged completion labels
into a nominally RGB-only controller. A task description at inference is allowed;
new target-specific action code is adaptation, not just a language instruction.
No target simulator candidate branching online in learned-model arms.

Source/dev/calibration/test IDs must be disjoint at trajectory **and** asset/family
level, according to the named shift. Do not split overlapping windows at random.
All six families can be explored as development now, but then final claims must use
unseen instances/conditions. To claim unseen-family transfer, reserve a family
before controller work or run separately frozen leave-one-family-out development
and training; merely relabeling already explored tasks as OOD is invalid.

Frozen-transfer and optional target-adaptation results will be separate tables.
If adaptation is later authorized, register a fixed episode budget beforehand and
give comparators the same data. Episode memory resets between evaluation episodes;
cross-episode updates require a separately named online-adaptation condition.

## Six requested tasks: measurable physical content and missing prerequisites

| Family | Full sequence and physical evidence required | Current local readiness |
|---|---|---|
| Refrigerator retrieval | approach closed door → open → reach among clutter → grasp unseen item → extract → place/release → close; door joints, item retention and collision contacts | no refrigerator environment or compatible mobile/arm policy adapter installed |
| Garment folding | locate/grasp cloth → unfold/spread → align/fold → release; mesh configuration, layer/edge alignment, self-contact and settled final shape | Genesis PBD cloth backend executable; no garment task, validated gripper coupling, perception or VLA folding skill |
| Dish washing | acquire dish/tool → contact-guided scrub/rinse → measure residue removal → place/release | no validated residue-removal/water/tool model; moving a dish or setting a `clean` flag is insufficient |
| Vegetable cutting | grasp/position tool and vegetable → successive cuts → separated pieces meeting size/retention specification | no validated fracture/cutting model, calibrated tool forces or task-compatible policy |
| Garlic peeling | stabilize clove → initiate peel → propagate separation → remove peel while retaining flesh | no layer/adhesion/separation assets or contact model; object scaling/deletion is not peeling |
| Raw-meat handling | first define manipulation: deformable grasp/transfer/portion placement; cutting/trimming is a separate scope | no calibrated tissue/slip/adhesion model; a soft-body proxy cannot establish biological realism |

First integration priorities are articulated refrigerator retrieval and deformable
cloth handling/folding, because they exercise distinct failures without requiring
an unimplemented fracture or adhesion model. All six remain in the suite; unsupported
tasks are marked blocked, not silently replaced with rigid-box surrogates or scored
as policy failures. Task horizons, geometric tolerances and event thresholds must
be frozen after metric fixtures, before comparing controller outcomes.

## Experiments and controls

**U0 (executed first): backend qualification.** Inspect installed backends and run
actual Genesis cloth dynamics. A passive drape is a backend test, not folding or a
VLA/hybrid trial. Record failure logs and independent-process numerical differences.
This is separate from target task-training data and from evaluation scores.

**U1 (next, prerequisite-dependent): full-reset actor capability and monitor audit.**
Build refrigerator and cloth environments with metric fixtures and compatible
actor/action mappings. Use naturally occurring grasps/slips/stalls, full-task
rollouts and all outcomes. Do not start from scripted successful grasps as A1 did.
Baseline inability to perform a skill is a real limitation; a correction wrapper
must not quietly substitute a scripted complete-task policy. No target holdout is
used to decide thresholds. U1 trajectories used for design are development data.

**U2: paired repair benefit.** Same initial objects/materials/disturbance seeds,
sensor access, query/control/wall-time budgets and terminal rules across frozen VLA,
generic measured-feedback repair, and full hybrid. Share gates/candidates for
ranking attribution, and separately compare full systems. Include a predictor-free
gate arm when attributing prediction. Report full completion, local recovery,
remaining-task completion after handoff, new harms, and unrecovered failures.
Disturbances are evaluator-controlled forces, slips, pose/visual changes or delays;
injection identity/timing is hidden from the controller. All cases remain in denominators.

**U3: OOD evaluation.** Freeze on source/dev/calibration only; measure named held-out
object/material/layout/family shifts without target tuning. Include benign cases and
both-failure cases. Publish paired confidence intervals clustered by independent
scene/object/seed, not thousands of correlated action steps. A successful resumption
must lead to useful remaining-task progress and final completion, not just low force.

## Real-time recovery is a separate experiment

Current VLA median is~0.71s/action; A2decision computation~4.5ms excludes VLA and
perception/sensing. A20Hz simulator and2x MP4 do not establish real-time control.
Proposed architecture keeps VLA as a slow proposal source and runs fresh-feedback
monitor/local repair on a separate fast loop; this is **not yet implemented**.
First register deadlines and max observation/action age from the selected task,
then measure timestamped sensing→detection→command→physical response, p50/p95/p99,
deadline misses, stale-proposal discards and recovery time. In clocked tests the
world advances while inference runs; the synchronous pause-and-query result is a
separate condition. Compare equal wall time and actor query budgets.

## Sources inspected 19 Oct 2026

* Local Genesis1.4.2: `engine/materials/PBD/cloth.py`, `engine/entities/pbd_entity.py`,
  `options/solvers.py` confirm deformable-particle APIs, not household robot skills.
* [BEHAVIOR installation](https://behavior.stanford.edu/getting_started/installation.html):
  OmniGibson/IsaacSim and datasets require a separate environment; none is installed
  here. Installation and assets alone do not provide a matching frozen actor.
* [OmniGibson transition rules](https://behavior.stanford.edu/omnigibson/transition_rules.html):
  slicing removes an object and spawns halves upon rule conditions. Such a benchmark
  can measure task sequencing but cannot establish continuum fracture/contact-force
  prediction. Apply the same fidelity distinction to cleaning/material transformations.

## Videos and records

Default human-review artifact is same-case **VLA left / hybrid right** MP4, aligned
by decision index/simulated time or explicitly labeled wall time for clocked tests.
Use identical playback speed; label completed-panel held frames, phase, route,
observed event, correction count, final outcome and latency when available.
Single-arm raw recordings remain evidence, but do not label a baseline-only or
backend-only probe as a paired efficacy video. Publish representative rescue,
preservation, harm and both-failure videos plus an all-case manifest through Git LFS.
Use `scripts/render_paired_review.py` for the current verified A2 schema; future
runners must export its pair manifest or an explicitly documented equivalent.
