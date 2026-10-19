# Runtime verification–repair: continuity, visual grounding and ICLR evidence target

## User clarification and scientific claim

The central theme remains an external **runtime verification and repair framework**
for frozen VLA policies. Preserve the five layers: subgoals → frozen VLA → risk/event
monitor → dual-mode gate → local repair. Verification chooses when to accept an
action, whether a repair restored the required conditions, and when to return control.
Differentiable prediction is one possible evidence source, not the definition of
verification. This clarification supersedes wording that suggests verification was
replaced by a stand-alone recovery controller. The R1 launch protocol is unchanged.

Testable hypothesis: under equal observations and budgets, checking phase-specific
physical premises and revalidating them before handoff improves complete-task success
over capable reactive feedback and planning baselines, without excessive intervention
or progress loss. Five layers, external rules and a finite-state machine alone are
not established novelty. Compare the mechanism with runtime assurance/shielding,
task-and-motion planning, behavior trees, failure monitoring and VLA repair literature.

Do not argue that a probabilistic policy cannot perform precise manipulation. It can;
the question is when explicit external checks add complementary value. Perception,
constraints and repair can also be wrong. Unless assumptions and proofs are supplied,
runtime verification means empirical condition checking, not formal certification.

## Exact boundary of current refrigerator evidence

* Native actor runs use agentview RGB. MuJoCo supplies physical articulation, contact
  and friction. This is not yet an integrated visual-physical verifier.
* `side_grasp_v5_pixels` runs fixed waypoint targets and fixed phase durations with
  privileged EEF pose feedback. It neither identifies an object from pixels nor
  estimates its pose online. Grasp/contact measurements are recorded rather than
  controlling phase transitions or VLA handoff. No task-compatible learned physics
  predictor or predictor-based action screening is active in this run.
* Its stable contact extraction is useful engineering evidence. It is not a full
  opening/retrieval/placement/closure episode, a multi-object localization result or
  a VLA-plus-wrapper efficacy result.
* A2 establishes intervention gains in one assembly continuation setting (8/20 versus
  18/20), with one harm. Shared predictive gating prevents interpreting its feedback
  arm as prediction-free. Phase/prediction-ranking advantages are not established.
* Refrigerator retrieval and cabinet placement share rigid-body grasp, transport,
  clearance and support reasoning. Opening a hinged door, extracting through a shelf
  and releasing into stable support require different predicates. Shared structure
  motivates transfer tests; it does not itself establish transfer.

## Proposed verifier interface and closed-loop mechanism (not implemented)

Each decision consumes a timestamped observation-derived state estimate, uncertainty,
current phase, proposed action and remaining budget. Each condition returns
`satisfied / violated / unknown`, with evidence source, margin and observation age.
Unknown does not count as satisfied. Use a bounded re-observation or retreat when
appropriate, record its cost, and terminate explicitly if the budget is exhausted.

Candidate phase contracts:

| Transition | Evidence to check | Bounded repair candidate | Revalidation before handoff |
|---|---|---|---|
| Approach → open | Handle location/uncertainty, approach clearance, feasible hinge motion | Re-align grip or change approach | Sufficient observed door aperture and free extraction path |
| Approach → grasp | Correct object identity, pose uncertainty, jaw/shelf clearance | Reposition and regrasp | Retention under a short lift; two-sided evidence if sensors permit |
| Grasp → extract | Retention, object/shelf clearance, progress | Reduce motion, back off, recover grip | Object retained and outside the obstructed region |
| Transport → release | Target support and release clearance | Align placement | Released, supported, stationary over a dwell interval |
| Close → finish | Door sweep free of hand/object | Withdraw, retry closure | Door closed, object retained at destination, arm clear and dwell |

Predicates use the declared sensor contract. Simulator contact pairs and true poses
are privileged unless explicitly a separate oracle arm. Geometric margins must include
estimated uncertainty; they are not collision guarantees under arbitrary model error.
Repair consumes a bounded budget and returns to a freshly queried, phase-valid actor
proposal. Measure repair occupancy and the remaining task completed by the VLA to
detect a wrapper that has become a full scripted task solver.

## Multi-object visual grounding development ladder

1. Re-estimate the single object's location from RGB-D or calibrated multiview images;
   randomize its supported position without hard-coded target coordinates. Keep an
   oracle-pose arm to separate perception from manipulation error. Choose and freeze
   the sensor contract before comparative trials; RGB-D is a proposed addition, not
   a current actor capability.
2. Add three objects with distinguishable semantics; vary target identity, position
   and ordering. Language grounding → instance mask/track → 3D pose or grasp region
   plus uncertainty → collision-aware approach → feedback retention verification.
3. Add up to five objects where shelf capacity and gripper access permit. Include
   similar-looking instances, partial occlusion and held-out layouts. Fully occluded
   targets require information gathering; do not invent precise locations. Moving a
   blocker is a separately specified subtask with its own budget and baseline access.
4. Score correct-target retrieval, wrong-object grasp, localization/rotation error
   where identifiable, uncertainty coverage, distractor displacement/drop and full
   task completion. Define precision against jaw/clearance tolerance; do not promise
   millimetre accuracy from the existing video. Use true poses only for offline scoring.

Layouts are validated for initial support and feasible access before actor outcomes.
Development and evaluation objects/layouts are disjoint on named axes; no tuning on
confirmation failures. A task-appropriate observation adapter must be shared by all
arms that need it, or its extra information must be isolated by an explicit ablation.

## ICLR-quality evidence program

The official [ICLR 2027 reviewer guide](https://www.iclr.cc/Conferences/2027/ReviewerGuidelines)
(consulted 19 Oct 2026) emphasizes a specific question, motivation/positioning, correct
and scientifically rigorous support, and significant new knowledge. It explicitly
does not require state-of-the-art scores. The targets below are our research design,
not a conference checklist or acceptance guarantee.

| Evidence target | Required comparison or artifact |
|---|---|
| Mechanism beyond a task script | Shared condition/uncertainty/budget/handoff engine; disclosed per-task predicates and primitives; strong behavior-tree/reactive baseline using the same primitives |
| Strong nominal capability | Frozen task-capable actor shared across arms; shared phase prompts; declared adaptation data; complete nominal task runs before interpreting recovery |
| Increment beyond more compute | Match action/query/time limits and observation permissions; task-compatible replanning baseline, not redundant H1 queries |
| Causal attribution | Same repair/different triggers; common trigger states/different repairs; immediate versus revalidated handoff; progress-memory ablation; downstream completion and harm |
| Visual and physical contribution | Oracle versus estimated state; visual-only versus visual + geometric/contact checks with common sensor access; prediction added only with a compatible predictor |
| Breadth of mechanism | Refrigerator plus cloth/garment primary families; assembly mechanism study; second frozen policy backbone where integration permits |
| Meaningful generalization | Held-out objects, layouts, physical parameters and occlusion strata; adapter/data cost disclosed; no unqualified zero-shot claim |
| Credible effect size | Paired outcomes and clustered confidence intervals; independent seeds/assets; all failures; sample sizes frozen from development precision/power requirements |
| Deployment relevance | Physics continues during inference; latency/stale observations and exclusive command ownership measured; pursue real-robot replication of representative recovery events |
| Reproducible demonstration | Full resets, uncut same-speed paired videos, all-case index, measured time, source/protocol hashes and independently restored artifacts |

Two task families alone do not ensure breadth; multiple controlled variants and
mechanistic attribution matter. A second backbone and real hardware strengthen this
project's claim but are not universal ICLR requirements. A simulator-only paper needs
especially convincing generality, uncertainty tests and nonprivileged observations.
Prediction fidelity must translate into decisions and downstream benefit, rather
than merely adding a physical-model module to the architecture diagram.

## Execution order and decision gates

1. Finish the frozen 12-case R1 cohort and audit action/interface failures. Do not tune
   its prompts/layout from partial outcomes. If basic skill remains absent, move to
   declared task training or a capable frozen actor shared across comparisons.
2. Qualify physical handle opening separately, then a complete actor trajectory and
   stable endpoint fixtures. Retain the extraction servo as a feasibility reference.
3. Build observation-derived localization and the three-valued verifier; begin with
   measured-feedback checks and bounded regrasp/clearance repair. Test revalidation
   and handoff against a same-primitives reactive controller.
4. Expand to three/five objects and held-out conditions; advance cloth physical grasp
   and folding after refrigerator basic-skill qualification, reusing the same core.
5. Freeze confirmation, execute causal ablations and full episodes, then second-backbone
   and clocked/real-robot evidence. Favor the strongest established claim over adding
   modules without incremental benefit.

This document is a new design target; none of its proposed visual localization,
tri-valued contracts, full refrigerator controller or hardware results is claimed
implemented by the existing R1 probes.
