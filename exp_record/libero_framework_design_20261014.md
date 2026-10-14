# Task-grounded framework comparison and architecture decision — 14 Oct 2026

## Current evidence, not an efficacy claim

All baseline/continuation jobs have finished; no learned verifier has been evaluated on this robot task. Frozen full OpenVLA native H1/k1 reaches native success on3/3development roots, but released seated1second completion is1/3 (descriptive Wilson95%6.15–79.23%; fixed development cases). The3/3native metric is not a valid substitute. Root0requires313steps; root1never fits the full collision-box footprint; root2meets all placement conditions for0.3seconds then resumes book/gripper contact. Exact prefix replay rules out the new measurement code as the cause of changed outcomes. Safety/CVR remain unmeasured, not zero.

This makes the task useful for testing **alignment and preservation of completed placement**, but it does not yet guarantee usable intervention headroom or learned efficacy. A successful scientific design can reject the framework as well as support it. Do not promise a positive result or tune the acceptance definition until it appears.

## Delivered review and causal diagnostics

`scripts/render_libero_review.py` produces every trajectory as an annotated2x MP4 plus a three-panel overview. `scripts/summarize_libero_continuation.py` now invokes it automatically after full artifact verification. Playback is2x **simulated time**, not inference wall time: original20fps frames are retained and encoded at40fps. Overview duration13.025seconds; shorter root0freezes with an explicit ENDED label. These are different initial states under the same actor, not a baseline-versus-framework montage.

Run review directory: `/home/ami/scratch/libero_continuation_20261013_a/review/`. Files: `all_roots_2x.mp4`, `root_0_2x.mp4`, `root_1_2x.mp4`, `root_2_2x.mp4`, `index.html`, `review.json`, and per-root causal diagnostic timelines. Source framejcorresponds to the state after actionj−1; overlays use that row, never future rowj. Video frame counts/fps and hashes were checked and a representative rendered frame inspected. Browser preview is unavailable in the current disconnected desktop session; local video files are complete.

`src/adapters/libero_diagnostics.py` uses only current/past logged measurements. It reports signed collision-box footprint margin, floor error, valid-placement preservation opportunity, re-contact after valid placement, and sustained slow-contact misalignment. These are **privileged diagnostic rules**, not a deployed image-based detector or a learned predictor.

-Root2has a current-state placement-preservation opportunity at zero-based step188; first observed re-contact at194. There are five intervening actor actions before the action at194produces the first renewed-contact interval. This is an intervention opportunity, not6step-ahead predictive accuracy.
-Root0has placement opportunities beginning283and no subsequent re-contact before termination.
-The conservative slow-contact stall rule does **not** flag root1: its qualifying streak never reaches20intervals (final16), despite clear cumulative misplacement. Therefore low instantaneous speed/one dwell threshold is not a sufficient general detector. No threshold was swept to force retrospective coverage. A task-error trend over measured history is the next hypothesis to test.
-Root2's final footprint excess≈0.038mm is too small to treat as a robust standalone failure; persistent gripper contact and insufficient dwell independently fail acceptance.

## What should be detected and what should change

| Observed failure | Useful measurements/prediction | Smallest intervention hypothesis | Essential comparator |
|---|---|---|---|
| Root1: misaligned/tilted book, persistent grasp/contact, no complete footprint containment | Relative box/slot pose, signed face clearances, approach velocity, contact mode and **task-error trend** over measured history | Bounded lateral/orientation correction, or unloading/re-approach if the contact mode makes direct correction harmful | Same-state geometric alignment rule, not only unmodified VLA |
| Root2: placement briefly valid, then later commands contact it again | Gripper/book separation, approach direction, predicted re-contact and loss of placement under the next candidate/control prefix | Small withdrawal/hold/open-gripper candidate where appropriate; continued monitoring through dwell and actor resumption | Simple phase-aware hold/retreat/termination logic |
| Contact force/impulse excursions | Pair-resolved normal/tangential force, integrated contact impulse, slip and penetration at physics substeps | Contact-mode-dependent correction that preserves task progress | Matched slowing/force-limiting control; lowering force while abandoning placement is not a success |

Do not prevent necessary grasp/support contact. Do not assume that every hold or opening action is safe. The zero-correction original proposal must remain an explicit candidate; restoration to the actor and completion must be evaluated, not just local contact suppression.

## A comparison capable of identifying the framework's contribution

Use a single frozen task-capable actor, controller, task geometry and task acceptance definition. Separate the following evidence stages; never put privileged oracle numbers in the learned result column.

### Stage1: bounded privileged mechanism diagnostic — recommended next execution

Use exact simulator geometry/dynamics for **all control comparators that use state**. This asks whether physical lookahead can choose useful bounded corrections beyond a simple rule. Current3roots locate mechanisms only; a fresh, preregistered small development cohort must check that the mechanism is not specific to root2. Use causal phase triggers computed online, not hardcoded episode IDs, successful future labels or the observed step194.

Four logical arms for attribution:

1. **A — frozen VLA continuation.** Same completion/time budget; native predicate is not a stop rule.
2. **B — matched simple phase/geometric controller.** Same current-state/history estimates, phase detector, correction candidates and decision opportunities; rank by current geometry/contact rules, with no future physics. Include an explicitly validated hold/retreat strategy rather than a weak “slow everything” baseline.
3. **O — physics-lookahead selector.** Same actor, phase detector and finite candidate family, but rank candidates using exact predicted future task/contact evolution. This is privileged mechanism evidence only. If Odoes not beat Bunder complete execution, refine the intervention formulation; another learned risk head is not the next move.
4. **L — learned physical selector**, only after Stage1headroom: same candidate family, acceptance/calibration and budgets as O/B, with an allowed-input learned predictor. This arm does not exist yet and must not be filled with oracle scores or an untrained placeholder.

Keep a small fixed candidate budget (design targetK=6, not a sweep): unchanged proposal, reduced pose increment, hold, bounded separation/retreat, bounded alignment correction, and a phase-appropriate gripper/pose variant. Exact bounds, duration, coordinate transformations and the gripper conditions must pass controller/action tests and be frozen before rollout. These are controller candidates, not purported future native VLA actions. Do not expand the candidate set only for the proposed method. Bmust see the same candidates; matching candidate enumeration does not mean Buses oracle rollout outputs. Report actual simulator/model/VLA queries separately, including any extra oracle queries.

Branch restoration must preserve controller goals, gripper state, simulation warm-start/internal integration state, environment counters and estimator history. Stored qpos/qvel alone is insufficient. A safe initial implementation can deterministically replay the complete executed action prefix from the same initial state into each branch, thereby reconstructing controller history, and assert state/action-prefix identity. Branch evaluation may access only information available by the decision; future baseline action tapes cannot stand in for future actions after an intervention. Resume the real actor on the resulting images and run to the same terminal criteria/budget.

Stage1must distinguish three possibilities: (i) simple phase handling already repairs the failure, so the contribution is not learned physics; (ii) physics selection adds complete-task benefit beyond the simple control, justifying a learned replacement; (iii) all corrections lose progress/resumption quality, requiring controller/repair redesign. Oracle gains establish opportunity, not guaranteed learnability or a global bound.

### Stage2: learned comparison under realistic observations

The current VLA consumes agent-view RGB only; geometry/contact values in our logs are simulator diagnostics. A learned runtime must obtain its task/contact belief from explicitly allowed images, robot proprioception/gripper measurements and any genuinely available contact sensor. True object pose, collision lists, hidden mass/friction and regime IDs cannot silently become runtime features. If wrist RGB, depth or force sensing is added, expose the same information/history to the simple control and predictor ablations. Compare Lto Bon identical state estimates to isolate prediction benefit; Ais the reference task policy, not sufficient alone to rule out added-sensor benefits.

Training labels may use simulator state/contact, but training inputs must follow the deployed estimator path. Collect **fresh training roots** covering ordinary grasp, support, placement, natural failures and bounded intervention/resumption; do not reuse the old0/1/2audit branches as training supervision. Split related branches/perturbations by root. Fit normalizers/calibration on their own registered splits. Physics parameters should be hidden from runtime input even when labels are simulated.

Before efficacy, establish contact-label stability: timestep/solver sensitivity, effective contact compliance (MuJoCo can clamp time constants relative to timestep), contact-pair force/impulse definitions and mechanically justified limits. Changing timestep while inadvertently changing effective compliance confounds numerical and physical effects. A raw force spike or a retrospectively chosen threshold is not a safety label. Until then report released completion and diagnostic contact distributions, with CVR/safe explicitly null.

Nominal feasibility precedes bounded observation/execution/dynamics disturbances. Preregister amplitudes, single-source and combined cases, shared exogenous noise schedules and estimator/control access. Input-pose corruption is a sensor proxy, not visual perception robustness. Use noise to test robustness, not to manufacture a weak baseline.

### Formal protocol compatibility still open

Current robot pilots use genuine nativeH1/k1. They do not satisfy the historical genuineH10/k5 framework protocol; repeating an H1action does not fix that. The immediate H1diagnostic can establish task/control headroom, but must remain labeled as such. Before formal learned-framework testing, establish a genuine task-compatible H10proposal adapter (and a capable robot-task baseline at k5), or obtain an explicit protocol revision with all methods re-registered under the new cadence. Changing cadence only for the framework is forbidden. Keep full7B and inference microbatch≤8; batch1has been used here.

## Evidence-driven architecture adjustment

The old architecture should not simply be widened or retuned for this robot:

1. `models/immediate_gate.py` hardcodes10D planar observations/2D actions. Its repeated-current-context R5event head is not a7D Panda verifier. `models/nets.py:OrbiSimDynamics.rollout` rolls out state but scores each action's risk against the same measured context, so risk does not follow preceding candidate actions/contact transitions. That is poorly suited to “placed now, hit again several steps later.” Preserve it as a legacy reference, not a silently transplanted method.
2. **First proposed model change: measured-history-conditioned residual physical prediction.** Start small: a fixed short window of relative task geometry/velocities, EEF/gripper response and permitted contact evidence, with uncertainty, then a causal action-conditioned rollout. A concatenated history MLP is a baseline; a GRU is justified only if it beats equal-input/history controls. Separate estimation of current contact mode from evolution under candidate actions. Hidden friction/mass may affect the learned belief through observed responses, not via simulator tags.
3. Predict observable/mechanically meaningful consequences: relative pose/velocity increments, clearance/progress, contact-mode transitions/re-contact and calibrated force/impulse summaries where labels are valid. Use known coordinate/kinematic structure plus learned residuals; handle rotations and per-channel action units explicitly. Avoid one scalar risk score that confuses helpful support contact with harmful collision.
4. **Repair change: bounded6D pose candidates plus an explicit discrete gripper branch.** Old scalar amplitude scaling cannot turn a misalignment into a lateral/orientation correction or preserve an already seated book. Keep trigger probability, continuous physical prediction and candidate ranking in separate units. Do not feed event probabilities into legacy repair constants. Use finite candidate selection first; gradient-based repair remains conditional on useful-gradient checks near contact and equal candidate budgets.
5. **Continuation-aware acceptance.** Predict whether correction preserves progress and the ensuing actor execution, not only whether the next prefix looks quiet. Retain the measured dwell/contact-mode state across decisions, and investigate a separately trained continuation-value/resumption signal only if one-step/short-rollout prediction still selects locally safe but globally harmful actions. An artificial terminal reward or a distance penalty alone does not demonstrate this property.

Do not change the actor, estimator, physics objective, candidate family and termination rule together in the main comparison. Freeze the actor first; ablate measured history and physical prediction after demonstrating a usable full-system benefit. If actor fine-tuning becomes necessary for task capability, include the same fine-tuned actor alone as the baseline and re-freeze it before attributing framework gains.

## Acceptance and data sufficiency

Report task-native success, geometry-aware released/stable completion, validated CVR and safe completion, intervention frequency, complete paired rescue/harm, root-level uncertainty, and stress-cell results. Evaluate entire execution including repair recovery. Compare Lagainst both Aand B; gains only over Aweaken the physics-specific claim. Preserve the project targets (≥+3pp OOD safe completion, positive independent paired95% lower bound, bounded≤2pp SR loss and seed/cohort consistency) for the formal study, not these three development examples.

The official task provides50initial states (shape50×45), all exactly distinct; SHA256`6cd3937ba18d64e72223222af36888d26ea201840415adbc4dae590454d0a5a8`. Only0/1/2have been consumed here. Do not assume50states provide sufficient independent power for a3pp claim; perturbation replicas and candidate branches do not increase the number of independent roots. Register task-specific train/dev/calibration/confirmation roots and power/precision before learned efficacy testing. The historical toy final-test seed base remains untouched.

**Decision:** next active research deliverable is one frozen bounded privileged-vs-simple-control mechanism protocol on this task, plus contact-label/observation-interface validation. The current deliverable is reviewed evidence, executable causal diagnostics and automatic accelerated video. No new training, actor fine-tuning or claimed framework-efficacy rollout was launched in this review.
