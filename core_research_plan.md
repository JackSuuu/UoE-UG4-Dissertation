# Core Research Plan: Verification and Repair for Frozen VLAs

**Period:** October 2026 – September 2027

**Updated:** 21 October 2026

**Registered dissertation topic:** *Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization*

> **Research subject:** a runtime verifier coupled to bounded, model-generated repair and verified policy resumption.
>
> **Primary evidence:** improved complete-task reliability, including rescues and harms, under matched resources.
>
> **Application domain:** unstructured, multi-stage manipulation, initially microwave and refrigerator tasks.
>
> **Physical prediction:** an explicit research component inside the verifier, alongside measured feedback and task constraints.

This document restores the comprehensive structure of the [earlier research plan](https://github.com/JackSuuu/UoE-UG4-Dissertation/blob/465e0a2/core_research_plan.md): contribution choice, research questions, timeline, phase details, metrics, implementation, thesis structure, risks and immediate steps. It incorporates subsequent evidence and scope decisions. The [current experiment record](exp_record/agent_notes/current_goal_and_experiment_queue_20261019.md) holds executable protocols, frozen configurations, process status and results.

## 0. Research Framing and Scope

### 0.1 Why verification and repair remain the main line

A frozen VLA may possess the actions needed for a task yet fail to recognize a bad contact transition, distinguish intentional release from grasp loss, preserve a completed stage, or resume useful behavior after correction. A runtime verifier should identify when intervention is warranted and whether a proposed or executed recovery is adequate. Repair should respond to that evidence while retaining the ability to finish the task.

The research problem is therefore the **coupling between verification, intervention and recovery**, rather than task completion considered in isolation. A detector can be accurate but intervene too late; a local correction can appear successful but damage later stages; repeated repairs can exhaust the budget. These failures motivate studying the entire loop.

Unstructured manipulation provides the setting in which that loop matters. Complete-task success supplies the strongest outcome measure. Neither replaces the verifier–repair mechanism as the methodological contribution.

### 0.2 What “unstructured” means in this project

The target conditions include variation in object position/orientation, receptacle layout, contact state and physical properties, together with multi-stage tasks whose intermediate transitions are not perfectly repeatable. Each evaluated variation must be defined and recorded.

Native LIBERO microwave initial states currently provide a **controlled development benchmark**. The existing custom refrigerator scene is another controlled setting. Neither alone establishes robustness to arbitrary environments. Claims about unstructured generalization require reserved initial states, declared shifts and task transfer with the method frozen.

### 0.3 Scope commitments

- Retain the five-layer architecture and OpenVLA as the immediate backbone.
- Use full paired task episodes as the primary experimental unit; investigate failures from those trajectories.
- Generate every numerical task and repair action with the model. Geometry and contact measurements can verify behavior, but do not supply handwritten approach routes, hinge servos or gripper overrides.
- Develop physical prediction within the existing verifier interface and assess its actual value. Measured verification remains a legitimate configuration while task-specific prediction is being adapted.
- Keep cloth, additional platforms and a second backbone outside the immediate execution queue. They are conditional extensions, not substitutes for improving the current method.
- Distinguish implemented components, research hypotheses and future milestones throughout the dissertation.

## 1. Main Contribution and Boundaries

### 1.1 Contribution hierarchy

| Direction | Role | Intended contribution | Required evidence |
|---|---|---|---|
| **A. Runtime verification coupled to repair** | **Main** | A phase-aware verification–repair–reverification loop with explicit recovery and resumption contracts | Complete paired tasks, component attribution, verified handoff outcomes and retained adverse cases |
| **B. Physical evidence and prediction** | **Core mechanism study within A** | Determine when prospective physical evidence improves intervention timing, acceptance or preservation beyond current/past measurements | Task-compatible forecasts, calibrated evaluation, actual decision influence and matched predictor-on/off comparisons |
| **C. Resource-aware implementation** | Supporting | Bound intervention cost and establish a measured latency/memory envelope | Query/action accounting, timing decomposition and comparisons under common limits |
| **D. Interpretable verification traces** | Supporting | An auditable chain from observed/predicted evidence to a decision, action and outcome | Reconstructable execution traces and independent replay |

The expected contribution is not established merely by connecting five modules. It must identify a useful verification/repair design principle, show where it helps and harms, and establish its limits through controlled experiments. Free-form reasoning generation, a new general-purpose VLA and a separate motion planner are not current deliverables.

### 1.2 Relationship to neural physics and prior work

Physical prediction provides prospective evidence: what a model-generated action may do to contact, containment, grasp stability or other declared physical conditions. Measured reverification checks what actually happened. Their interaction retains the dissertation's neural-physics motivation without assuming that every intervention needs a forecast.

The repository contains self-developed predictor and reference-verifier implementations inspired by prior work. `AssemblyDynamics` uses an assembly-specific state/action contract; `RefCheckVLA` is a local reference implementation. The official OrbiSim adapter is a skeleton. These are not evidence of a loaded official OrbiSim or CheckVLA system, nor of valid native microwave/fridge prediction.

Differentiability is a possible property of a predictor, not the definition of the verifier. Gradient-based repair, visual-world-model comparisons and asynchronous execution become research extensions only if they address an observed limitation and can be compared through the same framework.

### 1.3 Current evidence and its implications

| Evidence | Measured result | Consequence for the plan |
|---|---|---|
| First native microwave comparison | Bare VLA4/6; framework4/6; one rescue, one harm;12/12exact replays | The loop affects outcomes, but net completion benefit is not established |
| Its recovery transactions | All11exited unverified at their local cap; root4completed after an actual handoff | Local intervention and downstream success do not yet demonstrate verified recovery followed by successful resumption |
| Root3/root4trajectory analysis | Release/loss confusion, repeated failed reacquisition, changing acceptance targets and missing preservation of new progress | Revise monitoring and recovery contracts within the same controller |
| Historical refrigerator R2 | Closed-start arms never opened; open-start intervention grasped/extracted but did not finish | Handwritten geometric corrections demonstrate neither current model-generated repair nor full-task competence |
| Earlier assembly experiments | Feedback/hybrid gains were observed; predictive ranking added no demonstrated advantage | Measured-feedback gains cannot be attributed to physical prediction |

The first six native roots are reused development cases. Historical push/assembly results retain their original protocols and limitations; they cannot be pooled with current native evidence. See [published native comparison](backups/framework_comparison_20261019/README.md), [fridge R2](exp_record/agent_notes/fridge_r2_results_20261019.md) and [assembly results](exp_record/agent_notes/assembly_a2_results_20261019.md).

## 2. Core Research Question and Hypotheses

> **How should a runtime verifier be coupled to bounded, model-generated repair so that a frozen VLA can detect and recover from execution failures, preserve task progress, and resume reliably in unstructured multi-stage manipulation? When does physical prediction improve this loop beyond measured feedback?**

The original signal-quality → closed-loop-benefit → system-feasibility structure is retained. Closed-loop benefit is the headline; signal quality explains it, and system feasibility establishes its practical limits.

| Research question | Falsifiable hypothesis | Evaluation and possible rejection |
|---|---|---|
| **RQ1 — Verification quality and decision relevance**: Which current/past measurements, phase context and physical forecasts provide useful evidence for intervention and recovery acceptance? | **H1a:** phase-aware verification reduces inappropriate interventions during valid transitions without losing useful early interventions. **H1b:** task-compatible physical forecasts provide additional timely evidence in some contact regimes. | Compare false interventions, missed events, timing and erroneous recovery acceptance, then check their full-task consequences. Reject benefit if improved scores do not yield better decisions or only defer failures. |
| **RQ2 — Verification–repair coupling and closed-loop benefit**: Does the verifier–repair loop improve completion, and do preservation, reverification and actual handoff account for that benefit? | **H2a:** the complete framework improves paired task completion over bare VLA under common limits. **H2b:** explicit recovery contracts and reverification reduce premature exits, regressions and repeated ineffective repair relative to matched module ablations. | Measure net rescues minus harms, complete success, verified-recovery-to-handoff outcomes and intervention cost. Reject the proposed advantage if gains vanish under matched resources or local recovery fails to persist. |
| **RQ3 — Robustness and usable resource envelope**: Under which task variations, observation assumptions and budgets does the same verifier–repair method remain useful? | **H3:** a frozen configuration retains a positive completion/cost trade-off across declared held-out conditions without scene-specific repair routes. | Evaluate reserved states, task transfer and measured latency/memory/query costs. Narrow the claim if success requires retuning, privileged information unavailable in the target setting, or extra resources. |

**Target headline, not an established result:** the same frozen VLA completes more full tasks because verification guides effective, bounded repair and reliable resumption—not because it receives more actions, easier endpoints or a hidden scripted controller.

Physical safety is evaluated only where constraints and labels are valid. Full-task success is the current primary endpoint. A later **constraint-respecting success** measure can require both completion and absence of a preregistered violation, but it must be reported alongside ordinary success and cannot retrospectively replace native benchmark labels.

## 3. Timeline Overview

The dates below are planning targets for the original one-year horizon. Progress is judged by evidence, not by treating a launch or integration milestone as a scientific result. Documentation and failure analysis accompany every full-task batch.

| Phase | Calendar target | Main work | Deliverable / decision gate |
|---|---|---|---|
| **P1 — Establish and improve the real loop** | Oct–Nov2026 | Iterate the existing microwave controller; integrate refrigerator task adapters | Reproducible full paired trajectories; distinguish verified recovery, budget exit, actual handoff and completion |
| **P2 — Mechanism attribution** | Nov2026–Jan2027 | Within-framework monitoring, preservation, reverification and retry ablations | Evidence about which verifier–repair mechanisms explain net gains and harms |
| **P3 — Physical-prediction contribution** | Dec2026–Mar2027 | Adapt state/action contract, train/calibrate applicable predictor, integrate into verifier | Forecast fidelity and decision-use analysis plus a matched complete-task predictor-on/off comparison |
| **P4 — Robustness and transfer** | Feb–Jun2027 | Freeze a chosen configuration; evaluate reserved states and declared task/condition shifts | Held-out reliability results with task-wise uncertainty and failure categories |
| **P5 — Resource and external validation** | Apr–Jul2027 | Measure the usable compute envelope; conditional sensor/real-data or second-backbone validation | Latency/memory/cost results and an explicit external-validity boundary |
| **P6 — Consolidation and writing** | Jul–Sep2027 | Remaining attribution checks, reproducible release, thesis and all-case demos | Draft by late August; final thesis and evidence package by September |

Prediction preparation can proceed alongside framework iteration, but does not block the current full-task batch. Additional sweeps are motivated by observed failure modes rather than scheduled simply to fill the calendar. Allow approximately one month of buffer across later phases.

## 4. Phase Details and Experimental Design

### P1. The actual five-layer verification–repair loop

| Layer | Method responsibility | Observable output |
|---|---|---|
| **1. High-level subgoals** | Track prerequisites, stage transitions and achieved milestones | Active task goal, satisfied/violated/unknown predicates and transition reason |
| **2. Frozen VLA** | Generate nominal and repair actions from fresh observations and instructions | Checkpoint, input image, prompt, model output and unchanged executed action |
| **3. Risk/event monitor** | Detect progress loss, release, contact transition, stagnation and uncertainty; consume forecasts when available | Causal evidence, event timestamp, forecast provenance and missing inputs |
| **4. Dual-mode gate** | Choose nominal tracking or contact/recovery management | Entry/continuation/exit decision, action ownership and remaining resources |
| **5. Local repair and reverification** | Pursue a bounded recovery contract, preserve progress, and return control | Acceptance checks, preservation status, exit reason and actual nominal handoff |

#### Verifier contract

At time *t*, the verifier operates on available observation history, task-stage context, previous actions and, when implemented, action-conditioned forecasts. It assesses three related questions:

1. **Intervention:** is there sufficient evidence that continuing normally needs correction?
2. **Recovery:** has the local objective been satisfied while required progress is preserved?
3. **Resumption:** can the transaction exit, and what happens after the nominal policy actually acts?

Predicates can be satisfied, violated or unknown. Unknown evidence is recorded rather than silently converted into success. Verification outputs remain separate from the gate's budget-aware decision and from the VLA's action generation.

#### Repair and handoff contract

Each transaction records its entry evidence, recovery target, acceptance alternatives, preservation conditions and resource bounds. The recovery contract stays fixed while the action prompt may change with current observations. Acceptance streaks for different alternatives cannot be combined into a false pass.

Newly achieved durable milestones enter preservation checks. Unsuccessful repair cannot repeat merely because the same flag remains latched; a retry needs declared fresh evidence or changed context. Exits distinguish verified local recovery, unverified budget exhaustion, complete task success during repair and technical termination.

**A handoff exists only after a fresh nominal VLA action executes.** Verification success, a request to hand back, and completion after handoff are separate recorded events. Full native task success remains absorbing; extra success-dwell actions are not inserted into a benchmark that stops on first success.

#### Current microwave batch

| Item | Frozen configuration |
|---|---|
| Task | Put the yellow-and-white mug in the microwave and close it; original `In AND Close` endpoint |
| Cells | Roots0–5 × bare VLA / revised framework =12fresh full episodes |
| Actor | Frozen OpenVLA LIBERO-10, revision80970322773f81baa2e22fe495d0487b93a05cfa; TF4.40.1; greedy H1; seed0; batch1 |
| Inputs | Same native camera and initial full-task instruction; scoped repair language is the treatment |
| Lifetime limits |520model queries and520task actions plus10separate initialization noops;1,200s rollout excluding model load |
| Repair limits |32actions per transaction and160total, within the lifetime limits |
| Evidence | Privileged current/past measurements; learned prediction disabled |
| Verification of execution | Fresh exact replay per episode; saved model inputs/actions; six full paired videos |

Revision `ab48b63` implements release/loss handling, stable acceptance, milestone preservation and evidence-based retry in the existing controller. It passed16model-free checks and was launched on21October. This is a **bundled development iteration**, so its result cannot isolate the causal contribution of each code change. Live status and exact thresholds remain in the current record.

#### Refrigerator follow-on

Adapt task observations, goal predicates and legal instructions while reusing the gate, transaction and resumption design. The intended closed-start, empty-hand sequence is **open → grasp/extract → place/release → close → clear the workspace**. Freeze supported-placement, release, closure and dwell semantics before launching.

The planned first cohort is seeds0–2 × bare VLA / framework: six full episodes with600query/action caps and1,200s rollout limits, matching initialization and stopping rules within each pair. Record the actual scene diversity produced by those seeds. This is an integration/development cohort; the closed-start actor's full-task competence is unresolved.

Open-start runs, if separately declared, test the remaining sequence only. Historical geometric door/grasp primitives are not repair fallbacks. The next step after failure is analysis of the observed full trajectory and an explicit revision, not manually completing the missing stage.

### P2. Attribute gains to verification and repair

The first comparison is bare VLA versus the complete framework. Follow-up ablations use switches within the **same actual runner and controller** and continue to execute full tasks. They do not introduce unrelated replacement methods or an isolated candidate-study prerequisite.

| Comparison | Mechanism tested | Matching requirements |
|---|---|---|
| Bare VLA vs complete framework | Overall intervention benefit | Actor, initial conditions, input preprocessing, native cadence, resource limits and endpoint |
| Full monitor vs selected event/phase evidence removed | Verification context and trigger timing | Repair generator, acceptance logic and budget |
| Full recovery contract vs simplified current-state exit | Value of temporal reverification | Entry trigger, model generation rule, input access and lifetime budget |
| Preservation enabled vs disabled | Protection of attained task progress | Same repair target, trigger and action-generation contract |
| Evidence-based retry vs cooldown-only retry | Avoiding repeated ineffective intervention | Same event definitions, local cap and total repair budget |
| Measured verifier vs measured-plus-predictive verifier | Incremental prospective evidence | Same actor, measurements, candidate-generation rule and resource envelope |

These are planned attribution comparisons, not already implemented or scheduled concurrent batches. Select the smallest set needed to explain the full-run observations, preregister their scope, and report interactions when one-component interpretations are inadequate.

Changing the controller changes the states and candidate actions subsequently encountered. Match the **generation protocol**, not an impossible identical future trajectory. If additional model proposals are introduced, account for all queries and distinguish equal maximum budgets from realized action exposure. A later replay analysis may explain a decision but does not prove the alternative full trajectory would succeed.

### P3. Adapt physical prediction inside the verifier

**Question:** can forecasting an execution consequence improve the verifier–repair loop in a way that current/past measurements cannot?

1. **Specify the native contract.** Declare object/receptacle-relative state, motion history, contact/grasp measurements and the seven-coordinate action convention. Identify exactly which features are privileged and which could be estimated from sensors. Do not pass hidden simulator parameters as undeclared inputs.
2. **Choose task-relevant targets.** Candidate targets are next-step relative pose, contact/grasp transition, containment loss and uncertainty. Force-related targets require a defined measurement and constraint; assembly force/tilt outputs do not already predict native task progress.
3. **Obtain declared training/calibration data.** Use permitted training states or datasets with frozen provenance, then separate calibration from confirmation. Previously inspected development traces remain development data even if reused for training.
4. **Reuse and adapt the existing implementation.** Start from the available dynamics/verifier interfaces. Record retraining, normalization, horizon and uncertainty estimation; incompatible weights cannot be relabeled as native prediction.
5. **Wire the forecast to a decision.** Save inputs, candidate/model action, predicted outcome and uncertainty, the verifier assessment, and the gate or acceptance decision influenced by it. A forecast computed only for logging does not demonstrate online contribution.
6. **Evaluate the full task.** Compare prediction enabled/disabled with matched resources. Report accurate but decision-irrelevant predictions, abstentions, false confidence and harmful interventions as well as benefits.

The current actor supplies H1 actions. Longer prospective horizons require an explicit future-action assumption or model-generated proposals with their query cost counted. Prediction horizon, action execution horizon and full task length are distinct; they must not be silently equated.

The online framework never selects an action using its future simulator branch outcome. Forecast reliability checks support the integrated experiment and do not reopen the canceled H96/candidate-study queue. If prediction fails to add value, report that result while retaining verification and repair as the main research subject.

### P4. Held-out robustness and task transfer

Evaluate a frozen configuration over separately declared axes rather than an uncontrolled Cartesian expansion:

| Axis | Planned evaluation | Interpretation boundary |
|---|---|---|
| Initial states | Reserved native roots and independently registered fridge initializations | Reused development roots do not count as held-out evidence |
| Geometric variation | Object poses, receptacle/layout configurations and clearance | No per-test motion routes or thresholds |
| Physical variation | Feasible friction/mass/contact changes with documented ranges | Report feasibility and shift magnitude; do not infer hidden parameters online |
| Task transfer | Microwave placement/closure to refrigerator retrieval | Declare all task adapters and distinguish logic reuse from zero-shot transfer |
| Observation access | Privileged measurements vs a specified estimated subset, if implemented | Attribute gains to available information as well as algorithms |
| Policy backbone | Optional second native-capable policy after the method is stable | Pair each policy with its own framework arm; cross-backbone scores do not isolate weights alone |

Native split: roots0–5 development,6–9 reserved calibration,10–19 reserved training,20–49 final confirmation. This numbering does not automatically define the fridge split; that must be registered independently. Freeze model, thresholds, task instructions and selection rules before confirmation. Confirmation outcomes may be reported but not used to select the version advertised as held out.

### P5. Resource envelope and external validation

Measure VLA inference, preprocessing/communication, verifier inference, simulation, repair queries, end-to-end latency and peak memory separately. Current GPU2/3parallelism accelerates paired experiments; it is not evidence of a two-GPU deployment requirement. Deployment claims use per-arm measurements under stated hardware.

Asynchronous scheduling, additional candidate generation or model compression are justified only by a measured bottleneck. A dual-rate or real-time controller is not currently implemented, and no50msdeadline is assumed achieved. Any scheduler change requires control-age and action-ownership checks in the same framework.

If accessible later, use sensor-based replay or real manipulation data to test whether verification signals remain meaningful outside privileged simulation. Such offline tests cannot establish closed-loop real-robot completion. If suitable data/hardware is unavailable, report the limitation and complete the declared simulation generalization study.

## 5. Evaluation, Statistics and Thesis Figures

### 5.1 Metrics and denominators

| Category | Measures | Reporting rule |
|---|---|---|
| **Full-task outcome** | Native success, termination reason, paired rescue/harm and net success difference | Main endpoint; preserve original task criteria and all planned cells |
| Verification | Incorrect interventions during valid transitions, missed declared events, intervention timing, false acceptance | Define event labels independently of the controller where possible; report unknown/ambiguous cases |
| Repair | Transaction count, accepted recovery, preservation loss, repeated failed repair and budget exits | Local outcomes do not imply task completion |
| Resumption | Executed handoffs, subsequent regression, completion after verified handoff, completion during repair | Give transaction-level and episode-level denominators separately |
| Prediction | State/transition error, calibration, uncertainty, decision changes and resulting harms/rescues | Use prediction metrics only for an actually loaded predictor with valid labels |
| Physical constraints | Measured contact/force/other violations and constraint-respecting completion | State definitions and measurement limits; no safety guarantee from surrogate acceptance |
| Resources | Queries, actions, repair share, time-to-success, rollout time, latency and memory | Include failures and exhausted budgets; separate model load from execution |

Event labels need care: a frame-level predicate change is not automatically a failure, and an eventual unsuccessful episode does not establish that an earlier intervention was necessary. Annotate valid releases, genuine loss and ambiguous transitions using declared evidence. Estimated error rates inherit that labeling uncertainty.

Completion conditional on verified handoff is useful descriptively, but those handoffs are selected by the controller. A higher conditional rate alone is not causal evidence; the full paired experiment and appropriate component comparisons carry that claim.

### 5.2 Paired design and statistical analysis

- Match initial simulator state, camera/preprocessing, model/checkpoint, nominal instruction/cadence, initialization, maximum queries/actions/time and stopping rules within each pair.
- The actor sees the declared native inputs. Privileged diagnostics are logged in both arms but influence decisions only in the framework arm. The first comparison tests that complete treatment; later information-matched ablations separate information access from decision logic.
- Freeze and log every actual proposal, executed action, verification event and input association. Never compare only the successful cases or substitute a repeated run for a failed one without reporting it.
- For six reused development roots, report exact case counts and paired outcomes; avoid claims of statistical generalization. Plan confirmation sample size from a preregistered meaningful paired effect and expected discordant-pair rate, rather than stopping when a favorable result appears.
- For confirmation, report paired confidence intervals and, where appropriate, exact paired tests. Resample episodes/initial-state pairs rather than treating correlated steps or multiple repairs as independent observations. Report task-specific effects before any declared pooled estimate.
- Technical/replay failures remain separate from valid task failures. Report planned, executed, valid and invalid counts; any conservative all-planned bound must be labeled as such. No silent removal or zero-imputation into a behavioral success-rate estimate.
- Component comparisons are exploratory until their selection procedure and confirmation split are frozen. State multiple-comparison and repeated-development limits explicitly.

### 5.3 Figures and deliverables

| Figure / artifact | Content | Question |
|---|---|---|
| **F1 — Method** | Five layers, verifier evidence, repair transaction and actual handoff | Architecture |
| **F2 — Verification cases** | Valid release vs genuine loss; evidence, decisions and their downstream effects | RQ1 |
| **F3 — Headline outcome** | Paired full success, rescues and harms, with all planned/valid counts | RQ2 |
| **F4 — Recovery funnel** | Trigger → repair → verified exit → executed handoff → full completion | RQ2 |
| **F5 — Mechanism ablations** | Preservation, reverification and retry comparisons through the same framework | RQ1/RQ2 |
| **F6 — Physical prediction** | Forecast fidelity/calibration, decision influence and predictor-on/off full-task outcomes | RQ1/RQ2 |
| **F7 — Generalization** | Frozen-method outcomes across reserved states, tasks and declared shifts | RQ3 |
| **F8 — Resource envelope** | Queries, latency, memory and completion/cost trade-offs | RQ3 |
| **F9 — All-case video/evidence index** | Full synchronized baseline-left/framework-right videos, traces and recoverable archives | Reproducibility |

## 6. Existing Implementation, Gaps and Reproducibility

| Component | Current implementation / evidence | Remaining work |
|---|---|---|
| Canonical experiment lifecycle | `scripts/run_framework_comparison.py`: prepare, launch, replay, report and videos | Extend task adapters without duplicating version-specific runners |
| Five-layer measured controller | `src/adapters/native_framework.py`; v2source frozen at`ab48b63` | Evaluate complete outcomes and improve only from recorded evidence |
| Frozen model action generation | Official greedy OpenVLA interface with saved inputs, tokens and executed actions | Fridge task-input integration and full-task capability remain unresolved |
| Recovery contract checks | `scripts/test_native_framework.py`;16model-free checks at launch | Add checks when genuine transition defects are found |
| Predictive components | Existing dynamics/reference-verifier interfaces; assembly-specific predictor | Native features, training/calibration, forecast calls and integrated decision use |
| Task transfer | Completed native microwave benchmark integration; historical fridge runtime/evidence | Reuse current repair transactions with fridge task adapters and model actions |
| Generalization | Development roots identified; confirmation reserved | Freeze before evaluating held-out conditions |
| Evidence publication | First native batch has complete videos, raw archives and independent remote recovery | Repeat publication/recovery for each completed new batch |

**Information boundary:** current verification uses privileged simulator measurements; it is not sensor-only deployment. **Execution boundary:** all numerical task/repair commands are model outputs; observation geometry is not an action generator. **Prediction boundary:** measured feedback does not imply a learned forecast was used.

Long jobs run detached from the terminal on physicalGPU2/3, neverGPU0; current actor batch is1and any microbatch is at most8. Source/configuration hashes and checkpoint provenance are frozen before execution. A running batch's pinned sources and protocol stay immutable.

Each batch delivers a completion table, all-case paired videos, stepwise evidence/control ownership/budgets/exits, saved inputs/actions, exact-replay reports and Git/LFS archives with independently checked recovery. Full videos use common simulation time and explicit end-frame holds. Documentation is English: this plan defines the research; the current record holds execution details; weekly files summarize findings. Other experiment records remain under `exp_record/agent_notes/`.

## 7. Target Thesis Structure

1. **Introduction:** reliability failures of frozen VLAs, motivation for runtime verification and repair, and the unstructured manipulation setting.
2. **Background and related work:** VLA execution, task/physical verification, world models and neural physics, model-generated recovery and policy resumption.
3. **Method:** five-layer framework, evidence representation, dual-mode gate, recovery contracts, preservation, reverification and handoff.
4. **Verification quality:** event/phase evidence, ambiguity and error analysis; physical prediction where actually implemented (RQ1).
5. **Closed-loop experiments:** complete tasks, rescues/harms, component attribution and persistent recovery (RQ2).
6. **Robustness and feasibility:** reserved states, task/condition transfer, observation assumptions and resource envelope (RQ3).
7. **Discussion:** causal and external-validity limits, negative results, predictor contribution, and the boundary between recoverable errors and missing policy capability.
8. **Conclusion:** supported claims, remaining gaps and future work.

The thesis does not depend on every hypothesis being positive. A reproducible account of why verification fails to translate into repair or resumption is a legitimate result, provided it is tested rather than inferred from a few attractive clips.

## 8. Risks and Decision Gates

| Risk | Evidence to examine | Response within the research scope |
|---|---|---|
| VLA lacks a required task behavior | Full trajectories show no useful task/repair behavior despite valid inputs | Report the capability boundary; investigate task-input/model adaptation with separate provenance, not a scripted repair route |
| Verifier confuses release or settling with failure | Interventions disrupt successful nominal transitions | Improve event/phase evidence; retain true-loss tests and all-case paired evaluation |
| Local recovery does not survive handoff | Accepted milestones regress after nominal resumption | Examine contract sufficiency and resumption semantics; measure full continuation |
| Repair repeatedly consumes budget without changing context | Repeated identical events and local-cap exits | Require changed evidence; quantify lost time and completion cost |
| Predictor is inaccurate or adds no useful information | Forecast errors, uncertainty mismatch or no positive paired increment | Improve the native contract/calibration if justified; report negative predictive value explicitly |
| Privileged state drives the apparent advantage | Gains disappear with matched evidence or estimated inputs | Narrow the claim and quantify the observation-access effect |
| Overfitting to six roots or one scene | Gains disappear after freezing on reserved conditions | Retain adverse cases; evaluate held-out transfer and limit generalization claims |
| Replay/infrastructure failures obscure behavior | Mismatched state/images, incomplete queries or inconsistent ownership | Preserve invalid artifacts and resolve the technical cause before making efficacy claims |
| Scope expands into new platforms, policies or hand-coded controllers | New infrastructure replaces work on the observed loop defects | Follow the single framework/task queue and require an explicit research justification |
| Runtime cost eliminates any benefit | Extra queries, latency or reduced executed exposure explain outcome changes | Report common-budget trade-offs and optimize the measured bottleneck |

**Decision gates:**

- **G1 — Trustworthy execution:** full-loop execution, model-action provenance and independent replay are available. Passing establishes infrastructure, not efficacy.
- **G2 — Development evidence:** completed paired runs identify the net benefit/harm and concrete failure mechanisms. A null result motivates a bounded revision, not a silent increase in budget or relaxed success criterion.
- **G3 — Mechanistic evidence:** within-framework comparisons support the claimed role of verification, repair acceptance or resumption. A bundled change alone is insufficient for component attribution.
- **G4 — Prediction evidence:** native-compatible forecasts actually participate in decisions and their incremental full-task effect is measured. Merely loading a model does not pass.
- **G5 — Generalization and feasibility:** a frozen method is evaluated on declared held-out conditions with uncertainty and costs; failure narrows the claim rather than triggering test-set retuning.

These gates govern claims and prioritization. They do not reinstate an isolated-diagnostic queue that must finish before the next complete-framework experiment can run.

## 9. Immediate Next Steps

1. **Complete the launched v2microwave batch.** Retain all12cells, fresh replays, the six full paired videos and both budget/technical failures. The exact frozen source and live state are in the current record.
2. **Evaluate the verifier–repair chain, not just the final score.** Check which triggers changed, whether recovery was truly verified, whether nominal handoffs executed, and whether subsequent task progress persisted. Report the complete success table first, followed by its mechanism evidence.
3. **Use those trajectories to choose the next bounded change or attribution comparison.** Avoid changing multiple unrelated components without declaring the comparison bundled.
4. **Integrate closed-start fridge manipulation through the same framework.** Freeze task predicates, initialization, actor inputs and model-generated repair interface before the planned six-episode development cohort.
5. **Specify and adapt the native physical predictor.** Establish state/action and target contracts using permitted data, then test its incremental role in complete framework runs. This remains a planned research work package, not a claim about the currently running measured-only controller.
6. **Publish and verify evidence.** Update the existing current record and weekly summary, archive the full batch and check fresh remote recovery. Preserve reserved confirmation states and user-owned planning edits.

The comprehensive plan supplies the research structure; it is not a replacement launch protocol. The current experiment continues under its already frozen configuration.
