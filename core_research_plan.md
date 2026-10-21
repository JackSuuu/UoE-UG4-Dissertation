# Research Plan: Completing Unstructured Manipulation Tasks

**Updated: 21 October 2026**

## 1. Objective

**Improve end-to-end completion of unstructured, multi-stage manipulation tasks using our five-layer verification and repair framework around a frozen VLA.**

The immediate task families are **microwave manipulation and refrigerator retrieval**. Success means completing the requested task, including the remaining sequence after an intervention. A grasp, a local recovery, or a successful handoff alone is not the primary outcome.

We retain OpenVLA while improving the existing framework. Native LIBERO microwave episodes provide a reproducible development setting; refrigerator manipulation extends the task sequence. Generalization to unstructured conditions must subsequently be tested through declared variations in initial state, object placement, layout and contact conditions. Success in a fixed simulator scene does not establish that generalization.

## 2. Method: One Five-Layer Framework

| Layer | Responsibility |
|---|---|
| **1. High-level subgoals** | Track task stages, prerequisites and achieved milestones. Separate local recovery from the next task objective. |
| **2. Frozen VLA** | Generate nominal and repair actions from current observations and the appropriate instruction. |
| **3. Risk and event-boundary monitoring** | Observe progress, contact transitions, grasp loss, release, stagnation and uncertainty. Incorporate applicable physical forecasts. |
| **4. Dual-mode gate** | Select nominal tracking or bounded intervention using the available verification evidence. |
| **5. Local repair and reverification** | Execute model-generated corrections, check a declared recovery contract, and return control to a fresh nominal VLA action. |

**Verification is central; physical prediction is a supporting module.** Measured state, contact and geometric constraints, task progress, and learned forecasts can contribute within the same framework. Prediction is not the only admissible evidence source. Its incremental value must be measured when a task-compatible predictor is available.

Every treatment experiment runs the actual five-layer framework from task initialization to success or termination. Bare VLA is the paired control. All task and repair numerical commands come from the model; geometry supplies verification evidence rather than scripted motion routes. Initialization noops are separately counted.

The current implementation uses synchronous, greedy OpenVLA queries and privileged simulator measurements for verification. High-level subgoals are declared task semantics. A learned microwave/fridge predictor and sensor-only state estimation are not yet implemented. These limitations are reported explicitly.

## 3. Research Questions and Measures

1. **Task completion:** Does the framework improve full-task success under matched initial conditions and resource limits?
2. **Recovery quality:** Does it rescue failures without disrupting successful nominal behavior, and does verified recovery lead to actual handoff and downstream completion?
3. **Generalization:** Do improvements persist on reserved initial states and declared unstructured variations without scene-specific motion scripts?
4. **Evidence contribution:** Which monitoring and verification components, including physical prediction, contribute to these outcomes?

The primary measure is **complete-task success rate**. Report paired rescues and harms alongside it. Secondary measures include verified recoveries, executed handoffs, completion after verified handoff, completion during repair, milestone regressions, repeated failed interventions, queries, actions and elapsed time. Technical failures and replay mismatches remain visible and separate from valid task failures.

## 4. Current Evidence

| Experiment | Observation | Interpretation |
|---|---|---|
| First native microwave comparison, six paired initial states | Bare VLA **4/6**, framework **4/6**; one rescue and one harm; 12/12 exact replays | Real outcome changes, but no net completion gain yet |
| Recovery in that comparison | All 11 transactions exited unverified; the rescued episode completed after an actual handoff | No demonstrated verified-recovery → handoff → completion chain |
| Historical refrigerator R2 | Neither closed-start arm opened the door; open-start intervention grasped/extracted but did not finish | Handwritten geometric corrections; not evidence for the current model-action repair method |

The observed microwave defects are release/loss confusion, a moving recovery target, missing preservation of new progress, and repeated intervention on an unchanged failure flag. These motivate the next revision. [Published paired evidence](backups/framework_comparison_20261019/README.md).

## 5. Experiment Sequence

### A. Microwave: revise the framework and run complete paired episodes

**Question:** Can phase-aware monitoring and stable recovery contracts reduce harm, retain rescue, and improve complete-task success?

| Item | Fixed design |
|---|---|
| Task | Put the yellow-and-white mug in the microwave and close it; original native `In AND Close` endpoint |
| Cohort | Development initial states 0–5 × bare VLA / revised framework: **12 fresh episodes** |
| Actor | Same frozen OpenVLA LIBERO-10 checkpoint, Transformers 4.40.1, greedy H1, batch 1, seed 0 |
| Inputs | Same native camera and initial full-task instruction; scoped repair language is the framework intervention |
| Budget | At most 520 queries and 520 task actions, plus 10 initialization noops; 1,200 s per rollout excluding model load |
| Repair limits | 32 actions per transaction; 160 repair actions within the shared lifetime budget |
| Termination | First native success, resource limit, or recorded technical failure |
| Verification | Current/past measured evidence; learned prediction disabled for this revision |
| Validation | Fresh exact replay of every episode and full paired videos for all six cases |

The revision distinguishes release near previously attained containment from genuine loss; freezes recovery acceptance independently of changing action prompts; preserves new durable milestones; and requires changed evidence before repeating a failed repair. Root-specific branches and manually generated corrections are excluded.

This is one bundled controller iteration, not a causal isolation of each change. Its fresh paired outcomes are compared with the historical revision as development evidence. Budgets, success criteria and actor weights remain fixed.

### B. Refrigerator: extend the same framework to a complete task

Reuse the gate, recovery transaction and handoff implementation through task-specific observation, goal and instruction adapters. Start closed-door and empty-handed: **open → grasp/extract → place/release → close → clear the workspace**. Freeze supported placement, release, closure and dwell checks before execution.

The planned first cohort is seeds 0–2 × bare VLA / framework, six full episodes in the existing fixed scene/camera configuration. Each arm has 600 model queries/actions and a 1,200 s rollout cap, with identical initialization and stopping rules. Record how much the seeds actually vary the scene. Open-start experiments, if separately declared, are remaining-sequence evaluations and cannot count as closed-start completion.

The task adapter must use model-generated repair, not the historical door/grasp servos. Full-task actor competence in this custom scene remains unresolved; complete failed attempts are useful evidence for the next framework iteration. This cohort follows the microwave revision rather than launching an independent replacement controller.

### C. Confirmation and component attribution

After development, freeze the method before evaluating reserved initial states and task variations. For the native split, roots 0–5 are reused development cases, 6–9 are reserved for calibration, 10–19 for training, and 20–49 for final confirmation. Register every use; never present development results as held-out generalization.

Evaluate physical prediction through matched predictor-on/off comparisons **within the same framework**. Reuse existing predictor/verifier interfaces, but adapt and validate native state/action features and task-relevant targets before loading a predictor. Assembly-specific weights do not establish microwave or fridge prediction capability. Log forecast calls, uncertainty, errors and actual decision influence.

## 6. Execution and Deliverables

- **One runner:** `scripts/run_framework_comparison.py`.
- **One controller:** `src/adapters/native_framework.py`.
- **One current record:** [protocol, freeze, launch and results](exp_record/agent_notes/current_goal_and_experiment_queue_20261019.md).
- **Checks:** meaningful model-free transition tests before freezing, then full paired episodes and independent replay. Diagnose complete trajectories rather than adding isolated candidate/horizon studies as prerequisites.
- **Compute:** detached jobs on physical GPUs 2/3; exclude GPU 0; actor batch 1 and any microbatch at most 8. Preserve PID, source hashes and actual query counts.
- **Each batch:** all-case completion table, rescue/harm accounting, complete synchronized videos, stepwise verification/ownership/budget/exit traces, raw evidence and Git/LFS publication with recovery verification.
- **Records:** English weekly summaries in `exp_record/week*.md`; all detailed records in `exp_record/agent_notes/`.

This plan defines the research objective and sequence. Live execution status belongs in the current record; a launch is not a result. Historical plans remain in Git, including [the earlier cumulative plan](https://github.com/JackSuuu/UoE-UG4-Dissertation/blob/465e0a2/core_research_plan.md).

**Registered dissertation topic:** *Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization*.
