# Verification and Repair for Frozen VLAs

University of Edinburgh UG4 dissertation: **runtime verification coupled to bounded, model-generated repair**, evaluated through complete-task reliability in unstructured manipulation.

## Objective

**Core research question:** how should a verifier guide intervention, check recovery and support reliable resumption of a frozen VLA? When does physical prediction improve this loop beyond measured feedback?

The **verifier–repair loop is the method**, **full-task completion is the primary outcome**, and **unstructured microwave/refrigerator manipulation is the application and generalization setting**. The five-layer architecture below implements this research subject.

The current microwave benchmark is a controlled development setting. The new runtime generates task semantics and recovery contracts from **natural-language instructions, images and a scene entity inventory** using a vision-capable language model. Its controller is task-independent; real-model task transfer remains to be evaluated.

**[Research plan](core_research_plan.md)** · **[Current protocol and run status](exp_record/agent_notes/current_goal_and_experiment_queue_20261019.md)** · **[Week 5](exp_record/week5.md)**

## Our Five-Layer Framework

1. **High-level subgoals:** track stages, prerequisites and achieved milestones.
2. **Frozen VLA:** generate nominal and repair actions from current observations.
3. **Risk/event monitoring:** observe progress, contact transitions, release, loss and uncertainty.
4. **Dual-mode gate:** decide between nominal execution and bounded intervention.
5. **Local repair and reverification:** check recovery, preserve progress, and execute a fresh nominal handoff.

```mermaid
flowchart TD
    Task[Task instruction and current observation] --> Goals[1. High-level subgoals]
    Goals --> VLA[2. Frozen VLA]
    Goals --> Monitor[3. Risk and event monitor]
    Obs[Measured state and optional physical forecasts] --> Monitor
    Monitor --> Gate{4. Dual-mode gate}
    Gate -->|Nominal instruction| VLA
    Gate -->|Recovery needed| Repair[5. Bounded model-generated repair and rechecks]
    Repair -->|Scoped repair instruction| VLA
    VLA --> Exec[Execute unchanged model action]
    Exec --> Monitor
    Exec --> Goals
    Monitor --> Repair
    Repair -->|Verified recovery or labeled budget exit| Gate
    Exec --> End[Complete-task outcome]
```

**Verification is central.** The planner proposes semantic goals; measured predicates determine whether they hold. Unknown evidence never verifies a contract. Every task/repair numerical action comes from frozen OpenVLA. A learned physical predictor is not enabled. Historical microwave results below used manually declared semantics, not the new generated-task implementation.

## Task-agnostic runtime

- Entry point: [`scripts/run_task.py`](scripts/run_task.py).
- Architecture, GPU-machine command and scorer interfaces: [runtime guide](docs/task_runtime.md).
- CPU integration checks: `python -B scripts/test_task_runtime.py` (standard library only).
- Main components: `src/planning/`, `src/controller.py`, `src/runtime.py`, `src/verification/physical_verifier.py` and environment/model bridges in `src/adapters/`.
- No per-task Python adapter or JSON goal configuration. The LIBERO bridge supplies scene IDs and measurements for any supported task; the MLLM generates the task and recovery records.
- The new path has CPU integration coverage. Real MLLM/OpenVLA rollouts, grounding accuracy, and general-purpose learned A/B artifacts are pending GPU validation.

## Experiments and Evidence

**Latest completed paired experiment (v2):** bare VLA **4/6**, framework **4/6**, with zero rescues and zero harms. All 12 episodes passed exact replay. The previous root3 harm disappeared, but the root4 rescue was lost. One repair passed local verification and actually handed back; containment was lost on the first nominal action and the task failed. No net completion improvement or verified-recovery-to-completion gain has yet been established.

- [Latest v2 comparison, six full paired videos and raw evidence](backups/framework_comparison_20261021/README.md)
- [First comparison: one rescue and one harm](backups/framework_comparison_20261019/README.md)
- [Historical refrigerator R2](exp_record/agent_notes/fridge_r2_results_20261019.md): geometric correction actions, local extraction progress, no full completion; separate from the current model-generated repair method.

**Historical post-v2 study:** a complete-task placement-prompt / measured-resumption-gate comparison was launched from frozen source `528a91c`. Its launch and results belong to that archived protocol; see the [experiment record](exp_record/agent_notes/current_goal_and_experiment_queue_20261019.md) and [session recovery status](exp_record/agent_notes/session_recovery_status_20261023.md). These outcomes are not results of the generated-task runtime.

Every treatment run uses the real five-layer framework. Paired arms share initial conditions, actor inputs and resource limits; all repair queries count against the lifetime budget. Full videos show baseline left/framework right, synchronized simulation time, control ownership, evidence, verification, exits and actual handoffs. Failures and harms remain included.

## Code and Records

| Path | Purpose |
|---|---|
| [`core_research_plan.md`](core_research_plan.md) | Current objective, method, experiment sequence and evaluation |
| [`scripts/run_task.py`](scripts/run_task.py) | Generated-task runtime and LIBERO execution |
| [`scripts/run_framework_comparison.py`](scripts/run_framework_comparison.py) | Historical microwave preparation, execution and replay |
| [`src/controller.py`](src/controller.py) | Task-independent budgets, repair verification and handoff |
| [`src/legacy/microwave_controller.py`](src/legacy/microwave_controller.py) | Historical measured microwave controller |
| [`src/README.md`](src/README.md) | Component layout, implementation status and Phase 1 history |
| [`scripts/test_native_framework.py`](scripts/test_native_framework.py) | Model-free recovery and execution-contract checks |
| [`exp_record/`](exp_record/agent_notes/experiment_record_index.md) | Weekly summaries; detailed protocols and results under `agent_notes/` |
| [`backups/`](backups/README.md) | Published videos, raw evidence and recovery instructions |
| [`third_party/LIBERO`](third_party/LIBERO) | Pinned benchmark submodule |
| `notes/` | Proposals, literature and historical planning |

Run current controller checks with `python -B scripts/test_task_runtime.py`; `scripts/test_native_framework.py` checks the historical protocol. Old source-pinned experiments require their archived sources. Inspect runner options with `--help`.

Large artifacts use Git LFS. Clone with submodules and follow each backup's scoped LFS recovery instructions.

**Registered dissertation topic:** *Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization*.
