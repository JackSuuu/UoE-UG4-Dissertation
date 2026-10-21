# Verification and Repair for Frozen VLAs

University of Edinburgh UG4 dissertation: **runtime verification coupled to bounded, model-generated repair**, evaluated through complete-task reliability in unstructured manipulation.

## Objective

**Core research question:** how should a verifier guide intervention, check recovery and support reliable resumption of a frozen VLA? When does physical prediction improve this loop beyond measured feedback?

The **verifier–repair loop is the method**, **full-task completion is the primary outcome**, and **unstructured microwave/refrigerator manipulation is the application and generalization setting**. The five-layer architecture below implements this research subject.

The current microwave benchmark is a controlled development setting. Generalization to unstructured object placements, layouts and contact conditions requires subsequent held-out evaluation.

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

**Verification is central.** Physical prediction can contribute alongside measured contact, geometric constraints and task progress within this framework. Current native runs use privileged measured verification; a learned microwave/fridge predictor is not enabled. High-level subgoals are declared task semantics, and every task/repair numerical action comes from OpenVLA.

## Experiments and Evidence

**Latest completed paired experiment:** bare VLA **4/6**, framework **4/6**, with one rescue and one harm. All 12 episodes passed exact replay. All 11 repair transactions exited unverified; one rescued episode completed after handoff. No net completion improvement or verified-recovery-to-completion gain has yet been established.

- [Complete comparison, six paired videos and recoverable raw evidence](backups/framework_comparison_20261019/README.md)
- [Historical refrigerator R2](exp_record/agent_notes/fridge_r2_results_20261019.md): geometric correction actions, local extraction progress, no full completion; separate from the current model-generated repair method.

**Current iteration (launched 21 October):** all six microwave initial states with fresh bare-VLA/framework pairs after correcting release/loss handling, recovery acceptance, milestone preservation and repeated failed intervention. Revision `ab48b63` passed 16 controller checks and is running on GPUs 2/3. The same framework then extends to closed-start refrigerator retrieval. Exact freeze, process status and results are maintained in the [current record](exp_record/agent_notes/current_goal_and_experiment_queue_20261019.md).

Every treatment run uses the real five-layer framework. Paired arms share initial conditions, actor inputs and resource limits; all repair queries count against the lifetime budget. Full videos show baseline left/framework right, synchronized simulation time, control ownership, evidence, verification, exits and actual handoffs. Failures and harms remain included.

## Code and Records

| Path | Purpose |
|---|---|
| [`core_research_plan.md`](core_research_plan.md) | Current objective, method, experiment sequence and evaluation |
| [`scripts/run_framework_comparison.py`](scripts/run_framework_comparison.py) | Canonical preparation, execution, replay, reporting and video runner |
| [`src/adapters/native_framework.py`](src/adapters/native_framework.py) | Current five-layer native controller |
| [`scripts/test_native_framework.py`](scripts/test_native_framework.py) | Model-free recovery and execution-contract checks |
| [`exp_record/`](exp_record/agent_notes/experiment_record_index.md) | Weekly summaries; detailed protocols and results under `agent_notes/` |
| [`backups/`](backups/README.md) | Published videos, raw evidence and recovery instructions |
| [`third_party/LIBERO`](third_party/LIBERO) | Pinned benchmark submodule |
| `notes/` | Proposals, literature and historical planning |

Run controller checks with `python -B scripts/test_native_framework.py`. Experiment execution uses the pinned server environments and source manifests documented in the current record; inspect runner options with `--help`.

Large artifacts use Git LFS. Clone with submodules and follow each backup's scoped LFS recovery instructions.

**Registered dissertation topic:** *Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization*.
