# Experiment record

Running log of experiment progress against
[`core_research_plan.md`](../core_research_plan.md),
one file per week. Inside a file, the newest entry is at the top.

**Current research decision:** [Local VLA precision execution with physical feedback under uncertainty](local_vla_scope_20261013.md). LLM/edge-cloud are future work. Establish one task/actor baseline and physical-intervention feasibility under [the task-first sequence](simulation_task_first_20261013.md) before further verifier training. This supersedes immediate gate-fine-tuning priorities in [the earlier mainline reset](research_mainline_20261013.md); completed experimental results remain unchanged.

**Implementation:** [LIBERO full7B integration and pilot launch](libero_integration_20261013.md), with [frozen native-action feasibility protocol](libero_baseline_protocol_20261013.md).

**Latest pilot outcome:** [Native3/3; released stable placement remains unestablished](libero_baseline_results_20261013.md). Includes the site-predicate contact erratum and the next continuation-baseline requirement.

**Completed continuation:** [Released placement1/3 and failure-stage audit](libero_continuation_results_20261013.md), [launch and checks](libero_continuation_launch_20261013.md), [frozen protocol](libero_continuation_protocol_20261013.md). No active rollout/training job remains.

**14 Oct review:** [Accelerated videos, causal diagnostics and physics-specific comparison/architecture design](libero_framework_design_20261014.md). Next is a bounded privileged-vs-simple-control mechanism protocol, before learned training.

**Authoritative architecture:** [User-locked five layers](libero_framework_design_20261014.md#authoritative-architecture-lock--user-decision-14-oct-2026), [concrete implementation and bowl/drawer long-task plan](libero_five_layer_implementation_20261014.md). High-level subgoals → frozen OpenVLA → risk/event monitoring → dual-mode gate → bounded local repair.

**Completed preservation study:** [Verified results](libero_preservation_results_20261014.md), [protocol](libero_preservation_protocol_20261014.md), [initial launch](libero_preservation_launch_20261014.md), [renderer failure/fix/restart](libero_preservation_renderer_fix_20261014.md). All24`_b`rollouts and independent checks complete, six method-comparison videos generated. Acquisition VLA/hold/retreat/physics=2/6,3/6,2/6,3/6; completed-and-retained=1/6,2/6,1/6,1/6. No physics advantage over hold. Still a privileged H1/k1 subsystem diagnostic, not the full five-layer implementation or learned efficacy.

**Recovery and attribution:** [GitHub backup, baseline clarification and staged generalization](recovery_and_generalization_20261014.md); [download videos / restore artifacts](../backups/README.md).

**Latest completed study:** [R5 results](immediate_gate_results_20261013.md): partial event-vs-regression signal, all new gates fail policy-relative safe completion; no promotion or new experiment. The [frozen numerical protocol](immediate_gate_20261013.md) remains unchanged.

| week | dates | file | main events |
|---|---|---|---|
| 1 | to 29 Sep 2026 | [week1.md](week1.md) | Genesis bring-up and demotion; Task A → box peg, two-fingertip pusher; open-loop chunk protocol; three pipeline defects; v5; repair audit; RQ2b shift ladder |
| 2 | 30 Sep – 4 Oct 2026 | [week2.md](week2.md) | CheckVLA's gradient branch and hard prefix are inert; v6/v7; rate-matched τ; OpenVLA-7B on transformers 5.x; demo + feature pipeline; chunk heads; DAgger; first OpenVLA headline (confounded by 5-step chunk, fixed at 10: oracle +0.131) |
| 3 | 5–11 Oct 2026 | [week3.md](week3.md) | cadence confound corrected; decision-only calibration; small matched-protocol validation; expanded confirmation and candidate repair launched |
| 4 | 12–18 Oct 2026 | [week4.md](week4.md) | temporal-risk negative; gate × repair and boundary complete; [transition results + SEAL feasibility](repair_transition_results_20261013.md): incremental +2.01pp vs original, overall learned efficacy unestablished; 52 tests |

## Conventions

- **English**, one `### Update <date> — <one-line finding>` per entry. The title is
  the result, not the activity.
- Record **what was measured and the number**, then what it means. Say which
  earlier reading it overturns, and edit that entry with a pointer forward
  rather than deleting it.
- A new week starts a new file `weekN.md`; add a row to the table above.
- Commands for long runs are recorded with `setsid nohup ... < /dev/null`.
  Anything started with a plain `&` dies with the launching shell.
