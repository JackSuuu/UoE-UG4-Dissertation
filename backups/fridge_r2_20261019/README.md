# R2: actual frozen VLA versus five-layer runtime

* [Protocol](../../exp_record/agent_notes/fridge_r2_framework_protocol_20261019.md)
* [Execution record](../../exp_record/agent_notes/fridge_r2_execution_20261019.md)
* [Detached launch snapshot](launch_snapshot.json), source freeze`c4afde8`
* [100-step interface smoke](interface_smoke.json)
* [Positive/negative physical evaluator fixtures](endpoint_qualification.json)

Four600-action/query-capped runs are launched on GPU1/3: seed0closed/open ×
baseline/framework. Both arms actually query the frozen VLA; the right arm applies
conditional, budgeted feedback repairs and revalidated handoff. Both use shared
phase-prompt rules. Privileged simulator geometry/contact is explicitly disclosed.

The first logged steps confirm actual VLA execution on both sides, followed by repair
ownership on the framework side while baseline continues VLA control. This is launch
evidence, not a successful full task or a completed comparison. The detached supervisor
automatically replays each run and renders both left-baseline/right-framework pairs.
Completed trajectories/videos will be archived after results arrive.

Endpoint fixtures intentionally install object state for evaluator qualification;
they are not actor episodes and do not contribute to task-success results.
