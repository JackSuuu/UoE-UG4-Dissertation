# R2: actual frozen VLA versus five-layer runtime

## Completed paired videos — baseline left / wrapper right

* **[Closed start: neither arm opens the door](closed_baseline_left_framework_right.mp4)**
* **[Open start: wrapper grasps/extracts, but does not complete](open_baseline_left_framework_right.mp4)**
* [Detailed results and generalization limits](../../exp_record/agent_notes/fridge_r2_results_20261019.md)
* [All-case metrics and replay reports](results.json), [final supervisor state](launch_final.json)
* [Complete archive](complete.tar.gz), [200-member SHA256 manifest](complete_manifest.json)

Each video is601frames at20fps with common simulation time, matched initial state,
600actions/queries per side, phase/owner/event labels and terminal-frame labels.
Both sides query the same frozen actor with shared phase prompts; only the right
executes corrections. **No full-task completion; no demonstrated cross-scene repair.**
The wrapper uses privileged simulator feedback and scene-specific primitives.

All4rollouts finished. Three have exact independent replay; the open wrapper has
exact physics/controller/decisions/agentview but two wrist frames differ by three
RGB channels total, each by1. The supervisor retains `complete_with_issues`.

**Event-label correction:** original video/log events use one string. `retain_grip`
overwrites two verified local exits at steps245/302. A frozen replay audit records
those exits separately; neither immediately executes an unmodified VLA action and
neither leads to completed retrieval. See `results.json` and the detailed report.

`complete.tar.gz` retains all raw actor/wrapper outputs, states, images, frozen
sources, replay reports, handoff audit, both paired videos, evaluator/smoke artifacts,
and all door diagnosis variants (including interrupted v4). Door branches are
post-hoc scene-specific development, not framework efficacy or transfer trials.
The archived current door-source snapshot includes an unvalidated v4 change.

* [Protocol](../../exp_record/agent_notes/fridge_r2_framework_protocol_20261019.md)
* [Execution record](../../exp_record/agent_notes/fridge_r2_execution_20261019.md)
* [Detached launch snapshot](launch_snapshot.json), source freeze`c4afde8`
* [100-step interface smoke](interface_smoke.json)
* [Positive/negative physical evaluator fixtures](endpoint_qualification.json)

Endpoint fixtures intentionally install object state for evaluator qualification;
they are not actor episodes and do not contribute to task-success results.
