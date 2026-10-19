# Full native comparison: baseline vs measured-verification framework

Frozen execution: `09d4b2a`. Six reused development initial states, two complete
episodes per state, at most520model calls/actions each. Same frozen OpenVLA,
native observations and initial states. Framework monitor uses privileged current
and past simulator measurements, not learned prediction. All task/repair numeric
actions come from the VLA. Scoped task-semantic repair instructions are part of
the complete intervention. This is not an isolated verification ablation.

**12/12 exact replays. Baseline4/6; framework4/6. One rescue, one harm.**
All11repair transactions exited unverified at their32-action cap. Actual VLA
handoffs occurred; root4subsequently completed. No verified-repair→handoff success.
Full per-cell results and records: [report](report.md), [data](report.json).

## Six full videos

Left baseline; right framework. Aligned by simulator action index,20fps native
simulator time, not wall-clock speed. Early endings hold a labeled frame. Causal
overlays show subgoal, trigger/evidence, owner, query budget, checks, exit and handoff.

| Root | Baseline / framework | Full video |
|---|---|---|
|0|failure / failure|[Watch](videos/root0_baseline_left_framework_right.mp4)|
|1|success / success|[Watch](videos/root1_baseline_left_framework_right.mp4)|
|2|success / success|[Watch](videos/root2_baseline_left_framework_right.mp4)|
|3|success / failure: harm|[Watch](videos/root3_baseline_left_framework_right.mp4)|
|4|failure / success: rescue|[Watch](videos/root4_baseline_left_framework_right.mp4)|
|5|success / success|[Watch](videos/root5_baseline_left_framework_right.mp4)|

GitHub: use **Download raw file** if inline playback is unavailable.

Root3: baseline self-recovers from containment loss; framework instead repeatedly
requests acquisition and fails. Root4: early intervention achieves containment,
then changes to closing; local verification remains unmet, budget exit returns
to the VLA and the complete task later succeeds. Root0still fails despite160repair
actions. Root1/2/5complete scientific trajectories are identical across arms.

Seven archives preserve12,676members: preparation, controller tests, all12episodes,
model inputs/outputs/actions, full physics/image tapes, independent replay proofs,
supervisor logs and reports. Six videos are also directly browsable through LFS.
`manifest.json` pins every member. Local verification and fresh remote recovery
passed:7archives,12,676restored members and6videos from`d1a48a1`. Recomputed
all5088model-query/action/input associations and aggregate outcomes from the
restored artifacts. [Recovery proof](remote_recovery.json).
