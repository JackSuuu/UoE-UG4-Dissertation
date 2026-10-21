# Fridge candidate v1: no opening/task progress, integration only

Artifacts: `/home/ami/scratch/experiment3_candidates_fridge_20261019_a`.
Pre-run source/protocol/camera/model freeze SHA256:
`56b3f5a3aa2bcd823e8463cb3634685d02278dd4c9ca263ef5fb4c2395dfc7f8`.
Closed-start seed0, fixed reframed camera, shared six-stage observer/prompts,
same frozen official TF4.40.1 actor, GPU2/batch1; no scripted task actions.

| Arm | Queries | Actions | Sample calls | Changed actions | Actual handoffs | Task stages achieved |
|---|---:|---:|---:|---:|---:|---|
| Nominal |600|600|0|0|0|none|
| Simple feedback |600|552|48|6|2|none|
| Current checks/rechecks |600|504|96|12|4|none|

Every action remained under the `open the fridge` prompt. Maximum door angles were
approximately2.20e-14/2.02e-14/1.85e-14rad, numerical-scale changes, not opening.
No grasp, native milestone, strict dwell or full-sequence completion occurred.
All six actual handoffs were local-cap exits explicitly labeled unverified; no
recheck-success handoff occurred. They prove routing execution only, not recovery.

Simple samples duplicated their same-observation greedy proposal33/48 times
(68.75%); checked samples59/96 times (61.46%). These candidate/transaction statistics
do not establish whether any rejected candidate would have helped. Query spending
also changes realized action exposure. The bundled comparisons do not isolate
selection, recheck and handoff causal effects.

Inference totals158.15/160.16/161.03s; end-to-end query totals173.46/175.58/176.26s;
query medians0.2840/0.2862/0.2871s. Episode walls182.26/183.83/183.82s. All arms used
their equal600-call cap; no hidden nominal candidate calls. Full actor readiness and
formal framework efficacy remain unresolved. No real-robot validity is claimed.

## Evidence

* `metrics.json`, `freeze.json`, `launch.json`, per-arm states/images/query/action/
  synchronous decision tapes, camera config, model XML and logs.
* `replay_{arm}/verification.json`:1,656 total exact action replays, zero state gap,
  no image/controller/physical/stage/gate/handoff mismatches; initial states identical.
* `nominal_left_verification_right.mp4` and
  `nominal_left_simple_feedback_right.mp4`:640frames each,20fps, actual per-step
  pre-action evidence/checks/choices/stage status, explicit held-final-frame padding.
* Six new fridge tests passed: local support permits release before full native
  success; native success does not replace support; unknown support is not permission;
  unknown resets rechecks; exact model action selection; open-start scope separation.

Frozen v1 remains unchanged. The authorized next diagnostic asks Q1 using offline
counterfactual branches at the first three eligible verification local entries,
with selector choices frozen before outcomes. That oracle is evaluation-only and
cannot retroactively authorize online repairs or establish Q2 causal attribution.

Q1 is now complete with no useful candidate at its three sampled opportunities;
see [Q1 results](../experiment3_fridge_q1_20261019/results.md).
[Published paired videos, complete archives and independent remote recovery proof](../../../backups/autonomous_candidates_20261019/README.md).
