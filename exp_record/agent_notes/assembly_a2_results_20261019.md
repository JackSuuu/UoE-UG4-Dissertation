# A2 complete: useful integrated feedback, no predictive-ranking contribution

Read-only inspection19Oct:30/30 actual A1 state trajectories audited;80/80 A2live
episodes independently replay-verified, no unresolved entries,161subprocess jobs
exit0,55.943minutes. Source/protocol freeze:`6d44dfe`; no new manipulation rollout
was launched during this inspection.

| Arm | Regression success | Prospective development success | Total | Mean episode peak nut/peg load |
|---|---:|---:|---:|---:|
|VLA|3/10|5/10|8/20|69.124N|
|feedback80|9/10|9/10|18/20|27.969N|
|phase_geometry80|9/10|9/10|18/20|30.387N|
|hybrid80|9/10|9/10|18/20|30.387N|

Hybrid vs VLA:11rescues,1harm,7both-success,1both-failure. Hybrid vs feedback:
1rescue,1harm,17both-success,1both-failure. Hybrid/phase_geometry all20trajectories
have identical pre-action fingerprints, effective actions, post-action fingerprints
and endpoints. Prediction-ranking overrides=0;1600geometry repairs. Reasons:
1332untrusted,266no accepted better candidate,2insufficient predicted advantage.
Shared predictive gating remains active; no predictor-free trigger comparison exists.

Feedback uses1442changes, exhausts80in13/20; hybrid uses1600changes, exhausts80in20/20.
Mean episode steps: feedback117.3, hybrid125.0 (failure cap included; not successful-only
efficiency). Mean peak loads are simulated nut/peg sums, not certified safety metrics.
VLA actor median~0.712s, hybrid decision median~4.49ms; full loop is not20Hz real-time.

Failure3001/c4is a phase-management harm: VLA and feedback succeed, hybrid remains
unseated/touching at160.3100/c0fails all arms' complete criterion; hybrid is released
and native-successful but geometric containment fails. Both stay in the final table.

All80original videos exist. Added20two-panel matched comparisons in
`/home/ami/scratch/assembly_a2_20261016_a/paired_review/index.html`. Published highlights
are selected deterministically: first both-success3000/c0, rescue3000/c1, harm3001/c4,
both-failure3100/c0, and first prospective rescue3100/c2. All20cases are in the manifest.
See [paired-video directory](../../backups/assembly_a2_paired_review_20261019/README.md).
Full original697-file archive is locally hash-verified; paired videos are a separate
later artifact, not retroactively listed in that frozen archive.

Conclusion: the runtime feedback framework is useful relative to bare VLA here;
extra phase/ranking machinery has not improved the shared-gate geometric baseline.
Proceed to the user's unstructured/long-horizon questions rather than further
optimizing this development task. New task-family/real-time claims remain untested.
