# Experiment3 native-book mechanism preflight: negative exploratory result

Frozen protocol/source manifest SHA256:
`2b6761766e99d79d7ebaeb55eac5eac616e673ce0bbdfccf43b3a5bce4548bef`.
Data: `/home/ami/scratch/experiment3_candidates_book_20261019_a`.
All three arms used GPU2, the same frozen official TF4.40.1 OpenVLA, native
book/init0/seed0, shared600-call/600-action/1200s maxima, and strict20-step dwell
termination. No manual motion or future simulator candidate screening was used.

| Arm | Calls | Actions | Sample calls | Changed actions | Native milestone | Strict/max dwell | Actual handoffs |
|---|---:|---:|---:|---:|---|---|---:|
| Nominal greedy |600|600|0|0|step358|false /0|0|
| Simple feedback |600|474|126|22|none|false /0|5|
| Command-check/reverification |600|530|70|10|none|false /0|2|

All first grasps occurred at step67. The baseline continued for241 actions after
its native milestone, without ever satisfying even one strict-placement interval.
The sampled arms consumed their common query budget earlier; they do not have
matched realized action exposure. No strict-placement candidate occurred in any
arm. This one development initial state provides no efficacy estimate or fridge
transfer claim. The baseline attaining native success while sampled arms did not
is retained as adverse exploratory evidence, not filtered away.

Sampling duplicated the same-observation greedy action in72/126 simple samples
(57.1%) and54/70 verification samples (77.1%). The verification arm's two actual
handoffs at steps213/499 were **local-cap exits**, not verified-progress exits.
Its final transaction was censored by the query budget. Simple exits comprised
four local-cap exits and one single-measured-progress exit; its final transaction
was also censored. Thus there is no demonstrated successful reverification-driven
handoff, validated repair, uncertainty-aware verification, or downstream completion.
Command checks never claimed knowledge of future collision/support/progress.

Model inference totals were159.30/160.55/161.07s (nominal/simple/verification).
Query wall totals including preprocessing/IPC were174.85/176.58/176.91s; medians
0.2861/0.2883/0.2886s. Episode wall times were185.15/184.84/186.74s. Simulation
paused during inference. No20Hz real-world inference-rate claim is made.

## Evidence

* `freeze.json`:29 frozen source hashes, model weight/config/code hashes and
  inspected LIBERO/robosuite continuation sources, created before any trial.
* Each arm: `steps.jsonl`, `queries.jsonl`, `states.npz`, `images.npz`,
  `status.json`, model XML, initial and policy-input images, worker logs.
* `replay_{arm}/verification.json`: independent process replays of all1604
  executed actions, exact state/controller/image/diagnostic and gate recomputation,
  zero mismatches and zero maximum state gap. Model re-query identity is not tested.
* `audit.json`: candidate provenance, accounting, genuine handoff events, identical
  initial fingerprints,29 source hashes and MP4 frame counts checked; passed.
* Six gate/budget/handoff unit tests passed before source freeze.
* `nominal_left_verification_right.mp4` and
  `nominal_left_simple_feedback_right.mp4`:1280x930,640frames,32s,20fps.
  Every pre-action frame has its actual saved decision trace. Shorter runs explicitly
  hold their final frame; both end with40 held frames, not added physics or dwell.
  First intervention frame201 and held frame639 were visually inspected.

No frozen book source/protocol is changed after running. Following user steering,
further work moves to a separately versioned full-fridge integration; book results
remain a failed mechanism preflight, without further tuning on this initial state.
