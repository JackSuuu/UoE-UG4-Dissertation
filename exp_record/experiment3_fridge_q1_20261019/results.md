# Q1 completed: no useful candidate available; Q2 selection remains unassessable

Artifacts: `/home/ami/scratch/experiment3_fridge_q1_20261019_a`.
Status: **completed_exact_verified**, detached cohort finished, GPU2 released.
Pre-branch protocol/source/plan freeze SHA256:
`8af817ad19b2ed2e64a2edb41d56914698e2fbd2d360491c7840ba579ba4b17f`.

The first three eligible local-entry rows from the exact-replayed fridge v1
verification arm were368,403,443. All nine saved candidates, including duplicates,
were evaluated under the frozen protocol. No outcome-selected opportunity or
branch was substituted. Each branch executed the saved candidate,11 fresh greedy
actions, and20 fresh greedy nominal-continuation actions on its own actual RGB and
shared phase prompt. The entire experiment is **offline evaluation only**.

| Source entry | Action duplicate groups | Frozen simple choice | Frozen verification choice | Useful candidates at1 /12 /32 |
|---|---|---:|---:|---|
|368|[0,1,2]|1|0|0 /0 /0|
|403|[0,2], [1]|1|0|0 /0 /0|
|443|[0], [1], [2]|1|1|0 /0 /0|

Candidate indices0/1/2 mean original greedy/sample/sample. Duplicate counts give
six distinct state/action pairs across nine branches, not nine independent samples.
All duplicate-branch state tapes, image tapes and subsequent model action sequences
were exactly identical within their duplicate groups (`postrun_audit.json`).

## Q1 and Q2 interpretation

* At all three horizons:0/3 opportunities had any useful candidate;0/9 candidates
  satisfied the opening stage. No phase was advanced and no task completion occurred.
* Maximum32-action opening goal-gap reduction was1.3322676295501878e-15rad,
  numerical-scale and far below the frozen.01rad descriptive useful-opening threshold.
  Different arm motions/commands are not counted as task progress.
* Both selectors' useful rate **conditional on useful availability is undefined**
  (`null`), because the denominator is zero. This is inability to assess useful
  selection, not evidence that verification and simple feedback are equivalent.
* Rejected-useful candidate count was zero. Offline oracle directional-margin regret
  was zero here because no candidate moved the door meaningfully; this does not
  establish optimal selection or verifier competence.
* No guarded prior-achievement loss or newly worsened item-penetration proxy was
  detected. No prior task-stage milestone had been achieved at these source states.
  Physical damage and validated contact safety remain **unknown**; these are not
  harm-free or safety-certified rollouts.
* The source states are biased toward the v1 verification policy's distribution,
  with one seed/camera/checkpoint and short horizons. They do not establish that
  every possible model candidate or longer continuation is incapable.

The20-step tail is counterfactual nominal continuation, **not an actual online
successful repair/handoff**. Fridge v1's six original actual handoffs were cap exits,
not recheck-success exits. Q2 selection/recheck/real-handoff causal factor experiments
remain gated on useful proposal/actor capability; the bundled three-arm pilot does
not isolate those effects. No cloth work or additional tuning is undertaken here.

## Exactness, costs, and resource release

Each of nine branches and each independent replay reconstructed a fresh environment
by replaying the full original action prefix. All before/after source-prefix
fingerprints, entry raw states/images/controller fingerprints and read_state/stage
histories matched exactly. All288 branch actions then replayed with zero state gap,
matching images, contacts, stage evidence and action provenance. No approximate
state fork, tolerance change, geometric action, rollout-informed selector or training.

*288 new branch actions;279 fresh official TF4.40.1 model calls; batch1/GPU2.
* Original source verification arm:600 model calls. The nine saved candidate calls
  were part of those600, with2.5646s recorded model time; reused, not fresh Q1 calls.
* Fresh Q1 model inference:76.4964s; query wall including preprocessing/IPC:98.6665s.
* Separately reported overhead:3,642 source-prefix replay actions during execution,
  another3,642 during independent replay,288 independent branch-replay actions,
  and360 standard settling actions across18 fresh environments.
* Entire detached cohort including replay/review:349.81s, within3600s. Each branch
  and its independent replay stayed within the shared1200s branch bound.
* Post-completion `nvidia-smi` compute-app listing contained only GPU1 activity;
  no GPU2 process remained. GPU2 is available to the main session.

## Files for integration/archive

Within the artifact root:

* `plan.json`: chronological opportunity selection, all candidates/duplicates,
  common prior-action history and both selector choices frozen before outcomes.
* `freeze.json`, `source/`, `base_source/`, `camera_config.json`: Q1 and exact v1
  source snapshots; model/code/environment and original-tape hashes.
* `results.json`, `launch.json`, `postrun_audit.json`: outcomes/accounting and checks.
* `step{368,403,443}_candidate{0,1,2}/`: `prefix_identity.json`, `steps.jsonl`,
  `queries.jsonl`, `states.npz`, `images.npz`, `before.png`, `after.png`, status/logs.
* `replay_step*_candidate*/`: independent exact-prefix/replay evidence.
* `step368_candidates_offline.mp4`, `step403_candidates_offline.mp4`,
  `step443_candidates_offline.mp4`:1536x850,72frames/3.6s each at20fps;32 real
  pre-action frames plus40 explicitly held final frames.
* `all_opportunities_candidates_offline.mp4`: concatenation of the three actual
  branch comparisons. `step*_frame{0,12,71}.png` are annotated stills; source403
  first frame and source443 final hold were visually inspected.

New implementation: `scripts/experiment3_q1_{metrics,run,launch,review}.py`,
`scripts/test_experiment3_q1.py`; five pre-freeze progress-definition tests passed.
All13 owned frozen book/fridge/Q1 implementation files still match their run hashes.
Main-owned files and frozen prior code were not edited; no commits or pushes.
