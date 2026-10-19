# Parallel observation / shared actor / autonomous candidate experiments

[Execution and video evidence contract](../../exp_record/agent_notes/parallel_diagnosis_execution_20261019.md).
User requested parallel work and per-frame decision evidence. Traces describe actual
pre-action observations, rule decisions, action source and later measured outcomes;
they are not internal model reasoning or sufficient causal proof by themselves.

## Experiment 1 preflight and launch

### Completed cohort

All six600-action actor runs have zero measured grasps/native successes. This is
an open-start grasp diagnosis; full retrieval is unscored. All six corrected cached
replays pass exactly; original300s timeout records remain preserved. Camera pairs
match initial physics/controller/wrist images. Actual seed item-position separation
is only3.4–15.5mm, so this is narrow development coverage, not a generalization test.
The fixed framing change does not establish an occlusion solution or skill benefit.

* [Seed0 video](seed0_original_left_framed_right.mp4)
* [Seed1 video](seed1_original_left_framed_right.mp4)
* [Seed2 video](seed2_original_left_framed_right.mp4)

Original camera left / frozen framed camera right; actor-only both sides,20fps
simulation playback,641frames each (initial+600actual post-action frames+40extra
labelled holds). Inference pauses physics. Annotation audit preserves pre-action
evidence/action timing; no added dwell from held frames.

[Completed evidence archive](observation_cohort.tar.gz) · [SHA256 manifest](manifest.json)
· [Detailed results](../../exp_record/agent_notes/fridge_observation_results_20261019.md).
Includes all rollouts, raw images/states/actions, frozen sources, original failed
replays, successful cached replays, finalizer/renderer provenance, annotation audit,
review images/videos and logs. Upstream weights/runtime installs and disposable
`__pycache__` are excluded; pinned provenance remains in frozen sources/manifests.

### Historical preflight

* [Fixed camera calibration](camera_config.json): seed0 reset geometry only, no
  action-rollout selection. Same frozen pose reused across seeds0/1/2.
* [One-action independent exact replay](observation_preflight_replay.json).
* [Actual one-action trace preview](trace_preflight.png): actor-only, verification
  inactive, no intervention. This is not a completed experiment video.
* [Detached launch snapshot](observation_launch_snapshot.json): GPU1, six episodes,
  source freeze`7a158cd`, supervisor3673371. Results pending at this snapshot.
* [207-member preflight archive](observation_preflight.tar.gz) and
  [SHA256 manifest](observation_preflight_manifest.json): all scene calibration/smoke
  records and frozen sources, one-action actor tape and independent replay.
* [Independent remote preflight recovery](preflight_remote_verified.json):207/207
  members downloaded from GitHub/LFS and SHA256 verified from`29917e9`.

Experiments2and3are delegated to independent GPU3andGPU2work respectively. Native
task mechanism pilots do not establish full refrigerator recovery. Actual outcomes,
failures and annotated comparative videos will be indexed after completion.

Those launch statements are historical. Completed
[shared actor results/videos](../shared_actor_20261019/README.md) and
[autonomous candidate/Q1 results/videos](../autonomous_candidates_20261019/README.md)
are published separately; neither demonstrates full fridge recovery.
