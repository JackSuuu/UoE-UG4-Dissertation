# Parallel observation / shared actor / autonomous candidate experiments

[Execution and video evidence contract](../../exp_record/agent_notes/parallel_diagnosis_execution_20261019.md).
User requested parallel work and per-frame decision evidence. Traces describe actual
pre-action observations, rule decisions, action source and later measured outcomes;
they are not internal model reasoning or sufficient causal proof by themselves.

## Experiment 1 preflight and launch

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

Experiments2and3are delegated to independent GPU3andGPU2work respectively. Native
task mechanism pilots do not establish full refrigerator recovery. Actual outcomes,
failures and annotated comparative videos will be indexed after completion.
