# L1 launch and preflight evidence

[Actual launch record](../../exp_record/agent_notes/native_long_l1_launch_20261019.md).
The24-episode cohort is running; this archive contains launch/preflight evidence,
not its future task results.

`manifest.json` hashes2,280archive members: frozen job plan, source/package/model
inventories, GPU mappings, four scene/replay checks, one model-query check and
labelled synthetic orchestration/summary/archive checks. Original scene checks
executed40initialization noops, replay checks another40; no task-solving actions.
The model interface check made one query and executed no robot action.

`initial_health.json` records the actual supervisor/worker PIDs, physical GPUs,
EGL mapping and first4model actions per worker after cohort launch. It is a
point-in-time health record, not a completion or framework-benefit claim.

Verify/restore with `scripts/backup_research_progress.py`. Full episode artifacts
will be archived separately after the cohort stops. Independent remote recovery
is recorded separately once performed.

First fresh-clone recovery attempt timed out during LFS download after180s; both
archives remained pointers and no restoration was claimed. The failure is retained
in `remote_recovery_attempt1.json`. The bounded retry succeeded: a fresh GitHub
clone/LFS download restored and hash-verified all2,280launch members plus35R0
source members. See `remote_recovery_passed.json`. This proof covers launch/R0
artifacts only; the six-episode result archive has a separate recovery check.
