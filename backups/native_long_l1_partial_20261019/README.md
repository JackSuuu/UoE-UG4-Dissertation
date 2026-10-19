# First six native long-task episodes — original stopped cohort

Four archives preserve4,128files from the stopped original24-cell plan, including
all6executed actor episodes and replay attempts,18blocked cells, source manifests,
both-camera tapes, physical/controller state fingerprints, query/action logs and
raw videos. Manifest verification passed locally. Remote recovery is separate.

Published at `0038f38`. First fresh-clone recovery timed out in the LFS pull after
600s with no downloaded objects; all four files remained pointers. Evidence is
retained in `remote_recovery_attempt1.json`. Diagnosis found serial ordinary-Git
blob fetching during LFS full-tree size scanning in the partial clone. The bounded
retry now independently restored all4archives/4,128members (757,735,833bytes), with
all archive and member hashes passing the pinned verifier. See
`remote_recovery_passed.json`; this specifically certifies the six-episode archive.

`partial_results.json` records the6observed outcomes:

| Task | Root0 | Root1 | Root2 | Integrity |
|---|---|---|---|---|
| Microwave | failure,520actions | success,444 | success,397 | all exact replay |
| Two pots | recorded success,500 | failure,520 | failure,520 | wrist-only mismatch at1/1/3frames |

Total2,901model queries and2,901actor actions;60initialization noops are separate.
The independent replays execute the same2,901recorded actions and60initialization
steps with zero model queries. Physics/integration maximum gap is zero in all six.
Two-pot wrist-image mismatches remain failed exact-replay results, not silently
accepted tolerances. Native success is distinct from released/stationary/dwell
criteria; no framework intervention or efficacy claim is made.

The18unstarted cells are not failures of the actor. Their later continuation must
reference this cohort without overwriting it or repeating its completed episodes.
Full24-cell task selection remains provisional.

Verify with `python scripts/backup_research_progress.py verify backups/native_long_l1_partial_20261019`.
