# M1 detached launch — 15 Oct 2026

**Later status correction:** the supervisor stopped on a worker replay assertion. An isolated unchanged-verifier retry passes that trajectory; root cause is unresolved. Nine completed episodes now have replay proofs; remaining interrupted/unstarted episodes are not task failures. See [validation/repair contract and audit](libero_validation_repair_contract_20261015.md). The details below describe the original launch, not a currently running batch.

Frozen implementation/protocol/grid/preflights pushed as **a37e354** before launch.

- Run: `/home/ami/scratch/libero_failure_mining_20261015_a`.
- Supervisor PID:1499667; workers1499671/1499672/1499673, healthy physical GPUs1/2/3, roots0/1/2.
- Phase: `all_case_paired_discovery`; planned39configurations×5arms=195live full-actor continuations.
- Supervisor and workers start in separate sessions with detached stdin and persistent logs; SSH termination does not stop them.
- All cases/all methods run; clean baseline success does not cancel intervention arms. Each root worker runs each case's VLA/hold/timed/phase/verified in sequence, then renders its verified five-method review before proceeding.
- Historical D1baseline success is not counted as a new framework result. Recorded-action preflight successes are infrastructure evidence, not task efficacy.

Launch command/PID capture: [JSON](../backups/libero_mining_launch_20261015.json).
Protocol: [M1](libero_failure_mining_protocol_20261015.md).
Literature boundary: [SEAL/CheckVLA audit](libero_literature_integration_20261015.md).
Preflights: [five-arm delay/sham](../backups/libero_mining_preflight_20261015.json), [x/yphysical fault](../backups/libero_mining_pose_preflight_20261015.json).

## Inspect progress without conflating activity and benefit

From the repository:

```bash
python scripts/status_libero_mining.py /home/ami/scratch/libero_failure_mining_20261015_a
```

Reports completed/verified counts, actual framework action changes, active step/reason and paired rescue/harm only for already verified pairs. Partial results are development discovery, not independent final evidence. `launch.json` holds overall status and worker exit codes. Per-worker logs are`worker_gpu_1.log`etc. Supervisor log is`/home/ami/scratch/libero_failure_mining_20261015_a_supervisor.log`.

Each completed case exports`review/<case_id>_methods_2x.mp4`and`review/<case_id>_trace.html`. Each episode stores`steps.jsonl`,`decision_trace.jsonl`,`substeps.jsonl`,`states.npz`,`result.json`and independent`verification.json`. A case's framework change is`selected != nominal`; external fault is`effective != selected`. Candidate forecast output, subsequent observed mismatch, repair budget and handoff remain explicit.

After all episodes verify, the supervisor produces`discovery_map.md`,`verified_summary.json`,`index.html`and the complete recovery archive in`backups/libero_failure_mining_20261015_a/`. GitHub/LFS upload and independent remote restore must be recorded after completion. At launch there is **no new verified efficacy conclusion and no completed M1video uploaded**.
