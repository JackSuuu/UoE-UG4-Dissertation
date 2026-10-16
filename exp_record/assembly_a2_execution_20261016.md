# A2 execution record

User authorized integration of shared predictive gating and geometric feedback,
prioritizing full-task performance and diagnosis on actually encountered states.
Protocol: `assembly_a2_protocol_20261016.md`; original A1 sources remain hash-exact.

## Preflight

`/home/ami/scratch/assembly_a2_preflight_20261016_a/report.json`: passed seven
mechanism invariants and four32-step physical fixtures, one per arm, with separate
process replay of images/physics/full forecasts/monitor/decisions/actions/outcomes.
Nominal proposals were recorded A1 actions, **not** online actor inference. These
fixtures are not efficacy results. Changed intervals:0/19/29/29. Hybrid predictions
were rejected as unreliable at all29 interventions: fallback genuinely executes;
an invariant test separately verifies a trusted prediction can override geometry.
No reliability thresholds were tuned after these outcomes.

## Scheduled execution

Read-only audit of all30 actual A1 live trajectories precedes80 live episodes:
seeds3000/3001 regression and3100/3101 prospective development, five offsets,
four arms. All intervention caps80, common horizon160, exact original endpoint.
The source checkpoint is copied/hash-checked, never retrained in this integration.
Healthy GPUs1/2/3; detached supervisor with independent process replay after each
episode; failures preserve partial evidence and stop new scheduling. Full output
will be locally archived after successful completion.

Run directory: `/home/ami/scratch/assembly_a2_20261016_a`.
Read-only command: `python scripts/status_assembly_hybrid.py /home/ami/scratch/assembly_a2_20261016_a`.
Launch details and actual progress are recorded below after starting the supervisor.

## Launched

Frozen implementation/protocol/preflight committed and pushed as
`6d44dfe120444ff2cec26aab503c634a05f351fb` before launch. Detached supervisor2251239
started the actual-state audit worker2251243 on GPU2. At the recorded launch
snapshot:9/30 original trajectories audited,0/80 new live trajectories verified,
no failed jobs; phase=`actual_state_audit`. Completion of the audit automatically
starts the registered four-arm comparison on GPUs1/2/3. This is execution underway,
not a statement that the hybrid improves efficacy.

Independent GitHub/LFS recovery verified all29 preflight archive members and all19
frozen source/protocol files. Proof:
`backups/assembly_a2_preflight_remote_verified_20261016.json`.
Launch status snapshot:`backups/assembly_a2_launch_snapshot_20261016.json`.
The original user modification in`notes/Experiment_plan_1year.md` is preserved.
Full online result/archive remote backup remains pending experiment completion.
