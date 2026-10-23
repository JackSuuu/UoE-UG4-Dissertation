# Session recovery: experiment inventory (23 October 2026)

Read-only inspection of saved experiment outputs and current processes. No new
rollouts or replay/audit jobs were executed. This note supplements the stale
"results pending" entry in `repair_data_expansion_and_fridge_20261022.md`.

## Paths and execution state

The available project is `/home/jack/UoE-UG4-Dissertation`, with experiment outputs
under `/home/jack/scratch`. Historical manifests and commands still contain
`/home/ami/...`; preserve those original provenance records. Old commands require
path/runtime validation before reuse.

No matching experiment actor or supervisor process was found on this host during
inspection. `scratch/repair_expansion_queue_20261022/status.json` still says
`started`, but only the fridge stage has recorded completion. The training stage
has no supervisor return code or aggregate `results.json`. The cause of the
interruption is not established by the inspected logs.

## Fridge pilot: complete, negative result

Source: `/home/jack/scratch/fridge_repair_candidates_20261022_r2/`.

- All 10 episodes completed and have replay return code 0.
- `analysis/report.json`: `status=passed`, no invalid cases. This is an integrity
  result, not task success.
- Closed-door full task: baseline and slots0–3 all fail (0/5 completion).
- Initially-open grasp diagnostic: baseline and slots0–3 all fail to grasp (0/5).
- All eight repair transactions exhaust their 32-action local budget without
  verified recovery. Closed-door repairs start at step29; open-door at step44.
- 6,000 executed actions and 6,024 actor queries across all 10 episodes.
- Candidate diversity exists (four distinct first actions closed, three open),
  but no tested candidate yields a successful recovery.

## Microwave training expansion: partial results

Source: `/home/jack/scratch/framework_repair_training_20261022_r2/study/`.
Planned roots10–19; at most50 episodes, with siblings conditional on admission.

- 20 episodes have finalized `*.result.json` files; all required recorded replay
  and audit return codes are 0 (bare baselines do not need joint-model audits).
- Two additional episodes have completed actor outputs but incomplete supervisory
  finalization (details below): 22 actor-completed episodes in total.
- Six sibling jobs on roots10/14 were correctly skipped because slot0 admitted
  no repair. These are not failed episodes.
- Roots15/17/18/19 have no episode directories in this run.

|Root|Bare baseline|Greedy framework / slot0|Sampled slots1/2/3|
|---|---|---|---|
|10|Success|Success|Not applicable: no repair admission|
|11|Failure|Failure|All fail; finalized|
|12|Failure|Failure|All fail; finalized|
|13|Success|Failure|Slots1/2 fail; slot3 succeeds, audit pending|
|14|Success|Success|Not applicable: no repair admission|
|16|Failure|Failure|Slot1 actor finished with failure, checks pending; slots2/3 absent|

Across the six completed baseline/slot0 pairs: baseline3/6 versus framework2/6,
zero rescues and one harm (root13). This is an incomplete training cohort, not
the final ten-root result or held-out evaluation.

### Unfinalized outputs worth recovering

1. `candidate3_root13`: actor reports valid native success in447 actions. Saved
   `replays/candidate3_root13/verification.json` says `exact_replay_passed`, with
   459 frames and zero mismatches. No independent joint audit or finalized result
   file is present. Repair entered294, exhausted32 actions at325, handed off326,
   and completed later. `verified=false`, so this is **not a verified recovery
   chain**. Bare root13 also succeeds: this candidate avoids slot0's observed harm,
   rather than rescuing a failed bare baseline. Actor ended
   `2026-10-22T23:23:02Z`; replay finished about38 seconds later.
2. `candidate1_root16`: actor status says valid completion at the520-action budget,
   native failure,523 queries. No replay/audit or finalized result is present.
   Actor ended `2026-10-22T23:25:04Z`.

The queue's last printed starts were these two cases. Their completed tapes should
be checked/finalized before deciding which missing episodes need new execution.
The current expansion runner creates output directories with `exist_ok=False`;
it is not a resume-capable command for the existing study directory.

## Earlier positive evidence remains intact

`/home/jack/scratch/framework_repair_candidates_20261022_r1/analysis/report.md`
and `framework_repair_candidates_20261022.md` retain the earlier root4 slot3
success:506 actions,509 queries,11 repair actions. The recorded verified exit
at158 and actual nominal handoff159 are followed by347 nominal actions through
completion. All10 episodes in that earlier collector passed replay and joint
audits. This demonstrates a useful candidate exists on that reused development
case, not that a learned scorer can select it reliably. Root0 still fails for all
four slots. The latest expansion has not yet established an overall benefit.
