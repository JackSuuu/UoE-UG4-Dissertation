# Current goal and result-driven experiment queue

## Authoritative scope after the latest discussion

Improve complete-task reliability of already capable frozen policies through
verification, bounded model-generated repair, reverification and actual policy
handoff. Preserve the five layers: high-level subgoals, frozen VLA, risk/event
monitoring, dual-mode gating, local repair. Predictive verification must earn its
place through incremental evidence over measured feedback.

OpenVLA + native LIBERO-Long is the immediate mechanism-development platform.
RoboCasa365 + a compatible native policy is the planned second platform. This
supersedes the blanket exclusion of a second model/platform in the earlier task
selection note, while retaining its LIBERO cohort and historical cleanup record.
No RoboCasa checkpoint is selected or qualified yet. Two different model/platform
pairs demonstrate replication across pairs; separating model and platform effects
requires an additional shared-platform/model comparison.

Latest execution status: **all24baseline actor episodes completed**. Microwave4/6,
two designated mugs3/6, drawer4/6native successes; all18tapes exact. Two pots3/6
recorded successes, with four strict wrist-image replay issues retained. Microwave
meets the original per-task admission rule and takes priority for frozen L2A.
[Complete screening results](native_long_l1_results_20261019.md). L2Av1completed:
18/18valid/exact branches,552calls; six greedy controls reproduce. Neither failure
root has useful sampled32-action gain, so the precommitted L2Bgate is false.
Stop v1. Failure analysis supports preparing a timing-only v2(first grasp+120
instead of+180), with the same six roots/sources/seeds/H32and576-call cap; no v2
model query yet. Root4's new boundary precedes original loss by15actions;
preserving grasp alone still does not meet the useful-progress label. No framework gain.
[L2A results](native_microwave_l2a_results_20261019.md).

Historical22-cell checkpoint: Microwave4/6successes,
two partial-progress failures, all6exact replay: the highest-priority task meets
the frozen per-task L2admission rule. Two pots3/6recorded successes,4wrist-only
replay issues. Two mugs2/5successes, drawer4/5; all ten completed tapes exact.
Two-mug/drawer root3remain untouched after a further two-pot wrist-image mismatch;
their final continuation is now running onGPU1(supervisor3735624), while microwave
L2protocol/runner preparation proceeds. [Final-cell launch](native_long_l1_final_cells_20261019.md).
No L2model branches or L3framework comparisons have run. Historical continuation:
L1continuation launched under `53a74b3`, supervisor
3727880, in `/home/ami/scratch/native_long_l1_20261019_b`;18untouched cells on
GPUs1/2/3. Combined reporting preflight passed and the first three two-mug workers
entered setup/model loading. [Decision and diagnosis](native_long_l1_continuation_20261019.md).
Original cohort paused after6/24actor episodes under frozen source
`63fecbd`. Microwave roots0–2:2successes/3completed, all exact replay. Two-pot
roots0–2:1recorded success/3completed; all physics/integration states match but
wrist-only image differences at1/1/3frames caused the prescribed lane stops.
Fixed-state redraws demonstrate wrist-only rendering nondeterminism; retain
original replay failures and the strict gate for new episodes. R0source research
completed. See [actual launch and preflight evidence](native_long_l1_launch_20261019.md).
L2/L3/R1remain unlaunched and their downstream branch budgets must be frozen from
the preceding evidence. Labels below are new queue labels, not renamings of
historical Experiment1/2/3.

## What existing evidence establishes

* Native book successes and historical drawer completion demonstrate some task
  competence. They do not establish current four-task cohort reliability.
* Custom refrigerator pilots, observation controls and sampled candidate branches
  have not established complete autonomous retrieval or framework improvement.
* Late stagnation-only triggering, novelty-based candidate ranking and unverified
  budget-exit handoffs are actionable framework weaknesses. Actor/scene mismatch
  also remains; it cannot explain away those framework defects.
* Exact replay supports execution integrity, not policy competence or efficacy.
* Historical assembly feedback gains do not isolate a predictive ranking benefit.

## Queue L1 — native capability and natural failure map

Run four tasks × official development roots0–5 =24actor-only episodes:
microwave mug+close, two moka pots, two mugs on designated plates, bowl+drawer close.
Use the existing pinned LIBERO10 OpenVLA and official TF4.40.1 inference, native
H1, observations/instructions and success predicates. Budget520model actions and
queries plus10labelled initialization steps; episode wall cap1200s excluding load,
outer worker cap1800s. Preserve calibration6–9, reserved training10–19 (unused),
and confirmation20–49. No new actor training is scheduled.

Save each full-goal result, constituent-predicate attainment/loss, grasp/release
evidence, timing, model-action provenance and replay/video evidence. Separate
infrastructure failure, native failure and supplementary physical diagnostics.
Only validated physical endpoints can become additional success metrics.

Use the previously proposed development eligibility rule: at least2full successes
and2partial-progress failures; choose eligible tasks in the declared order above.
This is a small-sample development rule, not a population reliability estimate.
Retain every task's result. If none qualifies, report that result before declaring
any further task/perturbation expansion; do not silently relax selection rules.

Deliverable: a four-row capability table and an event-aligned failure map. This
determines which failure states to study, rather than immediately changing actors.

## Queue L2 — locate the broken link on development states

Start with recorded L1 episodes. Compare existing stagnation alarms with proposed
early progress-loss/departure/event alarms in shadow mode, including clean
successes. Offline annotation may identify onset; online alarms must be causal.
Do not assume an alarm is timely until a tested recovery branch supports that.

Then preregister a bounded set of early/mid/late snapshots around failures and
matched successful transitions (initial target: up to12snapshots, chosen by a
deterministic event rule before comparing branch outcomes). Freeze candidate
generator, count, continuation horizon and query/action/time caps before launch.
Generate candidates only with the frozen model; include unchanged continuation.
Do not repeat identical greedy queries as if they were diverse candidates.
Record any prompt/decoding/context differences and give controls the same source.

Branch from identical physics/controller/history state. Check candidate progress,
preservation of achieved goals, actual handoff and downstream outcome. Branch
outcomes are offline labels, never information to the online selector.

Result-to-change mapping:

| Evidence | Next framework change |
|---|---|
| Earlier states recover, old trigger states do not | Earlier event/progress detection |
| No useful candidates under the frozen proposal budget | Repair proposal/context mechanism; no selector efficacy claim |
| Useful candidates exist but verifier rejects/ranks them poorly | Task-directed verification/ranking |
| Local repair succeeds but completion regresses after handoff | Rechecking, progress preservation and handoff conditions |
| Successful nominal episodes are harmed | Trigger calibration and intervention restraint |

## Queue L3 — paired development comparison and attribution

First compare nominal VLA, same-source simple feedback, and the full framework on
all six development roots of the selected task (18episode cells per frozen
version, counting reusable baseline cells only if the protocol matches exactly).
Include both initially successful and failed roots. Compare matched information
and resource caps; report actual usage, not only identical caps. Define cap
exhaustion and stopping semantics before launch.

Once useful candidates exist, isolate selection and rechecking using a2×2design:
simple/verified selection × single/continuous rechecking. Shared triggers and
candidate source isolate these factors; evaluate trigger changes separately on
shared trajectories or with the repair mechanism fixed. Full end-to-end trials
still matter because methods produce different future states.

For each revision: freeze a version, name the predicted failure reduction, change
one mechanism, rerun the same development cases and retain regressions. Repeated
development outcomes are engineering evidence, not held-out efficacy estimates.
Use calibration roots only after mechanism development. Leave20–49sealed until
the selected implementation and final protocol are fixed. Held-out failures must
not become tuning feedback for the same claimed confirmation result.

Common report: full native success; rescue/harm; trigger exposure and timing;
useful-candidate availability; verified recovery; actual nominal-action handoff;
post-handoff completion; calls/actions/latency. Supplementary physical metrics
remain separate. Video: baseline left/framework right, aligned clock, labelled
holds, candidate source, verification evidence, controller owner and budget exits.

## Queue R0/R1 — second-platform readiness, then replication

R0 can proceed independently of L1: audit RoboCasa365 versions, public native
checkpoints, task support, observation/action/chunk contracts, reproducible entry
points, compute needs and CheckVLA code/weight availability. Do not select a policy
from an atomic CloseFridge example as evidence of full composite-task competence.

Candidate tasks: Prepare Coffee for entry-level qualification; Steam in Microwave
and Store Leftovers in Bowl for longer composition/progress preservation. Final
selection and episode counts depend on verified checkpoint support and are frozen
before viewing new outcomes. First reproduce actor-only full native completion.
Then port the same framework contracts and report exactly which components were
reused, calibrated or retrained. LIBERO mechanism evidence gates a large second-
platform efficacy campaign; it need not block source research or capability checks.

Compare each policy against its own matched controls. CheckVLA published aggregate
scores are contextual references unless backbone, tasks, training and evaluation
are matched. A small task subset is not a full50-task benchmark result.

## Immediate deliverables and operating discipline

1. Freeze and execute L1; publish all24attempts and their failure map.
2. Complete R0 inventory while L1 runs; record a concrete readiness decision.
3. Use L1 evidence to freeze L2, then fix the identified mechanism and run L3.
4. Reserve final evaluation and cross-platform replication for a frozen framework.

Use detached supervised jobs, healthy physical GPUs1–3 with matching CUDA/EGL,
batch1for the current actor, durable logs and per-episode status. User annual-plan
edits remain untouched. Keep one status board with planned/running/completed/
blocked states; launching a worker is not completion. Current states: L1actor-complete24/24
(four replay issues retained),
R0research complete, L2/L3/R1planned. The physical-to-EGL index map is audited
separately; equal numeric IDs are incorrect on this host. Future task or actor changes require an explicit reason
and a versioned protocol, preserving the prior results.
