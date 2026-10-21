# Current goal and result-driven experiment queue

## Active direction — preserve OpenVLA and optimize the predictive verification loop

User reaffirmed21October: retain OpenVLA; focus on refrigerator/microwave-style
multi-stage contact manipulation; keep physical prediction and physical
verification central. Optimize the existing method through direct full-episode
experiments, rather than accumulating replacement methods or isolated diagnostics.
Backbone/platform expansion is not the immediate main line.

Reuse the existing five layers and prediction/verifier interfaces. Numerical
task and repair commands must come from the frozen policy/model, not scene-known
approach waypoints, hinge-arc servos or overwritten gripper commands. Physical
geometry remains legitimate verification evidence; it must not silently become
the action generator. Historical fridge geometric interventions remain evidence
about that older controller, not results for model-generated repair.

Concrete integration gap rechecked: `AssemblyDynamics` already subclasses the
existing ensemble dynamics model and supports autoregressive action-conditioned
state/risk prediction. Its40D nut/peg state and force/tilt risk labels are
assembly-specific. Reuse this implementation/interface with declared native
state/action features and task-relevant targets; do not reuse incompatible weights
or pretend force/deformation outputs already predict containment/grasp loss.
`RefCheckVLA` is the existing self-developed reference verifier; the official
OrbiSim adapter remains a skeleton, not a loaded pretrained component.

The next implementation work is one coherent change in the canonical comparison
entrypoint: wire a genuinely applicable predictor into pre-action verification,
retain measured post-action rechecks, correct release/loss classification and
repair/next-subgoal separation, and use model-generated repair proposals. Training
or calibration needed for native inputs must use declared data splits; the old
six-root results remain development evidence. Inference must not select actions
using future simulator branch outcomes. Forecast confidence/error, actual policy
ownership and verified versus budget-exit handoff must be visible in the trace.

Microwave provides the current paired rescue/harm regression cases; fridge is the
next manipulation target using the same framework, with a task-specific observation
adapter and goal predicates rather than a new hand-coded controller. Each completed
iteration should deliver the same full-task comparison table and full paired videos,
plus a short explanation of changed behavior and prediction's actual decision role.
A measured-only result must remain labeled measured-only. Full-system gain and
incremental predictive gain are different questions; the existing measured-only
version is retained as a later attribution control. No new batch has been launched
under this direction yet; no prediction readiness is claimed.

## 21 October follow-up — actual verifier, fridge provenance, root3/root4

Read-only audit following the user's three questions; no new actor episodes or
threshold changes. The user referenced20251019; the existing evidence is dated
20261019. Rechecked all13archived-source hashes for each of the four R2episodes,
and archived controller/runner hashes across the12native comparison episodes.

**Terminology correction:** the native comparison wired the five architectural
layers, but did **not** integrate the learned physics predictor. Its frozen plan
explicitly declares `learned_predictor=false`, `future_simulation=false` and
`privileged_current_past_endpoint_measurements`. Neither native execution nor the
R2fridge runner loads or calls `RefCheckVLA`/a learned dynamics checkpoint. The
historical assembly runner does: `load_model` loads `AssemblyDynamics(40,7)` and
`assembly_dynamics.pt`, wraps `RefCheckVLA`, then calls `forecast`. Its task-specific
state/training contract is not a validated native microwave dynamics model.
Five-layer measured feedback is an integrated development implementation, not a
completed predictive-physics framework. Do not imply predictive integration from
the architectural layer count or the word "verification".

### Fridge R2 really used the wrapper, but its repair source was geometric

Frozen source: `seed0_closed_framework/source/src/adapters/fridge_runtime.py`
under `/home/ami/scratch/fridge_r2_framework_20261019_a`. `TaskMonitor`, frozen
OpenVLA, `sense`/progress monitor, `FridgeRepair.decide` and `primitive`/`verified`
actually execute. In the runner, baseline executes `proposal`; framework executes
`candidate`. Recomputed all2400rows: baseline changes0/600actions per episode;
framework changes420/600per episode. The paired trajectories first diverge at29
(closed) and51(open); all preceding scientific prefixes match.

The closed wrapper visibly approaches the door, but neither closed arm opens it:
maximum angle in both is2.1981868044697224e-14rad. The open wrapper grasps at242
and achieves held extraction at299; baseline does neither. All four runs fail
remaining-sequence completion. This is real qualitative/local capability gain,
not a statistically established full-task success improvement. Repair uses
scene-specific geometry/pose servo and hand-coded target stages, including direct
gripper edits; these are not VLA-generated repair actions. Shared phase prompts
also mean the baseline is not completely unwrapped OpenVLA. Preserve the original
open-framework strict wrist-image replay issue and corrected local-exit/handoff
audit; neither is erased by this provenance check.

### Native root3: loss classification and sticky repeated intervention

All steps below are zero-based. Both arms first grasp at136, attain In at195–197,
then the nominal action opens the gripper at198; grasp and In become false. At199
the framework's latched `containment_lost` flag immediately selects `acquire`.
The source chooses acquire for every uncontained state without held-clear grasp,
even when the mug is at the microwave entrance/on its collision geometry.
Baseline never regrasps after198: it restores In at314and closes at351(352actions).
Thus an extra grasp is not a necessary stage of the actual successful trajectory.
Framework reacquisition never succeeds; five32-action transactions separated by
eight nominal actions consume160repair actions. No new evidence is required to
rearm the persistent loss flag. The framework mug stays near
(-0.027726,0.269664,0.945086)m from230onwards, with microwave contact.

At199the baseline and framework produce different model actions from the same
state under different instructions. The paired design establishes an integrated
intervention harm here, but does not isolate first-trigger timing, prompt choice
and later repeated repairs individually. Near-boundary release/settling must be
distinguished from genuine falling/irrecoverable loss; merely delaying every
grasp-loss trigger could remove root4's useful early intervention.

### Native root4: real model-action rescue, moving verification target

Both arms grasp at95and lose the grasp predicate at147while mug–gripper and
mug–microwave contacts remain. At148baseline continues retreat/open
([.002,-.771,.327,...,-1]); framework's acquisition-conditioned VLA instead
commands([.267,.811,.220,...,+1]) and re-establishes grasp. These are the unchanged
mapped model actions, not numeric edits by the wrapper. Baseline later remains
contact-constrained and loses grasp at230; it never attains In. Framework attains
In at155and retains it through completion at377(378actions).

At156the controller changes the active transaction from acquire to close because
In=true and the instantaneous grasp predicate=false, despite continuing mug–hand
contact. This resets the streak and moves the verification target to native full
completion. When grasp becomes true again at158, the active close goal does not
refeed to release. In also was not true at entry, so the static preservation list
does not promote newly acquired containment to a protected milestone.
The32-action repair exits unverified at179; fresh full-instruction VLA executes
at180and later succeeds. Existing verification further demands three consecutive
native-success endpoints for a close repair, but native episode termination occurs
on the first: the special terminal path records completion without a verified local
exit. Terminal success and nonterminal three-check recovery need separate semantics.
These findings explain current bookkeeping/control weaknesses; they do not prove
that an earlier verified handoff would reproduce the observed rescue.

### One next iteration, same direct comparison

Keep the same canonical runner, six pairs, full-task outcomes and full videos.
Within the controller: distinguish release/settling from loss; keep the repair
recovery contract separate from the next task subgoal; verify useful task progress
with preserved milestones before advancing; require fresh failure evidence or
changed repair context before repeating an ineffective transaction. Record actual
handoff and downstream outcomes. Do not fix root3by a blanket delay that suppresses
root4, re-label old unverified exits, or add manual motion routes. Proposed changes
remain untested. Learned physics must be explicitly accounted for: a future
integration needs a native observation/action contract and a predictor that really
forecasts it; report model/version/calls/decision influence. Existing closed-loop
logs can support that work without another isolated rollout prerequisite.

### Actor scope and possible second policy

Existing native OpenVLA screening already provides a bounded capability map:
microwave4/6, designated mugs3/6, bowl+drawer4/6(all exact); two pots3/6recorded,
with four strict replay issues retained. Custom fridge full-task skill is unproven.
No exhaustive new task sweep is needed before fixing the observed wrapper defects.

Checked official OpenPI README and LIBERO example on21October: an explicitly
released `pi05_libero` checkpoint and matching simulator interface exist; its
reported LIBERO10score is92.4%, not a measurement on our six cases. The checked
official public repository lists pi0/pi0-FAST/pi0.5, not a pi2checkpoint. Confirm
what the user's `pi_2` refers to before treating it as an available model.
Recommendation: retain OpenVLA for this controlled controller iteration, and use
an available native LIBERO policy such as pi0.5as the next backbone comparison.
For each backbone compare its own bare policy vs the same framework, with native
inputs/action cadence declared and matched within that pair. Camera/state/chunk
differences prevent attributing a cross-backbone score change to weights alone.
Stronger native manipulation would not repair incorrect gating/handoff logic or
automatically establish custom-fridge capability. No downloads or model calls
were made during this source review.

Sources: [OpenPI](https://github.com/Physical-Intelligence/openpi),
[official LIBERO checkpoint/results](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md),
[R2 original results](fridge_r2_results_20261019.md),
[native paired evidence](../../backups/framework_comparison_20261019/README.md).

## Current execution directive — direct complete-framework comparison

**Completed direct comparison (`09d4b2a`):12/12valid, exact fresh replays.**
Bare VLA4/6; measured-verification framework4/6; one rescue(root4), one harm(root3).
Queries/actions:2531baseline,2557framework (5088total), plus120initialization
actions and zero-query replays. All six full paired videos are in the run's
`report/`; publication directory is `backups/framework_comparison_20261019`.
Published at`d1a48a1`; independently restored7archives/12,676members/6videos
from a fresh remote clone, then recomputed all5088query/action/input associations,
aggregate outcomes and six paired divergence points. Recovery proof is in the
same publication directory. No further model calls were used for this audit.

Framework actually intervened on roots0/3/4, using160/160/32model-generated repair
actions. Roots1/2/5never intervened and their complete scientific trajectories
match baseline. Other pairs are identical until first intervention at zero-based
actions254/199/148respectively. No measured repair transaction passed verification:
all11exited at the32-action cap, with actual original-instruction VLA handoffs.
Root4completed at378actions after handoff at180(zero-based), but this is an
**unverified-budget-exit handoff**, not verified repair→handoff success.

The paired run directly identifies concrete issues: root3briefly achieved In at
195–197then lost it at198; the baseline recovered unaided and completed at352
actions, while framework intervened at199and repeatedly requested acquisition,
exhausting160repair actions and failing. Root4intervened at148after elevated
grasp loss, achieved In at155, switched local goal to closing at156, and exited
unverified at179before eventual nominal completion. This exposes both an
over-intervention harm and a verification/goal-refeed mismatch: useful acquired
containment did not satisfy the transaction's revised closing objective. These
are trace-supported development findings, not held-out estimates or isolated
predictor effects. Root0's repeated acquire prompts still made no useful recovery.
Next iteration should address these observed gate/refeed/repeated-repair issues
within the same canonical entrypoint and repeat the full two-arm comparison;
no isolated candidate study is added. No second comparison has been launched.

The user's latest instruction supersedes the diagnostic-first queue below:
**run bare VLA versus the complete five-layer framework end to end**, on the same
native task, initial states, observations and total budgets. Primary output is
full-task outcomes and paired full-episode videos. Diagnose failures from those
actual runs; do not require further isolated candidate/horizon studies first.

Use one canonical runner and this current record. Reuse existing framework
modules; all five layers must actually execute, including online verification,
bounded model-generated repair, reverification and real frozen-VLA handoff.
Do not label sampled-policy branches as the complete framework. Start with two
arms only; additional ablations follow a concrete issue in the paired runs.

H96 diagnostic was stopped on user instruction: supervisor3765267exited,
192queries on C0roots0/1, sampling never enabled, no active v3processes remain.
Its partial artifacts remain at `/home/ami/scratch/native_microwave_l2a_20261019_v3`.
The generic supervisor status `aborted`/`infrastructure_or_integrity_failure`
reflects external termination here, not an established actor/integrity failure.
No H96 capability conclusion or automatic restart. Historical evidence below is
retained for reference, not the active execution plan.

Implemented direct comparison: `scripts/run_framework_comparison.py` (prepare,
run, replay, report/video) with `src/adapters/native_framework.py` (five-layer
controller). Microwave roots0–5 × baseline/framework;520model queries and520
executed actions per episode, ten separate initialization actions, same1200s
rollout cap. One model query per action; repair counts inside the lifetime.
Framework uses task-semantic subgoal tracking and privileged **measured** online
verification, not a trained predictive verifier. All numeric actions come from
the unchanged frozen VLA; no geometric motion fallback. A separate LLM planner
is not required by the locked architecture and is not a new prerequisite.
Per-repair32actions/total160, measured progress/preservation checks, verified or
explicitly unverified exits, and fresh original-instruction action execution on
handoff. Baseline keeps the native full instruction and logs the same monitor
in shadow. Complete-system differences include repair instruction conditioning;
this two-arm experiment does not isolate a predictive-model contribution.

Launch preparation completed:12cells frozen at
`/home/ami/scratch/framework_comparison_freeze_20261019/plan.json`, SHA256
`4c9a12af16f4e53a24f4fe7174c3928c90233c471d4e4607b6e1f8c35493e694`.
Controller11/11CPU checks passed, including preservation, unverified budget exit
and executed handoff. GPU2runs baseline, GPU3runs framework, paired by root;
fresh zero-query replays and six full videos follow automatically. Output:
`/home/ami/scratch/framework_comparison_run_20261019`. Actual process and query
status is recorded in its `launch.json` and each episode's `status.json`.

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
Stop v1. Timing-only v2also completed18/18valid/exact cells,576calls; unchanged
failure-root gate false, so no L2B. Root4/C2retains held-clear level1but this equals
its snapshot reference and is not useful progress. Only root3/C2(original-success
control) gains containment. [V2results](native_microwave_l2_v2_results_20261019.md).
Root4/C2moves at most0.954mm while retaining grasp: contact-constrained plateau,
not measured delayed placement. Prepare one final horizon-only v3(H96,1,728new
calls), same states/sources/seeds, every fresh H32prefix must exactly match v2
before query33. If valid and negative, stop further horizon tuning. No framework gain.
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
L3/R1remain unlaunched; further branch budgets must be frozen from
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
R0research complete, L2Av1/v2complete/gatesfalse, L3/R1planned.
The physical-to-EGL index map is audited
separately; equal numeric IDs are incorrect on this host. Future task or actor changes require an explicit reason
and a versioned protocol, preserving the prior results.
