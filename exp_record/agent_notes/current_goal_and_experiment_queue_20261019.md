# Current goal and result-driven experiment queue

## Completed v2results — no net gain; local acceptance is not resumability

The21October `ab48b63` batch is complete:12/12valid episodes and12/12fresh exact
replays. Bare VLA4/6; framework4/6;0paired rescues and0harms. Each arm used2531
queries/actions (5062combined plus120initialization noops). Six full paired videos
are generated. [All-case results and evidence](../../backups/framework_comparison_20261021/README.md).

| Root | Baseline | Framework v1 | Framework v2 | v2actions | v2repair actions |
|---|---|---|---|---:|---:|
|0|Fail|Fail|Fail|520|96|
|1|Success|Success|Success|444|0|
|2|Success|Success|Success|397|0|
|3|Success|Fail|Success|352|0|
|4|Fail|Success|Fail|520|80|
|5|Success|Success|Success|298|0|

The revision removes root3's old harm but loses root4's old rescue. Framework
repair actions fall352→176, transactions11→6, unverified exits11→5. One local
repair now passes its declared contract and executes a handoff, but no repaired
episode finishes. Reduced repair occupancy does not imply halved total query cost.

**Verified from complete traces (zero-based steps):**

* Root3has no repair; its complete scientific trajectory equals fresh baseline.
  Release-transition classification is active198–351; full task completes at351.
  This is preservation of nominal self-recovery, not a successful repair example.
* Root4intervenes at148as before. V1/v2science matches through155, then prompt/
  action diverges at156: old close vs new contain-release. V2In is true at155,
  156and161–163only. At163the3-endpoint contained alternative passes the fixed
  acquisition contract and In is promoted to preservation. Mug–gripper contact
  remains true throughout161–163; position moves6.571mm from162to163. There is
  no detachment, settling, independent-support or nominal-continuation check in
  that acceptance alternative. At164the actual nominal action loses In; at165
  a second repair starts. Two later32-action repairs do not restore In, and the
  episode fails at520actions. All numerical actions are saved VLA outputs.
* Root0still fails after3transactions/96repair actions. The old repeated entry
  at294is suppressed; later context changes permit entries411and473. No transaction
  gains accepted recovery. Retry gating is not evidence of better repair actions.

**Interpretation:** the code correctly accepted its declared local condition,
but the condition did not establish a state from which nominal execution would
preserve progress. The important verifier–repair gap is therefore between local
progress and reliable resumption. The root4prompt changes before handoff; this
bundled run does not isolate prompt effects from acceptance/exit timing. Neither
“three In checks imply stable recovery” nor “a later handoff would definitely
restore success” is supported. No learned predictor was loaded in this batch.

Integrity review recomputed5062query/action/input associations and checked all
zero-query replay proofs. All six v2baseline trajectories match their v1science.
Within v2, roots1/2/3/5match end to end; roots0/4first diverge at254/148, exactly
their first interventions. Physical fingerprints, observations/diagnostics,
actions, tokens and contacts are compared; timing/controller-version fields are
not treated as physical divergence. The post-run audit is saved as
`/home/ami/scratch/framework_comparison_run_20261021/report/review.json`.

**Next research focus, not yet implemented/launched:** strengthen the verifier's
resumption contract using phase-appropriate release/contact and measured motion
evidence; keep recovery acceptance distinct from full task completion. Examine
repair-prompt suitability alongside the root4handoff failure rather than assume
that a more conservative exit alone fixes it. Use the same full six-pair framework
comparison, fixed actor/budgets and all-case outcomes for a declared next revision.
Fridge remains the next task adapter; reserved confirmation roots remain unused.

## Research framing clarification — verifier–repair remains the main contribution

On21October the user asked to restore the earlier comprehensive research-plan
structure in English. The previous condensed plan overemphasized the application
objective. The methodological subject is **runtime verification coupled to bounded
model-generated repair, reverification and reliable resumption**. Complete-task
success is its primary evaluation; unstructured manipulation is the application
and generalization setting. Microwave/fridge priorities and the five layers remain.

The core plan now restores contribution boundaries, falsifiable RQ1signal-quality /
RQ2closed-loop-benefit / RQ3robustness-and-feasibility questions, a one-year timeline,
detailed work packages, within-framework ablations, prediction adaptation, metrics
and statistical design, implementation gaps, thesis chapters, risks and gates.
Physical prediction is an explicit mechanism study inside the verifier; its
incremental contribution still requires actual native integration and experiments.
The README uses the same framing. This documentation revision changes neither the
running v2sources nor its frozen protocol; no additional experiment is launched.

## 21 October execution — full-task objective and framework v2

User requested an English, consolidated research plan and an updated README,
and authorized starting experiments. The primary objective is **completion of
unstructured multi-stage manipulation tasks**, with microwave and refrigerator
manipulation as the immediate task families. Native microwave remains a controlled
development setting, not evidence of broad unstructured generalization.

Implemented `native_microwave_measured_framework_v2` in the existing controller.
The actual five-layer runner, actor, initial states, endpoints and lifetime budgets
are retained. This is a bundled development revision; individual changes are not
causally isolated:

* Release context requires previous durable In, a recent grasp→release transition,
  no table contact, and distance ≤0.050m from the most recent durable-In position.
  Only this event gets3executed endpoints of contact-transition grace; subsequent
  exemption requires measured mug–microwave contact. Missing/out-of-region/table
  evidence ends it. Contact is not claimed to prove support or native containment.
  Loss before any durable containment still intervenes on the next query. Other
  stagnation monitors remain active; no root IDs or numeric motion commands exist.
* Entry freezes acceptance alternatives: acquire→held-clear OR contained;
  transfer→contained; release→contained AND not grasped AND no mug–gripper contact;
  close→native completion. Each alternative has its own3-endpoint streak, so
  alternating alternatives cannot accumulate a false pass. Prompt updates do not
  reset or replace this contract. Contact/unknown detachment requests release
  rather than assuming a false grasp flag means free to close.
* Newly durable In is promoted to the repair's preservation set. Violated or
  unknown preservation prevents verified exit. Actual terminal success is recorded
  separately as completed-during-repair, with no fabricated handoff or dwell.
* A failed trigger stores its exit context. After cooldown it needs3new endpoints
  showing the same changed known grasp/held/In/Close/table/microwave-contact value,
  or each ≥0.020m from its failed-exit position. An unchanged latched flag or one
  flickering endpoint cannot justify repeating that failed transaction.

All thresholds are declared development choices, not calibrated safety limits.
The six reused roots are development evidence. Learned prediction remains disabled;
the measured verification participates in entry, continuation, rechecks and exits.
No future simulator branches select actions.

Validation:16/16CPU contract tests passed via
`/home/ami/miniconda3/bin/python -B scripts/test_native_framework.py`.
Checks include retained early loss intervention, conditional release grace and its
expiry, fixed acceptance, promoted preservation, independent acceptance streaks,
retries, both repair caps, budget ties, completion during repair, actual handoff,
post-handoff regression, missing data and immutable action/decision contracts.
The runner's updated full-video overlays and legacy summary loading also passed
a2-frame zero-query smoke at `/tmp/opencode/framework_v2_video_check_mp23yt18/`.
README and core plan are now English, with local links checked.

**Historical launch checkpoint; batch now completed (results above).** Source commit
`ab48b63f0fc44ed4ecebc6c625045586e4c56688`; detached supervisor PID/session
`4082390`. Baseline uses physicalGPU2/EGL1; framework uses physicalGPU3/EGL0,
both UUID-masked and actor batch1. Both device identities were verified. At the
launch-health checkpoint, both root0workers were in actor phase with152requested
queries and151executed actions each; the last executed step150used unchanged
saved model output and controller versionv2. The supervisor was alive. No
completed result is claimed at this checkpoint.

* Freeze: `/home/ami/scratch/framework_comparison_freeze_20261021/plan.json`.
* Plan SHA256: `7fac280cd77cf1b2cd57e6594e7f923937860672bdf84f81928351f4c1d23c5e`.
* Run: `/home/ami/scratch/framework_comparison_run_20261021/`.
* Supervisor log: `/home/ami/scratch/framework_comparison_freeze_20261021/supervisor.log`.
* Durable live status: `launch.json` and each cell's `status.json` in the run.
* The detached supervisor runs all12fresh episodes, fresh exact replays after
  each pair, then automatically writes `report/report.json`, `report/report.md`
  and all six full comparison videos. Git/LFS artifact publication and remote
  recovery verification follow completion; the launcher does not itself publish.

## Frozen cohort design — the real five-layer framework,21October

The latest user instruction governs this queue and supersedes older execution
orders below. Retain OpenVLA and the five layers: subgoals → frozen VLA →
risk/event monitoring → dual-mode gate → bounded model-generated repair/recheck/
actual handoff. The objective is complete-task reliability in microwave and fridge
manipulation. **Physical prediction is one possible evidence source within this
framework, not the only verification method or a prerequisite for every run.**
Measured contact/support, phase progress, uncertainty and applicable prediction
can work together. Every treatment run must execute our real framework end to end;
bare VLA is the paired control. A geometric servo or isolated sampled branch is
not a substitute treatment. The core plan now contains only current goals/design;
historical decisions and results remain here or at their frozen commits.

### Microwave batch — frozen and launched

| Item | Planned configuration |
|---|---|
| Question | Can phase-aware loss handling, stable recovery contracts and evidence-based retry reduce harm while retaining rescue? |
| Cells | roots0–5 × bare VLA / revised real framework =12fresh complete episodes |
| Task | Native mug In microwave AND Close; no added opening stage |
| Actor | Same frozen LIBERO10OpenVLA revision8097032, TF4.40.1, greedy H1, seed0, native agentview/language |
| Budgets |520queries and520actions per cell,10separate initialization noops,1200s rollout excluding actor load; repairs count inside |
| Repair budget |32actions per transaction,160total; no budget increase to manufacture recovery |
| Evidence | Existing measured contact/goal/history verification; learned predictor not enabled in this planned first revision |
| Runtime | Existing `run_framework_comparison.py` and `native_framework.py`; GPU2/3 paired, batch1, detached supervisor |
| Outcomes | Native success, paired rescues/harms, verified repair, actual handoff, downstream completion, regressions and cost |
| Delivery | All12tapes/replays,6full paired videos, one table and current-record update |

Implementation contract, frozen before fresh model calls:

1. Distinguish release/settling near the receptacle from an actual falling or
   unrecovered loss. Use causal current/past evidence, not a root ID or the known
   later baseline result. Do not suppress all grasp-loss events by a global delay.
2. Give each repair an explicit, stable recovery target, separate from the next
   task subgoal. A newly attained relevant task milestone can satisfy an explicitly
   declared recovery alternative after rechecks; it does not silently change the
   target to a harder full-task stage. Promote durable acquired goals to preservation
   constraints. Missing support/release evidence stays unknown.
3. A failed transaction cannot rearm indefinitely on the same latched flag alone.
   Require renewed evidence or a materially changed recovery context; keep all
   failure/abstention/budget outcomes in the trace. No hand-coded movement fallback.
4. Keep nonterminal recovery checks distinct from absorbing native task success.
   Count actual handoff only after a fresh full-instruction VLA action executes;
   count completed-during-repair separately. Never relabel an old budget exit.

Exact thresholds and evidence schemas are frozen in the source/config and plan
identified above. CPU transition tests cover transient loss, retained true loss,
milestone protection, failed retry suppression and actual handoff. They are
implementation checks, not efficacy studies. The full pair batch is launched
directly; no isolated candidate-bank qualification or horizon sweep is scheduled.

### Fridge follow-on — same controller, task adapter only

Reuse the actual framework's gate/transaction/handoff with fridge state/goal
adapters. Planned first cohort: existing fixed scene and camera, seed0–2 × two
arms =6closed-start, empty-hand full-sequence episodes;600queries/actions and1200s
per arm. Both arms get the same full-task instruction initially; scoped repair
language belongs to the framework intervention. Preserve the registered released/
supported placement, closed-door and clearance dwell endpoint. Open-start runs,
if later declared, are remaining-sequence cases only. The old R2manual door/grasp
primitives are not executable repair fallbacks. Freeze the scene, initialization,
endpoint and shared-model interface before launch; these are integration tasks,
not a reason to replace the real framework with another method.

### Prediction and evidence policy

The existing self-developed `AssemblyDynamics`/`RefCheckVLA` implementation can be
reused, but its nut/peg40D state and force/tilt targets are not native microwave or
fridge models. The official OrbiSim adapter remains a skeleton. Applicable learned
prediction may later supply prospective risk/progress evidence to the same gate;
declare model/version, training/calibration data, calls, confidence and actual
decision influence. Do not force overrides to prove participation, use future
simulator outcomes online, or claim prediction is enabled when it is absent.
Predictor-on/off attribution is a later same-framework comparison, not a competing
method or a blocker for the next12episodes. Roots6–9calibration,10–19training and
20–49confirmation remain reserved; roots0–5are reused development cases.

Current status: framework v2 completed and replay-verified as recorded above.
Historical first comparison remains4/6vs4/6with root4rescue/root3harm; v2is4/6vs4/6
with neither rescue nor harm. The user-owned annual plan is untouched.

## Historical evidence and superseded execution orders

The sections below preserve what was measured or proposed at the time. Only the
active design above determines the next run.

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
