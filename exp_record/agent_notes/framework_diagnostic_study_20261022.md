# Framework diagnostic study — 22 October 2026

## Research purpose and protocol

The user requested additional experiments and detailed logs to decide how to
adjust the existing verification–repair architecture. The primary question is
whether intervention is warranted and produces useful recovery, not whether a
different task-specific prompt can finish root0.

The controller, trained artifacts, prompt dictionary and action generation are
frozen under the existing plan:

- Plan: `/home/ami/scratch/multi_root_control_general_plan/plan.json`
- Plan SHA256: `11d1ebc243a27383a7ddf09de9bfeac505572a8910a2f47e12217020eac1f9bd`
- Study outputs: `/home/ami/scratch/framework_diagnostic_20261022_r1/`
- Existing eight episodes: `/home/ami/scratch/multi_root_control_general_20261022/`
- Three arms over microwave roots0–5: bare VLA; full joint v6 with fusion gate;
  measured-controller v2 with the same fusion gate, without joint components.
- Reuse the eight completed baseline/joint root0–3 episodes; collect the four
  missing root4–5 episodes and six fresh measured-fusion episodes.
- Primary endpoint: complete native In AND Close success. Frozen greedy H1
  OpenVLA, 520 actions/queries, 10 initialization noops, 32 local / 160 total
  repair-action caps, unchanged mapped VLA proposals.
- GPU2/3 only, one serial queue per lane, at most two actors. Every valid episode
  receives exact tape replay; joint episodes also receive independent neural
  recomputation. Infrastructure failures remain distinct from task failures.
- This is reused-root, single-task development evidence. It is not a held-out
  generalization estimate; roots1/2/5 were used for artifact training.

The additional arm removes **joint stage inference, episodic retrieval,
prediction and risk together**. It is a bundled diagnostic ablation, not a
risk-only causal attribution. The same fixed stage instructions remain available
in both framework arms. No oracle skills or numerical action modifications are
enabled.

The canonical plan still contains historical 36-cell factorial metadata. The
study supervisor's separately frozen `protocol.json` specifies the selected
cells and actual modes; the unused prompt-factorial cells are not scheduled.

## Instrumentation

- `scripts/run_framework_diagnostic_study.py`: canonical workers, exclusive
  output creation, explicit mode/provenance checks for reuse, bounded child
  processes, two GPU queues, replay and joint audit.
- `scripts/analyze_framework_diagnostics.py`: per-action JSONL/CSV, target-bound
  contacts, measured cup/EEF displacement, raw/native/mapped actions, risk and
  prediction evidence, acceptance streaks, fusion exit vetoes, budgets,
  transaction summaries, exact-prefix alignment and reconstructed retry reasons.
- `scripts/audit_framework_nominal_risk.py`: observational inference on saved
  baseline policy images and states; no actor calls or fitting. Evaluates the
  exact training-label function on future nominal tape only for offline analysis.
  Region-dependent exit prediction is explicitly unavailable in this diagnostic;
  stage, risk and no-lift computation do not use the region.

All action indices are **zero based**. Before action t is observation t; after
action t is observation t+1. Later repairs are on diverged states: a baseline
window at the same action index is not a same-state counterfactual.

Gripper convention was checked against the frozen candidate worker's inversion
and installed `PandaGripper.format_action`: mapped +1 closes, mapped -1 opens.
Native proposal and mapped gripper command are different representations.

## Initial eight-episode findings (subsequently replayed and audited)

| Root | Baseline | Full joint | First joint repair action |
|---|---|---|---|
| 0 | fail,520 | fail,520 |104: joint_no_lift|
| 1 | success,444 | fail,520 |89: joint_outcome_risk|
| 2 | success,397 | fail,520 |145: joint_outcome_risk|
| 3 | success,352 | success,352 |none|

1. On roots0/1/2, the first numerical action divergence is precisely the first
   repair action. The pre-action fingerprint is identical through that entry.
   Root3 has identical full action tapes and no joint repair.
2. At roots1/2's first joint entry, the next24baseline actions receive label0
   from the **existing training definition** (`observed_nominal_window_without_defined_violation`).
   Thus the first risk triggers are false positives against that short-horizon
   definition as well as preceding observed full-task harms. This does not prove
   that disabling just those triggers would restore success under later triggers.
3. All15joint repairs on roots0–2 use32actions, meet no acceptance alternative
   on any step, and exit unverified. Fusion blocks **zero eligible exits** in
   these repairs. A stricter fusion threshold is not the immediate cause of their
   failure to recover.
4. All480mapped gripper commands during these repairs command closing. Earlier
   opening-dominant interpretations were sign-reversed. This observation alone
   does not establish the mechanical cause of no lift.
5. Same-trigger retries are reconstructable: root0 R2 is admitted by >=20mm
   displacement; other same-trigger retries in these three cases are admitted
   by three stable changed table-contact endpoints. A new trigger name also
   starts a fresh failure-memory entry. These are candidates for contract-based
   retry analysis, not proof that every table-contact change is spurious.

The initial CPU nominal-shadow run reproduces first risk-trigger indices89/145,
but its scores differ from the recorded GPU scores by approximately0.00127 and
0.00030. A strict1e-5 score-agreement check failed. CPU results are retained as
exploratory evidence. The subsequent GPU check below establishes agreement on
the shared prefixes; the CPU scores are not numerically equivalent.
This does not affect the labels computed from recorded baseline states.

## Completed controls and additional root4 evidence

All12baseline/joint episodes are now valid, with12/12exact tape replays and
6/6independent joint recomputations passing without mismatch. Baseline succeeds
on roots1/2/3/5 (4/6); full joint succeeds on roots3/5 (2/6). Root4 now ends
normally at520actions, not an infrastructure failure. Roots3/5 have identical
baseline/joint action tapes and no joint repairs. Total joint repair cost is640
actions across20transactions, with zero verified repairs.

Root4 is importantly different from roots0–2: R1/R2 satisfy local recovery
conditions, and the fusion layer records24/14eligible-but-blocked endpoints.
These counts include any cap-tied endpoints; they do not all represent a gate
as the exclusive exit blocker. Local recovery does not itself prove that a
nominal continuation would succeed, so these are not automatically false
rejections. Root4's first risk entry at98also has label0on the shared-state
baseline nominal24window. This adds a handoff/stage-selection diagnostic case
alongside roots1/2's intervention harms.

For root4 R1,25/32requested stages differ from the measured-stage reference:
the object is already held clear but the joint selection remains `acquire`.
This reference is a declared rule, not independent stage ground truth. Eligible
fusion probabilities span0.116–0.715 in R1 and0.979–0.981 in R2, below the fixed
0.9934257threshold; neither risk nor prediction veto is active at those eligible
endpoints. Consequently, it is necessary to distinguish stage selection from
handoff-score rejection and to evaluate actual nominal continuation before
loosening the gate.

The CPU-only nominal-shadow diagnostics cover all six baseline tapes. The first
risk-eligible observations are129/89/145/none/98/none for roots0–5; no-lift
eligibility first appears at104/117/198/none/189/none. Successful roots1/2 contain
many eligible observations: delayed/no-lift signals cannot be equated with
inevitable full-task failure. Subsequent GPU inference confirms the shared-prefix
scores and decisions (details below).

Removing just the first risk trigger is not known to resolve roots1/2: nominal
shadow no-lift conditions later become eligible at117/198 even on their
successful baseline tapes. Their complete-task success coexists with some true
short-horizon stagnation labels. A useful intervention gate must distinguish
detecting a local condition from evidence that spending repair budget helps.

## Final results: 18 valid episodes

Ten new episodes were collected and eight existing episodes reused. All runs
finished normally; none of the 18 outcomes is an infrastructure failure.

| Root | Baseline | Full joint | Measured + fusion |
|---|---|---|---|
| 0 | fail, 520 | fail, 520 | fail, 520 |
| 1 | success, 444 | fail, 520 | success, 444 |
| 2 | success, 397 | fail, 520 | success, 397 |
| 3 | success, 352 | success, 352 | success, 352 |
| 4 | fail, 520 | fail, 520 | fail, 520 |
| 5 | success, 298 | success, 298 | success, 298 |

| Arm | Complete success | Paired rescues / harms vs baseline | Repair actions | Transactions | Verified repairs |
|---|---|---|---|---|---|
| Baseline | 4/6 | — | 0 | 0 | 0 |
| Full joint | 2/6 | 0 / 2 | 640 | 20 | 0 |
| Measured + fusion | 4/6 | 0 / 0 | 176 | 6 | 1 |

Measured + fusion performs **no repairs** on roots1/2/3/5 and its full numerical
action tapes equal baseline on all four. It therefore avoids the two observed
joint harms, but provides no evidence of a task-level rescue. Its repairs occur
only on roots0/4, which still fail. Full joint adds 464 repair actions across the
six roots, with two fewer completions. These repair actions are included within
the shared per-episode 520-action cap, not added outside it.

### A verified exit that fails immediately after handoff

Measured + fusion root4 R1 enters at action148 after elevated grasp loss. It
passes the `contained` alternative after action163 (three consecutive endpoints;
fusion probability 0.994094, above 0.9934257), and hands off at action164.
After that **first nominal action**, `in_goal` becomes false, and repair R2
starts at165. The first repair used16actions; R2/R3 each used32.

At the verified exit, target/gripper contact remains true although the grasp
predicate is false. The target moves about11.29mm from the containment anchor
after the handoff action. These measurements document a fragile accepted state;
they do not establish which mechanical interaction caused the loss. The one
verified transaction is not a successful stable handoff or complete-task rescue.
Together with full joint root4's blocked local recoveries, this argues for
continuation-based verification data rather than a uniform threshold reduction.

### Final validation and provenance

- 18/18 exact replays pass: 8,008 frames, zero mismatches.
- 6/6 independent joint audits pass: 2,736 observations and 640 handoff scores.
- Both measured-fusion episodes with repairs pass independent fusion audit:
  176 scores, zero skipped scores or mismatches. Four other episodes have no
  handoff scores to audit.
- All 7,792 actor actions equal the recorded mapped VLA proposals; no numerical
  corrections or oracle options. This is total retained episode cost, including
  the eight reused episodes, not the incremental cost of this study.
- All 12 treatment/baseline pairs have identical selected initial-state and scene
  hashes. Exported action/observation indexing, query counts, source hashes and
  same-trigger retry-admission reasons pass consistency checks.
- GPU nominal-shadow inference matches all 1,090 shared-prefix observations
  across six roots: exact risk-score agreement, matching selected stages,
  high-risk streaks and no-lift outputs. Inference ran on the canonical GPU3 lane;
  the independent episode audits used each episode's recorded lane. CPU results
  remain separately retained with their numerical discrepancy record.

Evidence under `/home/ami/scratch/framework_diagnostic_20261022_r1/`:

- `analysis_final18/`: per-action JSONL/CSV, repair summaries, paired windows and
  aggregate `report.json` / `report.md`.
- `plots_final18/root{0..5}.png`: measured progress, repair occupancy and risk.
- `nominal_risk_gpu6/`: baseline observational inference and nominal24 labels.
- `final_validation.json`: validation counts and audit-file hashes.
- `controls/` and `measured/`: frozen protocols, launch logs, results, replay
  reports and newly collected raw episodes. Reused episode paths are in results.
- `analysis_sources/`: frozen copies of the analysis scripts used for the final
  exports. A compact copy of reports/protocols is retained beside this note in
  `framework_diagnostic_evidence_20261022/`.

## Evidence-led architectural changes to test next

These are proposed changes, not efficacy claims or already implemented repairs.
The verification–repair–reverification framework remains the research object.

### 1. Separate anomaly evidence from intervention admission

**Smallest first change:** retain joint feature computation, logging, stage
selection and handoff evaluation, but make joint-only risk/no-lift/predicted-exit
signals advisory at the admission boundary. Measured triggers still admit
repairs. This isolates the admission pathway from the bundled joint-off arm
completed here; it does not discard the joint modules.

Inputs are measured violations, anomaly evidence, current objective, progress
history and prior repair outcomes. The explicit output is `admit / observe` plus
a reason. On `observe`, the action request remains nominal; on `admit`, the
existing repair request is issued. Merely adding diagnostic fields to a record
would not change behavior. Admission benefit should subsequently be learned or
validated from matched nominal/repair continuations, rather than interpreting
a 24-action stagnation label as evidence that intervention improves completion.

### 2. Make repair continuation and retries depend on the recovery objective

Use the existing measured acceptance alternatives and preservation predicates
to track progress, stagnation and regression after each action. Persistent
absence of objective progress should end the transaction before the cap; it
should not automatically consume32actions and reopen under another trigger.
Key failed attempts by recovery objective and relevant context, including the
attempted behavior, rather than trigger name alone. Require a material change
in relevant evidence or an actually different proposal for retry. A stable
table-contact change alone is not demonstrated to make the same repair useful.

The actual action-generation consequence must be explicit: resume nominal when
an attempt is unproductive, or request a next admissible behavior when measured
progress invalidates the current one. Root4's `acquire` requests after held-clear
progress motivate checking this compatibility. A structured failure-conditioned
candidate generator would be a separate implementation and needs actual
candidate generation/evaluation; the current H1 wrapper and risk head do not
provide that capability. This study cannot establish that frozen VLA is unable
to execute a better recovery representation.

### 3. Verify preservation under nominal continuation

Train/evaluate handoff decisions on actual post-repair nominal continuations,
with preserved predicates, regression timing and complete-task outcome logged
separately. Root4 provides both rejected local-recovery states and an accepted
state that regresses one action later. A provisional handoff/reverification
window must count its actions in the same total budget and record failures;
calling a local predicate streak “recovered” is insufficient.

For the next incremental test, freeze the admission-only contrast first, keep
the model, instruction dictionary, initial states and caps matched, and report
complete success, rescues/harms, repair cost and post-handoff regression. Test
progress-aware continuation and handoff changes separately afterward. Exact
progress windows and thresholds must be declared before running that contrast;
this study does not select them retrospectively to maximize these six outcomes.

## Rollback and evidence-retention boundary

These runs use the rollback-era controller without the uncommitted release /
regrasp state machine. The old `/home/ami/scratch/multi_root_control_20261022/`
directory was previously deleted by the assistant, contrary to the evidence-
preservation requirement; its detailed logs and root4 infrastructure traceback
cannot be audited here. Its outcomes are not pooled into these 18 episodes.
The separately retained root0 state-machine directory remains a different
historical experiment. All current outputs, failed checks and preliminary
exports are retained. No commit or push was made for this study.
