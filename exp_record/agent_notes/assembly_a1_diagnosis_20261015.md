# A1 post-hoc failure diagnosis and video review

## Scope and frozen originals

A1 completed30/30 trajectories with exact replay. Full endpoint: VLA3/10,
framework6/10, shared-trigger geometric ranking8/10. The four framework failures
are3000/c1,3000/c3,3001/c2,3001/c3. This is failure-selected development diagnosis,
not a new unbiased evaluation. No original controller, checkpoint, endpoint or
trace is modified. Original traces must match both status and replay SHA256.

All30 originals: `/home/ami/scratch/assembly_a1_20261015_a/live_s{seed}_c{case}_{arm}/annotated_2x.mp4`.
Review index: `/home/ami/scratch/assembly_a1_20261015_a/review/index.html`.
Four three-arm comparisons: `review/s{seed}_c{case}_comparison_2x.mp4`.
Columns VLA/framework/feedback; playback2x simulated time; shorter successful
clips hold an explicitly labeled final frame. Original video images are pre-action,
while their load/dwell annotation is post-action. Contact sheets sample0/9/18/59/159.
Desktop browser preview was unavailable in this session; paths remain directly usable.

## Registered diagnostic computations (before their outcomes)

1. Recompute one-step learned prediction error against the **actual executed**
   action's next state; force and tilt head errors separately. Do not compare the
   three-step pulse/settle forecast against later different online actor actions.
   Decompose candidate objective into lateral geometry, vertical geometry and risk;
   report actual vs predicted direction of lateral error change. Compute all30
   trajectories, not just failures, for context.
2. For each failed framework trace, select first repair, worst one-step positional
   error among repairs, last repair consuming budget40, and first subsequent
   forecast-preferred repair blocked by budget. Replay all8 candidate3-step
   pulse/settle plans from each selected exact historical prefix. These are
   **privileged offline diagnostic branches**, never online learned predictions.
   Every prefix action, full integration state and endpoint/load must be exact;
   no hidden controller snapshot approximations. Report true local objective and
   ranking regret; GT risk has no ensemble uncertainty, so report that comparison
   limitation. Neither local ranking nor local progress proves final-task rescue.
3. **Budget intervention, registered before running:** only for the same four
   failed episodes, double changed-action cap40→80; preserve horizon160, same actor,
   model weights, sensing, trigger, candidates, objective and endpoint. A separately
   generated/hashed function changes only the budget condition and its video label;
   the original source remains unchanged. Require exact images, nominal actions,
   selected actions and forecasts through original budget exhaustion. Independent
   process action/state/image/outcome replay follows each new rollout. Report all
   four outcomes, including new failures; no threshold tuning or test-set claim.
   This tests whether **more of the same correction** helps, not whether arbitrary
   better corrections could rescue the case.

Diagnosis output: `/home/ami/scratch/assembly_a1_diagnosis_20261015_a`.
Budget probe output: `/home/ami/scratch/assembly_a1_budget80_20261015_a`.

## Initial descriptive findings (before counterfactual outcomes)

*3000/c1*: full endpoint fails only detachment at termination; native success starts91,
seated starts92. All40 repairs preserve grasp; budget exhausted65;70 later preferred
repairs suppressed. Repair-only one-step xyz error mean2.35mm. This is not terminal
failure to insert. Candidate set contains no explicit open/retract repair (all
non-nominal candidates close the gripper); more closed-grasp pulses cannot be assumed
to solve final release.

*3000/c3*: loss of gripper contact first occurs8, before first repair9;33/40 repairs
occur when the previous measured endpoint is detached. Budget exhausted59. Repair
xyz error mean35.61mm; at18 predicted next tilt0.0044deg vs measured77.09deg, xyz
error58.01mm. Terminal tilt81.22deg. No grasp-conditioned gate or regrasp candidate
exists, and residual memory is diagnostic-only. Budget is spent after lost coupling.

*3001/c2*: budget exhausted72;48 later preferred repairs suppressed. Native success
starts91 but the frozen geometric endpoint never seats. Final minimum cross-section
clearance approximately−0.09565mm; detached and stable. Report as a near-boundary
geometric rejection, not robust jam/damage evidence. Do not relax the original
endpoint post-hoc. Repair xyz error mean3.23mm, much larger than this final margin.

*3001/c3*:32/40 corrections increase lateral error by>.1mm;25 of these predict an
improvement>.1mm. Lateral error at first repair≈33.1mm grows to≈55.2mm after last
repair59; terminal≈64.9mm.15 positive-y and22 positive-x corrections.17 later
preferred repairs suppressed. At13 the selected action's next load prediction1.63N
vs actual17.55N. At18, +y receives **worse predicted geometry** than −y but a lower
risk penalty, so risk outweighs alignment. This directly motivates true candidate
ranking checks rather than assuming the budget alone caused failure.

All four have120 nominal intervals with nonzero rotation commands; training had no
rotation commands. Even zero-rotation correction contexts inherit earlier actor
rotations/contact states. Out-of-distribution exposure is measured; causal model
improvement would require a separate data-coverage experiment, not an assertion.

## Completed counterfactual evidence

All16 selected events×8 candidates=128 three-step physical branches completed.
Every historical prefix was bit-exact against recorded integration state, observed
load and endpoint. All four budget80 continuations completed with exact original
prefixes through the40th correction (steps65/59/72/59) and independent full replay.
The cap and its displayed label were the only controller changes.

| Case | Original terminal failure | Budget80 full endpoint | Actual changes / steps |
|---|---|---|---|
|3000/c1|seated, gripper still touching|PASS|80 /148|
|3000/c3|detached, tilted/hung at peg|FAIL|46 /160|
|3001/c2|stable/released, clearance−0.09565mm|PASS|47 /101|
|3001/c3|large lateral displacement|FAIL|70 /160|

**Budget causality is case-dependent.** Additional permission for the same controller
rescues3000/c1 and3001/c2 under identical horizon/endpoint. This supports a causal
contribution of the40-change cap in those two deterministic continuations, not a
general recommendation for80 or a revised original headline score. In3000/c1 native
success occurs later (139 rather than91), yet complete release succeeds: earlier
native success was not better final completion. In3001/c2 only7 additional changes
are needed. The other two do not even consume all80, so their remaining failure is
not explained by that upper cap.3001/c3 ends about102.3mm laterally displaced versus
64.9mm originally after30 extra changes, with21 extra+y corrections.

**Lost coupling and unreliable predictions (3000/c3).** At sampled steps18/59/78,
all8 candidate outcomes on the nut over3 steps are exactly identical. The model
nevertheless predicts distinct scores/objectives; the apparent ranking has no real
short-horizon control effect. Preferred3-step positional errors are120.7/129.4/129.3mm.
At18, first-step tilt is predicted0.0044±0.0073deg (ensemble mean±std) while actual
is77.09deg. Existing residual memory does not veto forecasts, and the gate does not
require a valid grasp or represent regrasp. Additional budget gives46 changes and
still fails; native success at145 nevertheless leaves the nut tilted90deg. This
is direct evidence of a model/applicability failure, not just insufficient budget.

**Direction/risk-ranking error (3001/c3).** True three-step objective favors retract
at13 and−y at18/59/60; learned ranking chooses+y/+y/+y/+x respectively. At18 predicted
geometry already slightly favors−y, but the risk penalty for+y is0.416 vs0.515 for−y,
reversing the selection. The model and score favor reducing predicted short-term
risk over useful alignment. At all four sampled states the chosen candidate still
beats the nominal action on the true local objective, so the correct conclusion is
**suboptimal recovery ranking and inadequate long-horizon completion alignment**,
not that every correction is worse than doing the nominal action. Exact true-best
ranking at these samples does not establish whole-task recovery for that alternative.

**Imperfect ranking need not mean ineffective repair (3000/c1 and3001/c2).** All
four sampled choices in3000/c1 improve the true local objective relative to nominal,
although none is the true best. In3001/c2, learned+x is the true best at8/72/73;
at34 learned+y is inferior to−x but still beats nominal. Additional-budget full-task
rescues confirm useful recovery remained possible in these two episodes.

**Verification acceptance gap.** In the four original failures,153/160 selected
repairs still score above the trigger threshold;152/160 have all8 candidates above
that threshold. Source requires only a>.01 objective improvement over nominal,
not a below-threshold candidate or a prediction-validity condition. Tau is an
empirical trigger, not a certified safety bound; nevertheless "triggered verification"
must not be described as "candidate passed verification." The existing RefCheckVLA
check/score are integrated, but assembly acceptance is only relative ranking.

## Consequences for the next version (not executed here)

1. Address predictor support and validity: real-actor rotational/contact-loss
   contexts, grasp-state conditioning, and calibrated residual/uncertainty checks.
   Re-evaluate held-out prediction and candidate ranking before attributing benefits.
2. Add explicit repair applicability/acceptance within the five-layer framework:
   no unsupported claims of moving a detached nut; distinguish incomplete/rejected
   predictions from genuinely verified progress. A grasp recovery phase needs
   compatible candidates rather than repeated empty lateral moves.
3. Evaluate phase-aware completion/progress scoring separately from the risk head;
   include orientation/containment and release requirements. A decreasing predicted
   force alone is not evidence of useful repair.
4. Compare budget policy separately: cap40 vs cap80 is now a concrete diagnostic,
   but progress-conditioned allocation needs an independently frozen comparison
   including original successes/no-harm cases. Do not pool these selected rescues
   into A1's original6/10 or claim superiority to feedback8/10.

Artifacts: `mechanism_summary.json`, `failure_timelines.png`,16`branch_*.json`
files in the diagnosis directory;4 new annotated videos plus `matched_prefix.json`
and `replay.json` in the budget directory. Original A1 archive310files and4 comparison
videos were independently restored from GitHub/LFS (`fecf4ba`), proof
`backups/assembly_a1_results_remote_verified_20261015.json`.

Final diagnosis/budget archives were pushed as`9b29c78` and independently downloaded
from GitHub/LFS:55 diagnosis files and47 budget-probe files all passed byte counts
and SHA256. Proof:`backups/assembly_a1_diagnosis_remote_verified_20261015.json`.
All12 original A1 frozen source/protocol hashes remain unchanged. Both diagnostic
supervisors are finished; no further rollout or training cohort was launched.
