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
