# R5 objective comparison complete: relative loss signal, failed task-level efficacy

## Decision

`immediate_gate_20261013_a` completed the registered24collection batches/768trajectories, all3paired training seeds (3000updates each), and180evaluation batches/5760trajectories. All workers exited successfully. **Do not promote any new gate or proceed to independent confirmation.** Event loss improves the point estimate over matched regression in all3seeds, but only one seed has a positive paired95% lower bound. All event seeds substantially reduce safe completion versus both the policy and frozen original verifier, including on ID cells.

The numerical [preregistration](immediate_gate_20261013.md) remains unchanged. This report supersedes launch/progress entries, not that protocol. The tested objective change is insufficient as a solution; it does not establish that event objectives are universally ineffective or identify one unique cause of failure. No new collector, loss/threshold sweep, architecture or repair study is launched.

## Closed-loop OOD results

14fixed OOD cells,448paired episodes per arm. Percentages; rescue/harm refers to safe-success outcomes versus policy. Intervals are paired4000-draw within-cell bootstrap95% intervals, conditional on this reused validation grid, not multiplicity-corrected or independent-test evidence.

| Arm | SR | CVR | Safe success | Δsafe vs policy [95% CI], pp | Safe rescue/harm |
|---|---:|---:|---:|---:|---:|
| Policy alone |87.05|29.24|65.63|—|—|
| Frozen H10 gate |84.60|28.79|64.73|−0.89 [−2.90,+0.89]|8/12|
| Frozen R5 gate, new calibration |85.49|31.70|62.72|−2.90 [−4.69,−1.34]|2/15|
| Regression s0 |85.49|40.63|53.57|−12.05 [−15.85,−8.26]|15/69|
| Event s0 |86.38|39.51|54.91|−10.71 [−13.62,−7.81]|3/51|
| Regression s1 |86.38|44.64|50.45|−15.18 [−18.97,−11.16]|15/83|
| Event s1 |85.71|40.85|53.79|−11.83 [−14.96,−8.93]|1/54|
| Regression s2 |86.61|41.29|53.57|−12.05 [−15.63,−8.48]|14/68|
| Event s2 |86.38|36.61|57.81|−7.81 [−10.27,−5.36]|2/37|

| Event seed vs policy | ΔSR [95% CI], pp | ΔCVR [95% CI], pp |
|---|---:|---:|
|0|−0.67 [−2.90,+1.56]|+10.27 [+7.59,+13.17]|
|1|−1.34 [−3.79,+0.89]|+11.61 [+8.71,+14.51]|
|2|−0.67 [−2.68,+1.34]|+7.37 [+4.91,+9.82]|

All3event seeds also lose safe success against frozen H10:−9.82[−13.39,−6.47],−10.94[−14.29,−7.59],−6.92[−10.04,−4.02]pp respectively, with rescue/harm12/56,10/59,12/43. Original none/frozen outcomes and live risks replayed successfully, so a changed policy baseline does not explain this result.

## The matched objective hypothesis

| Event vs same-seed regression | ΔSR [95% CI], pp | ΔCVR [95% CI], pp | Δsafe [95% CI], pp | Safe rescue/harm |
|---|---:|---:|---:|---:|
|s0|+0.89 [−1.56,+3.57]|−1.12 [−4.46,+2.23]|+1.34 [−2.01,+4.91]|35/29|
|s1|−0.67 [−3.57,+2.23]|−3.79 [−7.37,0.00]|+3.35 [−0.45,+7.14]|45/30|
|s2|−0.22 [−3.13,+2.68]|−4.69 [−8.04,−1.34]|+4.24 [+0.67,+7.81]|44/25|

This is a consistent point-direction signal, **not** the registered all-seed positive-CI result and not an improvement over the policy. Both new arms are much worse than the frozen gate. The matched event/regression contrast isolates their objective under this common data/sampler/architecture recipe; new-vs-frozen also changes training distribution, normalization, sampling, auxiliary objectives and horizon/calibration, so it cannot attribute the larger regression to one of those factors alone.

## ID and stress checks

ID n192/arm, SR/CVR/safe(%):

| Arm | SR | CVR | Safe |
|---|---:|---:|---:|
|None|94.79|2.60|92.71|
|Frozen H10|95.31|2.08|93.23|
|Frozen R5|95.31|2.08|93.23|
|Regression s0|94.79|18.75|78.13|
|Event s0|94.27|16.15|79.17|
|Regression s1|93.23|20.31|75.52|
|Event s1|94.79|18.23|77.08|
|Regression s2|93.75|19.79|76.04|
|Event s2|95.83|14.06|82.29|

Hence this is not exclusively failure outside the physical training range. All event seeds flag13/14 OOD cells under the registered safe-loss or>2pp SR-loss stress rule. At friction1.8/mass1.5, safe delta vs policy is−31.25/−31.25/−28.13pp; at friction1.4/mass.5 it is−21.88/−25.00/−18.75pp. Individual cells contain32episodes and these are descriptive stress signals; pooled conclusions do not rely on selecting them.

## Support passed; its coverage is narrow

All predeclared R5 root/bin minima passed. Eligible train/dev/calibration rows4605/2328/2273; distinct roots128/64/64. Immediate-event roots120/62/61; safe-contact roots125/64/62. Per-bin immediate roots train[23,33,29,35],dev[15,18,12,17],calibration[20,14,15,12]. This does not retroactively pass the failed H10 anticipation studies.

Training has128original-proposal immediate-event decisions: **120brake-resumption,3policy-alone,5frozen-controller**. Thus93.75%of positive decisions come from one handcrafted collector. Development has62positive decisions, all from brake-resumption; policy-alone and frozen-controller dev branches have no immediate positives. Training has just2H10-only rows on1root shared across collectors, dev/calibration zero.

The event gate achieves62/62dev recall in every seed versus56/62,55/62,57/62for regression (90.32/88.71/91.94%). But **both frozen H10 and frozen R5 already detect62/62**. Those positive labels therefore do not demonstrate coverage of the original gate's missed hazards. Development false-trigger rates over2266R5-safe decisions are event5.56/5.83/4.72%, regression6.84/6.66/6.66%, frozen H10 3.35%, frozen R5 4.63%. Recall1.0 and a degenerate empirical bootstrap interval do not establish perfect population recall.

Calibration uses992R5-safe policy-alone decisions across62roots. Event taus are6.375e−10/1.077e−10/8.332e−10; regression .42037/.40808/.40081; frozen R5 .51649 and legacy frozen H10 .55549. Very small event thresholds reflect the calibrated upper tail of its score distribution, not calibrated physical event probabilities or evidence by themselves of a numerical bug. No post-hoc threshold changes are made. This study cannot isolate score calibration from representation/data effects as the sole cause.

**Interpretation:** the screen established enough examples of an immediate brake-resumption event, not representative coverage of policy-generated hazards or old-gate misses. This limits the offline-to-closed-loop inference. It is a documented support-design limitation, not permission to silently replace the criteria and rerun a collector sweep.

## What the saved trajectories establish about harm

Among event-vs-policy safe harms51/54/37, respectively47/52/35include a new violation, and6/4/4include task-success loss (categories overlap). All those new-violation harms have a gate trigger before or at first violation; no safe-harm episode has zero triggers. Thus most loss is not simply stopping safely before completion. The altered gate/unchanged repair controller creates many more unsafe outcomes in paired trajectories. This does **not** identify each trigger as a false positive or establish which repair/release step caused the later failure; unmodified-proposal counterfactual labels for every new trajectory were not recorded.

OOD triggered-decision rates are frozen H10 9.67%, event10.16/11.33/10.02%; event triggered-episode rates61.61/62.72/59.60%vs frozen66.96%. Consequently a simple statement that the new gate “always intervenes much more” is unsupported. Which states receive interventions and their subsequent trajectories matter. No new branch rollout is necessary to establish the failed efficacy result.

## Integrity and disposition

Independent completion verification checked all48collection artifact hashes, all180evaluation NPZ hashes, collection live-prefix/previous-violation parity, disjoint roots,3training summaries/checkpoints/paired normalizers, source/protocol/code hashes, per-episode risk/first-violation/trigger reconstruction,16-call/512-row cadence and one gate/eight repair queries per applicable row.40none/frozen batches replay earlier outcomes and live risk. All9workers exited normally. Recorded in ignored `completion_verification.json`.

Aggregate SHA256: `20b19983a239ed8795d39f4c6aa019752158b2a98198ef8cd165deebb51dadcd`.

Close this fixed R5 objective configuration as **failed policy-relative efficacy, partial relative-objective signal**. Keep the original verifier/direct and completed ramp as references; no new gate is accepted, no independent-test seeds are used, and no precision-task transfer claim follows. The overarching goal is unchanged: safe precision completion. Any subsequent proposal needs a specific evidentiary reason beyond “dev recall improved”; this result does not automatically authorize another network, collector, loss or smoothing branch.
