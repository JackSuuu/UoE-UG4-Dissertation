# Boundary/gate diagnosis completion — 13 Oct 2026

The [registered protocol](event_boundary_20261013.md) remains unchanged. All three `event_boundary_20261013_a_{early,middle,late}` shards completed **28/28 batches each**, automatic analyses are complete, and their launcher/Python processes exited. Total **84 batches / 2,688 trajectories / 128 shared training roots**. No gate was trained and no new job was launched in this review.

## Integrity and primary result

Rechecked **168 raw artifact hashes** and each analysis's source-summary hash. All **four none batches**, including states, contexts, proposals, scale labels, seven-model scores and reversal labels/scores, agree **bit-exactly across all three GPUs**. Each shard's analysis also verifies live-trace violation masks, executed-prefix labels and coverage. This completion review did not freshly relabel the simulator.

**Every perturbed onset/arm fails the registered reachability screen.** Post-pulse pre-violation anticipatory root counts:

| Collector | Early .20 m | Middle .12 m | Late .06 m |
|---|---:|---:|---:|
| hard_drive | 1 | 0 | 0 |
| hard_brake | 0 | 0 | 0 |
| gain_110 | 0 | 2 | 2 |
| ramp_030 | 0 | 1 | 2 |
| ramp_060 | 1 | 3 | 3 |
| brake_ramp | 0 | 0 | 0 |

Threshold was ≥24 roots per onset/arm, ≥2 per physics stratum and ≥90% pulse survival. Across all branches/onsets there are only **5 unique anticipatory roots**, not the sum of correlated table cells. **Zero** anticipatory originals occur at the very next decision after a pulse. None has zero anticipation and four immediate-hazard roots.

Weaker pulses substantially improve survival: `ramp_030` survives **100%** of pre-violation pulses at every onset; `gain_110` survives **100/99.21/100%**. But survival alone does not create anticipatory event coverage. The tested overshoot hypothesis is therefore only a partial explanation of the previous pilot failure; these schedules do not solve it, and the result does not prove no reachable anticipatory state exists.

## New targeted finding: hazards at policy resumption

Recorded original-proposal risk traces and pulse timing localize most early-brake hazards to the next policy decision:

| Onset / collector | Pre-violation R5-unsafe originals | At first decision after pulse | Of those, violation at first executed step |
|---|---:|---:|---:|
| early hard_brake | 118 | 114 | 69 |
| early brake_ramp | 127 | 125 | 125 |
| middle hard_brake | 35 | 30 | 19 |
| middle brake_ramp | 34 | 29 | 29 |

Each listed next-decision case belongs to a distinct root. After the single pulse, the controller executes the unmodified policy proposal. Early `brake_ramp` has **100% pulse survival**, yet the next proposal first violates on its first executed step in **125 roots**. This directly motivates a resumption-transition diagnosis. It does **not** yet distinguish abrupt commanded velocity, loss/re-establishment of contact, policy time/state mismatch, or another dynamical mechanism.

Collector outcomes (n128 paired roots, percent); these are exploration trajectories, **not learned-verifier efficacy**:

| Collector | SR / CVR / safe | Safe rescue/harm vs none | ΔSR [paired 95%, pp] | ΔCVR [paired 95%, pp] | Δsafe [paired 95%, pp] |
|---|---|---|---|---|---|
| none | 96.09 / 3.13 / 93.75 | — | — | — | — |
| early hard_brake | 94.53 / 94.53 / 4.69 | 0/114 | −1.56 [−6.25,+2.34] | +91.41 [86.72,96.09] | −89.06 [−94.53,−83.59] |
| early brake_ramp | 91.41 / 99.22 / 0.78 | 0/119 | −4.69 [−9.38,0] | +96.09 [92.19,99.22] | −92.97 [−96.88,−88.28] |

All 21 onset/arm SR/CVR/safe rates, paired counts and intervals are in the three run-local `analysis.json` files; bootstrap resamples paired roots, not trajectories as independent samples.

## Frozen-gate diagnosis

The 64-root diagnostic-check halves are reused training/design roots, not untouched evaluation. Immediate hazards provide adequate event counts in early hard_drive/hard_brake/brake_ramp and middle hard_brake/brake_ramp; anticipatory endpoints remain too sparse. Preserve all model seeds.

For early hard_brake, the check half contains 60 positive and 63 safe roots. At existing thresholds, original predictor recall is **90.0% [81.97,96.83]**, safe-trigger rate **0.44% [0,1.31]**, precision **98.18%**. Static s0 recall is **21.67% [11.67,32.27]**, safe-trigger rate **4.80% [1.85,8.09]**; sequence s0 recall **3.33% [0,8.62]**, safe-trigger rate **11.79% [7.87,16.37]**. Intervals are root-cluster bootstrap, conditional on this collector.

Across all three seeds on this same arm, static recall is **20–25%**, GRU **3.33–11.67%**. The registered nominal-5%-safe recalibration yields static **21.67–25%**, GRU **5–11.67%**, while actual safe-trigger rates are **7.42–11.35%** and **10.04–17.03%**, respectively. Thus the **tested recalibration does not rescue these heads**; this does not rule out every threshold/rule or identify architecture as the sole cause. Original predictor recall under that recalibration drops to 81.67%, so lower false-trigger rate alone is not an improvement.

On unperturbed none states, reversing action order flips H10 labels in **272 pre-violation decisions across 126 roots**. Original predictor and all three static heads retain invariant max scores (≤1e−5 difference) for all 272 flips; none of the three GRU heads is invariant on these flips. This confirms order blindness for the static scoring families on the synthetic probe. Yet GRU remains a poor detector on the real resumption events: **order sensitivity is not sufficient for efficacy**. Synthetic reversed candidates do not establish the frequency of the issue among actual actor proposals.

## Decision after this completed budget

- Do not repeat a broader onset/amplitude sweep or silently weaken anticipation criteria. No collector qualifies for production event-gate training.
- Prioritize **paired resumption-transition diagnosis** on training-only roots: localize force/contact state and action discontinuity at pulse end, then compare original resumption with a fixed-budget smooth bridge and a pause/re-contact control, preserving actual history and k5. This is a next experiment to register, not an already implemented mechanism or demonstrated repair.
- Separate **immediate-event gate data/objective controls** from the unresolved anticipation-support question. Strong original-predictor recall in these immediate events is a useful baseline, not a transfer claim. Any R5-only learning study needs its own stated scope and cannot be relabeled as a successful H10 gate.
- Subsequent architecture changes must test action-order representation and measured history under matched data/loss/budget. These results support neither promoting the old GRU nor claiming threshold tuning alone fixes the framework.

No production training, independent confirmation, final test or simulator-transfer experiment was launched during this completion check.
