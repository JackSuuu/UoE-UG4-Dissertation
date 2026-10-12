# Temporal-risk validation: negative result and controller diagnosis (12 Oct 2026)

## Completion update: closed-loop factorial diagnosis

`risk_controller_diagnosis_20261012_a` has completed **20 cells × 11 arms × 32 episodes**; `summary.json` is complete and launcher/Python PIDs 591432/591435 have exited. Rechecked all **220 raw NPZ hashes, saved episode vectors, violations against recorded risk traces and H10/k5 cadence counts**, plus none and all three L/L arms against the original experiment. All passed. The launch/pending statements below are historical.

Pooled OOD n=448 per arm. Percentages below; rescue/harm and safe deltas are relative to policy-alone. L = learned; G = exact simulator. The G gate explicitly replaces both the risk estimator and its learned-quantile decision rule with the exact physical H10 event `risk > 1`; these results do not isolate model error from threshold semantics.

| Arm | SR | CVR | Safe success | Safe rescue / harm | Δ safe [paired 95% CI], pp |
|---|---:|---:|---:|---:|---:|
| None | 87.05 | 29.24 | 65.63 | — | — |
| G/G | 82.59 | 12.05 | 76.12 | 47 / 0 | +10.49 [+8.04, +12.95] |
| Frozen L/L | 84.60 | 28.79 | 64.73 | 8 / 12 | −0.89 [−2.90, +0.89] |
| Frozen G/L | 80.80 | 8.04 | 77.46 | 53 / 0 | +11.83 [+9.38, +14.29] |
| Frozen L/G | 86.61 | 27.23 | 66.52 | 5 / 1 | +0.89 [+0.00, +2.01] |
| Static s0 L/L | 82.37 | 39.96 | 53.57 | 2 / 56 | −12.05 [−14.73, −9.37] |
| Static s0 G/L | 83.71 | 8.04 | 78.79 | 59 / 0 | +13.17 [+10.49, +15.62] |
| Static s0 L/G | 83.48 | 42.63 | 51.79 | 2 / 64 | −13.84 [−16.52, −11.38] |
| Sequence s0 L/L | 83.71 | 54.24 | 41.29 | 6 / 115 | −24.33 [−28.12, −20.76] |
| Sequence s0 G/L | 81.25 | 8.48 | 76.79 | 50 / 0 | +11.16 [+8.93, +13.62] |
| Sequence s0 L/G | 85.49 | 47.32 | 47.99 | 6 / 85 | −17.63 [−20.98, −14.29] |

Direct safe-success differences versus the corresponding L/L:

- Frozen: G/L **+12.72 pp [+9.82,+15.40]**, rescue/harm **59/2**; L/G **+1.79 pp [approximately 0,+3.57]**, **13/5**.
- Static s0: G/L **+25.22 pp [+21.43,+29.02]**, **115/2**; L/G **−1.79 pp [−4.91,+1.12]**, **21/29**.
- Sequence s0: G/L **+35.49 pp [+31.25,+39.73]**, **162/3**; L/G **+6.70 pp [+3.79,+9.60]**, **39/9**.

Intervals remain conditional on this fixed parameter grid, using paired episode bootstrap; this diagnostic ran only training seed 0 for the new architectures. Replacing the gate is the largest tested lever for all three heads. Repair-score correction also helps the GRU, so repair is not irrelevant. Exact repair is not uniformly superior with an unchanged search/acceptance rule, and G/G is not a performance upper bound. Privileged GT uses hidden physical parameters and full simulator state, so it does not prove an equally effective learned gate can be inferred from the allowed one-step observation context.

Completion remains a separate bottleneck: G/L SR is **3.35–6.25 pp below policy-alone**; for sequence G/L, ΔSR is **−5.80 pp [−8.48,−3.35]** and ΔCVR **−20.76 pp [−23.21,−18.30]**. Zero observed safe harms against none on this cohort is not a safety guarantee. These privileged diagnostic gains do not meet the learned-verifier transfer gate or justify hardware transfer.

### Next priority: learn the event gate while freezing repair

1. First perform a fresh **training-only contact/event coverage pilot**, retaining the current physical training ranges. Sample reachable near-contact and post-repair states with explicitly recorded collector interventions, original proposals and the same eight scales. Keep entire episodes/branches in one split and normalizers train-only. Measure independent episode counts for safe-contact, R5-unsafe and H10-only-unsafe events; correlated windows must not substitute for event diversity. Validation audit labels remain diagnosis-only. Choose numerical coverage criteria prospectively in the pilot protocol before production training; none are claimed already satisfied.
2. Make the runtime interface **learned event gate → frozen original repair predictor/rule**. Supervision distinguishes `y5 = max(risk[:5]) > 1` and `y10 = max(risk[:10]) > 1`, retaining anticipation. Hidden simulator parameters are offline labels only. Separate the gate's decision objective from the repair score's magnitude/ranking objective.
3. On the same accepted training data, first retrain matched regression-gate controls to isolate coverage effects, then compare an event-aware gate objective with unchanged architectures, information, parameter/budget controls and repair. Static and causal controls remain necessary. Longer history, physics-parameter inference and directional candidates remain separate ablations.
4. Use disjoint development/calibration episodes to assess event recall at matched safe-trigger rates, precision, calibration and robustness across contact stages. Register threshold-selection rules before inspecting the next closed-loop validation. If boundary discrimination remains weak, test whether the allowed observation context is sufficient rather than endlessly tuning a single threshold.
5. Evaluate closed-loop SR/CVR/safe success and paired rescue/harm. Gate improvements that sacrifice completion still require a separate acceptance/continuation study; current H10 labels are not policy-conditioned continuation targets. Independent confirmation comes only after a validation-selected method is frozen.

No subsequent training job has been launched by this completion review. The immediate work item is the coverage-pilot and gate-ablation protocol, not a wider GRU or a larger candidate sweep.

## Decision

Do not promote either newly trained head. All three GRUs are worse than their matched static controls in closed-loop safe success, although their development regression and same-state candidate ranking are better. Stop architecture expansion; isolate event detection/gating from repair and continuation first. This rejects the tested model/data/loss/calibration/controller combination, not every temporal architecture or every verifier approach.

## Completed experiment and integrity

`src/results/push_torch/sequence_risk_20261012_a/`: collection, training, offline validation and all 20 cells × 8 arms × 32 episodes completed. This is validation, not confirmation or final test. Full OpenVLA-7B, the frozen H10 DAgger×2 velocity head, inference microbatch 8 and k=5 were preserved. The three training seeds share the same evaluation episodes; they are not three independent episode cohorts.

Post-run `experiments/analyze_sequence_risk.py` checks passed: source/model/data hashes, disjoint episode splits, exact train-only normalization statistics, baseline/frozen episode parity with the earlier candidate validation, all 160 arm/cell cadence counts, and outcome/aggregate consistency. Existing checkpoints and source-hashed experiment code were retained.

Sources: `evaluation.json` (paired outcomes/CIs), `training.json` (models/calibration/development), `offline_validation.json` (same-state comparisons), and the new `src/results/push_torch/sequence_diagnosis_20261012_b/analysis.json` (composition, integrity and local substitutions). Raw result files remain local; this record preserves the main findings in Git.

## Closed-loop outcomes

Pooled OOD: 14 fixed parameter cells, **448 episodes per arm**. SR/CVR/safe values are percentages; deltas and intervals are percentage points. Safe rescue/harm counts are paired against policy-alone. Intervals are 4,000-resample, cell-stratified paired bootstrap intervals conditional on this fixed grid, not new-task or training-seed uncertainty.

| Arm | SR | CVR | Safe success | Safe rescue / harm | Δ safe vs none [95% CI] |
|---|---:|---:|---:|---:|---:|
| None | 87.05 | 29.24 | 65.63 | — | — |
| Frozen anchor | 84.60 | 28.79 | 64.73 | 8 / 12 | −0.89 [−2.90, +0.89] |
| Static s0 | 82.37 | 39.96 | 53.57 | 2 / 56 | −12.05 [−14.73, −9.37] |
| Static s1 | 84.60 | 43.08 | 51.34 | 2 / 66 | −14.29 [−17.19, −11.38] |
| Static s2 | 81.92 | 38.17 | 53.57 | 1 / 55 | −12.05 [−14.96, −9.38] |
| Sequence s0 | 83.71 | 54.24 | 41.29 | 6 / 115 | −24.33 [−28.12, −20.76] |
| Sequence s1 | 80.80 | 57.37 | 36.83 | 7 / 136 | −28.79 [−32.37, −25.22] |
| Sequence s2 | 83.93 | 47.10 | 47.32 | 9 / 91 | −18.30 [−21.88, −14.73] |

The anchor interval differs slightly from the previous record because this runner uses a different bootstrap RNG seed; the episodes, point estimates and rescue/harm counts reproduce exactly.

Direct matched architecture contrast (sequence minus static):

| Training seed | Δ SR [95% CI] | Δ CVR [95% CI] | Δ safe [95% CI] | Safe rescue / harm vs static |
|---|---:|---:|---:|---:|
| 0 | +1.34 [−1.56, +4.46] | +14.29 [+9.82, +18.75] | −12.28 [−16.52, −8.04] | 26 / 81 |
| 1 | −3.79 [−6.70, −0.67] | +14.29 [+10.04, +18.75] | −14.51 [−18.75, −10.27] | 24 / 89 |
| 2 | +2.01 [−0.67, +4.69] | +8.93 [+4.69, +13.17] | −6.25 [−10.27, −2.23] | 30 / 58 |

Regression also occurs on the six ID grid cells (192 episodes/arm): none safe success **92.71%**, frozen **93.23%**, static **78.65–85.42%**, sequence **66.15–72.40%**. These ranges describe training-seed point estimates, not confidence intervals. OOD extrapolation alone cannot explain the whole failure.

## Representation improves some proxies, not the decision outcome

- Development per-step MAE: static **0.02100–0.02146**, sequence **0.01815–0.01836**. This averages both channels, including push's identically zero deformation channel, and many safe/no-contact steps.
- On identical policy-alone OOD states before the first violation, there are 5,572 decisions: **131 unsafe R5 originals** and **248 unsafe H10 originals**. Frozen trigger recall is **37.40% / 26.21%** for R5/H10; static is **4.58–5.34% / 2.42–3.23%**; sequence is **15.27–17.56% / 9.68–11.69%**. Thus sequence improves over static but both new heads miss more actual hazards than the anchor.
- H10-safe proposal trigger rate: frozen **4.88%**, static **7.72–8.23%**, sequence **6.99–7.89%**. A finite-horizon-safe trigger is not automatically causally harmful; anticipation beyond H10 can matter.
- To avoid rewarding prediction ties, the new diagnostic uses a **fixed GT denominator** for candidate ordering. There are 90,744 GT-distinguishable H10 pairs, including **3,050 safety-boundary-crossing pairs**. On the latter, frozen has **1,424 inversions / 41 prediction ties**; static seeds have **906/19, 735/35, 702/24**; sequence seeds have **393/0, 443/0, 370/0**. Improved within-state ordering does not establish reliable across-state event detection or useful intervention timing.

Calibration thresholds: frozen **0.555492**; static **0.291903 / 0.287303 / 0.284446**; sequence **0.396668 / 0.390008 / 0.398618**. Raw thresholds have different score scales and should not be compared as intervention aggressiveness by themselves. On the same OOD states, unsafe H10 original scores have medians **0.120–0.135** for static and **0.249–0.264** for sequence, below their respective thresholds. Independent safe-score quantile calibration controls neither OOD recall nor intervention benefit.

## Training coverage bottleneck

| Split | Episodes | Decisions | Unsafe H10 originals | Unsafe R5 originals | Unsafe candidates / all candidates |
|---|---:|---:|---:|---:|---:|
| Train | 384 | 6,144 | 20 | 20 | 152 / 49,152 |
| Development | 128 | 2,048 | 3 | 2 | 19 / 16,384 |
| Calibration | 128 | 2,048 | 8 | 7 | 64 / 16,384 |

Only 50 training decisions from 34 episodes contain any unsafe candidate. The 335 hot decisions (>0.8 for at least one candidate) span 198 episodes, but oversampling them cannot create missing event diversity. Training originals contain **zero H10-unsafe/R5-safe anticipation examples**; this statement is about original proposals, not all scaled candidates. The frozen-verifier collector contributes only four unsafe originals in 3,072 training decisions. Development recall 100% rests on just two R5 / three H10 unsafe originals and is not persuasive evidence of reliable detection.

These observations make coverage and regression-objective mismatch plausible causes. They do not, alone, establish that simply adding episodes or reweighting MSE will solve the task. New-state distributions induced by the learned repair also remain unlabelled in this first dataset.

## Completed same-state gate × repair substitutions

`sequence_diagnosis_20261012_b` examines all 14 OOD cells on the **same policy-alone pre-first-violation trajectory**, for all six new heads and the anchor. L = learned component; G = privileged exact simulator component. The G gate uses **true H10 risk > 1**; the G repair substitutes exact risk into the existing six-step bisection, target `min(0.8, 0.75*s0)`, floor 0.05 and responsiveness guard. It is not the legacy four-scale `gt_shadow` controller. H10 labels describe open-loop continuation; R5 is the executed-prefix scope.

| Head | L gate / L repair: H10 local rescues | G gate / L repair | L gate / G repair |
|---|---:|---:|---:|
| Frozen | 16 | 106 | 16 |
| Static s0 / s1 / s2 | 4 / 6 / 5 | 109 / 109 / 107 | 4 / 6 / 5 |
| Sequence s0 / s1 / s2 | 17 / 17 / 19 | 107 / 107 / 107 | 18 / 18 / 20 |

G/G rescues **110/248 H10-unsafe originals**, and **16/131 R5-unsafe originals**, under this specific search/acceptance rule. These counts are not an upper bound over all actions, policies or earlier interventions. The local contrasts prioritize gating/detection for the next closed-loop diagnosis, but do not establish closed-loop gains.

On these saved baseline states, **none of the six new L/L repairs creates an R5 safety-label violation from an R5-safe original**, despite the large closed-loop harms above. Their own intervened trajectories differ, so this does not locate every new harm beyond five steps; it does show that this local check is insufficient to explain or predict episode-level harm.

The algorithm also permits execution without reaching its own target: changed L/L repairs above the relative target number **35/30/32** for static and **51/69/73** for sequence. Responsiveness is not an acceptance certificate. Altering this guard remains a separate ablation: earlier candidate-acceptance experiments did not establish net benefit.

Numerical provenance: first CPU attempt `sequence_diagnosis_20261012_a` correctly stopped at the strict simulator-label parity assertion (maximum observed difference ≈0.01378 in a ten-step risk trace). The successful `_b` run used the same healthy CUDA GPU as the source audit and passed original-label parity in every examined cell. No tolerance was relaxed and the failed attempt is not pooled into results.

## Next experiment: matched closed-loop factorial diagnosis

Implemented `experiments/diagnose_risk_controller.py` and `scripts/diagnose_risk_controller.sh`. For the anchor, static s0 and sequence s0, run **L/L, G/L, L/G**, plus shared **G/G and policy-alone**: 11 arms × 20 cells × 32 episodes. Seed 0 is a diagnostic convention, not a best-seed choice; any architecture-level follow-up must report the other training seeds too. This run diagnoses components rather than proposing a deployable GT-assisted method.

- Reuse validation episode seeds 26000–26019. Confirmation/final-test stay unused for selection.
- Retain full OpenVLA, H10/k5, microbatch 8, gradient/hard-prefix off and the same bisection rule and scoring-call structure. GT compute cost and privileged information are explicit; no latency-matched or information-matched learned-method claim is made.
- Policy-alone and all L/L arms must reproduce saved outcomes; record per-episode outcomes, per-step risk/score/trigger traces, source hashes, SR/CVR/safe success, paired rescue/harm and fixed-grid CIs versus none and each corresponding L/L arm.
- G/L improvement implicates detection/gating plus its trajectory consequences; L/G improvement implicates repair scoring/selection plus its trajectory consequences. If G/G remains weak, investigate action-family limitations, the search/acceptance rule and horizon/continuation mismatch. A poor G/G is not proof that every verifier approach is impossible.
- Before another training run, define adequate independent episode/event coverage in both train and development; separately test event-aware loss or stratified collection while retaining a matched static control. Do not combine new data, loss, gate and candidate freedom in one claimed architectural improvement.

Verification: **42 tests passed** (37 existing + 5 new); CPU BC end-to-end smoke completed all 11 arms. Smoke outcomes are not research efficacy evidence. Production launched as `risk_controller_diagnosis_20261012_a`, detached launcher PID **591432**, healthy GPU UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`, using `setsid nohup` and stdin `/dev/null`. Log: `~/scratch/risk_controller_diagnosis_20261012_a.log`. Formal closed-loop factorial outcomes are pending at this entry.

Launch verification reached cell 0: policy-alone and frozen L/L completed and passed source episode replay checks; G/G also completed. The full grid is still running, so no pooled conclusion is drawn from this first cell.

LIBERO single-task baseline feasibility can still proceed separately. Neither new head meets the previously registered verifier-transfer criterion; hardware remains deferred.
