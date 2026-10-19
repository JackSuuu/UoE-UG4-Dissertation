# Research reset: one question, one active intervention — 13 Oct 2026

This decision supersedes the SEAL-recovery next-experiment sketch and the older architecture-first priority lists. Historical protocols/results remain immutable. The user's concern is justified: successive diagnoses became new projects without a sufficiently explicit link to the primary outcome. The next action is not another candidate-family or simulator expansion.

## The one research question

**With the same full OpenVLA policy, observations and action-execution opportunities, can a learned physics verifier improve safe task completion by intervening on the right proposals, without materially reducing task completion?**

The fixed box-push task is the mechanism testbed. One everyday simulated manipulation task is the subsequent transfer test, not a substitute for success on the testbed. Differentiable physics/local repair remains the contribution direction; current successful components do not demonstrate effective gradient-based repair.

## Why the earlier result sounded stronger

These are different experiments, not a time series of one model becoming worse:

1. **Early stand-in/BC-era results and oracle results were overgeneralized.** Week1's historical v5 table reports policy SR/CVR/safe82.3/27.2/68.5%, learned physics80.4/20.5/70.4%, and GT shadow80.5/1.5/80.1%. Learned safe improvement was about1.9pp; the much larger11.6pp belongs to privileged GT. The table's old “VLA alone” naming and “upper bound” language do not establish full-OpenVLA learned efficacy or a global bound. Early `vis` runs also skipped the7B language backbone.
2. **The old execution comparison was confounded.** The8Oct audit found that policy-alone cached k5 chunks while non-triggering verifier arms replanned each step. The earlier claimed all-arm fix was incomplete. Old headline numbers therefore cannot isolate verifier benefit at matched control opportunities. We have not measured how much of every historical gain this defect explains, and must not attribute all of it to the defect.
3. **Under corrected H10/k5, the small sample overstated the larger-cohort result.** The n8/cell OpenVLA validation suggested roughly+3.6–4.5pp safe. On896 OOD episodes, frozen decision-tau physics gives SR/CVR/safe82.92/28.35/63.50% vs policy84.71/31.92/62.28%: Δsafe+1.23[+0.11,+2.46]pp, rescue/harm21/10; ΔSR−1.79[−3.24,−0.33]pp. Legacy tau gives+1.56[+0.45,+2.68]pp safe, rescue/harm20/6. These small positive results remain valid; they are not erased by later results.
4. **On a different validation cohort (n448 OOD), benefit was not stable.** Policy87.05/29.24/65.63%; original verifier84.60/28.79/64.73%, Δsafe−0.89[−2.90,+0.89]pp, rescue/harm8/12. Frozen ramp84.82/26.79/66.74%, Δsafe vs policy+1.12[−0.67,+2.68]pp, rescue/harm10/5, ΔSR−2.23pp. Against the original verifier ramp improves+2.01[+0.89,+3.35]pp, rescue/harm9/0. A component improvement is not the same as a reliable advantage over the policy.

Thus the correction is **“small, cohort-dependent learned benefit; strong privileged diagnostic benefit”**, not “a previously established large learned gain disappeared.” Sources: week1 v5 historical table; week3 8Oct cadence audit; week4 12Oct confirmation; repair-transition completion report. Do not compare baselines across cohorts as if their difficulty were identical.

## What we have learned, and what remains unknown

- Replacing gate estimation/threshold semantics with a GT event gives the largest tested gain while keeping learned repair: OOD safe77.46% vs original64.73% and policy65.63%. This selects **intervention decision/gating** as the active bottleneck to investigate. Privileged physics and changed threshold semantics mean it does not prove that allowed observations suffice to learn the same gate. GT-gated SR80.80% also remains below policy87.05%.
- Repair can create later harm. The frozen ramp recovers7/12 harms and retains8/8 original rescues, but does not meet the policy-relative efficacy target. This is a completed bounded component result; pause and further ramp sweeps are not the next priority.
- Static action-order blindness is real on synthetic probes, but the tested GRUs failed actual detection. Neither a larger temporal model nor a new action family is justified as the next default.
- The new-head training data had20 unsafe original decisions and zero H10-only unsafe originals. Both subsequent collectors failed anticipation support. Consequently a full H10 event-gate training run is **blocked**, not “ready if we simply train longer.” Immediate R5 work cannot be labelled anticipatory H10 learning.

## One active change: intervention decision objective, with repair frozen

**Hypothesis:** the current scalar-regression-plus-safe-quantile gate fails to distinguish harmful original proposals reliably; an event-oriented decision objective may improve intervention selection without changing repair or adding actions. This hypothesis is unproven and may fail because of coverage or observability.

The next preregistration must specify a bounded, **R5-immediate-event** comparison, with the H10 anticipation deficiency explicit:

- Preserve full actor, H10 proposals/k5 execution, original frozen repair predictor/bisection, actual observation history, and hidden-parameter isolation. Keep original direct controller as the causal comparison; retain ramp as a completed reference, rather than combining new gate and new repair schedules in the first test.
- Compare the frozen gate, a same-formulation regression control on the new permitted training data, and an event-loss gate on the **same data and architecture/input budget**. Coverage-only versus objective effects must remain separable; report all predefined training seeds.
- Freeze data eligibility, root-grouped train/dev/calibration splits, gate integration and threshold rules before collecting/training. Existing validation/branch/audit labels remain diagnosis-only. Fresh data support must be checked against numerical R5 criteria specified beforehand; no claim that the previous failed full-H10 pilot already qualifies.
- If that bounded support check fails, stop this study as blocked. Do not repeat another broad collector sweep, silently relax the criterion, extend physical ranges, or launch SEAL/GRU work as a substitute. Report which observable-data assumption prevents the hypothesis being tested and revise the research scope explicitly.
- If support passes, run one fixed comparison through **closed-loop SR/CVR/safe success, paired rescue/harm and uncertainty**. Offline recall/MAE improvement alone does not pass. R5 success is a component result; H10 anticipation remains a separate unresolved requirement.
- If event loss adds no discrimination over matched regression, reject that tested objective; do not automatically widen the network. If detection improves but task outcome does not, the result establishes the need for a separately justified acceptance/repair study. No automatic branch expansion follows either outcome.

This document chooses the research question and next intervention, **not a launch-ready numerical protocol**. No training/collection job is launched by the reset. Writing the exact bounded protocol is the immediate deliverable; it must acknowledge the existing data-support blocker rather than treating it as solved.

## Outcome gate and parked work

Overall candidate acceptance remains≥+3pp OOD safe success vs policy, positive independent paired95% lower bound, no>2pp SR loss, stress-cell review and established training-seed/cohort robustness. Freeze before unused independent testing; already consumed confirmation seeds cannot become new confirmation. A validation increment or oracle result does not pass this gate.

**Parked:** SEAL/non-collinear candidate studies, further smoothing, larger/longer-history architectures, broad coverage sweeps and full LIBERO/RoboCasa verifier transfer. SEAL is an optional future repair mechanism if reliable detection still leaves completion failures; its availability is not a research objective. Benchmark feasibility may be revisited after the mainline decision, but is not the current active experiment. Hardware/latency work remains deferred.

Every proposed run must answer four questions in advance: **which observed failure; which single change; which policy-relative outcome; which stop/next decision**. If it cannot, it does not launch. Documentation below this decision is historical unless explicitly reinstated.
