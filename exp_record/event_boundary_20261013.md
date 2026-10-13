# Boundary reachability and frozen-gate diagnosis — registered 13 Oct 2026

## Purpose and hypotheses

The preceding coverage pilot failed: train/dev anticipatory original-proposal roots were 0/1. Strong drive pulses first violated within their five-step execution in 117/125 and 57/64 pre-violation cases. This experiment targets that failure rather than assuming another network will fix it.

- **H1, collector overshoot:** weaker/gradual or earlier pulses can reach pre-violation states where the unchanged policy proposes an H10-unsafe, R5-safe chunk. Falsifiable within the fixed tested schedules, not a universal reachability claim.
- **H2, gate decision rule:** frozen scores contain useful discrimination, but existing thresholds miss the shifted contact events. Test frozen thresholds against diagnostic safe-score thresholds on root-disjoint design halves.
- **H3, order representation:** holding context and the action multiset fixed, reversing candidate order can change the true H10 event while a static max-score remains invariant. This identifies a structural blind spot on the probed candidates; reversal is synthetic and cannot alone establish its prevalence under the actor or justify promoting the previously failed GRU.

## Fixed design and parallelism

Three onset shards share **the same 128 training roots**, reset seeds **74000–74003**, 32 roots each, and replay full trajectories from reset. These are the preceding pilot's **training** roots, reused for design only. No pilot-dev, validation-audit, confirmation or final-test episodes/labels are used. All branches of a root stay together. Replaying entire prefixes costs more than snapshot branching but directly preserves camera appearance, history and physical reachability.

Each shard uses first positive observed x-goal gap ≤ its fixed threshold: **early 0.20 m / middle 0.12 m / late 0.06 m**. Every collector applies at most one five-step pulse at an existing decision boundary; the unmodified policy resumes afterward. Seven arms per onset:

| Arm | Applied pulse, j=1…5 |
|---|---|
| none | Original proposal throughout; replicated across devices as replay control |
| hard_drive | Previous gain U[1.25,2], x bias U[.03,.12] m/s, y bias U[−.02,.02] m/s; onset now fixed |
| hard_brake | Previous per-root scale uniformly from [.05,.25,.55]; onset now fixed |
| gain_110 | Original proposal ×1.10 |
| ramp_030 | Original proposal + x bias .03×j/5 m/s |
| ramp_060 | Original proposal + x bias .06×j/5 m/s |
| brake_ramp | Original proposal ×(1−.75×j/5), ending at .25 |

Random constants reuse the preceding pilot's per-root draw, including its original draw ordering. All actions remain clipped at ±0.5 m/s. No future label, learned score or hidden parameter selects actions or onset. Physical ranges stay friction [.5,1.5], mass [.6,1.6]. Full OpenVLA-7B + frozen H10 DAgger×2 head, microbatch8, open-loop k5, 80 steps, 16 decisions for all arms. **3×7×4=84 batches / 2,688 executed trajectories**, but only **128 independent roots**. Duplicate none branches and training seeds do not multiply sample size.

Use three independently health-checked RTX A5000 devices (one process/shard):
- early: UUID `GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0`;
- middle: UUID `GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8`;
- late: UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`.

Each process loads its own full actor and all seven frozen risk models (original + static/sequence seeds 0/1/2). Model/calibration/source hashes are checked; frozen models are **observers**, never new online gates or repairs in these collector rollouts. Production runs use `setsid nohup ... < /dev/null`, offline loading and CPU thread caps. No model/data package upgrades are needed.

## Labels, checks and primary endpoints

At every actual decision save measured obs/obs_prev/a_prev, original proposal and the same eight scale labels, original state/physical multipliers for offline provenance, executed plan, pulse timing, prior-violation flag, all seven frozen per-step scores, live actions/risks and final outcomes. GT labels are offline only. A separate reversed-original probe is never executed, never offered to repair and never counted as original-proposal coverage. It adds one fixed diagnostic label query per state, not a deployment candidate.

Assert exact paired initialization/proposals/parameters, original none-source replay (rtol1e−5/atol1e−6), non-destructive label calls, actual previous-action history, 16-call policy/render cadence and executed-prefix label parity (rtol1e−5/atol1e−6). Stop on failed parity rather than loosening tolerances. Save atomic per-batch tensors, manifests and hashes. Recompute hashes, coverage and prior-violation masks during analysis.

**Primary reachability endpoints, root deduplicated:**
1. Fraction of pre-violation pulses whose entire applied five-step prefix remains safe.
2. Number of roots with **post-pulse, pre-first-violation** original proposals satisfying R5≤1 and H10>1; separately count the very next decision and any later decision. All-stage coverage is also reported, but cannot substitute for post-pulse evidence.
3. Each candidate onset/arm passes the **design feasibility screen** only with ≥**24 post-pulse anticipatory roots**, ≥**2 in each** fixed friction/mass stratum (splits 1.0/1.1), and ≥**90% pulse survival**. A pass is not efficacy or deployment acceptance. The old production-data criteria (train immediate/anticipation/safe-contact 32/24/64; dev16/12/32 plus strata) remain required in a subsequent fresh collector pilot.
4. Report final SR/CVR/safe success together with paired rescue/harm and paired-root 95% intervals against none. Collector outcomes are training/design diagnostics, not verifier efficacy results. Broadening exploration must not be presented as an improved runtime policy.

**Budget/stop:** exactly the registered 84 batches; no automatic expansion, threshold lowering, physical-range expansion, event relabeling, gate fitting or model promotion. If all screens fail, report that the tested schedules fail; investigate event support/other reachable families before more production data.

## Secondary frozen-gate analysis

Keep every static/sequence training seed; never select the best seed. Each shard's sorted root IDs split into **64 threshold-fit roots (74000–74001)** and **64 diagnostic-check roots (74002–74003)**. All paired arms of each root stay together. These are already-used training roots for experimental design, **not independent validation or a new calibration set for deployment**.

For each model, retain its frozen production tau. Additionally fit safe-score quantiles at nominal **1% and 5%** trigger rates from pooled pre-violation, H10-safe originals of the fit half, using the existing quantile rule. Apply each threshold unchanged to the check half for every arm. Compare all H10/R5 recalls, safe-trigger rates, precision and missed/false-trigger counts, plus anticipation recall against H10-safe negatives. Immediate H10 hazards must not be mislabeled as anticipation negatives. Report actual check-half trigger rates; nominal targets do not guarantee matched rates under shifted distributions. The same H10 score is used for all endpoints so a horizon-specific score change is not confounded with threshold selection.

Use root-cluster 95% intervals (1,000 bootstrap draws) preserving all correlated decisions. Flag endpoints with fewer than **12 positive roots or 32 safe roots** as insufficient for comparative gate conclusions; descriptive results may still be shown. Intervals are conditional on each design arm; no correction or confirmatory claim for selection across the 18 perturbed onset/arm combinations.

Order probe: report pre-violation H10 event-flip decisions/roots and event flips for which reversed/original max scores differ by ≤1e−5, separately for all seven models. Risk changes under synthetic reversal are not new original-proposal event coverage, learned safety gains, or continuation-value labels.

## Decision map: algorithm versus architecture

| Observed pattern | Supported next action | Not established by that pattern |
|---|---|---|
| Weaker/earlier/gradual collector passes reachability screen | Freeze observation-only collector, repeat on fresh train/design roots; then coverage-only regression controls with original repair fixed | That more data alone fixes the gate |
| Adequate event roots; design thresholds improve detection at acceptable measured false triggers | Prioritize a separately calibrated event gate and a threshold/rule ablation before increasing model capacity | Closed-loop completion gain or OOD calibration guarantees |
| Adequate roots; existing scores still poorly separate events | First compare regression versus event-aware loss on the same accepted data and matched static/causal architectures | That architecture or missing observations are already proven causes |
| Frequent event flips with invariant static scores | Test an order-sensitive event representation with matched controls; retain all seeds | That the existing GRU is good, or synthetic reversals are common actor errors |
| Event-aware training still fails after adequate coverage | Test longer measured history / dynamics-identification context with controlled input ablations; privileged parameters only as diagnostic upper-reference | Permission to provide hidden physics to learned runtime models |
| Detection improves but completion regresses | Separate acceptance/repair/continuation ablations with frozen gate | That detection recall alone validates the framework |

This experiment **targets the evidence needed to choose a change**. It does not precommit to a larger network, a new GRU, or a new repair search. Formal learned efficacy and transfer still require independent frozen-method SR/CVR/safe evaluation under the established protocol.

## Verification before production

Three boundary-collector tests cover exact ramps, bounded actions, unchanged original proposals, one-pulse caching/history/commitment and onset behavior. CPU BC smoke `event_boundary_smoke_20261013_a` completed seven collectors, seven frozen-model score/reversal probes and automatic analysis; its tiny-sample screen correctly failed. This smoke tests implementation, not OpenVLA efficacy. Production launch and later measured results are recorded chronologically in `week4.md`; this registered numerical protocol is kept fixed after launch.
