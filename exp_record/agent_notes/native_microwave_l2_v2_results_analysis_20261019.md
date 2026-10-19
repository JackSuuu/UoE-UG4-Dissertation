# Native microwave L2 v2: results and next-mechanism analysis (2026-10-19)

**Stop v2: 18/18 valid cells, 576/576 requested calls, and the frozen L2B gate is false. Root 4 C2 preserves an elevated grasp through H32 but does not place the mug; its motion is a contact-constrained plateau, not demonstrated delayed placement. Recommend one new, bounded, separately preregistered v3 horizon test, H32→H96, with everything else scientifically fixed and fresh H32 prefixes required to match v2 exactly. This note is analysis/recommendation, not a new protocol or launch authorization.**

## Evidence, indexing and limits

- Execution: `/home/ami/scratch/native_microwave_l2a_20261019_v2/`; completed audit: sibling `_v2_audit/{summary.json,results.md}`. The audit reports all six fresh greedy controls exact, all 18 identities valid, global integrity passed, and no sampled failure-root gain.
- Reproducer and detailed measurements: `/home/ami/scratch/native_microwave_l2_v2_analysis_20261019/{analyze.py,evidence.json,endpoint_trace.csv}`. The extractor reads saved JSON only, hashes its 46 input files, independently recomputes all **36** H12/H32 labels against the audit, and emits **594** endpoint records (18 × snapshot plus 32 actions). No model queries, simulator execution, GPU use, or new outcomes were produced by this analysis.
- Local action counts **k=1..32** are one-based; **k=0** is the snapshot. Absolute post-action indices are zero-based. Root 4 snapshot is **215**, so local k maps to post-step **215+k**. Root 0 snapshot is 193. Position measurements below are world-coordinate millimetres; action coordinates are controller commands, not millimetres.
- Contacts and grasp are logged **post-action endpoints**. “Throughout” below means every observed endpoint, including the snapshot where stated; it does not rule out unobserved substep losses/contact changes. Contact entries include positive-margin proximity as well as negative signed distances. They do not by themselves measure collision severity, force, or prove the cause of a stall.
- Six reused development roots, including two original failures (0/4) and four original successes (1/2/3/5), are not 18 independent tasks or held-out validation. Privileged predicates/contact measurements and offline oracle comparisons are not actor or selector inputs.

## 1. Complete local result

Each cell executes exactly 32 actions. There are **zero native completions in all 18 branches**, all U12 are false, and no snapshot-true goal predicate is lost. Root 5 is the only root with a snapshot-true goal (`In`); it preserves that goal in every branch.

Triples are C0/C1/C2; b is the frozen snapshot reference level.

| Root | Original L1 result | b | p12 | p32 | U32 |
|---|---|---:|---|---|---|
| 0 | failure | 0 | 0/0/0 | 0/0/0 | F/F/F |
| 1 | success | 0 | 0/0/0 | 0/0/0 | F/F/F |
| 2 | success | 0 | 0/0/0 | 0/0/0 | F/F/F |
| 3 | success | 0 | 0/0/0 | 0/0/2 | F/F/T |
| 4 | failure | 1 | 1/1/1 | 0/0/1 | F/F/F |
| 5 | success | 2 | 2/2/2 | 2/2/2 | F/F/F |

Thus **0/4 sampled failure-root candidates** have U32 gains. Root 3 C2 is the only sampled useful gain over matched C0, on an originally successful control. Root 4 C2's p32=1 equals b=1: preservation cannot satisfy U32. The v2 hypothesis of useful root-4 placement within H32 is unsupported under its valid bounded test, even though retention differs physically.

## 2. Root 4: exact retention, loss, height and contact timing

The common snapshot has mug z **0.977433 m**, **74.500 mm** above original first-grasp height, grasp=true, both pads and `microwave_1_g4` contacting the mug, no table contact, and neither goal predicate true.

| Measurement over 32 post-action endpoints | C0 | C1 | C2 |
|---|---:|---:|---:|
| Grasp=true / held_clear=true endpoints | 14 / 14 | 16 / 16 | 32 / 32 |
| First grasp and held-clear loss, local k (absolute step) | 15 (230) | 17 (232) | none |
| Both-pad contact endpoints | 14 | 16 | 32 |
| Any mug–gripper contact endpoints | 14 | 23 | 32 |
| Mug–table contact endpoints | 11 | 0 | 0 |
| Mug–microwave contact endpoints | 22 | 30 | 32 |
| Gripper–microwave contact endpoints | 24 | 22 | 32 |
| Final mug z, m | 0.902897 | 0.976547 | 0.977876 |
| Final height above first grasp, mm | −0.036 | 73.615 | 74.943 |
| Maximum mug displacement from snapshot, mm | 79.547 | 17.448 | 0.954 |
| Maximum mug rise above snapshot, mm | 3.056 | 3.713 | 0.443 |
| Maximum EEF displacement from snapshot, mm | 15.532 | 18.852 | 1.261 |

**C0: release followed by a fall.** Closing commands occur at k1–12, opening at **k13–15 (228–230)**, then closing at k16–32. Both pads retain contact through k14, and grasp is false from k15 onward without reacquisition. Mug–microwave contact is present at k1–12, k16–20 and k28–32. Table contact begins at **k22 (237)** and persists through H32. Final mug displacement is **(+3.198, −27.599, −74.537) mm**. Neither `In` nor `Close` is ever attained.

**C1: later grasp loss, but no table fall within H32.** Opening occurs at **k16–18** and k24. Grasp is true through k16, then false at every endpoint k17–32. One pad remains at k17; subsequent intermittent single-finger contacts (k20, k27, k29–32) do not restore grasp. Mug–microwave contact is present at k1–15 and **every k18–32**; the latter contacts involve `microwave_1_g7`. Height remains **70.125–78.214 mm above first grasp** over snapshot+branch. Final delta is **(−5.585, −12.377, −0.886) mm**. This is elevated, ungrasped contact with microwave geometry, not the same physical drop as C0, and not containment. `In` and `Close` remain false throughout.

**C2: no observed transient grasp or held-clear loss.** Grasp, held_clear, both-pad contact, mug–`microwave_1_g4` contact and gripper–microwave contact hold at **k0–32**; table contact is absent. All 32 commands close the gripper. Final delta is only **(+0.720, −0.442, +0.443) mm**. From k24 to k32 the mug moves **0.215 mm net**, with delta **(+0.078, −0.110, +0.167) mm**. It is not visibly progressing toward containment in these position/predicate measurements: `In=false` and `Close=false` at all 33 observed states.

C2's microwave contact pairs are mug geoms g8/g9/g10 against `microwave_1_g4`, and `gripper0_hand_collision` against microwave g6/g8. Logged mug–microwave signed distances span **−0.074 to +0.998 mm**; hand–microwave entries span **+0.842 to +0.963 mm**. The latter are positive-margin contacts, not logged penetration. Together with sub-millimetre mug motion despite repeated positive-y commands, these observations are **consistent with a contact-constrained stall**. They do not establish permanent jamming or a specific collision-based causal explanation. Door qpos is exactly unchanged at **−1.960378871 rad in all three branches**.

The observed effect is therefore avoidance/delay of the nominal disengagement through H32, not successful transfer. A later release, continued stall, or delayed placement after H32 remains unobserved. Opening and subsequent grasp loss are temporally associated in C0/C1; these multicoordinate, closed-loop traces do not isolate gripper opening as the cause or prove that forcing closure would repair the task.

## 3. Source diversity: boundary duplication is not whole-trajectory collapse

All three first calls per root use the same audited policy PNG/prompt. Boundary token and mapped-action uniqueness agree:

| Root | Unique boundary outputs /3 | Duplicate group | Unique 32-action vectors C0/C1/C2 | C1/C2 actions differing from aligned C0 |
|---|---:|---|---|---|
| 0 | 1 | C0=C1=C2 | 13/20/16 | 29/26 |
| 1 | 3 | none | 5/12/11 | 21/22 |
| 2 | 1 | C0=C1=C2 | 16/15/15 | 22/17 |
| 3 | 3 | none | 29/29/32 | 31/31 |
| 4 | 1 | C0=C1=C2 | 23/21/8 | 24/26 |
| 5 | 3 | none | 26/28/29 | 30/32 |

There are **12 distinct within-root boundary outputs across 18 allocated cells**, not 12 draws from one observation. Unique token-vector counts also equal the tabulated unique action-vector counts.

- **Root 4:** C1 first differs from C0 at **k4**, C2 at **k3**; each first divergence has an exactly matching before fingerprint. Across aligned trajectories, **all seven coordinates differ from C0** for each sample. Within C2 itself, the numbers of unique values in coordinates 1–7 are **2/6/3/5/2/8/1**: all six arm coordinates vary but its gripper is always closed. “All-coordinate diversity” relative to C0 must not be misreported as C2 exploring both gripper signs. C0/C1/C2 close/open counts are **29/3, 28/4, 32/0**. C2 y commands range **+0.525499 to +0.766991**, yet mug net y is −0.442 mm and EEF maximum displacement only 1.261 mm. Action diversity therefore does not establish useful physical alternatives.
- **Root 0:** both samples first differ at **k4** on an identical before state. Unlike v1's rotation-only differences, v2 samples differ from aligned greedy in **coordinates 2–6**, including y/z translation. All 96 commands nevertheless share x **+0.002127114** and closing **+1**. All z commands are **≤+0.000341382**, with sampled minima **−0.133061980**. This is restricted tested support, not total action identity or a measured property of the entire source distribution.

After the first divergence, aligned comparisons mostly compare different observations. They describe closed-loop consequences, not same-observation sampling entropy or a causal attribution to an individual later coordinate.

## 4. Root 0: same persistent low-height failure phase, not the same narrow v1 collapse

All three branches retain grasp and both-pad contact at k0–32. None ever attains held_clear, `In`, or `Close`; no mug–microwave contact occurs. Table-contact counts are **28/31/26 of 32** (C0/C1/C2).

| Root-0 measurement, mm | C0 | C1 | C2 |
|---|---:|---:|---:|
| Maximum mug displacement from snapshot | 6.933 | 4.674 | 7.493 |
| Maximum rise above snapshot | 0.906 | 1.072 | 0.670 |
| Maximum height above first grasp | 1.909 | 2.074 | 1.672 |
| Maximum EEF displacement from snapshot | 4.978 | 7.868 | 6.769 |

Final mug deltas are **(+5.279,+4.478,+0.370)**, **(+2.795,+3.590,+1.072)** and **(+5.895,+4.592,+0.102) mm**. Brief table-contact absence does not approach the >20 mm held-clear criterion. There is somewhat broader action/position variation than v1, but no lift or transfer. Longer observation is not specifically supported as a remedy for root 0; it remains a required negative/control case in a common-design experiment.

## 5. All successful controls and possible harms

**Root 1:** all candidates retain grasp through H32, with no goal attainment. Final heights above first grasp are **2.523/5.305/2.396 mm**, and table-contact counts are **6/2/3**. These modest differences do not demonstrate completion benefit or harm. The original tape first exceeds +20 mm at post-step **244**, local **38**, outside H32; first future `In` is local **71**, native success **237**.

**Root 2:** first grasp reacquisition is **k17/k17/k18**; C2 has no mug–gripper contact at k16–17, then regains it. All remain grasped thereafter. Final heights above first grasp are **16.369/13.948/10.250 mm**. Relative to C0, sampled endpoint heights are lower by **2.422/6.119 mm**, and C2's reacquisition is one action later. These are local adverse differences, not measured native task failures or proven eventual damage. No candidate attains held_clear or either goal by H32. Original first +20 mm rise is local **36**, first future `In` **81**, and native success **181**.

**Root 3:** C0/C1 never attain `In` during H32. C2 first attains it at **k15 (post-step 271)** and retains it at every endpoint **k15–32**, giving 18 consecutive contained endpoints and p32=2/U32=true. No branch grasps the mug or reaches `Close`. C2 moves the mug **+55.550 mm in y** versus C0's effectively stationary mug; it does not regrasp. Door start qpos is −1.432186 rad; endpoints are **−0.463917/−0.316955/−0.019115 rad**, but the authoritative native `Close` remains false even for C2. Original greedy first reattains `In` at local **58 (314)** and succeeds at **95 (351)**. This is accelerated containment on an originally successful root, not recovered failure or an observed native completion.

**Root 5:** all three retain snapshot `In` at every endpoint, never grasp the mug, and have exactly identical mug-position traces. All have `Close=false`; door qpos stays **−1.880591808 rad**. The EEF moves about **253–256 mm** while the mug stays effectively stationary, consistent with preparation for the closing phase rather than placement loss. Original native completion is local **85 (297)**.

**What is and is not measured:** no snapshot-goal destruction occurs, and no successful-control native harm is observed within this local window because **none of their fresh C0 branches completes within H32 either**. Full-lifetime sampled outcomes for all four successful controls are unmeasured. Do not report “no harms” or subtract original L1 successes from truncated sampled outcomes. Root 2's adverse local differences must remain in reporting even though labels tie.

## 6. Comparison with v1: timing evidence at different states

The plans use the same allocated source identities and local-offset seed lists for all 18 cells (verified by the extractor); the frozen protocols retain sampling settings/model/interface. The intervention boundary moved uniformly from **g+180 to g+120**, so each v2 snapshot is **60 actions earlier**, with different state/images and possibly different reference b. Equal seed numbers do not make the generated trajectories equal at these different observations.

- V1 root 4 began already disengaged, with no regrasp/contact with the mug during its local probes and essentially stationary object traces. V2 begins engaged and separates release/fall, elevated ungrasped contact, and retained-grasp plateau. This supports the importance of **which physical phase is probed**, but **both versions have zero useful failure-root gains** under their frozen labels. In v2 root 4 b rises from 0 to 1, so preserving its starting state cannot be counted as new progress.
- V1 root 0 varied only rotational coordinates 5/6 relative to C0; v2 varies 2–6, yet still never lifts. “The source is completely collapsed in both” would be false; “both bounded probes fail to produce useful acquisition” is supported.
- V1 root 3 C1/C2 completed locally at **29/31** versus original greedy **35** from its later boundary. V2 C2 only attains containment by H32; these are different physical windows, not evidence that the earlier timing harms full outcomes. V1 root 5 completion at **25/26/25** similarly cannot be compared as an overall success rate to v2's earlier, unfinished branches.
- V1 sampled roots 1/2 achieved containment in the later interval, whereas v2 H32 ends before even the original first substantial lifts. This horizon/boundary alignment matters; absence of early U is not a validated failure alarm.

These results are **same-source/seed, changed-timing local evidence**, not matched full-lifetime outcome comparisons. No v2 sampled suffix beyond H32 may be borrowed from v1 or from the original greedy tape.

## 7. One next-mechanism recommendation: a final bounded local horizon test

**Recommend a new v3 experiment whose only scientific mechanism change from v2 is local horizon 32→96.** Retain g+120 for all roots, all six roots × C0/C1/C2, identical original snapshots, model/interface/full instruction, stochastic settings, seed arithmetic, labels and reference b. Extend the same local seed formula to k=0..95; do not shift seeds by absolute time. The sampled source must remain sampled after action 32—no H32 handoff to greedy, which would test a different mechanism.

This recommendation has limited but concrete information value:

1. C2 root 4 remains engaged at H32, unlike the late v1 probe. H96 can distinguish delayed useful placement from prolonged contact-constrained holding or later disengagement **under this exact source/seed trajectory**. The observed plateau makes a positive result uncertain; it is not evidence of an imminent placement.
2. A fixed common H96 reaches the original root-1/2 containment times (71/81) and root-3/5 native times (95/85). Those are **original-tape timing landmarks**, not predicted sampled outcomes. This gives the retained successful controls an interpretable longer window instead of testing only early acquisition/pre-closing phases.
3. A single fixed horizon factor isolates this unresolved time-budget explanation more cleanly than simultaneously changing decoding, prompt and boundary. H96 is development-informed, not independently optimized or held-out.

The conservative budget is **18×96 = 1,728 new requested model calls/actions**, or **12,096 action tokens**. This includes **576 newly queried prefix calls**; it is not merely 18×64 appended calls. Original-prefix reconstruction remains **3,909 recorded actions**; evaluation including local actions is ≤**5,637**, independent replay ≤**5,637**, and 36×10 initialization actions give ≤**11,634 total simulator actions**. Early native stopping reduces work without reallocating saved calls. Original 520-action lifetime remains unchanged; every snapshot has at least 263 remaining actions, so H96 fits.

**Required gates before interpreting or advancing such a separately designed experiment:**

- **New preregistration and freeze before queries.** Explicit v3 identity, H96 cap, nested H12/H32/H96 readouts, seed extension, resource/watchdog caps, all artifact/runtime hashes and stopping rules. Record that the design is informed by v1/v2. Do not edit v2's frozen gate or label retention as success.
- **Fresh reconstruction and prefix identity.** Reconstruct from the original prefixes, verify exact integration/controller/diagnostic and both-image boundary identity, then newly query every branch. Require each branch's first **32 tokens/native actions/mapped actions, policy inputs and resulting state/image/diagnostic tape** to exactly match its corresponding v2 branch. A prefix mismatch invalidates the “horizon only” interpretation; no replacement seed or approximate match.
- **Complete matched controls.** All six fresh C0 H96 trajectories must reproduce the corresponding original greedy suffix, with native early success absorbing; C0 validation/replay must gate sampling. Independently replay every completed branch with zero model queries, retain all 18 identities and duplicates, and distinguish invalid/censored/missing cases from task failures. Fresh original matching alone is insufficient for C1/C2: their v2-prefix matches are separately required.
- **Failure-root gain, not retention or control gain.** Any proposed post-audit capability gate should require all 18 valid, all exact controls/prefixes/replays/global integrity, and sampled **U96=true on root 0 or 4 with matched C0 U96=false**, using the unchanged b and predicate-loss rule. Root-4 p96=1, root-3 acceleration or source diversity alone cannot pass it. Report native completions and control regressions separately.
- **Hard stop against endless tuning.** If this valid H96 test has no failure-root gain, stop this same-source/g+120/horizon-extension line; do not automatically try H160, more seeds, a root-specific timestamp, or an appended rollout. A subsequent study would need a substantively different, independently justified candidate source, not another clock adjustment. A positive proxy result only motivates separately designing matched full-lifetime outcome evaluation, including every successful control; it does not establish recovery, absence of harm, a selector, or L3 eligibility.

The plateau is a reason to bound this horizon test tightly, not enough evidence to assert that every longer test is uninformative or that a particular replacement source will solve it. **V2 remains stopped with L2B=false. Any H96 work is a new v3 local experiment, not an eligible v2 L2B continuation or a hidden extension of the failed gate.** No new protocol, launch, GPU allocation, commit, shared-file edit or agent delegation is part of this analysis.
