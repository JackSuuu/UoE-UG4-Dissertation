# Microwave L2A v1: failure mechanism analysis (2026-10-19)

**Stop v1: the failure-root capability gate is false. Recommend one separately preregistered change: move the common snapshot timer from first grasp +180 to +120 actions, keeping the same sources and 32-action horizon.** This tests an earlier transfer state before root 4 loses the mug; it is a development-informed hypothesis, not established prevention or recovery.

## Evidence and scope

Read-only CPU analysis of `/home/ami/scratch/native_microwave_l2a_20261019_v1`, its sibling `_audit`, and the six original L1 tapes. Reproducer and detailed measurements: `/home/ami/scratch/native_microwave_l2a_failure_analysis_20261019/{analyze.py,evidence.json}`. No model, GPU, simulator, or new rollout was used. Existing audit establishes **18/18 valid cells, exact greedy suffix reproduction and replay checks, 552 requested/returned/executed calls against the 576 cap**. All 18 cells are retained; none is censored or replaced.

These are **six reused development roots**, including two original failures and four original successes, not held-out validation or 18 independent tasks. All predicates, grasp/contact measurements, oracle labels and eventual root classifications are offline diagnostics. They are not actor/selector inputs. Post-action indices are zero-based; local action counts are one-based. Contact counts concern logged **endpoints**, not continuous/substep contact guarantees. Position differences below are measured world-coordinate millimetres; mapped commands are controller inputs, not millimetres.

## 1. Complete bounded result, including negatives

Every cell has `U12=false`. At H32:

| Root | Original L1 result | C0/C1/C2 actions | C0/C1/C2 U32 | Native successes within branch |
|---|---|---|---|---|
| 0 | failure | 32/32/32 | F/F/F | none |
| 1 | success | 32/32/32 | T/T/T | none; containment only |
| 2 | success | 32/32/32 | T/T/T | none; containment only |
| 3 | success | 32/29/31 | F/T/T | C1 at 29, C2 at 31 |
| 4 | failure | 32/32/32 | F/F/F | none |
| 5 | success | 25/26/25 | T/T/T | all, at their terminal action |

There are **0/4 sampled useful failure-root candidates**, and 0/6 useful cells including their greedy controls. Both failure roots have `p12=p32=0`, no transient held-clear acquisition, no `In`, and no `Close`. Root 3 is the sole sampled U32 gain over C0, on an already-successful original root. Its original greedy suffix succeeds after **35** actions; the exact fresh C0 prefix ends at 32. Thus 29/31 versus 35 supports earlier completion on this control and exposes horizon sensitivity, not failure recovery. Root 5 C1 is **one action slower** than C0, and C2 ties. No snapshot-true goal predicate is lost in any branch. Full long-horizon preservation for unfinished sampled controls 1/2 was not tested.

## 2. Source collapse versus diversity that does not help

All three first queries per root use the audited identical boundary PNG/prompt. Token and mapped-action uniqueness agree exactly:

| Root | Unique seven-token sequences / mapped actions, out of 3 | Duplicate group |
|---|---|---|
| 0 | 1 / 1 | C0=C1=C2 |
| 1 | 1 / 1 | C0=C1=C2 |
| 2 | 1 / 1 | C0=C1=C2 |
| 3 | 3 / 3 | none |
| 4 | 3 / 3 | none |
| 5 | 2 / 2 | C0=C2 |

The sum is **11 distinct within-root outputs across 18 allocated cells**, not a pooled same-observation sample. Duplicates arise in tokens already; no additional boundary collapse is introduced by action mapping. Three draws do not establish the full source distribution's support or entropy.

- **Root 0:** both samples first differ from C0 at local action **2**, when their before-state/images still match C0. Across each 32-action trajectory, C0/C1/C2 have **4/6/6** unique token/action vectors. C1 differs from time-aligned C0 at **17/32** actions, C2 at **22/32**, but **only rotational coordinates 5 and 6** vary (one-based). All 96 actions share translation `(0.0021271, −0.0019695, +0.0003414)`, rotational coordinate 4 `−0.0004643`, and closing command `+1`. This is boundary duplication plus severely restricted command diversity in the tested trajectories, not complete closed-loop identity.
- **Root 4:** all three boundary vectors differ. C1 changes y translation and rotational coordinate 4; C2 changes rotational coordinate 4. C0/C1/C2 subsequently have **16/29/19** unique vectors. C1/C2 differ from time-aligned C0 on **32/32 and 27/32** actions; over the traces, **all seven** coordinates differ, including gripper commands. Closing-command counts are **7/13/8**, opening counts **25/19/24**. This failure cannot be explained by duplicate proposals alone.
- Roots 1/2 show why first-call duplication alone is insufficient: their sampled traces subsequently differ from greedy on **28/32 and 28/32**, and **28/32 and 27/32**, respectively, while reaching containment. Conversely, root 4 shows why uniqueness alone is insufficient for useful physical alternatives.

After each first divergent action, time-aligned comparisons are descriptive **closed-loop trajectory comparisons on different observations**, not estimates of same-observation sampling diversity. The source/seed intervention changes the subsequent action-observation chain; these tapes do not isolate any particular later command's causal contribution.

## 3. Physical consequences and grasp/contact phases

All entries in each triple below are C0/C1/C2. Displacement maxima include the common snapshot and all recorded branch endpoints.

| Failure-root measurement | Root 0 | Root 4 |
|---|---|---|
| Grasp-true endpoints /32 | 32/32/32 | 0/0/0 |
| Mug–gripper contact endpoints /32 | 32/32/32 | 0/0/0 |
| Mug–table contact endpoints /32 | 27/28/25 | 32/32/32 |
| Mug–microwave contact endpoints /32 | 0/0/0 | 32/32/32 |
| Maximum mug displacement from snapshot, mm | 3.711/3.813/3.654 | 0.017234/0.017234/0.017234 |
| Maximum mug rise from snapshot, mm | 0.723/0.693/0.704 | 0.000914/0.000914/0.000914 |
| Maximum EEF displacement from snapshot, mm | 2.178/2.255/2.109 | 16.676/24.177/12.212 |

**Root 0: persistent low-height acquisition.** Final mug deltas are `(3.595,−0.570,+0.723)`, `(3.723,−0.456,+0.657)`, `(3.553,−0.482,+0.704)` mm. Maximum rise relative to the original first grasp is only **2.297/2.267/2.278 mm**, far below the 20 mm held-clear threshold. Both pads contact the mug; frequent table contact persists. Rotational sampling produces small physical differences without a lift/transfer phase. Grasp=true is therefore not sufficient evidence of usable acquisition. The logs do not identify whether perception, decoded action support, contact geometry or controller response is the underlying cause.

**Root 4: robot movement without mug engagement.** Mug contacts are exclusively `table_collision` and `microwave_1_g7`; no branch recontacts or regrasps the mug during its 32 actions. The entire C2 mug-position trace equals C0 exactly; C1's largest coordinate discrepancy from C0 is **3.47e−18 m**. The EEF moves differently, and commands include both opening and closing, but these differences do not reach the object at logged endpoints. The door remains at **−1.96038 rad** in all three branches. This is distinct but unhelpful late-state diversity, rather than a marginal useful lift rejected by a dwell threshold.

**Matched controls distinguish normal task phases:**

- Roots **1/2** begin held clear, have **zero mug–table endpoints** in all six branches, and translate the mug approximately **95–138 mm in +y** over H32. First `In` occurs at local **11/11/11** for root 1 and **21/24/22** for root 2; the latter samples delay containment by **3/1 actions**. Root 1 grasp losses first occur at **25/26/26**, after containment; root 2 C0/C2 first lose grasp at **27/32**, also after containment, while C1 stays grasped throughout. These release differences are not failure alarms. All six satisfy the final eight-endpoint containment dwell, but none closes the door within H32.
- Roots **3/5** start contained with no target grasp. Every branch preserves `In` at every observed endpoint, has no mug–gripper contact, and finishes or advances the closing phase. Root 3 door qpos changes from **−0.27206** to **−0.10427/+0.01561/+0.00867 rad**; only the samples reach native `Close` within H32. Root 5 all finish from door qpos **−1.88059 rad**. Mug displacement during closing is compatible with retained containment; more object motion alone would not establish recovery.

## 4. Why root 4's current snapshot is late

The original chronological tape already shows a different, earlier phase:

1. First grasp **95**; first >20 mm rise **107**, unlike root 0's persistent low-height behavior.
2. At **215 = g+120**, grasp is true, mug z **0.977433 m** (**+74.500 mm** versus first grasp), both gripper pads and `microwave_1_g4` contact it, table contact is absent, and `In=false`. The currently executed command closes the gripper. Thus a reachable pre-loss transfer boundary exists using a prefix-only timer.
3. At **229**, the opening command `−1` accompanies retained grasp; at **230**, grasp becomes false and mug contacts disappear. Mug z subsequently falls from **0.980041 m at 230 to 0.902906 m at 262**, a **77.135 mm** fall, with table support returning. This temporal association does not prove that suppressing opening would fix placement.
4. The v1 boundary **275** is **45 actions after grasp loss**, already table/microwave-supported, with `In=false`. The subsequent L2A probes start from this disengaged state. The original greedy episode never reacquires grasp or either goal through **519**; its later unsuccessful continuation does not rule out a longer sampled recovery.

The evidence supports separating root 0's acquisition limitation from root 4's lift/transfer then uncontained release and late non-reacquisition. No image interpretation or unlogged intention is required for those distinctions.

## 5. One next mechanism change and exact proposed budget

**Proposed revision: use `t = first_grasp + 120` for every root, rather than +180.** Keep C0 greedy, C1/C2's frozen sampling settings and seed schedules, full native instruction/interface, one branch per candidate, 12/32 nested readouts and exact reconstruction/replay. This is one snapshot-timing intervention, not an outcome-triggered failure detector. The 120 offset is chosen from these development tapes and must be declared as such in a **new preregistration before any new branch**.

Why prioritize timing: root 4 already has substantial source diversity at +180 with effectively identical stationary mug traces, whereas +120 observes a held, elevated mug **15 actions before** the original grasp-loss endpoint. A 32-action branch from 215 spans original actions **216–247**, covering that transition. It tests whether the same source can produce useful placement while the object is still engaged. Root 0's earlier boundary remains a table-contact acquisition case, so this proposal does **not** presume a remedy for both failure mechanisms.

Horizon and source remain plausible unresolved limitations. Root 3 establishes a three-action horizon cutoff effect on a successful control, but neither failure has near-threshold physical progress at H32. A longer horizon alone would principally extend the already disengaged root-4 probe. A different source could help root 0's restricted translation, but the present data do not identify a specific replacement source, and root 4 is not source-collapsed. Earlier timing is consequently the most concrete single-factor next diagnostic, not a demonstrated superior treatment.

| Proposed resource | Exact cap |
|---|---:|
| Same six roots × C0/C1/C2 | 18 branches |
| Post-action snapshot indices, roots 0–5 | 193, 206, 215, 256, 215, 212 |
| State / next-action indices | 194, 207, 216, 257, 216, 213 |
| New actor calls/actions | **18×32 = 576** |
| Generated action tokens | **4,032** |
| Recorded prefix reconstruction per evaluation pass | **3×1,303 = 3,909** actions, zero queries |
| Prefix + local actions per pass | **4,485** |
| Independent recorded-action replay pass | **≤4,485**, zero queries |
| 36 scene initializations ×10 actions | **360** |
| Total simulator-action upper bound, including replay/init | **9,330** |

Retain all four original-success roots as matched C0/C1/C2 controls, run all 18 allocated cells, retain duplicates/negatives, and permit no replacement seeds, retries or automatic continuation. Early native stopping reduces actual calls without reallocating savings. Reuse the v1 operational caps (180 s readiness, 30 s/query, 120 s branch excluding load, 7,200 s phase) and treat timeout cells as censored; freeze any implementation-specific wrapper limits in the new revision.

Recompute snapshot reference levels using the unchanged rule: at +120 they are **0,0,0,0,1,2**. Root 4 merely retaining elevated grasp is **preservation, not U=true**; do not relabel it recovery. Root 3 is now within its original temporary `In`-loss interval, and original successful roots 1/2 do not first rise >20 mm until g+158/g+156, beyond the proposed g+152 endpoint. Hence negative short-horizon controls at the earlier clock are expected possibilities, not evidence that earlier intervention is universally harmful. Report physical retention and predicate components descriptively, separately from the fixed useful-progress and native-success labels.

**Decision boundary:** v1 remains stopped, with no L2B or L3 entitlement. This note recommends a new bounded development experiment; it does not modify the frozen protocol, launch it, establish any recovered failure, or turn offline oracle availability into a deployable selector. Any claim of recovery still requires native completion against matched greedy, and later held-out evaluation must use genuinely new cases.
