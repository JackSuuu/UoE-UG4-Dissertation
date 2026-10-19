# Native microwave L2 v2 — earlier-boundary candidate reachability (2026-10-19)

**Protocol ID: `native_microwave_l2_v2_20261019`. Freeze this design before any v2 model query or simulator branch. Execute L2A only: six snapshots × three candidates × at most 32 actions = 576 calls. Pause after its complete audit; no automatic L2B.**

The sole mechanism change from frozen v1 is the snapshot clock: **first target grasp +180 → +120 actions** for every root. Sources, model/interface, instruction, seed schedules, 18 identities, 12/32 nested readouts, label rule, original 520-action lifetime and failure-root continuation gate remain fixed. Reference levels change because the same rule is evaluated at earlier states. V1 remains stopped with its negative result intact.

This is a **development-informed preregistration for new v2 branches**, after observing L1 and L2A v1; it is not preregistration before those observations or held-out confirmation. All six roots are reused development cases, not 18 independent tasks. No runtime selector or universal early-failure alarm is claimed. This design work is CPU-only saved-log extraction, with zero model queries or simulator actions.

Design directory: `/home/ami/scratch/native_microwave_l2_v2_design_20261019/`. Its `analyze.py` adapts v1's offline extractor; `snapshot_manifest.json` retains the launcher-facing v1 schema and pins original source paths, snapshot fingerprints, source hashes and seeds; `evidence.json` records extraction checks and original-tape measurements. `design_freeze.json` hashes this note, extractor and both JSON artifacts. The launcher must pin that freeze's exact SHA-256 and freeze its own implementation/configuration before querying. No v1 protocol or implementation is modified by this note.

## 1. Rationale, hypothesis and falsifier

Source cohort: `/home/ami/scratch/native_long_l1_20261019_c/report/{summary.json,results.md}`, which lists all **24 planned task/root episodes**. The extractor verifies that the six microwave artifact paths, outcomes, lengths and passed replay status agree with the earlier `b/report`. Microwave has four original native successes and two partial-progress failures, with all six microwave cases completed, consistent and replay-passed. The report's provisional wider-screening flag and other tasks' replay issues are not evidence against those six microwave identities; 24 listed episodes does not imply 24 exact replays. Task/root selection is unchanged.

V1 L2A used 552/576 calls and had 18/18 valid identities, but **no sampled U32 gain on original failure roots 0 or 4**. Root 0's observed alternatives varied only two rotational coordinates, yielding <0.724 mm maximum mug rise over 32 actions. Root 4 had distinct boundary actions and subsequent differences in all seven command coordinates, yet no mug recontact and only 0.01724 mm maximum mug displacement. Root 3's sampled completion at 29/31 versus original greedy 35 actions was a successful-control acceleration, not failure recovery. Quantitative source: `native_microwave_l2a_failure_analysis_20261019.md` and the v1 audit, whose hashes are recorded in v2 evidence.

At root 4's earlier `g+120=215` endpoint, the mug is grasped, **74.500 mm above first-grasp height**, contacting both gripper pads and microwave geometry, without table contact or `In`. The original grasp-loss endpoint is 230, followed by a 77.135 mm fall by 262; v1 sampled only at 275. The proposed local interval **216–247** spans that original loss transition while beginning with object engagement. Temporal association of opening and loss does not establish that suppressing opening would fix placement.

**Mechanism hypothesis:** the v1 root-4 boundary is too late for this bounded source to find useful transfer alternatives; sampling the same actor before object disengagement can yield durable containment or native success within H32 when matched fresh greedy cannot. Its operational prediction is `U32(C1 or C2)=true` and `U32(C0)=false` on root 4. Root 0 is retained as a distinct persistent-acquisition probe, not assumed to share this mechanism.

**Bounded falsifier:** with valid exact controls and replays, neither sampled root-4 branch satisfies that prediction. Held-clear preservation alone does not support it. If a sampled root-0 gain nevertheless occurs, the general continuation gate may pass, but the specific root-4 hypothesis remains unsupported. If neither failure root has a sampled U32 gain, stop v2: the earlier timing failed to establish useful alternatives under this fixed source/horizon/seed budget. Invalid or censored evidence is inconclusive, not a mechanistic failure. Any negative is bounded by these cases and horizons; it does not establish impossibility of longer recovery or different sources.

## 2. One causal snapshot per root, with full matched controls

Scan each original trajectory chronologically. Latch `g`, the first post-action observation with `grasped[white_yellow_mug_1]=true`. At **post-action `t=g+120`**, collect the boundary if native success has not already terminated the episode, **regardless of current grasp, height, goal predicates or eventual outcome**. No timer reset, loss-triggered timestamp, fallback, root-specific offset or second snapshot. A missing grasp, early termination or missing/inconsistent boundary makes that root unavailable; do not replace it.

All indices below are zero-based post-action indices; state/frame `s=t+1` is the boundary before the next actor action. Current/past simulator grasp is privileged instrumentation for collection, not an RGB-only monitor or actor input. Prefix truncation must return the same snapshot from `rows[:t+1]` and none from `rows[:t]`.

| Root | First grasp g | Post-step t | State / next action s | Remaining R=520−s | Reference b | Boundary context |
|---|---:|---:|---:|---:|---:|---|
| 0 | 73 | **193** | 194 | 326 | **0** | grasped, table contact, neither goal |
| 1 | 86 | **206** | 207 | 313 | **0** | grasped, table contact, neither goal |
| 2 | 95 | **215** | 216 | 304 | **0** | not grasped, low-height gripper contacts, neither goal |
| 3 | 136 | **256** | 257 | 263 | **0** | not grasped, microwave contact, temporary `In` loss |
| 4 | 95 | **215** | 216 | 304 | **1** | held clear, microwave contact, not contained |
| 5 | 92 | **212** | 213 | 307 | **2** | contained, not grasped, door open |

Retain original-success roots **1,2,3,5**, with all three candidates each. Roots 1/2 originally first rise >20 mm only at g+158/g+156, beyond the new H32 endpoint g+152. Root 3's earlier snapshot lies in its naturally recovered `In`-loss interval. Thus early low-height/placement-loss observations and negative short-horizon controls are not validated failure alarms.

**Restoration:** reconstruct each boundary independently from the original official initial state, seed 0, native scene and the ten original initialization actions, then replay recorded mapped prefix actions `0..s−1`, with **zero actor queries**. Require exact integration/controller fingerprints, both RGB images and diagnostics against the saved snapshot. Preserve controller history; a naked qpos/qvel reset is insufficient. Reject an approximate restoration rather than substituting it. The extractor verifies saved frame `s`, after-action fingerprint `t`, next before-action fingerprint `s`, and diagnostics agree; these offline checks do not substitute for execution-time reconstruction verification.

## 3. Frozen sources, interface and candidate identities

Each root has exactly three closed-loop trajectories:

| Identity | Source for local offsets k=0..31 | Request seed |
|---|---|---|
| C0 | nominal greedy | **0** at every call |
| C1 | stochastic sample | `20261019 + 10000*root + 100 + k` |
| C2 | stochastic sample | `20261019 + 10000*root + 200 + k` |

The manifest's arithmetic seed formula applies to sampled candidate indices 1/2; C0 always uses zero. These are the **same local-offset schedules as v1**, not shifted absolute-time seeds.

Frozen actor: `openvla/openvla-7b-finetuned-libero-10`, checkpoint revision `80970322773f81baa2e22fe495d0487b93a05cfa`, official base-code revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`; Transformers **4.40.1**, BF16, SDPA, eval/inference mode, batch **1**, normalization key `libero_10`, original native camera preprocessing and official gripper mapping. Preserve native task/reset/physics. Use audited physical-GPU UUID/EGL preflight in execution; no guessed ordinal mapping.

Full unchanged instruction:

> put the yellow and white mug in the microwave and close it

Exact text prompt: `In: What action should the robot take to put the yellow and white mug in the microwave and close it?\nOut:`. Greedy uses `do_sample=False`, one beam, one return. Sampling uses `do_sample=True, temperature=0.7, top_p=0.95, top_k=50, num_beams=1, num_return_sequences=1`. Log resolved configuration and all seven returned tokens, native proposals and mapped actions. Official `predict_action` owns the seven-token generation budget; do not duplicate `max_new_tokens` through sampling kwargs.

All three first calls per root receive byte-identical original snapshot policy PNG and the identical instruction/prompt. Thereafter each branch obtains one new proposal from its **own fresh RGB** before every executed action. These are not repeated fixed-action probes or three proposals per timestep. No rewritten subgoal, privileged-state token, saved future observation, synthetic image, manual motion, interpolation, fallback or rejection sampling reaches the actor.

Execute each branch's sole valid proposal directly. Require seven finite coordinates and the unchanged audited mapping; invalid generation is an unusable cell, not repaired by clipping/redraw. Charge every requested call, including interrupted/invalid requests. Retain duplicates within their allocated identities. No temperature sweep, additional seeds, free draws, replacement branches or outcome-driven retries. Same-observation diversity claims apply only at the common boundary or subsequently proven identical inputs.

## 4. L2A execution order and stopping contract

1. Run **all six C0 branches**, in root-order dispatch, with exact fresh-model token/native/mapped-action comparison against the original greedy suffix and exact resulting local tape comparison. Each C0 receives a fresh-process, **zero-query independent action replay**, including boundary reconstruction, physics/integration/controller/images and diagnostic/label recomputation.
2. **All six C0 identity and replay checks must pass before any C1 or C2 model query.** A mismatch blocks sampling and interpretation, rather than becoming apparent repair benefit. Record all preallocated unrun cells as blocked/missing, not native failures.
3. Run C1 roots 0–5, then C2 roots 0–5, with independent recorded-action replay for every completed branch. Preserve the phase barriers; at most two independent branch workers, each batch one. Complete the 18-cell audit before interpreting the continuation gate.

Each trajectory has **at most 32 new model calls/actions**, with readouts after **12 and 32** actions on that same trajectory. Twelve is a nested readout, not another arm or handoff. Stop early only for native full conjunction, invalid generation, infrastructure failure, or the fixed resource caps. Do not stop/change source for proxy gain, grasp loss, predicate flicker, or another branch's outcome. Native success is absorbing for endpoint reporting; report the real action count rather than fictitious post-success actions. Count all attempts against both branch and aggregate caps.

L2A operational caps: **180 s worker readiness, 30 s per requested generation, 120 s branch wall including scene/prefix work but excluding worker load, 120 s independent replay, 7,200 s whole phase including load/replay**. Zero automatic retries. Outer process watchdogs, if required, must be frozen and cannot extend these inner scientific caps. Interruptions retain artifacts and attempted-call accounting and produce explicit infrastructure/censored cells. Early stopping never reallocates unused calls. The actor lifetime remains **520 actions** from the original episode; reconstruction/replay work is charged separately, not against new-query allowances or as extra lifetime actions.

## 5. Unchanged offline labels and required reporting

Record native `In`, native `Close`, native conjunction, grasp, object/EEF positions, endpoint contacts (and available substep contacts), and executed commands separately. Native simultaneous full conjunction is the task endpoint. The short-horizon labels remain exploratory instrumentation, not a new success definition.

`held_clear(d)` means current target grasp=true, mug z is **strictly >20 mm above its original first-grasp z**, and no mug–table **endpoint** contact is logged. This does not certify continuous contact-free support or stable lift. Reference `b=2` if snapshot `In` is true, else `1` if held_clear is true, else `0`. V2 references, roots 0–5: **`[0,0,0,0,1,2]`**.

At horizon h, take the highest applicable level:

- `p_h=3`: native full success occurred by h, absorbing.
- `p_h=2`: `In` is true at each of the final **eight** post-action observations ending at h.
- `p_h=1`: held_clear is true at each of those final eight observations.
- `p_h=0`: none of the above.

`U_h = (p_h > b) AND no goal predicate true at the snapshot became false anywhere in the branch prefix through h`. Only snapshot-true predicates enter this loss condition; also report other gains/losses independently. Except for absorbing native early success, fewer than h observations makes U **unknown/censored**, not false. An early success after action 12 does not alter the earlier U12. Direct native success gets level 3 without requiring any acquisition stage.

**Root 4 preservation alone is not progress:** keeping its lifted grasp at p=1 equals b=1 and yields U=false. Useful root-4 progress requires durable containment or full native success. Grasp retention, height or avoiding the original drop may be descriptive mechanism evidence but cannot replace the frozen U gate or native recovery endpoint.

Original-tape reference measurements (not new v2 branch outcomes):

| Root | b | Original p12 | Original p32 | Original U12/U32 |
|---|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | false/false |
| 1 | 0 | 0 | 0 | false/false |
| 2 | 0 | 0 | 0 | false/false |
| 3 | 0 | 0 | 0 | false/false |
| 4 | 1 | 1 | 0 | false/false |
| 5 | 2 | 2 | 2 | false/false |

Report all **6×3×2** labels with components, first attainment/loss times, actual lengths, censorship, boundary token/action uniqueness and duplicate groups, original/fresh nominal identity, every replay check, and per-root sampled gains over C0. Retain all valid-but-unhelpful candidates and full controls. Distinguish missing, invalid, censored, infrastructure and mismatch cells from native task failures. Provide physical phase/contact changes even if the gate fails.

Any “best available candidate” summary is **offline oracle availability**, not an executed policy or framework success rate. No future labels, simulator state, native success, U or winning-branch identity is supplied to an actor/selector to substitute an executed action. No selector is implemented or evaluated here.

## 6. Frozen continuation gate; L2B is optional and unlaunched

**Pause after complete L2A audit.** Eligibility for separately scheduled L2B requires: **all 18 cells valid**, all six fresh greedy controls exactly reproduced, all reconstruction/replay/global integrity checks passed, and at least one sampled branch on original failure root **0 or 4** with **U32=true while its C0 has U32=false**. This is the unchanged failure-root experiment gate. Root-3/other-control gain, diversity alone, root-4 held-clear retention, or any incomplete audit cannot pass it. If it fails, stop v2 without resampling, timestamp adjustment or continuation. If it passes, report eligibility and await separate scheduling; no automatic launch or immediate L2B is authorized by this design task.

To make the optional budget unambiguous, its scientific contract is frozen now: continue **all 18 preallocated identities**, not only proxy winners, with the same original lifetime. Restore each unfinished branch's H32 state by replaying its original prefix plus its own local tape and checking its boundary fingerprint. At exactly local action count 32 hand off to **fresh greedy H1** queries, identical model/interface/instruction, until native success or original actor index 519 has executed. Each remaining branch gets at most **R−32** further calls. Already natively successful branches are absorbing and receive no new calls. No second intervention, new seed bank, manual recovery, reset/regrasp script or post-success dwell. Fresh independent zero-query replay verifies every completed continuation. Optional L2B caps are 300 s branch wall excluding worker load, otherwise the same 180/30/120/7,200 s readiness/query/replay/phase caps, zero retries.

Report native `Y(root,candidate)` and censoring for every identity. A recovered failure requires sampled Y=true and matched fresh C0 Y=false; successful-control damage is C0 Y=true and sample Y=false. Local progress alone is not recovery. No L3 entitlement follows from a proxy gain; full recovery and reproduced controls would first justify designing a separately preregistered selector study. Earlier branching provides 60 more remaining lifetime actions than v1, while L2A's intervention horizon stays 32; any optional full-lifetime comparison must state that consequence of the timing change.

## 7. Exact ceilings and freeze requirements

`sum(s)=1303`, `sum(R)=1817`. All maxima are conservative ceilings; success or censored termination reduces actual work without replacement draws.

| Resource | L2A | Additional optional L2B | Combined upper bound |
|---|---:|---:|---:|
| Candidate identities | 18 | same 18 | 18 |
| New model calls/actions | **576** | **4,875** | **5,451** |
| Generated action tokens, seven/call | **4,032** | **34,125** | **38,157** |
| Prefix construction + evaluation simulator actions, excluding initialization | **4,485** | **9,360** | **13,845** |
| Independent recorded-action replay actions, excluding initialization | **≤4,485** | **≤9,360** | **≤13,845** |
| Initialization actions, both construction and replay, ten/scene | **360** | **360** | **720** |
| Total simulator actions including replay/initialization | **≤9,330** | **≤19,080** | **≤28,410** |

Arithmetic: L2A reconstruction is `3×1303=3909` recorded actions, plus `18×32=576` local actions. Optional remaining query allowance is `3×(1817−6×32)=4875`; combined new calls are `3×1817=5451`. Optional L2B reconstructs each prefix+local tape and follows the remaining lifetime, bounded by `18×520=9360` simulator actions per construction/evaluation pass, with the same bound for independent replay. Reconstruction/replay uses **zero model calls**. No separate calibration-query allowance exists: greedy identity checks consume allocated C0 calls. No new wall-time estimate is asserted from v1 throughput.

Before launch, bind the exact note/extractor/evidence/manifest freeze, all original source tapes and manifests, checkpoint/base-code/runtime sources, preprocessing/prompt and boundary PNG hashes, seed/generation configuration, new launcher/worker/auditor implementation hashes and physical-device preflight into an immutable execution plan. Require plan protocol ID **`native_microwave_l2_v2_20261019`**, not v1. Any later change to timing, source, seeds, horizon, label, gate or relevant runtime semantics requires a new version before queries. This design contains no implementation edits, commits, model/simulator runs or authorization to bypass the post-L2A pause.
