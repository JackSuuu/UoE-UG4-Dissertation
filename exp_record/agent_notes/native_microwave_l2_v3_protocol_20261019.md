# Native microwave L2 v3 — final horizon-only local test (2026-10-19)

**Protocol ID: `native_microwave_l2_v3_20261019`. Freeze before any v3 model query or simulator branch. One final local experiment: six original g+120 snapshots × three unchanged candidate sources × at most 96 fresh actions = 1,728 NEW requested model calls. The first 576 calls freshly reproduce v2; every branch must pass exact prefix identity before requesting action 33. Pause after the complete audit. No automatic L2B or full-outcome continuation is authorized.**

The sole scientific mechanism change from v2 is **H32→H96**, with nested **12/32/96** readouts of one trajectory. Sources, original snapshots, model/interface/full instruction, seed arithmetic, decoding, labels/references, cohort and original 520-action lifetime remain fixed. Resource ceilings scale only to accommodate the longer local evaluation/replay. This is **development-informed preregistration after L1/v1/v2**, on six reused development roots, not held-out confirmation or 18 independent tasks.

## 1. Rationale, hypothesis and terminal decision

V2 completed 18/18 valid identities with 576 calls, exact controls/replays and **L2B gate=false**. No sampled U32 gain occurred on original failures 0/4. Root 4 C2 retained grasp/held_clear at every logged endpoint through H32, whereas C0/C1 first lost grasp at local 15/17. But C2 moved the mug only **0.954 mm maximum**, remained in microwave contact, and never attained `In`. Its p32=1 equals b=1. This is a contact-constrained plateau, not demonstrated delayed placement or recovery. Root 0 never lifted usefully. Root 3 C2's containment gain occurred on an originally successful control.

**Hypothesis:** the fixed H32 horizon truncates a useful later outcome available from the same source and seed trajectories at g+120. Operational prediction: at least one sampled candidate on root **0 or 4** has **U96=true while matched C0 has U96=false**, under complete exact identity/replay controls. The root-4-specific interpretation requires a gain on root 4; a root-0-only gain does not prove delayed root-4 placement. Root-4 held-clear retention alone remains insufficient.

H96 tests whether the retained branch places later, stays contact-constrained, or disengages. Its plateau makes a positive result uncertain. Original control timing landmarks—containment at local 71/81 on roots 1/2 and native completion at 95/85 on roots 3/5—make a common H96 window interpretable, but do not predict sampled outcomes. No v1/original suffix is substituted for an unobserved sampled v3 suffix.

**Hard stop:** if all validity prerequisites pass but no original-failure sampled U96 gain exists, end this **source/g+120/horizon-extension line**. No further duration extension, seed search, root-specific timestamp or horizon tuning, automatic resampling, or appended rollout. A positive proxy gate only recommends designing a **separately frozen full-outcome/handoff study**; it does not authorize one or establish full recovery, absence of control harm, a runtime selector, or L3. Invalid/censored evidence is inconclusive rather than a native failure; zero retries and no replacement cells still apply.

## 2. Frozen snapshots and cohort

Use the unchanged prefix-only rule: latch first post-action target grasp g; take post-step **t=g+120** if the original episode has not already terminated, irrespective of current grasp, predicates or eventual outcome. No reset of g, fallback, outcome filtering or replacement. State/frame **s=t+1** is the boundary before the next actor action. Indices t/g/s are zero-based; local executed-action counts are one-based. Original initialization consists of ten recorded initialization actions, separately charged.

| Root | Original L1 result | g | t | s | Remaining R=520−s | Reference b |
|---|---|---:|---:|---:|---:|---:|
| 0 | failure | 73 | 193 | 194 | 326 | 0 |
| 1 | success | 86 | 206 | 207 | 313 | 0 |
| 2 | success | 95 | 215 | 216 | 304 | 0 |
| 3 | success | 136 | 256 | 257 | 263 | 0 |
| 4 | failure | 95 | 215 | 216 | 304 | 1 |
| 5 | success | 92 | 212 | 213 | 307 | 2 |

All 18 identities, including all four originally successful roots, remain allocated. Root 0 is a persistent low-height acquisition probe, not assumed to share root 4's mechanism. Privileged grasp/predicate/contact diagnostics support offline collection/reporting, not actor inputs or an RGB-only failure detector.

Independently reconstruct every boundary from the original official initial state, seed 0, unchanged native scene/physics and initialization, then replay mapped original actions **0..s−1** with zero actor queries. Require exact integration state, controller history, both RGB images and diagnostics against the pinned original and v2 boundary. A naked qpos/qvel reset or approximate match is invalid. The offline extractor checks prefix truncation and saved boundary consistency; execution-time reconstruction verification remains mandatory.

## 3. Unchanged actor, source, instruction and seeds

Actor: `openvla/openvla-7b-finetuned-libero-10`, checkpoint revision **`80970322773f81baa2e22fe495d0487b93a05cfa`**, base-code revision **`47a0ec7fc4ec123775a391911046cf33cf9ed83f`**; Transformers **4.40.1**, BF16, SDPA, eval/inference mode, batch **1**, normalization `libero_10`, original native camera preprocessing and official gripper mapping.

Full instruction remains:

> put the yellow and white mug in the microwave and close it

Exact prompt: `In: What action should the robot take to put the yellow and white mug in the microwave and close it?\nOut:`.

For local offsets **k=0..95**:

| Identity | Source | Request seed |
|---|---|---|
| C0 | nominal greedy | 0 at every call |
| C1 | stochastic sample | `20261019 + 10000*root + 100 + k` |
| C2 | stochastic sample | `20261019 + 10000*root + 200 + k` |

The first 32 offsets exactly reuse v2's seeds; offsets 32–95 extend the same arithmetic without a reset or absolute-time shift. Greedy uses `do_sample=False`, one beam/return. Sampling uses **`do_sample=True, temperature=0.7, top_p=0.95, top_k=50, num_beams=1, num_return_sequences=1`**. Official `predict_action` owns seven-token generation; do not supply a second `max_new_tokens` through sampling kwargs. Log resolved configuration, all tokens, native and mapped actions, seeds and PNG hashes.

The first query of each candidate uses the same original snapshot policy PNG/prompt; every later query receives its own branch's fresh RGB. Execute the sole valid seven-finite-coordinate proposal directly with the unchanged mapping. No clipping repair, redraw, prompt/subgoal rewriting, privileged token, manual motion, interpolation, rejection sampling, extra seed, fallback or replacement. Retain duplicates. **C1/C2 remain sampled after action 32; C0 remains greedy. There is no source handoff at 12, 32 or any proxy event.**

## 4. Mandatory fresh-v2-prefix identity barrier

All 18 v3 branches are freshly reconstructed and newly queried from local offset zero. Saved v2 actions may be used only as comparison/replay evidence; **do not execute a replayed v2 prefix in place of the first 32 new actor calls**. The 576 fresh prefix calls consume the v3 budget.

Compare every branch to its **own corresponding v2 identity**, not merely to C0. Exact requirements cover:

1. All **32 query inputs/outputs**: local/absolute index, unchanged source/seed/instruction/prompt/resolved generation settings, seven token IDs, seven native coordinates, seven mapped coordinates, policy PNG byte hash (plus recorded policy/raw RGB hashes).
2. The **33 frames** comprising the reconstructed boundary and each post-action endpoint: exact integration/controller fingerprints, both RGB images, complete diagnostics including predicates/grasp/contacts, and saved numerical state/integration arrays. Require exact dtype/shape/array bytes and exact semantic diagnostic values, not rounded floats or tolerance-based matching. Check every before/after transition against these frames.
3. Expected policy PNG files, actions/state/image arrays and frame records must agree with their pinned reference artifacts. Runtime bookkeeping (PID, timestamps, elapsed time, local output paths) is not scientific identity and is excluded explicitly; scientific fields above may not be excluded.

**Fail closed before requesting query 33 (zero-based query index 32).** Verify incrementally and stop immediately on a known mismatch; after action 32, persist a successful complete-prefix comparison certificate bound to the v3 plan and v2 reference hashes before any query 33 request is emitted. Missing certificate, incomplete artifacts, mismatch, or premature success within the reference prefix blocks extension. All v2 references contain 32 nonterminal actions, so a new earlier native success is an identity mismatch, not a valid early recovery. Classify failed-prefix cells as **invalid identity, not native task failures**. They cannot pass the experiment gate; do not retry or replace them.

The design's `v2_prefix_manifest.json` supplies 18 reference entries, 576 expected queries and 594 frame descriptors, with exact artifact hashes and canonical semantic hashes. Complete frozen tapes remain the authority. Diagnostic hashes use sorted-key compact UTF-8 JSON; array hashes include C-order bytes, dtype and shape. This offline reference preparation is not an implementation or validation of the future runtime barrier. The separately frozen launcher/worker/auditor must demonstrate the barrier, including rejection of a bad/missing prefix certificate before any 33rd request, before scientific execution.

## 5. Dispatch, full greedy identity, replay and stopping

1. Dispatch **C0 roots 0–5** first, at most two independent workers. Each must pass the fresh H32 prefix gate and then match the **entire original greedy suffix through local H96 or the original native early-success endpoint**, including fresh tokens/native/mapped actions, inputs, resulting integration/controller/images and diagnostics. No original full-success label is substituted for a newly executed comparison.
2. Independently replay each C0's recorded actions in a fresh process with **zero actor queries**, including original-prefix reconstruction and all local endpoints. All **six complete C0 original-suffix identities and strict replays must pass before any C1/C2 model query**. Failure blocks sampling; report allocated unrun cells explicitly rather than as task failures.
3. Dispatch **C1 roots 0–5, then C2 roots 0–5**, preserving phase barriers/root-order dispatch. Each has the same per-cell fresh-prefix barrier and independent zero-query replay of every completed local tape. Any global identity/integrity failure blocks interpretation; no valid subset can substitute for the complete experiment gate.
4. Audit all 18 allocated identities and all three nested horizons, then **pause**. No automatic continuation or winner-only rollout.

Native simultaneous full conjunction is absorbing: stop immediately on native success and record the real call/action count; no fictitious post-success steps or extra dwell. Other stop reasons are invalid generation, identity mismatch, infrastructure failure or fixed resource caps. Do not stop/change source for U gain, predicate flicker, grasp loss, contact plateau or another branch's result. Charge all requested calls, including interrupted/invalid ones, against per-cell and aggregate ceilings. Unused calls are never reallocated. Actor lifetime remains indices **0..519** inclusive; reconstruction/replay are separate zero-query work, not new lifetime allowances.

## 6. Fixed execution resources

Only the following resource scaling accompanies H96:

| Limit | Frozen v3 cap |
|---|---:|
| Worker readiness | 180 s |
| Each requested generation | 30 s |
| Local branch wall, excluding model load, including setup/reconstruction, prefix checking and local actions | 300 s |
| Independent replay inner wall | 180 s |
| Branch child outer watchdog | 540 s |
| Replay outer watchdog | 210 s |
| Whole phase, including loads/replays | 14,400 s |
| Parallel branch workers / batch per worker | 2 / 1 |
| Retries / replacement candidates | 0 / 0 |

Use only **physical GPUs 2 and 3; never GPUs 0 or 1**. The v2 reference UUID/EGL lanes are GPU2 `GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8` / EGL1 and GPU3 `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b` / EGL0. These historical lanes must be checked by fresh physical-UUID, visible-CUDA and EGL preflight before execution, not assumed from an ordinal. Freeze the execution plan/environment and proof of renderer/actor device alignment. Outer watchdogs cannot extend scientific inner caps. Timeouts retain artifacts and explicit censored/infrastructure accounting; they are not native failures.

## 7. Unchanged labels, original-suffix references and reporting

`held_clear` means target grasp=true, mug z **strictly >0.020 m above the original first-grasp z**, and no mug–table **endpoint** contact. This is endpoint instrumentation, not a continuous collision-free/stable-support certificate. Reference b is 2 for snapshot `In`, else 1 for held_clear, else 0: **[0,0,0,0,1,2]**.

At h=12,32,96, p_h is the highest applicable level:

- **3**: native full conjunction occurred by h, absorbing;
- **2**: `In` true at all final eight post-action observations ending at h;
- **1**: held_clear true at those final eight observations;
- **0**: none of these.

`U_h = (p_h > b) AND no snapshot-true goal predicate became false anywhere through h`. Only snapshot-true goals enter that loss condition; separately report all predicate/grasp gains and losses. Fewer than h observed actions without native success yields unknown/censored, not false. A later success does not change an earlier readout. A native early success needs no eight-step dwell. Root 4 preservation at level 1 is never useful progress relative to b=1.

The offline extractor computes these **original-tape reference labels**, not new v3 outcomes:

| Root | Original p12 / p32 / p96 | Original U12 / U32 / U96 | Actions observed by H96 | First native local action by H96 |
|---|---|---|---:|---:|
| 0 | 0 / 0 / 0 | F / F / F | 96 | none |
| 1 | 0 / 0 / 2 | F / F / T | 96 | none |
| 2 | 0 / 0 / 2 | F / F / T | 96 | none |
| 3 | 0 / 0 / 3 | F / F / T | 95 | 95 |
| 4 | 1 / 0 / 0 | F / F / F | 96 | none |
| 5 | 2 / 2 / 3 | F / F / T | 85 | 85 |

Report all **54 root×candidate×horizon** labels with components, first attainment/loss, actual lengths, native outcomes and censoring; all 18 prefix certificates and replays; six original/fresh C0 suffix checks; boundary token/action uniqueness and duplicate groups; source/command/contact/height timing; and original-failure gains plus successful-control regressions. Same-observation diversity claims are limited to identical audited inputs. Full-lifetime sampled harms outside H96 remain unmeasured; do not equate incomplete sampled branches to original native failures or report absence of full-outcome harm.

“Best available candidate” is **offline oracle availability**, not an executed selector/framework success rate. No native/U/future/oracle winner information becomes an actor input or substitutes an executed action. No selector is implemented by this protocol.

## 8. Post-audit proxy gate and stopping contract

The only positive experiment-level decision requires **all 18 valid**, all reconstruction/global integrity checks passed, **all 18 exact fresh-v2-prefix identities**, all six full fresh-C0/original-suffix identities, every required independent replay passed, and **at least one sampled U96=true on original failure root 0 or 4 while that root's matched fresh C0 has U96=false**.

Root-3/other successful-control gain, root-4 held-clear retention, source diversity or an incomplete audit cannot pass. **Gate=false on complete valid evidence ends this source/g+120/horizon line.** Any invalidity blocks the capability conclusion and does not license replacements under this frozen design. Preserve v1/v2 negative results and v2's stopped L2B state.

**Gate=true only recommends a separately designed, separately preregistered/frozen full-outcome and source-handoff study with matched controls.** No L2B budget, handoff step, full-lifetime rollout or automatic launch is authorized here. Native recovery and preservation/harm require the appropriate matched outcome evaluation; proxy availability alone never establishes a deployable selection policy or L3 entitlement.

## 9. Exact conservative budgets

`sum(s)=1303`, `sum(R)=1817`, and the minimum R is 263, so every H96 fits the unchanged lifetime. All resource ceilings are unreallocated caps; native early stopping or unusable cells reduce actual work.

| Resource | Upper bound |
|---|---:|
| Snapshots × candidates | 6 × 3 = 18 |
| New requested model calls / new local actions | **18 × 96 = 1,728** |
| Fresh first-32 calls included above | **18 × 32 = 576** |
| Calls beyond those prefixes included above | 18 × 64 = 1,152 |
| Generated action tokens, seven/call | **12,096** |
| Original recorded-prefix reconstruction across 18 branches | **3 × 1,303 = 3,909** actions, zero queries |
| Reconstruction + local evaluation, excluding initialization | **3,909 + 1,728 = 5,637** simulator actions |
| Independent reconstruction + recorded-local-action replay | **≤5,637** simulator actions, zero queries |
| 36 scene initializations × 10 actions | **360** simulator actions |
| Total including evaluation/replay/initialization | **≤11,634** simulator actions |
| Authorized continuation queries/actions | **0** |

Exact C0 controls should stop at local 95/85 on roots 3/5, so their expected total is **564**, rather than 576; if all identities remain valid, that alone reduces the 1,728 ceiling by 12. Keep the conservative allocation unchanged and do not spend those savings elsewhere. There is no additional calibration-query budget; all fresh nominal/prefix checks consume allocated calls. GPU/runtime preflight and recorded-action replay have no model allowance.

## 10. Frozen artifacts, schema and execution prerequisites

Design directory: `/home/ami/scratch/native_microwave_l2_v3_design_20261019/`.

- `analyze.py`: CPU-only saved-JSON/NumPy extraction and hashing; no model/simulator imports or execution. It verifies v2's prior freeze, complete valid negative audit, unchanged snapshot/seed source evidence, prefix-only boundary consistency, old H12/H32 original labels, and extracts H96 labels. It builds and checks all 18 prefix references from saved tapes.
- `snapshot_manifest.json`: independent v3 launcher-facing manifest preserving the prior per-snapshot schema and exact six snapshot records; only version/horizon/readouts/resources/gating-related top-level contracts change. `snapshot_manifest.schema.json` supplies a Draft 2020-12 schema; the extractor additionally asserts cross-record indices/references and arithmetic invariants.
- `v2_prefix_manifest.json`: all 18 corresponding v2 directories, scientific artifact hashes, exact expected query/frame descriptors and canonical semantic prefix hashes. Original v2 files remain immutable reference data.
- `evidence.json`: original suffix labels at 12/32/96, extraction checks, full original tape/source provenance and input hash inventory.
- `design_freeze.json`: pins **this note, extractor, schema, evidence, snapshot manifest, prefix manifest, v2 audit/plan/launch/status/provenance and all 18 v2 scientific prefix artifact sets**, including queries/responses/tokens/native/mapped arrays, state/image arrays, frame files, policy PNGs, reconstruction/controller/device evidence, archived sources and replay certificates. It also pins original episode tapes, initial states, scene/model, source manifests/archives and prior design/report provenance. It excludes mutable logs/caches; runtime bookkeeping is not prefix identity. The freeze itself is identified by its externally reported SHA-256, avoiding a self-referential hash.

Reproduce offline with `python /home/ami/scratch/native_microwave_l2_v3_design_20261019/analyze.py`; verify an existing freeze without writes using the same command plus `--verify`. Schema validation is offline; no network retrieval of schema resources is needed.

Before querying, the execution owner must bind this exact design-freeze SHA and all referenced artifact hashes into a **new v3 immutable launch plan**, with independently frozen new launcher/worker/auditor code, unchanged actor/runtime/preprocessing and prompt hashes, per-cell seeds, scientific/resource caps, hardware preflight and phase barriers. A v1/v2 launcher must not silently treat this manifest as a continuation. Demonstrate the pre-query-33 fail-closed barrier and all-six-C0 sampling barrier with offline fixtures before launch. Any subsequent change to snapshots, source, instruction, seeds, labels, identity requirements, horizon or gate requires a new approved design, not an in-place relaxation. This design task itself performs **zero queries, zero simulator actions and zero GPU operations**.
