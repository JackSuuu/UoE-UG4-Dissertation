# M1: batch development failure discovery with physical execution proofs

## Decision and purpose (15 Oct 2026, before any M1 live actor outcomes)

The user explicitly requests broad, customized scene discovery instead of iterating one easy nominal experiment, plus integration of established SEAL/CheckVLA ideas. This supersedes the earlier narrow D2-only next step. Historical D1/D2 records and the five-layer architecture remain intact. M1 implements a **39-configuration development discovery batch**, not 39 distinct tasks and not a final efficacy benchmark. It asks which execution faults expose useful, recoverable headroom for a frozen actor plus local repair. Every case runs every arm, including sham successes and harmful repairs. A positive selected example is a mechanism demonstration, not an unbiased overall score.

First batch uses the already audited bowl/drawer long task and three consumed development roots (0/1/2); this allows broad fault screening now without conflating new task adapters and invalid task predicates. Book/caddy is the next cross-task transfer after the fault map. All calibration/training/final reservations remain unchanged; no root20–49 is opened.

## Fixed case grid and full task endpoint

Each root has 13 configurations:

- sham (release delay0);
- actuator release delay4/8/12/16 control intervals (0.2/0.4/0.6/0.8 simulated seconds);
- external world-x/world-y pose-command bias of ±10mm/±20mm cumulative requested displacement, spread over four intervals (8 configurations).

Thus 3×13=39 case identities; five arms give **195 live actor continuations**. This is one fixed development grid, not a severity search repeatedly extended until ours wins. No combined faults in M1. Bias is added after method selection and clipped to native controller input limits; actual EEF/object displacement is NOT asserted equal to requested displacement. No object teleportation, object mass/friction change, hidden-state reset or artificial success label.

Use previously verified clean action prefixes to first release opportunity at decisions169/147/163. All arms replay the same original-root prefix and must reproduce integration/controller/camera fingerprints before continuation. Only continuation steps query live frozen OpenVLA on each arm's actual images. Prefix actions/decodes are explicitly marked replayed, not newly queried. This is a matched-prefix recovery study, not an unrestricted start-to-finish detector assessment. Methods receive no case ID, injection anchor/duration/intensity/active label or future fault schedule. They receive current/past measured geometry/contact, actual selected-command history, and the current H1 proposal. The evaluator applies the same actuator fault AFTER selection/repair.

The unchanged D1 full physical completion criterion and600-interval total cap include prefix length; stop at20 consecutive complete intervals, not native point-in-drawer success. Canonical instruction, frozen LIBERO-10 checkpoint, nativeH1/k1 and actual observation cadence are common. No same-frame deterministic resampling described as replanning. Each continuation interval has one query, including intervention intervals. No training or model expansion. No learned or validated safe-success claim.

## Common causal event and bounded controllers

Entry: previous measured native-in region AND bowl/gripper contact AND drawer not closed AND current nominal proposal asks open. This is a **release opportunity**, not proof of failure. All three known anchors precede stable support; root2 lacks full footprint. Early holding/opening may harm normal placement, which is why all sham cases are mandatory.

There is one transaction per episode with at most40 selected intervention intervals. The existing retained-placement/released/clear3-interval histories govern transitions. Outside-native-region for3 consecutive measured intervals aborts with `release_envelope_lost`. Temporary lack of full retained placement at entry does not immediately abort (otherwise all three early entries abort). No retrigger. Only current/past evidence is available. Failed or exhausted returns are logged as such. Remaining drawer closing is always real VLA execution.

| Arm | Executed response |
|---|---|
| `vla` | Original live actor proposal, with monitor and shadow candidate evidence. |
| `gate_hold` | Zero6D pose + open; return after3 retained/detached intervals, loss, or40-step cap. This stronger release-conditioned simple control is deliberately distinct from D1's clearance-conditioned hold. |
| `timed_repair` |10hold/open then30outward/open; only envelope loss interrupts. |
| `phase_repair` |Hold/open until3retained/detached intervals; outward/open while released and not currently clear; hold to accumulate3stable clear intervals, then return. |
| `verified_repair` |Same phase controller; at the two registered verification opportunities only, may substitute a physically verified candidate first action. |

Outward primitive is a5mm requested translation along drawer-local negative y, zero orientation and open gripper. No vertical thrust under the cabinet shelf. These are bounded engineering heuristics, not certified collision-free actions. `gate_hold` and `timed_repair` are strong same-sensing controls; improvement over actor-only is insufficient for the full framework claim.

## Literature-inspired integration and attribution boundaries

Primary sources checked 15 Oct:

- SEAL paper [2510.16281](https://arxiv.org/abs/2510.16281), [official project](https://yilin-wu98.github.io/steering-reasoning-vla/), [official NVlabs/actalign README](https://github.com/NVlabs/actalign). Official policy is reasoning π0 with sampled text-plan-conditioned action sequences, simulator prediction to next reasoning switch, and VLM alignment verification. Official README currently lists model repos as TBD. Published gains do not guarantee ours.
- CheckVLA paper [2607.26789v1](https://arxiv.org/html/2607.26789v1), [author page](https://scpsyl.github.io/). It uses frozen action-conditioned learned feature prediction, temporal risk, episode-level functional conformal calibration, latency-aware flow-based suffix rewriting and policy/risk memory. No official code/checkpoint entry was located in these sources/search. Existing `checkvla_official.py` is a non-executable skeleton and is NOT called.

**Implemented SEAL-inspired logic:** propose → predict → verify → execute → observe. At first entry and first subsequent released event (maximum two opportunities), evaluate exactly three candidates: nominal first action, phase-controller first action, hold/open. Each is a3-interval local pulse-and-settle controller plan: first action followed by two zero-pose steps with its selected gripper command. These are NOT future VLA predictions or action chunks. Branches independently replay the entire actually executed prefix and must match the live integration/controller/camera hash. Future actuator faults are neither known nor reproduced in forecasts; prediction errors during a fault remain visible in the proof.

Rank candidates lexicographically by: intervals outside native region; intervals without retained placement; intervals with bowl/cabinet penetration>1mm; bowl/gripper contact intervals; intervals lacking clearance; deviation from nominal command; candidate index. This frozen physical subgoal heuristic may be wrong. Log all candidates, predicted measurements, rank keys, chosen first action, and actual observed effect. No claim of calibrated physical risk or semantic VLM verification.

All arms receive the same maximum2opportunities×3candidates×3forecast intervals=18forecast intervals, computed in shadow for controls. Only `verified_repair` uses the chosen action while its repair transaction remains active. Opportunity realizations depend on each arm's measured trajectory and are reported, as are prefix-replay steps and prediction wall time. Simulation branches are sequential, candidate batch3≤8. No extra VLA calls for predictions. Compare verified versus phase to test this added mechanism. If verified loses, report it; no online candidate/ranking retuning within M1.

**Implemented CheckVLA-inspired evidence interfaces:** causally bind previously selected open commands to later persistent bowl contact; maintain a capped16-event bank with original evidence indices, phase, release, retention and clearance. Existing measured-history release/clearance counters drive actual phase decisions. The new command/consequence residual and keyframe bank are **diagnostic only** in M1, not a trained/certified alarm or memory-conditioned actor. They help locate missed/late events and regressions without pretending a logging addition proves policy benefit. No conformal bound is claimed from three development roots. Real CheckVLA calibration/model/flow-rewriter integration remains pending; H1 has no deployable chunk suffix. Adding those requires genuine compatible modules and new matched controls, not renaming this controller.

Full trace: measured evidence→phase/history→entry/exit predicates→mode/reason→candidate forecasts→nominal/selected command→externally effective command→observed effect→handoff→remaining task outcome. Fault metadata is appended only as evaluator context after decision. This is a physical execution proof, not a fabricated internal language-model chain of thought.

## Discovery output and decisions

Every case gets all five labeled2xvideos in a single comparison and a clickable decision timeline. Automatic discovery map includes four paired categories: both succeed, framework rescue, framework harm, both fail; compare both phase and verified against actor, hold, timed and each other. Publish all severities/directions, non-triggering cases, budget exhaustion, harmful corrections and later-stage failures. Report task SR, native SR, actual changed actions, trigger/exit timing, prediction/observation disagreements and complete-task paired rescue/harm.

Uncertainty uses root clusters (all variants travel together); only3consumed roots means estimates are descriptive, not confirmation. The discovery map may select a promising **fault family and mechanism** for the next frozen-method independent cohort. Do not pool selected wins as the headline success rate. Calibration and independent confirmation will be registered after discovery, with all failures and denominators retained.

If VLA solves the whole grid, there is no demonstrated necessity in these conditions: broaden to a distinct phase/task/failure family on development data, without claiming benefit. If all controllers fail, diagnose recoverability/coverage/controller limits. If hold/timed tie or beat ours, credit the simple mechanism and simplify or make one failure-supported repair change. If phase or verified yields reproducible rescues beyond both controls, freeze the mechanism for unseen development transfer and later independent confirmation. No published architecture automatically guarantees transfer efficacy.

## Execution integrity

Before launch: unit tests for fault invariance, causal gate/handoff/budget, no retrigger, forecast candidate semantics and command/consequence history; real-simulator pipeline smoke using recorded-command fixtures, explicitly not online efficacy. Five methods' physical actions are replayed independently, including intervention and external faults, and candidate branches replayed again. Live actor load is strictly checked against D1build metadata; every live anchor must reproduce its historical deterministic decode.

Detached supervisor and root-sharded workers use healthy GPUs1/2/3 with matching numeric CUDA/EGL IDs. Every episode verifies state/controller/camera hashes,25substeps, measurement histories, selected versus effective actions, trace, termination/query caps and video lengths before the next case. Sources, protocol, task, case grid and actor identity are hashed. Failure stops the batch explicitly, preserving completed/partial records. Each completed five-arm case renders immediately. After all195episodes verify, produce discovery tables and archive automatically. GitHub/LFS push and independent remote restoration follow completion; a launched job is not a finished result.
