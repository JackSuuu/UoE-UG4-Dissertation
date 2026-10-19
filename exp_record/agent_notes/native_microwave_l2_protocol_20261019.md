# Native microwave L2 — frozen candidate-reachability protocol (2026-10-19)

**Recommendation: run the six-snapshot, 18-branch L2A below first (at most 576 model calls).** Establish whether this frozen actor can produce locally useful alternatives before designing an online selector. L2B is a separately gated, fully specified continuation audit; L3 is not scheduled here.

Status: **design v1 frozen before any L2 model query or simulator branch**. This work only reads saved logs and source; no model, GPU, environment, or rollout was run. The rule is development-informed, not preregistered before L1 and not held-out validation. Machine-readable snapshot identifiers, fingerprints, source hashes and budgets are in `/home/ami/scratch/native_microwave_l2_design_20261019/snapshot_manifest.json`; `analyze.py` reproduces the CPU/offline extraction and `evidence.json` contains its measurements. The sibling `design_freeze.json` binds this note and evidence by SHA-256. Any subsequent change to snapshot rule, decoding, seeds, horizons, labels or gate needs a new version before new branches.

## 1. Selection is established by L1, independently of candidate outcomes

Source: `/home/ami/scratch/native_long_l1_20261019_b/report/{results.md,summary.json}`, which combines original cohort `a` roots 0–2 and continuation `b` roots 3–5. Microwave has **4 native successes, 2 meaningful-progress failures, six completed/consistent episodes, and six exact replays** (zero state/integration gaps). It satisfies the frozen eligibility rule and is the highest-priority task. The combined report's overall `selection.provisional` flag concerns the unfinished wider screening; it does not erase microwave's completed six-case eligibility. No lower-priority task's candidate outcomes enter this choice.

| Root | Native result / actions | First grasp `g` | First mug rise >20 mm after `g` (delay) | First `In` | First `Close` |
|---|---|---:|---|---:|---:|
| 0 | failure / 520 | 73 | none | none | none |
| 1 | success / 444 | 86 | 244 (+158) | 277 | 443 |
| 2 | success / 397 | 95 | 251 (+156) | 296 | 396 |
| 3 | success / 352 | 136 | 155 (+19) | 195 | 351 |
| 4 | failure / 520 | 95 | 107 (+12) | none | none |
| 5 | success / 298 | 92 | 107 (+15) | 143 | 297 |

All `t` values are zero-based **post-action** indices; state/frame index is `t+1`. A rise is relative to the logged position at first grasp, not a stable-lift certificate. Contact statements below refer to logged endpoints unless specified otherwise.

### All-six interpretation, including the new root 4 failure

- **Root 0:** grasp remains true at `73–519`, but maximum mug rise after first grasp is only **5.12 mm**. Table contact occurs at 344/447 endpoints. The mug never attains `In`; the door never attains `Close`. The mapped z command is the same approximately `+0.000341` throughout `320–519`. This is persistent contact-defined acquisition with little vertical progress, rather than an observed lift followed by a drop.
- **Roots 1/2:** eventual successes remain near table height for a long time. Their first 20 mm mug/EEF rise is **158/156 actions after first grasp**. Root 2 has grasp losses at `115,119,146,161,202,302`, including natural reacquisitions. Consequently +40/+60 timers and isolated grasp losses cannot serve as validated failure alarms.
- **Root 3:** grasp `136–197`; `In` true at `195–197`, false at `198–313`, and true again from `314`. It succeeds at `351` without another logged grasp. This is a successful control for a substantial placement-loss interval, not just one-frame predicate flicker.
- **Root 4:** grasp intervals `95–146,151,155–156,158–229`. It lifts promptly and reaches **80.00 mm maximum rise**; it therefore contradicts a universal “microwave failures never lift” account. At `195` the elevated mug contacts both gripper pads and microwave geometry while `In` is false. Around `229–230` mapped gripper commands are `-1` (opening under the audited mapping); at `230` grasp becomes false and no mug endpoint contacts are logged. Mug z falls from **0.98004 m at 230 to about 0.90291 m at 262** (−77.14 mm); table contact appears at 26/33 endpoints in that window. At `275` it contacts table and microwave geometry, without grasp or `In`. It never reacquires the logged grasp. During `320–519`, table contact occurs at all 200 endpoints; mug motion ranges are only **3.95/16.93/1.45 mm**, while EEF ranges are **31.48/37.46/39.07 mm**. Evidence supports lift/transfer followed by loss and table-supported non-placement, followed by unsuccessful reacquisition/transfer. It does not establish the unique perception/planning/contact cause or a counterfactual benefit from suppressing opening.
- **Root 5:** `In` is true from `143`; grasp losses at `153` and `160` do not prevent success at `297`. Grasp ending after containment can be compatible with correct release. Root 3 also shows why temporary `In` loss must be checked against successful continuation.

Thus root 0 and root 4 require different capability probes: leaving persistent low-height contact versus reacquiring/transferring a previously lifted but unplaced object. The selected snapshot rule covers both without using eventual failure as an inclusion condition.

## 2. Exactly one causal snapshot per root

Process each original trajectory in chronological order, using only its current/past diagnostic prefix:

1. Latch `g`, the first post-action observation with `grasped[white_yellow_mug_1] == true`.
2. At post-action `t = g + 180`, save the branch boundary if native success has not already terminated the episode. **Collect regardless of current grasp, object height, goal predicates, or eventual outcome.** No timer reset on grasp loss/reacquisition.
3. No second snapshot, fallback timestamp, search over losses, or root-specific adjustment. If no grasp, termination before the boundary, or missing/inconsistent data prevents a snapshot, record that root as unavailable rather than replacing it.

The 180-action wait accommodates the observed 156/158-action successful lift delays. It is an event-aligned collection timer, **not a declaration of failure**, and the eventual online detector has not been designed. Current/past simulator grasp is privileged instrumentation; this study does not claim an RGB-only monitor.

| Root | Post-step `t` | State index / next actor action `s=t+1` | Remaining `R=520-s` | Snapshot context |
|---|---:|---:|---:|---|
| 0 | 253 | 254 | 266 | grasp true, table contact, neither goal |
| 1 | 266 | 267 | 253 | grasp true, lifted, neither goal |
| 2 | 275 | 276 | 244 | grasp true, lifted, neither goal |
| 3 | 316 | 317 | 203 | `In` true, grasp false, door open |
| 4 | 275 | 276 | 244 | grasp false, table contact, neither goal |
| 5 | 272 | 273 | 247 | `In` true, grasp false, door open |

The extractor checked prefix truncation: the same snapshot is returned from `rows[:t+1]`, none from `rows[:t]`; frame `s`, post-action fingerprint at `t`, and before-action fingerprint at `s` agree. This is log consistency, not a new restoration experiment.

**Restoration contract for execution:** use each root's original official initial state, seed 0, native scene, and ten original initialization actions, then replay its **recorded mapped prefix** `0..s-1`, with zero actor calls. Verify exact integration/controller fingerprints, both RGB images and diagnostics against the saved boundary. This reconstructs controller history as well as physics. A naked `set_state(qpos,qvel)` is insufficient. Reconstruction mismatch blocks that root; no approximate-reset substitute. Prefix actions are recorded native policy actions, not new hand-authored motions. Charge them separately as reconstruction work, not again against the branch's remaining actor budget.

## 3. Candidate bank: two sources, three fixed branches

Each of six snapshots gets exactly these three branch identities:

| Candidate | Source for local actions 0–31 | Request seed at local offset `k` |
|---|---|---|
| `C0` | nominal greedy | 0 |
| `C1` | stochastic sample | `20261019 + 10000*root + 100 + k` |
| `C2` | stochastic sample | `20261019 + 10000*root + 200 + k` |

Shared model/interface: frozen `openvla/openvla-7b-finetuned-libero-10`, revision `80970322773f81baa2e22fe495d0487b93a05cfa`; official base code revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`; TF **4.40.1**, BF16, SDPA, eval/inference mode, batch 1, normalization key `libero_10`, original native camera preprocessing and official gripper mapping. Full unchanged instruction:

> put the yellow and white mug in the microwave and close it

Greedy: `do_sample=False`, one beam/one return. Sampling: `do_sample=True, temperature=0.7, top_p=0.95, top_k=50, num_beams=1, num_return_sequences=1`. Explicit `top_k=50` fixes the TF4 default inherited by the inspected sampled worker. Official `predict_action` itself requests the seven action tokens; do **not** also supply `max_new_tokens` through kwargs (duplicate argument). Log the resolved generation configuration and all seven returned tokens/native/mapped action coordinates. One generation attempt per call; no token rejection sampling, temperature sweep, redraw on duplicate, candidate replacement, or outcome-driven retry.

**Lock the semantics:** these are **fresh closed-loop model branches**, not a frozen first action held/repeated for 12 or 32 steps. All three first calls receive byte-identical original snapshot policy PNG and identical instruction/prompt. Thereafter each branch makes one new model call on its own freshly observed RGB before every action, retaining the same full instruction and fixed source/seed schedule. No saved future observation, rewritten subgoal, privileged-state token, synthetic image, action interpolation, manual recovery trajectory or hand-motion script reaches the actor. Observations legitimately diverge after actions diverge. Same-observation action-diversity claims apply only at the common boundary (or subsequently proven identical observations), not to arbitrary time-aligned divergent branches.

There are **three candidate trajectories total per snapshot**, not three proposals per timestep and not a growing search tree. Each branch executes its sole model proposal directly. Validate the audited seven-coordinate finite action contract and mapping before execution; record invalid generations as unusable candidate cells without clipping, redrawing or scripted fallback. Infrastructure failures remain distinct from valid-but-unhelpful proposals. Count requested calls even if interrupted or rejected.

## 4. L2A: nested 12/32-action reachability experiment

- Restore original boundary independently for each `C0/C1/C2`; run **at most 32 new actor actions**. Measure after action counts **12 and 32 on the same trajectory**. Twelve is a nested readout, not another intervention duration or independently sampled branch. There is no 12-action handoff arm in v1.
- Stop early only at native full conjunction, invalid generation, infrastructure failure, or the fixed action/query/wall cap. Native success is absorbing for endpoint reporting; report actual actions, rather than physically executing fictitious post-success actions. If success is first observed after 12 but before 32, it does not alter the earlier 12-action label.
- No branch stops or changes source because a proxy looks good/bad, grasp is lost, `In` toggles, or another branch succeeds. No selector runs.
- Run all six `C0` local controls first, then `C1` and `C2` in root order. C0's newly generated tokens/actions and resulting local tape must match the original nominal suffix up to its stopping boundary; action replay alone does not establish model re-query identity. A mismatch is a model/runtime reproducibility issue, not repair benefit; stop before interpreting source differences.
- Fresh-process, zero-query action replay checks each completed local tape, including boundary reconstruction, exact physics/integration/controller/images and diagnostic/label recomputation. Keep all 18 planned cells in the report; missing, invalid, censored and mismatched cells are not ordinary native task failures.

### Offline labels — frozen now; never selector inputs

Record native `In`, native `Close`, native conjunction, grasp, positions, endpoint/substep contacts, and executed commands separately. Native success remains the task endpoint. The following proxy only makes short-horizon capability legible:

`held_clear(d)` means current target grasp is true, mug z is **>20 mm above that root's original first-grasp z**, and no mug–table endpoint contact is logged. It is a measured lifted-acquisition proxy, not stable support or continuous contact certification.

Snapshot reference `b`: **2** if `In` is currently true; else **1** if `held_clear` is true; else **0**. The six reference values are `0,1,1,2,0,2`. At horizon `h`, define `p_h` as the highest applicable value:

- **3:** native full success has occurred by `h` (absorbing endpoint).
- **2:** `In` is true at each of the final **8** post-action observations ending at `h`.
- **1:** `held_clear` is true at each of those final **8** observations.
- **0:** none of the above.

Define **`U_h = (p_h > b) AND no goal predicate that was true at the snapshot became false in the branch prefix through h`**. Also report each component, first attainment times, final level, any intermediate losses, and actual observation count. Except for native early success, fewer than `h` observations makes `U_h` censored/unknown, not false. Acquisition/placement labels do not impose mandatory task stages: direct native success receives 3 even without observed grasp. For a contained mug, retaining `In` is a preservation result; merely moving it is not useful progress. Lift/`In` dwell thresholds are fixed exploratory proxies and do not redefine native completion.

On the **original recorded nominal suffix**, these labels give `U_12=false` for all six; `U_32=true` only for roots **1,2,5**. Root 5 reaches native success after 25 further actions; root 3 needs 35 and therefore has `U_32=false` despite successful near-term continuation. Root 4's mug displacement over its original next 32 actions is approximately `(−0.0002, −0.0172, +0.0004)` mm, versus root 0's `(3.59, −0.57, +0.72)` mm. These are **reference tape measurements, not L2 model-branch results**. They illustrate why a negative 12/32 proxy is not proof of no longer-horizon recovery.

Report boundary action/token uniqueness and duplicates, valid candidate count, the full `6 roots × 3 candidates × 2 horizons` table, original/fresh nominal agreement, per-root sampled proxy gains over C0, and control predicate losses. Retain duplicates in their preallocated cells; no free extra draws. Summaries of “best available candidate” are explicitly **offline oracle availability**, not an executable policy or framework success rate. Neither U, future simulator states, full-success labels, nor the identity of a successful branch may be supplied to a runtime selector/actor or used to substitute its executed action. L2 has no runtime selector at all.

## 5. Precommitted L2B continuation gate and budget

**Pause after the complete L2A audit.** Proceed to L2B only if all six greedy controls reproduce and required local replay checks pass, and at least one sampled branch on original failure root **0 or 4** has `U_32=true` while its C0 has `U_32=false`. This is an experiment-level capability gate, not action selection. Otherwise stop v1 and report source diversity, physical changes and the bounded negative result. No new seeds or timestamp changes within v1.

If the gate is met and continuation execution is scheduled, continue **all 18 preallocated branches**, including both failures and all four successful controls; do not shortlist proxy winners. Restore each branch's H32 state by replaying its original prefix plus its own saved local action tape, verifying its fingerprint before any query. Branches already natively successful need no further actions. Every remaining branch hands off at exactly local action count 32 to **fresh greedy H1 model calls**, same checkpoint/interface/full instruction, until native success or the **original 520-action cap**. No second local episode, resampling, reset/regrasp routine, or post-success dwell. There is at most one continuation per candidate, consuming at most `R−32` further calls/actions. Preserve source and handoff attribution in the tape. Verify completed continuations with fresh zero-query action replay.

L2B reports `Y(root,candidate)` = native success within that remaining budget, together with censoring and physical diagnostics. A recovered failure requires `Y(Ci)=true` and fresh `Y(C0)=false` at the same snapshot; nominal-success control damage is `Y(C0)=true, Y(Ci)=false`. Report all 12 sampled cells and both failure roots, without presenting oracle maxima as achieved deployment performance. This design tests late-state reachability. Root 4 may already require more than a 32-action local intervention by step 275; a negative result cannot rule out earlier prevention or another bounded source.

## 6. Exact resource bounds and implementation reuse

| Resource | L2A | Additional L2B if gate passes | Combined upper bound |
|---|---:|---:|---:|
| Snapshots / candidate identities | 6 / 18 | same identities | 6 / 18 |
| Model calls / newly executed model actions | 576 | 3,795 | **4,371** |
| Generated action tokens (7/call) | 4,032 | 26,565 | 30,597 |
| Model-guided local length per branch | 32 (12 nested) | 0 | 32 |
| New prefix construction + evaluation simulator actions, excluding initialization | 5,565 | 9,360 | 14,925 |
| Independent recorded-action replay work, excluding initialization | ≤5,565 | ≤9,360 | ≤14,925 |

Calculations: `sum(s)=1663`, `sum(R)=1457`; L2A `3*1663 + 18*32 = 5565`; L2B reconstructs prefix+local and follows at most the remainder, `18*520 = 9360`. Reconstruction/replay requires **zero model calls**. Up to 72 fresh scene constructions across both phases and their verification add **720 labeled initialization actions**. Counts are deliberately conservative: early native stopping saves actions/calls (e.g. the original root-5 C0 local branch stops after 25), never creates replacement draws. There is no extra calibration-query allowance; greedy identity checks are counted C0 calls.

Execution caps: worker readiness **180 s**; each requested generation **30 s**; each L2A branch **120 s** and each L2B continuation **300 s**, including scene/prefix work but excluding worker load; each independent replay **120 s**; each phase's whole job **7,200 s** including load/replay. Zero automatic retries; interruptions consume attempted calls and retain artifacts. These operational stops yield explicitly censored/infrastructure cells. A phase cannot bypass its model/action cap to compensate for unused cells.

Observed L1 episode-wall cost is approximately **0.342 s/action** weighted over the six cases. Rough planning estimate: L2A's model/step core ≤**3.3 serial minutes**, plus prefix reconstruction, source loading, logging and replay; budget about **8–15 minutes** for the small experiment. L2B adds up to **21.7 serial minutes** at that rate, plus reconstruction/replay (roughly **30–45 minutes** additional end to end). These are planning estimates, not measured sampled-worker performance or guarantees. Persistent batch-one inference can reuse one loaded model; no actor training or new checkpoint is needed.

Inspected reusable components:

- `scripts/experiment3_candidate_worker.py`: existing official TF4 sampled/greedy worker, per-request `torch.manual_seed`, `temperature=.7/top_p=.95`, seven-token capture, same gripper mapping; `experiment3_candidate_run.py:CandidateActor.query` records the policy PNG hash. Reuse that bounded source interface, explicitly fixing top-k and seed/generation metadata.
- **Adapt device setup before use:** the old candidate worker/launcher hard-code both CUDA and EGL to numeric `2`; current audited native lanes map physical GPU 1→EGL2, 2→EGL1, 3→EGL0. Use the native episode's UUID/EGL preflight and pinned snapshot/code manifests, not the old launcher's device assertions. Nothing was launched here.
- `scripts/run_native_long_episode.py`: native task initialization, diagnostics, tapes, fingerprints, original action replay and full-goal stopping. Reuse the saved root-specific source manifests and verify restore fidelity.
- `experiment3_candidate_gate.py` is book-specific (600-call shared cap, early stagnation heuristics, 12-action local owner, novelty/continuity/release checks, strict-completion monitor). **Do not transplant it into L2.** It would confound source reachability with a selector, reject legitimate release, and import a different endpoint. `experiment3_candidate_audit.py` supplies useful query/action attribution and duplicate-accounting patterns; its book-specific assumptions need replacement.

Before execution, a launcher must freeze its exact implementation/configuration and hashes against this design, including observation PNG/prompt hashes, seed schedule, checkpoint/base-code/runtime sources and all six boundary fingerprints. The existing scripts are reusable parts, not an already implemented microwave L2 runner.

## 7. Result-driven next decision

1. **Now:** implement/freeze and execute **L2A only**, with all six roots and 576-call cap. This note is the protocol; no framework-gain claim exists yet.
2. **No distinct samples:** report candidate-source collapse at the common observation; do not tune a verifier over duplicates. **Distinct but no failure-root U32 gain:** stop v1; report the tested source/horizon limitation. A separately frozen earlier-event or longer-local-horizon study is a possible next decision, not a hidden extension of this bank.
3. **Local gain on a failure:** run the fixed **all-bank L2B** audit. Local lifted/contained motion alone is not recovery. If no sampled branch yields full native recovery over fresh C0, remain at capability diagnosis rather than proceed to L3.
4. **At least one verified full failure recovery plus reproduced nominal controls:** candidate reachability is established on this development bank. Then preregister a small L3 comparison with identical candidate budgets and a deployable current/past-evidence selector versus a shared-candidate feedback control. The oracle identities and future labels remain evaluation-only. Any such L3 is a new protocol; this six-root reused bank is not held-out confirmation.
