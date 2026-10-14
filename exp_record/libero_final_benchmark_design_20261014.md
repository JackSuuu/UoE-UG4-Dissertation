# LIBERO final benchmark: clean-task preservation and controlled recovery

## Authoritative scope correction after D1

The user explicitly authorizes **controlled perturbation experiments** and asks for a final benchmark capable of testing framework intervention and task benefit. The five-layer architecture remains fixed. Clean VLA3/3on D1is a clean anchor, not a reason to abandon the task or stop the research. D1's original no-headroom stop remains a correctly executed historical protocol. This new authorization supersedes its suggested return-to-book-only next step: continue the **drawer task with controlled release faults**, then frozen-method transfer and the final clean/stress benchmark.

The research claim to establish is: **an event/phase-aware guardrail around the same frozen VLA improves complete-task recovery under specified execution faults, beyond a matched reactive geometric controller, while preserving clean-task performance.** A positive outcome is a hypothesis to demonstrate, not a metric to engineer by selecting only favorable final cases. Use recoverable physical faults, paired cases, explicit explanations and independent final roots.

### Evidence needed for an efficacy claim

1. Actual measured fault or deviation, timely observable monitor response, a changed executed action, and completion of the remaining task—not only a successful detector score or a restored subgoal.
2. Higher paired full-task success than the **same frozen actor** and the strong simple geometric shield under identical fault schedules/sensing/control caps. Report rescues and harms, including failures and non-triggering cases.
3. Clean-task paired non-inferiority/no-harm assessment, including uncertainty. Clean3/3is not this assessment.
4. Detector substitutions share the same repair backend; repair ablations share the same physical detector. This separates useful sensing, event timing and correction behavior.
5. Freeze implementation/thresholds/case rules before opening final roots. A physical-state version is explicitly privileged; a learned or image-only claim requires the actual estimator/model implementation.

## 1. Final environment matrix

Only LIBERO is needed for the main online benchmark. Historical planar simulations remain developmental evidence, not an additional primary benchmark. The pinned registry is `libero_10`; LIBERO-Long is a common literature label, not a second installed suite here.

| Block | Tasks / conditions | Question |
|---|---|---|
| Clean no-harm | Three preregisteredSpatialand threeObjecttasks, plus clean long-task anchors | Does the plugin preserve a task-capable actor and avoid unnecessary/harmful corrections? |
| Long-task completion | Book→caddy and bowl→bottom-drawer→close, unperturbed | Does physical completion survive the task's remaining actions? |
| Controlled execution faults | Those same two tasks, sham + two release-delay severities + two lateral-command-pulse severities | Does the framework recover physical relations and finish, beyond nominal VLA/simple control? |

Fixed clean inventory, selected by registry order before their outcomes are observed:

- Spatial: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`; `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`; `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- Object: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`; `pick_up_the_cream_cheese_and_place_it_in_the_basket`; `pick_up_the_salad_dressing_and_place_it_in_the_basket`.
- Long: the existing exact book/caddy and bowl/drawer task strings in the respective protocol files.

This is an **eight-task selected LIBERO benchmark**, not the official all-task LIBERO score or a reproduced LIBERO-Plus/RoboRecover benchmark. Keep suite/task results separate. Per-task accepted task-completion definitions and allowed contacts must be audited before use; do not transplant the book's box criterion to every object.

The current cached actor is LIBERO-10-finetuned. Spatial/Object require capability checks and immutable **suite-matched** official checkpoint/statistics references. Freeze one actor per suite and use exactly that actor across every method within its suite; name the different weights explicitly. Poor transfer from the10checkpoint cannot be presented as the clean Spatial/Object baseline. No actor fine-tuning is part of this experiment.

### Split and bounded evaluation size

Each task's official initial-state index is task-specific. Preserve the already consumed book0–5and drawer0–2as development. For new cases:0–5development,6–9calibration,10–19reserved training only if a learned component is later justified,20–49final confirmation. Existing reservations are respected; no final labels enter tuning. No learned model is currently scheduled.

Planned final case matrix, once the method passes development:

- CleanSpatial/Object:6tasks ×30roots =180case identities.
- Long clean/stress:2tasks ×30roots ×5conditions =300case identities (clean counted once, not duplicated in the other block).
- Core four methods on480cases =1,920episodes.
- Two detector-substitution comparators on the300long clean/stress cases =600additional episodes.
- Total upper planned core/attribution workload:2,520episodes; launch in detached task/suite batches only after measured throughput and storage estimates. No large final sweep now.

Repeated severities on one root are correlated: confidence intervals resample root clusters within tasks, never treat frames or all perturbation variants as independent trials. Report tasks separately and macro averages. Six clean tasks supply180root/task pairs but only six task identities; no claim of generalization to an arbitrary task population. A2ppno-harm margin must be accompanied by its uncertainty; sample size is not a guarantee of passing it. For perspective, with independent cases and zero harms, at least149cases are needed merely for a one-sided95%binomial upper bound below2%; clustering and observed discordance can demand more. Power planning uses development/calibration only, before any final outcomes.

## 2. Fault design and next concrete experiment D2

### D2: delayed release at the contact transition

Keep the drawer scene and canonical task instruction. First release opportunity is the first nominal open command while the previous measured interval has bowl/gripper contact, the bowl reference point is in the drawer region, and the drawer is not closed. This is an **injection opportunity**, not a success/safety label. It intentionally does not wait for1sdwell or a stable-support event; D1root2showed that those can arrive too late.

From the already verified D1clean prefixes, the causal anchor decisions are **169,147,163** on roots0/1/2. `scripts/prepare_libero_stress.py` freezes all nine development case identities, including sham/4/8control-interval release delays, without consulting method outcomes. Durations correspond to0/0.2/0.4seconds of simulated actuator latency at20Hz; they are development values, not asserted hardware measurements or finalized calibration choices.

`ReleaseDelay.apply` acts **after** policy/gate/repair selection: during the registered interval, the physical gripper command remains close (+1), while all six selected pose coordinates are unchanged. At expiration it passes the selected command through. This subjects VLA and repair actions to the same physical actuator defect. Log proposal→selected command→faulted effective command separately. A method cannot bypass the fault by replacing the nominal actor action. The fault active bit, duration, anchor ID and future recovery label never enter the monitor/controller; only ordinary measured geometry/contact/proprioception can reveal delayed detachment.

**Injector preflight passed:** `scripts/smoke_libero_stress.py` replays root0's full actual169-action prefix independently for sham/4/8delay branches. All starting integration/controller/camera hashes match; sham reproduces the next eight nominal intervals exactly; nonzero delays produce different physical states with the specified gripper command and unchanged selected6Dpose. With recorded nominal actions as test stimuli, the8-delay branch retains bowl/gripper contact throughout those eight intervals, whereas sham detaches during that window. [Regression report](../backups/libero_stress_smoke_20261014.json). This validates a physical delay mechanism, **not** live-policy failure/recovery or complete-task efficacy; no future actor actions were queried in this regression.

For the initial mechanism experiment, replay the identical clean action prefix from the original root, verify full integration/controller/camera hashes at the anchor, and continue each arm with real VLA queries on its own actual observations. This is a **matched-prefix recovery experiment**. All arms begin the intervention-enabled continuation at the same physical state; sham cases use the identical setup. Include prefix length in the same600step total budget. This restriction is common to every arm and is not represented as unrestricted start-to-finish event timing. The later final online study must separately specify whether faults are fixed-time or causal event-triggered; retain no-event episodes and report realized injection rates rather than excluding them.

Targeted gate change to test: enter a release-pending transaction at the observed release opportunity; suppress unsafe pose continuation while contact persists; use measured separation/release history to transition to bounded outward clearance and return control for closing. The monitor must not require already-detached stable placement before managing release. Compare against same-sensing hold/open and fixed timed repair. This is a new layer3–5version, not a silent change to D1. All actual decisions must have the existing pre-action evidence→predicate→mode→action→outcome proof.

**D2go/no-go differs from D1:** clean success does not cancel the stress cases. Execute the registered matched stress/sham cases, including harmless disturbances and failures, rather than stopping when clean baselines succeed. Check physical validity, common-prefix identity and finite bounded commands first. Report saturation (all methods fail), triviality (all succeed), conditional recovery and harms; development revisions are versioned and retained. Never choose a final severity because it maximizes the ours-minus-baseline gap.

### Second fault family after the release mechanism works

Brief lateral action-command bias during approach/insertion. Development candidate magnitudes correspond to2–3cm **cumulative requested** displacement using the audited native OSC scale; actual object/EEF displacement must be measured. Signed axes, duration, seed and saturation rule are fixed before comparisons. Do not call a3cmOSCrequest an exact3cmobject translation. No state teleportation, interpenetrating object relocation, hidden-state reset or unrecorded controller reset.

Calibration roots establish that faults are finite, physically valid and sometimes recoverable without access to our final results. A scripted recovery feasibility check is explicitly privileged, separately labeled and never the baseline score. Freeze mild/moderate magnitudes after calibration, retain the clean condition, and publish all severity cells. No per-method intensity adjustment. Do not mix several faults until individual-family effects are understood.

## 3. Baselines and attribution matrix

| Method | Detector / timing | Response | Interpretation |
|---|---|---|---|
| Frozen VLA | Actual checkpoint's native cadence | Original action | Currently **H1closed-loop**, one query on each new image; not native open-loop chunking |
| Reactive geometric shield | Same measured physical event | Fixed bounded hold/retract rule | Strong simple same-information comparator; not a CBF implementation |
| Event suspension control | Our event monitor | Hold pose with explicitly specified gripper behavior | Tests bounded suspension; opening the gripper is an action and must be disclosed |
| Ours | Phase-aware physical monitor + dual-mode gate | Bounded local repair + measured handoff | Core complete-task recovery claim |
| Temporal action-jump + shared repair | Previous/current native7Daction changes, translation/rotation/gripper scored separately | The **same repair backend and cap as ours** | H1-compatible detector substitution; inspired action consistency, **not reproduced ACC** |
| Feature-distance + shared repair | Frozen feature distance to a separate reference success distribution | The **same repair backend and cap as ours** | Detector substitution; feature layer, normalization, reference set and threshold frozen before test |

Before the final core run, D2retains the fixed10hold+30retract control to distinguish state-conditioned release/exit timing from a simple timed primitive. This is a targeted development ablation, not a new architecture. Additional within-method removals (no phase-aware return, no measured-history debounce) are performed one at a time only when they test the demonstrated mechanism.

Feature-reference fitting (including covariance/normalization) uses designated reference/training roots, not the threshold-calibration roots or final roots. Thresholds are calibrated separately on6–9with episode/root-level false-alarm reporting; temporal frames are not independent calibration cases. The pretrained actor's own training exposure is outside this guardrail split, so "held-out roots" means held out from guardrail development, not proven unseen by OpenVLA pretraining/fine-tuning.

Key baseline corrections:

- NativeH1already replans on every new observation. Requerying a deterministic actor on the same observation is not an independent control. Fixed-Kaction repetition can be an explicitly named **cadence/action-repeat ablation**, but weakening the baseline cannot establish repair-specific benefit.
- Genuine overlapping action chunks are required for VLA-FAILACC. Do not create chunks from future observations or call H1jumpsACC. If a genuine chunk-capable frozen policy is introduced later, run its actor-only baseline and all matched arms in a **separate actor/cadence block**; do not mix it with this H1table or silently replace historicalH10/k5results.
- ACTtemporal ensembling is an action aggregation mechanism, not automatically a failure detector. A handcrafted hard stop is not a control-barrier-function method. A generic embedding distance is not automatically V-JEPAor VLA-FAILLLMD.
- True literature reproduction needs the published score, feature extraction, calibration, thresholding and implementation checks. Otherwise use the transparent descriptive/inspired name above. Selecting a20-line approximation does not reproduce an entire paper.
- In the initial privileged-state block, the reactive shield and repair/controller ablations have the same simulator-state sensing access and phase information as ours. Action-only/feature-only detectors are explicitly **input-restricted ablations**, not evidence that the algorithm wins with equal modalities. A deployment/learned claim requires an estimated-observation block with actual sensing errors and latency. Do not hide extra geometry behind the phrase "same backbone".

## 4. Metrics with unambiguous denominators

### Primary outcomes

- Native task SR and separately registered physical full-task SR; episode is the denominator, including all non-triggering/no-injection cases in the intent-to-evaluate table.
- Paired ΔSR, rescue and harm counts versus VLA **and the strong shield**. Cluster uncertainty by task/root; all severity variants travel together. Report each task/severity and predefined pooled weighting.
- Clean harm rate and paired ΔSR with uncertainty; target no more than2ppclean SR loss. Preserve the historical efficacy target of≥+3pppooled OODsafecompletion with a positive paired95%lower bound only for a future **validated safe** endpoint. This current guardrail study can claim physical completion gain, not substitute it for safe completion.

### Recovery outcomes

The user-proposed success-among-triggered ratio is useful but can be gamed by rarely triggering. Report all of:

1. **Fault-case completion:** successful whole-task episodes / all independently valid injected cases (same case denominator across methods).
2. **Baseline-failure rescue:** ours successful among matched VLA failures / matched VLA failures; alongside harms among VLA successes. This is a conditional descriptive contrast, not selective removal from the main table.
3. **Trigger-conditioned recovery:** successful whole-task episodes / truly faulty cases that triggered; report numerator, denominator, missed-fault count and zero-denominatornull. Re-establishing a subgoal without closing/completing is not recovery.

An imposed actuator fault is not automatically an irreversible failure. Case eligibility, observed physical deviation and ultimate task failure are separate labels. No historical offline0.909value is imported.

### Detection timing and quality

Use independently defined, phase-specific physical events and annotated/validated onset times. For lead-time scoring, run detectors in **shadow mode on a shared uncorrected perturbed rollout** and compare alarm time with that rollout's event time. In a repaired rollout the harmful event may never happen; do not invent its counterfactual timestamp. Report detection coverage, miss rate, early false alarms and lead-time distribution together. A detector that never alerts does not have zero-latency perfect performance.

Use the VLA-FAILpublished **AUCPDTdefinition and reference evaluation** if reproduced; otherwise report clearly named precision/recall versus time-to-event curves. Do not relabel a homemade AUC asAUCPDT. No historical19-step lead is a LIBERO result without matching source artifact/task/event/cadence provenance.

### Physical contact outcomes

Freeze contact-pair×task-phase allowed/forbidden categories and onset/debounce rules on calibration cases. Necessary grasp, drawer-handle actuation and bowl support are allowed contacts. Report episode-level unintended-contact incidence and event counts per simulated second; numerical sensitivity is checked on near-contact fixtures. Damaging-contact/force thresholds require their own justification. Until these checks exist, Collision Rate/CVR/safe success are null, not zero. A contact count reduction cannot replace task success.

### Cost and observability

Record per-step actor, measurement/estimation, detector, gate/repair and total wall time; CUDA timings require synchronized boundaries/events. Report warmup separately, median/p95latency, CPU/GPU/model, actual policy calls, intervention ratio and added simulation steps. Compare with the same runtime settings. Do not claim milliseconds,74ms, realtime or cheaper-than-VLMuntil measured online. A20Hzsimulator does not mean the7Bcontrol loop runs at20Hzwall time.

Every intervention retains measured preconditions, numeric thresholds, consecutive counts, route/reason, remaining budget, nominal/selected/effective commands, post-action state and handoff evidence. Fault metadata is stored in evaluator-only logs alongside, not fed to the policy or explanation builder. Videos show actual interventions, successful recovery and harmful/failed interventions; methods are named correctly.

Framework intervention is `selected != nominal`; fault actuation is `effective != selected`. Never credit external disturbance-induced action changes to the repair module. Physical replay executes the effective command, while the causal gate proof reconstructs the selected command using only pre-action ordinary measurements.

## 5. Final deliverables and experiment order

Final report tables:

1. Clean no-harm: task/suite, VLAand methodSR, paired harms/Δ/CI, clean intervention incidence, overhead.
2. Long controlled recovery: method × task × fault/severity, whole-taskSR, valid injected cases, recovery denominators, rescues/harms, contact outcomes and cost.
3. Detector/backend attribution: shared repair with action/feature/physical gates, and shared physical monitor with shield/suspension/full repair. Detection quality is separate from complete-task gain.
4. Failure audit: trigger coverage **and timing**, recovery-budget exhaustion, release/alignment/handoff failure, later-stage task failure; representative matched videos and observable chains.

Execution order is fixed: D2single-family release-delay mechanism with explicit traces → measured physical headroom against simple controls → freeze/calibrate the monitor and add the two detector comparators → frozen-method second-task transfer and clean-task capability audits → one final held-out benchmark. Architectural layers do not change. No large final benchmark, additional model training or unrelated environment expansion is launched now.

## 6. Literature/provenance check (14 Oct 2026)

- [LIBERO-Plus official repository](https://github.com/sylvestf/LIBERO-plus), [paper2510.13626](https://arxiv.org/abs/2510.13626): seven dimensions including layout, cameras, initial robot states, language, lighting, background and sensor noise. This supports robustness benchmarking, but does **not** establish that every cited project uses the same contact-frame actuator injection.
- [VLA-FAIL2606.21386v2](https://arxiv.org/abs/2606.21386v2): LLMDlast-layer token-feature distance; ACCoverlapping-chunk consistency; AUCPDTjoint detection/timing metric. The actual nativeH1adapter does not meet ACC's chunk-overlap assumption.
- [RoboRecover2609.28952v1](https://arxiv.org/abs/2609.28952v1): execution-induced deviation states reconstructed by action-prefix replay,2,000scenarios across RoboTwin/LIBERO and fixed platform-specific800/200splits. It motivates a paired-prefix recovery protocol; our own cases are not its benchmark score.
- [VLA-FixBenchICML2026record](https://icml.cc/virtual/2026/poster/64203): fault dataset and VLM–VLAdiagnosis/recovery evaluation. Its reported expert upper-bound improvement is an external result, not our baseline or a reproduced controller.

A targeted search of current tracked experiment Markdown did not establish the user-mentionedV-JEPA74ms/AUC0.727vs0.996, online recovery0.909,19-step lead or task0's80%as the claimed LIBERO evidence. Some matching numbers occur in unrelated historical records. Treat those claims as **unverified provenance**, not false by assumption and not numbers ready for the thesis. Retrieve original artifact/task/split/metric definitions before citing them.
