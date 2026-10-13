# Week 4 (12–18 Oct 2026) — Verifier reliability before simulated-arm transfer

Newest entry at the top. Index and conventions: [README.md](README.md).

---

### Update 13 Oct — Continuation complete: released placement1/3; alignment and re-contact failures identified

[Verified results and trajectory audit](libero_continuation_results_20261013.md). Root0passes at313steps; roots1/2reach520without completing. All original-prefix tokens/actions/states agree exactly;25substeps/interval, geometric conditions/dwell and video lengths verified. Root1never fits the full footprint and remains in gripper contact after grasping. Root2passes all conditions for6intervals, then renewed gripper contact interrupts stability. Root0establishes one detached seated placement but not intentional gripper opening or safe retreat. Next mechanism design must compare physics-guided repair against simple matched phase/termination controls; no learned gain or safety rate claimed. All jobs complete, no new intervention/training launched.

---

### Update 13 Oct — Full-policy continuation launched with geometry/release/dwell acceptance

[Protocol](libero_continuation_protocol_20261013.md), [implementation and checks](libero_continuation_launch_20261013.md). Protocol/source `9f39e2b` pushed before three detached workers launched; same roots/actor/control and520total policy steps. Native success no longer terminates the rollout.5tests pass,185step original replay matches every state exactly with substep instrumentation. A separate synthetic placed-book fixture passes60intervals, demonstrating metric satisfiability only. Its native predicate is false, so geometry-aware completion must be presented as an alternative task acceptance rather than a nested benchmark metric. Full-policy outcomes pending at this entry; no physics intervention or training launched.

---

### Update 13 Oct — Pilot completed: native3/3, but release/stable placement not established

[Results and source-interpretation erratum](libero_baseline_results_20261013.md). All three workers completed, native success at185/229/166steps; artifact verification passed and a local HTML video report was produced. Terminal commands remain closed and book/gripper contacts persist in all three. Deeper predicate tracing found `SiteObjectState.check_contact` always returns True, so this task does not require actual caddy contact despite the generic `In` call. The frozen protocol's contact interpretation is corrected in the results, without rewriting the protocol. Next is a bounded continuation/release/stability baseline protocol, before physics intervention or learned training. CVR/safe/strict precision remain unmeasured. No jobs remain running.

---

### Update 13 Oct — LIBERO full7B baseline integrated; three complete pilot episodes launched

[Implementation and checks](libero_integration_20261013.md), [frozen pilot protocol](libero_baseline_protocol_20261013.md). Environment and native-model one-action smokes passed; full checkpoint strict loading and exact upstream image preprocessing verified. Project-local adapters resolve two MuJoCo3.14/robosuite1.4.1 ABI changes.12unittest checks and standalone TF5/action-map checks passed. Protocol/runner commit `bf7c6c6` pushed before launching three fixed development initial states on healthy GPUs1/2/3 with detached `setsid nohup` workers. Pilot root `/home/ami/scratch/libero_book_pilot_20261013_a`; completion/results pending at this entry. No learned training/intervention launched. Actual collision geometry and predicate limitations recorded; native SR must remain distinct from stable precision and safety.

---

### Update 13 Oct — Scope fixed to local VLA; precision tasks and noise treatment specified

The user clarified the broader physical-verification vision and then explicitly limited current work to **local VLA execution**; LLM planning/edge-cloud collaboration are future work. [Scope and task decision](local_vla_scope_20261013.md) separates target-simulator training data from held-out evaluation, physical state/uncertainty estimation from “removing noise,” and full-task benefit from model fine-tuning alone.

Checked official BDDL for book-in-caddy (LIBERO-10, In predicate), bowl-in-drawer-and-close (In+Close), and bottle-on-rack (On, not verified tight insertion). Recommend book-in-caddy as the first precision-oriented candidate, drawer composite second, square-nut placement as a later controlled mechanism task. Actual geometry/force/precision acceptance still needs local validation. Checked public full7B LIBERO-10 checkpoint revision `80970322773f81baa2e22fe495d0487b93a05cfa`. Drawer-opening remains an optional integration smoke, not the final precision demonstration. No benchmark installation, training, rollout or LLM implementation started in this scope update.

---

### Update 13 Oct — User confirms task-first acceptance; target-simulator feasibility audited

[Decision and source audit](simulation_task_first_20261013.md): stop making another toy-task fine-tuning variation the immediate milestone. First establish a complete single-task robot-simulation policy baseline, then test bounded privileged physical intervention on the same task, then demonstrate learned efficacy. This revises execution priority without erasing negative results or treating oracle gains as learnability.

Verified official LIBERO-Goal task `open_the_middle_drawer_of_the_cabinet` and the public task-suite full7B checkpoint (HF revision `fa5ae1e7509348889295bba8e08621d8b55e9baf`). This is the first integration target, not a measured task result or a certified precision metric. Local MuJoCo3.14.0 exists; LIBERO/robosuite and task weights do not. Existing planar H10 adapter is incompatible with the7D Panda OSC action convention; official H1evaluation can only be a feasibility baseline, never a matched H10/k5 verifier comparison. Dependency pins/native TF5decode require checks before additive-only setup. No new installation, weight download, training or rollout launched during this audit.

---

### Update 13 Oct — R5 study complete: all event gates harm policy-relative safe completion

[Completion report](immediate_gate_results_20261013.md):24collection batches,3paired training seeds and180evaluation batches/5760trajectories complete; all workers exited. Independently verified48collection artifacts,180evaluation NPZs,40baseline replay batches, episode/live-risk/trigger reconstruction, source hashes, disjoint roots and cadence/query accounting. Aggregate SHA256 `20b19983a239ed8795d39f4c6aa019752158b2a98198ef8cd165deebb51dadcd`.

OOD n448/arm: policy SR/CVR/safe87.05/29.24/65.63%; frozen84.60/28.79/64.73%; event s0/s1/s2 safe54.91/53.79/57.81%, CVR39.51/40.85/36.61%. Event vs policy Δsafe−10.71[−13.62,−7.81]/−11.83[−14.96,−8.93]/−7.81[−10.27,−5.36]pp, rescue/harm3/51,1/54,2/37. Event vs matched regression+1.34[−2.01,+4.91]/+3.35[−0.45,+7.14]/+4.24[+0.67,+7.81]pp: point direction passes, all-seedCI and policy-relative screens fail. ID also regresses.

R5 support passed, but120/128positive training decisions and all62dev positives came from handcrafted brake-resumption. Both frozen gates already detect62/62dev events. Event recall100%versus regression88.7–91.9%does not demonstrate coverage of old-gate misses. Most safe harms include new violations rather than just completion loss; trigger frequency alone is not an adequate explanation. Close this configuration without promotion, tuning or independent confirmation. No new experiment launched; the following launch/progress entries are historical.

---

### Update 13 Oct — R5 study launched on three healthy GPUs; architecture review completed

Production `immediate_gate_20261013_a` launched at15:49UTC from preregistration/implementation commit **`a4b468e`** (pushed before launch). Detached coordinator PID675479; collection workers675483/675484/675485 on healthy GPU UUIDs `3f4f72ef…` / `99b39a1b…` / `c164f9d9…`. Verified coordinator SID675479 and stdin `/dev/null`; all3workers were alive in full-OpenVLA loading at the initial check. This is a launch statement, not a support/training/efficacy result. The coordinator automatically enforces collection→support→training→evaluation→integrity; support failure stops it before training.

Subsequent startup verification: all3 GPUs reached full-actor inference (~15.9GB each); independently checked the first12completed batches/24artifact hashes, applied-prefix labels against actual live risk, previous-violation masks and frozen code hashes. The first32roots/split already show that immediate positives are concentrated in the brake-resumption collector (train32,dev31,calibration30roots); natural/frozen positives are0/0/1roots respectively. This is **collector-induced R5 support**, not evidence of natural H10 anticipation or deployed gain. Collector-stratified reporting and the closed-loop validation are essential; no protocol is changed in response.

```bash
setsid nohup /home/ami/miniconda3/bin/python -u scripts/run_immediate_gate.py \
  --run immediate_gate_20261013_a \
  > /home/ami/scratch/immediate_gate_20261013_a.log 2>&1 < /dev/null &
```

Launch manifest: `/home/ami/scratch/immediate_gate_20261013_a_launch.json`. Progress/artifacts: `src/results/push_torch/immediate_gate_20261013_a/coordinator.json`, split collection manifests, then `support.json`, training summaries and final `summary.json`. New code/protocol files are frozen by hashes; documentation status updates do not change those files.

[Architecture review](architecture_review_20261013.md) ties suggestions to precision completion: decoupled gate/continuous repair now; compact observed-contact/causal-prefix representation only if justified by residual misses; completion/continuation-aware repair only if detection improves but task completion does not. Important task-metric finding: present success is position<2cm and speed<.05m/s, with no explicit yaw/depth/dwell success requirement. The6mm geometry is not a demonstrated6mm precision metric. Future transfer requires an explicit mechanically valid precision-success definition. No new architecture/task branch launched.

---

### Update 13 Oct — R5 objective experiment registered; user authorizes execution

[Numerical protocol](immediate_gate_20261013.md) freezes fresh roots76000/77000/78000,24collection batches/768trajectories, per-split root/bin support criteria and automatic stop. On support pass only: paired regression/event training with identical initialization/minibatches,3seeds×3000updates, then9arms×20validation cells×32episodes with original H10 repair frozen. Frozen-H10 and frozen-R5 controls expose horizon/calibration effects. No confirmation/final roots consumed. New-vs-frozen is not called a pure data-only contrast because legacy training procedure differs; event-vs-regression is the matched objective contrast.

All57 semantic tests pass, including5 new boundary tests. CPU smoke `immediate_gate_smoke_20261013_b` completed all stages and verified5raw artifacts; synthetic training labels are explicitly marked and never used as efficacy evidence. Earlier smoke `_a` stopped because its tiny BC training collection had zero unsafe originals; this motivated the explicit smoke-only fixture, with production eligibility unchanged. Actual launch status is recorded separately; the protocol and implementation must be committed before production collection.

---

### Update 13 Oct — Research reset: one gate hypothesis, no new branch launch

The user correctly challenged the growing collection of loosely connected experiments. [The mainline decision](research_mainline_20261013.md) now supersedes the SEAL candidate-study next-step sketch. One objective: learned-verifier **safe task-completion gain over the same full OpenVLA**, with matched observations/control and bounded SR loss. Early BC/vis results, privileged GT gains, cadence-confounded results, small samples and corrected full-actor confirmation must not be conflated. The corrected896-episode confirmation's+1.23pp learned safe gain remains valid, but is small and not consistent in direction on the448-episode validation cohort; the new+2.01pp ramp increment is relative to the old verifier, not to policy.

**Single next intervention:** a bounded R5 immediate-event gate-objective comparison with original repair frozen, separating same-data regression from event loss. Exact collection/support/integration/calibration rules require preregistration; this is not a launch-ready protocol. H10 anticipation support remains failed/blocked and cannot be replaced by R5 success. If the bounded fresh-support check fails, stop and explicitly revise scope rather than opening another collector/SEAL/GRU branch. All final judgments use SR/CVR/safe, paired rescue/harm and uncertainty. SEAL, candidate expansion, smoothing sweeps, larger temporal architectures and full verifier transfer are parked. No new job launched.

---

### Update 13 Oct — Repair-transition complete; incremental gain, overall efficacy still unestablished

**180/180 batches /5,760 trajectories complete**, all workers exited normally. Independently rechecked180 NPZ hashes, all episode vectors, live-risk violation flags, prefix-label parity and three shard-summary hashes. Full results, uncertainty, stress review and SEAL feasibility: [completion report](repair_transition_results_20261013.md). This completion supersedes the following launch/progress entries.

OOD n448/arm SR/CVR/safe(%): policy87.05/29.24/65.63; learned direct84.60/28.79/64.73; ramp84.82/26.79/66.74; flat85.04/26.56/66.52; pause85.04/29.24/64.51. Ramp vs direct Δsafe+2.01[+0.89,+3.35]pp, safe rescue/harm9/0, restores7/12 original harms and preserves8/8 original rescues. Ramp vs policy Δsafe+1.12[−0.67,+2.68]pp, rescue/harm10/5, ΔSR−2.23pp: **incremental screen passes; broader efficacy screen fails**. Ramp vs flat remains inconclusive. Pause reduces local jump/incidents but restores0/12 original harms and does not improve task-level safe success. GT ramp vs GT direct Δsafe+0.45[−1.12,+2.01]pp, rescue/harm8/6: continuity is not a universal repair solution. All results reuse validation roots, not independent confirmation.

**SEAL assessment:** checked official paper/project and repositories `NVlabs/actalign`, `NVlabs/Libero-10-r`. Code is public; documented pretrained reasoning-model checkpoints remainTBD. Local actor is deterministic full OpenVLA+H10 planar head, not a text-plan-conditioned stochastic reasoning policy. Local LIBERO/robosuite/RoboCasa are absent. Recommend a bounded SEAL-inspired **fixed-K=8 non-collinear candidate + continuation-witness audit** before learned selection, retaining scale-only controls and hidden-physics isolation. The previous failed scale/progress experiment is not evidence against all steering/re-contact candidates, but no new candidate benefit is established. Official variable-horizon simulator/VLM semantic verification is a distinct comparison and cannot stand in for learned contact-force verification. No new jobs started; no broad smoothing sweep planned.

---

### Update 13 Oct — Actual-verifier transition ablation launched; tie the next algorithm to measured harms

**Progress review (171/180 batches at status check, ~46 min):** shard2 completed54/54 and exited normally; shards0/1 active at59/63 and58/63. The aggregate is not yet complete. All20 cells already contain none/learned_direct/learned_ramp/learned_flat; independently verified their80 raw hashes, episode vectors and live-risk violation flags. This permits a complete-cohort primary comparison, while pause/GT controls and final aggregate checks remain pending.

OOD n448, SR/CVR/safe (%): none87.05/29.24/65.63; direct84.60/28.79/64.73; ramp84.82/26.79/66.74; flat85.04/26.56/66.52. Ramp vs direct: Δsafe **+2.01[+0.89,+3.35]pp**, paired safe rescue/harm9/0; ΔSR+0.22[−0.67,+1.12]pp, ΔCVR−2.01[−3.35,−0.89]pp. It restores7/12 original harms and retains8/8 original rescues. Ramp vs none: Δsafe+1.12[−0.67,+2.68]pp, rescue/harm10/5; ΔSR−2.23[−4.24,−0.22]pp, ΔCVR−2.46[−4.24,−0.67]pp. Ramp vs flat Δsafe+0.22[−0.45,+0.89]pp. Thus the frozen primary meets the incremental algorithm screen on this reused validation cohort, but fails the broader policy-relative target; gradual interpolation has not demonstrated superiority over constant blending. No independent-efficacy or neural-architecture conclusion is warranted. Intervals use the registered4000 paired, cell-stratified draws and seeds.

The collector-only resumption proposal is superseded by a **direct framework ablation**, [registered before launch](repair_transition_20261013.md), implementation/protocol commit `0318a41`. The target is safe task completion under the actual learned verifier, not another exploration-coverage sweep.

**Premise audit:** original frozen learned verifier OOD n448 has 129 violations /12 safe harms; 340 pre-violation post-repair decisions. Sixteen first violations occur in the chunk immediately after an execution-changing repair, including **all 12 harms**. Only seven occur in a direct-release chunk (four harms), with **zero first-step release violations**. Eight harms occur while repair continues. This narrows the hypothesis to adjacent repaired chunks; it does not transplant the collector's first-step-failure mechanism to the verifier. Saved audit: `src/results/push_torch/repair_transition_20261013_a/actual_repair_audit.json`, with source hashes and per-cell attribution.

**Algorithm:** when the previous real repair changed execution, blend the new proposal's committed prefix from the previous executed velocity with weights .2/.4/.6/.8/1.0 **before** unchanged gate/repair. Keep the suffix unchanged. Controls are direct, constant .6 blend and five-step pause; transforms do not re-arm themselves, and no-op/abstention does not arm one. No neural weights, thresholds, bisection budget or H10/k5 opportunity changes. Exactly one gate query per decision and eight repair risk queries per triggered row are asserted.

**Nine arms:** policy alone; learned gate×four transitions; privileged GT gate×four transitions, all with original learned repair. Primary fixed as learned_ramp. Twenty validation cells (seeds26000–26019), n640/arm, n448 OOD; **180 batches /5,760 trajectories**, not independent confirmation. Three health-checked RTX A5000s divide cells by index modulo3, with all nine methods paired on each cell:

| Shard | Cells / arm batches | GPU UUID | Detached launcher |
|---|---|---|---:|
| 0 | 7 /63 | `GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0` | 671696 |
| 1 | 7 /63 | `GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8` | 671697 |
| 2 | 6 /54 | `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b` | 671698 |

Shared run root `src/results/push_torch/repair_transition_20261013_a/`; logs `~/scratch/repair_transition_20261013_a_shard{0,1,2}.log`; launcher manifest `~/scratch/repair_transition_20261013_a_launch.json`. `setsid nohup`, stdin `/dev/null`, offline loading and thread caps preserve jobs across SSH exit. The last completed worker aggregates once under an exclusive lock. Current state is startup/collection; no result claimed.

**Startup check:** all three workers completed their first none arm (**3/180 arm batches** at this check), matching source factorial episode outcomes and live risk traces. Python PIDs shard0/1/2 **671707/671705/671706** are active. The first cells are deliberately the original stress-grid cells, not a selected favorable subset; partial per-cell metrics are not pooled efficacy results.

**Decision:** report SR/CVR/safe, paired rescue/harm/CIs, original-harm reversal and rescue retention. Incremental benefit requires positive paired lower bound vs direct and ≤2pp SR loss; the broader ≥3pp safe gain vs policy/positive CI/completion/stress requirements remain separate. GT-only benefits are diagnostic. Smoother traces without task-level gain fail the objective. If the fixed rule fails, stop this transition family and return to event-gate data/objective/observability; no unbounded ramp tuning.

**Checks:** four new semantic/controller tests plus existing suite: **52 tests pass**. Nine-arm CPU BC smoke `repair_transition_smoke_20261013_a` completed source-independent rollout, gate/repair query accounting, prefix parity, pre-branch matching and automatic aggregate analysis. Production enforces none/learned_direct/gt_direct replay against the prior factorial. No dependencies changed or model training started.

---

### Update 13 Oct — Boundary diagnosis complete: anticipation still sparse; hazards concentrate at policy resumption

All three shards completed **84/84 batches, 2,688 trajectories over 128 paired training roots**, including automatic analysis. Launcher/Python processes exited. Rechecked **168 raw hashes**, analysis→source hashes, and all four none batches' **bit-exact cross-GPU** state/context/proposal/labels/seven-model scores/order probes.

All 18 perturbed onset/arm combinations fail the registered screen: at most **3 anticipatory roots per arm**, only **5 unique roots across all combinations**, and **0** at the first post-pulse decision. Weaker/gradual pulses largely survive, so direct pulse overshoot is not the entire explanation. Early brake_ramp survives 100% of pulses but yields **125 distinct roots whose next original-policy proposal violates at its first executed step**. The mechanism of this policy-resumption hazard still needs a paired test.

On adequate immediate-event cohorts, the tested safe-score recalibration does not repair the new heads' low recall. Original predictor and static heads are invariant on **272 none-state reversal event flips /126 roots**, while GRU is order-sensitive but still poorly detects actual resumption events. Full uncertainty, collector SR/CVR/safe and paired rescue/harm are in [the completion report](event_boundary_results_20261013.md). These are reused training-root diagnostics, not new learned-method efficacy results.

**Next direction:** register a targeted resumption-transition comparison and localize the contact/action discontinuity; avoid another indiscriminate sweep. Immediate-event learning controls and anticipation-support diagnosis must have separate, explicit scopes. **No new job or training launched in this check.** The numerical protocol remains unchanged.

---

### Update 13 Oct — Three-GPU boundary/gate diagnosis registered and launched

**Historical launch entry, superseded by the completion review above.**

[Fixed protocol and decision map](event_boundary_20261013.md), committed as `e103cfd` before launch. Test collector overshoot, threshold versus score discrimination, and action-order blindness before selecting an algorithm/architecture change. Three CUDA-health-checked RTX A5000s run identical seven-arm designs at observed gap onsets **.20/.12/.06 m**. All use the same **128 training roots** (74000–74003), complete replay from reset, full frozen OpenVLA H10/k5/microbatch8, existing physics ranges and eight scale labels. **84 batches / 2,688 trajectories**, not 2,688 independent roots.

| Shard / run suffix | GPU UUID | Detached launcher PID |
|---|---|---:|
| `event_boundary_20261013_a_early` | `GPU-3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0` | 662796 |
| `event_boundary_20261013_a_middle` | `GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8` | 662797 |
| `event_boundary_20261013_a_late` | `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b` | 662798 |

Logs: `~/scratch/<run_id>.log`; outputs: `src/results/push_torch/<run_id>/`. Each launcher uses `setsid nohup`, stdin `/dev/null`, offline loading and thread caps, and automatically runs per-shard analysis after successful collection. Launch manifest: `~/scratch/event_boundary_20261013_a_launch.json`. Current state is production collection, not a completed result.

**Startup verified:** each shard completed its first none batch (**3/84 batches at this check**) and passed original-pilot replay. Separately compared all three GPUs' saved first-batch state, measured context, proposals, eight-scale labels, all seven model scores, and reversed-action labels/scores: **bit-exact cross-GPU parity**. Python PIDs early/middle/late **662806/662807/662805** were active. These checks establish successful matched execution, not a coverage result.

**Readout:** post-pulse pre-violation anticipatory root coverage, pulse survival, collector SR/CVR/safe and paired rescue/harm/CIs. Feasibility screen ≥24 anticipatory roots, ≥2 per physics stratum, ≥90% pulse survival. Observe original predictor and all six static/GRU checkpoints without changing collector decisions. Use root-disjoint 64/64 diagnostic threshold-fit/check halves, actual safe-trigger rates, recall/precision and cluster CIs; sparse endpoints cannot decide model quality. Offline reversed-order probes are never executed or counted as original-policy coverage. No automatic gate training or method promotion.

**Verification:** **48 targeted tests passed**. Seven-arm CPU BC smoke `event_boundary_smoke_20261013_a` completed collection, frozen-model scores, order probes and automatic analysis; source OpenVLA replay is additionally enforced in production on each GPU. No dependencies changed. Next decisions follow the registered data/threshold/objective/representation decision map rather than unconditional architecture replacement.

---

### Update 13 Oct — Coverage pilot fails anticipation screen; drive pulses mostly violate immediately

All **24 batches / 768 trajectories** of `event_gate_coverage_20261012_a` completed; PIDs 598105/598108 exited. Recomputed 48 artifact hashes, coverage, paired initialization/proposals/parameters, cadence, prior-violation masks, executed-prefix parity and root split isolation. Reproducible analysis: `analyze_event_gate_pilot.py` and run-local `analysis.json`.

Unique original-proposal pre-violation event roots, immediate/anticipation-only/safe-contact: train **33/0/126** versus required **32/24/64**; dev **15/1/64** versus **16/12/32**. Coverage screen **fails**. Drive first violates during **117/125 train** and **57/64 dev** pre-violation pulses. Brake supplies **26/14 immediate roots**, but **0/1 anticipatory roots**. Any-scale aggregation also leaves anticipation essentially absent. More collector violations do not imply usable gate supervision.

Full collector SR/CVR/safe, paired rescue/harm and root-paired 95% intervals are in the [completion review](event_gate_coverage_20261012.md#completion-review--13-oct-2026). These are in-range data-collector outcomes, not new verifier efficacy results. Next: pre-register a small training-only boundary-reachability diagnosis of earlier/weaker/gradual pulses and braking/re-contact, with actual history and H10/k5 preserved. Freeze a revised observation-only collector only after measuring pulse survival and next-policy anticipation; re-pilot on fresh roots before fitting gates. **No new job launched in this review.**

---

### Update 12 Oct — Event-gate coverage pilot implemented with prospective root-level criteria

**Historical launch entry; superseded by the 13 Oct completion review above.**

**Production launched:** `event_gate_coverage_20261012_a`, detached launcher PID `598105`, healthy GPU UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`, `setsid nohup` with stdin `/dev/null`. Log `~/scratch/event_gate_coverage_20261012_a.log`; output under `src/results/push_torch/event_gate_coverage_20261012_a/`. Collection is active; no gate fitting or new efficacy result yet.

Proceeding from the completed factorial diagnosis to a **coverage pilot**, before another gate training run. [Registered protocol](event_gate_coverage_20261012.md): 128 train and 64 design-dev root initializations, each paired across policy-alone, frozen verifier, one bounded drive pulse and one bounded brake pulse. Total 768 trajectories, with branch correlation explicitly grouped by root; same physical ranges, full OpenVLA H10/k5/microbatch8 and eight labelled scales. Pulse decisions use current observed goal gap and seeded constants, not future GT labels. No runtime action-family expansion is claimed.

Screen original proposals before prior violation; require train immediate/anticipation-only/safe-contact coverage in at least **32/24/64 unique roots**, dev **16/12/32**, with all four physical strata represented (minimum 2 train / 1 dev root per event/stratum). Criteria are recorded before production labels and are pilot feasibility thresholds, not efficacy guarantees. Preserve paired initialization/proposal/parameter agreement and exact executed-prefix label checks. Do not train automatically on insufficient coverage. After coverage, separately test event gates with frozen original repair, matching static/causal controls and separating data from loss changes.

**Verification:** three new tests plus 42 existing targeted tests pass (**45 total**); final eight-batch CPU BC smoke completed and correctly rejected insufficient tiny-sample coverage. Files: `collect_event_gate_pilot.py`, corresponding launcher and tests. Production launch status follows in the dedicated protocol record.

---

### Update 12 Oct — Closed-loop gate × repair diagnosis complete: event gating is the main tested lever

`risk_controller_diagnosis_20261012_a` completed all **20 cells × 11 arms × 32 episodes**; PIDs 591432/591435 exited. Independently checked 220 raw hashes, episode vectors, violations against recorded risk traces, cadence, and none/L/L replay parity. Full tables and next protocol are in [the updated postmortem](sequence_risk_20261012.md#completion-update-closed-loop-factorial-diagnosis).

OOD n=448/arm, SR/CVR/safe (%): none **87.05/29.24/65.63**; G/G **82.59/12.05/76.12**; frozen G/L **80.80/8.04/77.46**, static s0 G/L **83.71/8.04/78.79**, sequence s0 G/L **81.25/8.48/76.79**. Their safe rescue/harm versus none are **47/0**, **53/0**, **59/0**, **50/0**, respectively. Sequence s0 L/G is only **85.49/47.32/47.99**, rescue/harm **6/85**, compared with its L/L **83.71/54.24/41.29**, **6/115**.

Sequence G/L improves safe success over L/L by **+35.49 pp [+31.25,+39.73]**, paired rescue/harm **162/3**; L/G improves by **+6.70 pp [+3.79,+9.60]**, **39/9**. G/L improves versus none by **+11.16 pp [+8.93,+13.62]**, but SR drops **−5.80 pp [−8.48,−3.35]** while CVR drops **−20.76 pp [−23.21,−18.30]**. These are fixed-grid paired 95% intervals and privileged diagnostic effects, not learned-method gains. The G gate replaces estimator and threshold semantics together (exact H10 event >1); it does not isolate calibration from representation. GT has hidden-parameter/full-state access, so learnability from current observations remains open.

**Direction:** prioritize an independently trained event gate with **frozen original repair**, beginning with a fresh training-only contact/event-coverage pilot. Separate R5 and H10 event labels; require meaningful independent episode diversity, retain static/causal controls, and isolate data coverage from event-loss changes. Register calibration/selection rules on disjoint dev/calibration data. Do not conflate lower violation with retained completion or use GT diagnostic gains to claim transfer readiness. No new training job was launched in this completion review; main plan synchronized.

---

### Update 12 Oct — Temporal-risk experiment complete: strong negative result; factorial diagnosis launched

**Decision:** neither new head is promoted. Full results, uncertainty, integrity checks, data coverage and causal limits are recorded in [sequence_risk_20261012.md](sequence_risk_20261012.md). All 20 cells × 8 arms completed; baseline and frozen anchor reproduce prior episode outcomes. This is validation, not independent confirmation.

OOD n=448/arm, SR/CVR/safe (%): none **87.05/29.24/65.63**; frozen **84.60/28.79/64.73**; static s0/s1/s2 **82.37/39.96/53.57**, **84.60/43.08/51.34**, **81.92/38.17/53.57**; sequence s0/s1/s2 **83.71/54.24/41.29**, **80.80/57.37/36.83**, **83.93/47.10/47.32**. Safe rescue/harm versus none: frozen **8/12**, static **2/56, 2/66, 1/55**, sequence **6/115, 7/136, 9/91**. Direct sequence-minus-static safe deltas: **−12.28 [−16.52,−8.04]**, **−14.51 [−18.75,−10.27]**, **−6.25 [−10.27,−2.23] pp** (paired fixed-grid 95% CIs). Training seeds share evaluation episodes; they do not triple independent n. ID cells also regress substantially.

**Diagnosis:** GRU improves development MAE and fixed-state candidate ordering, but its H10 unsafe recall is only 9.68–11.69% on the policy-alone OOD trajectory, versus 26.21% for the frozen anchor. New training has just **20 unsafe originals / 6,144 decisions**, zero H10-only unsafe originals, and **152 unsafe scale candidates / 49,152** across 34 episodes. Development's 100% recall is based on just two R5 / three H10 unsafe originals. This is inadequate evidence for detection readiness, despite matched training budgets.

**Completed offline factorial diagnosis:** on the same 5,572 baseline pre-violation OOD states, replacing only GRU gating with exact H10 risk raises local H10 rescues from **17/17/19 to 107/107/107**; replacing only repair scoring gives **18/18/20**. Exact gate+repair yields 110/248 H10 and 16/131 R5 local rescues under the same bisection rule. These are local counterfactuals, not closed-loop success estimates or global controllability bounds. Responsiveness still permits repairs above their own relative target (51/69/73 GRU cases). Prioritize detection/gating and subsequent trajectory consequences, rather than model width.

**Implementation and verification:** added `analyze_sequence_risk.py`, `diagnose_risk_controller.py`, launcher and five tests; **42 tests pass**, and the new 11-arm CPU BC end-to-end smoke completes. The offline CPU attempt `_a` stopped on strict simulator-label parity; successful `sequence_diagnosis_20261012_b` uses the source CUDA GPU and passes per-cell label agreement. No source checkpoints, thresholds or controller files were edited. Integrity checks include source/data/model hashes, split isolation, exact train-only normalizers, baseline replay, cadence and outcome aggregates.

**Next run launched:** `risk_controller_diagnosis_20261012_a`, launcher PID **591432**, healthy GPU `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`, `setsid nohup` with `/dev/null` stdin. Log `~/scratch/risk_controller_diagnosis_20261012_a.log`. Eleven arms: none, shared exact gate/exact repair, and L/L, G/L, L/G for frozen/static_s0/sequence_s0. Seed 0 is a diagnostic convention, not best-seed selection. All 20 validation cells, n=32, full OpenVLA H10/k5/microbatch8; same six-step bisection/relative target/abstention. Exact gate uses physical risk >1, not a learned score quantile. None/L/L must replay existing outcomes. Privileged diagnosis only; formal factorial results remain pending. Plan synchronized; no new verifier-transfer readiness claim.

---

### Update 12 Oct — Temporal-risk architecture implemented and matched-data experiment launched

**Task:** address the representation bottleneck first, rather than jointly changing detection, action freedom and feedback. Added `models/sequence_risk.py`, `experiments/validate_sequence_risk.py`, `tests/test_sequence_risk.py` and `scripts/validate_sequence_risk.sh`. Existing weights, thresholds, controller and source-hashed files remain intact.

**Architecture isolation:** two risk-only ensembles (E=5) share the same measured obs/previous-obs/previous-action inputs and training-only normalizers. Static: three-layer action-wise MLP, hidden 157, 269,265 parameters. Sequence: context-initialized causal GRU, hidden 128, 269,450 parameters (about 0.07% difference). Step j depends only on candidate actions 1…j. Neither consumes predicted observations; both wrap the same frozen state/progress dynamics. Risk/state decoupling is common to both new controls, with the original frozen predictor as a separate anchor. GPU eval-mode input gradients use a non-cuDNN path because cuDNN inference RNNs do not retain backward reserve; runtime gradient repair remains off.

**Fresh data and split isolation:** train 384 episodes (12 batches, reset seeds 71000–71011); development 128 (72000–72003); calibration 128 (73000–73003), all n=32 and H10/k5. These are fresh, reserved streams, separate from the audit/confirmation/final-test. Sample actual physical parameters within the existing predictor ranges: friction multipliers [0.5,1.5], mass [0.6,1.6], without expanding coverage. Train/dev alternate whole batches from uncorrected OpenVLA and frozen bisection trajectories; calibration is uncorrected policy only, with no action noise. Offline functional simulator labels cover the same eight scale candidates at each real decision. Preserve all stages and episode grouping. No future GT labels or hidden parameter inputs enter deployment. Fit normalization on training decisions only.

**Training contract:** seeds 0/1/2 for each architecture, 3,000 fixed iterations, batch 256, matched sampled decision/candidate rows, 30% hot-decision sampling, per-step risk MSE with targets clipped at 3, identical AdamW/cosine schedule and gradient clipping. Fixed final checkpoints; no choosing a lucky seed/iteration. Parameters and optimization steps are matched, not claimed equal FLOPs; per-architecture training wall time is recorded. New heads use independently calibrated 95th-percentile safe original H10 decision scores (α=.05); correlated chunk calibration is not an OOD safety certificate.

**Validation pipeline:** (1) identical saved audit states/candidates from policy-alone and frozen-verifier trajectories: prefix/H10 detection, safe-proposal trigger rates and risk-ranking inversions; no fitting on these validation labels; (2) full 20-cell × 32-episode grid with none, frozen anchor, static seeds 0/1/2 and sequence seeds 0/1/2. Full OpenVLA-7B and the original H10 DAgger×2 head, microbatch 8, k=5 and frozen bisection semantics are retained. Baseline/frozen episode outcomes must reproduce the original candidate-validation source. Report SR/CVR/safe success, paired rescue/harm and fixed-grid CIs versus none and direct same-seed sequence-versus-static contrasts. Runtime bisection still searches continuous scales; interpolation beyond the eight labelled scales is shared by both heads. Original and new predictors' data/objectives differ, so only the matched static/sequence contrast isolates temporal representation.

**Verification:** all eight new tests pass (parameter budget, static permutation equivariance, sequence causality/action-history response, CPU/GPU eval-mode action gradients, train-only normalization, split isolation, vectorized simulator labels versus serial labels). All 29 existing audit/attribution/cadence/candidate tests pass: 37 total. CPU BC end-to-end collection/training/calibration/rollout smoke completed under `sequence_risk_smoke_20261012_a`; its tiny three-iteration models are implementation checks, not efficacy evidence. The separate full-grid offline smoke exactly reproduced the frozen audit's 59 true prefix triggers / 86 misses / 337 safe-prefix triggers on its OOD bisection trajectories.

**Production launch:** `sequence_risk_20261012_a`, launcher PID `587112`, healthy GPU `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`, `setsid nohup` with stdin `/dev/null`, offline model loading and CPU thread caps. Log `~/scratch/sequence_risk_20261012_a.log`; artifacts `src/results/push_torch/sequence_risk_20261012_a/`. Startup reached full OpenVLA CUDA loading. Batch datasets/manifests, model checkpoints, per-arm/per-cell evaluation partials and stage status are saved atomically. Formal results remain pending; a successful launch or smoke is not an improvement claim.

**Latest launch check:** Python PID `587115` active; first 3/12 training batches completed (96 episodes), containing nonzero unsafe-candidate fractions of 1.51%, 0.12% and 1.37%. Stage remains collection, `complete=false`. These are data-composition statistics, not model performance. Version the matched-protocol/audit prerequisites together with the new experiment so the remote code includes its local dependencies; model/data artifacts remain separately hashed local files.

**Prospective decisions:** promote temporal representation only with matched-state and closed-loop evidence, not MAE alone. If both new heads improve similarly, attribute gains cautiously to data or shared risk/state decoupling. If sequence improves risk discrimination but not safe success, next isolate gate/continuation decisions while fixing the model. If neither improves, inspect coverage/history/controllability before another size increase. Do not tune confirmation or final-test outcomes.

**Transfer gates, registered before this run reports outcomes:** LIBERO single-task policy/control feasibility may proceed independently now; official verifier transfer still follows stable efficacy. Proposed project go/no-go is ≥+3 pp pooled OOD safe-success improvement versus policy-alone with a positive lower fixed-grid paired 95% bound on an independent episode test, consistent point-estimate direction across all three reported training seeds, and a second independent episode cohort confirming direction. An SR loss >2 pp or substantial stress-cell regression triggers review. These are prospective project criteria, not universal benchmark standards or safety certification. Validation screens methods; it cannot itself satisfy the independent-test gate.

**Embodiment readiness:** environment check found MuJoCo 3.14.0 available; robosuite/LIBERO/RoboCasa absent. The current planar two-action head is not an arm controller. A task-compatible VLA/action head, EE/gripper frame/unit conventions, camera/state adapters and validated native success/contact labels are required; retrain/recalibrate the verifier for the new simulator. Physical hardware remains deferred until target-embodiment simulation and accessible sensing/control interfaces are verified. There is no pushing-success percentage that automatically establishes hardware readiness. Chinese scope and gates are synchronized in `core_research_plan.md`.

---

### Update 12 Oct — Intervention branches complete; architecture and algorithm evaluation

**Completion verified:** `branch_intervention_branch_20261012_a/summary.json` is `complete=true`, all eight cells saved, and launcher/Python processes exited normally. Every whole-batch unmodified control replay and unselected-row parity check passed. Rechecked the audit-summary hash, all eight raw branch hashes, exactly one skip per selected episode and saved NPZ/summary outcomes. No new efficacy run or training was performed for this evaluation.

| Hindsight-selected bisection group | n | original SR / CVR / safe counts | after one skipped intervention: SR / CVR / safe counts |
|---|---:|---|---|
| Paired safe harms | 12 | 9 / 12 / 0 | 11 / 0 / 11 |
| Paired safe rescues | 8 | 8 / 0 / 8 | 7 / 8 / 0 |

Skipping the first execution-changing repair eliminated all 12 harms' violations; 11 regained safe task completion and one remained unsuccessful. All eight rescues lost safety after the skip. These are diagnostic, outcome-selected rows from the validation grid; 11/12 is not an estimated deployable rescue rate, and net 11−8 is not a new method's improvement. Full-grid validation SR/CVR/safe-success and paired CIs remain as reported below.

**Causal scope:** with initial state/history/proposal matched and the subsequent policy/verifier held fixed as functions, changing one early repair changes later safety outcomes in both directions. This establishes a total effect of that intervention plus downstream replanning, not a decomposition into inertia, contact geometry, policy response or later verifier actions. Original harm trajectories first violated 5–8 steps after the intervention; after skipping, seven rescue trajectories first violate 4–8 steps later, while the remaining rescue (cell 12, row 23) violates 46 steps later. A short safe prefix is insufficient, and H10 open-loop safety is not identical to closed-loop continuation value.

**Threshold discrimination check:** first-intervention learned max-risk scores span 0.5634–0.9440 for harms and 0.5610–0.7873 for rescues (current τ=0.5554924). A single monotone threshold cannot suppress all these harms while preserving all these rescues; this does not exclude an average benefit from recalibration. All 12 harms' original H10 chunks are GT-safe; seven of eight rescues' original H10 chunks are unsafe, but the long-delay rescue is H10-safe. This supports improving predictive discrimination while retaining useful anticipation, rather than suppressing interventions wholesale.

**Framework assessment:** evaluation/replay infrastructure is now useful for mechanism work. End-to-end learned-verifier efficacy remains unestablished across independent seed sets: confirmation gave small positive deltas; this validation remains below baseline point estimates. The branch supplies stronger evidence for intervention discrimination and temporal consequences as design priorities. It does not prove a specific neural architecture will improve safe success. Current predictor risk is order-insensitive, runtime feedback is one-step observation/action context, repair is scale-only, and the controller does not assess the policy-conditioned consequences beyond its committed prefix.

**Recommended first architecture ablation:** direct causal sequence-risk prediction: initialize a small recurrent latent from the existing obs/previous-obs/previous-action context, update it with each candidate action, and emit per-step physical risk. This conditions step j on actions 1…j without feeding back drifted predicted observations. Retain H10/k5, the eight scales, information access and ensemble size. Use matched-data, approximately parameter-matched static-risk versus GRU-risk controls with the same training loss/budget and a common unchanged state/progress predictor; retain the original frozen checkpoint as an anchor. Splitting the risk/state networks must be common to the matched new controls, not silently confounded with temporal representation. Longer history, event/ranking losses and larger candidate families are separate later ablations.

**Training protocol concern from source inspection:** `train.py:train_orbisim` fits normalizers using all `DynWindows.O` before its episode train/validation split is used for batches. This is an internal held-out normalization leak, not evidence of leakage into the separate current audit grid. Future matched-data training must fit statistics on training episodes only and keep all branches/candidates from an episode within the same split. Generate fresh disjoint training-seed policy decisions/candidate labels; audit validation labels and these 20 selected outcomes never become training targets. Recalibrate new scores on independent calibration episodes. Confirmation/final-test remain excluded from tuning.

**Algorithm sequence:** first compare risk formulations while keeping controller/repair rules fixed. Next, with the model fixed, calibrate/report execution-prefix R5 separately from counterfactual lookahead R10 and ablate the intervention gate; retain anticipation. Then test candidate acceptance based on absolute risk, paired risk improvement and minimal modification, with original proposal included, rather than relying exclusively on a fixed 25% surrogate-score drop. Marginal ensemble uncertainty is not a calibrated paired improvement bound. Any later continuation/recoverability value head needs separate policy/controller-versioned branch targets and fresh data; do not relabel open-loop H10 risk as closed-loop value.

**Repair/feedback gate:** no-safe-scale cases warrant same-state zero/retreat and earlier-decision witnesses before adding directional repair. Keep candidate count and execution opportunities matched. Add longer motion/action history independently; force/contact sensing is a separate information intervention. Gradient optimization stays a later ablation after verifying finite-perturbation and true-risk ranking reliability. Everyday drawer/holder task feasibility remains appropriate in parallel, but manipulation transfer and faster inner loops require task-native labels, new predictor/calibration and classical information/control-matched feedback baselines.

**Synchronization:** detailed Chinese design, ablation order and gates are in `core_research_plan.md`. This is a results-backed design recommendation; the proposed risk architecture, gate and continuation head have not been implemented or trained.

---

### Update 12 Oct — Full audit completed; CPU attribution and skip-one-intervention branch launched

**Completed:** `verifier_audit_20261012_a` finished all 20 cells × 32 episodes × four arms (80 raw NPZs). `summary.json` has `complete=true`; every arm replay matched source outcomes/cadence and actual five-step prefix risk. The original detached processes exited normally. Previous entries saying the audit is active describe earlier checks, not its latest state.

**Reproducible attribution:** new CPU-only `src/experiments/analyze_verifier_audit.py` requires a completed full-grid validation audit and verifies the source hash, raw episode outcomes and prefix/live labels. It hashes all raw inputs and emits stage-conditioned decision diagnostics, one witness per first actual violation, terminal harm/success-loss/safety-loss flags, and first execution divergence from the paired baseline. Output: `src/results/push_torch/audit_attribution_20261012_a/analysis.json` and `report.md`. Raw artifacts remain local/ignored. No predictor fitting, threshold search or deployment/controller change occurred.

**Pooled OOD outcomes reproduced, n=448/arm:**

| arm | SR % | CVR % | safe success % | paired safe rescue / harm | safe delta pp [fixed-grid paired 95% CI] |
|---|---:|---:|---:|---:|---|
| none | 87.05 | 29.24 | 65.625 | — | — |
| bisection | 84.60 | 28.79 | 64.73 | 8 / 12 | −0.89 [−2.68,+0.89] |
| discrete | 84.15 | 29.24 | 64.73 | 5 / 9 | −0.89 [−2.46,+0.67] |
| progress | 84.82 | 29.02 | 65.18 | 7 / 9 | −0.45 [−2.23,+1.34] |

Intervals are copied from the frozen validation source, not newly inferred from correlated audit decisions. This replay is not independent efficacy evidence.

**Local five-step evidence (own arm trajectories, all stages before first violation including its decision):** bisection has 145 unsafe proposals, 59 true triggers and 86 misses. Of the misses, 14 have a truly safe tested scale and 72 do not. Triggered execution still unsafe with a safe scale available: 8; without one: 35. Its 129 actually violating OOD episodes partition at their first violating decision as 14 missed-with-safe-scale, 72 missed-without-safe-scale, 8 triggered-with-safe-scale and 35 triggered-without-safe-scale. Thus 107/129 have no safe choice among the eight tested scales at that boundary. This is not proof that all actions are unsafe, nor an end-to-end upper bound for a new detector: earlier decisions could alter the reachable state. Discrete/progress first-violation categories are 14/73/6/38 and 14/73/5/38 respectively, in the same order.

**Lookahead and trigger mismatch:** bisection H10 labels have 265 unsafe proposals, 78 true triggers and 187 misses; 95 misses have a safe scale. Its 396 scoped triggers include 337 prefix-safe proposals, of which 318 are also H10-safe. Prefix-safe triggers may legitimately anticipate beyond step five; full-H safety is still only continuing that chunk without later replanning. Neither category alone identifies causally harmful intervention. Bisection risk-ranking inversions are 18,712/92,045 comparable prefix candidate pairs (~20.3%); pairs are correlated local samples, not 92,045 independent trials.

**Paired trajectory witnesses change the diagnosis:** all 12 bisection safe-harm episodes have later violations; only three also lose terminal success. All nine discrete and nine progress harms have later violations; only two in each arm also lose success. At every harm's first execution-changing intervention, state/proposal match baseline, the original H10 proposal is GT-safe, and the applied prefix is GT-safe. The first actual violation follows 5–8 control steps later. The intervention increases five-step goal distance by roughly 10–42 mm in these harms. This is compatible with altered contact/replanning trajectories, not a demonstration that slowing down uniquely caused the violation. Zero local induced-prefix hazards does NOT exonerate later harm. For bisection rescues, 7/8 first-divergence original H10 proposals are unsafe despite 7/8 having safe prefixes; useful anticipation and unnecessary intervention must be distinguished, not simply disabled together.

**Representation and progress:** all 11,186 bisection order-probe pairs preserve predicted full-H max score; 1,838 change true safety label, and 3,605 change true risk by >0.1. These are correlated reversed/half-shifted offline chunks, possibly off-policy. They establish order-insensitivity, not its contribution to safe-success loss. Among ≥2 learned-accepted AND GT-prefix-safe scales, >1 mm predicted-choice progress regret occurs in 9/2,208 bisection decisions, 5/2,181 discrete decisions and 7/2,228 progress decisions. The triggered subsets are 9/297, 5/302 and 6/329. This conditional goal-distance result does not vindicate overall progress prediction or task-completion semantics; it offers little support for making a larger progress head the immediate fix.

**Next minimal counterfactual, launched but not yet completed:** `src/experiments/audit_intervention_branch.py` and `scripts/audit_intervention_branch.sh` select ALL bisection pooled-OOD paired harms/rescues (12+8 rows across cells 0,1,2,3,7,12,15,19). Skip exactly its first execution-changing intervention per selected episode, then continue the original frozen verifier. Preserve the whole original 32-row batch, renderer/appearance generation, forward microbatch 8 and H10/k5; first reproduce the unmodified whole-batch replay per cell and require unaffected-row outcomes and pre-branch state/proposal/plan agreement. The diagnostic updates cached plan, current action and previous-action history together without changing pointers/cadence. Hindsight outcome selection is offline only: no outcome label or simulator future enters a deployment rule. Results will measure harm reversibility and lost rescues under this particular one-shot intervention, NOT a deployable method's SR.

**Detached launch:** launcher PID `584133`; healthy GPU UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`; launched with `setsid nohup` and stdin `/dev/null`, offline model loading and CPU thread caps. Log `~/scratch/intervention_branch_20261012_a.log`; output `src/results/push_torch/branch_intervention_branch_20261012_a/`. Startup reached CUDA setup/model loading; version warnings alone are not a replay failure. Branch efficacy/attribution claims await checked per-cell/final outputs.

**Subsequent partial check:** Python PID `584136` remains active; cells 0 and 1 completed (2/8), with unmodified replay controls and unselected-row parity passing. Among these cells' three original harms, skipping the first divergent intervention restored safe success in two; the third avoided violation but still failed completion. All five original rescues in these cells lost safety (and hence safe success) after the skip. These hindsight-selected local branch cases are not independent efficacy trials or the final result. They already rule out treating blanket intervention suppression as a demonstrated fix. Full result still pending.

**Tests:** seven new attribution/skip-branch tests (including full synthetic control/branch, unselected-row parity, exactly-once skip and action-history/pointer semantics), seven existing audit tests, five cadence tests and ten candidate/pairing tests pass: 29 total.

**Design gate:** first distinguish unnecessary safe-lookahead intervention from useful anticipation via the paired branches. Then select one minimal risk/temporal or timing/repair ablation, not a combined redesign. If no safe scale is present, test zero/retreat or earlier-decision offline witnesses before declaring a richer action family necessary. Any training uses new disjoint training-seed data; audit validation labels are not training targets. Matched-data old/new formulation comparisons and independent recalibration are required; confirmation/final-test remain excluded from tuning. The authoritative Chinese update is in `core_research_plan.md`.

---

### Update 12 Oct — Research focus: architecture-first precision contact in everyday manipulation

**User/supervision direction:** focus on fine, contact-sensitive work grounded in everyday life, rather than only an industrial insertion example or a broad kitchen composite task. The contribution remains the differentiable physics verifier and constrained local-repair architecture. Larger MLPs, more data, wider physics ranges or more candidates alone are not the main research objective.

**Candidate shortlist, not verified benchmark IDs:** gently close a drawer/cabinet to its seated endpoint; accurately place a cup/bottle into a narrow holder/storage slot; align and seat a storage-container lid; press an appliance button to its effective travel without over-pushing. First investigate the drawer/closure and constrained-placement candidates, then choose only one with credible geometry/contact labels, compatible policy/control conventions and a baseline able to reach the contact stage. Lid compliance/snap-fit, airtight sealing and appliance functionality are not assumed to exist. Peg/connector insertion may support mechanism stress testing; daily-life contact manipulation should anchor the eventual demo. Kitchen composite tasks remain later options.

**Architecture questions:** (1) Can the risk representation distinguish action order and anticipate future contact rather than repeatedly scoring current-state actions? (2) Is current motion context sufficient to infer dynamics, or is longer motion/contact feedback history required? (3) Is the finite repair family expressive enough to align, retreat and resume, or does it merely brake? (4) Does the progress signal track actual completion/contact stage rather than just goal distance? (5) Are intervention opportunities too coarse for the contact dynamics, and what matched control/feedback baselines are needed if the architecture changes cadence?

**Partial audit evidence, explicitly limited:** the first full replay cell (friction 0.2 / mass 0.5, n=32) matched original outcomes and prefix risks for all four arms. In the bisection arm, 448 pre-first-violation action-order probe pairs preserve the learned full-H score within 1e-5; 89 change the true full-H violation label, and 158 change true risk by more than 0.1. At the same cell, 23/25 GT-unsafe five-step proposals did not trigger, with 2 true triggers and 12 triggers on prefix-safe proposals. Prefix-safe triggers can still be legitimate lookahead anticipation, not automatically false alarms. These decisions/probes are correlated samples from one cell, not independent trials or the pooled audit result. The reordered chunks may be outside normal policy proposals. They demonstrate a representational limitation and detection mismatch to investigate, not that this explains all failures. At the latest log check, cells 0 and 1 had all four replay arms saved and cell 2 was partially saved; the audit was still running, not complete.

**Current versus proposed architecture:** current state/action history is one step; force/torque history is not a predictor input. Current repair is scale-based, with gradient and hard-prefix branches off. The intended design space is `VLA intent/proposal → observed motion/contact history → temporal physics verification → constrained local repair → execution → new feedback`. Longer history, direct chunk-risk prediction, directional/retraction repair, contact-stage progress and faster feedback are hypotheses, not implemented components or guaranteed improvements. Simulator force labels are not evidence of real sensor feedback. EE pose and object/contact sensing must be specified separately.

**Audit-to-design gate:** finish the frozen validation replay; organize findings as mechanism, supporting evidence, unresolved alternatives and smallest isolating ablation. If existing safe candidates are misranked, address prediction/ranking; if the finite set has no useful candidate, investigate a richer but bounded repair family; if temporal representation is implicated, compare a temporal formulation against the existing formulation retrained on identical data. Use targeted trajectory counterfactuals where local witnesses do not explain final harm/rescue. Do not launch all changes together or default to scale-up.

**Fairness and task validity:** retain H10/chunk_k=5 for the active audit and current arms. Future inner-loop/feedback changes require matched information and control opportunities, plus an appropriate classical feedback/impedance baseline. Declare privileged state access, keep true mass/friction hidden from runtime prediction, and control data/candidate budgets. Validate simulator contact/solver behavior and task-native success/risk definitions before imposing thresholds. Report SR/CVR/safe success, paired harm/rescue and uncertainty, supplemented by alignment/travel/contact-stage metrics where valid. Benchmark task IDs, checkpoint/action adapters and dependencies remain unverified; public tasks and custom precision variants must be labelled separately.

**Plan synchronization:** the authoritative Chinese scope/design section is at the top of `core_research_plan.md`; this English record preserves the decision and evidence. Continue the active audit without changing its source artifacts or thresholds. No new training, algorithm switch, task implementation, physical robot work or verifier transfer is authorized/claimed by this direction update. The documentation push is separate from pending local implementation changes.

---

### Update 12 Oct — Frozen validation decision replay audit launched; no algorithm changes

**Purpose:** identify evidence for specific failure mechanisms before designing another algorithm. This is an offline audit, not a new efficacy sweep, training run, threshold search, or benchmark transfer. All existing weights, calibration files and controller logic remain unchanged.

**Source and scope:** replay `rq2_candidate_validation_20261009_a.json` only (validation seeds 26000–26019, n=32/cell, all 20 cells, all original 32 episode rows). Arms are policy alone, bisection, discrete repair and progress repair. Preserve full OpenVLA `llm` H10 DAgger×2, original forward microbatch 8, and execution horizon 5. Confirmation/test sources are explicitly rejected by the audit script. Replay all rows, including seated/zero-action stages, rather than subsetting interesting episodes and inadvertently changing initial-state/appearance generation or inference batch math.

**Implemented:** `src/experiments/audit_verifier.py` and `_audit.py`. At actual policy decisions, snapshot current state, observation history, proposal, selected plan and trigger. For each fixed candidate scale, record learned per-step risk, learned goal-distance prediction, functional simulator per-step risk and true goal distance at steps 5 and 10. Record the applied chunk identically. GT future labels are offline only and never enter Controller or candidate selection. Simulator state equality is checked after labelling; prefix GT labels must reproduce actual executed risks. Each replay must match the original episode success, violation, first violation/trigger/success, intervention counts and final task costs, plus policy-call cadence. A mismatch fails the run rather than being silently treated as evidence.

**Diagnostic questions:** decision-level detection confusion over prefix vs lookahead; misses where a truly safe scale exists; selected unsafe repairs despite an available safe scale; no safe option within the finite scale set; originally safe proposals made unsafe by repair in a same-state local counterfactual; predicted acceptance versus true safety; candidate risk-order inversions; learned-progress regret among comparable candidates. Primary summary counts use decisions up to first episode violation and retain both seated and unseated stages; all later records remain in NPZs for inspection. Safety after five executed steps does not imply safety under later replanning. These categories overlap and are not exclusive full-episode causal explanations.

**Temporal-order probe:** reverse and circularly shift the proposed action sequence offline, retaining its action multiset. Test whether the full-H learned score remains unchanged while true risk or its violation label changes. The current risk formulation's full-H score is order-invariant by construction (also checked by a unit test); frequency and importance on these trajectories still require audit labels. These diagnostic probes are not proposed runtime repairs and may be outside the policy's normal proposal distribution. A witness demonstrates a representational limitation, not by itself that this limitation caused a particular original-policy failure.

**Outputs:** per-cell/per-arm compressed NPZs contain raw state/context/action/risk/progress traces, outcome timing, and paired safe-harm/rescue flags. Partial cell and run summaries are saved atomically. Completed summary reports pooled OOD counts with explicit decision scope; audit runtime is not reported as policy latency. Artifact/source hashes are checked before replay; source result and new audit code hashes are recorded. Diagnostic tolerances (not control constraints) are recorded: risk pair-ranking tie tolerance 1e-4, order-score tolerance 1e-5, true order-risk difference 0.1, progress regret 1 mm. Do not tune runtime thresholds from these post-hoc labels.

**Checks:** seven new audit CPU tests pass, as do the existing five cadence and ten candidate/pairing tests (22 total). A one-cell, five-arm BC implementation replay (n=4) completed, reproduced source outcomes and actual prefix risks, and saved raw/pooled diagnostic outputs. It is a pipeline check, not OpenVLA causal evidence. Python compilation and shell syntax checks passed.

**Launch:** `scripts/audit_verifier_validation.sh`, run `verifier_audit_20261012_a`, detached with `setsid nohup ... < /dev/null`, offline model loading, healthy GPU UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`. Launcher PID 579005, Python PID 579008. Log `/home/ami/scratch/verifier_audit_20261012_a.log`; outputs `src/results/push_torch/audit_verifier_audit_20261012_a/`. At the startup check the process was alive and model initialization warnings were present; no completed OpenVLA replay cell had yet been observed.

**Next:** require replay agreement, inspect pooled diagnostics and matched harm/rescue cases, then use specific counterfactual replay branches if a full-episode attribution remains ambiguous. Only afterwards choose the smallest relevant algorithm ablation. Direct chunk-risk prediction, longer history and recovery are hypotheses, not preselected solutions. Reserved final-test seed base 46000 remains unused.

---

### Update 12 Oct — Expanded confirmation shows only a small gain; candidate repair has no established net benefit

**Status:** both detached jobs completed, final JSON files have `complete=true` and 20/20 cells, and neither launcher nor evaluation process remains active. Checked actual checkpoint/calibration/source SHA-256 against current files: no missing or changed artifacts. All arms use `matched_open_loop_v2`, full OpenVLA-7B `llm` H10 DAgger×2, `chunk_k=5`. Cadence is 16 policy calls and 1,024 policy rows per cell for confirmation, 16 calls and 512 rows for candidate validation. No predictor retraining took place in these experiments.

#### 1. Independent fixed-method confirmation

Result: `src/results/push_torch/rq2_verifier_confirmation_20261009_a.json`. Seeds 16000–16019, n=64/cell; 1,280 episodes/arm on the full grid, 896 episodes/arm over 14 pooled OOD cells.

| Arm | SR | CVR | Safe success | Paired safe delta vs none (percentage points; 95% CI) | Safe rescue / harm |
|---|---:|---:|---:|---:|---:|
| Policy alone | 84.71% | 31.92% | 62.28% | — | — |
| Physics, frozen decision taus | 82.92% | 28.35% | 63.50% | +1.23 [+0.11, +2.46] | 21 / 10 |
| Physics, frozen legacy taus | 83.48% | 28.35% | 63.84% | +1.56 [+0.45, +2.68] | 20 / 6 |
| Finite-horizon GT shadow reference | 81.36% | 8.71% | 77.01% | +14.73 [+12.72, +16.85] | 133 / 1 |

These are within-cell paired-episode bootstrap intervals (4,000 draws) conditional on this fixed parameter grid. They are not guarantees on new tasks/OOD distributions, nor a correction for multiple comparisons. The learned verifier has a small positive signal in confirmation, not the large robust improvement needed for the intended contribution. Decision-tau physics also reduces task success by 1.79 pp (paired SR CI −3.24 to −0.33 pp). The earlier n=8 signal (+3.6–4.5 pp safe success) overestimated the gain observed here.

**Bands (descriptive/post-hoc):** at friction 0.2, safe success is baseline 4.30%, physics decision-tau 8.59%, GT shadow 42.19%; learned CVR remains 83.59%. On the other ten OOD cells, baseline and decision-tau physics both achieve 85.47% safe success (7 rescues / 7 harms), versus GT shadow 90.94%. In the friction 1.8 / mass 2.0 cell, decision-tau physics lowers safe success from 76.56% to 68.75%, with 0 rescues / 5 harms; CVR rises from 18.75% to 26.56%. Do not tune thresholds by these true parameter labels.

**Trigger timing:** among 254 violating OOD episodes in the decision-tau learned arm, 157 (61.8%) violate before their first trigger or never trigger; 34 never trigger. This is a trajectory-level timing observation, not a decision-level recall estimate and not proof that prediction error is the only cause. Common cadence, forecast horizon, score calibration and modeled action controllability must be disentangled by replaying the preceding policy decisions. Mean GT risk reduction on triggered chunks is 34.7%, but that local audit does not imply prevention of the first episode violation. Triggered-chunk risk averages are also selected on the verifier's trigger and cannot establish global detection precision.

#### 2. Candidate-repair algorithm validation

Result: `src/results/push_torch/rq2_candidate_validation_20261009_a.json`. Seeds 26000–26019, n=32/cell; 640 episodes/arm on the full grid, 448 pooled OOD. Fixed original predictor and decision-only taus.

| Arm | SR | CVR | Safe success | Paired safe delta vs none (pp; 95% CI) | Safe rescue / harm |
|---|---:|---:|---:|---:|---:|
| Policy alone | 87.05% | 29.24% | 65.63% | — | — |
| Current bisection | 84.60% | 28.79% | 64.73% | −0.89 [−2.68, +0.89] | 8 / 12 |
| Discrete minimal modification | 84.15% | 29.24% | 64.73% | −0.89 [−2.46, +0.67] | 5 / 9 |
| Risk-filtered learned progress | 84.82% | 29.02% | 65.18% | −0.45 [−2.23, +1.34] | 7 / 9 |

Progress vs bisection adds 0.45 pp safe success (5 rescues / 3 harms; independently recomputed paired CI −0.67 to +1.57 pp). Progress vs discrete adds 0.45 pp (only 2 extra safe successes, no paired harms); this is not a substantive demonstrated improvement. All three methods remain below the policy-alone point estimate. Progress increases the local mean GT risk reduction from 33.2% to 38.1%, but that does not translate into positive overall safe success. Stall decision rates are 6.77% baseline, 7.52% bisection, 7.56% discrete, 7.46% progress: no demonstrated stall reduction. Current stall is a diagnostic proxy, not a cause attribution.

At friction 1.8 / mass 2.0, baseline safe success is 93.75%, bisection 75.00%, discrete/progress 84.38%. Repeated harm in this band makes it a useful validation replay target, not a license for a cell-specific runtime rule. Results from the two seed sets must not be compared as though their baseline difficulty were identical.

#### 3. Week 4 priorities — proposed, not yet launched

1. **Decision-level replay audit on validation trajectories.** Replay decisions before first violations, false triggers on GT-safe proposals, and matched harm/rescue trajectories. For the same eight candidate scales, compare learned vs simulator risk over both the actual five-step execution prefix and ten-step lookahead. Audit risk ranking, predicted vs true progress, whether a candidate actually meets its target, and whether the best available scale can prevent violation. Keep privileged GT labels offline only. Do not conclude that progress drift, late detection, or stall is the sole bottleneck from aggregate rates.
2. **If the audit confirms temporal-risk mismatch, train a direct chunk-risk head.** Current `OrbiSimDynamics.rollout` computes risk from the same current context for every proposed action, while its state rollout is autoregressive. Thus H10 does not by itself establish correct future contact anticipation. Use simulator-labelled full candidate chunks to predict per-step risk directly from current motion context plus the complete action sequence, avoiding state-rollout drift in the risk path. Retain differentiability and hidden friction/mass; distinguish executed-prefix and lookahead labels. Compare the existing risk formulation retrained on the same candidate dataset against the new formulation, to separate data coverage from temporal-model effects. Do not simply increase MLP width or reopen the old autoregressive-risk branch.
3. **Isolate history conditioning next, if needed.** The existing predictor already sees one previous observation/action. Test a longer history separately to infer hidden dynamics from observed response; never input true mass/friction. Keep repair mechanism fixed while testing the risk model, and recalibrate new scores on independent calibration episodes rather than reusing old taus or tuning on test cells.
4. **Only then revisit repair/recovery.** Keep candidate/progress as ablations, not the promoted default. If reliable risk/progress ranking is achieved but repair still stalls, test stall recovery with explicitly defined proposals. No simultaneous risk-head, candidate-source and recovery changes.
5. **Freeze before final testing and transfer.** Reserved final-test seed base 46000 remains unused. Report SR/CVR/safe success, paired rescue/harm, bands and uncertainty. Verifier transfer to LIBERO remains gated on stable, practically meaningful net benefit. LIBERO dependency/checkpoint/baseline feasibility may be prepared separately; physical robots and kernel optimization remain deferred.

**No new long job was launched during this results inspection.** Week 4 recording and the root plan were updated; existing experiment artifacts and uncommitted user work are preserved.
