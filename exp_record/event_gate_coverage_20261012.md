# Event-gate coverage pilot protocol (12 Oct 2026)

## Completion review — 13 Oct 2026

**Complete, coverage screen failed.** All 24 batches / 768 trajectories finished normally; launcher/Python 598105/598108 are no longer running. No gate training was started. The launch section below is historical.

`analyze_event_gate_pilot.py` independently verified **48 artifact hashes**, recomputed coverage, paired initial states/proposals/parameters, 16-decision cadence, prior-violation masks from live traces, executed-prefix parity and train/dev root isolation. Reproducible report: `src/results/push_torch/event_gate_coverage_20261012_a/analysis.json`. This review uses stored labels and traces; it does not claim a fresh simulator relabeling audit.

| Unique roots, original proposal before prior violation | Train observed / required | Design-dev observed / required |
|---|---:|---:|
| Immediate R5 unsafe | **33 / 32** | **15 / 16** |
| Anticipation-only (R5 safe, H10 unsafe) | **0 / 24** | **1 / 12** |
| Safe contact | **126 / 64** | **64 / 32** |

Train immediate and safe-contact also pass their four-stratum criteria. Anticipation fails both count and diversity criteria. Across collectors, immediate root counts are none/frozen/drive/brake = **4/7/8/26 train**, **0/0/7/14 dev**; these are overlapping paired roots, not additive independent counts. Only brake yields a pre-violation anticipatory original, on one dev root. Broadening to any of eight scales still gives only **one train root** (none) and **one dev root** (brake), so candidate aggregation cannot repair the coverage deficit.

### Collector attribution

- **Drive overshoots the usable collection region:** of 125 train and 64 dev pulses applied before prior violation, **117/125 (93.6%)** and **57/64 (89.1%)** first violate during the five-step pulse itself. Drive produces four anticipatory original decisions per split only *after* prior violation; these do not satisfy the registered pre-violation objective. Final drive CVR is 97.66%/100%.
- **Brake yields immediate hazards but almost no anticipation:** 26 train / 14 dev immediate roots; only **1/125 train** and **1/64 dev** pre-violation pulses first violate during the pulse. Later policy behavior is therefore worth investigating, but no particular re-contact mechanism has yet been established.
- Velocity bounds alone did not prevent the exploration pulse from immediately overloading contact. More violating collector trajectories are not equivalent to more usable original-proposal event labels. This is a failed collector design, not a new learned-verifier efficacy failure; no new gate has been fitted.

### Collector outcomes and paired uncertainty

These are in-range **training/design-data collector outcomes**, not OOD efficacy estimates. SR/CVR/safe are percentages; deltas and 95% intervals are percentage points versus the paired none branch. Bootstrap resamples root pairs within each split (4,000 draws); branches are not independent observations. Zero-width intervals for identical observed outcomes do not imply zero population uncertainty.

| Split / collector | SR / CVR / safe | Safe rescue / harm | ΔSR [95% CI] | ΔCVR [95% CI] | Δsafe [95% CI] |
|---|---|---|---|---|---|
| Train none (n128) | 96.09 / 3.13 / 93.75 | — | — | — | — |
| Train frozen | 95.31 / 1.56 / 93.75 | 0 / 0 | −0.78 [−2.34,0] | −1.56 [−3.91,0] | 0 [0,0] |
| Train drive | 23.44 / 97.66 / 2.34 | 0 / 117 | −72.66 [−80.47,−64.06] | +94.53 [90.63,98.44] | −91.41 [−96.09,−85.94] |
| Train brake | 96.09 / 21.09 / 77.34 | 1 / 22 | 0 [0,0] | +17.97 [10.94,25.00] | −16.41 [−23.44,−9.38] |
| Dev none (n64) | 93.75 / 0 / 93.75 | — | — | — | — |
| Dev frozen | 93.75 / 0 / 93.75 | 0 / 0 | 0 [0,0] | 0 [0,0] | 0 [0,0] |
| Dev drive | 21.88 / 100 / 0 | 0 / 60 | −71.88 [−82.81,−60.94] | +100 [100,100] | −93.75 [−98.44,−87.50] |
| Dev brake | 92.19 / 23.44 / 70.31 | 0 / 15 | −1.56 [−4.69,0] | +23.44 [14.02,34.38] | −23.44 [−34.38,−14.02] |

### Next experiment: boundary-reachability before production collection

1. First use this pilot's **training-only** saved pre-pulse states to diagnose the onset/amplitude response of a small fixed set of weaker, gradual five-step perturbations and braking/re-contact branches. Measure survival through the pulse and the next unchanged OpenVLA proposal's R5/H10 events. Preserve actual executed history, RGB rendering and H10/k5; simulated state teleportation or edited velocity is not a reachable-state witness. Selecting branches from offline labels is collector design only, never learned-runtime GT feedback.
2. The tested hypothesis is **the current onset/amplitude jumps directly into violation rather than reaching a pre-violation anticipatory region**. Earlier onset, weaker amplitude and gradual ramps are hypotheses to compare, not established fixes. Register their numerical schedules and screening/stop rules before obtaining new labels. Keep none and current brake references; root-group all branches. Do not blindly scale this failed drive collector or count post-violation windows as successful anticipation coverage.
3. If a reachable anticipatory region is found, freeze the observation-only collector rule and run a fresh train/design-dev pilot at the original physics ranges with the same root-level coverage criteria. If not, test whether anticipation is scarce under these in-range reachable states versus whether the collector fails to reach it; do not silently broaden physics ranges or relabel R5 hazards as H10-only.
4. Only after adequate coverage, run matched coverage-only regression controls followed by event-aware gate objectives, with original repair frozen and fresh calibration. Report SR/CVR/safe, paired rescue/harm and uncertainty before transfer decisions.

This review adds analysis and updates the plan; **no new collection or training job was launched**. The v2 numerical protocol remains to be registered.

## Question and decision

Can a fixed, bounded collector intervention produce diverse **reachable, pre-violation states** on which the unchanged OpenVLA's original proposal contains immediate or anticipatory contact hazards, without expanding the physical training ranges?

This is the next experiment following the completed gate × repair diagnosis. It is a data/label-feasibility experiment, not a new verifier efficacy result. No new gate is trained automatically. If coverage is inadequate, revise the collector hypothesis before spending another full training/evaluation budget. If it passes, use the evidence to design the matched regression-versus-event gate ablation with the original repair frozen.

## Registered collection design

- Full OpenVLA-7B + frozen H10 DAgger×2 head; forward microbatch 8; 16 decisions over 80 steps; k=5 for all collectors. Original weights, thresholds and runtime controller implementations are preserved.
- Training: four reset seeds **74000–74003**, each with 32 root episodes. Development/design: two reset seeds **75000–75001**, each with 32 roots. These blocks were checked against existing code/records before reservation. No confirmation, validation-grid or final-test episode is collected here.
- Each root is rolled out four times: **none**, **frozen verifier**, **one drive pulse**, **one brake pulse**. Total: **128 training roots × 4 = 512 trajectories**, **64 development roots × 4 = 256 trajectories**. There are **192 independent root initializations**, not 768 independent roots. Every branch and candidate of a root remains in the same split. Initial state, original proposal and physical parameters must match exactly across the four collectors.
- Hidden parameters are sampled identically for each paired root within existing ranges: friction multipliers **[0.5,1.5]**, mass **[0.6,1.6]**. Neither collector decisions nor subsequent learned gate inputs receive hidden parameters.
- Pulse trigger: first decision with observed positive x-goal gap below a per-episode onset sampled uniformly in **[0.04,0.16] m**. At most one five-step pulse is applied per episode. Subsequent policy calls use the actual resulting state and previous executed action.
- Drive pulse: proposal gain sampled in **[1.25,2.0]**, forward bias **[0.03,0.12] m/s**, lateral bias **[−0.02,+0.02] m/s**, clipped to simulator velocity limits ±0.5 m/s. Brake pulse: one pre-sampled factor from **[0.05,0.25,0.55]**. Both use current observation and reset-seeded constants only; future GT labels never select a pulse.
- These are offline data-collector interventions, not new deployment candidates. Candidate labels still cover the original policy proposal and the fixed eight scales **[1,.85,.70,.55,.40,.25,.15,.05]** at every actual decision. Record all contexts, states, original candidate risk traces, applied plans, physical multipliers, collector identity/pulse, prior-violation status and root IDs. States/parameters are stored for provenance and offline labels, not model features.
- For every applied plan, simulated first-five-step risk labels must match actual execution. Preserve per-step live actions/risks and final outcomes. This checks cache/pointer/history consistency as well as non-destructive shadow labels. Save batches and manifests atomically with hashes.

## Pre-registered pilot screening criteria

Count **original proposals only**, at decisions before any prior executed violation. Count unique root IDs, deduplicating decisions and collector branches.

| Event | Definition | Train minimum roots | Dev/design minimum roots |
|---|---|---:|---:|
| Immediate | max true risk at steps 1–5 > 1 | 32 | 16 |
| Anticipation-only | steps 1–5 safe, max true risk at steps 1–10 > 1 | 24 | 12 |
| Safe contact | 0.1 ≤ max true H10 risk ≤ 1 | 64 | 32 |

Additionally require each category in all four fixed physical strata (friction split at 1.0; mass at 1.1), with at least **2 train roots / 1 dev root per stratum**. The 0.1 safe-contact cutoff is a sampling/composition proxy, not a change to the physical violation limit 1. Report collector-specific coverage alongside aggregate coverage so an unsuccessful intervention is visible.

These are **pilot feasibility screens**, registered before production labels, not power calculations, generalization guarantees or deployment acceptance criteria. They deliberately prevent a few correlated hazardous windows from passing the screen. A pass still requires a fresh, adequately powered development/calibration protocol for model selection and inference; a fail blocks automatic training. The pilot dev split is for collector design and must not later be described as untouched final evaluation.

## Next algorithmic experiment after adequate coverage

Keep the original repair predictor and bisection fixed. Compare a separately learned gate against the current regression-score gate on matched data/architecture/budget. Supervise R5 and H10 events separately and retain anticipation; first isolate coverage effects, then the event-aware objective. Gate calibration uses fresh policy-decision episodes after collector selection. Any threshold rule is fixed using development/calibration, not the validation audit or confirmation/final-test. Report task completion alongside violation and safe success; the privileged diagnostic still lost 3.35–6.25 pp SR.

The completed factorial diagnosis supports prioritizing gating, but GT changes both information quality and decision semantics. It does not show that the allowed observation context is sufficient for an equally effective learned gate. Persistent failure after adequate coverage should motivate an observability/history test, not unconditional model scaling.

## Implementation verification

Added `src/experiments/collect_event_gate_pilot.py`, `scripts/collect_event_gate_pilot.sh`, and three tests for event categories/post-violation exclusion, deduplicated root counting, and pulse cache/history/commitment correctness. **45 targeted tests passed**, including existing CUDA differentiability and matched-control tests. Final CPU BC smoke `event_gate_coverage_smoke_20261012_b` completed all eight train/dev collector batches, including exact paired initialization/proposal/parameter checks and executed-prefix label parity. Its coverage screen failed, as expected for a tiny implementation smoke; it supplies no production coverage or efficacy evidence.

Production protocol and thresholds are written atomically before model loading and collection.

## Production launch

Launched **`event_gate_coverage_20261012_a`**, detached launcher PID **598105**, on healthy GPU UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`, using `setsid nohup`, stdin `/dev/null`, offline model loading and CPU thread caps. Log: `~/scratch/event_gate_coverage_20261012_a.log`. Output: `src/results/push_torch/event_gate_coverage_20261012_a/`. Full OpenVLA collection is the active job; event-gate fitting and efficacy evaluation have not started. The `passes_pilot` outcome is pending and must not be inferred from the successful launch or smoke.
