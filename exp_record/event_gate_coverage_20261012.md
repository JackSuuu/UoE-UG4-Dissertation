# Event-gate coverage pilot protocol (12 Oct 2026)

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
