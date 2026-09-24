# Experiment Plan — Phase 1: Execution-Time Physical Verification with Genesis

**Dissertation:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization
**Framework:** VLA (base policy) + OrbiSim-Dynamics (fast differentiable object-centric predictor) + CheckVLA (runtime verifier/trigger) + Genesis (GPU ground-truth simulator & long-horizon rollout backend)
**Machine:** Local dev on Apple Silicon (MPS) for iteration · NVIDIA GPU (local or cloud) for Genesis + long-horizon differentiable rollouts
**Duration:** 3–4 weeks
**Builds on:** Phase 0 (`dissertation_exp/`, MLP BC + physics-guided residual correction, MuJoCo, completed) and Phase 1 draft benchmark plan (`Experiment_Phase_0.md`, Robosuite NutAssembly, Diffusion Policy)

> ✅ Note on sources (updated 2026-09-24): Both prior-work papers are now confirmed as real, citable arXiv papers:
> - **OrbiSim** — arXiv `2605.16395`, Jiajian Li et al. Redefines world models as fully differentiable physics engines, decoupled into OrbiSim-Dynamics + OrbiSim-Vision, enabling end-to-end differentiability.
> - **CheckVLA** — arXiv `2607.26789`, Yushan Liu et al. Core mechanism: action-conditioned world model + calibrated-threshold trigger + latency-aware suffix repair. Ablating the action-conditioning drops timely recall from 77.9% to 48.6%.
>
> This means the contribution framing shifts from "design reference, not yet verified" to a precise, defensible claim:
>
> **Contribution statement:** Use Genesis as ground truth to validate OrbiSim-Dynamics as the core predictor component of a CheckVLA-style verifier, and to resolve OrbiSim-Dynamics' long-horizon differentiable-rollout systems bottleneck (memory growth under backprop-through-time).
>
> Genesis itself (`genesis-world`, GitHub: Genesis-Embodied-AI/Genesis) remains real and should still be checked against its official docs for current API/backend support — see §3.6 below for known differentiability limits that directly affect RQ1.

---

## 1. Research Question

> Can a fast, differentiable, object-centric physics predictor — continuously checked at runtime against a high-fidelity GPU simulator (Genesis) — detect and correct long-horizon physical constraint violations (excess deformation, contact-force overload) in a VLA policy's action sequence, **without** the system stalling on full-fidelity re-simulation at every control step?

This targets three sub-questions with independent, falsifiable experiments:

| Sub-question | What it tests |
|---|---|
| RQ1 (narrowed) | Within the CheckVLA trigger logic specifically, does replacing the visual world-model predictor with OrbiSim-Dynamics' differentiable object-centric predictor produce **better-calibrated risk signals** (trigger precision/recall, timely-recall) and **earlier/more accurate repair timing**, relative to Genesis ground truth — and does this hold only within the subset of contact regimes where Genesis provides valid gradients (see §3.6 gradient path audit)? |
| RQ2 | Does CheckVLA-style intervention (trigger + correction), instantiated with the OrbiSim-Dynamics predictor, improve task success / reduce constraint violations vs. (a) an uncorrected VLA baseline and (b) a CheckVLA variant using an observation-only / vision world-model predictor, under distribution shift (novel friction/mass/deformable stiffness)? |
| RQ3 (systems contribution) | Do truncated checkpointing + adaptive gradient truncation + fast/slow dual-engine scheduling keep peak GPU memory and wall-clock latency bounded as horizon length increases, compared to naive full backprop-through-time in Genesis? |

RQ3 is the "hard, systems" contribution that differentiates this from a pure-ML manipulation paper — it should be the centerpiece figure of Phase 1.

**Why RQ1 was narrowed:** OrbiSim's own paper already reports predictor fidelity and RL control performance for OrbiSim-Dynamics in isolation, so re-measuring raw predictor accuracy would be a redundant, weaker replication. CheckVLA's paper already shows action-conditioned signals beat observation-only signals, but using a *visual* world model. The open, falsifiable, non-redundant question this dissertation can answer is narrower and more specific: does swapping in a *differentiable physics* predictor (OrbiSim-Dynamics) inside the CheckVLA trigger — instead of a visual world model — further improve calibration and repair timing, and where does that improvement break down due to Genesis's differentiability limits.

---

## 2. Benchmark & Task

### 2.1 Primary task: deformable-object contact-rich manipulation in Genesis

| Property | Value |
|---|---|
| Simulator | Genesis (`genesis-world`), rigid + MPM/FEM soft-body solver |
| Task A (rigid, control) | Peg-in-hole / socket insertion — reuse geometry from Robosuite NutAssembly for continuity with Phase 0/1 draft |
| Task B (deformable, headline) | Cloth/garment folding corner-grasp — the flagship "deformable manipulation" scenario from the proposal |
| Horizon | 50–150 control steps (long enough to make BPTT memory/chaos issues visible) |
| Distribution shift axes | friction (as in Phase 0/1), cloth stiffness / bending modulus, object mass |

**Why keep Task A (rigid) as a control:** it lets you validate the Genesis pipeline and the verifier logic on a *simple, already-understood* task (reusing Phase 0/1 infrastructure) before trusting results on the harder deformable task, where ground truth intuition is weaker.

### 2.2 OOD test protocol (extends Phase 0/1)

| Condition | Train value | OOD test values |
|---|---|---|
| Friction | nominal | ±40%, ±80% |
| Cloth stiffness (Task B only) | nominal | 0.5×, 2×, 4× |
| Object mass | nominal | ±50%, ×2 |

Same protocol style as `Experiment_Phase_0.md` §2.2 — reuse the friction/mass grid, add stiffness for the deformable task.

---

## 3. System Components

### 3.1 Base policy (VLA proxy)

Reuse Phase 1 draft's Diffusion Policy (or the simpler MLP BC from Phase 0 if time-constrained) as the "VLA" stand-in. Do **not** build a real multimodal VLA from scratch in Phase 1 — that is out of scope for a 3–4 week window. Document this substitution explicitly as a limitation in the write-up.

### 3.2 OrbiSim-Dynamics (fast engine): differentiable object-centric predictor

- Lightweight learned or analytic differentiable model that predicts object-centric state deltas (position, contact points, deformation field summary) N steps ahead, given the policy's proposed action sequence.
- Implementation option (time-boxed): a small MLP/GNN trained to imitate short Genesis rollouts (distillation), differentiable end-to-end in PyTorch — this is the pragmatic Phase 1 scope, rather than a from-scratch novel differentiable contact model.
- Runs at every control step, in-process, on GPU SRAM — must be fast (<5ms/step target).

### 3.3 CheckVLA (validator / trigger)

- Consumes OrbiSim-Dynamics' predicted trajectory + a confidence/variance signal.
- Triggers intervention when:
  1. predicted deformation gradient exceeds a tear/damage threshold, or
  2. predicted contact force exceeds a torque/force limit, or
  3. predictor confidence collapses (entering a chaotic contact regime).
- On trigger: fall back to (a) querying Genesis for a corrected/re-planned short rollout, or (b) applying the Phase 0 physics-correction residual as an immediate cheap fallback while Genesis catches up asynchronously.

### 3.4 Genesis (slow engine / ground truth)

- Provides the authoritative rollout used for (i) validating OrbiSim-Dynamics' predictions (RQ1), (ii) supplying gradients for online policy correction when CheckVLA triggers (RQ2), and (iii) the long-horizon backprop-through-time benchmark for the systems contribution (RQ3).

### 3.5 The three systems optimizations (RQ3 — core contribution)

1. **Truncated differentiable checkpointing** — only retain the last *N* physics states in GPU memory; offload older states to CPU; recompute on backward pass as needed. Measure peak GPU memory vs. horizon length, with/without checkpointing.
2. **Adaptive gradient truncation ("gradient stabilizer")** — detect high-variance/chaotic contact windows (via OrbiSim-Dynamics confidence or gradient-norm spikes) and clip/redirect to a smoothed relaxation gradient from Genesis in those windows only. Measure gradient norm stability and downstream policy-update quality with/without this.
3. **Fast/slow dual-engine async scheduling** — pipeline OrbiSim-Dynamics (fast, every step) against Genesis (slow, background sync) using CUDA streams / async job queue. Measure end-to-end control-loop latency (ms/step) and how often CheckVLA's decision is stale relative to Genesis ground truth.

These three are independently testable ablations — see §4.

### 3.6 Gradient path audit (new — do this first, Week 1)

Genesis's differentiable-physics support has explicit, documented limits that directly bound what RQ1/RQ3 can measure:

- Differentiable mode must be explicitly enabled per-object (`requires_grad=True`); it is not on by default.
- **Peak memory under backprop-through-time grows linearly with horizon length** — this is exactly the systems problem RQ3 targets, but it means RQ1's "Gradient Agreement" metric is *coupled* to RQ3's memory/checkpointing work from day one, not a separable earlier step.
- **Not all operations are differentiable.** Contact/collision resolution can return zero or undefined gradients in some regimes; elliptical friction cones are not supported in the differentiable path; the SAP (Semi-Analytic Primal) coupler does not support differentiation.

**Action (Week 1, before RQ1 experiments start):** run a small audit sweep over Task A and Task B contact conditions, tagging each state/contact regime as `gradient-valid` or `gradient-invalid` (zero/NaN/undefined gradient from Genesis). Use this map to:
1. Restrict RQ1's "Gradient Agreement" metric to `gradient-valid` regions only, and report the fraction of the OOD grid that falls outside valid coverage as a limitation.
2. Scope CheckVLA's OrbiSim-Dynamics-based trigger to only claim gradient-based confidence inside `gradient-valid` regions; fall back to state-error-only confidence elsewhere.

### 3.7 Baseline arm: vision world-model CheckVLA (new)

To isolate the effect of swapping in OrbiSim-Dynamics (narrowed RQ1/RQ2), the evaluation must include a CheckVLA variant using an observation-only / vision world-model predictor (as in the original CheckVLA paper) as a direct baseline. Without this arm, there is no way to attribute improvements in trigger calibration or repair timing to the differentiable-physics predictor specifically, rather than to the CheckVLA scaffolding itself. This is now a required condition in the Week 4 evaluation grid (§5, §6).

---

## 4. Evaluation Metrics

| Metric | Definition | Maps to |
|---|---|---|
| Success Rate (SR) | task completion % across OOD grid | RQ2 |
| Constraint Violation Rate (CVR) | % episodes exceeding force/deformation thresholds | RQ2 |
| Predictor Fidelity | state-space error (OrbiSim vs Genesis) over rollout horizon, **restricted to gradient-valid regions per §3.6 audit** | RQ1 |
| Gradient Agreement | cosine similarity of OrbiSim vs Genesis gradients at matched states, **reported only within gradient-valid regions; coverage fraction reported separately** | RQ1 |
| Trigger calibration (precision/recall, timely recall) | CheckVLA trigger accuracy vs. ground-truth violation, comparing OrbiSim-Dynamics predictor vs. vision-world-model predictor (§3.7) | RQ1, RQ2 |
| Peak GPU memory (MB) vs horizon length | with/without checkpointing | RQ3 |
| Wall-clock latency (ms/control step) | with/without fast/slow scheduling | RQ3 |
| Gradient norm variance | with/without adaptive truncation | RQ3 |
| Recovery Rate (RR) | reuse definition from `Experiment_Phase_0.md` §4 | RQ2, ties to Phase 0/1 |

**Target claims to validate:**
- OrbiSim-Dynamics-based CheckVLA trigger improves timely-recall and/or repair-timing accuracy vs. the vision-world-model CheckVLA baseline, within gradient-valid regions.
- CheckVLA-triggered correction reduces CVR by ≥30% vs. uncorrected baseline under OOD shift.
- Checkpointing reduces peak memory by ≥1 order of magnitude at horizon ≥100 steps, with SR/gradient-quality degradation <5%.
- Fast/slow scheduling keeps control-loop latency within real-time budget (define target, e.g. <50ms/step) while Genesis sync latency stays decoupled from the hot loop.

---

## 5. Experimental Steps (3–4 week plan)

### Week 1 — Genesis integration + control-task sanity check + gradient path audit
- Install Genesis on the GPU machine; verify rigid-body + MPM/soft-body demos run.
- Port Task A (peg-in-hole) into Genesis; confirm parity with the existing MuJoCo/Robosuite version from Phase 0/1 (same friction/mass grid, same SR ballpark) — this is a *reproduction* checkpoint before trusting Genesis for anything new.
- Stand up basic profiling: FPS, memory, per-step latency for rigid vs soft-body sim.
- **Run the gradient path audit (§3.6):** sweep Task A/B contact conditions, tag `gradient-valid` vs `gradient-invalid` regions (zero/NaN/undefined gradients, unsupported friction-cone/coupler paths). This map gates what RQ1 can measure and where CheckVLA's OrbiSim-based confidence signal is trustworthy.
- **Start a minimal checkpointing prototype in parallel (moved up from Week 3):** crude version — keep only the last ~5 physics states resident on GPU, offload the rest to CPU, recompute on backward pass. Goal at this stage is only to confirm it reduces peak memory at all; refine in Week 2.

### Week 2 — OrbiSim-Dynamics predictor + CheckVLA trigger logic + checkpointing v1
- Implement the fast object-centric predictor (distilled from short Genesis rollouts on Task A, then Task B).
- Implement CheckVLA trigger conditions (deformation/force thresholds, confidence collapse), scoped to gradient-valid regions per the Week 1 audit.
- Implement the vision-world-model CheckVLA baseline (§3.7) so RQ1/RQ2 have a direct comparison arm.
- Validate narrowed RQ1: trigger calibration + repair-timing accuracy, OrbiSim-Dynamics vs vision-world-model predictor, within gradient-valid regions.
- Refine the Week 1 checkpointing prototype into a real truncated-checkpointing implementation; get an early peak-memory-vs-horizon curve, even if rough.

### Week 3 — Remaining systems optimizations (RQ3) + Task B (deformable) integration
- Finish benchmarking checkpointing (peak memory vs horizon length, with/without) using the Week 2 implementation.
- Implement adaptive gradient truncation; benchmark gradient stability.
- Implement fast/slow async scheduling; benchmark control-loop latency.
- Bring up Task B (cloth folding) in Genesis; extend OOD grid to stiffness; re-run the gradient path audit for Task B's contact modes.

### Week 4 — Full evaluation grid + ablations + figures
- Run full OOD grid (Task A + Task B) for: [uncorrected VLA baseline, Phase-0-style residual correction, CheckVLA + vision-world-model predictor, CheckVLA + OrbiSim-Dynamics predictor (full pipeline)].
- Run RQ3 ablations: [no checkpointing / checkpointing only / + adaptive truncation / + fast-slow scheduling — full system].
- Generate figures (see §6) and write up results, including the gradient-valid coverage fraction as an explicit limitation.

---

## 6. Planned Figures

| Figure | Content | Maps to |
|---|---|---|
| Fig A | SR/CVR heatmap over OOD grid — baseline vs Phase-0 residual vs CheckVLA(vision) vs CheckVLA(OrbiSim-Dynamics) | RQ2 |
| Fig B | Predictor-vs-Genesis state error over rollout horizon (fidelity decay curve), gradient-valid region annotated | RQ1 |
| Fig B2 | Gradient-valid vs gradient-invalid coverage map over Task A/B contact conditions (audit result) | RQ1, RQ3 |
| Fig C | Peak GPU memory vs horizon length — with/without checkpointing (log-scale) | RQ3 |
| Fig D | Control-loop latency (ms/step) — with/without fast/slow scheduling | RQ3 |
| Fig E | Gradient norm trace over a chaotic-contact episode — with/without adaptive truncation | RQ3 |
| Fig F | CheckVLA trigger trace on one Task B episode: predicted deformation vs threshold vs intervention point, OrbiSim-Dynamics vs vision-world-model predictor overlaid | RQ1, RQ2 |

---

## 7. Connection to Dissertation Narrative

| Proposal / Revised-Proposal Section | This Experiment |
|---|---|
| Genesis integration (revised proposal §1) | Core simulator backend, Task A + B |
| Causal/structured Physical CoT | CheckVLA trigger reasoning trace = a constrained instance of Physical CoT (Observation → Constraint → Intervention) |
| "Long-horizon differentiable gradient" systems challenge | Directly addressed by §3.5 (checkpointing, gradient stabilizer, dual-engine scheduling) |
| Expected Outcome 2 (simulation env supporting gradient backprop) | Genesis + OrbiSim-Dynamics pipeline built in Weeks 1–3 |
| Outstanding completion criterion | RQ3 systems contribution + Task B zero-shot generalization results |
| Phase 0 (`dissertation_exp/`) | Physics-correction residual reused as one baseline arm; friction/mass OOD grid reused |
| Phase 1 draft (`Experiment_Phase_0.md`) | Task A geometry/metrics reused for Genesis parity check |

---

## 8. Risk & Mitigation

| Risk | Probability | Mitigation |
|---|---|---|
| Genesis install/build issues on target GPU | Medium | Budget Day 1–2 of Week 1 as buffer; fallback to Docker image if available; keep MuJoCo path as backup for Task A only |
| Genesis's differentiable path lacks gradient support for key contact modes (elliptical friction cone, SAP coupler, some collision paths) | Medium-High | Week 1 gradient path audit (§3.6) makes this explicit up front; scope RQ1/CheckVLA gradient-based confidence to gradient-valid regions only, and report coverage fraction as a limitation rather than treating it as a late-discovered blocker |
| OrbiSim-Dynamics distillation predictor doesn't match Genesis well within gradient-valid regions (narrowed RQ1 fails) | Medium-High | Treat as a valid negative result — report calibration limits, restrict CheckVLA triggers to regimes where predictor is validated |
| Deformable task (Task B) too unstable/slow to iterate on in 1 week | Medium | Keep Task A as the primary result; Task B becomes a smaller-scale qualitative demo if time runs out |
| Checkpointing/scheduling engineering takes longer than 1 week (this is real systems work) | High | Highest-risk, highest-value item — minimal checkpointing prototype now starts Week 1 in parallel with the gradient audit, refined in Week 2, fully benchmarked in Week 3, rather than starting cold in Week 3 |
| Cloud GPU cost/availability | Low-Medium | Confirm quota before Week 1; scope horizon lengths/episode counts to fit budget |
| Missing baseline arm makes it impossible to attribute gains to OrbiSim-Dynamics specifically | Medium | Vision-world-model CheckVLA baseline (§3.7) is now a required Week 2 deliverable, not optional |

---

## 9. Deliverables at End of Phase 1

1. Genesis-based Task A + Task B environments with reproducible OOD grids.
2. Gradient path audit (§3.6): map of gradient-valid vs gradient-invalid contact regimes in Genesis for Task A/B.
3. OrbiSim-Dynamics predictor + CheckVLA trigger implementation with narrowed-RQ1 calibration report, benchmarked against a vision-world-model CheckVLA baseline.
4. Three systems optimizations (checkpointing, gradient stabilizer, fast/slow scheduling) each with an isolated ablation benchmark — checkpointing prototyped from Week 1, not Week 3.
5. Full SR/CVR comparison: baseline vs Phase-0 residual vs CheckVLA(vision) vs CheckVLA(OrbiSim-Dynamics), across OOD grid.
6. Figures A, B, B2, C–F, ready for inclusion in the dissertation systems-contribution chapter.

---

*Last updated: 2026-09-24*
