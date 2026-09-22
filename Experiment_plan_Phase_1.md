# Experiment Plan — Phase 1: Execution-Time Physical Verification with Genesis

**Dissertation:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization
**Framework:** VLA (base policy) + OrbiSim-Dynamics (fast differentiable object-centric predictor) + CheckVLA (runtime verifier/trigger) + Genesis (GPU ground-truth simulator & long-horizon rollout backend)
**Machine:** Local dev on Apple Silicon (MPS) for iteration · NVIDIA GPU (local or cloud) for Genesis + long-horizon differentiable rollouts
**Duration:** 3–4 weeks
**Builds on:** Phase 0 (`dissertation_exp/`, MLP BC + physics-guided residual correction, MuJoCo, completed) and Phase 1 draft benchmark plan (`Experiment_Phase_0.md`, Robosuite NutAssembly, Diffusion Policy)

> ⚠️ Note on sources: This plan operationalizes the "CheckVLA + OrbiSim-Dynamics + Genesis" architecture as discussed. I have **not independently verified** that "OrbiSim-Dynamics" and "CheckVLA" correspond to specific published papers — treat their names/descriptions here as *design references*, not confirmed citations. Before writing them into the dissertation as related work, locate and check the actual papers (arXiv IDs, authors, venues) yourself. Genesis itself (`genesis-world`, GitHub: Genesis-Embodied-AI/Genesis) is real and should be verified against its official docs for current API/backend support.

---

## 1. Research Question

> Can a fast, differentiable, object-centric physics predictor — continuously checked at runtime against a high-fidelity GPU simulator (Genesis) — detect and correct long-horizon physical constraint violations (excess deformation, contact-force overload) in a VLA policy's action sequence, **without** the system stalling on full-fidelity re-simulation at every control step?

This targets three sub-questions with independent, falsifiable experiments:

| Sub-question | What it tests |
|---|---|
| RQ1 | Does the OrbiSim-style fast predictor's forward rollout agree with Genesis ground truth well enough (state + gradient) to be trustworthy as a runtime verifier? |
| RQ2 | Does CheckVLA-style intervention (trigger + correction) improve task success / reduce constraint violations vs. an uncorrected VLA baseline, under distribution shift (novel friction/mass/deformable stiffness)? |
| RQ3 (systems contribution) | Do truncated checkpointing + adaptive gradient truncation + fast/slow dual-engine scheduling keep peak GPU memory and wall-clock latency bounded as horizon length increases, compared to naive full backprop-through-time in Genesis? |

RQ3 is the "hard, systems" contribution that differentiates this from a pure-ML manipulation paper — it should be the centerpiece figure of Phase 1.

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

---

## 4. Evaluation Metrics

| Metric | Definition | Maps to |
|---|---|---|
| Success Rate (SR) | task completion % across OOD grid | RQ2 |
| Constraint Violation Rate (CVR) | % episodes exceeding force/deformation thresholds | RQ2 |
| Predictor Fidelity | state-space error (OrbiSim vs Genesis) over rollout horizon | RQ1 |
| Gradient Agreement | cosine similarity of OrbiSim vs Genesis gradients at matched states | RQ1 |
| Peak GPU memory (MB) vs horizon length | with/without checkpointing | RQ3 |
| Wall-clock latency (ms/control step) | with/without fast/slow scheduling | RQ3 |
| Gradient norm variance | with/without adaptive truncation | RQ3 |
| Recovery Rate (RR) | reuse definition from `Experiment_Phase_0.md` §4 | RQ2, ties to Phase 0/1 |

**Target claims to validate:**
- CheckVLA-triggered correction reduces CVR by ≥30% vs. uncorrected baseline under OOD shift.
- Checkpointing reduces peak memory by ≥1 order of magnitude at horizon ≥100 steps, with SR/gradient-quality degradation <5%.
- Fast/slow scheduling keeps control-loop latency within real-time budget (define target, e.g. <50ms/step) while Genesis sync latency stays decoupled from the hot loop.

---

## 5. Experimental Steps (3–4 week plan)

### Week 1 — Genesis integration + control-task sanity check
- Install Genesis on the GPU machine; verify rigid-body + MPM/soft-body demos run.
- Port Task A (peg-in-hole) into Genesis; confirm parity with the existing MuJoCo/Robosuite version from Phase 0/1 (same friction/mass grid, same SR ballpark) — this is a *reproduction* checkpoint before trusting Genesis for anything new.
- Stand up basic profiling: FPS, memory, per-step latency for rigid vs soft-body sim.

### Week 2 — OrbiSim-Dynamics predictor + CheckVLA trigger logic
- Implement the fast object-centric predictor (distilled from short Genesis rollouts on Task A, then Task B).
- Implement CheckVLA trigger conditions (deformation/force thresholds, confidence collapse).
- Validate RQ1: predictor fidelity + gradient agreement vs Genesis on held-out rollouts.

### Week 3 — Systems optimizations (RQ3) + Task B (deformable) integration
- Implement truncated checkpointing; benchmark peak memory vs horizon length (with/without).
- Implement adaptive gradient truncation; benchmark gradient stability.
- Implement fast/slow async scheduling; benchmark control-loop latency.
- Bring up Task B (cloth folding) in Genesis; extend OOD grid to stiffness.

### Week 4 — Full evaluation grid + ablations + figures
- Run full OOD grid (Task A + Task B) for: [uncorrected VLA baseline, Phase-0-style residual correction, full CheckVLA+OrbiSim+Genesis pipeline].
- Run RQ3 ablations: [no checkpointing / checkpointing only / + adaptive truncation / + fast-slow scheduling — full system].
- Generate figures (see §6) and write up results.

---

## 6. Planned Figures

| Figure | Content | Maps to |
|---|---|---|
| Fig A | SR/CVR heatmap over OOD grid — baseline vs Phase-0 residual vs full pipeline | RQ2 |
| Fig B | Predictor-vs-Genesis state error over rollout horizon (fidelity decay curve) | RQ1 |
| Fig C | Peak GPU memory vs horizon length — with/without checkpointing (log-scale) | RQ3 |
| Fig D | Control-loop latency (ms/step) — with/without fast/slow scheduling | RQ3 |
| Fig E | Gradient norm trace over a chaotic-contact episode — with/without adaptive truncation | RQ3 |
| Fig F | CheckVLA trigger trace on one Task B episode: predicted deformation vs threshold vs intervention point | RQ2 |

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
| OrbiSim-Dynamics distillation predictor doesn't match Genesis well (RQ1 fails) | Medium-High | Treat as a valid negative result — report fidelity limits, restrict CheckVLA triggers to regimes where predictor is validated |
| Deformable task (Task B) too unstable/slow to iterate on in 1 week | Medium | Keep Task A as the primary result; Task B becomes a smaller-scale qualitative demo if time runs out |
| Checkpointing/scheduling engineering takes longer than 1 week (this is real systems work) | High | This is the highest-risk, highest-value item — start a minimal version in Week 2 in parallel, not only Week 3 |
| Cloud GPU cost/availability | Low-Medium | Confirm quota before Week 1; scope horizon lengths/episode counts to fit budget |
| "OrbiSim-Dynamics"/"CheckVLA" naming doesn't match a real citable paper | Medium | Verify literature before final write-up (see note at top); rename as your own architecture if no exact match found, and cite the closest real prior work (e.g., differentiable object-centric contact models, runtime verification for RL) instead |

---

## 9. Deliverables at End of Phase 1

1. Genesis-based Task A + Task B environments with reproducible OOD grids.
2. OrbiSim-Dynamics predictor + CheckVLA trigger implementation with RQ1 fidelity report.
3. Three systems optimizations (checkpointing, gradient stabilizer, fast/slow scheduling) each with an isolated ablation benchmark.
4. Full SR/CVR comparison: baseline vs Phase-0 residual vs full pipeline, across OOD grid.
5. Figures A–F, ready for inclusion in the dissertation systems-contribution chapter.

---

*Last updated: 2026-09-22*
