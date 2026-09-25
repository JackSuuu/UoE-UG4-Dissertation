# Experiment Plan — One-Year Version (Oct 2026 – Sep 2027)

**Dissertation:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization
**Replaces:** `Experiment_plan_Phase_1.md` (3–4 week validation plan). That plan remains the blueprint for Months 1–3 infrastructure.
**Prior work used:** OrbiSim (Li et al., arXiv 2605.16395), CheckVLA (Liu et al., arXiv 2607.26789), Genesis (`genesis-world`)
**Last updated:** 2026-09-25

---

## 0. What changes with one year instead of one month

A one-month plan can only run **comparative validation experiments**. A one-year plan should **build something with a clear contribution boundary**. That means:

1. **Research loop, not engineering pipeline.** "Baseline, then VLA, then demo" is an engineering order. The research order is: fix the one question to answer in 12 months → build only the infrastructure that question needs → iterate, with room for failure and change of direction.
2. **Pick one main contribution.** Of the three candidate directions (physical reasoning chain / differentiable verifier / system optimisation), one is the main contribution and the others support it. The three are not treated equally.
3. **Real-world evidence becomes necessary.** A year of simulation-only work invites the objection "self-validation inside a simulator". This plan keeps a small, low-risk real-data validation (offline, no full closed loop).

---

## 1. Choosing the main contribution (decision needed — Month 1, Week 2)

| Direction | What the thesis becomes | Fit with the proposal | 1-year risk | Recommended role |
|---|---|---|---|---|
| **A. Differentiable verifier** (CheckVLA-style, physics predictor inside) | Replacing the visual world model with a differentiable physics predictor as the runtime verifier: *when is its signal reliable, and for which constraints* | High — directly "neural physics engine inside an embodied foundation model" | Medium | **Main contribution (recommended)** |
| **B. System optimisation** (checkpointing, gradient truncation, fast/slow scheduling) | A gradient-transport system for differentiable physics in long-horizon VLA loops | Medium — close to a systems paper, further from "physical understanding" | Medium-low | **Support**: makes A feasible at long horizons; its own chapter |
| **C. Physical reasoning chain** (explicit causal physical-constraint chain in VLA reasoning) | VLA reasoning that explicitly includes physical-constraint causality | Highest ambition | High — three open sub-problems (extracting constraints / organising them into a chain / validating the chain) | **Light support only**: the verifier's trigger trace (observation → constraint → intervention) is treated as a constrained, checkable instance of a physical reasoning chain; no free-form CoT generation |

**Recommendation: A main, B support, C as interpretation layer.** This keeps the proposal's narrative, keeps B's systems depth as a second contribution chapter, and avoids C's open-endedness.

> If you choose **B** as main: the core experiments in §4 become the "workload" and §5 becomes the centre. If you choose **C**: this plan needs a rewrite. Do not start Month 2 until this is decided.

The rest of this document assumes **A main / B support / C light**.

---

## 2. Core research question

> **In long-horizon, contact-rich manipulation, how does the quality of the signal from a differentiable physics predictor — used as the core of a runtime verifier — change with task complexity, prediction horizon and constraint type? And what system-level design keeps that signal usable within acceptable resource cost?**

Three levels, each an upgrade of the Phase-1 RQ:

| RQ | Upgrade of | Question | Falsifiable hypothesis |
|---|---|---|---|
| **RQ1 — Signal quality** | Phase-1 RQ1 | Not "is the prediction accurate" but "under which conditions are the predictor's risk and gradient signals reliable" (horizon, contact regime, OOD distance, Genesis gradient-path validity) | H1: trigger AUROC and gradient agreement fall off sharply beyond a critical horizon *h\** that depends on contact regime. Beyond *h\** the physics predictor is no better than the vision WM. |
| **RQ2 — Trigger effectiveness per constraint type** | Phase-1 RQ2 | Not "does triggering improve SR" but "how do force / deformation / contact-mode constraints differ in what they need from the trigger signal" (lead time, calibration, repairability) | H2: the physics predictor's advantage over the vision WM is largest for constraints with hidden physical state (force, internal strain), and smallest for visually salient ones (large visible deformation). |
| **RQ3 — System feasibility** | Phase-1 RQ3 | Not "does checkpointing reduce memory" but "within which resource envelope (GPU memory, control latency) the verifier architecture remains usable, and what it degrades to outside it" | H3: with checkpointing + adaptive truncation + async scheduling, verifier quality at horizon *h\** is kept within budget (e.g. <50 ms/step, one consumer GPU), and naive BPTT cannot reach *h\**. |

**Primary deliverable:** a *reliability map* — signal quality as a function of (horizon × constraint type × contact regime × OOD shift) — plus the system that makes the usable region of that map reachable.

---

## 3. Timeline overview

Core experiments and system optimisation run **in parallel, not in series**. System limits decide which experiment designs are feasible, and experiment needs decide what to optimise.

| Phase | Months | Calendar | Main task | Milestone / gate |
|---|---|---|---|---|
| **P1 Infrastructure** | 1–3 | Oct–Dec 2026 | Genesis integration, real components in place of stand-ins, gradient path audit, end-to-end predict → trigger → repair | **G1 (end Dec):** end-to-end loop runs in Genesis with real OrbiSim-Dynamics (or a justified substitute) |
| **P2 Core experiments** | 4–7 | Jan–Apr 2027 | Vary horizon, task complexity, constraint type; measure signal quality & trigger effectiveness | **G2 (end Mar):** first reliability map for Task A; *h\** located or H1 rejected |
| **P3 System optimisation** | 6–9 (parallel) | Mar–Jun 2027 | Checkpointing / truncation / scheduling ablations at the horizons P2 needs | **G3 (end Jun):** usable-envelope result (H3) |
| **P4 Real-data validation** | 8–10 | May–Jul 2027 | Offline trigger tests on real robot trajectories | **G4 (end Jul):** signal meaningful on real data, or a documented sim-to-real gap |
| **P5 Converge & write** | 10–12 | Jul–Sep 2027 | Complementary experiments, figures, writing | Full thesis draft end Aug; final Sep |

Buffer: about 1 month total, spread across P2/P3 overlap. If a gate fails, see §8.

---

## 4. Phase details

### P1 — Infrastructure (Months 1–3)

Starting point: the `src/` framework (plug-in interfaces, stand-ins, Genesis backend written but never run). See `src/README.md` for component status.

| Month | Work | Output |
|---|---|---|
| **M1** | Run Genesis on the server: `audit_gradients.py --backend genesis --genesis_probe`, Franka demo recording. Fix API issues. **Decide main contribution (§1).** Literature check: is OrbiSim / CheckVLA code public? | Genesis working; Franka demo mp4; decision recorded |
| **M2** | Replace stand-ins: real OrbiSim-Dynamics via `adapters/orbisim_official.py` (or, if unavailable, a documented re-implementation following the paper — named "OrbiSim-style", not "OrbiSim"). Real/official CheckVLA logic if public; otherwise the reference verifier with its design written up. Risk head for the predictor (route a or b in the adapter). | Real components behind the interfaces |
| **M3** | **Gradient path audit on Genesis** for Task A (rigid) and a Genesis-native deformable task (MPM/FEM, since PBD cloth is not differentiable). End-to-end loop on Genesis. Minimal checkpointing prototype started (the P3 system track starts early, as in the Phase-1 review). | Audit map (Fig B2); **G1** |

Base policy decision (M2): keep the BC/Diffusion-policy proxy for the core experiments (cheap, controllable), and use a real VLA (OpenVLA or a chunked VLA) only for a **transfer check** in P2/P5. The thesis question is about the verifier, not the policy.

### P2 — Core experiments (Months 4–7)

A factorial design over the axes the RQs name. Every cell compares **physics predictor vs. vision WM vs. hybrid** inside the same verifier.

| Axis | Levels | Notes |
|---|---|---|
| Prediction horizon *h* | 5, 10, 20, 40, 80 steps | locate *h\** (H1) |
| Constraint type | contact force · deformation/strain · contact-mode change (slip/stick, making/breaking contact) | H2 |
| Task complexity | T1 rigid push-insertion → T2 multi-contact rigid (e.g. peg-in-hole with tight clearance) → T3 deformable (Genesis MPM/FEM) | increasing contact richness |
| OOD shift | friction, mass, stiffness grids (from Phase 1), plus distance-to-training-range | calibration under shift |
| Predictor | physics (OrbiSim-Dynamics) · vision WM · hybrid (physics state + visual residual) | the hybrid is optional, added if the first two are close |

Measurements for every cell:
- **Signal quality:** risk calibration (reliability curve, ECE), AUROC, gradient agreement with Genesis (gradient-valid regions only), fidelity decay vs horizon
- **Trigger effectiveness:** precision/recall, timely recall, lead time, repair success (does the repaired suffix avoid the violation in Genesis), SR/CVR under closed loop
- **Cost:** predictor ms/step, memory

Order: M4 T1 all axes → M5 T2 → M6–7 T3. After **G2 (end Mar)**, T2/T3 cover only the informative axis levels, pruned by what T1 showed.

### P3 — System optimisation (Months 6–9, parallel with P2)

The three components already exist in `src/systems/` as prototypes. P3 turns them into a studied system:

| Component | Research question | Ablation |
|---|---|---|
| Truncated checkpointing + CPU offload | memory vs horizon vs recompute cost; best segment size per task | naive / GPU ckpt / offload / offload+truncation, horizons up to 400 steps |
| Adaptive gradient truncation | does relaxation/clipping in chaotic contact windows improve repair quality, or only gradient variance? | none / clip / smooth-relax, per contact regime from the audit |
| Fast/slow async scheduling | how stale can slow-engine (Genesis) verdicts be before they stop helping? | fast-only / async (varying slow latency) / sync |

Output: the **usable-resource envelope** (Fig: verifier quality vs memory budget × latency budget), answering H3.

### P4 — Real-data validation (Months 8–10)

Minimal and offline — **no full VLA closed loop on hardware**:
1. Get real trajectories with physical measurements (contact force from a wrist F/T sensor; deformation from vision/markers where possible). Options, in order of preference: (a) a lab robot at Edinburgh, if access can be arranged — **ask the supervisor in M1**; (b) public manipulation datasets that include force/torque — candidates to be identified and checked in M1.
2. Run the verifiers offline on these trajectories: does the physics predictor's trigger fire before measured force/deformation limit crossings?
3. Report the same RQ1/RQ2 metrics on real data, and the sim-to-real gap relative to Genesis.

If no real data with force labels can be obtained by M6, P4 shrinks to a **sim-to-sim transfer** study (train in the torch GT, test in Genesis, or across Genesis solver settings), and this is stated as a limitation.

### P5 — Convergence & writing (Months 10–12)

- Complementary experiments requested by the gates and supervisor feedback
- Real-VLA transfer check: the verifier on top of a real VLA policy for a subset of Task A cells
- Demo: Franka arm video in Genesis (`src/demo/`), normal vs OOD vs OOD + verifier
- Writing: chapter plan in §7

---

## 5. Metrics & figures (thesis-level)

| Figure | Content | RQ |
|---|---|---|
| F1 | System diagram: policy → predictor → verifier → repair; fast/slow engines | — |
| F2 | Gradient path audit map (Genesis), per task × contact regime | RQ1 |
| F3 | **Reliability map:** AUROC / calibration vs horizon × constraint type, physics vs vision | RQ1 (headline) |
| F4 | Fidelity and gradient-agreement decay vs horizon, with *h\** marked | RQ1 |
| F5 | Trigger lead time & repair success per constraint type | RQ2 |
| F6 | SR/CVR over OOD grids, 4–5 arms | RQ2 |
| F7 | Memory / latency vs horizon, with and without each optimisation | RQ3 |
| F8 | Usable envelope: verifier quality vs resource budget | RQ3 |
| F9 | Real-data offline trigger traces vs measured force | P4 |
| F10 | Verifier trace as a physical reasoning chain (observation → constraint → intervention) | C (interpretation) |

---

## 6. What the existing `src/` code covers

| Needed | Exists in `src/` | Gap |
|---|---|---|
| Plug-in interfaces for policy / predictor / verifier / GT | `interfaces.py`, `registry.py` | — |
| Genesis GT | `sims/genesis_push.py` (Task A, never run) | run and fix; add T2 and a differentiable deformable task (MPM/FEM) |
| Real OrbiSim / CheckVLA | adapter templates only | M2 |
| Gradient path audit | `experiments/audit_gradients.py` (torch GT + Genesis probe) | per-regime Genesis audit |
| RQ1/RQ2 metrics | `rq1_calibration.py`, `rq2_eval.py` | horizon & constraint-type sweeps, calibration curves/ECE, repair success |
| System track | `systems/bptt.py`, `systems/scheduler.py` | Genesis-native checkpointing; latency sweeps |
| Demo | `demo/genesis_franka_demo.py` (never run) | run; add target marker, OOD comparison video |
| Real data | — | P4 |

Known issue carried over: the stand-in predictor under-predicts rare risk spikes (fix written, unverified). Real predictors will need the same check: rare-event risk calibration is part of RQ1.

---

## 7. Thesis structure (target)

1. Introduction — physical reliability of VLAs; why runtime verification
2. Background — VLAs, world models, differentiable physics (Genesis, OrbiSim), runtime verification (CheckVLA)
3. System — architecture, predictors, verifier, Genesis integration, gradient path audit
4. **Signal quality & trigger effectiveness (RQ1, RQ2)** — main contribution
5. **Making it usable: system design (RQ3)** — supporting contribution
6. Real-data validation (P4)
7. Discussion — the verifier as a physical reasoning chain; limitations (policy proxy, Genesis gradient coverage, sim-to-real)
8. Conclusion

---

## 8. Risks & decision gates

| Risk | Probability | Mitigation / fallback |
|---|---|---|
| OrbiSim / CheckVLA code not public | Medium-High | Documented re-implementations named "OrbiSim-style" / "CheckVLA-style"; the contribution statement is about *differentiable physics predictors* as a class, with OrbiSim as the reference design |
| Genesis gradients unavailable for key contact modes | Medium-High | The audit makes it a finding, not a blocker; claims restricted to gradient-valid regions; torch GT as differentiable fallback |
| Differentiable deformable task in Genesis too unstable | Medium | T3 reduced to a smaller qualitative study; T1/T2 carry RQ1/RQ2 |
| H1 rejected (no clear *h\**, physics ≈ vision everywhere) | Medium | Still a publishable negative result; pivot the main contribution to B (system envelope) at G2 |
| No real data with force labels | Medium | Sim-to-sim transfer study (see P4) |
| Scope creep into direction C | Medium | C is only the interpretation of verifier traces; no free-form reasoning generation |
| Compute | Low-Medium | Confirm GPU allocation in M1; P3 results give the budget for P2 grid sizes |

**Gates:**
- **G1 (end Dec 2026):** end-to-end loop on Genesis with real or justified components. If not met by mid-Jan → run P2 on the torch GT and treat Genesis as validation only.
- **G2 (end Mar 2027):** *h\** located or H1 clearly rejected. If rejected → re-weight toward B.
- **G3 (end Jun 2027):** usable envelope measured.
- **G4 (end Jul 2027):** real or sim-to-sim validation done.

---

## 9. Immediate next steps (next 2–3 weeks)

1. **Decide the main contribution** (§1) — ideally with the supervisor.
2. Server: install `genesis-world`, run the Genesis gradient probe and the Franka demo recording; send errors back for fixing.
3. Literature check: is code public for OrbiSim (2605.16395) and CheckVLA (2607.26789)?
4. Ask about lab robot / F/T sensor access for P4; list candidate public datasets that include force labels.
5. Confirm GPU budget.
