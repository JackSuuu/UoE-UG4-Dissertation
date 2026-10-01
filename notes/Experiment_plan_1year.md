# Experiment Plan — One-Year Version (Oct 2026 – Sep 2027)

**Dissertation:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization
**Replaces:** `Experiment_plan_Phase_1.md` (3–4 week validation plan). That plan remains the blueprint for Months 1–3 infrastructure.
**Prior work used:** OrbiSim (Li et al., arXiv 2605.16395), CheckVLA (Liu et al., arXiv 2607.26789), Genesis (`genesis-world`)
**Last updated:** 2026-09-29

> **Revision 2026-09-29 (pending supervisor confirmation):**
> 1. The **headline result** is now closed-loop: *does adding the verifier to a real VLA improve safe task completion?* RQ1 explains that result and RQ3 shows it fits the compute budget.
> 2. New primary metric: **safe success** (task completed **and** no constraint violation during the episode).
> 3. A real VLA moves from a late "transfer check" to the **main RQ2 policy**. Integration and model selection happen in M2 (Nov). BC stays as the cheap policy for large sweeps.
> 4. **Task A uses a box peg** in both simulators (Genesis 1.4 differentiable mode does not detect cylinder–sphere contacts).
> 5. **Compute budget: one RTX A5000 (24 GB)** for VLA + verifier together, to mimic local/edge compute.
> 6. **Task A = box peg pushed by a two-fingertip pusher** (single-point pushing of a box is unstable; see `exp_record/week1.md`, 29 Sep).
> 7. **Public VLA benchmark added (P2b / P5):** a subset of LIBERO tasks with physical perturbations (friction/mass) and force limits, current VLAs with vs. without the verifier. This follows how CheckVLA (RoboCasa365), SAFE (LIBERO, SimplerEnv) and OrbiSim (robosuite Push etc.) are evaluated.
> 8. **Deformable scope (T3) made explicit:** a simple deformable task with a clear strain/force constraint (e.g. lift-and-fold one cloth corner without over-stretching, or press a soft object below a strain limit). Full garment folding with a VLA is out of scope: current folding VLAs rely on large real-robot datasets, and the physics predictor would need full cloth dynamics.

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

## 1a. Revised 30 Sep (start of Week 2) — what the first week of measurements changed

Four things are now settled by data rather than by argument. Three of them contradict earlier parts of this document.

**1. Genesis is demoted from GT engine to optional cross-check.** Two blocking measurements: with the same expert and initial states, SR is 1.00 in torch and 0.06 in Genesis (the peg drifts sideways and goes around the end of the wall, for both box and cylinder, cause still unknown), and it costs 34–52 ms/step against torch's 6–7. Neither is fixable inside the budget we have. **The GT remains the hand-written differentiable torch sim** (`sims/torch_push.py`: box peg, two fingertips, 10-d state, analytic box–point contact). Genesis is retained only as a *transfer* target for P2b: if the predictor transfers to it, that is evidence the learned signal is physics rather than torch-sim artefact. This is a downgrade in ambition and an upgrade in honesty — the earlier plan assumed Genesis would be the differentiable engine inside the loop, and the measurements say it cannot be trusted to be.

**2. The verifier is a distilled surrogate, not the physics engine run in the loop.** This is a real divergence from how A was originally framed ("use Genesis as an analytical world model"). In practice we *train* the risk model: `OrbiSimDynamics` is an MLP ensemble (hidden 256 × E=5, 15000 iterations) and `VisionWM` is a GRU. The engine is only the GT the surrogate is scored against. Consequences to accept rather than hide:
- the contribution becomes *"a cheap differentiable physics surrogate that can be run in the VLA loop, and a characterisation of when its signal is trustworthy"* — which is closer to OrbiSim's own distillation claim than to "run the engine inline";
- the verifier reads true simulator state (friction and mass stay hidden), so it is **privileged**. This must be stated wherever the VLA framing is used, and P2b's LIBERO transfer is what tests whether it survives without privilege;
- the surrogate is 24–30 ms/step against the vision model's 451 ms, so the cheap predictor is also the one that works — a systems result, reported as such.

**3. CheckVLA's gradient branch and hard prefix are both inert on this task, and that is now measured rather than assumed.** `suffix_repair` holds a hard prefix, and the violating contact is the *first* step of the chunk, so it cannot change the one thing that matters: GT risk 1.193 → 1.193, against 0.125 for the bisected down-scale on the same chunks. It nonetheless scores *lower* on the predictor (0.300 vs 0.256, winning on 29% of envs), so the "keep whichever the predictor rates safer" selection picked it and discarded the bisection (1.193 → 0.419). This is what produces the 49% no-op rate in the repair audit. It is **not** the mechanism behind the RQ3 `stab` table, which is a different code path (direct BPTT through the simulator, no prefix, no verifier) and fails for the opposite reason: unregularised BPTT is the *best* of the three there (CVR 0.88 vs 1.00), so over-regularisation is what hurts. Two independent mechanisms, one conclusion — gradients through this contact model are not worth optimising against. Both are now off by default (`--use_grad`, `--hard_prefix`) and kept only as ablations. Measured on friction 0.2/mass 1.5 with 128 envs: bisection 1.073 → 0.105, with the gradient branch 1.073 → 0.333.

**Consequence for the numbers already collected:** the v5 headline table was produced with the gradient branch *enabled* (it was the old unconditional default), so every RQ2/RQ2b/RQ3-sched number for the `checkvla_*` arms describes the contaminated selection. Those arms must be re-run before the table is quoted anywhere. The `none` and `gt_shadow` arms are unaffected.

**4. What we may claim about the headline, and what we may not.** The current claim is *not* "the privileged predictor repairs better" — the repair audit falsifies that: per intervention the vision model lowers the true risk of the chunk it executes by 41.4% against orbisim's 28.9%, with half the no-op rate. The claim is that the two differ in **how conservative they are** (interventions 3721 vs 1394; recall 0.934 vs 0.377; lead 8.8 vs 5.2 steps), and a CVR comparison taken at each predictor's own calibrated tau is **confounded by that**. `rq2_matched_tau.py` exists to remove the confound; until it is in, the orbisim-over-vision gap is suggestive, not established.

---

## 1b. How this differs from CheckVLA (30 Sep)

CheckVLA's code is not public, so this is a comparison of *claims and mechanisms* against the paper, and is stated as such in the write-up. It is written here because the Week-1 plan described A as "CheckVLA with a physics predictor swapped in", which understates the difference in both directions.

| | CheckVLA (2607.26789) | This thesis |
|---|---|---|
| **Verifier signal** | frozen visual world model; risk = distance between predicted and observed features | differentiable physics surrogate; risk = a **named physical quantity** (contact force / penetration) against a threshold |
| **What a violation means** | latent feature divergence — not identifiable or nameable | a specific constraint, so violations are **attributable and countable per constraint type** |
| **Required privilege** | pixels only — deployable as-is | reads true simulator state (friction/mass stay hidden) — **privileged**. A real disadvantage; the LIBERO transfer is the test that could remove it |
| **Response to a violation** | discard the chunk, re-query the policy (replanning) | **repair before execution**: bisected down-scale certified by the same signal, so the task still completes |
| **Cost** | visual WM rollout | 24–30 ms/step vs 451 ms — and the cheap one is also the accurate one (RQ1), a systems finding in its own right |
| **Question asked** | does runtime verification improve success? | **when is the verifier's signal trustworthy enough to act on?** |
| **Evidence produced** | end-to-end benchmark comparison | mechanism-level: reliability map, repair audit, rate-matched comparison, controllability, shift boundary, resource envelope |

**The one-sentence delta.** CheckVLA asks *whether* a visual world model can catch failures; we ask *when* a physics-derived risk signal is calibrated enough to be acted on, we measure that boundary explicitly, and we ship a repair operator certified by the same signal. The headline artefact is the boundary, not a win.

**Three findings that CheckVLA's design could not have produced, and which are the actual novelty:**

1. **AUROC does not predict safety.** The pixel model detects better (AUROC 0.991, timely recall 1.00) and buys −0.4% CVR; the state model detects worse (0.863, 0.56) and buys −24.6%. Ranked by detection quality the conclusion comes out backwards. Any work that reports detection metrics as evidence of safety will get this ordering wrong.
2. **The mechanism is trigger calibration, not repair quality.** Per intervention the pixel model is the *better* repairer (true risk of the executed chunk −41.4% vs −28.9%, half the no-op rate) and still loses, purely because it fires 2.7× more often. The gap is conformal-calibration granularity — a per-regime τ, not a better model. This is a finding about how to use *any* verifier, CheckVLA's included.
3. **CheckVLA's own two components are inert when the violating contact is the first action of the chunk** — the latency-aware hard prefix and the gradient repair fail for the same reason (neither can change the step that violates), and the gradient branch *over-claims* on the score while changing nothing, so a "keep whichever scores safer" selection picks it. Reproducing a published mechanism faithfully is what surfaced this; a from-scratch design would have avoided the bug instead of measuring it. The generalisable form is not "drop the hard prefix" but *a latency-aware repair is sound only if the risk is not concentrated in the committed prefix* — a property of the task, so it belongs in a check rather than in a constant.

**What we must not claim:** that the physics verifier wins *because it is physics*. On the metric that should matter most (CVR reduction at a fair operating point) the two predictors are not yet separated, and the honest statement is "we can characterise the difference, not yet exploit it". Claiming physics-beats-vision before the rate-matched comparison lands would be exactly the confound we identified.

---

## 2. Core research question

> **In long-horizon, contact-rich manipulation, how does the quality of the signal from a differentiable physics predictor — used as the core of a runtime verifier — change with task complexity, prediction horizon and constraint type? And what system-level design keeps that signal usable within acceptable resource cost?**

Three levels, each an upgrade of the Phase-1 RQ:

| RQ | Upgrade of | Question | Falsifiable hypothesis |
|---|---|---|---|
| **RQ1 — Signal quality** | Phase-1 RQ1 | Not "is the prediction accurate" but "under which conditions are the predictor's risk and gradient signals reliable" (horizon, contact regime, OOD distance, Genesis gradient-path validity) | H1: trigger AUROC and gradient agreement fall off sharply beyond a critical horizon *h\** that depends on contact regime. Beyond *h\** the physics predictor is no better than the vision WM. |
| **RQ2 — Closed-loop benefit on a real VLA, per constraint type** | Phase-1 RQ2 | **Headline:** does adding the verifier to a real VLA raise **safe success** (success with no violation) under physical OOD shift, and is the gain due to the physics predictor rather than the CheckVLA scaffolding (vs. the vision-WM arm)? Then: how force / deformation / contact-mode constraints differ in what they need from the trigger (lead time, calibration, repairability) | H2a: VLA + physics verifier > VLA alone and > VLA + vision verifier on safe success in OOD cells. H2b: the physics predictor's advantage over the vision WM is largest for constraints with hidden physical state (force, internal strain), and smallest for visually salient ones (large visible deformation). |
| **RQ3 — System feasibility** | Phase-1 RQ3 | Not "does checkpointing reduce memory" but "within which resource envelope (GPU memory, control latency) the verifier architecture remains usable, and what it degrades to outside it" | H3: with checkpointing + adaptive truncation + async scheduling, verifier quality at horizon *h\** is kept within budget (e.g. <50 ms/step, one consumer GPU), and naive BPTT cannot reach *h\**. |

**Headline result:** safe success of a real VLA **with vs. without** the verifier over the Task A OOD grid (and a demo video of the same comparison), run on one A5000.

**Supporting deliverables:** a *reliability map* (RQ1): signal quality as a function of (horizon × constraint type × contact regime × OOD shift), which explains where and why the headline gain holds; plus the system (RQ3) that makes the usable region reachable within the single-GPU budget.

A verifier can only fix failures with a *physical* cause. Before the main VLA runs, check that the base VLA's failures in the OOD cells are mainly physical (overshoot, excess force) and not perception or grasping errors.

---

## 3. Timeline overview

Core experiments and system optimisation run **in parallel, not in series**. System limits decide which experiment designs are feasible, and experiment needs decide what to optimise.

| Phase | Months | Calendar | Main task | Milestone / gate |
|---|---|---|---|---|
| **P1 Infrastructure** | 1–3 | Oct–Dec 2026 | Real components in place of stand-ins (especially a real VLA policy), gradient path audit, end-to-end predict → trigger → repair. **GT stays the hand-written differentiable torch sim; Genesis is demoted to an optional cross-check** (see §1a) | **G1 (end Dec):** end-to-end loop runs with a real VLA policy as the actor |
| **P2a Releasable physics benchmark** | 2–4 | Nov 2026 – Jan 2027 | Freeze Task A into a runnable protocol with the actor as an argument; ≥2 actors on the same verifier | **G1b (end Feb):** third party reproduces in ≤2 h on one A5000, or demoted to internal protocol |
| **P2 Core experiments** | 4–7 | Jan–Apr 2027 | Vary horizon, task complexity, constraint type; measure signal quality & trigger effectiveness | **G2 (end Mar):** first reliability map for Task A; *h\** located or H1 rejected |
| **P3 System optimisation** | 6–9 (parallel) | Mar–Jun 2027 | Checkpointing / truncation / scheduling ablations at the horizons P2 needs | **G3 (end Jun):** usable-envelope result (H3) |
| **P2b Public VLA benchmark** | 7–10 (parallel) | Apr–Jul 2027 | LIBERO subset + physical perturbations and force limits; predictor retrained on robosuite state; current VLAs with vs. without verifier | **G2b (end Jul):** with/without-verifier results on the LIBERO subset, or a documented reason it could not transfer |
| **P4 Real-data validation** | 8–10 | May–Jul 2027 | Offline trigger tests on real robot trajectories | **G4 (end Jul):** signal meaningful on real data, or a documented sim-to-real gap |
| **P5 Converge & write** | 10–12 | Jul–Sep 2027 | Complementary experiments, figures, writing | Full thesis draft end Aug; final Sep |

Buffer: about 1 month total, spread across P2/P3 overlap. If a gate fails, see §8.

---

## 4. Phase details

### P1 — Infrastructure (Months 1–3)

Starting point: the `src/` framework (plug-in interfaces, stand-ins, Genesis backend written but never run). See `src/README.md` for component status.

| Month | Work | Output |
|---|---|---|
| **M1** | ~~Run Genesis on the server~~, Franka demo recording. **Decide main contribution (§1).** Literature check: is OrbiSim / CheckVLA code public? **Task A → box peg in the torch GT; Genesis ↔ torch parity.** | Decision recorded; box Task A in the torch GT. **Parity failed (30 Sep) → Genesis demoted, see §1a.1** |
| **M2** | Replace stand-ins: real OrbiSim-Dynamics via `adapters/orbisim_official.py` (or, if unavailable, a documented re-implementation following the paper — named "OrbiSim-style", not "OrbiSim"). Real/official CheckVLA logic if public; otherwise the reference verifier with its design written up (add event-driven keyframe banks; align hard prefixing with the paper). Risk head for the predictor (route a or b in the adapter). **VLA selection and integration:** short-list 2–3 chunk-output VLAs, measure memory and ms/step on one A5000, pick one; wire Genesis camera (`render_rgb`) and the Franka action mapping; collect Genesis Task A demos and fine-tune; baseline VLA (no verifier) on the OOD grid, with its failure causes classified. | Real components behind the interfaces; chosen VLA running in the loop with baseline OOD numbers |
| **M3** | **Gradient path audit on Genesis** for Task A (rigid) and a Genesis-native deformable task (MPM/FEM, since PBD cloth is not differentiable). End-to-end loop on Genesis. Minimal checkpointing prototype started (the P3 system track starts early, as in the Phase-1 review). | Audit map (Fig B2); **G1** |

Base policy decision (revised 2026-09-29): the **real VLA is the policy for the headline RQ2 runs** (with vs. without verifier, safe success). The BC proxy (behaviour cloning: a small network trained by supervised learning to imitate the scripted expert's action chunks; a VLA is the same idea at large scale with images and language) is kept for the large sweeps (RQ1 horizon × constraint × OOD, RQ3 ablations), where running thousands of VLA episodes is too slow on one A5000. Every BC-based conclusion used to explain the headline should be spot-checked on the VLA.

### P2 — Core experiments (Months 4–7)

A factorial design over the axes the RQs name. Every cell compares **physics predictor vs. vision WM vs. hybrid** inside the same verifier.

| Axis | Levels | Notes |
|---|---|---|
| Prediction horizon *h* | 5, 10, 20, 40, 80 steps | locate *h\** (H1) |
| Constraint type | contact force · deformation/strain · contact-mode change (slip/stick, making/breaking contact) | H2 |
| Task complexity | T1 rigid push-insertion (box peg, two-fingertip pusher) → T2 multi-contact rigid (e.g. peg-in-hole with tight clearance) → T3 simple deformable task with a strain/force limit (Genesis MPM/FEM; e.g. lift-and-fold one cloth corner, or press a soft object) | increasing contact richness; T3 is deliberately not full garment folding |
| OOD shift | friction, mass, stiffness grids (from Phase 1), plus distance-to-training-range | calibration under shift |
| Predictor | physics (OrbiSim-Dynamics) · vision WM · hybrid (physics state + visual residual) | the hybrid is optional, added if the first two are close |
| Policy | VLA (headline arms: none / vision verifier / physics verifier) · BC (full sweep) | VLA on the informative cells only if one A5000 cannot cover the full grid |

Measurements for every cell:
- **Signal quality:** risk calibration (reliability curve, ECE), AUROC, gradient agreement with Genesis (gradient-valid regions only), fidelity decay vs horizon
- **Trigger effectiveness:** precision/recall, timely recall, lead time, repair success (does the repaired suffix avoid the violation in Genesis), **safe success** (primary), SR and CVR under closed loop. Reporting SR and CVR next to safe success shows whether the verifier helps or only makes the policy more cautious.
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

### P2b — Public VLA benchmark (Months 7–10, parallel with P3/P4)

**Why:** reviewers expect results on a public benchmark with existing VLAs, as in related work:

| Paper | Benchmark | Policies | Reported |
|---|---|---|---|
| CheckVLA (2607.26789) | RoboCasa365 (sim, 365 household mobile-manipulation tasks) | several VLAs under one training recipe | success rate vs. periodic replanning; timely recall at a matched 5% false-alarm rate; sim only |
| SAFE (NeurIPS 2025) | LIBERO, SimplerEnv, plus real robot | OpenVLA, π0-FAST, … | failure-detection ROC-AUC, detection time |
| OrbiSim (2605.16395) | robosuite Push, Isaac Lab Stack, AdaManip, Physion Drape | own policies | prediction fidelity, RL success |

**Problem:** public benchmarks do not test physical constraints. Most failures are grasping or placement errors, which a physics verifier cannot fix.

**Plan:** "LIBERO + physics":
1. Choose a LIBERO subset where contact matters (pushing, placing, inserting). LIBERO runs on robosuite/MuJoCo, which exposes contact forces and lets us vary friction and mass.
2. Add physical OOD perturbations (friction, mass) and per-task force limits; define safe success as in §4.
3. Retrain the physics predictor on robosuite object state (same architecture as for Genesis; the vision WM likewise), and recalibrate tau per policy.
4. Run 1–2 current VLAs that fit one A5000 (from the M2 short-list; e.g. OpenVLA-OFT or π0-FAST): none / vision verifier / physics verifier.
5. Report safe success, SR, CVR, and the detection metrics used by CheckVLA/SAFE (timely recall at matched false-alarm rate, AUROC), so the numbers can be compared.

**Why LIBERO rather than RoboCasa365:** LIBERO is the most common VLA benchmark, uses a fixed-base Franka (as our demo), and is lighter to run on one GPU. RoboCasa365 is mobile manipulation and heavier.

**Prerequisites:** the chosen VLA runs in the loop (G1) and the Task A results have located where the physics signal is reliable (G2). If robosuite transfer fails, report the Task A results as the main evidence and the benchmark attempt as a limitation.

### P2a — Releasable physics-constraint benchmark (Months 2–4, **before** P2b, not after)

Introduced 30 Sep. The problem with going straight to P2b is that it couples three unknowns — a VLA we have never run, a new simulator, and a new benchmark — so a failure in any of them costs a semester and the failure is diagnosed slowly. P2a removes the two unknowns we control.

**What it is:** the Task A setting, frozen into a benchmark protocol that others can run, with the actor treated as a variable rather than part of the task. Released as `physicsbench-push` (or similar): seeded initial states, the published friction × mass perturbation grid, fixed episode length and success/violation predicates, and an eval harness taking a policy as an argument.

**Why it is a contribution and not just our testbed:** CheckVLA evaluates on RoboCasa365, which contains **no physical constraints** — its failures are grasping and placement errors that no physics verifier can address. There is therefore no public benchmark on which a physics verifier can be shown to help. Producing one, together with the honest split of which failures a physics verifier *can* and *cannot* fix, is a standalone contribution and it is the one most likely to be cited.

**Actor ladder** (each row is an arm, same protocol, same verifier — the verifier is never retrained per actor):
1. scripted expert (sanity: verifier must not break a correct policy)
2. `bc` (current stand-in — the row every later arm is compared against)
3. small chunk-output VLA, run open-loop over `chunk_k` like every other arm
4. the VLA used in P2b, if time permits

**Numbers to produce:** safe success / SR / CVR per actor × per perturbation cell, plus the detection metrics (AUROC, timely recall at matched false-alarm rate) and the resource cost. The interesting result is the *interaction*: the same verifier helps a weak actor more than a strong one, because a strong actor's failures are less often physical. That is a claim about verifier scope, and it is only visible with ≥2 actors — which is why a single-actor testbed could never have shown it.

**Decision gate G1b (end Feb 2027):** if the harness is not runnable by others (documented, one-command, ≤2 h on one A5000), it stays an internal protocol and P2b carries the external-validity burden alone.

**Prerequisites:** none beyond what already runs. This is the cheapest credible external artifact in the plan.

**Camera status (30 Sep, done).** The blocker for actors 3–4 was that `Env.render_rgb` raised `NotImplementedError` for the torch backend, so the VLA had no input and Genesis — the *optional* cross-check — could not be made to carry it. `sims/camera.py` is now a batched analytic ray-cast renderer on the torch GT: a table plane plus five yaw-oriented boxes (peg, two fingertips, gripper body, end-stop wall), Lambert + Blinn-Phong, per-env randomised appearance. It is a **pure renderer** — reads the state tensor, returns pixels, never writes `self.state` and never calls `sim.step` — so no result already computed with `--backend torch` is affected by its presence. The wall and the printed seat are in the scene on purpose: the thesis is about the contact-force constraint at the end-stop, so a camera that hid the wall could not support the claim.

Whether the camera is *worth having* is measured, not asserted (`src/tests/test_camera_observability.py`): a ridge from pixels to each task variable, fitted on 2048 states and scored on 1024 unseen. Held-out R² **0.974** for `bx`, **0.979** for `by` and `ty`, and **0.974** for the gap `x_w − bx` — the clearance the constraint actually depends on. Yaw is 0.40, as it should be, since a square is symmetric under 90°. The same test asserts the complementary fact, which is the more useful one for the argument: **friction and mass are invisible** (R² −0.087, −0.070). They are not in the state tensor, so they cannot appear in the image, and a visual policy therefore cannot substitute for the privileged physics signal. That is a claim about the *design*, and it is now falsifiable.

Cost, 64 envs at 224×224 on one A5000: **42 ms** at `ss=1`, 146 ms at `ss=2`, so `ss` defaults to 1 — supersampling is 3.5× for smoother edges, and 146 ms is 29 ms/step even amortised over a 5-step chunk, i.e. more than half the 50 ms budget spent on antialiasing. The frame is cached per chunk (the same reason the policy is called once per chunk), which `src/tests/test_rgb_cache.py` verifies is action-identical to rendering every step, at 5.0× fewer rendered rows.

### P4 — Real-data validation (Months 8–10)

Minimal and offline — **no full VLA closed loop on hardware**:
1. Get real trajectories with physical measurements (contact force from a wrist F/T sensor; deformation from vision/markers where possible). Options, in order of preference: (a) a lab robot at Edinburgh, if access can be arranged — **ask the supervisor in M1**; (b) public manipulation datasets that include force/torque — candidates to be identified and checked in M1.
2. Run the verifiers offline on these trajectories: does the physics predictor's trigger fire before measured force/deformation limit crossings?
3. Report the same RQ1/RQ2 metrics on real data, and the sim-to-real gap relative to Genesis.

If no real data with force labels can be obtained by M6, P4 shrinks to a **sim-to-sim transfer** study (train in the torch GT, test in Genesis, or across Genesis solver settings), and this is stated as a limitation.

### P5 — Convergence & writing (Months 10–12)

- Complementary experiments requested by the gates and supervisor feedback
- Final VLA runs: headline with/without-verifier comparison on the chosen VLA over the Task A OOD grid (building on the P2 results), plus the final LIBERO + physics numbers (P2b)
- Demo: Franka arm video in Genesis (`src/demo/`), normal vs OOD vs OOD + verifier
- Writing: chapter plan in §7

---

## 5. Metrics & figures (thesis-level)

| Figure | Content | RQ |
|---|---|---|
| F1 | System diagram: policy → predictor → verifier → repair; fast/slow engines | — |
| F2 | Gradient path audit map, per task × contact regime (torch GT; Genesis columns only where the probe ran) | RQ1 |
| **F6b** | **P2a: safe success by actor × perturbation cell** — the same verifier on 2–4 actors, showing the gain depends on how physical the actor's failures are | RQ2 (P2a) |
| F3 | **Reliability map:** AUROC / calibration vs horizon × constraint type, physics vs vision | RQ1 (headline) |
| F4 | Fidelity and gradient-agreement decay vs horizon, with *h\** marked | RQ1 |
| F5 | Trigger lead time & repair success per constraint type | RQ2 |
| F6 | SR/CVR over OOD grids, 4–5 arms | RQ2 |
| **F0** | **Headline: safe success of the VLA over the OOD grid — none vs. vision verifier vs. physics verifier (with SR and CVR alongside)** | RQ2 (headline) |
| F11 | Demo video: same seed and OOD cell, VLA without vs. with verifier (force, risk score and trigger overlaid); episode chosen to be representative of F0, not the best case | RQ2 |
| F12 | VLA + verifier memory and ms/step on one A5000 vs. the 50 ms budget | RQ3 |
| F13 | Public benchmark: LIBERO + physics subset — safe success, SR, CVR for none / vision / physics verifier on 1–2 VLAs, plus timely recall at matched false-alarm rate | RQ2 (P2b) |
| F7 | Memory / latency vs horizon, with and without each optimisation | RQ3 |
| F8 | Usable envelope: verifier quality vs resource budget | RQ3 |
| F9 | Real-data offline trigger traces vs measured force | P4 |
| F10 | Verifier trace as a physical reasoning chain (observation → constraint → intervention) | C (interpretation) |

---

## 6. What the existing `src/` code covers

| Needed | Exists in `src/` | Gap |
|---|---|---|
| Plug-in interfaces for policy / predictor / verifier / GT | `interfaces.py`, `registry.py` | — |
| **GT engine** | `sims/torch_push.py` — hand-written differentiable box–point contact (the working GT); `sims/genesis_push.py` (parity fails, 30 Sep) | T2 multi-contact; Genesis demoted to optional transfer target, not a deliverable |
| Releasable benchmark (P2a) | `experiments/rq2_eval.py` arms + the OOD grid already do most of it | freeze protocol, seed the initial states, take the actor as an argument, one-command harness, docs |
| Real OrbiSim / CheckVLA | adapter templates only | M2 |
| Gradient path audit | `experiments/audit_gradients.py` (torch GT + Genesis probe) | per-regime Genesis audit |
| RQ1/RQ2 metrics | `rq1_calibration.py`, `rq2_eval.py` | horizon & constraint-type sweeps, calibration curves/ECE, repair success |
| System track | `systems/bptt.py`, `systems/scheduler.py` | Genesis-native checkpointing; latency sweeps |
| Demo | `demo/genesis_franka_demo.py` (never run) | run; add target marker, OOD comparison video |
| Real VLA policy | `adapters/openvla_policy.py` (skeleton, single-action OpenVLA) | M2: chunk-output VLA selection, fine-tuning; **camera now done** (`sims/camera.py`, gap 0.974 held out, friction/mass provably invisible) — remaining: 2-D velocity action → EE-delta mapping, demos |
| Public benchmark | — | P2b: LIBERO install, physics perturbation + force-limit wrapper, robosuite state adapter for the predictor, eval script |
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
| ~~Genesis gradients unavailable for key contact modes~~ | **Realised (30 Sep)** | Genesis parity fails (SR 0.06 vs 1.00) and it is 5× slower. Not mitigated — *removed from scope*. torch GT is the engine; Genesis is an optional transfer target for P2b |
| A releasable benchmark cannot be made reproducible by others | Medium | G1b forces a one-command ≤2 h harness on one A5000, or P2a is demoted to an internal protocol rather than a claimed artefact |
| Differentiable deformable task in Genesis too unstable | Medium | T3 reduced to a smaller qualitative study; T1/T2 carry RQ1/RQ2 |
| H1 rejected (no clear *h\**, physics ≈ vision everywhere) | Medium | Still a publishable negative result; pivot the main contribution to B (system envelope) at G2 |
| No real data with force labels | Medium | Sim-to-sim transfer study (see P4) |
| Scope creep into direction C | Medium | C is only the interpretation of verifier traces; no free-form reasoning generation |
| Compute | Low-Medium | Budget fixed to one A5000 (24 GB); P3 results give the budget for P2 grid sizes |
| VLA too large or slow for one A5000 alongside the verifier | Medium-High | Measure in M2 before choosing; prefer smaller chunk-output VLAs; run VLA arms only on informative cells; the gap to the 50 ms budget becomes an RQ3 result |
| Base VLA fails in OOD cells for non-physical reasons (perception, grasping) | Medium | Classify failure causes in M2; fine-tune on Genesis Task A demos; report which failures the verifier can and cannot address |
| Verifier lowers SR by making the policy over-cautious | Medium | Safe success as the primary metric, with SR and CVR reported next to it; tune the repair margin and conformal alpha on held-out cells |
| Public-benchmark failures are not physical, so the verifier has nothing to fix | Medium-High | Choose contact-heavy LIBERO tasks and add physical perturbations; classify failure causes; report the split |
| Predictor does not transfer to robosuite state | Medium | Same architecture retrained on robosuite; if it still fails, keep Task A as the main evidence and document the gap |

**Gates:**
- **G1 (end Dec 2026):** end-to-end loop on the torch GT with real or justified components, **and the chosen VLA running in the loop with baseline (no-verifier) OOD numbers**. The Genesis half of the original gate is removed (30 Sep, §1a.1).
- **G1b (end Feb 2027):** P2a harness runnable by a third party in ≤2 h on one A5000, else demoted to an internal protocol.
- **G2 (end Mar 2027):** *h\** located or H1 clearly rejected. If rejected → re-weight toward B.
- **G3 (end Jun 2027):** usable envelope measured.
- **G2b (end Jul 2027):** LIBERO + physics with/without-verifier results, or a documented reason the transfer failed.
- **G4 (end Jul 2027):** real or sim-to-sim validation done.

---

## 9. Immediate next steps (next 2–3 weeks)

Done in Week 1: main contribution decided (A); Genesis installed and gradient probe working; OrbiSim / CheckVLA code confirmed not public; compute fixed to one A5000.

**Done in Week 2** (see `exp_record/week2.md` for the measurements): full torch pipeline at 64 envs with the safe-success metric; orbiim / vision / vision_noact trained and conformally calibrated; RQ1 (detection + controllability), RQ2 (pooled OOD, 5 arms), RQ2b (shift ladder), RQ3 (memory + scheduling), the repair audit, and the two empty-step defects in the normaliser and the rollout risk. Genesis parity measured and failed → demoted (§1a.1). The gradient repair branch and the hard prefix both measured inert → off by default (§1a.3). Peak latency is 183 ms/step mean, 441 ms p95, so the 50 ms budget is **not** met at `chunk_k=1` but amortises to ~37 ms/step at `chunk_k=5`.

1. **Run the rate-matched τ sweep** (`experiments/rq2_matched_tau.py`). The headline orbisim-over-vision gap is confounded by the two predictors' different conservatism until this lands; it is the single highest-value thing outstanding. Smoke test in progress.
2. **Freeze the P2a protocol** (P2a, new): seed the initial states, make the actor a harness argument, write the one-command entry point, add the scripted-expert arm. This is the cheapest credible external artefact and it needs no new dependency.
3. **Start the VLA short-list properly** — this is now the only thing between us and "the verifier is on a VLA", and everything downstream (P2b, F0, F13) depends on it. Measure one candidate's memory and step latency *with the verifier resident* on the A5000 before committing, since 3.6× over budget is the current state.
4. Read out ablation (`scratch/readout_ablation.py`, written but never run at full scale): current-state vs extrapolated-state vs delta read-out. Note the trap recorded in `exp_record/week1.md` — the extrapolated read-out loses on uniformly sampled validation windows and wins under the real RQ1 protocol, so this must be judged on the RQ1 protocol only.
5. Any change to `models/nets.py` requires retraining both predictors (~300 s each) or the results silently use stale weights.
6. **Supervisor:** confirm the 30 Sep revision (Genesis out, surrogate framing, P2a ahead of P2b, contribution delta in §1b); report the GPU-0 fault (`ERR!`/`N/A`) to the admin; ask about lab robot / F/T sensor access for P4.
