# Week 2 (from 30 Sep 2026) — Plan phase P1, Month 1

Newest entry at the top. Index and conventions: [README.md](README.md).

---

### Update 1 Oct (evening) — first closed-loop OpenVLA actor: the vision readout works, the LLM readout fails in distribution

Chunk heads trained on the cached OpenVLA features (54k frames, 3000 episodes; validation = 300 held-out *episodes*).

**Open loop, held-out R² of the next 5 actions** (mean predictor ≈ 0.00 in every case, so the split is not leaking):

| readout | ridge | MLP head |
|---|---|---|
| `llm` — LLM last-layer hidden state, last prompt token | 0.852 | 0.947 |
| `vis` — mean projected image patch, before the LLM | 0.946 | 0.979 |

The pre-LLM visual feature beats the post-LLM one. For this control task, 32 LLM layers and a last-token readout lose spatial information rather than add it.

**A mistake of mine, measured and reverted.** I had subsampled the "already seated" frames (all-zero target, 38% of frames) to 10%, expecting them to bias the head towards slow pushing. Error by episode phase on held-out episodes showed the opposite. Push-phase error was unchanged (0.0096 vs 0.0106 m/s), but at the seat the head crept forward on **36%** of frames (error 0.018 m/s) against **2%** (0.0035) with all frames kept. Creeping into a seated block builds wall force. `--done_keep` now defaults to 1.0. A global R² hid this; the per-phase table found it.

**Closed loop, actor only (arm `none`), `--quick` (8 envs × 20 cells), pooled OOD:**

| actor | SR | CVR | safe success |
|---|---|---|---|
| `bc` (reference) | 0.839 | 0.286 | 0.670 |
| OpenVLA `vis`, 10% seated frames | 0.723 | 0.402 | 0.554 |
| **OpenVLA `vis`, all frames** | **0.812** | **0.357** | **0.589** |
| OpenVLA `llm`, 10% seated frames | 0.393 | 0.714 | 0.214 |
| OpenVLA `llm`, all frames | 0.554 | 0.643 | 0.339 |

**Per cell (the part that matters):** `vis` matches `bc` in distribution (SR 1.00 / CVR 0.00 on almost every cell with friction ≥ 0.6×) and fails mainly at friction 0.2×, the designed physical OOD failure the verifier exists to catch. `llm` fails **at nominal physics** too (friction 1.0×, mass 1.0×, the demo physics: SR 0.50, CVR 0.50). Its failures are not physical. They are closed-loop covariate shift (open-loop R² 0.947 does not survive compounding), which a physics verifier cannot fix.

This is the plan's §2 precondition ("check that the base VLA's failures in the OOD cells are mainly physical") applied for the first time: `vis` passes and `llm` does not. It is also the P2a actor × verifier interaction, now available from data rather than argument.

Caveats: 8 envs per cell, so per-cell SR moves in steps of 0.125 and is indicative only. The pooled numbers average over OOD cells.

### Note 1 Oct — related verifiers: SEAL (Wu et al., ICRA 2026) and IVE (Lee et al., CoRL 2025)

Literature positioning, read from the method sections, not the abstracts.

**SEAL** ("Do What You Say", arXiv 2510.16281). A reasoning VLA (π0 + text plans) writes a plan, then acts; the actions often fail the plan, worst under OOD. Runtime steering is **Hypothesize → Predict → Verify**: sample K=10 action sequences from the same VLA, predict each outcome, score it 0/1 with GPT-4o against the text plan, execute the first that passes (early exit). 94–97% ID, up to +15% on compositional tasks, 347 ms per decision; gains grow with K, with diminishing returns. **In simulation the Predict step uses parallel instances of the true simulator**, i.e. an oracle equivalent to our `gt_shadow` arm. A learned world model is left as future work. Self-reported limits: bounded by the base policy's proposals; the VLM misjudges fine-grained gripper–object contact.

**IVE** ("Imagine, Verify, Execute", arXiv 2505.07815). Not a VLA safety method; autonomous *exploration* for data collection. A VLM imagines scene-graph transitions; a VLM verifier judges feasibility using recent interaction history and returns yes/no + reason + suggested correction. Their ablation: removing the verifier degrades exploration only slightly; memory and explorer matter more.

**Positioning.** Three verifier types: semantic outcome-vs-plan (SEAL), feasibility (IVE), quantitative physical constraint (ours). Only ours checks a quantity that is invisible in pixels. We have evidence for that: friction/mass are not decodable from the camera (held-out R² −0.087 / −0.070), and SEAL itself reports VLM failures on fine contact. Our contribution is the step SEAL defers: replace the oracle simulator in Predict with a 30 ms learned surrogate and measure what is lost (`gt_shadow` 0.801 vs `checkvla_orbisim` 0.705 safe success).

**To adopt, by value:**
1. **Best-of-N candidates** scored by the physics verifier. Fixes the `scale_repair` limitation (it can only shrink, so it fails when risk is not monotone in magnitude). Needs candidate diversity: the chunk head is deterministic, so train an ensemble of heads on the cached features (minutes). Gives a safe-success-vs-K curve.
2. **Select on risk *and* progress**, not risk alone. Safety-only selection rewards not moving (measured: vision drives CVR to 0 at SR 0.05). The predictor already rolls the state forward, so predicted advance toward the seat is available.
3. **History-conditioned verification** (IVE's memory, turned into calibration). Compare predicted vs observed outcomes over recent steps to adapt τ online or infer friction. Targets the weakest result: far-OOD friction 0.2, learned 0.08 vs oracle 0.41 safe success.
4. **Visual-OOD stress test** (SEAL's taxonomy; they find viewpoint/background shifts hurt VLAs most). With our renderer: shift viewpoint/appearance. Expect the VLA and the vision verifier to degrade and the state-based physics verifier not to.
5. **Structured feedback** on each intervention: constraint, step, predicted force, repair taken. This is F10, the interpretation layer.

**Not adopted:** a VLM as verifier. Over the 50 ms budget by >7×, and blind to contact force.

### Update 1 Oct (later) — VLA actor pipeline: demos collected, features extracting

The OpenVLA-7B actor needs a head that emits an H×2 world-frame velocity chunk in one forward pass (`planar_head`). The pipeline is now demos → frozen-backbone feature cache → small MLP head → the same `Controller` and verifier as every other arm.

**Demos (`src/vla/collect_demos.py`).** 3000 scripted-expert episodes, 120,000 frames of 224×224 RGB plus the per-step action, in **4 minutes** (50 parallel envs, batched camera). What the data actually is:

| | value |
|---|---|
| physics | nominal only (friction 0.25, mass 0.5) |
| expert seated within the 40 recorded steps | 100% (median error 2.8 mm) |
| expert violations | 0% |
| eval episode length (`sim.T`) | 80 steps; demos stop at 40 because the expert has finished by then |
| initial state | peg x ±2 cm, y ±4 cm, seat y ±8 cm, yaw ±0.15 rad |
| appearance | randomised per env (lighting), as the appearance-invariance result requires |
| frames with zero action (episode already finished) | **45%** |

Nominal-only demos are deliberate. The VLA learns the *skill*; it never sees low friction or a wall strike. Under OOD friction its open-loop chunks overshoot into the wall, which is exactly the failure the verifier is meant to catch. The verifier is not trained on these demos, so the two arms are cleanly separated. This mirrors the `bc` setup, so the two actors are directly comparable.

The 45% zero-action frames would pull the head towards small velocities (a slow policy that never arrives). `train_chunk_head.py --done_keep 0.1` keeps 10% of them, so the head still learns to stop.

**The "silent death at ~110 episodes" was the launcher, not the code.** Every earlier collection run died around episode 100–110 with no traceback and no OOM. Cause: jobs were started with a plain `&` inside the tool shell, which is killed when the tool call hits its ~120 s timeout. 100 episodes took ~2 min at the old serial speed, which matches exactly. Three workarounds (chunking, `gc.collect`, CUDA resets) were built against the wrong diagnosis and have been removed. The standing rule applies: long jobs run under `setsid nohup ... < /dev/null`.

**Feature cache (`src/vla/extract_features.py`, running, ~100 min, GPU 2).** 54,000 frames (every 2nd step, only steps with a full 5-step future). The backbone is frozen, so features are the same every epoch: extracting once turns a >10 h training job into ~100 min plus minutes per head. Two readouts are cached so the choice is measured rather than assumed: `llm` (last-layer hidden state at the last prompt token, image and instruction fused) and `vis` (mean projected patch, vision only). Smoke-checked: features finite and frame-dependent; all 28 identical-feature pairs are identical images (finished episodes); chunk targets match the stored actions.

**Head training (`src/vla/train_chunk_head.py`)** reports held-out R² per chunk step, split by *episode* (a frame split would leak), against a mean predictor and a ridge readout. If the MLP cannot beat ridge, use ridge.

**Three silent bugs fixed before they produced numbers:**
1. The old trainer fed raw pixels to the model, skipping OpenVLA's processor (normalisation and the 6-channel DINOv2/SigLIP stack).
2. Train/test feature mismatch: the adapter used a forward hook, no 29871 terminator, no feature normalisation and its own copy of `ChunkHead`. A head trained on one feature and run on another does not raise; it just acts wrongly. The adapter now imports the trainer's head class and feature definition, and the normalisation is stored in the checkpoint.
3. `rq2_eval.py` always wrote `rq2.json`, so an OpenVLA run would have overwritten the v7 table. It now takes `--tag`, which defaults to the policy name for any non-`bc` policy (→ `rq2_openvla.json`). `run_p2a.sh` passed flags that do not exist (`--out`, bare `--camera`); fixed. The scripted-expert arm is documented as not yet implemented.

**Scope notes recorded for the write-up** (from discussion, not experiments):
- The verifier learns state+action → risk of a *named* constraint. Friction and mass are hidden from it, not learned. The runtime mechanism (calibration, trigger, repair, rate-matched evaluation) is task-agnostic. The GT simulator, state representation and predictor are per task.
- **Limitation:** `scale_repair` means "be more conservative" and only works when risk is monotone in action magnitude. True for wall strikes, false for e.g. grasp slip, where safety needs *more* grip force. A grasping task needs a different repair operator.
- **The "differentiable" part contributes less than expected:** gradient validity is 0–0.19 in `wall_contact`, the violating regime. The forward risk signal does the work, not the gradient. This is a finding, not a gap to hide.

**Next:** train heads on `llm` and `vis`, compare held-out R² against ridge; closed-loop smoke test on two cells; then `run_p2a.sh openvla` detached.

### Update 1 Oct (v7 complete) — rate-matched τ sweep lands, physics-signal claim survives and widens

Full re-evaluation with the corrected predictor, conformal calibration, and the rate-matched threshold sweep. The v6 headline table (which had the gradient branch on) is superseded; `none` and `gt_shadow` are unchanged because they never call `repair`.

**Headline, pooled OOD (20 cells, chunk_k=5):**

| arm | SR | CVR | CVR red. | safe success | interventions |
|---|---|---|---|---|---|
| `none` | 0.823 | 0.272 | — | 0.685 | — |
| `gt_shadow` (oracle) | 0.805 | 0.015 | −94.7% | **0.801** | 665 |
| `checkvla_orbisim` | 0.807 | 0.210 | −22.8% | **0.705** | 1572 |
| `checkvla_vision` | 0.826 | 0.272 | 0.0% | 0.695 | 4641 |
| `checkvla_vision_noact` | 0.819 | 0.285 | +4.8% | 0.689 | 4288 |

**The rate-matched sweep removes the confound.** Each predictor is split-conformally calibrated on its own score distribution, and the two distributions are not comparable — at their own calibrated τ, vision intervenes on 5.4% of steps against orbisim's 2.2%. The headline table therefore compared them at *different cost*, so "physics beats vision" was confounded with "vision fires 2.5× as often". `rq2_matched_tau.py` sweeps τ per predictor and records the **measured** intervention rate.

| measured rate | orbisim CVR / safe | vision CVR / safe |
|---|---|---|
| 0.0057 | 0.261 / 0.714 | 0.259 / 0.715 |
| 0.0084 | 0.259 / 0.711 | 0.261 / 0.713 |
| 0.0179 | **0.218 / 0.718** | 0.262 / 0.714 |
| 0.0293 | **0.183 / 0.722** | 0.268 / 0.714 |
| 0.0893 | 0.276 / 0.642 | 0.270 / 0.669 |
| 0.1192 | 0.268 / 0.690 | 0.270 / 0.654 |

The gap **survives matching and widens with rate**. At its own calibrated τ (τ 0.597 → rate 0.0218) orbisim gets CVR 0.210 / safe 0.715; vision at its own (τ 0.457 → rate 0.0540) gets CVR 0.272 / safe 0.714 — no better than doing nothing, while spending 2.5× the interventions.

Two further readings, both more interesting than the ordering itself:

- **vision has no useful operating point.** Its measured points never beat the 0.262 baseline at any rate that leaves the task intact; its next point up is rate 0.112, where SR has collapsed to 0.65. Suppression beating the task is a real failure mode of a verifier, and it is invisible to CVR alone.
- **vision can drive CVR to 0.00 — by never acting** (rate 0.000, SR 0.836). CVR is gameable by doing nothing, which is why safe success is the headline metric.

Fig H (`figH_rate_matched`) plots the **measured** sweep points, not the `at_rate()` interpolations. vision's fitted rate–τ exponent is −2.55 against orbisim's −2.16 and it jumps 0.021 → 0.117 between two of its own measured points; drawing a curve through that gap would assert resolution the data does not have.

**The systems blocker is gone.** `rq3_sched` fast_only: **30.2 ms mean / 45.9 ms p95** on the stress cell (friction 0.2, mass 2.0), against 183/441 in v6. The gradient branch was doing 25 BPTT iterations per intervention and was the entire reason the budget was missed. The oracle `sync` mode remains 494 ms / 1103 p95 — still not deployable, as always.

**The gradient audit says the GT gradient is useless exactly where it matters.** Overall valid fraction 0.306, but by regime: `free` 0.70–1.00, `pusher_contact` 0.29–0.85, **`wall_contact` 0.00–0.19**, `stuck` 0.00–0.02. The violating regime has the *worst* gradient validity. This is the mechanistic reason the gradient branch could never work here.

**Shift ladder (near/far OOD bands, friction/mass):** in-dist all 1.00; near (friction 0.4–0.6) all arms at CVR 0.00; far (friction 0.2, mass 0.5–2.0): `none` safe 0.04, `gt_shadow` 0.41, `orbisim` 0.08, `vision` 0.04. The learned gain lives in the friction 0.4–0.6 band, and vision is *worse than no verifier* in the far band.

**Unresolved defect:** `vision_noact`'s magnitude ratio hits the 0.05 floor — for an action-insensitive predictor the score is constant in the scale, so the bisection's accept test `score ≤ min(margin, 0.75·score₀)` can never be satisfied and the search falls through to the floor. It damps the task to a near stop without preventing anything (CVR 0.285 vs baseline 0.272, SR −0.03). It is the ablation's own control and it fails as expected, but it is a real bug in `scale_repair` and should be guarded before submission.

**VLA evidence scripts committed** (`src/vla/`): `vla_proof_openvla.py` (OpenVLA-7B loads, 14.06 GiB peak, greedy_decode is the ground truth; `predict_action` is broken on transformers 5.17), `vla_vision_conditioning.py` (head is alive, reads the scene: 10/10 distinct sequences on real frames), `vla_appearance_invariance.py` (appearance randomisation moves the head 0.99x as much as a scene change — demos must span appearances). `tests/test_decode_agreement.py` pins the generate() bug: it returns constant token 31872 for every input.

**Demo collection running** (3000 episodes, background). Next: train the chunk head (H×2 continuous head replacing the 7-token discretised head) on those demos, then wire the VLA into `Controller.act`.

---

### Update 1 Oct (overnight, run v6) — the confound is removed, and the physics-signal claim survives it

Full re-evaluation after the repair-path fix, plus the new rate-matched threshold experiment. Every `checkvla_*` number in the v5 table was produced with the inert gradient branch enabled and is now superseded; `none` and `gt_shadow` are unchanged (0.685 / 0.801 safe) because they never call `repair`.

**Headline, pooled OOD (14 cells, chunk_k=5):**

| arm | SR | CVR | CVR red. | safe success | interventions |
|---|---|---|---|---|---|
| `none` | 0.823 | 0.272 | — | 0.685 | — |
| `gt_shadow` (oracle) | 0.805 | 0.015 | −94.7% | **0.801** | 665 |
| `checkvla_orbisim` | 0.805 | 0.204 | −25.0% | **0.705** | 1572 |
| `checkvla_vision` | 0.754 | 0.268 | −1.6% | 0.623 | 4641 |
| `checkvla_vision_noact` | 0.766 | 0.278 | +2.0% | 0.571 | 4288 |

The fix moved the two arms in *opposite* directions, which is itself the result. `orbisim` was barely affected (safe 0.704 → 0.705) because bisection was usually winning the selection anyway. `vision` got substantially worse (0.695 → 0.623, SR 0.843 → 0.754): with the inert branch gone it now depends on bisection alone, which over-damps — mean magnitude ratio ×0.28 against orbisim's ×0.48. `vision_noact` got worse still (0.689 → 0.571). So the v5 near-tie between the two predictors was an artefact of the gradient branch flattering the worse one.

**The confound, and the experiment that removes it.** Each predictor is split-conformally calibrated on its own score distribution, and the two distributions are not comparable — at their own calibrated τ, vision intervenes on 6.5% of steps against orbisim's 2.2%. The headline table therefore compares them at *different cost*, so "physics beats vision" was confounded with "vision fires three times as often". `rq2_matched_tau.py` sweeps τ per predictor and records the **measured** intervention rate, inverting a power law fitted to round 1 to land on the requested rates (a single quantile of the uncorrected pool undershoots by 4–20× because intervening damps the actions and lowers every later score).

| measured rate | orbisim CVR / safe | vision CVR / safe |
|---|---|---|
| 0.010 | 0.254 / 0.712 | 0.261 / 0.713 |
| 0.020 | **0.209** / 0.717 | 0.265 / 0.711 |
| 0.040 | **0.177** / 0.709 | 0.265 / 0.681 |
| 0.080 | 0.252 / 0.642 | 0.255 / 0.573 |
| 0.160 | — | 0.129 / 0.193 |

The gap **survives matching and widens with rate**. At its own calibrated τ (τ 0.597 → rate 0.0217) orbisim gets CVR 0.204 / safe 0.715; vision at its own (τ 0.457 → rate 0.0651) gets CVR 0.265 / safe 0.644 — worse than doing nothing, while spending 3× the interventions.

Two further readings from the same curve, both more interesting than the ordering itself:

- **vision has no useful operating point.** Its measured points never beat the 0.262 baseline at any rate that leaves the task intact; its next point up is rate 0.134, where SR has collapsed to 0.42. Suppression beating the task is a real failure mode of a verifier, and it is invisible to CVR alone.
- **vision can drive CVR to 0.012 — by never acting** (rate 0.193, SR 0.046). CVR is gameable by doing nothing, which is why safe success is the headline metric. This is the cleanest possible motivation for the metric choice.

Fig H (`figH_rate_matched`) plots the **measured** sweep points, not the `at_rate()` interpolations. vision's fitted rate–τ exponent is −2.61 against orbisim's −2.16 and it jumps 0.019 → 0.134 between two of its own measured points; drawing a curve through that gap would assert resolution the data does not have. The r=0.005 and r=0.160 orbisim rows are absent for the same reason (no measured point in range) and are left blank.

**Two results that changed the plan:**

1. **The 50 ms budget is met.** `rq3_sched` fast_only: **30.2 ms mean / 45.9 ms p95** on the stress cell (friction 0.2, mass 2.0), against 183/441 in v5. The gradient branch was doing 25 BPTT iterations per intervention and was the entire reason the budget was missed. The oracle `sync` mode remains 494 ms / 1103 p95 — still not deployable, as always. **The systems blocker is gone**, which removes one of the three arguments against going to a VLA.
2. **The gradient audit says the GT gradient is useless exactly where it matters.** Overall valid fraction 0.306, but by regime: `free` 0.70–1.00, `pusher_contact` 0.29–0.85, **`wall_contact` 0.00–0.19**, `stuck` 0.00–0.02. The violating regime has the *worst* gradient validity. This is the mechanistic reason the gradient branch could never work here, and it is independent of the hard-prefix argument.

**Unchanged and still true:** RQ1 still ranks detection the other way (vision AUROC 0.991 / timely recall 1.00 vs orbisim 0.863 / 0.563). Detection quality does not predict the safety outcome, now demonstrated under rate matching rather than by confound. Shift ladder: in-dist 0.993 / 0.991 / 0.899, near 0.906 / 0.938 / 0.812, far 0.665 / 0.699 / 0.578 for none / orbisim / vision — the learned gain lives in the friction 0.2–0.6 band, and vision is now clearly *worse than no verifier* in every band.

**A defect found, not fixed:** `vision_noact`'s magnitude ratio collapses to the 0.05 floor. For an action-insensitive predictor the score is constant in the scale, so the bisection's accept test `score ≤ min(margin, 0.75·score₀)` can never be satisfied and the search falls through to the floor. It damps the task to a near stop without preventing anything (CVR 0.278 vs baseline 0.272, SR −0.057). Harmless to the conclusions — it is the ablation's own control and it fails as expected — but it is a real bug in `scale_repair` and should be guarded before submission.

**Decision taken from these results: go to a VLA, not more mechanism tuning.** Reasoning in `Experiment_plan_1year.md` §9. Short version: the claim we wanted is now defensible and further cells or seeds will not change the story; the remaining risk to the thesis is that nothing in it is a VLA. Two cheap defect fixes go in first.

Two figure bugs fixed in the same pass: `fig_g` had been silently producing nothing since it was written (rq1 keys predictors by bare role `orbisim`, rq2 by arm name `checkvla_orbisim`, so the join was empty and it returned before plotting), and `make_figures.py` did not accept `--chunk_k`.

### Update 30 Sep — CheckVLA's gradient branch and hard prefix are both *inert*, and that explains a week of contradictory results

I had been treating the RQ3 gradient table (every stabiliser worse than no repair: CVR 0.88 none / 1.00 clip / 1.00 relax) and the repair audit (orbisim: 49% of interventions are no-ops) as two separate oddities. They are the same bug, and finding it came from implementing CheckVLA's latency-aware hard prefix rather than from any experiment.

`suffix_repair` holds a hard prefix by construction: `prefix = chunk[:, :latency].detach()`. Our risk label is the max over the chunk, and the violating contact is the **first** step of the chunk — so the gradient branch is not optimising badly, it is optimising the only thing it is allowed to move while the one thing that matters stays fixed. Measured, 128 envs, predictor `orbisim`, friction 0.2 / mass 1.5, τ forced to 0 so the same proposed chunks are compared:

| repair candidate | GT risk of the executed chunk |
|---|---|
| as proposed | 1.193 |
| `scale_repair` (bisected down-scale) | **0.125** |
| `suffix_repair` (gradient, hard prefix) | **1.193 — unchanged** |
| what `repair()` actually returned | 0.419 |

The gradient branch does nothing *and still wins its own selection step*: it scores 0.300 against the bisection's 0.256 on the predictor, and takes that win on 29% of envs. So "keep whichever the predictor rates safer" was systematically discarding the sound repair in favour of the inert one. Nothing in the selection could have detected this, because the inert branch's score is a better *lie* than the sound branch's score is a truth. This accounts for the 49% no-op rate in the repair audit.

**It does not account for the RQ3 `stab` table, and I initially wrote that it did.** `rq3_systems.py:105` optimises the *whole* action sequence by direct BPTT through the simulator — no prefix held, no verifier in the loop — and reports CVR on the GT afterwards. It finds the opposite: *unregularised* BPTT is the best of the three (final CVR 0.88, against 1.00 for both clipping and relaxation), i.e. over-regularising the gradient makes the optimiser take worse steps. Two different mechanisms:

| | RQ3 `stab` | `suffix_repair` |
|---|---|---|
| what is optimised | full action sequence, direct BPTT through the sim | chunk suffix, through the predictor |
| why it fails | the gradient signal itself is poor; regularising it degrades the step | structurally cannot change the violating step (hard prefix) |
| CVR | 0.88 unregularised vs 1.00 stabilised | GT risk 1.193 → 1.193 (inert) |

One shared conclusion — gradients through this contact model are not worth optimising against — but the mechanisms are independent, and the repair path's failure is the more specific one. Keeping them apart matters for the write-up: fixing the gradient path would not have fixed the stab table, and vice versa.

**Two code changes follow, both measured:**

- `scale_repair(..., hard_prefix=)` — implemented CheckVLA's constraint on the path actually used, to test whether the same pathology applies there. It does, and worse: GT 1.004 → 0.110 (−89%, cleared 0.99) without the prefix against 1.004 → 0.708 (−29.5%, cleared 0.68) with it. The bisection drives the suffix to ×0.10, nearly a dead stop, and still cannot clear the limit, because it is forbidden from touching the violating step. Same at friction 1.0/mass 1.0: 0.136 vs 0.540. **Default off, kept as an ablation** (`--hard_prefix`).
- The gradient branch is now opt-in (`--use_grad`, default off). Default path: GT 1.073 → 0.105; with the gradient branch re-enabled: 1.073 → 0.333.

I had recommended this change on 30 Sep morning on the reasoning that "already-dispatched actions cannot be changed, and that constraint holds for a down-scale too". **That recommendation was wrong and the measurement is why.** CheckVLA's constraint encodes the assumption that the dispatched prefix is not the problem. On this task it is exactly the problem. The generalisable form of the finding is not "drop the hard prefix" but: *a latency-aware repair is only sound if the risk is not concentrated in the committed prefix*, which is a property of the task and belongs in a check, not in a constant.

Also corrected in the same pass: the RQ1 reading in the Week-1 entry below ("detection and repair are two independent capabilities, and the privileged-state advantage shows up in the second") — the repair audit falsifies it, see the audit entry. And Fig G's title and axes were rewritten around intervention count rather than a detection-vs-repair framing that no longer holds.

### Update 30 Sep — Week-2 orientation

Where things stand after the full-size run. Recorded here because three plan-level decisions this week came from measurement rather than argument, and the reasoning needs to survive:

1. **Genesis is out.** SR 0.06 against torch's 1.00 on the same expert and initial states, for both box and cylinder, and 34–52 ms/step against torch's 6–7. Not a bug to fix inside the budget — the engine cannot be trusted to be the GT, and it cannot fit the loop. `Experiment_plan_1year.md` §1a.1 records the demotion; the torch GT carries every result. Genesis survives only as a *transfer* target (does the predictor hold up in a different engine?), which is a smaller and honest question.
2. **The verifier is a distilled surrogate, not the engine in the loop.** This diverges from how A was first written ("use Genesis as an analytical world model"). `OrbiSimDynamics` is an MLP ensemble trained for 15000 iterations; `VisionWM` is a GRU. The engine is only the GT the surrogate is scored against. §1a.2 states the consequences: the contribution is closer to OrbiSim's own distillation claim than to running an engine inline, the verifier is **privileged** (true state, friction/mass hidden) and this must be stated wherever the VLA framing is used, and the surrogate's 24–30 ms/step against the vision model's 451 ms makes the cheap predictor also the accurate one.
3. **The headline claim has to be narrowed.** The repair audit says the pixel model is the better repairer per intervention and the state model still wins on CVR, purely by firing 2.7× less often. So the claim is not "physics repairs better" — it is that the two differ in conservatism, and a CVR comparison at each predictor's own τ is confounded. `experiments/rq2_matched_tau.py` removes the confound and is the highest-value thing outstanding.

New in the plan this week: **P2a**, a releasable physics-constraint benchmark, placed *before* P2b. The reason is that P2b couples three unknowns (a VLA never run, a new simulator, a new benchmark) so a failure in any of them costs a semester and is diagnosed slowly. P2a freezes what already runs into a protocol where the *actor* is the variable, needs no new dependency, and has a claimable artefact: CheckVLA evaluates on RoboCasa365, which has no physical constraints, so there is currently no public benchmark on which a physics verifier can be shown to help. G1b (end Feb) forces a one-command ≤2 h reproduction or demotes it to an internal protocol.

**Still the empty step:** every experiment runs `policy=bc`. `--policy openvela` exists in `registry.py` and has never been run. P2a's actor ladder (scripted expert → `bc` → small chunk-output VLA → P2b's VLA) is the cheapest way to make that step real, and it is now step 2 of the immediate next steps rather than a Month-2 item.

---
