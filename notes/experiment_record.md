# Experiment Record

Running log of experiment progress against `Experiment_plan_1year.md`.
Newest entry at the top.

---

## Sem 1 · Week 2 (w/c 5 Oct 2026) — Plan phase P1, Month 1

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

## Sem 1 · Week 1 (w/c 28 Sep 2026) — Plan phase P1, Month 1

### Update 29 Sep — Task A moved to a box peg with a two-fingertip pusher
**torch GT (`sims/torch_push.py`) rewritten:**
- Square peg (side 2·R_b) with yaw: state 8 → 10 dims `[…, th, w]` and obs 7 → 10 dims `[…, sin4th, cos4th, w/10]`. The new columns are appended, so existing indices are unchanged.
- **Pusher = two fingertips** 3 cm apart along y, like a closed Franka gripper. Each has a box–point contact with Coulomb friction μ_p = 0.5. Wall contact is per corner; floor friction couples translation and rotation (ellipsoidal limit surface).
- Seat clearance `gap` 2 mm → 6 mm (a yawed box is wider in x).
- Expert: "align, then insert straight". Heading = +x plus a lateral correction atan(15·e_y); the pusher stays centred on the box's back face; it slows down on the remaining insertion depth.

How we got there (expert on the torch GT, nominal cell):

| Version | SR | Violation rate | Problem |
|---|---|---|---|
| Single frictionless point | 0.11 | 0.89 | Force only along the face normal; any angled push spins the box |
| + fingertip friction | 0.09 | 0.91 | Near the seat, the expert steered sideways, cut through the box and spun it |
| + hold near seat, 6 mm gap, straight final approach | 0.84 | 0.00 | Floor friction treated translation and rotation independently (over-damped rotation, did not match Genesis) |
| Coupled floor friction | 0.12 | 0.00 | Single-point pushing of a box is unstable |
| **Two fingertips, centred on the back face, μ_p = 0.5** | **1.00** | **0.00** | — |

Final expert over the OOD grid (64 envs/cell): nominal and friction ≥ 0.6× → SR 0.83–1.00, violations ≈ 0; **low friction (0.2×) → violations 0.84–1.00**, the intended OOD failure. Gradients are finite (‖dJ/dA‖ ≈ 0.15 for both hard and smooth). μ_p = 1.0 was rejected: SR on light pegs dropped to 0.62–0.75 and the smooth gradients blew up to ~1e4.

**Genesis (`sims/genesis_push.py`):** box peg, two-fingertip pusher as one MJCF body (density must be set in the MJCF; the material rho is ignored for MJCF bodies, which gave a 0.008 kg pusher), yaw and yaw rate in the state.

**Parity (same 16 initial states, expert):**

| Check | Result |
|---|---|
| Aligned box, straight push | identical (SR 1.00 both, y error 0) |
| Floor friction only (slide / spin, no pusher) | matches |
| Nominal cell, full task | torch SR 1.00 vs Genesis 0.50; mean position gap 3 cm (was 19 cm with the single-point pusher) |
| Friction 1.8×, mass 2× | torch 0 violations vs Genesis 0.94 |

Remaining parity gaps:
1. **The force signal is defined differently.** torch uses max(fingertip normal force sum, wall force). Genesis uses the norm of the *net* contact force on the peg (including floor friction), which is also impulsive per step. Genesis needs per-contact forces (pusher–peg and wall–peg separately) before violation rates can be compared.
2. Genesis corrects lateral error less well (nominal |e_y| 3.4 cm vs 0.4 cm).

Earlier fix of the parity script: torch and Genesis had been reset with different random generators (CPU vs CUDA), so the Week 1 parity numbers compared different initial states.

**Quick pipeline on the box Task A** (`run_all.sh push torch --quick`, stand-ins, not reportable): runs end to end (EXIT 0).
- Expert violation rate at nominal 0.000; predictor training data has episode-violation rate 0.56.
- Calibration: safe-chunk fraction dropped to 0.47 (disk: 0.91); stand-in OrbiSim tau ≈ 4e-11 (still degenerate).
- RQ1 pooled AUROC: stand-in OrbiSim **0.31** (worse than chance), vision 0.58, vision_noact 0.52.
- RQ2 pooled OOD CVR: none 0.41, gt_shadow 0.25 (−39%), checkvla_vision 0.45, checkvla_orbisim 0.44.
- **Interventions create violations at nominal:** in the nominal cell `none` has CVR 0.00, but gt_shadow 0.38 and the CheckVLA arms 0.12–0.25. Repaired chunks seem to make the pusher strike the box. To investigate (repair size vs. loss of fingertip alignment); this is directly the "over-cautious / harmful verifier" risk.
- RQ3/stab: log grad-norm variance 47 with no stabiliser (disk: 0.15); clip 18.5, relax 21.3. Box contacts are far more chaotic, so the stabiliser matters more here.
- RQ3/sched: fast-only 18.8 ms, async 40.3 ms, sync (GT in the loop) 581 ms mean latency.
- RQ3/mem: naive 17.7 MB vs checkpointing 3.6 MB at T=50.

### Update 29 Sep (later) — execution protocol fixed to CheckVLA-style open-loop chunks
**Problem found:** the baseline `none` was replanning every step (closed-loop), while CheckVLA studies VLA policies executing action chunks open-loop. On the box task this matters a lot (64 envs/cell):

| Execution | Nominal SR / CVR | Low friction + heavy peg SR / CVR |
|---|---|---|
| `none`, replan every step | 0.84 / 0.03 | 0.67 / 0.81 |
| `none`, open-loop chunk k=5 | 0.27 / 0.73 | 0.00 / 1.00 |
| `gt_shadow` (GT check + repair) | 0.83 / 0.08 | 0.53 / **0.38** |

The earlier "interventions create violations at nominal" was mostly small-sample noise (quick: 8 envs/cell). The real issue was the protocol mismatch. The GT verifier upper bound is strong: CVR 0.81 → 0.38 in the hardest cell.

**Changes:**
- `Controller` and `verifier_rollout` now execute chunks open-loop for `--chunk_k` steps (default 5) in **all** arms, including the baseline. `chunk_k=1` gives closed-loop replanning.
- `safe_success` (success with no violation) added to `summarize` as the primary metric.
- `run_all.sh` split into `build` (collect/train/calibrate) and `eval` (RQ1–3, figures); `--chunk_k` is separated and passed only to the eval scripts.
- Full-size run launched in the background (`setsid nohup`, log `~/scratch/run_all_full.log`).

### Update 29 Sep (night) — full pipeline ran; three defects found and fixed
The first full-size run completed end to end (EXIT 0) but RQ1 reported **orbisim AUROC 0.372**, i.e. the privileged-state predictor was worse than chance while the pixel predictor scored 0.987. Chasing that down turned up three separate defects, each confirmed by an ablation before the fix.

**(1) `Normalizer` std floor poisoned the predictor** (`models/nets.py`).
`fit()` clamped std to `min=1e-4`. Obs dim 6 (`ty`) has an almost constant delta, so its std landed on 1e-4, and `_one` divides `(o - o_prev)` by that std — turning float noise into a ~1e4 spike. Every model consuming the normalised delta was poisoned; `VisionWM` never consumes it, which is exactly why it looked fine.
Ablation on the val split, max-over-chunk risk AUROC vs the GT hot label:

| std floor | read off rollout | read off true state |
|---|---|---|
| 1e-4 | 0.495 | 0.495 |
| 1e-2 | 0.971 | 0.996 |

After the fix, full 8k-iter training: val obs NMSE 0.55 (was 1.30, *worse* than predicting the mean), risk MAE 0.033 (was 0.10), AUROC 0.992 (was 0.372). A `w_obs` sweep (0.05/0.5/1.0) gave AUROC 0.99 throughout, so the pre-existing obs-loss weight was kept.

**(2) The risk head was reading a state it had drifted away from.**
`OrbiSimDynamics.rollout` extrapolated the state autoregressively and read risk off the extrapolated state. That cannot work here: contact dynamics are not predictable from the state alone — friction and mass are hidden generative parameters and contact makes the delta discontinuous. Measured 1-step delta NMSE: 0.77 with obs only, 0.48 *even with the true friction and mass*. So a drifted state is garbage input:

| | before | after |
|---|---|---|
| AUROC | 0.495 | 0.976 |
| abs d(risk)/d(action) per step | 1.65, 0.020, 0.010, 0.0049, 0.00027 | 0.80, 0.82, 0.79, 0.76, 0.84 |
| pred risk at chunk x1.0 / x0.5 / x0.0 | 1.97 / 4.05 / 5.97 | 0.33 / 0.30 / 0.28 |

The state rollout is kept — the gradient probe and the differentiable-repair path consume it — but risk is now read off the true current state plus each step's action, which is also the physically correct model: risk is a function of the current contact state and the commanded action.

**(3) Repair optimised a surrogate whose ranking inverts off-distribution.**
With AUROC 0.992 and a trigger rate near 1.0, RQ2 CVR still barely moved (0.272 → 0.266). On friction=0.2 / mass=1.5:

| chunk scale | 1.0 | 0.9 | 0.75 | 0.5 | 0.3 | 0.15 | 0.0 |
|---|---|---|---|---|---|---|---|
| GT risk | 0.337 | 0.309 | 0.267 | 0.197 | 0.145 | 0.094 | 0.005 |
| predicted risk | 0.412 | 0.413 | 0.415 | **0.421** | 0.369 | 0.249 | 0.122 |

GT falls monotonically but the prediction *rises* over 1.0 → 0.5, so gradient descent walked toward larger actions: the chunk moved 164% and the true violation rate went 0.00 → 0.27. Fixes, each measured:
- scale the whole chunk, not just the suffix (suffix-only scaling leaves the committed prefix's contact force; the full-chunk ladder is what moves it);
- accept on a **relative** test `score <= min(margin, 0.75 * score(chunk))` — an absolute margin is useless because the predictor carries a near-constant offset, so "first scale below 0.8" stops at 0.75 and leaves GT risk at 0.27;
- **bisect** the factor instead of walking a fixed ladder — with a ladder the repair was discontinuous in tau (at friction=1.0, tau 0.60/0.45/0.35/0.25 gave CVR 0.00/0.26/0.00/0.00, so lowering the threshold sometimes *raised* violations);
- keep the gradient result but score it against the down-scale and take the safer of the two, so repair degrades to "be more conservative" rather than optimising a surrogate that stopped tracking reality.

GT risk after repair, 256 envs per cell: friction 1.0/mass 1.0 0.413→0.154, friction 0.6/mass 1.0 0.338→0.115, friction 1.8/mass 0.5 0.441→0.138, friction 0.2/mass 1.5 0.337→0.092.

**(4) Metric bug.** `summarize`'s `intervention_rate` was "did this episode ever trigger", which with a ~0.15 per-step rate over 80 steps is ~1.00 for every arm — it hid the real trigger frequency entirely. Now reported per control step, with `episode_trigger_rate` kept alongside.

### Where this leaves the headline result — honest reading
`gt_shadow` (repair by GT-guided down-scaling) reaches CVR 0.015 pooled over OOD cells against a 0.272 baseline, so **the safety mechanism works when the risk model is right**. The learned predictors improve on the baseline but do not approach that bound: on friction=0.2 / mass=1.5 the tau sweep gives CVR 0.74/0.43/0.71/0.78/0.55 at tau 0.60/0.45/0.35/0.25/0.15, against none 1.00 and gt_shadow 0.02.

The reason is a property of the distilled predictor, not of the repair search: its training range is friction 0.5–1.5, and at friction 0.2 it cannot reliably rank down-scale factors. So the current stand-in supports the *mechanism* (detect → repair → safer) but not yet the *claim* that a distilled physics verifier closes most of the gap to a GT verifier. Two things would change that, in order of expected value:
1. widen the predictor's training range to cover the deployment range (friction 0.3–1.5) and report the residual gap on a *held-out* perturbation axis — this is the standard fix and costs one data collection;
2. report trigger calibration *per regime* rather than one global tau, so the conformal guarantee is stated on the distribution it was calibrated on instead of being extrapolated to friction 0.2.

Both are honest experiments; neither is a bug fix, and both belong in the next run rather than being folded in silently here.

### Update 29 Sep (later) — run v5: the fixed predictor, and why detection ≠ repair

Full pipeline re-run end-to-end with the corrected model so that tau is recalibrated against it (`run_all_v5.log`). tau: orbisim 0.597, vision 0.457, vision_noact 0.950.

RQ1 (pooled over the OOD grid, chunk-level labels):

| predictor | AUROC | precision | recall | timely |
|---|---|---|---|---|
| orbisim (privileged state) | 0.863 | 0.579 | 0.377 | 0.563 |
| vision (pixels) | **0.991** | 0.795 | 0.934 | **1.00** |
| vision_noact (action-blind ablation) | 0.750 | 0.257 | 0.233 | 0.850 |

RQ2 (pooled OOD cells, `chunk_k=5`, 64 envs/cell):

| arm | SR | CVR | CVR reduction | **safe success** |
|---|---|---|---|---|
| none (VLA alone) | 0.823 | 0.272 | – | 0.685 |
| gt_shadow (GT risk upper bound) | 0.805 | 0.015 | −94.7% | 0.801 |
| checkvla_vision | 0.843 | 0.271 | −0.4% | 0.695 |
| checkvla_orbisim | 0.804 | 0.205 | −24.6% | **0.704** |
| checkvla_vision_noact | 0.818 | 0.285 | +4.5% | 0.689 |

**The main finding, and it is not the one I was looking for.** The pixel world model detects violations *better* than the privileged-state one (0.991 vs 0.863, timely recall 1.00 vs 0.56) yet its repair does nothing at all: −0.4% CVR. The state model detects worse but repairs 25% of the violations away. So detection quality and repair quality are **not the same axis**, and the RQ1 metric set measures only the first.

⚠️ **Superseded by the repair audit below.** My first reading of this was "detection and repair are two independent capabilities, and the privileged-state advantage shows up in the second". The audit falsifies that: per intervention the *pixel* model is the better repairer (41.4% vs 28.9% GT-risk drop, half the no-op rate). What actually separates them is how often they fire. The correct statement is that the two differ in conservatism, and comparing CVR at each predictor's own tau is confounded by it. See "repair audit" below.

Averaging per-step latency, `checkvla_orbisim` is 24–30 ms against `checkvla_vision` 451 ms, i.e. the cheap predictor is also the one that works.

**Correction to the entry above.** I had compared the two risk read-outs (risk off the extrapolated state vs. off the current state) on a proxy — AUROC over uniformly sampled validation windows from the training episodes — and concluded from it that the current-state read-out was the better one (0.976 vs 0.495). The pipeline disagreed: under the RQ1 protocol the extrapolated-state read-out reaches AUROC 0.992 / risk-MAE 0.035, the current-state one 0.863 / 0.087. The proxy was the wrong distribution — RQ1 evaluates OOD cells with execution noise and a whole-chunk GT label, not in-distribution single windows. The conclusion "current-state read-out is better" was an artefact of measuring on the training distribution. The current-state read-out is nevertheless the one that has to stay, because the extrapolated state's gradient was noise (per-step |d(risk)/d(action)| 1.65 vs 0.00027) and repair with it was actively harmful. The open question is whether a read-out that anchors on the true state but *also* consumes the model's own predicted one-step delta can recover the ranking without giving up the usable gradient; that is now the top follow-up, measured with the RQ1 protocol rather than a proxy.

RQ3 unchanged in structure, worse in one place: gradient clipping cuts log-grad-norm variance 38.1 → 21.2 but CVR goes **0.88 → 1.00** (and `relax` 17.4 with 39 spikes, also 1.00), i.e. every form of gradient-based repair tested is worse than not repairing. Consistent with the read-out finding above, and it means the gradient path is currently the weakest part of the method, not a supporting detail. Memory at T=400: naive 1742 MB vs checkpointed 47 MB (cos 1.000); truncating the checkpoint to save more costs accuracy (cos 0.968). Scheduling: fast_only 183 ms mean / 441 ms p95, async 145 ms / 447 ms at staleness 1.57, sync 466 ms / 1037 ms. At `chunk_k=5` the 183 ms amortises to 37 ms/step, inside the 50 ms budget; the sync verifier does not fit.

### Update 29 Sep — repair audit: the gap is *when to intervene*, not *how well*

Added an audit that, on every intervention, shadow-rolls the chunk that was **executed** alongside the chunk the policy **proposed**, and reports the drop in GT risk between them. Trigger rate, AUROC, timely recall and controllability all describe the verifier; none of them say whether the repair changed the real dynamics. Kept out of `Controller.act` so it never contaminates the reported latency (a shadow rollout costs ~5 real steps, which is why `gt_shadow`'s 320 ms is not a deployable number — it is an upper bound only).

| arm | GT risk proposed → executed | drop | cleared | **no-op** | magnitude | n |
|---|---|---|---|---|---|---|
| gt_shadow | 2.289 → 0.741 | +63.2% | 0.94 | 0.01 | ×0.67 | 665 |
| checkvla_vision | 0.691 → 0.408 | **+41.4%** | 0.90 | 0.24 | ×0.51 | 3721 |
| checkvla_orbisim | 0.650 → 0.392 | +28.9% | 0.91 | **0.49** | ×0.78 | 1394 |
| checkvla_vision_noact | 0.765 → 0.765 | +0.0% | 0.82 | 1.00 | ×1.00 | 2606 |

**This overturns the reading I gave an hour earlier.** I had guessed orbisim won by intervening *less* — a calibration story. It is not that. Per intervention, vision is the **better repairer**: it lowers the true risk of the chunk it executes by 41.4% against orbisim's 28.9%, and half as many of its interventions are no-ops. It still loses, by 24.6% to 0.4% of CVR, because it fires 3721 times to orbisim's 1394 at recall 0.934 with 8.8 steps of lead — i.e. on chunks that were never going to violate. Its SR is the highest of any arm (0.843, above the 0.823 baseline) and its CVR is unchanged: the extra interventions push the box differently without removing a violation.

So the correct claim is **not** "the privileged-state model repairs better" (per-intervention it does not). It is that the two predictors differ in *how conservative they are*, and the CVR comparison at each predictor's own calibrated tau is confounded by that. The follow-up is a rate-matched sweep (`experiments/rq2_matched_tau.py`): sweep tau for each predictor and compare at a common intervention rate, so the comparison cannot be won by a looser threshold.

Sanity check on the whole audit chain: `vision_noact` returns `mag ×1.00`, `no-op 1.00`, drop +0.0% — the action-blind model never moves the chunk, because its score is insensitive to the action and the bisection has no signal to follow, so it returns the input. Any audit bug that silently broke the chunk would not produce that.

**Controllability result, which also refuted the hypothesis that motivated it.** Descending fraction of the scale ladder (1.0/0.75/0.5/0.25/0.0) on chunks the trigger acts on: orbisim 0.93, **vision 0.87**, vision_noact 0.00. Vision is nearly as controllable as orbisim, so "its signal is flat" is not the explanation either. The action-blind ablation at 0.00 is the metric behaving correctly. First version reported a *relative* drop, which is meaningless here: the predicted score carries a near-constant offset and can be near zero on individual chunks, so the denominator is unbounded — it reported means of −2127 and −593. Replaced with a Spearman correlation of score against scale.

### Update 29 Sep — RQ2b: the shift ladder (an honest boundary result)

The OOD grid asks *whether* the verifier helps at a fixed set of perturbed cells, which conflates "how far is this cell from the training range" with "is the verifier good". RQ2b makes shift magnitude the independent variable: one OOD axis moves at a time, the other is held nominal, the ladder spans the training range (friction 0.5–1.5) and continues past it, and `shift` is the signed log-distance from that range so 0 is exactly in-distribution.

| distance from training range | n cells | none | gt_shadow | checkvla_orbisim | checkvla_vision |
|---|---|---|---|---|---|
| in-dist | 11 | 0.99 | 1.00 | 0.99 | 1.00 |
| near | 2 | 0.91 | 0.95 | 0.94 | 0.95 |
| far | 7 | 0.67 | **0.82** | 0.70 | **0.65** |

This is a negative result and it is the honest boundary of the method. Inside the range the verifier is unnecessary (everything is 1.00 — nothing violates). Outside it the GT verifier's advantage opens up (+15 points) while the distilled ones do not help at all: orbisim +3, and **vision is 2 points *worse* than no verifier at all**. Friction 0.2 is the clearest case — none 0.03 safe / CVR 0.95, gt_shadow 0.31 / 0.05, orbisim 0.11 / 0.58, vision 0.06 / 0.94. The learned predictor cannot rank down-scale factors at friction 0.2, which is outside the range it was trained on; the *mechanism* still works, the distilled *risk model* does not.

**Mass is not a breaking axis for this task.** Over the whole ladder 0.4–2.5, safe success stays 0.95–1.00 and CVR at 0.00–0.03, in-distribution and out. Only friction breaks the box task — a slippery box slides into the wall; a heavy one still tracks the pusher. So the mass half of the RQ2 OOD grid is padding, and the shift curve carries information on one axis only. Worth stating in the write-up, because it also means "distribution shift" is not a scalar here and a single shift number would be misleading.

Caught by a `--quick` smoke test before it reached a full run: holding the other axis at nominal was first hard-coded to `"mass"`, which for the mass ladder produces `{"mass": v, "mass": 1.0}` → `{"mass": 1.0}`, so the entire second ladder silently re-evaluated the nominal cell (every rung reporting shift 0.00 and safe 1.00). The other axis is now derived from the ladder keys, with an assert on the ladder's arity.

### Update 29 Sep — new measurement: repair controllability

Added `controllability()` to `_verif.py` and wired it into RQ1, because the v5 result above cannot be explained without it. It re-scores every proposed chunk at a ladder of scales (1.0, 0.75, 0.5, 0.25, 0.0) and reports (a) the fraction of monotone descents in the *predicted* score and (b) the mean relative drop, each next to the same quantity measured on the GT shadow rollout. Chunks are restricted to those the trigger actually acts on, since the controllability of a chunk that is never repaired is irrelevant. This is the direct test of whether AUROC is a sufficient proxy for safe success, and it is what the thesis should report alongside it.

### Update 29 Sep — new experiment RQ2b: shift as the independent variable

`experiments/rq2_shift.py`. The OOD grid asks *whether* the verifier helps at a fixed set of perturbed cells, which conflates "is the cell far from the predictor's training range" with "does the verifier help". RQ2b makes the shift magnitude the independent variable — the question a practitioner actually has. One OOD axis moves at a time, the other is held nominal, and the ladder spans the training range (friction 0.5–1.5) and continues past it. `shift` is the signed log-distance from the training range, so 0 is exactly in-distribution.

A null result worth keeping: **the mass axis is not a breaking axis for this task at any value in 0.4–2.5** (safe 1.00, CVR 0.00 throughout, in-distribution and out). Only friction breaks the box task — a slippery box slides into the wall; a heavy one still tracks the pusher. The shift curve therefore carries information on one axis only, and the mass half of the OOD grid in RQ2 is padding.

Caught by a `--quick` smoke test before it reached a full run: holding the other axis at nominal was first hard-coded to `"mass"`, which for the mass ladder produces `{"mass": v, "mass": 1.0}` → `{"mass": 1.0}`, so the entire second ladder silently re-evaluated the nominal cell (every rung reporting shift 0.00 and safe 1.00). The other axis is now derived from the ladder keys, with an assert on the ladder's arity.

### Update 29 Sep — plan revision
`Experiment_plan_1year.md` revised (pending supervisor confirmation):
- **Headline result = closed loop on a real VLA:** safe success with vs. without the verifier over the Task A OOD grid, plus the matching demo video. RQ1 explains the result and RQ3 shows it fits one A5000.
- **New primary metric: safe success** (task completed **and** no violation). Reason: in the quick smoke run, `checkvla_orbisim` lowered CVR (0.46 → 0.31) but also SR (0.13 → 0.05). A verifier can look good on CVR alone just by being over-cautious.
- **VLA moves earlier:** selection and integration in M2 (Nov), baseline VLA OOD numbers are part of G1. BC stays for the large RQ1/RQ3 sweeps.
- RQ2 split into H2a (VLA + physics verifier beats VLA alone and VLA + vision verifier on safe success) and H2b (per-constraint-type advantage, as before).
- New figures F0 (headline), F11 (demo video), F12 (VLA + verifier cost on one A5000). New risks: VLA too slow for one A5000; non-physical VLA failures; over-cautious verifier.

Addendum to the Week 1 smoke-test numbers: the RQ1 gradient agreement between the stand-in OrbiSim and the torch GT was **negative** (cosine mean −0.29, median −0.45, n = 403). Stand-in and quick size, so not reportable, but it is another sign that the stand-in predictor is not usable as is.

### Update — Genesis bring-up (same week)
Constraint adopted: **one A5000 only** (`run_all.sh` now defaults to `CUDA_VISIBLE_DEVICES=0`) to mimic local/edge compute.
OrbiSim and CheckVLA code are **not public**, so both will be re-implemented from the papers ("OrbiSim-style", "CheckVLA-style").
CheckVLA gap vs the paper: conformal calibration ✅, action-conditioned WM ✅, latency-aware hard prefixing ⚠️ (ours repairs the suffix by gradient descent), **event-driven keyframe banks ❌ missing**.

**Genesis gradient audit** (`audit_gradients.py --backend genesis --genesis_probe --quick`), gradient of the task loss w.r.t. the pusher action chunk:

| Peg shape | Status (all cells, H = 10/50/100 sub-steps) | \|dJ/dA\| | Peg displacement |
|---|---|---|---|
| box (side 2·R_b) | valid | 3e-2 – 1.3e-1 | 2.6 – 18 cm |
| cylinder | zero | 0 | 0 (pusher passes through) |

Findings:
1. Genesis 1.4 rigid differentiable mode works: read state with `entity.get_state()` and differentiate with `scene.backward(loss)`. It requires the `approximate_implicitfast` integrator and no hibernation.
2. **In differentiable mode, cylinder–sphere contacts are not detected.** Box–sphere, box–box and sphere–sphere contacts are detected and give non-zero action gradients. This is a gradient-path limitation for RQ1 (plan §3.6).
3. The pusher was too light (≈0.02 kg vs a 0.5 kg peg) to push; it is now heavier. Genesis combines friction as `max(μa, μb)`, so only the floor carries μ.
4. **Parity with the torch GT fails.** Same expert and initial states, nominal cell: SR torch 1.00 vs Genesis 0.06. The Genesis peg drifts sideways and goes around the end of the wall, for both box and cylinder, even after the friction fix. Cause not found yet. **Genesis cannot be used as GT for RQ1/RQ2 until this is fixed.**
5. Cost at 16 envs: Genesis 34–52 ms/step vs torch GT 6–7 ms/step.

Decision (28 Sep): **Task A moves to a box peg in both simulators.** The torch GT needs a box–point contact model with rotation (yaw, yaw rate), so the state, obs, expert and predictors all change and must be retrained.

**VLA timing.** No real VLA has been used yet; every "policy" so far is the BC stand-in. Plan:

| Phase | When | What |
|---|---|---|
| Not yet | Sep–Oct | Prerequisites first: box Task A, Genesis parity, verifier working on BC |
| Early integration | P1 · M2–M3 (Nov–Dec) | Engineering only: load candidate VLAs on one A5000 and measure memory and per-step latency (feeds the RQ3 resource envelope); wire `render_rgb()` and the Franka action mapping; collect Task A demos in Genesis for fine-tuning |
| Transfer check | P2, after G2 (Mar–Apr 2027) | Verifier on top of the VLA on a subset of Task A cells: do the RQ1/RQ2 conclusions hold? |
| Final | P5 (Jul–Aug 2027) | Thesis VLA results and demo video (VLA vs VLA + verifier) |

Model choice (M2): CheckVLA verifies open-loop **action chunks**, so chunk-output VLAs are preferred over single-action OpenVLA-7B. Short-list 2–3 candidates and measure memory and latency on the A5000 before choosing.

### Decisions
- **Main contribution: Direction A (differentiable verifier)**, with B (systems) as support and C as interpretation only, as recommended in plan §1.
- RQ3 (systems) will draw on KTransformers (CPU/GPU hybrid inference, expert deferral, CUDA Graph) as the systems reference for later optimisation.

### Environment (server)
| Item | Value |
|---|---|
| GPUs | 4× NVIDIA RTX A5000 (24 GB each); experiments use a single GPU to mimic an edge device |
| OS | Ubuntu 22.04, no sudo |
| Python / torch | 3.13 (miniconda) / torch 2.14 (CUDA) |
| Genesis | `genesis-world` 1.4.2 |

### What was run
All runs are **`--quick` smoke tests with stand-in components**. They only show the pipeline runs end to end. **None of these numbers are reportable results.**

| Step | Command | Status | Notes |
|---|---|---|---|
| Full pipeline, torch GT | `bash run_all.sh push torch --quick` | ✅ runs end to end | audit → rq3/mem → collect → train → calibrate → rq1 → rq2 → rq3/stab → rq3/sched |
| Gradient audit + Genesis probe | `audit_gradients.py --task push --backend genesis --genesis_probe` | ⚠️ runs, result not trusted | see findings |
| Franka demo in browser | `demo/genesis_web_streamer.py --port 8080` | ✅ | Genesis physics + render, MJPEG stream over SSH tunnel |
| New RQ3 parts | `rq3_systems.py --part hybrid` / `--part deferral` | ⚠️ scaffolding only | see findings |

Smoke-test observations (quick sizes, stand-ins — indicative only):
- rq3/mem: checkpointing peak memory 0.8 MB vs naive 4.0 MB at T=50 (B=8); grad cosine vs naive ≈ 1.0 (0.997 with truncation).
- rq1 pooled AUROC: orbisim 0.53, vision 0.68, vision_noact 0.61. Calibrated orbisim tau ≈ 5.5e-15, so the stand-in trigger is essentially degenerate.
- rq2 pooled OOD CVR: none 0.46, gt_shadow 0.04, checkvla_orbisim 0.31, checkvla_vision 0.51.
- rq3/sched: fast_only mean latency 127 ms, async 175 ms, sync 92 ms. **All above the 50 ms/step budget**, and async is slower than fast_only.

### Findings / issues
1. ~~**Genesis gradient probe is inconclusive.**~~ **Resolved (see update below):** the probe was using the wrong API (`get_pos()` + `loss.backward()`). Genesis 1.4 rigid mode *is* differentiable.
2. The main table in `audit_gradients.py --backend genesis` is still computed on the **torch GT**. Only the probe rows touch Genesis.
3. The stand-in OrbiSim problem (known issue: under-predicts rare risk spikes) is still visible: AUROC is close to chance and tau is close to 0.
4. The RQ3 `hybrid` / `deferral` parts are **not results yet**:
   - arithmetic intensity comes from an analytical estimate, not from measurement;
   - the "fused" operator runs the same computation as the unfused one (1.00x);
   - CUDA Graph capture fails because there is no grad path in the capture;
   - deferral: 0 deferred gradients were consumed, and the 1.02x is a formula on submit time, not measured overlap.
5. `results/push_torch/` was overwritten by the quick run (the directory is gitignored).

### Code changes this week
- `sims/torch_push.py`, `sims/torch_cloth.py`: `init_state` samples on the generator's device. Fixes the CPU/CUDA generator errors, because `gs.init()` makes CUDA the default torch device.
- `sims/genesis_push.py`: CUDA generator in `reset`; the probe no longer calls `scene.reset()` and now reports a missing `grad_fn` explicitly instead of crashing.
- `systems/hybrid_engine.py` (new): arithmetic-intensity scheduler, fused step operator, CUDA Graph wrapper (prototype).
- `systems/expert_deferral.py` (new): immediate/deferred segment scheduler (prototype).
- `experiments/rq3_systems.py`: new `--part hybrid` and `--part deferral`.
- `demo/genesis_web_streamer.py` (new): headless Genesis Franka demo streamed to a browser (no VNC or sudo needed). Currently drives the arm with the **scripted expert only**: no BC policy, OrbiSim or CheckVLA yet.

### Not done yet (from plan §9 / M1)
- [ ] Full-size `run_all.sh push torch` (no `--quick`)
- [ ] `run_all.sh push genesis`
- [x] Fix the Genesis gradient probe against the Genesis 1.4 docs
- [ ] Genesis ↔ torch GT parity (peg lateral drift)
- [x] Literature check: OrbiSim / CheckVLA code not public → re-implement from papers
- [ ] Ask supervisor about lab robot / F/T sensor access (P4)
- [ ] Confirm GPU budget
- [ ] Wire BC + OrbiSim + CheckVLA and OOD physics into the web demo
- [ ] Turn the RQ3 hybrid/deferral prototypes into measured experiments

### Next week (Week 2) — proposed
1. ~~Task A → box peg~~ ✅ (29 Sep). Finish Genesis ↔ torch parity: measure pusher–peg and wall–peg contact forces separately in Genesis; look into weaker lateral correction.
2. Investigate why interventions (gt_shadow, CheckVLA repair) create violations at nominal on the box task.
3. Genesis gradient audit per contact regime (free / pusher_contact / wall_contact / stuck) → Fig B2.
4. Full-size torch pipeline (no `--quick`) → first real stand-in numbers; add **safe success** to `rq2_eval.py`; split `run_all.sh` into "build verifier" (collect/train/calibrate) and "evaluate" (RQ1–3).
5. Read the OrbiSim / CheckVLA papers and design the re-implementations (prep for M2).
6. Start the VLA short-list (chunk-output, fits one A5000 with the verifier; prefer models with LIBERO checkpoints for P2b).
7. Supervisor: confirm the 29 Sep plan revision (incl. box Task A and the LIBERO benchmark); report the GPU 0 fault to the admin; ask about lab robot / F/T sensor access (P4).
