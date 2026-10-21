# Week 3 (from 5 Oct 2026) — Plan phase P1, Month 1

Newest entry at the top. Index and conventions: [weekly index](agent_notes/experiment_record_index.md).

---

### Update 9 Oct — Expanded confirmation and candidate-repair validation launched

**Previous matched-protocol validation completed:** all three runs have 20/20 cells, n=8/cell, H=10, execution horizon=5, 16 policy calls/episode batch. Pooled OOD contains 14 cells (112 episodes/arm). Original predictor + old taus: physics SR/CVR/safe success = 85.71%/25.89%/64.29%; original + decision taus = 83.93%/24.11%/63.39%; mass2 + decision taus = 83.93%/24.11%/63.39%. The common baseline is 86.61%/32.14%/59.82%; oracle is 80.36%/6.25%/77.68%. This is a small positive signal, not established efficacy; recalibration and mass coverage have not demonstrated additional overall benefit. The predictor weights were not retrained in that run.

**Fixed-method confirmation:** `scripts/expand_verifier_confirmation.sh`, run `verifier_confirmation_20261009_a`, n=64/cell over all 20 cells (1,280 episodes/arm; 896 pooled OOD). Seeds 16000–16019 are disjoint from previous validation/calibration. Arms: identical full OpenVLA H10 DAgger×2 policy alone; finite-horizon GT shadow reference; original learned physics + frozen decision-only taus; original learned physics + frozen legacy taus. Original predictor only: adding another mass2 run is deferred because its small-sample outcomes did not improve the decision-tau result. This confirmation set must not be used to tune algorithm parameters; it is not a new-task generalization test.

**Algorithm validation:** `scripts/validate_candidate_repair.sh`, run `candidate_validation_20261009_a`, n=32/cell over all 20 cells (640 episodes/arm; 448 pooled OOD), seeds 26000–26019. Fixed original predictor and frozen decision-only calibration. Arms: policy alone, current bisection, discrete-scale minimal-modification selection, and risk-filtered learned-progress selection. All arms use H=10 / chunk_k=5 / full OpenVLA, with forward microbatch capped at 8 images for activation memory. No VLA or predictor weight updates.

**Candidate algorithm specification (frozen for this run):** scales `[1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.15, 0.05]`; evaluate each candidate explicitly, using the existing risk score and target `min(0.8, 0.75 * original_score)`. Require an actually accepted candidate plus the existing action-responsiveness guard; otherwise return the original proposal and count abstention. Discrete mode chooses the least-modified accepted scale. Progress mode chooses the accepted candidate with smallest ensemble-mean predicted goal distance at the executed prefix endpoint (step 5), breaking exact ties toward less modification. It never reads hidden friction/mass or GT future states. Progress mode performs eight additional learned state rollouts; both candidate modes evaluate eight candidate risks. Bisection also evaluates eight risks inside repair (original, six search steps, floor), excluding the common trigger check.

**Attribution and limitations:** discrete vs progress shares the same candidates/acceptance logic and isolates the progress objective. Bisection vs discrete changes both search and no-accepted-candidate handling; do not attribute that entire difference to search alone. Progress may fail due to state-rollout drift; this experiment tests its usefulness, not an assumed improvement. Accepted surrogate scores are not certified physical safety. Returning the original proposal on abstention is not a safe fallback. This is SEAL-inspired constrained candidate selection, not SEAL semantic verification; these arms are also not official CheckVLA reproductions. Gradients, hard-prefix constraints and stall recovery remain off. Stall is diagnostic only: three consecutive decisions with <1 mm goal-distance reduction outside the success region.

**Outputs/provenance:** per-episode final success, any violation, first violation/trigger/success, intervention counts, final task cost and stall counts are saved. Every completed cell is atomically checkpointed to `rq2_<run>.partial.json`; final outputs have `complete=true`. Metadata records per-arm tau/mode, actual checkpoint/calibration/source SHA-256, grid indices and split. Paired rescue/harm and 95% fixed-grid, within-cell episode-pair bootstrap intervals (4,000 draws) accompany SR/CVR/safe success. These intervals do not quantify uncertainty across new tasks or unseen parameter grids.

**Tests:** five existing cadence/calibration tests and ten new CPU candidate/pairing/microbatch tests passed. A six-arm CPU BC implementation smoke test on grid indices 0 and 4 completed with final JSON, correct 16 calls and 64 policy rows for n=4, correct per-arm metadata and paired outcome lengths. It is not an OpenVLA efficacy experiment. An earlier full-grid CPU smoke attempt hit the shell's 120-second timeout after 12 completed cells; its partial JSON remains labelled `implementation_smoke_20261009` and is not a failed CUDA training job.

**Detached launches:** both jobs use `setsid nohup ... < /dev/null`, offline model loading, and separate healthy GPU UUIDs. Confirmation launcher PID 6613 / Python 6617, GPU `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`, log `/home/ami/scratch/verifier_confirmation_20261009_a.log`. Algorithm launcher PID 6618 / Python 6621, GPU `GPU-99b39a1b-5fdd-1acb-398a-426672ebc1a8`, log `/home/ami/scratch/candidate_validation_20261009_a.log`. At the launch check both Python processes were alive and startup logs showed model initialization; no completed OpenVLA cells were yet observed. Do not assume completion from this entry.

**Next gate:** read completed paired results and band-specific harm/stall before choosing any new mechanism. Freeze choices on validation, then evaluate with a separately reserved final-test seed base (46000, not used here). If progress ranking fails, diagnose candidate risk/progress ranking against simulator labels before history-conditioned predictor retraining. Verifier transfer to LIBERO follows a stable net-gain gate; no physical robot work is started.

---

### Update 8 Oct — Protocol audit; verifier improvement before simulated-arm transfer

**Scope:** physical hardware is deferred. The active sequence is learned-verifier improvement, then a functioning LIBERO or RoboCasa365 simulated-arm pipeline. Box pushing remains a diagnostic testbed, not the final demonstration.

**Critical audit finding:** the legacy Controller cached non-triggering proposals only in the `none` arm. Learned-verifier and oracle arms replanned every step while not triggering; repaired chunks used `commit`. Thus the legacy headline is not a matched `chunk_k=5` comparison. Its numbers are historical observations, not evidence of verifier benefit/harm under the intended protocol. They must be repeated before causal conclusions or paper claims are made.

**Implemented:** `matched_open_loop_v2` caches every proposal for `chunk_k` steps and overwrites that plan when repaired, also for mixed triggering/non-triggering batches. Calibration now filters to actual policy decision times; the old implementation pooled stale-proposal scores from intervening steps. Added isolated calibration paths, deployment-matched noise option, separate rollout seed bases, checkpoint/calibration SHA-256 metadata, and policy-call cadence assertions. No predictor weights were changed.

**Tests:** five CPU regression tests pass (non-triggering learned/oracle parity with `none`, mixed batches, oracle repair cadence, calibration decision mask). RGB cache regression also passes on a healthy GPU: 6 policy/render calls over 30 steps, bit-identical cached/uncached actions.

**Validation entry point:** `scripts/validate_verifier_protocol.sh` uses the full OpenVLA `llm` H10 DAgger×2 head, original thresholds first, then separately calibrated original/mass2 predictors. Calibration uses episode seeds 42000–42005 and zero added execution noise; validation uses 6000–6019. Calibration files and result tags are new; legacy artefacts are preserved. Initial n=8/cell is diagnostic only, not a headline claim. Threshold search on test cells is not used.

**Launch:** run ID `protocol_v2_20261008_a`, detached with `setsid nohup`, launcher PID 4095551, log `~/scratch/protocol_v2_20261008_a.log`. GPU selected by UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b` (physical index 3, confirmed available at launch). Outcomes are pending.

**Provenance correction:** `push_torch/rq2_openvla_llm_h10_mass2.json` completed 20 cells but loaded the original predictor directory, despite its tag. Legacy pooled SR/CVR/safe are: none 0.840/0.326/0.614; oracle 0.706/0.045/0.688; physics 0.684/0.324/0.485; vision 0.761/0.372/0.506. This is not a full-LLM mass2 evaluation.

**Next gate:** establish a corrected-protocol baseline, isolate calibration/repair/stall failures, then run validation-only candidate/progress/recovery ablations. Transfer requires a new simulator action/state adapter, new predictor data/training and calibration. No LIBERO/RoboCasa rollout or learned-verifier improvement is claimed here.

---

### Update 7 Oct — Predictor coverage fix complete; full VLA headline with two predictor versions

**Corrected 8 Oct from the JSON files. These are legacy, cadence-confounded runs; see the protocol audit above.** Coverage collection/training completed, but effectiveness was not established.

**Mass-2 predictor (A) headline** (`rq2_openvla_vis_h10.json` in `push_torch_mass2`, mass 0.5–2.0, τ on OpenVLA):

| arm | SR | CVR | safe | Δ safe vs none |
|---|---|---|---|---|
| `none` | 0.852 | 0.359 | 0.596 | — |
| `gt_shadow` (oracle) | 0.746 | 0.073 | **0.727** | **+0.131** |
| `checkvla_orbisim` | 0.742 | 0.339 | 0.519 | −0.077 |
| `checkvla_vision` | 0.747 | 0.328 | 0.535 | −0.061 |

**Split by band:**

| band | `none` | oracle | `checkvla_orbisim` |
|---|---|---|---|
| friction 0.2× (target band) | 0.020 | 0.500 | 0.043 |
| friction ≥ 0.6× | 0.870 | 0.852 | 0.772 |

The earlier target-band 0.172 value belonged to the original predictor, not mass2. Mass2's observed target-band safe success was 0.043; it did not establish the claimed coverage benefit. Band comparisons also contain the legacy controller-cadence confound.

**Mass-2+vla predictor (B) headline** (`push_torch_mass2vla`, same mass range + VLA states added to training):

| arm | SR | CVR | safe | Δ safe vs A |
|---|---|---|---|---|
| `checkvla_orbisim` | 0.737 | 0.337 | 0.518 | −0.001 |

The observed safe-success values were A 0.519 and B 0.518, versus original 0.549. Neither coverage variant improved the original aggregate; this does not establish that coverage was the dominant cause. No significance claim is made.

**Per-policy recalibration:** τ recalibrated on OpenVLA rollouts (was on `bc`). `is_ood()` keeps original ranges (friction 0.5–1.5, mass 0.6–1.6), so pooled-OOD numbers stay comparable across predictors. `friction 0.2×` remains a true OOD test.

---

### Update 6 Oct (late) — full VLA actor passes "failures are physical" check after DAgger

**`llm` actor (full OpenVLA, LLM readout), 10-step head + DAgger rounds 1 and 2** (each round: the current policy drives 1024 episodes at nominal physics, the expert labels every visited state with its 10-step chunk from a cloned sim). Policy SR at nominal during collection: 0.60 (round 1), 0.90 (round 2). Closed loop, actor only, `--quick`:

| actor | friction ≥ 0.6×: SR / CVR / safe | friction 0.2×: SR / CVR / safe |
|---|---|---|
| `bc` | 0.95 / 0.01 / 0.95 | 0.62 / 0.97 / 0.03 |
| OpenVLA `vis` | 0.91 / 0.08 / 0.88 | 0.62 / 1.00 / 0.00 |
| OpenVLA `llm`, no DAgger | 0.58 / 0.52 / 0.47 | 0.44 / 0.94 / 0.06 |
| OpenVLA `llm`, 10-step + DAgger ×2 | **0.88 / 0.05 / 0.84** | 0.66 / 1.00 / 0.00 |

The full VLA now matches `vis` in distribution and fails only when physics changes, so it passes the plan's §2 precondition. It joins the headline as a genuine VLA, removing the "vision backbone only" caveat of `vis`.

**Predictor coverage fix, step A (running, `results/push_torch_mass2`).** Retrain the physics predictor with mass 0.5–2.0 (was 0.6–1.6); friction unchanged at 0.5–1.5, so friction 0.2× stays a true out-of-range test. Implementation: `--variant` writes to a separate results dir so the v7 predictor is untouched, and `--train_mass` overrides the range for **collection and τ calibration only**. `is_ood()` deliberately keeps the original ranges, so pooled-OOD numbers stay comparable across predictors. τ is recalibrated on OpenVLA rollouts (it was calibrated on `bc`).

**Step B (data collected):** `src/vla/collect_vla_dyn.py`, 1024 episodes of OpenVLA-driven rollouts with the expert data's correlated action noise, over the same ranges (episode-violation rate ~0.55–0.6, like the expert data). To be merged into the predictor's training data, so it also sees the states the VLA visits.
