# Experiment Record

Running log of experiment progress against `Experiment_plan_1year.md`.
Newest entry at the top.

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
