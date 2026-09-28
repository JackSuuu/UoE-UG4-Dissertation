# Experiment Record

Running log of experiment progress against `Experiment_plan_1year.md`.
Newest entry at the top.

---

## Sem 1 · Week 1 (w/c 28 Sep 2026) — Plan phase P1, Month 1

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
1. **Genesis gradient probe is inconclusive.** After the fixes, every cell/horizon reports that the peg position tensor has no `grad_fn`. This may be caused by how the probe uses the Genesis 1.4 API rather than Genesis itself lacking a gradient path. It must not be reported as an audit result until the probe has been checked against the Genesis differentiable-simulation docs.
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
- [ ] Fix the Genesis gradient probe against the Genesis 1.4 docs
- [ ] Literature check: is OrbiSim (2605.16395) / CheckVLA (2607.26789) code public?
- [ ] Ask supervisor about lab robot / F/T sensor access (P4)
- [ ] Confirm GPU budget
- [ ] Wire BC + OrbiSim + CheckVLA and OOD physics into the web demo
- [ ] Turn the RQ3 hybrid/deferral prototypes into measured experiments

### Next week (Week 2) — proposed
1. Full-size torch pipeline in the background → first real stand-in numbers.
2. Rewrite the Genesis probe, then decide whether RQ1 gradient agreement can use Genesis.
3. Code availability check for OrbiSim / CheckVLA.
