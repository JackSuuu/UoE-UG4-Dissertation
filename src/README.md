# phase1_exp — Phase 1 experiment framework

Code for `Experiment_plan_Phase_1.md`: VLA + OrbiSim-Dynamics + CheckVLA + Genesis.

> **Status: this is a framework with STAND-IN components.** Every slot in the
> system sits behind an interface (`interfaces.py`) and is picked in
> `registry.py`. The in-repo implementations are placeholders written by us —
> they are **not** the real VLA / OrbiSim / CheckVLA, and the Genesis backend
> has **never been executed**. Results from stand-ins must not be reported as
> results about the real components.

## Component status

| Slot | Stand-in in repo (default) | Real target (adapter) | Adapter status |
|---|---|---|---|
| Base policy (VLA) | `BCPolicy` — MLP behaviour cloning, obs → 10-step action chunk, trained at nominal physics | OpenVLA-7B — `adapters/openvla_policy.py` (`--policy openvla`) | skeleton: HF load + `predict_action` written, **action mapping / camera / fine-tuning TODO** |
| Fast predictor ("orbisim" role) | `OrbiSimDynamics` — our own distilled MLP ensemble (object-centric, differentiable). **Not OrbiSim's architecture or code.** | OrbiSim-Dynamics (arXiv 2605.16395) — `adapters/orbisim_official.py` (`--orbisim_impl official`) | template only — code availability not checked |
| Baseline predictor ("vision" role) | `VisionWM` — our own CNN+GRU risk world model on 32×32 synthetic renders | CheckVLA's visual WM — `adapters/checkvla_official.py::CheckVLAVisualWM` (`--vision_impl official`) | template only |
| Verifier | `checkvla/reference.py::RefCheckVLA` — our implementation from the paper's one-line description (conformal threshold + gradient suffix repair) | CheckVLA (arXiv 2607.26789) — `adapters/checkvla_official.py::CheckVLAOfficial` (`--verifier official`) | template only |
| GT simulator | `sims/torch_push.py`, `sims/torch_cloth.py` — our own differentiable PyTorch physics | Genesis — `sims/genesis_push.py` (`--backend genesis`, Task A only) | written against genesis-world 0.2/0.3 API, **never run** |

What *is* ours by design (not a stand-in): the experiment protocol, OOD grid,
gradient path audit, metrics, and the three RQ3 systems pieces
(`systems/bptt.py` checkpointing + gradient stabilizer, `systems/scheduler.py`
async fast/slow engine).

## Layout

```
interfaces.py          PolicyAdapter / PredictorAdapter / VerifierAdapter contracts
registry.py            chooses implementations from CLI flags
sims/                  GT simulators (Env interface in sims/base.py)
models/nets.py         stand-in policy + predictors
checkvla/reference.py  stand-in CheckVLA (score / check / repair)
checkvla/verifier.py   component-agnostic closed-loop Controller + episode runner
adapters/              skeletons for the real components  <-- fill in on the server
systems/               RQ3: checkpointed BPTT, gradient stabilizer, async scheduler
experiments/           audit, collect, train, calibrate, rq1, rq2, rq3, figures
run_all.sh             whole pipeline for one task/backend
```

## Replacing a component

1. Implement the adapter so it satisfies the Protocol in `interfaces.py`
   (read the docstring at the top of the adapter file — it lists every TODO).
2. Run with the flag, e.g.
   ```bash
   export ORBISIM_ROOT=/path/to/orbisim ORBISIM_CKPT=/path/to/ckpt
   python experiments/calibrate.py       --task push --orbisim_impl official --roles orbisim
   python experiments/rq1_calibration.py --task push --orbisim_impl official
   python experiments/rq2_eval.py        --task push --orbisim_impl official \
          --arms none gt_shadow checkvla_vision checkvla_orbisim
   ```
   `calibrate.py` merges the new tau into `taus.json`; always re-calibrate after
   swapping a predictor or verifier.
3. `train.py` only trains the **stand-ins**. A real predictor that does not
   output our risk channels needs a risk head — see route (a)/(b) in
   `adapters/orbisim_official.py`.

Key contracts (details in `interfaces.py`):
* risk is `(B, H, 2)` = `[force/force_limit, deform/deform_limit]`, >1 = violation
* predictor `risk(ctx, actions)` → `(mean, std)`; set `differentiable=True` only if
  gradients w.r.t. actions work (enables gradient suffix repair; otherwise the
  reference verifier falls back to down-scaling)
* policy returns a `(B, H, A)` chunk in the sim's action space (velocities)
* a real VLA needs `env.render_rgb()` → only `GenesisPushEnv(camera=True)` has it

## Genesis on the server

```bash
pip install genesis-world        # + torch matching your CUDA
python experiments/audit_gradients.py --task push --backend genesis --genesis_probe
```
The probe records `valid / zero / nan / error(msg)` per cell and horizon, so an
API mismatch shows up as `error` with the message instead of crashing. If
building the scene fails, the functions to adapt are `_build`, `_read`,
`_write`, `_contact_force`, `render_rgb` in `sims/genesis_push.py`.

## Running the pipeline

```bash
pip install -r requirements.txt
bash run_all.sh push torch            # Task A, torch GT, all stand-ins
bash run_all.sh cloth torch           # Task B
bash run_all.sh push genesis          # Task A with Genesis GT (once it runs)
bash run_all.sh push torch --quick    # tiny smoke-test sizes
```
Component flags pass through, e.g. `bash run_all.sh push torch --orbisim_impl official`.
Outputs go to `results/<task>_<backend>/` (json, npz, Fig A–F png/pdf).

## Known issues (not yet fixed)

* **Stand-in OrbiSim under-predicts rare risk spikes.** In a medium local run
  on push, predicted risk topped out around 0.3 while GT reached ~2.9, so the
  trigger never fired. `train.py` now oversamples violation windows and
  switches the risk loss to MSE, but this fix has **not been verified**.
* Cloth has no self-collision; Genesis PBD cloth is not wired (not differentiable).
* RQ3 memory numbers need CUDA (`torch.cuda.max_memory_allocated`).
