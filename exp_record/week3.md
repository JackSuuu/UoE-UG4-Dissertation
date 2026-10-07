# Week 3 (from 5 Oct 2026) — Plan phase P1, Month 1

Newest entry at the top. Index and conventions: [README.md](README.md).

---

### Update 7 Oct — Predictor coverage fix complete; full VLA headline with two predictor versions

**Mass-2 predictor (A) headline** (`rq2_openvla_vis_h10.json` in `push_torch_mass2`, mass 0.5–2.0, τ on OpenVLA):

| arm | SR | CVR | safe | Δ safe vs none |
|---|---|---|---|---|
| `none` | 0.852 | 0.359 | 0.596 | — |
| `gt_shadow` (oracle) | 0.746 | 0.073 | **0.727** | **+0.131** |
| `checkvla_orbisim` | 0.749 | 0.294 | 0.549 | −0.047 |
| `checkvla_vision` | 0.785 | 0.387 | 0.519 | −0.077 |

**Split by band:**

| band | `none` | oracle | `checkvla_orbisim` |
|---|---|---|---|
| friction 0.2× (target band) | 0.020 | 0.500 | **0.172** |
| friction ≥ 0.6× | 0.870 | 0.852 | 0.769 |

In the target band the learned verifier helps the VLA **8.6×** and captures **31% of the oracle's headroom**. The net loss is entirely in the normal-friction band, concentrated in the **heavy cells**, where it *raises* CVR while the oracle removes it: friction 1.4×/mass 2.0× 0.27 → 0.45 (oracle 0.03); 1.8×/2.0× 0.25 → 0.39 (oracle 0.03).

**Mass-2+vla predictor (B) headline** (`push_torch_mass2vla`, same mass range + VLA states added to training):

| arm | SR | CVR | safe | Δ safe vs A |
|---|---|---|---|---|
| `checkvla_orbisim` | 0.743 | 0.292 | 0.548 | −0.001 |

VLA-driven data gave **no measurable improvement** (0.549 → 0.548). The coverage gap was the dominant factor; VLA states added marginal benefit.

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
