# Week 3 (from 5 Oct 2026) — Plan phase P1, Month 1

Newest entry at the top. Index and conventions: [README.md](README.md).

---

### Update 6 Oct — the full 7B VLA now passes the "failures are physical" check; predictor coverage fix running

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
