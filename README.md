# UoE-UG4-Dissertation

Undergraduate (UG4) dissertation project — University of Edinburgh.

**Title:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization in Embodied Agents

**Current focus (14 Oct 2026):** task-first physical prediction and bounded correction for complete manipulation. The LIBERO book/caddy run is currently a frozen OpenVLA baseline with physical diagnostics, not an active physics-repair framework. Released-placement completion is 1/3 development roots; learned efficacy and generalization remain to be established.

**[Videos, artifact backup and recovery instructions](backups/README.md)** · **[Pinned LIBERO submodule](third_party/LIBERO)**. Large experiment files use Git LFS; clone with submodules and run `git lfs pull` to restore them.

## Contents

- [`core_research_plan.md`](core_research_plan.md) — authoritative research plan and current scope (Oct 2026 – Sep 2027)
- `notes/` — proposals, experiment plans, papers, figures and working notes
  - `Dissertation_Proposal_gsu.pdf` — original dissertation proposal
  - `Dissertation_Proposal_gsu_new_sep_19.md` — revised proposal (Genesis, causal/structured Physical CoT, VLA/Embodied Data Pyramid updates)
  - `Experiment_Phase_0.md` — Phase 0/1 benchmark plan (Robosuite NutAssembly, Diffusion Policy, OOD friction/mass evaluation)
  - `Experiment_plan_Phase_1.md` — Phase 1 plan: CheckVLA + OrbiSim-Dynamics + Genesis execution-time physical verification
  - `Experiment_plan_1year.md` — historical one-year planning notes; current scope is in the root plan above
  - `note.md` — working notes
- `exp_record/` — experiment progress log, one file per week (see `exp_record/README.md`)
- `src/` — Phase 1 experiment framework (see `src/README.md`)
