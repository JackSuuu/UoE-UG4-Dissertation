# UoE-UG4-Dissertation

Undergraduate (UG4) dissertation project — University of Edinburgh.

**Title:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization in Embodied Agents

**Current focus (14 Oct 2026):** task-first physical prediction and bounded correction for complete manipulation. The completed LIBERO book/caddy baseline has released-placement completion on 1/3 development roots. A first **privileged physics-selection** study is now implemented; see its [protocol and renderer-fix/restart record](exp_record/libero_preservation_renderer_fix_20261014.md). Learned efficacy and generalization remain to be established.

**Locked architecture:** high-level subgoals → frozen OpenVLA → risk/event-boundary monitoring → dual-mode gate → bounded local repair. [Authoritative design](exp_record/libero_framework_design_20261014.md) · [layer-by-layer implementation / next long-horizon task](exp_record/libero_five_layer_implementation_20261014.md).

**Latest diagnostic completed:** [24 verified trajectories and method comparison](exp_record/libero_preservation_results_20261014.md). Privileged lookahead and simple hold both reach placement on 3/6 development roots; lookahead does not outperform hold on continued execution. [Root 2 four-method 2x video](backups/libero_preservation_20261014_b/videos/root_2_methods_2x.mp4).

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
