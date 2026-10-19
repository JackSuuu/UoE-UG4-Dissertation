# UoE-UG4-Dissertation

Undergraduate (UG4) dissertation project — University of Edinburgh.

**Title:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization in Embodied Agents

**Current focus (19 Oct 2026):** test runtime repair on unstructured, long-horizon manipulation with explicit data provenance and OOD splits. [Week5](exp_record/week5.md) · [six-task capability/OOD design](exp_record/agent_notes/unstructured_ood_design_20261019.md). The current assembly controller is task-specific; deformable/cutting/cleaning capability and real-time operation are not established.

**Locked architecture:** high-level subgoals → frozen OpenVLA → risk/event-boundary monitoring → dual-mode gate → bounded local repair. [Authoritative design](exp_record/agent_notes/libero_framework_design_20261014.md) · [layer-by-layer implementation / next long-horizon task](exp_record/agent_notes/libero_five_layer_implementation_20261014.md).

**Latest completed experiment:** [A2:80/80verified trajectories](exp_record/agent_notes/assembly_a2_results_20261019.md). VLA8/20; shared-gate geometric feedback and hybrid18/20. Prediction ranking never overrides geometry, and the hybrid harms one baseline success. [VLA-left / hybrid-right paired videos, including negative cases](backups/assembly_a2_paired_review_20261019/README.md).

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
- `exp_record/` — weekly summaries; detailed protocols/results in `exp_record/agent_notes/` (see `exp_record/README.md`)
- `src/` — Phase 1 experiment framework (see `src/README.md`)
