# Experiment record

English weekly progress logs. Newest update first within each week; detailed
protocols, diagnostics and execution records are in [`agent_notes/`](agent_notes/README.md).
The authoritative research plan is [`core_research_plan.md`](../core_research_plan.md).

| Week | Dates | Main events |
|---|---|---|
| [1](week1.md) | to29Sep2026 | Genesis bring-up, rigid-task baselines, predictor/repair audit |
| [2](week2.md) | 30Sep–4Oct | CheckVLA-style mechanisms, OpenVLA integration and cadence confounds |
| [3](week3.md) | 5–11Oct | Matched cadence, calibration, confirmation and candidate repair |
| [4](week4.md) | 12–18Oct | Negative gate studies, fullVLA LIBERO transfer, interrupted M1; A1/A2 launched |
| [5](week5.md) | 19–25Oct | A2 completion, paired videos, unstructured capability and OOD design |

## Current findings

* [A2 results](agent_notes/assembly_a2_results_20261019.md):80/80verified trajectories.
  VLA8/20; shared-gate feedback and hybrid18/20. Prediction ranking never overrides
  geometry. No task-family generalization or real-time claim.
* [Unstructured tasks and data/OOD contract](agent_notes/unstructured_ood_design_20261019.md):
  refrigerator retrieval, clothes, dish washing, cutting, garlic and raw meat.
* [Paired videos](../backups/assembly_a2_paired_review_20261019/README.md): VLA left,
  hybrid right; representative rescues **and** harms/both-failures, all-case manifest.
* M1 remains stopped; its unresolved in-process replay discrepancy is historical,
  not resolved by later A1/A2 independent verification.

## Conventions

* One`weekN.md` per week; use`### Update <date> — <finding>` entries.
* State measured quantities first, then interpretation and untested limitations.
* Store non-week notes in`agent_notes/`; keep frozen protocols immutable. Historical
  launch paths/hashes resolve through the recorded commit/source archive.
* Default review video is a same-case baseline/hybrid pair with honest timing labels;
  raw single-arm recordings remain source evidence. Publish representative negative
  cases alongside positive cases through Git LFS.
* Detached jobs use`start_new_session=True`or`setsid nohup`; record PID/output/source
  freeze and independent recovery. Do not mistake a launched job for a completed run.
