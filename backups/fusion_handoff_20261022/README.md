# Outcome-trained visual–physical handoff fusion (22 Oct 2026)

Weights: [`../../src/checkpoints/fusion_handoff_v2_20261022/`](../../src/checkpoints/fusion_handoff_v2_20261022/).
Training output: `/home/ami/scratch/fusion_handoff_training_20261022_r2/`.
This directory retains the report, trajectory manifest, predictions and actual
pretrained visual features. Raw policy PNGs/diagnostics remain in the manifest's
source episodes; the JSON records their image/controller/initial-state hashes.

Frozen DINOv2 ViT-S/14 weights (88,283,115 bytes) were obtained from
<https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth>.
SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
They stay in the local Torch cache, not duplicated in Git. `encoder.json`
identifies the exact source hash and policy-RGB preprocessing. No random or
zero visual vectors are used by this classifier.

## Data and result

- Source: completed development prompt-refinement trajectories plus the latest
  bounded-correction root0 trajectory. Identical initial state + action tapes
  are deduplicated: **13 distinct trajectories**, not 37 independent runs.
- Train roots 1/2/5 (3 trajectories, 397 labelled contract samples); validation
  roots 3/4 (7 trajectories, 1,131 samples); evaluation root0 (3 trajectories,
  525 samples). Variants and adjacent frames never cross root boundaries.
- Inputs: real 384-D vision, 12 measured state values plus 12 validity masks,
  requested recovery contract. Training-only normalization; class 1 is pass.
- Labels: current contract plus preservation over the next 12 observed nominal
  actions, or native completion without regression. Repair interruption,
  missing evidence and right-censoring are not fabricated positive outcomes.
  Current false contracts are negative regardless of nominal ownership.
- Validation-selected threshold **0.993425726890564**, using only samples whose
  current contract is true. Conditional false acceptance **1/28 (3.57%)**;
  false rejection **193/283 (68.20%)**. These are correlated sample counts,
  not independent episodes or a statistical coverage guarantee.
- On 11 actual validation handoff observations (including budget exits),
  the classifier accepts one unsafe case and rejects both safe cases.
  Training roots contain no actual repair handoffs. This is a substantive
  data/efficacy limitation, not a reason to claim the head is calibrated.
- Root0 has **zero positive labels**. Rejection of its three current-true
  regression samples is useful diagnosis, not evidence of accepting safe
  handoffs, generalization, or complete-task improvement.

The original threshold-selection attempt is retained in
`/home/ami/scratch/fusion_handoff_training_20261022_r1/`. Its all-sample threshold
diluted conditional false acceptance; r2 corrects that population, without
using root0 to choose the threshold. No root split was changed after evaluation.

## Causal use and evaluation

The new loader rejects legacy weights and missing/zero visuals. The runner can
leave fusion off, record the same scores without controlling exits, or require
both measured recovery and MLP acceptance before a verified exit. Completion
and bounded unverified exits remain separate. Close-stage training is absent
and explicitly unsupported. The framework is not replaced by MLP-generated actions.

`scripts/run_fusion_handoff_ablation.py` runs the matched three-arm root0 test,
retaining failures, original actions/corrections, repair transactions, replay
and independent score-recomputation audits. Task-specific geometry remains
privileged; no scene reconstruction, RL, genuine suffix regeneration or
held-out task generalization is claimed.
