# Joint runtime artifact, 22 Oct 2026

`joint_components.pt` contains the trained outcome-risk MLP, training-only
normalization, an empirical validation threshold and **144 real visual
keyframes** from three completed successful training trajectories.

Training command (GPU2 UUID mask, not physical GPU0):

```bash
python -B -m src.training.train_joint_components \
  --fusion-dir /home/ami/scratch/fusion_handoff_training_20261022_r2 \
  --out /home/ami/scratch/joint_components_training_20261022_r1 --device cuda:0
```

Train roots 1/2/5; validation roots 3/4; root0 explicitly excluded from this
training/calibration script. The source trajectory manifest and pretrained
encoder provenance match the fusion artifact. Stage prototypes are distilled
from measured stage labels, not externally annotated semantic goals.

Risk targets are measured predicate/lift regression or sustained acquisition
stagnation over 24 observed nominal actions, not complete-task failure labels.
There are 92 normal / 17 violation training samples and 145 normal / 146 violation
validation samples. Threshold 0.2741200923919678 is the empirical 95th percentile
of normal validation scores. At this threshold the validation confusion is
**TP=0, FP=6, TN=139, FN=146** (risk-positive means intervention-worthy): zero
detected violations, 4.14% false interventions. This is a weak learned detector,
not a conformal guarantee or an established contribution. Do not reinterpret
the generic report's classifier field names as handoff acceptance metrics.

The joint runtime nevertheless consumes the frozen risk head causally, alongside
short-term prediction and measured checks; all consulted scores, effective
triggers and vetoes are recorded so its actual contribution is auditable.
Its bank has 77 acquire, 14 transfer, 11 contain_release and 42 close exemplars.
Retrieval and visual centroid scores jointly select physically admissible stages.
Online writes wait for 12 already observed endpoints of preservation; the bank
is reset between episodes and never seeded with root0 successes or future states.

Installed OpenVLA emits H1 actions. Repair is closed-loop frozen-policy generation
of the remaining execution, with stage-plan revisions, not cached multi-action
chunk regeneration. Numerical offsets and legacy random visual/risk fallback
paths are disabled in the joint experiment.
