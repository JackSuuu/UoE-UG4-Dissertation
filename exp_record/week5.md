# Week 5 (19–25 Oct 2026) — Unstructured transfer, honest OOD and paired evidence

Newest entry at the top. [Weekly index](README.md). Detailed records live in
[`agent_notes/`](agent_notes/README.md).

### Update 19 Oct — A2 paired videos and unstructured probe independently recovered

Pushed results, five representative comparison MP4s, U0probe and record reorganization
as`46a5371`. Independently fetched GitHub/LFS bytes and checked697A2archive members,
8U0probe members and all5standalone comparison videos against SHA256. Proof:
[`week5_remote_verified_20261019.json`](../backups/week5_remote_verified_20261019.json).
All20A2pairs are locally rendered; the representative5include negative outcomes.
No six-task robot efficacy cohort is running; U0is complete and remains a physics
backend qualification only.

### Update 19 Oct — Six unstructured task families require explicit capability and OOD tests

User prioritizes cutting vegetables, peeling garlic, raw-meat handling, refrigerator
retrieval, dish washing and garment folding. [U-series design and data/OOD contract](agent_notes/unstructured_ood_design_20261019.md)
defines what the current mixed controller supports, what source/target experience
means, task-specific physical criteria, full-reset trials and paired recovery controls.
The suggested LIBERO multi-object pilot was not launched before this steering.

Current A2 is task-specific: a frozen LIBERO10actor, a learned nut/peg predictor and
handcrafted assembly phases/geometry. Target-specific fitting or threshold/candidate
design is adaptation, even if actor weights stay fixed. New seeds alone are not
unseen-task generalization. Component-relative exposure and held-out families/assets
must be registered before inspecting evaluation outcomes. Unknown foundation-model
pretraining overlap stays unknown.

Installed capability audit: Genesis1.4.2and MuJoCo3.14available; OmniGibson/IsaacSim,
ManiSkill and SoftGym/PyFlex not installed. **U0actually executed** on healthy GPU2:
120steps of cloth draping over an asymmetric obstacle,1089particles,121recorded
states; maximum pair-distance change0.14776m confirms non-rigid evolution rather
than a rigid-cloth proxy. Independent-process repeat is **not bit-exact**: maximum
particle-coordinate difference~0.0000505764m. No post-hoc tolerance is declared a pass.
This establishes executable deformable backend primitives only—not garment folding,
robot manipulation, real-time control, calibrated material physics or OOD efficacy.
No target-task policy training or six-task hybrid evaluation has been performed.

Plan: articulated retrieval and cloth manipulation first; washing needs actual
residue/tool interaction, cutting needs validated fracture or an explicitly named
symbolic benchmark, peeling needs adhesion/separation, meat requires a specified
deformable handling operation and validated material assumptions. Runtime response
must be measured with a moving world and timestamps, separately from paused simulation.

### Update 19 Oct — A2 verified18/20; prediction ranking never overrides geometry

[Complete results](agent_notes/assembly_a2_results_20261019.md): all80live episodes
and30historical actual-state audits completed, with independent exact replay.
VLA8/20; feedback80, phase_geometry80and hybrid80each18/20. Old/new cohorts separately
score3/10+5/10forVLA and9/10+9/10for each intervention arm. Hybrid rescues11baseline
failures but harms1baseline success. Prediction override count0; hybrid and phase
geometry have identical executed trajectories in all20cases. Feedback achieves the
same completion with lower mean peak load27.97vs30.39N and fewer corrections1442vs1600.
These are small development results, not universal/generalization/safety evidence.

User requests paired video as the default deliverable. Rendered20matched VLA-left /
hybrid-right MP4s with synchronized decision index,2x simulated-time labels and held
end-frame labels. Five representative clips include preservation, old/new rescues,
harm and both-failure; all20pairs remain indexed. [Video index and download notes](../backups/assembly_a2_paired_review_20261019/README.md).

### Update 19 Oct — Weekly records reorganized without changing frozen protocols

Moved38non-week detailed records into`exp_record/agent_notes/`; weekly files stay
at the outer level. Relative navigation repaired, provenance/hashes retained in
`agent_notes/relocation_manifest.json`.32relocated files are byte-identical; six
historical result/launch notes change only relative Markdown links. Frozen protocol
bytes and source code are preserved. Historical launchers that require old paths
should be run from their frozen commit/source archive. Original`notes/`proposals
and the user's uncommitted one-year-plan edit are preserved.
