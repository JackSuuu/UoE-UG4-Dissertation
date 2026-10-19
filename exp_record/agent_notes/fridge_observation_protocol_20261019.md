# Experiment 1: fixed camera framing, pure actor, three paired initial states

User authorizes parallel experiments 1/2/3. This is experiment 1, on GPU1 only.
Experiment 2 separately qualifies shared actor skill; experiment 3 develops and
tests autonomous candidate gating on a native task, not claimed refrigerator efficacy.

## Frozen intervention before model outcomes

Six episodes: seeds0/1/2 × original robot-side camera / framed-side camera.
Same R2 open-fridge geometry, 20 stationary open-gripper settling intervals, same
official TF4.40.1 frozen LIBERO10 checkpoint and instruction
`pick up the cream cheese box inside the fridge`. 600model queries/actions and
1,200s episode wall cap excluding model load, stop on native environment done or
budget. Batch1, GPU1 with matching CUDA/EGL IDs. No action route or repair.
Open-start grasp diagnosis is not full opening/retrieval/placement/closing success.

Calibration uses seed0 reset only, with no actor queries. The original camera's
orientation and FOV remain fixed. Collect transformed collision/visual geometry
vertices for fridge, item and gripper (box/mesh; bounding sphere for other types).
Center on their world AABB midpoint, and choose distance along the existing optical
axis to contain the points within sqrt(.9)*.9 of sensor half-width/height, plus2cm
distance margin. sqrt(.9) corresponds to the existing policy crop;10% is a fixed
framing margin. No outcome-dependent viewpoint search, no per-seed recalibration.
Freeze resulting pose and geometry/calibration metadata once; use it for all seeds.
This is privileged reset-only observation calibration, not visual localization and
not an autonomous action. It changes framing/scale and potentially occlusion; it
does not guarantee depth visibility or isolate every component of visual shift.

Before actor queries, smoke all6scenes, assert exactly matching paired physical
states/controller and unchanged wrist images. Check reset0new framing by image and
geometry projection, not task success. Retain a failed visibility qualification
rather than searching camera positions using successful trajectories. Multiple
seeds may have little physical diversity; report actual initial hashes/displacements.

## Trace and evidence

Reuse the existing `libero_drawer.decision_trace` structure concept: pre-decision
evidence separated from post-action outcomes. Actor-only trace explicitly says no
gate/repair, model proposal equals executed action, no handoff. Logged current-state
geometry is diagnostic only and never enters actor action selection. This is an
observable execution proof, not internal model reasoning or causal explanation.
Source files and protocol are copied and SHA256 hashed per run before rollout.

Log raw two-camera images, simulator states/controller fingerprints, tokens,
native/executed actions, grasp/item/EEF/contact evidence, query times and termination.
Fresh-process frozen-source replay validates physics/controller/images, pre-action
trace evidence and post-action diagnostics. All cases and failures are retained.

## Review and interpretation

Produce original-left / framed-right camera diagnostic MP4 for every seed, same20fps
simulation clock, zero-intervention labels, evidence/action source and timing, held
end frames explicitly labeled. These are actor observation controls, not a fabricated
VLA/framework pair. Mechanism arm videos are experiment3's separate deliverable.

Primary endpoint: any measured grasp and item displacement; report near-zero command
frequency and unique output sequences descriptively. No success-rate inference from
three development seeds. Framing benefit would support an observation intervention;
continued failure would not prove the model is incapable under all possible views.
Observation changes must be shared with baseline/framework in later efficacy tests.
