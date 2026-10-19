# R1: contact extraction feasibility and frozen actor diagnosis

* [Revised research plan and fixed R1 protocol](../../exp_record/agent_notes/closed_loop_recovery_r1_20261019.md)
* [Execution record](../../exp_record/agent_notes/fridge_r1_execution_20261019.md)
* [Labeled physical extraction video](contact_extraction_diagnostic.mp4)
* [Measured diagnosis](diagnosis.json), [complete engineering archive manifest](manifest.json)
* [Actor launch snapshot](actor_launch_snapshot.json)
* [Independent GitHub/LFS restore](remote_verified.json):208archive members and the
  labeled video downloaded into a fresh repository and SHA256-verified.

`engineering.tar.gz` retains unsuccessful geometry/servo variants, initial launch
errors, source freezes where available, raw videos, physics states and replay reports.
The completed pixel replay is included; the newly launched actor cohort will be
archived after completion. The earliest inventory probes predate
per-run source copying and retain hashes rather than full intermediate source copies.

The video is **scripted geometric control, not OpenVLA or five-layer efficacy**.
Real pad contact lifts the item clear of the shelf and transports it outward by
0.173744m; grasp remains true from step335 through484 (150 control endpoints).
No particle/object teleportation, grasp attachment or actor prefix is involved.
This is not the complete opening/placement/closing task. Physics/controller replay
is exact; six post-action image fingerprints in v5 differ. A second instrumented
run localizes its differences to four wrist frames:30RGBchannel values differ by1.
All main-camera images (the actor's only visual input) are exact in that replay.
This is still not a fully bit-exact two-camera replay; no tolerance was widened.

![Physical extraction retained in the gripper](contact_extraction_final.png)

Twelve native-actor development trials were launched from frozen commit `080c9a4`,
without the scripted prefix. They compare canonical and basic-skill instructions
on closed/open initial doors over seeds0–2. Full-task success is not scored here.
