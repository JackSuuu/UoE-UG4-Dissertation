# Annotated native baseline videos

Six complete original episodes, including every failure. Each video shows agent
view left and diagnostic wrist view right, with recorded native subgoal predicates,
grasp evidence, action owner, consumed budget and exit. These are two cameras of
the same baseline; there is no framework comparison. Trigger/candidate/repair/
handoff are explicitly labelled absent. Privileged diagnostics are not policy inputs.

| Video | Native outcome | Actor actions | Replay |
|---|---|---:|---|
| [Microwave root0](videos/microwave_root0.mp4) | failure |520|exact|
| [Microwave root1](videos/microwave_root1.mp4) | success |444|exact|
| [Microwave root2](videos/microwave_root2.mp4) | success |397|exact|
| [Two pots root0](videos/two_pots_root0.mp4) | recorded success |500|wrist-image mismatch|
| [Two pots root1](videos/two_pots_root1.mp4) | failure |520|wrist-image mismatch|
| [Two pots root2](videos/two_pots_root2.mp4) | failure |520|wrist-image mismatch|

All2,967frames retained: reset plus10initialization frames and each actor action
endpoint. Playback20fps follows simulator action time, not inference wall time;
no holds or discarded intervals. Both raw camera arrays are rotated180degrees
for display; the agent display is uncropped, whereas the actor's official crop is
recorded separately in the raw episode evidence. Native success is not a dwell/
fully-released certificate. Video encoding is lossy; raw image tapes govern replay.

Frame counts,20fps and1024×800resolution verified for all six videos; the final
microwave-root1frame was inspected for readable goal/owner/budget/exit labels.
`manifest.json` hashes the videos and source/check metadata. Original complete
episode archive and independent recovery proof: `../native_long_l1_partial_20261019/`.

Fresh remote-only recovery passed for all six videos, three source/check members
and1,862rendering-diagnosis members. Proof: `remote_recovery_passed.json`.
