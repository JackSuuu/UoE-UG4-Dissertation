# L2A paired nominal-versus-sampled videos

Left: C0nominal greedy. Right: C1orC2sampled model candidate. These are local
candidate diagnostics, not complete-framework comparisons. All six roots and
both samples retained, including the negative failure-root cases.

| Root | C1comparison | C2comparison | Interpretation |
|---|---|---|---|
|0|[video](videos/root0_C0_vs_C1.mp4)|[video](videos/root0_C0_vs_C2.mp4)|original failure; no useful progress|
|1|[video](videos/root1_C0_vs_C1.mp4)|[video](videos/root1_C0_vs_C2.mp4)|original success; containment progress|
|2|[video](videos/root2_C0_vs_C1.mp4)|[video](videos/root2_C0_vs_C2.mp4)|original success; containment progress|
|3|[video](videos/root3_C0_vs_C1.mp4)|[video](videos/root3_C0_vs_C2.mp4)|original success; samples close sooner|
|4|[video](videos/root4_C0_vs_C1.mp4)|[video](videos/root4_C0_vs_C2.mp4)|original failure; no object recontact|
|5|[video](videos/root5_C0_vs_C1.mp4)|[video](videos/root5_C0_vs_C2.mp4)|original success; all complete locally|

All videos are2048×1440,20fps,33frames: the shared boundary followed by32aligned
action endpoints. Playback follows simulator action time, not model latency.
Early native exits are visibly labelled and their last frame held for alignment;
held frames are not additional actions. Main-camera images are rotated180degrees
and uncropped. Overlays show subgoals, candidate source/seed, action owner, budget,
exit and absent selector/verified-repair/handoff. Proxy labels appear only after
their measurement point and are explicitly offline.

Frame counts, same-boundary PNGs, source identities and142unchanged source hashes
checked; sample overlays visually inspected. The56-member metadata archive includes
frame maps, checks, inspection samples and renderer. Encoded video is lossy; raw
L2A tapes remain authoritative. L2Bgate was false; no continuation was performed.
