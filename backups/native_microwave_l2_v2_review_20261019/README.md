# V2 paired nominal-versus-sampled videos

Left: C0 greedy control. Right: C1 or C2 sampled model candidate, starting at
first grasp+120. All six roots and both samples retained. This is a local
candidate comparison, not full-framework evaluation.

| Root | C1 pair | C2 pair |
|---|---|---|
|0|[video](videos/root0_C0_vs_C1.mp4)|[video](videos/root0_C0_vs_C2.mp4)|
|1|[video](videos/root1_C0_vs_C1.mp4)|[video](videos/root1_C0_vs_C2.mp4)|
|2|[video](videos/root2_C0_vs_C1.mp4)|[video](videos/root2_C0_vs_C2.mp4)|
|3|[video](videos/root3_C0_vs_C1.mp4)|[video](videos/root3_C0_vs_C2.mp4)|
|4|[video](videos/root4_C0_vs_C1.mp4)|[video](videos/root4_C0_vs_C2.mp4)|
|5|[video](videos/root5_C0_vs_C1.mp4)|[video](videos/root5_C0_vs_C2.mp4)|

Root4/C2 retains held-clear level1, equal to its reference: U=false, not recovery.
Root3/C2 achieves containment on an originally successful root. No local branch
completes the native task; the failure-root L2B gate remains false.

Each video is2048×1440,20fps,33frames: shared initial boundary plus32action
endpoints. Playback is simulator time, not inference latency. Labels are offline
and appear only at their measurement horizon. Overlays identify source/seed,
subgoals, ownership, budgets, horizon exits and absent selector/repair/handoff.
Agentview images are rotated180degrees and uncropped.

Frame counts, identical initial PNGs,143unchanged source hashes and readable
overlays passed checks. The52-member evidence archive includes frame maps,
inspection samples, checks and renderer. Raw tapes govern exactness; video is
lossy. Local archive/video hashes passed; independent remote recovery is separate.

Independent remote-only recovery passed at `fdb1ef7`: all52metadata members and
12videos match their recorded sizes and hashes. See `remote_recovery_passed.json`.
