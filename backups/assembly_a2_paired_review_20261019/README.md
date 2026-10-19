# A2 paired videos: VLA baseline vs hybrid framework

Left: frozen VLA. Right: hybrid80. 2x simulated-time playback, **not wall-clock real-time**. Same initial state and decision-index alignment; shorter runs hold a labeled end frame. Original image is pre-action and load/endpoint annotation post-action.

All20 paired cases are indexed in `manifest.json`. Highlights include every observed outcome category, including harm and both-failure; no favorable-only selection. Hybrid prediction-ranking overrides were zero in A2; geometry and hybrid trajectories were identical.

- **both_success**, 3000/c0: [s3000_c0_vla_vs_hybrid80_2x.mp4](s3000_c0_vla_vs_hybrid80_2x.mp4)
- **rescue**, 3000/c1: [s3000_c1_vla_vs_hybrid80_2x.mp4](s3000_c1_vla_vs_hybrid80_2x.mp4)
- **harm**, 3001/c4: [s3001_c4_vla_vs_hybrid80_2x.mp4](s3001_c4_vla_vs_hybrid80_2x.mp4)
- **both_failure**, 3100/c0: [s3100_c0_vla_vs_hybrid80_2x.mp4](s3100_c0_vla_vs_hybrid80_2x.mp4)
- **rescue**, 3100/c2: [s3100_c2_vla_vs_hybrid80_2x.mp4](s3100_c2_vla_vs_hybrid80_2x.mp4)

MP4 files use Git LFS. On GitHub use **Download raw file** if no inline playback appears.
