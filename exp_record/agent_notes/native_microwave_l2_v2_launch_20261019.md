# V2 actual launch: timing-only g+120 experiment

Implementation and preparation frozen at `c7f0c74`. Detached command submitted:

```bash
setsid --wait /home/ami/miniconda3/bin/python -B -u scripts/run_native_microwave_l2_v2.py \
  --launch --plan /home/ami/scratch/native_microwave_l2a_v2_preparation_20261019/plan.json \
  --out /home/ami/scratch/native_microwave_l2a_20261019_v2 \
  > /home/ami/scratch/native_microwave_l2a_20261019_v2.supervisor.log 2>&1 < /dev/null
```

Plan SHA256`dc19052f70815610cb04e4f28c44b491433777f8d7574e2c32278c3e006c9c6e`.
PhysicalGPU2/EGL1andGPU3/EGL0, UUID-masked, batch1.18branches/576calls maximum;
six original-suffix-exact C0controls and fresh replays gate C1thenC2sampling.
Root4/t215smoke reconstructed228frames exactly with zero model queries.
V1implementations remain hash-identical; explicit version adapters pin all seven
implementation files and reject foreign controls/replays. All operational caps,
sources, seeds, H32, labels and L2Bgate unchanged from v1.

Command submission is not a branch result. L2Aonly; optional L2Brequires a complete
positive audit and separate scheduling. Root4held-grasp preservation alone remains
U=false. The new protocol tests earlier engaged-object timing; no runtime failure
alarm, selector or verified repair is asserted. All six matched roots remain.

Preparation/smoke archive: `backups/native_microwave_l2_v2_launch_20261019`.
Design/failure-analysis archive: `backups/native_microwave_l2_v2_design_20261019`.
