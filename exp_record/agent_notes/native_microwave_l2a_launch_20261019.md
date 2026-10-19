# Actual L2A execution launch

Frozen implementation/plan at commit `7e63364`. Detached command submitted:

```bash
setsid --wait /home/ami/miniconda3/bin/python -B -u scripts/run_native_microwave_l2.py \
  --launch --plan /home/ami/scratch/native_microwave_l2a_preparation_20261019_v1/plan.json \
  --out /home/ami/scratch/native_microwave_l2a_20261019_v1 \
  > /home/ami/scratch/native_microwave_l2a_20261019_v1.supervisor.log 2>&1 < /dev/null
```

Scope: L2A only,18model branches, at most576queries/actions, physicalGPU2/3,
batch1, exact prefix reconstruction and fresh replay per branch. C0greedy controls
must match original tokens/actions/policy images/resulting trajectories for all six
roots before C1/C2sampling. Any infrastructure/integrity issue blocks progression;
all missing and interrupted cells remain visible. No L2B or runtime selector.

Plan SHA256`9162f78ff15a71d5fc04ba4fa6e7c87fd4aa548a5f7a383bd556a62efabdfccb`.
The run's `launch.json` records actual process/cell states. Command submission is
not a completed model call or experiment result. Outcome/audit is pending.

Initial health: supervisor3739844in C0phase; root0worker3739886and root1worker
3739887running. At that snapshot query count was0and sampling remained disabled.
The offline audit implementation passes11regression cases and reproduces all six
original suffix proxy labels; it reads saved artifacts only and does not select
runtime actions. Validation artifacts:
`/home/ami/scratch/native_microwave_l2_audit_validation_20261019`.

Design and preparation archives: `backups/native_microwave_l2_design_20261019`
and `backups/native_microwave_l2a_launch_20261019`. Root0/t253reconstruction smoke
passed266frames with zero queries: `backups/native_microwave_l2_smoke_20261019`.
