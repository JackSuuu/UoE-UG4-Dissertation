# R2 actual framework execution

## Launch

User explicitly requests a real frozen-VLA versus VLA+framework comparison. Source
and [protocol](fridge_r2_framework_protocol_20261019.md) committed/pushed as`c4afde8`
before launch. Detached supervisor PID3653705, healthy GPU1baseline/GPU3framework.
Working output: `/home/ami/scratch/fridge_r2_framework_20261019_a`.
Supervisor log: `/home/ami/scratch/fridge_r2_framework_20261019_a.supervisor.log`.

Planned4runs: seed0closed/open × actor/framework,600actions/600queries per arm.
Both arms run the full frozen actor and use shared phase-prompt rules. The framework
adds online progress/grasp checks, conditional bounded corrections and revalidated
handoff. Privileged geometry/contact is declared; no visual localization or learned
predictor is claimed. Closed-door runs attempt the complete sequence; open-door
runs are separate diagnostics. Neither arm has a scripted prefix.

The supervisor replays each completed tape in a fresh process, verifying recorded
proposals → gate decisions → executed actions → physics/controller/task conditions.
It automatically renders left-baseline/right-framework videos under `paired/`,
including intervention ownership, event labels and explicit terminal-frame holds.
No successful outcome is claimed at launch. Negative or incomplete results remain
part of the cohort; changed thresholds/primitives require a new version.

## Checks before interpretation

Six guard tests pass: required event history and released dwell, open-start exclusion
from closed-task success, release dwell retaining the placement phase, bounded
unsuccessful exit, fresh-proposal handoff, and budgeted grip retention.
A100-step live environment smoke exercise triggered actual repair at step29and
executed71corrections; this validates integration, not a successful task.

`qualify_fridge_endpoint.py` additionally constructed three **evaluator-only** fixtures:
supported/released item inside placement bounds (positive), supported item outside
bounds (negative), and unsupported item above bounds (negative). All pass. None is
counted as full completion without the observed opening/grasp/extraction history.
These fixtures intentionally install object state and are not actor/repair episodes.
Output: `/home/ami/scratch/fridge_r2_endpoint_20261019_a`.

Endpoint checks remain development predicates. The full task and recovery effect
must be measured in the real actor rollouts; fixture success is not task success.
