# A1 assembly development execution record

## Initial preparation and failures (before launch)

User requested actual integration of our prediction/CheckVLA framework and assembly
experiments. Official adapters were inspected and remain unimplemented templates;
the OrbiSim author page still says "Code coming soon." We connect the repository's
self-developed learned dynamics and RefCheckVLA instead, explicitly labeled.

Physical setup development failures were not robot-policy trials: initially the
handle position was sampled before the nut settled; a20-step settling prefix fixed
that. A60-step transport stopped short; a fixed140-step transport reaches the entry.
The final320-step physical setup includes settling, approach, grasp, lift and
transport. Three fixture seeds/offsets passed bilateral-contact checks. No object
weld or teleport is used. No OpenVLA task capability has yet been established.

`assembly_a1_preflight_20261015_a/b/c` preserved failed pixel-exact replay checks.
In `_b` and `_c` state_max_error=0. In `_c`, at step60 exactly one RGB channel
at raw pixel(y204,x3) differed by one level,157 vs156. All physical state entries
were bitwise identical. The issue was not hidden by rounding hashes or tolerance.
Before any actor trials, A1 was set to disable shadow/reflection passes and GL
dithering uniformly. `_d` then passed both130-step fixtures, instrumentation
invariance and exact rendered-image/physical-state/action/metric replay. Independent
process replay of aligned case on GPU1 and misaligned case on GPU2 also passed.
This is a rendering-configuration workaround, not an isolated causal diagnosis of
the old renderer and not a resolution of M1's separate failure.

Aligned scripted insert/open/retract: complete dwell24 intervals; no nut/peg force
in the collision-free insertion. x+12mm fixture: peak nut/peg summed load58.413N,
no native success and zero full dwell. These are physical fixtures, not learned
framework wins. Shorter90-step earlier misaligned probe peaked41.781N. The load
is an instrumented simulator quantity, not a validated injury/damage threshold.

The neural preflight verifies8 candidates×3 steps, actual action-conditioned
CheckVLA scores, differentiability and nonzero action sensitivity. Random-weight
fixture sensitivity is not prediction accuracy; accuracy is measured after training.

## Frozen execution pipeline

Protocol: `exp_record/assembly_a1_protocol_20261015.md`.
Supervisor: `scripts/launch_assembly_development.py`.
Runtime: `scripts/run_assembly_development.py`.
Expected output: `/home/ami/scratch/assembly_a1_20261015_a`.

Stages:40 independently seeded collection episodes (6400 transitions),1800
minibatch8 training updates, independent calibration/validation reporting, freeze
checkpoint, then30 paired development continuations and per-episode isolated exact
replay. Annotated2x videos, all candidate forecasts and actual action differences
are saved. The nominal actor is the unchanged full7B LIBERO10 OpenVLA checkpoint;
its transfer to assembly is explicitly unestablished. No silent scripted replacement.

All original M1 results/harms remain. The new protocol is not a resumed/revised M1.
Launch status/PID is recorded below only after an actual detached launch.
