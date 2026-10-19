# Continuation baseline implementation and launch

Frozen [protocol](libero_continuation_protocol_20261013.md) and runtime code committed/pushed as `9f39e2b` **before** full-policy continuation launch. Run `/home/ami/scratch/libero_continuation_20261013_a`; three development roots0/1/2 on healthy physical GPUs1/2/3, PIDs691605/691606/691607. Each worker uses `setsid nohup` with detached stdin. Max520policy steps from original initial state; no native-predicate early stop, no scripted release, no physics intervention or training. Full outcomes are pending at this entry.

Implementation: `src/adapters/libero_completion.py`, `scripts/run_libero_continuation.py`, `scripts/summarize_libero_continuation.py`. Independent aggregation checks every saved acceptance component against geometry and substep contacts, recomputes dwell, verifies token/action/state agreement with the original baseline prefix, validates video/state lengths and hashes artifacts. A complete-run HTML report is emitted at `index.html`.

## Checks

-5tests passed:3new geometric/release/motion acceptance tests and2existing simulator ABI checks.
-Exact recorded-action replay of185root0steps with the monitor and500Hz contact recorder: all186states identical (max absolute difference0), first native success unchanged. An extra interface step confirmed native done does not lock the environment.
-After launch, ran an independent **synthetic positive metric fixture**, with no VLA: place the book's actual collision box upright in the rear compartment, initially0.1mm above the floor, open gripper stationary, then60control steps under normal physics. All60sampled intervals met the frozen definition; completion at interval19. This confirms the metric is physically satisfiable in this model, **not** that the policy can achieve it. Saved `/home/ami/scratch/libero_completion_positive_fixture.json`. No thresholds changed.

The synthetic fixture also exposes a useful distinction: geometric released placement passes while native success is false. The native predicate checks a body reference point offset from the collision box; for this valid upright placement, that reference point is outside the native target region. Therefore the new geometry-aware completion is an **alternative task acceptance definition, not a logically nested native-success-plus-extra-conditions predicate**. Both must be reported separately. The frozen monitor deliberately does not require native success. Do not advertise these results as unchanged official LIBERO benchmark SR or infer policy performance from the fixture.

Next on completion: inspect all full trajectories and component failures. If the same actor can complete released placement, proceed to physically justified force/impact/numerical checks and bounded privileged-intervention headroom. If it cannot, first locate the task/control failure before another learned experiment. Endpoint or substep force alone is not yet a validated safety label.
