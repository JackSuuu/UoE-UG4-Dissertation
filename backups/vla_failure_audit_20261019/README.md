# Actor failure audit — no scripted repair in the diagnostic cohort

## Completed actor-only controls

All five runs completed; 1,997 executed VLA actions pass independent exact replay.
[Detailed interpretation](../../exp_record/agent_notes/vla_failure_audit_results_20261019.md)
· [Metrics and pairing checks](results.json) · [Completed supervisor](completed_launch.json).

| Diagnostic video | Left | Right |
|---|---|---|
| [Book inference](book_decode.mp4) | Native camera, existing TF5 | Native camera, official TF4 |
| [Book camera](book_camera.mp4) | Native camera, official TF4 | Robot-side camera, official TF4 |
| [Fridge inference](fridge_decode.mp4) | Historical R2 actor TF5 | Official TF4 actor |
| [Cheese+butter native control](cheese_native.mp4) | Single actor-only native task | — |

All are **actor controls, with zero interventions**. There is no framework arm.
Common simulation-time playback is 1x at20fps. End-frame holds are labeled and add
no physics/dwell. Native success is not strict stable release: both book native runs
stop before any verified placement dwell. Book side-camera and both fridge actors
never grasp. Camera change includes loss of task visibility. [Contact sheet](contact_sheet.png).

[185-member complete cohort archive](actor_cohort.tar.gz) ·
[SHA256 manifest](actor_cohort_manifest.json). Includes all five original tapes,
images/states/videos, frozen source, logs and independent replay records, plus
postprocessing script and annotated review. The historical R2 comparator is preserved
in the [original R2 archive](../fridge_r2_20261019/README.md).

## Preflight and provenance evidence

* [Protocol](../../exp_record/agent_notes/vla_failure_audit_protocol_20261019.md)
* [Evidence and execution record](../../exp_record/agent_notes/vla_failure_audit_execution_20261019.md)
* [R2 action ownership](r2_ownership.json): open-wrapper382geometric-primitive actions,
  38grip-retention actions,180unmodified VLA actions. First grasp/extraction occur
  under the primitive; this is not autonomous repair evidence.
* [Pinned cached decode](pinned_cached.json) and [pinned uncached decode](pinned_uncached.json):
  all6processor inputs match,5/6actions match the existing stack. One differs,
  including gripper sign. Official cached/uncached outputs agree on all6cases.
* [Pinned book one-action exact replay](preflight_replay.json).
* [Preflight archive](preflight.tar.gz), [139-member manifest](preflight_manifest.json):
  fixed audit images/tensors, all reports/logs, scene and IPC preflights, and both
  failed/stopped relational-transport engineering probes. The transport route is
  not integrated or claimed as intelligent repair.
* [Independent remote recovery proof](preflight_remote_verified.json):139/139members
  downloaded and SHA256-verified from source freeze`1477011`.
* [Detached actor-only launch](launch_snapshot.json): supervisor3664969, GPU1/2/3.

The five trials each had a600-action/query cap; native done stopped three early.
Every executed control-interval action is a model proposal, with no repair or human
motion route. Original preflight reports and launch snapshots remain historical.
