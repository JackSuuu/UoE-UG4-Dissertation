# Shared actor qualification and refrigerator transfer

Five actor-only runs,3000model actions; all pass independent exact replay and
artifact checks. No framework intervention or training occurred.

* [Fridge video](fridge90_actor_only_annotated.mp4): closed-start left, open-start
  right, both LIBERO90 actor only. Neither opens/grasp-completes its tested skill.
  The right panel's open door is initialization, not learned opening.
* [Book video](actor_only_qualification_verified.mp4): three LIBERO10 initial
  states; native success3/3, additional strict completion0/3 despite continuation.
* [Fridge contact sheet](fridge90_endpoint_contact_sheet.png).
* [Book contact sheet](qualification_endpoint_contact_sheet.png).

Videos use20fps simulation-time playback,601actual state frames and40additional
labelled final holds; physics pauses during model inference. These are capability
controls, not baseline/framework comparison videos or real-time control results.

[Complete artifact archive](actor_qualification.tar.gz) ·
[Per-file SHA256 manifest](manifest.json).
Includes both cohorts' full state/image/action/contact tapes, source freezes,
independent replays, original and corrected review artifacts, logs, local and
remote dataset/checkpoint audits, plus checkpoint download/hash provenance.
Upstream model weights and disposable runtime installations are excluded; pinned
repositories/revisions and weight hashes are retained.

[Independent remote recovery proof](remote_verified.json): fresh GitHub/LFS
download from artifact commit`3de144b`,204/204archive members extracted and SHA256
verified, plus both standalone videos. An initial120-second download limit was
followed by a successful resumed LFS pull; no experimental rollout was retried.

Interpretation and provenance:
[Fridge result](../../exp_record/agent_notes/shared_actor_fridge90_results_20261019.md)
· [Book result](../../exp_record/agent_notes/shared_actor_qualification_results_20261019.md)
· [Remote-resource audit](../../exp_record/agent_notes/shared_actor_fridge_pretrained_audit_20261019.md).

The new actor uses its own `libero_90_no_noops` normalization. Its original
robot-side camera and fixed instructions differ from the separate framed-camera,
phase-prompt candidate pilot; this is not a paired checkpoint-only comparison.
No in-domain positive-control rollout was performed for the third-party LIBERO90
checkpoint. No shared full-fridge actor is qualified by these results.
