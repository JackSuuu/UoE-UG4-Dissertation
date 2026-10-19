# Autonomous model candidates: negative pilots and offline Q1 diagnostic

All evaluated actions originate from the frozen VLA. Online pilots use no manual
motion routes or future simulator branch selection. Q1 uses simulator branches
**only for offline evaluation**, with both selector choices frozen before outcomes.

## Actual baseline-left comparison videos

| Video | Left | Right | Outcome |
|---|---|---|---|
| [Fridge verification](fridge_nominal_left_verification_right.mp4) | Nominal VLA | Current checks/rechecks | Neither opens; no task completion |
| [Fridge simple feedback](fridge_nominal_left_simple_feedback_right.mp4) | Nominal VLA | Simple feedback | Neither opens; no task completion |
| [Book verification](book_nominal_left_verification_right.mp4) | Nominal VLA | Current checks/rechecks | Only nominal reaches native goal; neither strict completion |
| [Book simple feedback](book_nominal_left_simple_feedback_right.mp4) | Nominal VLA | Simple feedback | Only nominal reaches native goal; neither strict completion |

Each pair shares its initial condition,600-query/action caps and20fps simulation
playback; actual action counts differ because extra samples consume queries.
Synchronous pre-action evidence, candidate source, chosen command, control owner,
repair budget and actual exits/handoffs are annotated. Shorter tapes and terminal
frames have explicit holds, with no added physics/dwell. Simulation pauses during
inference. Fridge handoffs are all unverified cap exits; none demonstrates recovery.

## Offline Q1 video

[All three opportunities, all candidates](q1_all_opportunities_candidates_offline.mp4)
compares candidates0/1/2, not baseline/framework arms. The actual source entries
are368/403/443. Each candidate is followed by11fresh greedy actions and20fresh
greedy continuation actions, from independently reconstructed identical source
states. The20-action tail is not an actual online framework handoff.

All9branches (six distinct state/action pairs) show no useful opening progress at
horizons1/12/32.288branch actions and279fresh queries; maximum opening-progress
margin1.33e-15rad. Useful selection conditional on useful availability is undefined.
This is not proof of equal selectors, universal candidate failure or safe recovery.

## Complete evidence

[SHA256 manifest](manifest.json) covers three complete cohort archives and five
standalone videos. Archives include source/protocol/plan freezes, candidates and
query costs, all state/image/action tapes, gate traces, independent replays, review
videos/stills, logs and failures. Q1 also includes the frozen source used for prefix
reconstruction; the accompanying fridge archive contains the original source tapes.
Weights/runtime installations and disposable `__pycache__` are excluded; upstream
model/code revisions and hashes are retained.

* [Book archive](book_candidates.tar.gz) · [results](../../exp_record/experiment3_candidate_20261019/results.md)
* [Fridge archive](fridge_candidates.tar.gz) · [results](../../exp_record/experiment3_fridge_candidate_20261019/results.md)
* [Q1 archive](fridge_q1.tar.gz) · [results](../../exp_record/experiment3_fridge_q1_20261019/results.md)

17pre-freeze unit tests passed across the three implementations. Independent replay
passes for1604book,1656fridge and288Q1branch actions. Q1 additionally replays exact
original prefixes separately and reports that overhead. No Q2 factorial efficacy
cohort or cloth manipulation experiment was launched by these runs.
