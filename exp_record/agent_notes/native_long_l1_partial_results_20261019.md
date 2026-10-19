# Native long-task L1: partial results and diagnostic hypotheses (2026-10-19)

## Scope and integrity

Read-only analysis of saved cohort `/home/ami/scratch/native_long_l1_20261019_a`: **6/24 episodes completed; 18 unstarted**. No simulation, model queries, interventions, or task selection were performed. These are development roots, not held-out results.

| Task | Root 0 | Root 1 | Root 2 | Coverage / integrity |
|---|---|---|---|---|
| Microwave | Failure, 520 actions | Native success, 444 | Native success, 397 | 3/6; all three exact replays passed |
| Two pots | Native success, 500 | Failure, 520 | Failure, 520 | 3/6; exact physics/integration, wrist RGB mismatches unresolved |
| Two mugs; drawer | Unstarted | Unstarted | Unstarted | 0/6 each |

The two-pot replay reports identify **1/1/3 distinct mismatching wrist-image frames**, respectively: frame 482; frame 155; frames 237, 261, 275. The corresponding before/after fingerprint mismatches refer to those same images, not additional independent frames. All six replays have maximum state and integration gaps of zero and zero actor queries. RGB differences remain an integrity issue; no rendering cause is inferred here.

**No task qualifies yet.** The frozen rule requires six completed, consistent, replay-verified cases per task, including at least two native successes and two meaningful partial-progress failures. Microwave currently has only one observed failure; two pots only one observed success. The 18 unstarted cells are not failures.

## Timing and measurement conventions

Below, `t` is a zero-based **post-action** step; its saved frame/state index is `t+1`. True intervals include both endpoints. Each action advances approximately 0.05 simulated seconds; inference wall time is separate. Position differences use logged world-coordinate EEF/object positions, in millimetres. A “20 mm rise” means object or EEF z exceeds its value at the stated reference step by 20 mm; it is an exploratory motion descriptor, not a certified lift/stability predicate. Contact counts are endpoint observations, not continuous-contact duration.

Sources: each episode's `steps.jsonl`, `status.json`, `frames.jsonl`, and each replay's `verification.json`; the cohort report and frozen protocol supply coverage/selection context. Small reproducible extraction: `/home/ami/scratch/native_long_l1_partial_analysis_20261019/summarize.py`, with `evidence.json` beside it. It imports only JSON/path utilities and NumPy and reads saved logs.

## Observed predicate and grasp histories

### Microwave

| Root | Mug grasp true | Mug `In` true | Microwave `Close` true |
|---|---|---|---|
| 0 | 73–519 | Never | Never |
| 1 | 86–290 | 277–443 | 443 |
| 2 | 95–114, 118, 120–145, 147–160, 167–201, 232–301 | 296–396 | 396 |

- **Root 0 has contact-defined progress, not an established lift.** Grasp remains true for 447 observations, yet maximum mug rise from first grasp is only **5.12 mm** through the cap; table contact appears in **344/447** of those endpoints, including the last. Over the first 60 subsequent actions (`73→133`), mug displacement is `(13.75, 14.43, −1.34)` mm and EEF displacement `(39.27, 13.67, −3.44)` mm. By `t=519`, they are `(64.62, 20.97, 3.98)` and `(73.81, 2.47, −14.24)` mm relative to first grasp. This is small-height lateral motion followed by low-motion persistence, not “grasp then drop.” Both goal predicates stay false.
- **Successful controls initially look stalled too.** At first grasp +60, root 1's mug has moved only `(+0.46, −2.45, +1.48)` mm and EEF `(+2.80, −1.24, +2.02)` mm; table contact is present at 60/61 endpoints. Root 2's mug is `(+7.38, +4.26, −1.48)` mm and EEF `(+14.26, −3.72, −3.14)` mm, with table contact at 61/61 endpoints. Both later succeed, despite root 2's multiple grasp losses.
- The first **20 mm z rise of both mug and EEF** occurs at `t=244` in root 1 and `t=251` in root 2: **158/156 actions after first grasp** (7.9/7.8 simulated seconds). Root 0 never reaches this descriptor. By grasp +180 (`t=253/266/275`), maximum mug rise is **1.92/60.79/84.30 mm** for roots 0/1/2. This delayed separation is visible before native placement success; it cannot be assumed at grasp +40 or +60.
- Root 0's late stagnation has action evidence: every mapped z command at `t=260–519` is approximately `+0.000341`, while both successful roots eventually issue substantial positive and negative z commands. Root 0 EEF coordinate ranges over `t=320–380` are only `(1.34, 3.13, 0.51)` mm. This supports a near-zero-z-command/low-motion persistence hypothesis for that interval, **not** a conclusion that an attempted strong lift was physically blocked, nor an explanation of why the actor chose those commands.

### Two pots

The stove-on predicate is already true initially and remains true; it contributes no newly achieved manipulation progress. Both pots are initially off the goal region. All three observed episodes happen to manipulate pot 2 first; that is an observation, not a required task order.

| Root | Pot 2 grasp true | Pot 2 `On` true | Pot 1 grasp true / `On` true |
|---|---|---|---|
| 0 | 159–248, 250–314 | 317–499 | 429–498 / 499 |
| 1 | 124–183 | 175–178, 180, 185–519 | Never / never |
| 2 | 127–299, 309–311 | 203, 205–269, 271–298, 301–519 | Never / never |

- Pot 2's first 20 mm rise after first grasp occurs at **260/129/154** for roots 0/1/2, and the EEF crosses the same threshold at those steps. Thus even the successful two-pot control waits **101 actions** after first grasp before that rise; the failures lift their first pot sooner (**5/27 actions**). Root 0's pot 2 maximum positional excursion during its first 60 post-grasp actions is only **1.51 mm**. First-grasp latency alone does not rank eventual outcomes.
- Pot 2 placement flicker in roots 1/2 resolves in the recorded continuation. Their final uninterrupted `On` runs start at **185/301**, leaving **334/218 subsequent actions** without completing pot 1. Grasp loss and transient `On` loss are useful snapshot events, but neither establishes the eventual failure mechanism: the already placed pot remains on the stove at the cap.
- **The terminal unfinished component is pot 1 acquisition/transfer.** First logged gripper contact with pot 1 occurs at `t=407/270/406` in roots 0/1/2. The successful control shows a 20 mm EEF rise by `420`, object rise by `421`, and only later a true grasp at `429`. Thus the logged grasp predicate can also **lag physical lifting motion**.

| Pot 1 contact-aligned window | Root 0, eventual success | Root 1, failure | Root 2, failure |
|---|---:|---:|---:|
| Window end, contact +40 | 447 | 310 | 446 |
| Maximum object z rise in that window | 124.64 mm | 4.89 mm | 4.79 mm |
| Any logged grasp by window end | Yes | No | No |
| Window end, contact +60 | 467 | 330 | 466 |
| Maximum object z rise in that window | 149.17 mm | 4.89 mm | 12.19 mm |

At contact +60, root 1's pot 1 displacement is `(3.45, 1.12, 0.74)` mm versus EEF `(31.75, 27.81, 2.75)` mm; root 2's is `(1.66, 20.81, 0.81)` versus EEF `(46.65, 56.70, 9.32)` mm. Both have contact/motion without successful acquisition, rather than a demonstrated lift-then-drop of pot 1. Root 1 later has a low-motion tail (EEF ranges only `(4.36, 6.41, 9.14)` mm over `400–519`); root 2 still moves appreciably (ranges `(53.96, 110.49, 26.21)` mm). Calling both tails identical “stagnation” would hide this difference. These traces support inspecting contact geometry and commanded versus realised motion; they do not identify a unique perception, planning, or contact-mechanics cause.

## Native endpoint versus stronger physical completion

Keep the three native successes as recorded. They are not independently certified stable, fully released placements:

- Microwave successes end with grasp false and mug–microwave contact, but no post-success settling window is recorded. This supports containment/contact evidence, not sustained stability certification.
- Two-pot root 0 first attains the full conjunction at its last action, `t=499`. Pot 1's grasp flag is false, yet **finger/finger-pad contact remains** alongside stove-burner contact; its logged translational speed is about **0.052 m/s** and angular speed about **1.55 rad/s**. False grasp therefore does not establish complete release, and the last sample cannot establish stable support. Pot 2 is already settled on the burner.
- Two-pot failures end with pot 2 on the burner and pot 1 still contacting the table; microwave root 0 ends with mug–table and mug–gripper contact. These physical observations complement, rather than replace, the native predicate outcomes.

## Bounded L2 snapshot/trigger hypotheses — proposals only

No branch was executed. These post-hoc development hypotheses require freezing and validation after full six-case integrity review; they neither select a task nor establish repair gains.

1. **Early grasp/contact diagnostic window:** retain the causal pre-action state and event post-state, then assess the next **40 actions**, with one optional extension to **60**. Log mapped actions, EEF/object displacement and relative motion, endpoint/substep contacts, and predicate/grasp changes. For first microwave grasp, proposed +40 snapshots are `t=113/126/135` (state indices `114/127/136`); +60 are `133/146/155`. This window can expose unsupported “grasp implies lift” assumptions, but it is a **diagnostic collection trigger**, not a failure alarm: all three microwave cases lack material vertical progress there. Successful delayed-lift controls must be retained.
2. **Microwave persistence follow-up:** test a separately bounded **180-action observation horizon from first grasp**, with motion checks at +40/+60/+180. A candidate concern is low object rise plus persistent table contact and absent positive-z command activity, using only the prefix available then. The +180 snapshots `253/266/275` separate the observed low-rise case from the two controls while leaving 266 actions in the failed trace. The 180 horizon and 20 mm descriptor are motivated by these development traces, not validated thresholds; neither may be presented as a universally early or reliable failure detector.
3. **Unfinished-object acquisition window:** after one object has a placement predicate true for a proposed 20 consecutive observations, anchor on first physical gripper contact with any still-unplaced target, without imposing object identity/order. Compare the following **40 actions**, optionally extending to 60, for absent grasp **and** absent material object rise/transfer. Applied retrospectively to pot 1, +40 snapshots are `447/310/446` (state indices `448/311/447`). The failed traces still have **209/73** actions remaining, and the successful control has already lifted. At +60 the failures have 189/53 remaining. This is a plausible local acquisition diagnostic; the short remaining budget in root 2 is a practical constraint, not evidence of recoverability.
4. **Placement-loss snapshots with continuation controls:** retain before/after states at pot 2 `On` losses (`179,181` in root 1; `204,270,299` in root 2), but include the subsequent naturally restored predicate. Microwave root 2's grasp-loss/reacquisition history supplies another successful control. Triggering an immediate repair merely because a predicate or grasp toggles would conflate recoverable transients with unresolved failure.

Snapshot timestamps must be chosen from causal prefixes. The eventual outcome labels here are for retrospective comparison only. The main actionable distinction for later diagnostics is **contact-defined grasp versus actual object motion, followed by persistence or acquisition failure**, with successful slow and flickering controls retained.
