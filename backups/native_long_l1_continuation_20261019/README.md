# L1 continuation: sixteen additional native episodes

Eight locally verified archives preserve9,547files (1,982,355,051compressed bytes)
from `/home/ami/scratch/native_long_l1_20261019_b`. They contain16new actor episodes,
their original replay attempts, frozen continuation manifests/source, raw videos,
camera/state/action tapes and combined24-cell reporting. The six prior episodes
are referenced rather than duplicated; recover them separately from
`../native_long_l1_partial_20261019/`.

Combined coverage after this continuation is22/24:

| Task | Native successes / completed | Exact replay | Remaining |
|---|---:|---:|---|
| Microwave |4/6|6/6|none|
| Two pots |3/6|2/6;4wrist-only mismatches retained|none|
| Two mugs |2/5|5/5|root3|
| Drawer |4/5|5/5|root3|

Two-pot root3failed strict replay at wrist frames252/254/258/277 with zero
physics/integration gap; its lane correctly stopped before the last two cells.
All other new episodes passed. Prior wrist failures remain in the combined report.
The highest-priority microwave task meets the frozen per-task L2admission rule:
at least2successes and2meaningful partial-progress failures across6integrity-checked
episodes. This is baseline development evidence, not a framework improvement.

The final two untouched cells have a separate completion run. Source and result
archives here remain the stopped historical continuation; no files are overwritten
by subsequent runs. Remote recovery is tracked separately after publication.
