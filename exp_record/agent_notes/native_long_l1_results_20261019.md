# Native long-task baseline screening: all24actor episodes completed

## Outcomes

| Native task | Full successes | Exact replay | Meaningful failed episodes | Frozen eligibility |
|---|---:|---:|---:|---|
| Microwave |4/6|6/6|2|yes; highest priority|
| Two pots |3/6recorded|2/6|3, all replay-flagged|no|
| Two designated mugs |3/6|6/6|3|yes|
| Bowl and drawer |4/6|6/6|2|yes|

All24planned identities completed, with9,984model calls/actions and240separate
initialization noops. No actor episode was repeated. Final untouched two-mug root3
succeeded after257actions; drawer root3failed at520. Both passed fresh exact replay.
The original four two-pot wrist-image mismatches remain failed strict checks;
the combined supervisor consequently ends `complete_with_issues`. The report's
global `selection.provisional` remains true because these integrity issues remain.
Microwave's own six cases meet the unchanged per-task admission rule, independently
of the lower-priority two-pot issues. No selection rule was relaxed.

These are reused development roots0–5, not held-out efficacy data. Native full
success is distinct from released/stationary/dwell completion. No framework,
repair, selection, or handoff intervention ran in this baseline cohort.

## Evidence chain

* Original six episodes: `native_long_l1_20261019_a`; published archive
  `backups/native_long_l1_partial_20261019`,4,128members independently restored.
* Sixteen more episodes: `native_long_l1_20261019_b`; published archive
  `backups/native_long_l1_continuation_20261019`,9,547members independently restored.
* Final two episodes and combined report: `native_long_l1_20261019_c`;
  `report/summary.json` and `report/results.md` retain all24identities and original
  paths. Final-cell archive publication/recovery is tracked separately.

## Next experiment

The [frozen L2A protocol](native_microwave_l2_protocol_20261019.md) tests three
closed-loop model branches at each of six event-aligned microwave states, with
a576-call cap. Original nominal controls must reproduce before sampling starts.
Root0's actual post-action253boundary reconstructed exactly over266frames with
zero model queries on physicalGPU2/EGL1. Implementation and evidence are frozen
at `a22ccf6`; model-backed branch launch still awaits supervisor/audit validation.

Offline labels measure candidate availability. A useful short branch is not a
verified repair or downstream task recovery. The predefined all-bank L2B audit
is required before claiming reachability beyond local progress; L3 remains unrun.
