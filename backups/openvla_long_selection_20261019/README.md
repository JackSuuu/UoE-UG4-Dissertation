# Native OpenVLA / LIBERO-Long task-source audit

Read-only research, **zero model queries and robot actions**. This is not an
experiment result or evidence of framework benefit.

* [Task inventory and hashes](inventory.json): all ten native LIBERO10 tasks,
  exact initial/goal expressions, runtime BDDL equality and official initial-file
  hashes; four proposed complex-task candidates.
* [Pinned checkpoint card](model_card.md): the existing checkpoint is explicitly
  trained on LIBERO-10 / LIBERO-Long.
* [Official evaluator snapshot](official_eval.py.txt): fetched public source,
  not executed;520model actions after10initial settling steps for LIBERO10.
* [Paper snapshot](paper.html): OpenVLA arXiv2406.09246v3, Appendix E/Table12,
 53.7%±1.3%standard error on the whole long-horizon suite.
* [File integrity manifest](source_manifest.json).

The official evaluator URL is a moving `main` reference; the exact retrieved bytes
are retained and hashed. The card is pinned to the existing checkpoint revision,
and the paper URL is versioned. Published suite performance does not establish
local task-specific failure rates.

[Task selection and proposed screening plan](../../exp_record/agent_notes/openvla_native_long_selection_20261019.md).
No new cohort was started by this audit. RoboCasa/alternate-policy setup has been
cancelled and removed, with a separate deletion record.
