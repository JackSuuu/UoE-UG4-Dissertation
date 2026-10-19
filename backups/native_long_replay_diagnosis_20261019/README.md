# Wrist-rendering diagnosis and continuation validation evidence

Archive contains1,862hashed files:11fresh zero-query replay investigations,
lossless pixel evidence and fixed-sensor-time redraws, original tape consistency
checks, offline partial-failure analysis, continuation synthetic checks, combined
report preflight and original-report regression. Local archive verification passed.

Rendering nondeterminism is demonstrated at unchanged sensor-time state. Captured
differences affect1–5wrist pixels with channel delta±1; physical/controller/contact
and agentview comparisons remain exact over5,752compared frames. The original
strict replay failures remain failures. Original historical replay RGB was not
saved; numerical differences refer to fresh diagnostic runs. Root1/2runs onGPU1
are explicitly cross-device. No policy query, policy training or repair action.

[Reviewed decision](../../exp_record/agent_notes/native_long_l1_continuation_20261019.md).
The18untouched baseline cells run separately; this archive makes no claim of their
outcomes or framework improvement. Independent remote recovery is recorded later.
Remote-only recovery now passed for all1,862members; proof:
`../native_long_l1_review_20261019/remote_recovery_passed.json`.
