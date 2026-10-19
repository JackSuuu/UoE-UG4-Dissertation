# L2A frozen execution preparation

Four-member archive preserves the18-cell plan and three executable sources before
launch. Plan SHA256:
`9162f78ff15a71d5fc04ba4fa6e7c87fd4aa548a5f7a383bd556a62efabdfccb`.
Preparation: `/home/ami/scratch/native_microwave_l2a_preparation_20261019_v1`.

PhysicalGPU2/EGL1andGPU3/EGL0are UUID-masked; at most two simultaneous batch-one
workers. C0roots0–5must pass original-suffix identity plus fresh strict replay
before C1, then C2. All18cells retained; no retries or replacement candidates.
576calls maximum,32local actions per cell, no immediate continuation. Per-model
readiness180s/query30s; branch/replay120s; whole phase7200s. Detached supervisor
tracks/kills descendant groups and retains interrupted work.

The plan pins native evidence, protocol, observation/prompt hashes, source/runtime
files and checkpoint provenance. Root0's selected boundary smoke passed with zero
queries. Preparation is not execution; actual launch/results are logged separately.

All four preparation members independently recovered and hash-verified. Proof:
`../native_long_l1_final_cells_20261019/remote_recovery_passed.json`.
