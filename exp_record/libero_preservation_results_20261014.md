# Verified preservation diagnostic results — 14 Oct 2026

Run: `/home/ami/scratch/libero_preservation_20261014_b`. All24episodes completed; worker exits0/0/0. The corrected renderer was committed/pushed as `ec9ff87` before restart. Frozen protocol: [placement preservation](libero_preservation_protocol_20261014.md). Architecture interpretation: [locked five-layer guardrail](libero_framework_design_20261014.md).

## Outcomes

| Arm | Native-ever | Released placement (original20-interval dwell) | Completed and final20intervals valid after40actor-only steps | Actual changed actions |
|---|---:|---:|---:|---:|
| VLA | 6/6 | 2/6 | 1/6 | 0 |
| Hold-open | 6/6 | 3/6 | 2/6 | 58 |
| Retreat-open | 6/6 | 2/6 | 1/6 | 58 |
| Accurate-physics selector | 6/6 | 3/6 | 1/6 | 13 |

The third outcome uses the full six-root denominator; episodes without acquisition do not count as completed-and-retained. Their raw post-completion retention field remains null because no post-completion test occurred. Acquisition and resumption are distinct outcomes. No partial40-step tail occurred among completed episodes. Native-ever is not a substitute for released placement; no official-native-SR gain occurred. Collision/CVR/safe-success labels remain unvalidated/null.

| Root | VLA acquisition / retained | Hold acquisition / retained | Retreat acquisition / retained | Physics acquisition / retained |
|---|---|---|---|---|
| 0 | yes / yes | yes / yes | no / n.a. | yes / yes |
| 1 | no / n.a. | no / n.a. | no / n.a. | no / n.a. |
| 2 | no / n.a. | yes / no | yes / no | yes / no |
| 3 | no / n.a. | no / n.a. | no / n.a. | no / n.a. |
| 4 | no / n.a. | no / n.a. | no / n.a. | no / n.a. |
| 5 | yes / no | yes / yes | yes / yes | yes / no |

Roots0–2are the original consumed development cases;3–5are the preregistered additional development cohort. Report them separately: acquisition on0–2is1/3,2/3,1/3,2/3and on3–5is1/3for all four arms. No held-out OOD/final-test or broad-generalization claim is supported.

## Attribution

- Physics versus VLA: one acquisition rescue (root2), zero acquisition harms.
- Physics versus hold: zero acquisition rescues and harms; identical3/6acquisition. For completed-and-retained outcome, physics loses root5relative to hold.
- Physics versus retreat: one acquisition rescue (root0), zero acquisition harms. This avoids a harmful fixed retreat, but does not outperform the stronger hold comparator.
- Physics changes only13actions, all on root2, versus58for each fixed control. This is a sparsity/selection observation; it does not establish better complete-task outcomes or efficiency after accounting for expensive oracle replay/forecast queries.
- Roots1/3/4never reach the valid-placement trigger and receive zero corrections. This exposes the current late trigger's coverage limit; their exact earlier failure causes still require phase-specific audit rather than being universally labeled "jamming".
- Short pulse-and-settle forecasts do not establish compatibility with future **actor** commands. Root2still loses placement after actor resumption, and physics performs no intervention on root5where the simple controls improve retention.

**Conclusion:** the narrow accurate-physics preservation selector improves acquisition over VLA on one development case but provides **no acquisition advantage over hold-open and worse completed-and-retained counts**. Do not promote it as successful learned/physical-prediction efficacy. This is evidence for improving layers3–5inside the user-locked architecture: earlier phase-conditioned event coverage and a persistent, completion-aware gate/repair handoff. The five-layer architecture remains fixed.

Six development roots are not a powered significance study. The verified summary records paired rescue/harm counts and descriptive Wilson95% intervals:1/6rescue spans approximately3.0–56.4%;0/6harm has upper bound39.0%. These are intervals for root-level proportions, not a positive paired efficacy bound or a formal safety claim.

## Verification, videos and recovery

Independent aggregation recomputed every acceptance component from geometric/contact logs, dwell, native-action mapping, exact executed candidate selection, forecast ranking and first-step forecast/actual equality, common-prefix states/actions/tokens across all arms and historical baselines,20-opportunity/600forecast-step budgets, actor-only tail, and states/video lengths. It recorded216raw artifact hashes. All checks passed.

The supervisor initially marked failure **after all workers succeeded** because `list(imageio_reader)` attempted an excessive allocation from the FFmpeg reader's length hint. A frame comprehension fixed only video aggregation; the independent verifier was rerun successfully on the same untouched24episodes. Original `launch.json` and failure log are retained; `postprocess_recovery.json` records the successful recovery rather than rewriting that provenance.

Outputs: `verified_summary.json`, `index.html`, six`review/root_<i>_methods_2x.mp4`files. Each video shows the four methods on the same root at2x simulated time; ended episodes freeze with an explicit label, not extra simulated dwell. All source frames are retained. This comparison is privileged H1/k1, not genuineH10, not a learned verifier and not the complete five-layer architecture.

Recovery archive and standalone comparison videos: `backups/libero_preservation_20261014_b/`; use the existing SHA256 verification/restore script. Whole-task long-horizon implementation follows the [fixed layer-by-layer plan](libero_five_layer_implementation_20261014.md), with the next bowl/drawer task, native-H1/chunk compatibility and collision definitions explicitly audited before new efficacy claims.

**Remote recovery verified:** `037f73c`pushed all seven new LFS objects (93,764,850bytes). Independently fetched from GitHub, verified all six standalone videos and fully restored/hash-checked all232archived files. [Verification record](../backups/libero_preservation_b_remote_verified_20261014.json).
