# Native microwave: framework vs bare VLA

Five-layer deterministic framework with privileged measured verification; no learned predictor. Native In & Close only. Frozen official greedy H1 VLA, batch one; scoped language during bounded repair; no numeric action edits.

| Root | Arm | Valid | Native success observed | Actions / queries | Exit | Replay |
|---|---|---|---|---|---|---|
| 0 | baseline | True | False | 520 / 520 | action_budget | exact_replay_passed |
| 0 | framework | True | False | 520 / 520 | action_budget | exact_replay_passed |
| 1 | baseline | True | True | 444 / 444 | native_success | exact_replay_passed |
| 1 | framework | True | True | 444 / 444 | native_success | exact_replay_passed |
| 2 | baseline | True | True | 397 / 397 | native_success | exact_replay_passed |
| 2 | framework | True | True | 397 / 397 | native_success | exact_replay_passed |
| 3 | baseline | True | True | 352 / 352 | native_success | exact_replay_passed |
| 3 | framework | True | True | 352 / 352 | native_success | exact_replay_passed |
| 4 | baseline | True | False | 520 / 520 | action_budget | exact_replay_passed |
| 4 | framework | True | False | 520 / 520 | action_budget | exact_replay_passed |
| 5 | baseline | True | True | 298 / 298 | native_success | exact_replay_passed |
| 5 | framework | True | True | 298 / 298 | native_success | exact_replay_passed |

Valid-pair rescues: 0; harms: 0.

Invalid/incomplete/replay-mismatched cells are retained, never counted as native failures.

Full SR, trigger evidence, measured verification, actual handoffs, post-handoff success and all timing are in `report.json`.
