# Experiments (single home)

All active launch/check/collection/training CLI/summary/test code lives here.
The generic framework stays in neighbouring modules; experiments call the same
`Session`, never copy a controller. `scripts/` retains only archive utilities.
Historical launchers are removed, not redirected to an incompatible new API.

## Entry points — run from the repository root

| Command | Purpose |
|---|---|
| `python -m src.experiment.run_task --help` | Shared arguments and preflights |
| `python -m src.experiment.microwave --help` | LIBERO-10 task9/order0, four arms |
| `python -m src.experiment.dataset --help` | Executed-action labels from training episodes |
| `python -m src.experiment.build_memory --help` | Successful measured training keyframes |
| `python -m src.experiment.train --help` | Fit risk, recovery A or handoff B |
| `python -m src.experiment.summarize --help` | Paired outcomes; missing/failed runs stay unpaired |
| `python -B -m unittest discover -s src/experiment/tests` | CPU tests; extra torch test if installed |

## Microwave design

Start with [Week 5](../../exp_record/week5.md) and [GPU setup](../../docs/gpu_setup.md).
They define shell arrays `common` and `visual`, plus `RUN_ROOT`. No per-task goal
config/adaptor is required. The native task is placement **and close**; omit
`--instruction` for the benchmark or pass a natural-language instruction explicitly.

| Arm | Enabled | Purpose |
|---|---|---|
| baseline | Original-instruction frozen VLA, recovery disabled | Complete-task control |
| measured | Generated contracts and measured bounded repair | Verify/repair reference |
| context | Pretrained vision, generated-predicate subgoals, retrieval and causal forecasts enter recovery planning | Integration/data collection, not the full learned system |
| full | Context + trained risk gate + trained A/B + nonempty training memory | Full architecture, after artifact validation |

Run scene → planner → policy → components preflights first. Inspect actual images,
grounding, measured unknowns and encoder provenance. Then root0 pilot, matched
development roots0–5 and training collection. Empty retrieval is explicit, not
claimed useful memory. Full refuses missing/incompatible artifacts. The shared
runner supports ablations by omitting components or using `--monitor-mode log`;
label these as ablations, not `full`.

### Training candidate siblings (roots10–19 only)

First run just root10/slot0 and audit it. Once it passes, collect four siblings
per root, identically seeded/reset before the first repair. Each queries one
greedy + three sampled unchanged H1 proposals and executes only its fixed slot.
This ranker is a data-collection instrument, **not a learned Module A**. The
query budget remains520, including all candidates. Only the executed slot gets
a recovery label; no counterfactual outcome is invented for other proposals.

```bash
for root in {10..19}; do
  for slot in {0..3}; do
    python -B -m src.experiment.microwave --arm context \
      --init-index "$root" --seed "$root" --candidate-slot "$slot" \
      "${common[@]}" "${visual[@]}" \
      --out "$RUN_ROOT/train/root${root}_slot${slot}" || exit 1
  done
done
```

No repair trigger means no A/B example, not a fabricated negative. Identical
candidates do not establish ranking diversity. Count contexts and roots, not
40 siblings as40 independent scenes. Unknown/repair-censored continuations and
infrastructure errors are excluded, not labelled failure.

```bash
for role in risk recovery handoff; do
  python -B -m src.experiment.dataset --role "$role" \
    --episodes "$RUN_ROOT"/train/root*_slot* --out "$RUN_ROOT/$role.jsonl"
  python -B -m src.experiment.train --role "$role" \
    --dataset "$RUN_ROOT/$role.jsonl" --out "$RUN_ROOT/models/$role" --device cpu
done
python -B -m src.experiment.build_memory --episodes "$RUN_ROOT"/train/root*_slot* \
  --out "$RUN_ROOT/memory.json"
```

Training defaults to10–17, validation18–19. Both splits need both classes;
training stops if data are insufficient. Review labels, missingness, validation
TPR/FPR and candidate diversity before promotion. Risk predicts observed nominal24
progress loss, **not** proven intervention benefit. B predicts observed nominal24
contract survival under the exact executed nominal action, **not** repair-entry
eventual success. A predicts observed measured recovery for the executed first
proposal; later recovery policy is part of that outcome. None is a world model
or a causal guarantee. These models pool generated predicate grammar/truth; they
do not encode unrestricted text or prove language entailment.

Measured generated-task completion is absorbing: a completed observed nominal
endpoint is positive for B and negative for regression risk even if a temporary
local contract no longer holds. Completion during executed repair is positive
for A. This does not uncensor unexecuted actions, repair-owned nominal futures or
missing/nonconsecutive endpoints.

### Full development arm — after training/validation gates

```bash
python -B -m src.experiment.microwave --arm full --init-index 0 --seed 0 \
  "${common[@]}" "${visual[@]}" --memory-bank "$RUN_ROOT/memory.json" \
  --risk-checkpoint "$RUN_ROOT/models/risk/model.pt" \
  --recovery-checkpoint "$RUN_ROOT/models/recovery/model.pt" \
  --handoff-checkpoint "$RUN_ROOT/models/handoff/model.pt" \
  --out "$RUN_ROOT/dev/root0/full"
python -B -m src.experiment.summarize --root "$RUN_ROOT/dev" --treatment full
```

Keep roots6–9 for calibration and20–49 for confirmation; do not repeatedly tune
on them. Native full-task success is primary for the native instruction, with
generated completion separate. Pairing checks source/checkpoints/instruction,
generated goals/milestones, initial-state index/seed and limits. Semantic
agreement with the instruction still needs inspection. There is no independent
simulator replay yet: summaries are not replay-audited evidence.

Outputs: fresh directories with status/result JSON, events JSONL, exact PNGs,
selected state, BDDL and source snapshots. Monitor events retain visual/fusion
features, retrieved examples, forecasts, risk decisions and provenance. Failed
preflights are infrastructure failures, not task failures.
