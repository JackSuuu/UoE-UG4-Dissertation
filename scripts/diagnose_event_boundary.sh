#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
: "${RUN_ID:?set a unique RUN_ID}"
python -u experiments/diagnose_event_boundary.py \
  --source_run "$ROOT/src/results/push_torch/sequence_risk_20261012_a" \
  --pilot_run "$ROOT/src/results/push_torch/event_gate_coverage_20261012_a" \
  --output "$ROOT/src/results/push_torch/$RUN_ID" "$@"
python -u experiments/analyze_event_boundary.py --run "$ROOT/src/results/push_torch/$RUN_ID"
