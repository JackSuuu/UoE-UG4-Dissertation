#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-event_gate_coverage_$(date +%Y%m%d_%H%M%S)}"
python -u experiments/collect_event_gate_pilot.py \
  --source_run "$ROOT/src/results/push_torch/sequence_risk_20261012_a" \
  --output "$ROOT/src/results/push_torch/$RUN_ID" "$@"
