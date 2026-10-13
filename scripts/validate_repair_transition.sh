#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
: "${RUN_ID:?set a unique shared RUN_ID}"
python -u experiments/validate_repair_transition.py \
  --source_run "$ROOT/src/results/push_torch/sequence_risk_20261012_a" \
  --factorial_run "$ROOT/src/results/push_torch/risk_controller_diagnosis_20261012_a" \
  --output "$ROOT/src/results/push_torch/$RUN_ID" "$@"
