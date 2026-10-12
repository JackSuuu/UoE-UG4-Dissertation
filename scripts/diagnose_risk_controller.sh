#!/usr/bin/env bash
# Privileged component diagnosis. Launch with setsid nohup, GPU UUID and stdin /dev/null.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-risk_controller_diagnosis_$(date +%Y%m%d_%H%M%S)}"
python -u experiments/diagnose_risk_controller.py \
  --run "$ROOT/src/results/push_torch/sequence_risk_20261012_a" \
  --output "$ROOT/src/results/push_torch/$RUN_ID" \
  --models frozen static_s0 sequence_s0 "$@"
