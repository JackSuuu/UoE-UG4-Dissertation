#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-sequence_risk_$(date +%Y%m%d_%H%M%S)}"
echo "[sequence-risk] run=$RUN_ID GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u experiments/validate_sequence_risk.py \
  --source_result "$ROOT/src/results/push_torch/rq2_candidate_validation_20261009_a.json" \
  --audit_summary "$ROOT/src/results/push_torch/audit_verifier_audit_20261012_a/summary.json" \
  --output "$ROOT/src/results/push_torch/$RUN_ID"
echo "[sequence-risk] complete $RUN_ID"
