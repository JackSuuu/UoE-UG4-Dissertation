#!/usr/bin/env bash
# Offline replay only. Frozen validation data; never search thresholds here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-verifier_audit_$(date +%Y%m%d_%H%M%S)}"
echo "[audit] run=$RUN_ID GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u experiments/audit_verifier.py \
  --source_result "$ROOT/src/results/push_torch/rq2_candidate_validation_20261009_a.json" \
  --tag "$RUN_ID"
echo "[audit] complete $RUN_ID"
