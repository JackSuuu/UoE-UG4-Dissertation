#!/usr/bin/env bash
# Diagnostic hindsight-selected validation branches, not a deployment skip rule.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-intervention_branch_$(date +%Y%m%d_%H%M%S)}"
echo "[branch] run=$RUN_ID GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u experiments/audit_intervention_branch.py \
  --audit_summary "$ROOT/src/results/push_torch/audit_verifier_audit_20261012_a/summary.json" \
  --tag "$RUN_ID"
echo "[branch] complete $RUN_ID"
