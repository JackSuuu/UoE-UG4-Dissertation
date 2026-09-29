#!/usr/bin/env bash
# Full Phase-1 pipeline for one task/backend.
#   bash run_all.sh [push|cloth] [torch|genesis] [extra args, e.g. --quick]
# Order follows the plan: audit (W1) -> data/train (W2) -> calibrate -> RQ1 -> RQ2 -> RQ3 -> figures
# Split into two stages:
#   build  — collect data, train predictors, calibrate tau (produces the verifier)
#   eval   — RQ1, RQ2, RQ3, figures (loads the trained verifier)
# Usage:
#   bash run_all.sh build push torch          # build the verifier
#   bash run_all.sh eval  push torch          # evaluate it
#   bash run_all.sh       push torch          # both (default)
set -euo pipefail
cd "$(dirname "$0")"
# Single A5000 only: simulates an edge-device compute budget. Override with CUDA_VISIBLE_DEVICES=...
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Optional first arg: build | eval | all (default)
STAGE=all
if [ "$1" = "build" ] || [ "$1" = "eval" ]; then
  STAGE=$1; shift
fi
TASK=${1:-push}; BACKEND=${2:-torch}; shift $(( $# > 2 ? 2 : $# )) || true
# Separate --chunk_k (execution protocol, eval scripts only) from other args
CHUNK_K=5
REMAINDER=()
while [ $# -gt 0 ]; do
  case "$1" in
    --chunk_k) CHUNK_K=$2; shift 2 ;;
    *) REMAINDER+=("$1"); shift ;;
  esac
done
A="--task $TASK --backend $BACKEND ${REMAINDER[*]}"
GP=""; [ "$BACKEND" = "genesis" ] && GP="--genesis_probe"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "build" ]; then
  python experiments/collect_data.py    $A
  python experiments/train.py           $A --what all
  python experiments/calibrate.py       $A --chunk_k $CHUNK_K
fi
if [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
  python experiments/audit_gradients.py $A $GP
  python experiments/rq3_systems.py     $A --part mem      # checkpointing early (plan: W1-2)
  python experiments/rq1_calibration.py $A --chunk_k $CHUNK_K
  python experiments/rq2_eval.py        $A --with_noact --chunk_k $CHUNK_K
  python experiments/rq3_systems.py     $A --part stab --chunk_k $CHUNK_K
  python experiments/rq3_systems.py     $A --part sched --chunk_k $CHUNK_K
  python experiments/make_figures.py    $A
fi
echo "done -> results/${TASK}_${BACKEND}/"
