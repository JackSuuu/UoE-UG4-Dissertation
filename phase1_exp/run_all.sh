#!/usr/bin/env bash
# Full Phase-1 pipeline for one task/backend.
#   bash run_all.sh [push|cloth] [torch|genesis] [extra args, e.g. --quick]
# Order follows the plan: audit (W1) -> data/train (W2) -> calibrate -> RQ1 -> RQ2 -> RQ3 -> figures
set -euo pipefail
cd "$(dirname "$0")"
TASK=${1:-push}; BACKEND=${2:-torch}; shift $(( $# > 2 ? 2 : $# )) || true
A="--task $TASK --backend $BACKEND $*"
GP=""; [ "$BACKEND" = "genesis" ] && GP="--genesis_probe"

python experiments/audit_gradients.py $A $GP
python experiments/rq3_systems.py     $A --part mem      # checkpointing early (plan: W1-2)
python experiments/collect_data.py    $A
python experiments/train.py           $A --what all
python experiments/calibrate.py       $A
python experiments/rq1_calibration.py $A
python experiments/rq2_eval.py        $A --with_noact
python experiments/rq3_systems.py     $A --part stab
python experiments/rq3_systems.py     $A --part sched
python experiments/make_figures.py    $A
echo "done -> results/${TASK}_${BACKEND}/"
