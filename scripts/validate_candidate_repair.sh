#!/usr/bin/env bash
# Fixed predictor/thresholds, validation only. Do NOT tune on confirmation seeds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-candidate_validation_$(date +%Y%m%d_%H%M%S)}"
echo "[candidate-validation] run=$RUN_ID GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u experiments/rq2_eval.py \
  --task push --backend torch --policy openvla \
  --vla_head_path "$HOME/scratch/openvla_chunk_head_llm_h10_d2.pt" \
  --vla_batch_size 8 --chunk_k 5 --roles orbisim \
  --n_envs 32 --rollout_seed 26000 --split validation --save_episodes \
  --calibration_path "$ROOT/src/results/push_torch/taus_protocol_v2_20261008_a_llm_decisions.json" \
  --arms none checkvla_orbisim checkvla_orbisim_discrete checkvla_orbisim_progress --tag "$RUN_ID"
echo "[candidate-validation] complete $RUN_ID"
