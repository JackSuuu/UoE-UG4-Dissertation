#!/usr/bin/env bash
# Fixed-method confirmation on independent episodes. No parameter search here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
RUN_ID="${RUN_ID:-verifier_confirmation_$(date +%Y%m%d_%H%M%S)}"
echo "[confirmation] run=$RUN_ID GPU=${CUDA_VISIBLE_DEVICES:-unset}"
python -u experiments/rq2_eval.py \
  --task push --backend torch --policy openvla \
  --vla_head_path "$HOME/scratch/openvla_chunk_head_llm_h10_d2.pt" \
  --vla_batch_size 8 --chunk_k 5 --roles orbisim \
  --n_envs 64 --rollout_seed 16000 --split confirmation --save_episodes \
  --calibration_path "$ROOT/src/results/push_torch/taus_protocol_v2_20261008_a_llm_decisions.json" \
  --legacy_calibration_path "$ROOT/src/results/push_torch/taus.json" \
  --arms none gt_shadow checkvla_orbisim checkvla_orbisim_legacy_tau --tag "$RUN_ID"
echo "[confirmation] complete $RUN_ID"
