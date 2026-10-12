#!/usr/bin/env bash
# Sequential, actor-specific VALIDATION only. Never overwrite legacy taus/results.
# Launch with setsid nohup and select a healthy GPU by UUID, not ordinal.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"
export PYTHONUNBUFFERED=1
HEAD="$HOME/scratch/openvla_chunk_head_llm_h10_d2.pt"
RUN_ID="${RUN_ID:-protocol_v2_$(date +%Y%m%d_%H%M%S)}"
N_ENVS="${N_ENVS:-8}"
CALIB_ENVS="${CALIB_ENVS:-16}"
COMMON=(--task push --backend torch --policy openvla --vla_head_path "$HEAD"
        --chunk_k 5 --roles orbisim vision)
EVAL=(--n_envs "$N_ENVS" --rollout_seed 6000
      --arms none gt_shadow checkvla_orbisim checkvla_vision)
echo "[validation] run=$RUN_ID GPU=${CUDA_VISIBLE_DEVICES:-unset}"

# First isolate the cadence change, keeping old checkpoints and thresholds.
python -u experiments/rq2_eval.py "${COMMON[@]}" "${EVAL[@]}" \
  --tag "${RUN_ID}_llm_original_legacy_tau"

for VARIANT in original mass2; do
  EXTRA=()
  DIR="$ROOT/src/results/push_torch"
  if [[ "$VARIANT" == mass2 ]]; then
    EXTRA=(--variant mass2 --train_mass 0.5 2.0)
    DIR="${DIR}_mass2"
  fi
  TAUS="$DIR/taus_${RUN_ID}_llm_decisions.json"
  python -u experiments/calibrate.py "${COMMON[@]}" "${EXTRA[@]}" \
    --seed 41000 --n_cells 6 --n_envs "$CALIB_ENVS" --act_noise 0 \
    --calibration_path "$TAUS"
  python -u experiments/rq2_eval.py "${COMMON[@]}" "${EXTRA[@]}" "${EVAL[@]}" \
    --calibration_path "$TAUS" --tag "${RUN_ID}_llm_${VARIANT}_decision_tau"
done
echo "[validation] complete $RUN_ID (small validation sample, not headline evidence)"
