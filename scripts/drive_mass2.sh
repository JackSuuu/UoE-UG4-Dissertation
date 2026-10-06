#!/usr/bin/env bash
# Predictor coverage fix, step A: widen the mass training range to 0.5-2.0.
# Friction range unchanged (0.5-1.5), so friction 0.2x stays a true OOD test.
# Everything goes to results/push_torch_mass2; the v7 predictor is untouched.
#   setsid nohup bash ~/scratch/drive_mass2.sh > ~/scratch/drive_mass2.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/UoE-UG4-Dissertation/src
export CUDA_VISIBLE_DEVICES=${GPU:-0}
V="--task push --backend torch --variant mass2 --train_mass 0.5 2.0"
VLA="--policy openvla --vla_head_path $HOME/scratch/openvla_chunk_head_vis_h10.pt --chunk_k 5"
run() { echo "=== $1  $(date +%T)"; shift; "$@"; echo "--- EXIT $?  $(date +%T)"; }
run collect   python -u experiments/collect_data.py $V
run train     python -u experiments/train.py $V --what all
run calibrate python -u experiments/calibrate.py $V $VLA
run headline  python -u experiments/rq2_eval.py $V $VLA \
                --arms none gt_shadow checkvla_orbisim checkvla_vision --tag openvla_vis_h10
echo "ALL DONE mass2"
