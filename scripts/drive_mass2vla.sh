#!/usr/bin/env bash
# Predictor coverage fix, step B: step A's data (mass 0.5-2.0, noisy expert)
# plus OpenVLA-driven rollouts over the same ranges (vla/collect_vla_dyn.py),
# so the predictor also sees the states the VLA actually visits.
# Results in results/push_torch_mass2vla; compared against step A (mass2).
#   setsid nohup bash ~/UoE-UG4-Dissertation/scripts/drive_mass2vla.sh > ~/scratch/drive_mass2vla.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/UoE-UG4-Dissertation/src
export CUDA_VISIBLE_DEVICES=${GPU:-1}
A=results/push_torch_mass2
B=results/push_torch_mass2vla
mkdir -p $B
# wait for step A's data and the VLA rollouts
while [ ! -f $A/dyn_data.pt ] || ! grep -q "^done" ~/scratch/collect_vla_dyn.log; do sleep 30; done
cp $A/bc_data.pt $B/
python - <<PY
import torch
a = torch.load("$A/dyn_data.pt"); v = torch.load("$HOME/scratch/dyn_data_vla.pt")
assert a["param_names"] == v["param_names"], (a["param_names"], v["param_names"])
for k in ("states", "actions", "risk", "param_mults"):
    assert a[k].shape[1:] == v[k].shape[1:], (k, a[k].shape, v[k].shape)
m = {k: torch.cat([a[k], v[k]]) for k in ("states", "actions", "risk", "param_mults")}
m["param_names"] = a["param_names"]
torch.save(m, "$B/dyn_data.pt")
print(f"merged dyn_data: {a['states'].shape[0]} expert + {v['states'].shape[0]} VLA episodes")
PY
V="--task push --backend torch --variant mass2vla --train_mass 0.5 2.0"
VLA="--policy openvla --vla_head_path $HOME/scratch/openvla_chunk_head_vis_h10.pt --chunk_k 5"
run() { echo "=== $1  $(date +%T)"; shift; "$@"; echo "--- EXIT $?  $(date +%T)"; }
run train     python -u experiments/train.py $V --what all
run calibrate python -u experiments/calibrate.py $V $VLA
run headline  python -u experiments/rq2_eval.py $V $VLA \
                --arms none gt_shadow checkvla_orbisim checkvla_vision --tag openvla_vis_h10
echo "ALL DONE mass2vla"
