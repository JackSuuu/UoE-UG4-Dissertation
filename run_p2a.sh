#!/usr/bin/env bash
# P2a: the Task A physics-constraint benchmark with the actor as an argument.
#
#   bash run_p2a.sh bc                 # behaviour-cloning stand-in
#   bash run_p2a.sh openvla            # OpenVLA-7B + trained chunk head
#   bash run_p2a.sh openvla 2          # same, on CUDA device index 2
#
# Same OOD grid, verifier, taus and safe-success definition as the RQ2 table;
# only the actor changes. Results go to src/results/push_torch/rq2_<actor>.json
# (bc writes rq2.json), so an actor run never overwrites another.
#
# Not yet an arm: the scripted expert (needs a --policy expert in registry.py).
#
# Long runs: launch detached, or the shell that started them will kill them:
#   setsid nohup bash run_p2a.sh openvla > ~/scratch/p2a_openvla.log 2>&1 < /dev/null &
set -euo pipefail

ACTOR="${1:-openvla}"
GPU="${2:-0}"
ARMS="${ARMS:-none gt_shadow checkvla_orbisim checkvla_vision}"
HEAD="${HEAD:-$HOME/scratch/openvla_chunk_head.pt}"

cd "$(dirname "${BASH_SOURCE[0]}")/src"

case "$ACTOR" in
    bc)      EXTRA=() ;;
    openvla) EXTRA=(--vla_head_path "$HEAD" --action_mode planar_head)
             [ -f "$HEAD" ] || { echo "no chunk head at $HEAD -- train it first"; exit 1; } ;;
    *)       echo "unknown actor '$ACTOR' (bc | openvla)"; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES="$GPU" python -u experiments/rq2_eval.py \
    --task push --backend torch --chunk_k 5 \
    --policy "$ACTOR" --arms $ARMS "${EXTRA[@]}"
