#!/usr/bin/env bash
# P2a: Releasable Physics-Constraint Benchmark for Task A (box push)
#
# Usage:
#   bash run_p2a.sh --actor openvla_chunk --chunk_k 5 --device cuda:0
#   bash run_p2a.sh --actor scripted_expert
#   bash run_p2a.sh --actor bc
#
# Output: JSON with safe_success, SR, CVR per actor × per perturbation cell

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

ACTOR="openvla_chunk"
CHUNK_K=5
DEVICE="cuda:0"
HEAD_PATH="~/scratch/openvla_chunk_head.pt"
SEED=42
OUT_DIR="${SCRIPT_DIR}/results/p2a"
N_ENVS=64

usage() {
    cat <<EOF
P2a Physics-Constraint Benchmark -- Task A (box push)

Usage: $0 [options]

Options:
  --actor ACTOR        Actor to evaluate: scripted_expert | bc | openvla_chunk (default: openvla_chunk)
  --chunk_k K          Open-loop chunk length (default: 5)
  --device DEV         CUDA device (default: cuda:0)
  --head_path PATH     Chunk head checkpoint (default: ~/scratch/openvla_chunk_head.pt)
  --seed SEED          Random seed for initial states (default: 42)
  --out_dir DIR        Output directory (default: results/p2a)
  --n_envs N           Number of parallel envs (default: 64)
  -h, --help           Show this help

Actors:
  scripted_expert      Ground-truth expert (sanity: verifier must not break correct policy)
  bc                   Behaviour cloning stand-in (current proxy for large sweeps)
  openvla_chunk        OpenVLA-7B with trained chunk head (H×2 world-frame m/s)

The benchmark defines:
  - Seeded initial states (fixed per cell, reproducible)
  - Perturbation grid: friction × mass (same as RQ2 OOD grid)
  - Episode length: 40 steps (8s simulated)
  - Success: block reaches seat without wall contact
  - Violation: wall contact force > threshold
  - Safe success: success AND no violation
  - Metrics: SR, CVR, safe_success, intervention_rate, latency_ms
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --actor) ACTOR="$2"; shift 2 ;;
        --chunk_k) CHUNK_K="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --head_path) HEAD_PATH="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        --n_envs) N_ENVS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "${OUT_DIR}"

cd "${SRC_DIR}"

# Build the actor argument for the eval script
case "${ACTOR}" in
    scripted_expert)
        ACTOR_ARGS="--policy scripted_expert"
        ;;
    bc)
        ACTOR_ARGS="--policy bc"
        ;;
    openvla_chunk)
        ACTOR_ARGS="--policy openvla --vla_model openvla/openvla-7b --vla_head_path ${HEAD_PATH} --action_mode planar_head --chunk_k ${CHUNK_K}"
        ;;
    *)
        echo "Unknown actor: ${ACTOR}"
        exit 1
        ;;
esac

# Run evaluation on the OOD grid
python -u experiments/rq2_eval.py \
    --task push \
    --backend torch \
    --chunk_k "${CHUNK_K}" \
    --camera \
    --cam_res 224 \
    --cam_ss 1 \
    ${ACTOR_ARGS} \
    --seed "${SEED}" \
    --n_envs "${N_ENVS}" \
    --device "${DEVICE}" \
    --out "${OUT_DIR}/p2a_${ACTOR}_seed${SEED}.json" \
    2>&1 | tee "${OUT_DIR}/p2a_${ACTOR}_seed${SEED}.log"

echo "P2a eval complete. Results in ${OUT_DIR}/p2a_${ACTOR}_seed${SEED}.json"