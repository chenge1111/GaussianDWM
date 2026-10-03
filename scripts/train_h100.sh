#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MANIFEST:?Set MANIFEST to the training JSONL}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
CONFIG="${CONFIG:-configs/research/online_joint.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/online_joint}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODES="${NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TOTAL_PROCESSES=$((GPUS_PER_NODE * NODES))
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
args=(--config "$CONFIG" --manifest "$MANIFEST" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR")
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
accelerate launch --config_file configs/cluster/accelerate_h100.yaml \
  --num_processes "$TOTAL_PROCESSES" --num_machines "$NODES" \
  --machine_rank "$NODE_RANK" --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  -m gaussiandwm_research.train "${args[@]}"
