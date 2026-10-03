#!/usr/bin/env bash
#SBATCH --job-name=gdwm-online
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --output=gdwm-%j.log
# Add your partition/account/time and activate your environment before submission.
set -euo pipefail
export NODES="$SLURM_NNODES"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
srun --kill-on-bad-exit=1 bash -c 'export NODE_RANK="$SLURM_NODEID"; bash scripts/train_h100.sh'
