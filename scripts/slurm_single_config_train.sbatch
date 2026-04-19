#!/bin/bash
#SBATCH --partition=RTXA6Kq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node16
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd "${TRAIN_WORKDIR:-/dataset/HH/ankle-ct}"
mkdir -p autoresearch_logs

: "${TRAIN_CONFIG:?TRAIN_CONFIG must be set}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

.venv/bin/python train.py --config "$TRAIN_CONFIG"
