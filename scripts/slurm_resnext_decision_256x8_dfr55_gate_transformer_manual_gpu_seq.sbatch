#!/bin/bash
#SBATCH --job-name=ankle-dfr55-gtx-manual
#SBATCH --partition=RTXA6Kq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MANUAL_CUDA_VISIBLE_DEVICES:-0}"

.venv/bin/python scripts/run_dfr55_gate_transformer_multiseed.py \
  --launcher direct \
  --max-parallel 1 \
  --cpus-per-run 12
