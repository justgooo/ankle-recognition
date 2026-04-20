#!/bin/bash
#SBATCH --job-name=ankle-rx-attnp-eq
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=15
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

.venv/bin/python scripts/run_resnext_attnpool_equal_multiseed.py \
  --execute \
  --inside-allocation \
  --cpus-per-run 5 \
  --timeout 10800 \
  --max-parallel 3 \
  --min-free-gb 20 \
  --log-dir autoresearch_logs/resnext_attnpool_equal_multiseed
