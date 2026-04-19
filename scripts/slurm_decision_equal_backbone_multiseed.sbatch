#!/bin/bash
#SBATCH --job-name=ankle-dec-eq-seed
#SBATCH --partition=RTXA6Kq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --nodelist=node16
#SBATCH --output=autoresearch_logs/slurm_decision_equal_backbone_multiseed_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/run_decision_equal_backbone_multiseed.py \
  --execute \
  --cpus-per-run 8 \
  --timeout 14400 \
  --min-free-gb 40 \
  --log-dir autoresearch_logs/decision_equal_backbone_multiseed
