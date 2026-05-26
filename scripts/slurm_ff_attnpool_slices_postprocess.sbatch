#!/bin/bash
#SBATCH --job-name=ankle-ff-attnp-post
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

TRAIN_JOBID="${TRAIN_JOBID:-unknown}"

.venv/bin/python scripts/postprocess_ff_attnpool_slices.py \
  --study-dir runs/optuna_main_ff_attnpool_slices_3seed \
  --train-jobid "${TRAIN_JOBID}"
