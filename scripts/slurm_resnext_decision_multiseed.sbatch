#!/bin/bash
#SBATCH --job-name=ankle-dfr-ms
#SBATCH --partition=RTXA6Kq
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -z "${CONFIGS:-}" ]]; then
  echo "CONFIGS must be a space-separated list of three formal config paths."
  exit 2
fi

LOG_DIR="${LOG_DIR:-autoresearch_logs/resnext_decision_multiseed}"
read -r -a CONFIG_ARRAY <<< "$CONFIGS"

.venv/bin/python scripts/run_resnext_decision_multiseed.py \
  --launcher srun \
  --log-dir "$LOG_DIR" \
  --configs "${CONFIG_ARRAY[@]}"
