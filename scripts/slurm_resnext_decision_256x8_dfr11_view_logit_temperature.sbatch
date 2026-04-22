#!/bin/bash
#SBATCH --job-name=ankle-dfr11
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

.venv/bin/python scripts/run_train_with_config_env.py \
  --config configs/cmp_resnext_decision_256x8_dfr11_view_logit_temperature_formal_s42.yaml
