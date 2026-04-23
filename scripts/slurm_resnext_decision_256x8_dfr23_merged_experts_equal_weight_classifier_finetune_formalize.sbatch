#!/bin/bash
#SBATCH --job-name=ankle-dfr23f
#SBATCH --partition=V100q
#SBATCH --nodelist=node20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

.venv/bin/python scripts/train_decision_stagewise.py \
  --config configs/cmp_resnext_decision_256x8_dfr23_merged_experts_equal_weight_classifier_finetune_formalize_s42.yaml
