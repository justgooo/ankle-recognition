#!/bin/bash
#SBATCH --job-name=ankle-dfr01
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

export ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=1
export ANKLE_DECISION_ENABLE_CLASSIFIER_VIEW_CONTEXT=1

.venv/bin/python train.py \
  --config configs/cmp_resnext_decision_256x8_dfr01_classifier_context_formal_s42.yaml
