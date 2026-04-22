#!/bin/bash
#SBATCH --job-name=ankle-dfr10
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
export ANKLE_DECISION_FUSION_WEIGHT_FLOOR=0.05
export ANKLE_DECISION_TRAIN_VIEW_DROPOUT_PROB=0.15
export ANKLE_DECISION_TRAIN_AXIAL_BLUR_PROB=0.30
export ANKLE_DECISION_TRAIN_AXIAL_BLUR_KERNEL=9

.venv/bin/python train.py \
  --config configs/cmp_resnext_decision_256x8_dfr10_floor05_plus_robustness_formal_s42.yaml
