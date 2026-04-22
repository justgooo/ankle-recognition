#!/bin/bash
#SBATCH --job-name=ankle-dfr08
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

.venv/bin/python train.py \
  --config configs/cmp_resnext_decision_256x8_dfr08_bounded_gating_floor05_formal_s123.yaml
