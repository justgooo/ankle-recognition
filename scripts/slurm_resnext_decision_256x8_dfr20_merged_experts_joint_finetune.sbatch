#!/bin/bash
#SBATCH --job-name=ankle-dfr20
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs runs/resnext_decision_256x8_mainline/artifacts

if [ ! -f runs/resnext_decision_256x8_mainline/artifacts/dfr19_merged_single_view_experts_s42.pt ]; then
  .venv/bin/python scripts/merge_single_view_experts.py \
    --base-checkpoint runs/resnext_decision_256x8_matrix/formal/l3_no_mixer/s42/best.pt \
    --axial-checkpoint runs/resnext_decision_256x8_mainline/dfr16_single_view_axial_formal_s42/best.pt \
    --coronal-checkpoint runs/resnext_decision_256x8_mainline/dfr17_single_view_coronal_formal_s42/best.pt \
    --sagittal-checkpoint runs/resnext_decision_256x8_mainline/dfr18_single_view_sagittal_formal_s42/best.pt \
    --output-checkpoint runs/resnext_decision_256x8_mainline/artifacts/dfr19_merged_single_view_experts_s42.pt
fi

.venv/bin/python scripts/train_decision_stagewise.py \
  --config configs/cmp_resnext_decision_256x8_dfr20_merged_experts_joint_finetune_formal_s42.yaml
