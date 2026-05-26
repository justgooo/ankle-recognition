#!/bin/bash
#SBATCH --job-name=ankle-ff-3seed
#SBATCH --partition=RTXA6Kq
#SBATCH --nodes=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=60
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CONFIGS=(
  configs/paper_feature_fusion_gated_head_formal_s42.yaml
  configs/paper_feature_fusion_gated_head_formal_s123.yaml
  configs/paper_feature_fusion_gated_head_formal_s456.yaml
  configs/paper_feature_fusion_minimal_baseline_formal_s42.yaml
  configs/paper_feature_fusion_minimal_baseline_formal_s123.yaml
  configs/paper_feature_fusion_minimal_baseline_formal_s456.yaml
)

.venv/bin/python scripts/run_dfr44_evidence_gate_multiseed.py \
  --launcher srun \
  --max-parallel 5 \
  --cpus-per-run 12 \
  --min-free-gb 20 \
  --max-gpu-utilization 35 \
  --skip-telemetry \
  --log-dir autoresearch_logs/paper_feature_fusion_3seed_5gpu \
  --configs "${CONFIGS[@]}"
