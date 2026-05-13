#!/usr/bin/env bash
#SBATCH --job-name=ankle_dfr135_replay
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --chdir=/dataset/HH/ankle-ct
#SBATCH --output=autoresearch_logs/slurm_dfr135_frozen_replay_%j.out

set -euo pipefail

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

DFR135_GPU_ID="${DFR135_GPU_ID:-4}"
export CUDA_VISIBLE_DEVICES="${DFR135_GPU_ID}"
echo "DFR135 selected physical GPU id: ${DFR135_GPU_ID}"
echo "Overridden CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export TORCH_HOME=.torch-cache/torch
export HF_HOME=.torch-cache/hf
export XDG_CACHE_HOME=.torch-cache/xdg
export HUGGINGFACE_HUB_CACHE=.torch-cache/hf/hub
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CHECKPOINT=runs/optuna_main_autoloop/iter_0001_20260423_021418_retry1/trials/trial_0001/run/best.pt
CONFIG_N24=configs/cmp_resnext_decision_256x24_dfr135_frozen_dfr25_replay_s42.yaml
CONFIG_N32=configs/cmp_resnext_decision_256x32_dfr135_frozen_dfr25_replay_s42.yaml
OUT_N24=runs/resnext_decision_256x8_mainline/dfr135_frozen_dfr25_replay_256x24_s42/fusion_weight_analysis.json
OUT_N32=runs/resnext_decision_256x8_mainline/dfr135_frozen_dfr25_replay_256x32_s42/fusion_weight_analysis.json

.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

.venv/bin/python scripts/analyze_fusion_weights.py \
  --config "${CONFIG_N24}" \
  --checkpoint "${CHECKPOINT}" \
  --split val \
  --output-json "${OUT_N24}"

.venv/bin/python scripts/analyze_fusion_weights.py \
  --config "${CONFIG_N32}" \
  --checkpoint "${CHECKPOINT}" \
  --split val \
  --output-json "${OUT_N32}"

echo "End: $(date)"
