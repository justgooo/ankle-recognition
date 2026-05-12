#!/usr/bin/env bash
#SBATCH --job-name=ankle_3gpu_10round
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --chdir=/dataset/HH/ankle-ct
#SBATCH --output=autoresearch_logs/slurm_3gpu_10round_%j.out

set -euo pipefail

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

.venv/bin/python - <<'PY'
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

export TORCH_HOME=.torch-cache/torch
export HF_HOME=.torch-cache/hf
export XDG_CACHE_HOME=.torch-cache/xdg
export HUGGINGFACE_HUB_CACHE=.torch-cache/hf/hub

.venv/bin/python autoresearch_loop.py \
  --max-iterations 10 \
  --cooldown-seconds 30 \
  --default-lane main-study \
  --gpu-id 0 \
  --gpu-ids 0,1,2 \
  --max-workers 3 \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --main-search-template configs/optuna_main_search.yaml \
  --proxy-search-template configs/optuna_proxy_search.yaml \
  --formal-config configs/autoresearch_formal.yaml \
  --proxy-config configs/autoresearch_proxy.yaml \
  --campaign-note "Human requested a clean 3-GPU node and 10 autonomous rounds. Use the provided method list as inspiration only when it directly serves the DFR-25 ResNeXt 256x8 decision-fusion routing repair objective: modality/view dropout, co-regularization, learned gating, gradient balancing, view consistency or contrastive constraints, MoE, uncertainty-aware fusion, distillation, and cross-attention. Keep one discrete variable per round, do not repeat recently closed families without a specific new mechanism, and continue prioritizing learned full-fusion val_acc over matched equal-weight and DFR-25 while recording fusion-weight telemetry."
