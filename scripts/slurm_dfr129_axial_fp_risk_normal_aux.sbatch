#!/usr/bin/env bash
#SBATCH --job-name=ankle_dfr129
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --chdir=/dataset/HH/ankle-ct
#SBATCH --output=autoresearch_logs/slurm_dfr129_axial_fp_risk_normal_aux_%j.out

set -euo pipefail

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

DFR129_GPU_IDS=4,5,6
export CUDA_VISIBLE_DEVICES="${DFR129_GPU_IDS}"
echo "DFR129 selected physical GPU ids: ${DFR129_GPU_IDS}"
echo "Overridden CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export TORCH_HOME=.torch-cache/torch
export HF_HOME=.torch-cache/hf
export XDG_CACHE_HOME=.torch-cache/xdg
export HUGGINGFACE_HUB_CACHE=.torch-cache/hf/hub

.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

.venv/bin/python scripts/autoresearch_main.py \
  --search-config configs/optuna_main_search_resnext_decision_256x8_dfr129_axial_fp_risk_normal_aux.yaml \
  --gpu-ids "${DFR129_GPU_IDS}" \
  --max-workers 3 \
  --max-used-memory-mb 1024 \
  --max-utilization 20

.venv/bin/python scripts/monitor_optuna.py \
  --study-dir runs/optuna_main_resnext_decision_256x8_dfr129_axial_fp_risk_normal_aux
