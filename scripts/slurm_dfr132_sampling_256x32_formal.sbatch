#!/usr/bin/env bash
#SBATCH --job-name=ankle_dfr132_256x32
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --chdir=/dataset/HH/ankle-ct
#SBATCH --output=autoresearch_logs/slurm_dfr132_sampling_256x32_%j.out

set -euo pipefail

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

DFR132_GPU_ID="${DFR132_GPU_ID:-4}"
export CUDA_VISIBLE_DEVICES="${DFR132_GPU_ID}"
echo "DFR132 selected physical GPU id: ${DFR132_GPU_ID}"
echo "Overridden CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export TORCH_HOME=.torch-cache/torch
export HF_HOME=.torch-cache/hf
export XDG_CACHE_HOME=.torch-cache/xdg
export HUGGINGFACE_HUB_CACHE=.torch-cache/hf/hub
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CONFIG=configs/cmp_resnext_decision_256x32_dfr132_sampling_formal_s42.yaml
OUTDIR=runs/resnext_decision_256x8_mainline/dfr132_sampling_256x32_formal_s42

.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

.venv/bin/python scripts/run_train_with_config_env.py --config "${CONFIG}" --python .venv/bin/python

.venv/bin/python scripts/analyze_fusion_weights.py \
  --config "${CONFIG}" \
  --checkpoint "${OUTDIR}/best.pt" \
  --split val

echo "End: $(date)"
