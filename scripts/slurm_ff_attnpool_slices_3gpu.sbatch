#!/bin/bash
#SBATCH --job-name=ankle-ff-attnp-3g
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

GPU_IDS="${CUDA_VISIBLE_DEVICES:-auto}"
if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS=auto
fi

.venv/bin/python scripts/autoresearch_main.py \
  --search-config configs/optuna_main_search_ff_attnpool_slices_3seed.yaml \
  --trials 12 \
  --gpu-ids "${GPU_IDS}" \
  --max-workers 3 \
  --max-used-memory-mb 4096 \
  --max-utilization 60 \
  --worker-cooldown-seconds 5 \
  --tail-fill never
