#!/bin/bash
#SBATCH --job-name=ankle-ff-attnp-2g
#SBATCH --partition=PA100q,PH100q,NH100q,NA100q,RTXA6Kq,HPCAIq,V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=90G
#SBATCH --time=12:00:00
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

IDLE_GPU_IDS="$(.venv/bin/python scripts/check_idle_gpus.py \
  --gpu-ids "${GPU_IDS}" \
  --min-idle-count 1 \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --id-list-only)"
IFS=',' read -r -a IDLE_GPU_ARRAY <<< "${IDLE_GPU_IDS}"
WORKERS="${#IDLE_GPU_ARRAY[@]}"
echo "IDLE_GPU_IDS=${IDLE_GPU_IDS}"
echo "WORKERS=${WORKERS}"

.venv/bin/python scripts/autoresearch_main.py \
  --search-config configs/optuna_main_search_ff_attnpool_slices_3seed.yaml \
  --trials 12 \
  --gpu-ids "${IDLE_GPU_IDS}" \
  --max-workers "${WORKERS}" \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --worker-cooldown-seconds 5 \
  --tail-fill never
