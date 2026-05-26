#!/bin/bash
#SBATCH --job-name=ankle-ff-struct5
#SBATCH --partition=V100q,RTXA6Kq,PA100q,PH100q,NH100q,NA100q,HPCAIq
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=10:00:00
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
  --min-idle-count 3 \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --id-list-only)"
if [[ -z "${IDLE_GPU_IDS}" && "${GPU_IDS}" == 0,* ]]; then
  echo "No three idle GPUs with GPU0 included; retrying while tolerating GPU0 occupancy."
  REMAINING_GPU_IDS="${GPU_IDS#0,}"
  IDLE_GPU_IDS="$(.venv/bin/python scripts/check_idle_gpus.py \
    --gpu-ids "${REMAINING_GPU_IDS}" \
    --min-idle-count 3 \
    --max-used-memory-mb 1024 \
    --max-utilization 20 \
    --id-list-only)"
fi
echo "IDLE_GPU_IDS=${IDLE_GPU_IDS}"
if [[ -z "${IDLE_GPU_IDS}" ]]; then
  echo "No usable 3-GPU idle subset found in CUDA_VISIBLE_DEVICES=${GPU_IDS}" >&2
  exit 2
fi

mapfile -t CONFIGS < <(
  .venv/bin/python scripts/prepare_feature_fusion_structure_5round.py \
    | grep '^configs/.*\.yaml$'
)

.venv/bin/python scripts/backbone_preflight.py --config "${CONFIGS[@]}"

.venv/bin/python scripts/run_dfr44_evidence_gate_multiseed.py \
  --launcher srun \
  --max-parallel 3 \
  --cpus-per-run 12 \
  --min-free-gb 20 \
  --max-gpu-utilization 20 \
  --skip-telemetry \
  --log-dir autoresearch_logs/feature_fusion_structure_5round_3gpu \
  --configs "${CONFIGS[@]}"

.venv/bin/python scripts/postprocess_feature_fusion_structure_5round.py
