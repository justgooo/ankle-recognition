#!/bin/bash
#SBATCH --job-name=ankle-ff-id-ch6
#SBATCH --partition=V100q,RTXA6Kq,PA100q,PH100q,NH100q,NA100q,HPCAIq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --chdir=/dataset/HH/ankle-ct
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_HOME=.torch-cache/torch
export HF_HOME=.torch-cache/hf
export XDG_CACHE_HOME=.torch-cache/xdg
export HUGGINGFACE_HUB_CACHE=.torch-cache/hf/hub

echo "Start: $(date)"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

.venv/bin/python - <<'PY'
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [[ -z "${GPU_IDS}" || "${GPU_IDS}" == "unset" ]]; then
  GPU_IDS="0,1,2"
fi
FALLBACK_GPU_ID="${GPU_IDS%%,*}"

echo "AUTORESEARCH_GPU_IDS=${GPU_IDS}"
echo "AUTORESEARCH_FALLBACK_GPU_ID=${FALLBACK_GPU_ID}"

timeout 86400 .venv/bin/python autoresearch_loop.py \
  --max-iterations 6 \
  --cooldown-seconds 30 \
  --default-lane main-study \
  --gpu-id "${FALLBACK_GPU_ID}" \
  --gpu-ids "${GPU_IDS}" \
  --max-workers 3 \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --main-search-template configs/optuna_main_search.yaml \
  --proxy-search-template configs/optuna_proxy_search.yaml \
  --formal-config configs/autoresearch_formal.yaml \
  --proxy-config configs/autoresearch_proxy.yaml \
  --campaign-note "Human requested exactly six feature-fusion autoresearch rounds with 3-GPU parallel 3-seed main studies: first complete FF-IDSKIP-AR-01/02/03, then FF-CHGATE-AR-01/02/03. For the first three completed iterations, choose the earliest open FF-IDSKIP backlog item and keep every experiment in the identity-skip family: enable ANKLE_FEATURE_ENABLE_XVIEW_IDENTITY_SKIP_GATE=1 and change only one identity-skip gate property such as max_scale, gate input/normalization, bounded residual formulation, placement relative to existing post-mixer token flow, or a stability regularizer specific to that gate. Do not enable the channel gate in those identity-skip rounds unless it is an explicitly justified single control. For the next three completed iterations, choose the earliest open FF-CHGATE backlog item and keep every experiment in the channel-gate family: enable ANKLE_FEATURE_ENABLE_VIEW_TOKEN_CHANNEL_GATE=1 and change only one channel-gate property such as scale bounds, bottleneck/channel capacity, initialization, per-view sharing, normalization, or stability regularization. Do not switch to unrelated feature residuals, attention pooling, decision-fusion DFR work, or pairwise fixed/broken audit during this campaign. Each round must use fresh configs/optuna_main_search.yaml semantics with seeds 42/123/456, max-workers 3, current ResNeXt 256x8 feature-fusion gated-head anchor geometry/scalars, and must record gate parameter norms plus validation scale/correction ratios where applicable."
