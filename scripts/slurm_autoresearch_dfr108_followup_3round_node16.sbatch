#!/usr/bin/env bash
#SBATCH --job-name=ankle_dfr108_3round
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --chdir=/dataset/HH/ankle-ct
#SBATCH --output=autoresearch_logs/slurm_dfr108_followup_3round_%j.out

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
  --max-iterations 3 \
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
  --campaign-note "Human requested 3 more rounds in the axial-lock-in / non-axial-target direction after DFR-108. Latest state: DFR-106/108 audits found a clean tiny target set on DFR-25 seed42: all useful confirmed targets are label0 sagittal, covering several DFR-25 errors/non-axial true-margin samples with helpful/harmful 4/0; DFR-107 failed because the binary class-coded auxiliary let 44 non-target-only samples create axial-reinforcing CE pressure, leaving top-weight 94/0/0. Continue from this exact diagnosis. Prioritize label-invariant, target-only gate ranking objectives or a similarly scoped mechanism that has zero gradient on non-target samples and does not train per-view classifiers. Preserve DFR-25 ResNeXt 256x8 decision-fusion geometry/scalars/runtime. Each round must be one discrete variable, run main-study/formal, generate fusion_weight_analysis.json, and record mean weights plus top-weight counts. Do not repeat broad coronal/role/floor/blend/active-mask sweeps already discarded in DFR-57 through DFR-108."
