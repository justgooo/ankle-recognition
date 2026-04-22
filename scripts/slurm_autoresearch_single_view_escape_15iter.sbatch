#!/bin/bash
#SBATCH --job-name=ankle-sv-escape
#SBATCH --partition=RTXA6Kq
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export PATH="$HOME/.npm-global/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

GPU_IDS_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "$GPU_IDS_RAW" ]; then
  echo "CUDA_VISIBLE_DEVICES is empty; Slurm did not expose allocated GPUs."
  exit 1
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS_RAW"
FIRST_GPU="${GPU_ID_ARRAY[0]}"

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${GPU_IDS_RAW}"
echo "which_codex=$(command -v codex)"
if [ -f "$HOME/.codex/auth.json" ]; then
  echo "codex_auth=present"
else
  echo "codex_auth=missing"
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
.venv/bin/python -c "import torch; print('torch_cuda_available', torch.cuda.is_available()); print('torch_cuda_device_count', torch.cuda.device_count()); [print('torch_device', i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

check_allocated_gpu_idle() {
  local idle=0
  while IFS=, read -r raw_idx _ raw_used _ raw_util; do
    local idx="${raw_idx//[[:space:]]/}"
    local used="${raw_used//[^0-9]/}"
    local util="${raw_util//[^0-9]/}"
    local selected=0
    for gpu_id in "${GPU_ID_ARRAY[@]}"; do
      if [ "$idx" = "$gpu_id" ]; then
        selected=1
        break
      fi
    done
    if [ "$selected" -eq 0 ]; then
      continue
    fi
    if [ "${used:-0}" -le 1024 ] && [ "${util:-0}" -le 20 ]; then
      idle=$((idle + 1))
    else
      echo "allocated_gpu_busy index=${idx} used_mb=${used:-0} util=${util:-0}"
    fi
  done < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader)
  [ "$idle" -eq "${#GPU_ID_ARRAY[@]}" ]
}

for attempt in $(seq 1 20); do
  if check_allocated_gpu_idle; then
    echo "allocated GPUs are idle enough for autoresearch on attempt ${attempt}."
    break
  fi
  if [ "$attempt" -eq 20 ]; then
    echo "Allocated GPUs never became idle enough within the startup wait window; requeueing."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  sleep 30
done

CAMPAIGN_NOTE=$(cat <<'EOF'
- 当前 autoresearch 的唯一主方向是：摆脱目前 decision fusion 除 `equal-weight` 外最佳 learned 方案的单视角依赖，尤其是 `axial dominance` / `weak-view starvation`，并把真正的 multi-view full-fusion learned `val_acc` 做到明确高于 matched `equal-weight` control。
- 在完成上面目标之前，不允许切换到其他方向；不要切 backbone family，不要切到 feature fusion，不要把无关 side campaign、泛化 cleanup 或与单视角依赖无直接关系的小修小补当成主线。
- 本轮以及后续每一轮 inner agent 在行动前都必须至少做两轮自检：
  1. 这项改动是否直接减弱 single-view dependence，或直接增强弱视角在 full-fusion 中的有效贡献？
  2. 如果它成功，为什么它有机会把 multi-view full-fusion learned `val_acc` 推到 matched `equal-weight` 之上，而不是只改善单视角表现、稳定性或局部 calibration？
- 如果上面两问任意一问不能给出具体机制链路，该方向就不合格，不要执行。
- 只允许优先探索真正改变 evidence routing、缓解弱视角 starvation、或让 stronger per-view experts 在 full-fusion 下形成超过 `equal-weight` 净收益的修复。
- 本次作业固定运行 15 轮 outer autoresearch iterations；不要无限循环。
EOF
)

timeout 85800 .venv/bin/python autoresearch_loop.py \
  --max-iterations 15 \
  --cooldown-seconds 30 \
  --gpu-id "${FIRST_GPU}" \
  --default-lane main-study \
  --gpu-ids "${GPU_IDS_RAW}" \
  --max-workers 3 \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --main-search-template configs/optuna_main_search_resnext_decision_256x8.yaml \
  --proxy-search-template configs/optuna_proxy_search_resnext_decision_256x8.yaml \
  --formal-config configs/autoresearch_formal_resnext_decision_256x8.yaml \
  --proxy-config configs/autoresearch_proxy_resnext_decision_256x8.yaml \
  --min-disk-free-gb 20 \
  --campaign-note "$CAMPAIGN_NOTE"
