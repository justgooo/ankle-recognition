#!/bin/bash
#SBATCH --job-name=ankle-mainline-auto
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --nodelist=node20
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export PATH="$HOME/.npm-global/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "which_codex=$(command -v codex)"
if [ -f "$HOME/.codex/auth.json" ]; then
  echo "codex_auth=present"
else
  echo "codex_auth=missing"
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
.venv/bin/python -c "import torch; print('torch_cuda_available', torch.cuda.is_available()); print('torch_cuda_device_count', torch.cuda.device_count()); [print('torch_device', i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

CAMPAIGN_NOTE=$(cat <<'EOF'
- 人类已明确把 canonical 主线重排为：先做 matched 模块贡献分析，再尝试优化 weighting 比例。
- 在模块分析结论出来之前，不要把“自动优化加权比例”当默认第一步，也不要直接增加更复杂的 gating / weighting path 容量。
- 首批实验必须优先围绕当前 canonical decision-fusion 主线做可解释对照：`equal-weight` control、`minimal learned` 路径、`current learned` 路径；如需继续拆 richer reliability path，也只能一次改一个模块。
- 如果模块分析显示 simpler weighting 更强或更稳，应先把 simpler path 当主参考，再决定 learned weighting 是否值得保留。
EOF
)

timeout 43140 .venv/bin/python autoresearch_loop.py \
  --max-iterations 0 \
  --cooldown-seconds 30 \
  --gpu-id 1 \
  --default-lane main-study \
  --gpu-ids 0,1,2 \
  --max-workers 3 \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --main-search-template configs/optuna_main_search.yaml \
  --proxy-search-template configs/optuna_proxy_search.yaml \
  --formal-config configs/autoresearch_formal.yaml \
  --proxy-config configs/autoresearch_proxy.yaml \
  --min-disk-free-gb 20 \
  --campaign-note "$CAMPAIGN_NOTE"
