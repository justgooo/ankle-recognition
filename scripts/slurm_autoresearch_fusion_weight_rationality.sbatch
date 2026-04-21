#!/bin/bash
#SBATCH --job-name=ankle-fw-rationality
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --nodelist=node19
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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
- 这是一个人类明确要求的 bounded side campaign：外层 autoresearch_loop 只跑 3 轮，然后退出；不要无限循环。
- 当前已有 in-flight job `432763` (`ankle-rx256x8-l6s123`) 在 `node20` 上执行 `L6` scalar repair。不要重复 `L6`，不要复用该作业的 `study_root`、`output_dir`、日志路径或运行目录。
- 本 campaign 的默认锚点是 `256x8 ResNeXt decision fusion + L3-no-mixer` strongest learned branch；如需 control，可派生 `equal-weight` / single-view / leave-one-view-out，但不要切回 `512x16`。
- 三轮任务顺序固定：
  1. `FWR-01` single-view / leave-one-view-out matched control
  2. `FWR-02` fusion-weight telemetry (`top-weight hit rate` + weight-vs-margin/correctness consistency)
  3. `FWR-03` perturbation-based weight migration
- 每轮只做一个离散实验；如需新增 instrumentation，优先把分析产物写入各自 `output_dir` 的 json 文件。
- 默认优先当前 canonical `256x8` formal/main lane；只有 smoke/debug 才退到 proxy。
EOF
)

timeout 28740 .venv/bin/python autoresearch_loop.py \
  --max-iterations 3 \
  --cooldown-seconds 30 \
  --gpu-id 0 \
  --default-lane main-study \
  --gpu-ids auto \
  --max-workers 2 \
  --max-used-memory-mb 1024 \
  --max-utilization 20 \
  --main-search-template configs/optuna_main_search_resnext_decision_256x8.yaml \
  --proxy-search-template configs/optuna_proxy_search_resnext_decision_256x8.yaml \
  --formal-config configs/autoresearch_formal_resnext_decision_256x8.yaml \
  --proxy-config configs/autoresearch_proxy_resnext_decision_256x8.yaml \
  --min-disk-free-gb 20 \
  --campaign-note "$CAMPAIGN_NOTE"
