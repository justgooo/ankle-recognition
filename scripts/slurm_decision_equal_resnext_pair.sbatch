#!/bin/bash
#SBATCH --job-name=ankle-rx-eqctrl
#SBATCH --partition=RTXA6Kq
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

LEARNED_CFG="configs/cmp_decision_equal_resnext_learned_512x16_e20_s42.yaml"
EQUAL_CFG="configs/cmp_decision_equal_resnext_equal_512x16_e20_s42.yaml"

echo "hostname=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

.venv/bin/python -c "import sys, torch; count=torch.cuda.device_count(); print(f'visible_gpus={count}'); [print(i, torch.cuda.get_device_name(i), f'free_gb={torch.cuda.mem_get_info(i)[0]/1024**3:.2f}') for i in range(count)]; sys.exit(0 if count >= 3 and min(torch.cuda.mem_get_info(i)[0] for i in range(count)) >= 20 * 1024**3 else 2)"

.venv/bin/python scripts/backbone_preflight.py --config "$LEARNED_CFG" "$EQUAL_CFG"

srun --exclusive -N1 -n1 --gres=gpu:1 -c6 \
  --output=autoresearch_logs/%x_learned_%j.out \
  .venv/bin/python train.py --config "$LEARNED_CFG" &
PID_LEARNED=$!

srun --exclusive -N1 -n1 --gres=gpu:1 -c6 \
  --output=autoresearch_logs/%x_equal_%j.out \
  .venv/bin/python train.py --config "$EQUAL_CFG" &
PID_EQUAL=$!

status=0
if ! wait "$PID_LEARNED"; then
  status=1
fi
if ! wait "$PID_EQUAL"; then
  status=1
fi

exit "$status"
