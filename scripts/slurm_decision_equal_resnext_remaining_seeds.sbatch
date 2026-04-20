#!/bin/bash
#SBATCH --job-name=ankle-rx-eqrem
#SBATCH --partition=V100q
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

SEEDS="${SEEDS:-123 456}"

echo "hostname=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SEEDS=${SEEDS}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

.venv/bin/python -c "import sys, torch; count=torch.cuda.device_count(); print(f'visible_gpus={count}'); [print(i, torch.cuda.get_device_name(i), f'free_gb={torch.cuda.mem_get_info(i)[0]/1024**3:.2f}') for i in range(count)]; sys.exit(0 if count >= 3 and min(torch.cuda.mem_get_info(i)[0] for i in range(count)) >= 20 * 1024**3 else 2)"

CONFIGS=()
for seed in ${SEEDS}; do
  CONFIGS+=("configs/cmp_decision_equal_resnext_learned_512x16_e20_s${seed}.yaml")
  CONFIGS+=("configs/cmp_decision_equal_resnext_equal_512x16_e20_s${seed}.yaml")
done

.venv/bin/python scripts/backbone_preflight.py --config "${CONFIGS[@]}"

run_pair() {
  local seed="$1"
  local learned_cfg="configs/cmp_decision_equal_resnext_learned_512x16_e20_s${seed}.yaml"
  local equal_cfg="configs/cmp_decision_equal_resnext_equal_512x16_e20_s${seed}.yaml"

  echo "starting seed=${seed}"

  srun --exclusive -N1 -n1 --gres=gpu:1 -c6 \
    --output="autoresearch_logs/%x_s${seed}_learned_%j.out" \
    .venv/bin/python train.py --config "${learned_cfg}" &
  local pid_learned=$!

  srun --exclusive -N1 -n1 --gres=gpu:1 -c6 \
    --output="autoresearch_logs/%x_s${seed}_equal_%j.out" \
    .venv/bin/python train.py --config "${equal_cfg}" &
  local pid_equal=$!

  local status=0
  if ! wait "${pid_learned}"; then
    status=1
  fi
  if ! wait "${pid_equal}"; then
    status=1
  fi
  return "${status}"
}

for seed in ${SEEDS}; do
  run_pair "${seed}"
done
