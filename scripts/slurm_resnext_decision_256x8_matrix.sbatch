#!/bin/bash
#SBATCH --job-name=ankle-rx256x8-matrix
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PHASE="${PHASE:-formal}"
LANES="${LANES:-l0_equal l1_learned}"
SEEDS="${SEEDS:-42 123 456}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
CPUS_PER_RUN="${CPUS_PER_RUN:-12}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

echo "hostname=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PHASE=${PHASE}"
echo "LANES=${LANES}"
echo "SEEDS=${SEEDS}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

.venv/bin/python -c "import os, sys, torch; want=int(os.environ['MAX_PARALLEL']); need=float(os.environ['MIN_FREE_GB']); count=torch.cuda.device_count(); print(f'visible_gpus={count}'); [print(i, torch.cuda.get_device_name(i), f'free_gb={torch.cuda.mem_get_info(i)[0]/1024**3:.2f}') for i in range(count)]; sys.exit(0 if count >= want and min(torch.cuda.mem_get_info(i)[0] for i in range(count)) >= need * 1024**3 else 2)"

CONFIGS=()
for lane in ${LANES}; do
  for seed in ${SEEDS}; do
    CONFIGS+=("configs/generated_resnext_decision_256x8_matrix/${PHASE}/cmp_resnext_decision_256x8_${lane}_${PHASE}_s${seed}.yaml")
  done
done

.venv/bin/python scripts/backbone_preflight.py --config "${CONFIGS[@]}"

run_config() {
  local lane="$1"
  local seed="$2"
  local cfg="configs/generated_resnext_decision_256x8_matrix/${PHASE}/cmp_resnext_decision_256x8_${lane}_${PHASE}_s${seed}.yaml"
  echo "launch lane=${lane} seed=${seed} cfg=${cfg}"
  srun --exclusive -N1 -n1 --gres=gpu:1 -c"${CPUS_PER_RUN}" \
    --output="autoresearch_logs/%x_${lane}_s${seed}_%j.out" \
    .venv/bin/python scripts/run_train_with_config_env.py --config "${cfg}" &
  PIDS+=("$!")
}

drain_wave() {
  local status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  PIDS=()
  return "${status}"
}

PIDS=()
ACTIVE=0
for lane in ${LANES}; do
  for seed in ${SEEDS}; do
    run_config "${lane}" "${seed}"
    ACTIVE=$((ACTIVE + 1))
    if [ "${ACTIVE}" -ge "${MAX_PARALLEL}" ]; then
      drain_wave
      ACTIVE=0
    fi
  done
done

if [ "${#PIDS[@]}" -gt 0 ]; then
  drain_wave
fi
