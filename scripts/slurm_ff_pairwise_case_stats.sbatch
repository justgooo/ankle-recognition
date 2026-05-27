#!/bin/bash
#SBATCH --job-name=ankle-ff-pairstats
#SBATCH --partition=PA100q,PH100q,NH100q,NA100q,RTXA6Kq,HPCAIq,V100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --chdir=/dataset/HH/ankle-ct
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

mkdir -p autoresearch_logs

echo "jobid=${SLURM_JOB_ID}"
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
.venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

.venv/bin/python scripts/report_ff_pairwise_case_stats.py \
  --num-workers 4 \
  --bootstrap-samples 5000
