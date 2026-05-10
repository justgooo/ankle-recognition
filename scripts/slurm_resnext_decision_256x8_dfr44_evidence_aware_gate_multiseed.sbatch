#!/bin/bash
#SBATCH --job-name=ankle-dfr44-eag
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --exclude=node20
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/run_dfr44_evidence_gate_multiseed.py
