#!/bin/bash
#SBATCH --job-name=ankle-single-view
#SBATCH --partition=V100q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=autoresearch_logs/slurm_%x_%j.out

set -euo pipefail

cd /dataset/HH/ankle-ct
mkdir -p autoresearch_logs

: "${CONFIG_PATH:?CONFIG_PATH must be set}"
: "${ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK:?ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK must be set}"

export ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER="${ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER:-1}"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=${ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER}"
echo "ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK=${ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK}"

.venv/bin/python train.py --config "${CONFIG_PATH}"
