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
: "${ACTIVE_VIEW_INDEX:?ACTIVE_VIEW_INDEX must be set to 0, 1, or 2}"

export ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER="${ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER:-1}"
case "${ACTIVE_VIEW_INDEX}" in
  0)
    export ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK="1,0,0"
    ;;
  1)
    export ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK="0,1,0"
    ;;
  2)
    export ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK="0,0,1"
    ;;
  *)
    echo "ACTIVE_VIEW_INDEX must be 0, 1, or 2; got ${ACTIVE_VIEW_INDEX}" >&2
    exit 1
    ;;
esac

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "ACTIVE_VIEW_INDEX=${ACTIVE_VIEW_INDEX}"
echo "ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER=${ANKLE_DISABLE_FUSION_CROSS_VIEW_MIXER}"
echo "ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK=${ANKLE_DECISION_FORCE_ACTIVE_VIEW_MASK}"

.venv/bin/python train.py --config "${CONFIG_PATH}"
