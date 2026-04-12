#!/usr/bin/env bash
# ============================================================
# parallel_train.sh — 双 GPU 并行训练启动器
# ============================================================
# 在 GPU 0 (RTX 3090) 和 GPU 1 (RTX 4090) 上同时运行两个独立
# 的训练进程。每个进程使用独立的 output_dir，互不干扰。
#
# 用法:
#   ./scripts/parallel_train.sh [options]
#
# Options:
#   --slot0-config PATH    Slot 0 (3090) 的配置文件（默认 configs/autoresearch_proxy_slot0.yaml）
#   --slot1-config PATH    Slot 1 (4090) 的配置文件（默认 configs/autoresearch_proxy.yaml）
#   --slot0-only           只启动 slot 0
#   --slot1-only           只启动 slot 1
#   --timeout SECONDS      每个进程的超时时间（默认 3600）
#   --log-dir DIR          日志目录（默认 autoresearch_logs）
#
# 环境要求:
#   - Linux (Ubuntu)
#   - 两张 NVIDIA GPU (index 0 = 3090, index 1 = 4090)
#   - .venv/bin/python 可用
#
# CPU 策略:
#   不使用 taskset — 服务器 CPU 已被其他进程占用 ~70%，
#   让 OS 调度器自行分配。num_workers 已在配置文件中降为 1。
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----
SLOT0_CONFIG="configs/autoresearch_proxy_slot0.yaml"
SLOT1_CONFIG="configs/autoresearch_proxy.yaml"
SLOT0_ENABLED=true
SLOT1_ENABLED=true
TIMEOUT=3600
LOG_DIR="autoresearch_logs"
PID_DIR="/tmp/ankle_parallel"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slot0-config)   SLOT0_CONFIG="$2";  shift 2 ;;
        --slot1-config)   SLOT1_CONFIG="$2";  shift 2 ;;
        --slot0-only)     SLOT1_ENABLED=false; shift ;;
        --slot1-only)     SLOT0_ENABLED=false; shift ;;
        --timeout)        TIMEOUT="$2";       shift 2 ;;
        --log-dir)        LOG_DIR="$2";       shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "$PID_DIR"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# ---- Pre-flight checks ----
if ! test -f .venv/bin/python; then
    echo "ERROR: .venv/bin/python not found" >&2
    exit 1
fi

# Check GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "WARNING: Expected 2 GPUs, found ${GPU_COUNT}. Proceeding anyway..." >&2
fi

# Disk space check
FREE_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {gsub("G",""); print $4}')
echo "Disk free: ${FREE_GB} GB"
if [ "${FREE_GB}" -lt 10 ]; then
    echo "ERROR: Less than 10 GB free disk space. Aborting." >&2
    exit 1
fi

# ---- Functions ----
cleanup() {
    echo ""
    echo "Caught signal — cleaning up..."
    for pid_file in "$PID_DIR"/slot*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Killing PID $pid"
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    echo "Cleanup done."
    exit 1
}
trap cleanup SIGINT SIGTERM

launch_slot() {
    local slot_id="$1"
    local gpu_id="$2"
    local config="$3"
    local log_file="${LOG_DIR}/slot${slot_id}_${TIMESTAMP}.log"
    local pid_file="${PID_DIR}/slot${slot_id}.pid"

    echo "  Slot ${slot_id}: GPU=${gpu_id}, config=${config}"
    echo "  Slot ${slot_id}: log → ${log_file}"

    # Clean previous output if exists
    local exp_name
    exp_name=$(grep -oP '(?<=output_dir: ).*' "$config" 2>/dev/null || echo "")
    if [ -n "$exp_name" ] && [ -d "$exp_name" ]; then
        # Remove non-best checkpoints
        find "$exp_name" -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" \
        timeout "$TIMEOUT" \
        .venv/bin/python train.py --config "$config" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    echo "  Slot ${slot_id}: PID=${pid}"
}

wait_for_slot() {
    local slot_id="$1"
    local pid_file="${PID_DIR}/slot${slot_id}.pid"

    if [ ! -f "$pid_file" ]; then
        return 0
    fi

    local pid
    pid=$(cat "$pid_file")
    local exit_code=0
    wait "$pid" 2>/dev/null || exit_code=$?
    rm -f "$pid_file"

    local log_file
    log_file=$(ls -t "${LOG_DIR}/slot${slot_id}_"*.log 2>/dev/null | head -1)

    if [ "$exit_code" -eq 0 ]; then
        echo "  Slot ${slot_id}: ✅ completed successfully"
    elif [ "$exit_code" -eq 124 ]; then
        echo "  Slot ${slot_id}: ⏰ TIMEOUT after ${TIMEOUT}s"
    else
        echo "  Slot ${slot_id}: ❌ failed (exit code ${exit_code})"
    fi

    # Show last few lines of log
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "  Slot ${slot_id}: last 5 lines of log:"
        tail -5 "$log_file" | sed 's/^/    /'
    fi

    return "$exit_code"
}

# ============================================================
# Main
# ============================================================
echo ""
echo "============================================================"
echo " Parallel Training — Ankle CT Classifier"
echo " Timestamp:  ${TIMESTAMP}"
echo " Slot 0:     GPU 0 (RTX 3090) — ${SLOT0_ENABLED}"
echo " Slot 1:     GPU 1 (RTX 4090) — ${SLOT1_ENABLED}"
echo " Timeout:    ${TIMEOUT}s per slot"
echo " Workdir:    ${SCRIPT_DIR}"
echo "============================================================"
echo ""

# Show GPU status
echo "GPU status before training:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
echo ""

# Launch slots
echo "Launching training processes..."
if $SLOT0_ENABLED; then
    launch_slot 0 0 "$SLOT0_CONFIG"
fi
if $SLOT1_ENABLED; then
    launch_slot 1 1 "$SLOT1_CONFIG"
fi
echo ""

# Wait for completion
echo "Waiting for training processes to finish..."
echo "(Press Ctrl+C to abort both)"
echo ""

EXIT0=0
EXIT1=0

if $SLOT0_ENABLED; then
    wait_for_slot 0 || EXIT0=$?
fi
if $SLOT1_ENABLED; then
    wait_for_slot 1 || EXIT1=$?
fi

echo ""

# Show GPU status after
echo "GPU status after training:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true

echo ""
echo "============================================================"
echo " Results"
echo "  Slot 0: exit=${EXIT0}"
echo "  Slot 1: exit=${EXIT1}"
echo "============================================================"

# Exit with non-zero if either failed
if [ "$EXIT0" -ne 0 ] || [ "$EXIT1" -ne 0 ]; then
    exit 1
fi
