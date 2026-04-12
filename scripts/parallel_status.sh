#!/usr/bin/env bash
# ============================================================
# parallel_status.sh — 双槽位训练状态监控
# ============================================================
# 显示两个训练槽位的 PID、GPU 使用率、日志尾部等信息。
#
# 用法:
#   ./scripts/parallel_status.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="/tmp/ankle_parallel"
LOG_DIR="${SCRIPT_DIR}/autoresearch_logs"

echo ""
echo "============================================================"
echo " Parallel Training Status — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ---- GPU Status ----
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv 2>/dev/null || echo "  nvidia-smi unavailable"
echo ""

# ---- Slot Status ----
for slot_id in 0 1; do
    pid_file="${PID_DIR}/slot${slot_id}.pid"
    echo "--- Slot ${slot_id} ---"

    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            # Process is alive — get CPU/MEM usage
            proc_info=$(ps -p "$pid" -o pid,pcpu,pmem,etime,args --no-headers 2>/dev/null || echo "")
            echo "  Status: RUNNING (PID=${pid})"
            if [ -n "$proc_info" ]; then
                echo "  Process: ${proc_info}"
            fi
        else
            echo "  Status: DEAD (PID=${pid} no longer running)"
        fi
    else
        echo "  Status: NOT STARTED (no PID file)"
    fi

    # Show latest log
    latest_log=$(ls -t "${LOG_DIR}/slot${slot_id}_"*.log 2>/dev/null | head -1 || true)
    if [ -n "$latest_log" ] && [ -f "$latest_log" ]; then
        log_size=$(du -h "$latest_log" | cut -f1)
        echo "  Latest log: $(basename "$latest_log") (${log_size})"
        echo "  Last 3 lines:"
        tail -3 "$latest_log" | sed 's/^/    /'
    else
        echo "  No log files found"
    fi
    echo ""
done

# ---- System Resources ----
echo "=== System Resources ==="
echo "  CPU load: $(uptime | awk -F'[, ]+' '{print $(NF-2), $(NF-1), $NF}')"
echo "  Memory:"
free -h | grep -E '(Mem|Swap)' | sed 's/^/    /'
echo ""

# ---- Disk ----
echo "  Disk:"
df -h "$SCRIPT_DIR" | sed 's/^/    /'
echo ""
echo "============================================================"
