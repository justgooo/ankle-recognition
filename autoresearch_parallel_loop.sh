#!/usr/bin/env bash
# ============================================================
# autoresearch_parallel_loop.sh — 双 GPU 并行 Autoresearch 循环
# ============================================================
# 同时在 GPU 0 (RTX 3090) 和 GPU 1 (RTX 4090) 上各启动一个
# 独立的 Qoder/Agent 实验会话，让两张卡并行消化 backlog 中的
# 同等级实验任务。
#
# 核心设计：
#   - 两个 slot 完全独立：不同 output_dir、不同 GPU、不同日志
#   - 共享 results.tsv（通过 flock 互斥写入，由 Agent 实现）
#   - 共享 backlog.md（Agent 更新时需注意并发，实际上冲突极少）
#   - 不使用 taskset（CPU 已被其他进程占用 ~70%）
#   - num_workers=1（在配置文件中设置，节省 CPU）
#
# 用法:
#   ./autoresearch_parallel_loop.sh [options]
#
# Options:
#   --max-iterations N       每个 slot 的最大迭代次数（默认 25）
#   --cooldown-seconds N     两次实验之间的冷却时间（默认 30）
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/autoresearch_logs"
PID_DIR="/tmp/ankle_parallel"
MAX_ITERATIONS=25
COOLDOWN_SECONDS=30

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-iterations)   MAX_ITERATIONS="$2";   shift 2 ;;
        --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "$PID_DIR"

if ! command -v qoder &>/dev/null; then
    echo "ERROR: qoder CLI was not found in PATH." >&2
    exit 1
fi

# ============================================================
# Session prompts — each slot gets its own GPU assignment
# ============================================================
generate_prompt() {
    local slot_id="$1"
    local gpu_id="$2"
    local gpu_name="$3"
    local proxy_config="$4"
    local formal_config="$5"

    cat <<PROMPT_EOF
You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY.

YOUR TASK FOR THIS SESSION (do exactly ONE experiment on SLOT ${slot_id}):

⚠️ CRITICAL GPU ASSIGNMENT: You are running on **Slot ${slot_id}** with **${gpu_name}** (GPU index=${gpu_id}).
- Use CUDA_VISIBLE_DEVICES=${gpu_id} for ALL Python commands
- Use proxy config: ${proxy_config}
- Use formal config: ${formal_config}
- Tag all results with [slot${slot_id}/${gpu_name}] in the description column

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
3. Pick the HIGHEST PRIORITY uncompleted experiment from backlog.md
   - If another slot is currently working on an experiment (check autoresearch_logs/), pick a DIFFERENT one
4. Execute exactly ONE experiment:
   a. Make code changes within the allowed scope (see program.md)
   b. Git commit the changes before training
   c. Run proxy training: CUDA_VISIBLE_DEVICES=${gpu_id} .venv/bin/python train.py --config ${proxy_config} > run.log 2>&1
   d. If crash: handle per program.md crash rules
   e. If success: read summary.json for best_val.accuracy
   f. Compare with current best (val_acc)
   g. Record results in results.tsv (use flock if available):
      flock -x /tmp/ankle_results.lock -c 'echo -e "LINE" >> results.tsv' 2>/dev/null || echo -e "LINE" >> results.tsv
   h. Update backlog.md (move experiment to completed, update best record if keep)
5. Output a final summary line: EXPERIMENT_DONE: <status> | <description> | val_acc=<value> | slot=${slot_id}

CRITICAL CONSTRAINTS:
- Use CUDA_VISIBLE_DEVICES=${gpu_id} .venv/bin/python for ALL Python commands
- Shell is Bash on Ubuntu
- Follow timeout rules in program.md
- Do NOT ask for human input - decide autonomously
- Read logs with: tail -30 run.log (never read the whole log)
- When reading text files use UTF-8 (Python: Path(...).read_text(encoding="utf-8"))
- Results summary.json is at: runs/$(basename "${proxy_config}" .yaml)/summary.json
PROMPT_EOF
}

# ============================================================
# Cleanup handler
# ============================================================
cleanup() {
    echo ""
    echo "Caught signal — stopping all slots..."
    for pid_file in "$PID_DIR"/loop_slot*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Killing slot loop PID $pid and its children..."
                kill -- -"$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null || kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    echo "All slots stopped."
    exit 1
}
trap cleanup SIGINT SIGTERM

# ============================================================
# Per-slot loop function (runs in background)
# ============================================================
run_slot_loop() {
    local slot_id="$1"
    local gpu_id="$2"
    local gpu_name="$3"
    local proxy_config="$4"
    local formal_config="$5"
    local completed=0

    for ((i=1; i<=MAX_ITERATIONS; i++)); do
        TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
        LOG_FILE="${LOG_DIR}/slot${slot_id}_run_${i}_${TIMESTAMP}.log"
        LAST_MSG_FILE="${LOG_DIR}/slot${slot_id}_run_${i}_${TIMESTAMP}.last.txt"

        echo "[Slot ${slot_id}] ── Iteration ${i}/${MAX_ITERATIONS} ── $(date '+%H:%M:%S') ──"

        # Disk space check
        FREE_GB=$(df -BG "${SCRIPT_DIR}" | awk 'NR==2 {gsub("G",""); print $4}')
        if [ "${FREE_GB}" -lt 10 ]; then
            echo "[Slot ${slot_id}] Less than 10 GB free. Stopping this slot."
            break
        fi

        # Clean old checkpoints
        RUNS_DIR="${SCRIPT_DIR}/runs"
        if [ -d "$RUNS_DIR" ]; then
            find "$RUNS_DIR" -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
        fi

        # Generate prompt and run
        PROMPT=$(generate_prompt "$slot_id" "$gpu_id" "$gpu_name" "$proxy_config" "$formal_config")
        START_TIME=$(date +%s)

        EXIT_CODE=0
        printf '%s' "$PROMPT" | qoder exec \
            --dangerously-bypass-approvals-and-sandbox \
            --color never \
            -C "$SCRIPT_DIR" \
            -o "$LAST_MSG_FILE" \
            >"$LOG_FILE" 2>&1 || EXIT_CODE=$?

        completed=$((completed + 1))
        END_TIME=$(date +%s)
        DURATION_MIN=$(( (END_TIME - START_TIME) / 60 ))

        if [ "$EXIT_CODE" -eq 0 ]; then
            echo "[Slot ${slot_id}] Iteration ${i} done in ${DURATION_MIN} min"
            if [ -f "$LAST_MSG_FILE" ]; then
                LAST_LINE=$(tail -1 "$LAST_MSG_FILE")
                [ -n "$LAST_LINE" ] && echo "[Slot ${slot_id}] → ${LAST_LINE}"
                if [[ "$LAST_LINE" == CAMPAIGN_COMPLETE:* ]]; then
                    echo "[Slot ${slot_id}] Campaign complete. Stopping."
                    break
                fi
            fi
        else
            echo "[Slot ${slot_id}] Iteration ${i} FAILED (exit=${EXIT_CODE}) after ${DURATION_MIN} min"
        fi

        if [ "$i" -lt "$MAX_ITERATIONS" ]; then
            echo "[Slot ${slot_id}] Cooling down ${COOLDOWN_SECONDS}s..."
            sleep "$COOLDOWN_SECONDS"
        fi
    done

    echo "[Slot ${slot_id}] Loop finished: ${completed} iteration(s)"
}

# ============================================================
# Main
# ============================================================
echo ""
echo "============================================================"
echo " Parallel Autoresearch Loop — Ankle CT Classifier"
echo " Max iterations per slot: ${MAX_ITERATIONS}"
echo " Cooldown:                ${COOLDOWN_SECONDS}s"
echo " Slot 0: GPU 0 (RTX 3090)"
echo " Slot 1: GPU 1 (RTX 4090)"
echo " Workdir: ${SCRIPT_DIR}"
echo "============================================================"
echo ""

# Show initial GPU status
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
echo ""

# Launch both slot loops in background
echo "Starting Slot 0 (RTX 3090, GPU 0)..."
run_slot_loop 0 0 "RTX3090" \
    "configs/autoresearch_proxy_slot0.yaml" \
    "configs/autoresearch_formal_slot0.yaml" &
LOOP_PID0=$!
echo "$LOOP_PID0" > "${PID_DIR}/loop_slot0.pid"

echo "Starting Slot 1 (RTX 4090, GPU 1)..."
run_slot_loop 1 1 "RTX4090" \
    "configs/autoresearch_proxy.yaml" \
    "configs/autoresearch_formal.yaml" &
LOOP_PID1=$!
echo "$LOOP_PID1" > "${PID_DIR}/loop_slot1.pid"

echo ""
echo "Both slots running. PIDs: slot0=${LOOP_PID0}, slot1=${LOOP_PID1}"
echo "Monitor with: ./scripts/parallel_status.sh"
echo "Stop with:    Ctrl+C or kill $$"
echo ""

# Wait for both to finish
EXIT0=0
EXIT1=0
wait "$LOOP_PID0" 2>/dev/null || EXIT0=$?
wait "$LOOP_PID1" 2>/dev/null || EXIT1=$?

rm -f "${PID_DIR}/loop_slot0.pid" "${PID_DIR}/loop_slot1.pid"

echo ""
echo "============================================================"
echo " Parallel loop finished"
echo "  Slot 0 exit: ${EXIT0}"
echo "  Slot 1 exit: ${EXIT1}"
echo "============================================================"
