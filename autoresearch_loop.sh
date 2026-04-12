#!/usr/bin/env bash
# ============================================================
# autoresearch_loop.sh — 单 GPU 单进程 Autoresearch 循环
# ============================================================
# 在单张 GPU 上启动一个接一个的 Qoder exec 实验会话。
# 每次实验使用独立的 context window。
#
# 如需双 GPU 并行运行，请使用 autoresearch_parallel_loop.sh。
#
# Usage:
#   ./autoresearch_loop.sh [options]
#
# Options:
#   --max-iterations N       Max loop iterations (default: 50)
#   --cooldown-seconds N     Cooldown between experiments (default: 30)
#   --slot N                 GPU slot to use: 0 = RTX 3090, 1 = RTX 4090 (default: 1)
# ============================================================
set -euo pipefail

# Default: slot 1 (RTX 4090)
SLOT_ID=1
GPU_NAME="RTX4090"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/autoresearch_logs"
MAX_ITERATIONS=50
COOLDOWN_SECONDS=30
COMPLETED_ITERATIONS=0
WORKFLOW_MODE="classic"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-iterations)   MAX_ITERATIONS="$2";   shift 2 ;;
        --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
        --workflow)         WORKFLOW_MODE="$2";    shift 2 ;;
        --slot)
            SLOT_ID="$2"
            if [ "$SLOT_ID" -eq 0 ]; then
                GPU_NAME="RTX3090"
            else
                GPU_NAME="RTX4090"
            fi
            shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$SLOT_ID"

# Select config based on slot
if [ "$SLOT_ID" -eq 0 ]; then
    PROXY_CONFIG="configs/autoresearch_proxy_slot0.yaml"
    FORMAL_CONFIG="configs/autoresearch_formal_slot0.yaml"
else
    PROXY_CONFIG="configs/autoresearch_proxy.yaml"
    FORMAL_CONFIG="configs/autoresearch_formal.yaml"
fi

if [[ "$WORKFLOW_MODE" != "classic" && "$WORKFLOW_MODE" != "joint" ]]; then
    echo "ERROR: --workflow must be classic or joint" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

if ! command -v qoder &>/dev/null; then
    echo "ERROR: qoder CLI was not found in PATH." >&2
    exit 1
fi

# ============================================================
# Session prompt
# ============================================================
if [[ "$WORKFLOW_MODE" == "joint" ]]; then
    read -r -d '' PROMPT <<'EOF'
You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY.

YOUR TASK FOR THIS SESSION (do exactly ONE research iteration):

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
3. Pick the HIGHEST PRIORITY uncompleted direction from backlog.md
4. Execute exactly ONE high-level AutoResearch change:
   a. Make one minimal but meaningful research change inside the allowed files
   b. Git commit the change before any training
   c. Run one baseline/smoke proxy check with train.py, compare val_acc from summary.json
   d. If the candidate is stable, run a small local Optuna study:
       CUDA_VISIBLE_DEVICES=${SLOT_ID} .venv/bin/python scripts/run_optuna_proxy.py
   e. Summarize that study:
       CUDA_VISIBLE_DEVICES=${SLOT_ID} .venv/bin/python scripts/monitor_optuna.py --study-dir runs/optuna_proxy
   f. Compare the best tuned candidate against the current keep version using val_acc from summary.json
   g. Only if improved, optionally run the deeper confirmation pass:
       CUDA_VISIBLE_DEVICES=${SLOT_ID} .venv/bin/python scripts/run_optuna_main.py --source-study-dir runs/optuna_proxy --top-k 3
   h. Record the outcome in results.tsv and update backlog.md
5. Output a final summary line:
   EXPERIMENT_DONE: <status> | <description> | val_acc=<value>

CRITICAL CONSTRAINTS:
- Use CUDA_VISIBLE_DEVICES=${SLOT_ID} .venv/bin/python for ALL Python commands (${GPU_NAME}, GPU index=${SLOT_ID})
- Shell is Bash on Ubuntu
- Do not use test-set metrics for selection
- Follow timeout rules in program.md
- Do NOT ask for human input - decide autonomously
- Read logs with: tail -30 <logfile> (never read whole logs)
- Keep the Optuna layer external; do not rewrite train.py unless absolutely necessary
EOF
else
    read -r -d '' PROMPT <<'EOF'
You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY.

YOUR TASK FOR THIS SESSION (do exactly ONE experiment):

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
3. Pick the HIGHEST PRIORITY uncompleted experiment from backlog.md
4. Execute exactly ONE experiment:
   a. Make code changes within the allowed scope (see program.md)
   b. Git commit the changes before training
   c. Run proxy training: CUDA_VISIBLE_DEVICES=${SLOT_ID} .venv/bin/python train.py --config ${PROXY_CONFIG} > run.log 2>&1
   d. If crash: handle per program.md crash rules
   e. If success: read summary.json for best_val.accuracy
   f. Compare with current best (val_acc)
   g. Record results in results.tsv
   h. Update backlog.md (move experiment to completed, update best record if keep)
5. Output a final summary line: EXPERIMENT_DONE: <status> | <description> | val_acc=<value>

CRITICAL CONSTRAINTS:
- Use CUDA_VISIBLE_DEVICES=${SLOT_ID} .venv/bin/python for ALL Python commands (${GPU_NAME}, GPU index=${SLOT_ID})
- Shell is Bash on Ubuntu
- Follow timeout rules in program.md
- Do NOT ask for human input - decide autonomously
- Read logs with: tail -30 run.log (never read the whole log)
- When reading text files use UTF-8 (Python: Path(...).read_text(encoding="utf-8"))
EOF
fi

# ============================================================
# Main loop
# ============================================================
echo ""
echo "============================================================"
echo " Autoresearch Loop - Ankle CT Classifier (single slot)"
echo " Max iterations:  ${MAX_ITERATIONS}"
echo " Cooldown:        ${COOLDOWN_SECONDS}s between experiments"
echo " Workflow:        ${WORKFLOW_MODE}"
echo " Slot:            ${SLOT_ID} (${GPU_NAME}, CUDA_VISIBLE_DEVICES=${SLOT_ID})"
echo " Proxy config:    ${PROXY_CONFIG}"
echo " Formal config:   ${FORMAL_CONFIG}"
echo " Workdir:         ${SCRIPT_DIR}"
echo "============================================================"
echo ""

for ((i=1; i<=MAX_ITERATIONS; i++)); do
    TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/run_${i}_${TIMESTAMP}.log"
    LAST_MSG_FILE="${LOG_DIR}/run_${i}_${TIMESTAMP}.last.txt"

    echo ""
    echo "----------------------------------------"
    echo " Iteration ${i} / ${MAX_ITERATIONS}"
    echo " $(date "+%Y-%m-%d %H:%M:%S")"
    echo "----------------------------------------"

    # Disk space check
    FREE_GB=$(df -BG "${SCRIPT_DIR}" | awk 'NR==2 {gsub("G",""); print $4}')
    echo "  Disk free: ${FREE_GB} GB"
    if [ "${FREE_GB}" -lt 10 ]; then
        echo "  Less than 10 GB free. Stopping loop."
        break
    fi

    # GPU memory check
    if command -v nvidia-smi &>/dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n '2p' || true)
        [ -n "$GPU_MEM" ] && echo "  GPU 1 (4090) memory used: ${GPU_MEM} MB"
    fi

    # Cleanup non-best checkpoints from previous runs
    RUNS_DIR="${SCRIPT_DIR}/runs"
    if [ -d "$RUNS_DIR" ]; then
        find "$RUNS_DIR" -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
    fi

    echo "  Starting Qoder session..."
    echo "  Session log: autoresearch_logs/$(basename "$LOG_FILE")"
    START_TIME=$(date +%s)

    EXIT_CODE=0
    printf '%s' "$PROMPT" | qoder exec \
        --dangerously-bypass-approvals-and-sandbox \
        --color never \
        -C "$SCRIPT_DIR" \
        -o "$LAST_MSG_FILE" \
        >"$LOG_FILE" 2>&1 || EXIT_CODE=$?

    COMPLETED_ITERATIONS=$((COMPLETED_ITERATIONS + 1))
    END_TIME=$(date +%s)
    DURATION_MIN=$(( (END_TIME - START_TIME) / 60 ))

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "  Iteration ${i} completed in ${DURATION_MIN} min"
        if [ -f "$LAST_MSG_FILE" ]; then
            LAST_LINE=$(tail -1 "$LAST_MSG_FILE")
            [ -n "$LAST_LINE" ] && echo "  Final summary: ${LAST_LINE}"
            # Stop if campaign is complete
            if [[ "$LAST_LINE" == CAMPAIGN_COMPLETE:* ]]; then
                echo "  Campaign complete. Stopping loop."
                break
            fi
        fi
    else
        echo "  Iteration ${i} failed (exit=${EXIT_CODE}) after ${DURATION_MIN} min"
    fi

    if [ "$i" -lt "$MAX_ITERATIONS" ]; then
        echo "  Cooling down ${COOLDOWN_SECONDS}s..."
        sleep "$COOLDOWN_SECONDS"
    fi
done

echo ""
echo "============================================================"
echo " Loop finished: ${COMPLETED_ITERATIONS} iteration(s) attempted"
echo "============================================================"
