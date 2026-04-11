#!/usr/bin/env bash
# ============================================================
# run_paper_validation.sh — 论文就绪验证全流程
# ============================================================
# 按顺序执行:
#   1. Multi-seed 验证（freeze=3 最优配方 x 3 seeds，proxy）
#   2. 三种融合方式 formal 对比（Decision / Feature / Attention）
#
# 预计总耗时：3 x 30min (proxy) + 3 x 120min (formal) ≈ 7.5 小时
# ============================================================
set -euo pipefail

PYTHON=".venv/bin/python"
TIMEOUT_PROXY_S=3600    # 60 min
TIMEOUT_FORMAL_S=10800  # 180 min

export CUDA_VISIBLE_DEVICES=1  # 优先使用 RTX 4090

ts() { date "+%Y-%m-%d %H:%M:%S"; }

# ============================================================
# Helper: run one experiment (train + threshold eval)
# ============================================================
run_experiment() {
    local name="$1"
    local config_path="$2"
    local timeout_s="$3"

    local run_dir
    run_dir=$(grep "^output_dir:" "$config_path" | awk '{print $2}' | tr -d '"' | tr -d "'")

    echo ""
    echo "========================================"
    echo "[$(ts)] START: $name"
    echo "  Config:  $config_path"
    echo "  RunDir:  $run_dir"
    echo "========================================"

    local log_file="${name}.log"
    local err_file="${name}.err"

    # --- Train ---
    if ! timeout "$timeout_s" "$PYTHON" train.py --config "$config_path" \
            >"$log_file" 2>"$err_file"; then
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "[$(ts)] TIMEOUT: $name (killed after $((timeout_s/60)) min)"
        else
            echo "[$(ts)] CRASH: $name (exit code $exit_code)"
            tail -20 "$log_file"
        fi
        return
    fi
    echo "[$(ts)] Training done: $name"

    # --- Threshold evaluation ---
    "$PYTHON" tools/evaluate_threshold.py --run_dir "$run_dir" --config "$config_path"
    echo "[$(ts)] Threshold eval done: $name"

    # --- Print summary ---
    local summary_path="${run_dir}/summary.json"
    local threshold_path="${run_dir}/threshold_eval.json"
    if [ -f "$summary_path" ]; then
        echo "  summary.json:"
        cat "$summary_path"
    fi
    if [ -f "$threshold_path" ]; then
        echo "  threshold_eval.json (last 5 lines):"
        tail -5 "$threshold_path"
    fi

    echo "[$(ts)] FINISHED: $name"
}

# ============================================================
# Phase 1: Multi-seed validation (3 x proxy, ~30 min each)
# ============================================================
echo ""
echo "################################################################"
echo "#  PHASE 1: Multi-seed validation (VR-MS-01~03, freeze=3)     #"
echo "################################################################"

run_experiment "VR-MS-01_seed42"  "configs/multiseed_s42.yaml"  "$TIMEOUT_PROXY_S"
run_experiment "VR-MS-02_seed123" "configs/multiseed_s123.yaml" "$TIMEOUT_PROXY_S"
run_experiment "VR-MS-03_seed456" "configs/multiseed_s456.yaml" "$TIMEOUT_PROXY_S"

# --- Multi-seed summary ---
echo ""
echo "================================================================"
echo "MULTI-SEED SUMMARY"
echo "================================================================"

seeds=("42" "123" "456")
run_dirs=("runs/multiseed_s42" "runs/multiseed_s123" "runs/multiseed_s456")
accs=()
spes=()
aucs=()

for i in 0 1 2; do
    te_path="${run_dirs[$i]}/threshold_eval.json"
    sm_path="${run_dirs[$i]}/summary.json"
    if [ -f "$te_path" ] && [ -f "$sm_path" ]; then
        acc=$("$PYTHON" -c "import json; d=json.load(open('$te_path')); print(d['val']['accuracy'])")
        spe=$("$PYTHON" -c "import json; d=json.load(open('$te_path')); print(d['val']['specificity'])")
        auc=$("$PYTHON" -c "import json; d=json.load(open('$sm_path')); print(d['best_val']['auc'])")
        accs+=("$acc")
        spes+=("$spe")
        aucs+=("$auc")
        echo "  seed=${seeds[$i]}: no_miss_val_acc=$acc  no_miss_val_spe=$spe  val_AUC=$auc"
    else
        echo "  seed=${seeds[$i]}: MISSING RESULTS"
    fi
done

if [ ${#accs[@]} -eq 3 ]; then
    "$PYTHON" - <<EOF
accs = [${accs[0]}, ${accs[1]}, ${accs[2]}]
spes = [${spes[0]}, ${spes[1]}, ${spes[2]}]
aucs = [${aucs[0]}, ${aucs[1]}, ${aucs[2]}]
import math
def stats(v):
    m = sum(v)/3
    s = math.sqrt(sum((x-m)**2 for x in v)/3)
    return round(m,4), round(s,4)
ma,sa = stats(accs); ms,ss = stats(spes); mu,su = stats(aucs)
print(f"\n  === 3-SEED STATISTICS ===")
print(f"  no_miss_val_acc: {ma} +/- {sa}")
print(f"  no_miss_val_spe: {ms} +/- {ss}")
print(f"  val_AUC:         {mu} +/- {su}")
EOF
fi

# ============================================================
# Phase 2: Three-fusion formal comparison (~120 min each)
# ============================================================
echo ""
echo "################################################################"
echo "#  PHASE 2: Three-fusion formal comparison                    #"
echo "################################################################"

run_experiment "Formal_Decision"  "configs/formal_decision.yaml"  "$TIMEOUT_FORMAL_S"
run_experiment "Formal_Feature"   "configs/formal_feature.yaml"   "$TIMEOUT_FORMAL_S"
run_experiment "Formal_Attention" "configs/formal_attention.yaml" "$TIMEOUT_FORMAL_S"

# --- Formal comparison summary ---
echo ""
echo "================================================================"
echo "FORMAL COMPARISON SUMMARY"
echo "================================================================"

formal_names=("Decision" "Feature" "Attention")
formal_dirs=("runs/formal_decision" "runs/formal_feature" "runs/formal_attention")

for i in 0 1 2; do
    te_path="${formal_dirs[$i]}/threshold_eval.json"
    sm_path="${formal_dirs[$i]}/summary.json"
    if [ -f "$te_path" ] && [ -f "$sm_path" ]; then
        "$PYTHON" - <<EOF
import json
te = json.load(open('$te_path'))
sm = json.load(open('$sm_path'))
name = "${formal_names[$i]}"
acc  = round(te['val']['accuracy'],    4)
spe  = round(te['val']['specificity'], 4)
auc  = round(sm['best_val']['auc'],    4)
ta   = round(te['test']['accuracy'],    4)
ts_  = round(te['test']['specificity'], 4)
tse  = round(te['test']['sensitivity'], 4)
print(f"  {name} Fusion:")
print(f"    Val:  no_miss_acc={acc}  no_miss_spe={spe}  AUC={auc}")
print(f"    Test: acc={ta}  spe={ts_}  sen={tse}")
EOF
    else
        echo "  ${formal_names[$i]} Fusion: MISSING RESULTS"
    fi
done

# ============================================================
# DONE
# ============================================================
echo ""
echo "################################################################"
echo "#  ALL PAPER VALIDATION EXPERIMENTS COMPLETE                   #"
echo "#  $(ts)                                         #"
echo "################################################################"
