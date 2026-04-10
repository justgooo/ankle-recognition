# ============================================================
# run_paper_validation.ps1 — 论文就绪验证全流程
# ============================================================
# 按顺序执行:
#   1. Multi-seed 验证（freeze=3 最优配方 x 3 seeds，proxy）
#   2. 三种融合方式 formal 对比（Decision / Feature / Attention）
#
# 预计总耗时：3 x 30min (proxy) + 3 x 120min (formal) ≈ 7.5 小时
# ============================================================

$ErrorActionPreference = "Stop"
$PYTHON = ".\.venv\Scripts\python.exe"
$TIMEOUT_PROXY_MS  = 3600000   # 60 min
$TIMEOUT_FORMAL_MS = 10800000  # 180 min

# Timestamp helper
function TS { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }

# ============================================================
# Helper: run one experiment (train + threshold eval)
# ============================================================
function Run-Experiment {
    param(
        [string]$Name,
        [string]$ConfigPath,
        [int]$TimeoutMs
    )

    $runDir = (Get-Content $ConfigPath | Select-String "^output_dir:" | ForEach-Object { ($_ -split ":\s*", 2)[1].Trim() })

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "[$(TS)] START: $Name" -ForegroundColor Cyan
    Write-Host "  Config:  $ConfigPath" -ForegroundColor Gray
    Write-Host "  RunDir:  $runDir" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan

    # --- Train ---
    $logFile = "$Name.log"
    $errFile = "$Name.err"
    $proc = Start-Process $PYTHON -ArgumentList "train.py","--config","$ConfigPath" `
        -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
        -PassThru -NoNewWindow
    if (-not $proc.WaitForExit($TimeoutMs)) {
        $proc.Kill()
        Write-Host "[$(TS)] TIMEOUT: $Name (killed after $($TimeoutMs/60000) min)" -ForegroundColor Red
        return
    }
    if ($proc.ExitCode -ne 0) {
        Write-Host "[$(TS)] CRASH: $Name (exit code $($proc.ExitCode))" -ForegroundColor Red
        Get-Content $logFile -Tail 20
        return
    }
    Write-Host "[$(TS)] Training done: $Name" -ForegroundColor Green

    # --- Threshold evaluation ---
    & $PYTHON tools/evaluate_threshold.py --run_dir $runDir --config $ConfigPath
    Write-Host "[$(TS)] Threshold eval done: $Name" -ForegroundColor Green

    # --- Print summary ---
    $summaryPath = "$runDir/summary.json"
    $thresholdPath = "$runDir/threshold_eval.json"
    if (Test-Path $summaryPath)   { Write-Host "  summary.json:" -ForegroundColor Yellow; Get-Content $summaryPath | ConvertFrom-Json | Select-Object -ExpandProperty best_val | Format-Table }
    if (Test-Path $thresholdPath) { Write-Host "  threshold_eval.json:" -ForegroundColor Yellow; Get-Content $thresholdPath -Tail 5 }

    Write-Host "[$(TS)] FINISHED: $Name" -ForegroundColor Green
}


# ============================================================
# Phase 1: Multi-seed validation (3 x proxy, ~30 min each)
# ============================================================
Write-Host "`n" -NoNewline
Write-Host "################################################################" -ForegroundColor Magenta
Write-Host "#  PHASE 1: Multi-seed validation (VR-MS-01~03, freeze=3)     #" -ForegroundColor Magenta
Write-Host "################################################################" -ForegroundColor Magenta

Run-Experiment -Name "VR-MS-01_seed42"  -ConfigPath "configs/multiseed_s42.yaml"  -TimeoutMs $TIMEOUT_PROXY_MS
Run-Experiment -Name "VR-MS-02_seed123" -ConfigPath "configs/multiseed_s123.yaml" -TimeoutMs $TIMEOUT_PROXY_MS
Run-Experiment -Name "VR-MS-03_seed456" -ConfigPath "configs/multiseed_s456.yaml" -TimeoutMs $TIMEOUT_PROXY_MS

# --- Multi-seed summary ---
Write-Host "`n================================================================" -ForegroundColor Magenta
Write-Host "MULTI-SEED SUMMARY" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta

$seeds = @("42", "123", "456")
$runDirs = @("runs/multiseed_s42", "runs/multiseed_s123", "runs/multiseed_s456")
$accs = @()
$spes = @()
$aucs = @()

for ($i = 0; $i -lt 3; $i++) {
    $tePath = "$($runDirs[$i])/threshold_eval.json"
    $smPath = "$($runDirs[$i])/summary.json"
    if ((Test-Path $tePath) -and (Test-Path $smPath)) {
        $te = Get-Content $tePath | ConvertFrom-Json
        $sm = Get-Content $smPath | ConvertFrom-Json
        $acc = $te.val.accuracy
        $spe = $te.val.specificity
        $auc = $sm.best_val.auc
        $accs += $acc
        $spes += $spe
        $aucs += $auc
        Write-Host "  seed=$($seeds[$i]): no_miss_val_acc=$([math]::Round($acc,4))  no_miss_val_spe=$([math]::Round($spe,4))  val_AUC=$([math]::Round($auc,4))"
    } else {
        Write-Host "  seed=$($seeds[$i]): MISSING RESULTS" -ForegroundColor Red
    }
}

if ($accs.Count -eq 3) {
    $meanAcc = ($accs | Measure-Object -Average).Average
    $stdAcc  = [math]::Sqrt((($accs | ForEach-Object { ($_ - $meanAcc) * ($_ - $meanAcc) }) | Measure-Object -Sum).Sum / 3)
    $meanSpe = ($spes | Measure-Object -Average).Average
    $stdSpe  = [math]::Sqrt((($spes | ForEach-Object { ($_ - $meanSpe) * ($_ - $meanSpe) }) | Measure-Object -Sum).Sum / 3)
    $meanAuc = ($aucs | Measure-Object -Average).Average
    $stdAuc  = [math]::Sqrt((($aucs | ForEach-Object { ($_ - $meanAuc) * ($_ - $meanAuc) }) | Measure-Object -Sum).Sum / 3)

    Write-Host "`n  === 3-SEED STATISTICS ===" -ForegroundColor Yellow
    Write-Host "  no_miss_val_acc: $([math]::Round($meanAcc,4)) +/- $([math]::Round($stdAcc,4))"
    Write-Host "  no_miss_val_spe: $([math]::Round($meanSpe,4)) +/- $([math]::Round($stdSpe,4))"
    Write-Host "  val_AUC:         $([math]::Round($meanAuc,4)) +/- $([math]::Round($stdAuc,4))"
}


# ============================================================
# Phase 2: Three-fusion formal comparison (~120 min each)
# ============================================================
Write-Host "`n" -NoNewline
Write-Host "################################################################" -ForegroundColor Magenta
Write-Host "#  PHASE 2: Three-fusion formal comparison                    #" -ForegroundColor Magenta
Write-Host "################################################################" -ForegroundColor Magenta

Run-Experiment -Name "Formal_Decision"  -ConfigPath "configs/formal_decision.yaml"  -TimeoutMs $TIMEOUT_FORMAL_MS
Run-Experiment -Name "Formal_Feature"   -ConfigPath "configs/formal_feature.yaml"   -TimeoutMs $TIMEOUT_FORMAL_MS
Run-Experiment -Name "Formal_Attention" -ConfigPath "configs/formal_attention.yaml" -TimeoutMs $TIMEOUT_FORMAL_MS

# --- Formal comparison summary ---
Write-Host "`n================================================================" -ForegroundColor Magenta
Write-Host "FORMAL COMPARISON SUMMARY" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta

$formalNames = @("Decision", "Feature", "Attention")
$formalDirs  = @("runs/formal_decision", "runs/formal_feature", "runs/formal_attention")

for ($i = 0; $i -lt 3; $i++) {
    $tePath = "$($formalDirs[$i])/threshold_eval.json"
    $smPath = "$($formalDirs[$i])/summary.json"
    if ((Test-Path $tePath) -and (Test-Path $smPath)) {
        $te = Get-Content $tePath | ConvertFrom-Json
        $sm = Get-Content $smPath | ConvertFrom-Json
        $acc = $te.val.accuracy
        $spe = $te.val.specificity
        $auc = $sm.best_val.auc
        # Test set metrics
        $testAcc = $te.test.accuracy
        $testSpe = $te.test.specificity
        $testSen = $te.test.sensitivity
        Write-Host "  $($formalNames[$i]) Fusion:"
        Write-Host "    Val:  no_miss_acc=$([math]::Round($acc,4))  no_miss_spe=$([math]::Round($spe,4))  AUC=$([math]::Round($auc,4))"
        Write-Host "    Test: acc=$([math]::Round($testAcc,4))  spe=$([math]::Round($testSpe,4))  sen=$([math]::Round($testSen,4))"
    } else {
        Write-Host "  $($formalNames[$i]) Fusion: MISSING RESULTS" -ForegroundColor Red
    }
}


# ============================================================
# DONE
# ============================================================
Write-Host "`n" -NoNewline
Write-Host "################################################################" -ForegroundColor Green
Write-Host "#  ALL PAPER VALIDATION EXPERIMENTS COMPLETE                   #" -ForegroundColor Green
Write-Host "#  $(TS)                                         #" -ForegroundColor Green
Write-Host "################################################################" -ForegroundColor Green
