<#
.SYNOPSIS
    Autoresearch loop runner for Codex CLI.
.DESCRIPTION
    Starts one fresh Codex exec session per iteration so each experiment runs
    with a clean context window.
.EXAMPLE
    .\autoresearch_loop.ps1 -MaxIterations 20 -CooldownSeconds 30
#>

[CmdletBinding()]
param(
    [int]$MaxIterations = 50,
    [int]$CooldownSeconds = 30
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $PSCommandPath
$LogDir = Join-Path $ScriptDir "autoresearch_logs"
$CompletedIterations = 0
$CodexWorkDir = $ScriptDir
$CodexCommand = $null
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom

try {
    cmd /c chcp 65001 > $null
} catch {
    Write-Warning "Failed to switch console code page to UTF-8: $_"
}

if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "codex CLI was not found in PATH."
}

$CodexCmdPath = Join-Path $env:APPDATA "npm\codex.cmd"
if (Test-Path -LiteralPath $CodexCmdPath) {
    $CodexCommand = $CodexCmdPath
} else {
    $CodexResolved = Get-Command codex.cmd -ErrorAction SilentlyContinue
    if ($CodexResolved) {
        $CodexCommand = $CodexResolved.Source
    } else {
        $CodexResolved = Get-Command codex -ErrorAction Stop
        $CodexCommand = $CodexResolved.Source
    }
}

if ($ScriptDir -match '[^\u0000-\u007F]') {
    $AliasRoot = Join-Path $env:TEMP "codex-workspaces"
    $AliasDir = Join-Path $AliasRoot "ankle-ct-codex"

    if (-not (Test-Path -LiteralPath $AliasRoot)) {
        New-Item -ItemType Directory -Path $AliasRoot | Out-Null
    }

    if (-not (Test-Path -LiteralPath $AliasDir)) {
        New-Item -ItemType Junction -Path $AliasDir -Target $ScriptDir | Out-Null
    }

    $AliasItem = Get-Item -LiteralPath $AliasDir
    if ($AliasItem.LinkType -ne "Junction") {
        throw "ASCII workspace alias exists but is not a junction: $AliasDir"
    }

    $CodexWorkDir = $AliasDir
}

$Prompt = @'
You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY.

YOUR TASK FOR THIS SESSION (do exactly ONE experiment):

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
3. Pick the HIGHEST PRIORITY uncompleted experiment from backlog.md
4. Execute exactly ONE experiment:
   a. Make code changes within the allowed scope (see program.md)
   b. Git commit the changes before training
   c. Run proxy training: .\.venv\Scripts\python.exe train.py --config configs/autoresearch_proxy.yaml > run.log 2>&1
   d. If crash: handle per program.md crash rules
   e. If success: run threshold evaluation
   f. Compare with current best (zero-miss val_acc)
   g. Record results in results.tsv
   h. Update backlog.md (move experiment to completed, update best record if keep)
5. Output a final summary line: EXPERIMENT_DONE: <status> | <description> | no_miss_val_acc=<value>

CRITICAL CONSTRAINTS:
- Use .\.venv\Scripts\python.exe for ALL Python commands (never system python)
- Shell is PowerShell on Windows
- Follow timeout rules in program.md
- Do NOT ask for human input - decide autonomously
- Read logs with: Get-Content run.log -Tail 30 (never read the whole log)
- When reading backlog.md, program.md, AGENTS.md, results.tsv, or any UTF-8 text file, ALWAYS use explicit UTF-8 decoding:
  * PowerShell: Get-Content -Encoding UTF8 ...
  * or Python: Path(...).read_text(encoding="utf-8")
- Never use bare Get-Content without -Encoding UTF8 on markdown or TSV files
'@

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Autoresearch Loop - Ankle CT Classifier" -ForegroundColor Cyan
Write-Host (" Max iterations: {0}" -f $MaxIterations) -ForegroundColor Cyan
Write-Host (" Cooldown: {0}s between experiments" -f $CooldownSeconds) -ForegroundColor Cyan
Write-Host (" Codex workdir: {0}" -f $CodexWorkDir) -ForegroundColor Cyan
Write-Host (" Codex command: {0}" -f $CodexCommand) -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

for ($i = 1; $i -le $MaxIterations; $i++) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir ("run_{0}_{1}.log" -f $i, $Timestamp)
    $LastMessageFile = Join-Path $LogDir ("run_{0}_{1}.last.txt" -f $i, $Timestamp)
    $DisplayLogFile = ".\autoresearch_logs\" + (Split-Path $LogFile -Leaf)

    Write-Host ""
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Write-Host (" Iteration {0} / {1}" -f $i, $MaxIterations) -ForegroundColor Cyan
    Write-Host (" {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Cyan

    $FreeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    Write-Host ("  Disk free: {0} GB" -f $FreeGB) -ForegroundColor Gray
    if ($FreeGB -lt 10) {
        Write-Host "  Less than 10 GB free. Stopping loop." -ForegroundColor Red
        break
    }

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $GpuCheck = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $GpuCheck) {
            Write-Host ("  GPU memory used: {0} MB" -f (($GpuCheck | Select-Object -First 1).Trim())) -ForegroundColor Gray
        }
    }

    Write-Host "  Starting Codex session..." -ForegroundColor Yellow
    Write-Host ("  Session log: {0}" -f $DisplayLogFile) -ForegroundColor Gray
    $StartTime = Get-Date

    try {
        $Prompt | & $CodexCommand exec `
            --dangerously-bypass-approvals-and-sandbox `
            --color never `
            -C $CodexWorkDir `
            -o $LastMessageFile `
            - *> $LogFile

        $ExitCode = $LASTEXITCODE
    } catch {
        Write-Host ("  Codex invocation failed: {0}" -f $_) -ForegroundColor Red
        ("ERROR: {0}" -f $_) | Out-File -FilePath $LogFile -Append -Encoding utf8
        $ExitCode = 1
    }

    $CompletedIterations++
    $DurationMin = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)

    if ($ExitCode -eq 0) {
        Write-Host ("  Iteration {0} completed in {1} min" -f $i, $DurationMin) -ForegroundColor Green
        if (Test-Path -LiteralPath $LastMessageFile) {
            $LastLine = Get-Content -Path $LastMessageFile -Encoding UTF8 | Select-Object -Last 1
            if ($LastLine) {
                Write-Host ("  Final summary: {0}" -f $LastLine) -ForegroundColor Gray
            }
        }
    } else {
        Write-Host ("  Iteration {0} failed (exit={1}) after {2} min" -f $i, $ExitCode, $DurationMin) -ForegroundColor Yellow
    }

    $RunsDir = Join-Path $ScriptDir "runs"
    if (Test-Path -LiteralPath $RunsDir) {
        $ChkptFiles = Get-ChildItem -Path $RunsDir -Filter "*.pt" -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne "best.pt" }

        if ($ChkptFiles) {
            Write-Host "  Cleaning up non-best checkpoints..." -ForegroundColor Gray
            $ChkptFiles | Remove-Item -Force
        }
    }

    if ($i -lt $MaxIterations) {
        Write-Host ("  Cooling down {0}s..." -f $CooldownSeconds) -ForegroundColor Gray
        Start-Sleep -Seconds $CooldownSeconds
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host (" Loop finished: {0} iteration(s) attempted" -f $CompletedIterations) -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
