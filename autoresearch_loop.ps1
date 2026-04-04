<#
.SYNOPSIS
    Autoresearch loop runner for Codex CLI.
.DESCRIPTION
    Starts one fresh Codex exec session per iteration so each experiment runs
    with a clean context window.
.EXAMPLE
    .\autoresearch_loop.ps1 -MaxIterations 20 -CooldownSeconds 30
.EXAMPLE
    .\autoresearch_loop.ps1 -RerunPlanNumbers 2,6 -RunsPerPlan 8 -KeepNoMissAccThreshold 0.75 -ImplementationPlanPath "C:\Users\xxx\.gemini\antigravity\brain\547f61e5-499e-48f5-b53c-155cffb49664\implementation_plan.md.resolved"
#>

[CmdletBinding()]
param(
    [int]$MaxIterations = 50,
    [int]$CooldownSeconds = 30,
    [string]$OpenAIBaseUrl,
    [string]$OpenAIApiKey,
    [string]$OpenAIModel,
    [ValidateSet("openai-completions", "chat", "chat-completions", "openai-responses", "responses")]
    [string]$OpenAIApi = "openai-completions",
    [int]$OpenAIFailureThreshold = 5,
    [string]$ImplementationPlanPath,
    [int[]]$RerunPlanNumbers = @(),
    [ValidateRange(1, 100)]
    [int]$RunsPerPlan = 8,
    [ValidateRange(0, 1)]
    [double]$KeepNoMissAccThreshold = 0.75
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $PSCommandPath
$LogDir = Join-Path $ScriptDir "autoresearch_logs"
$CompletedIterations = 0
$CodexWorkDir = $ScriptDir
$CodexCommand = $null
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$PrimaryProviderName = "autoresearch_openai"
$PrimaryEnvKeyName = "AUTORESEARCH_OPENAI_API_KEY"
$OpenAIFailureStreak = 0
$FallbackToCodex = $false
$UseOpenAIPrimary = $false
$ResolvedOpenAIWireApi = $null
$ResolvedImplementationPlanPath = $null
$TargetedRerunMode = $false
$KeepThresholdText = $KeepNoMissAccThreshold.ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture)

[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom

function Resolve-CodexWireApi {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ApiMode
    )

    switch ($ApiMode.ToLowerInvariant()) {
        "openai-completions" { return "chat" }
        "chat" { return "chat" }
        "chat-completions" { return "chat" }
        "openai-responses" { return "responses" }
        "responses" { return "responses" }
        default {
            throw "Unsupported OpenAI api mode: $ApiMode"
        }
    }
}

function Get-UniquePlanNumbers {
    param(
        [int[]]$PlanNumbers
    )

    $Seen = @{}
    $Ordered = New-Object System.Collections.Generic.List[int]

    foreach ($PlanNumber in $PlanNumbers) {
        if (-not $Seen.ContainsKey($PlanNumber)) {
            $Seen[$PlanNumber] = $true
            [void]$Ordered.Add($PlanNumber)
        }
    }

    return $Ordered.ToArray()
}

function Get-PlanDisplayName {
    param(
        [Parameter(Mandatory = $true)]
        [int]$PlanNumber
    )

    switch ($PlanNumber) {
        2 { return "方案 2（View Reliability Gating）" }
        6 { return "方案 6（Asymmetric Safety-Biased Fusion）" }
        default { return ("方案 {0}" -f $PlanNumber) }
    }
}

function Format-PlanList {
    param(
        [int[]]$PlanNumbers
    )

    $Items = @($PlanNumbers | ForEach-Object { Get-PlanDisplayName -PlanNumber $_ })

    if ($Items.Count -eq 0) {
        return ""
    }

    if ($Items.Count -eq 1) {
        return $Items[0]
    }

    if ($Items.Count -eq 2) {
        return ("{0} 和 {1}" -f $Items[0], $Items[1])
    }

    return ((($Items | Select-Object -First ($Items.Count - 1)) -join "、") + " 和 " + $Items[-1])
}

function Get-TargetedPlanDefinitions {
    param(
        [int[]]$PlanNumbers
    )

    $Lines = foreach ($PlanNumber in $PlanNumbers) {
        switch ($PlanNumber) {
            2 { "- 方案 2 = View Reliability Gating（动态视角可靠度门控）" }
            6 { "- 方案 6 = Asymmetric Safety-Biased Fusion（非对称安全融合）" }
            default { "- 方案 $PlanNumber = use the implementation defined in the implementation plan / repository history" }
        }
    }

    return ($Lines -join "`n")
}

function Get-SessionPrompt {
    param(
        [switch]$TargetedRerunMode,
        [string]$ImplementationPlanPath,
        [int[]]$PlanNumbers,
        [int]$RunsPerPlan,
        [string]$KeepThresholdText
    )

    if ($TargetedRerunMode) {
        $PlanList = Format-PlanList -PlanNumbers $PlanNumbers
        $PlanDefinitions = Get-TargetedPlanDefinitions -PlanNumbers $PlanNumbers
        $RerunIds = @($PlanNumbers | ForEach-Object { "R{0}-01 ... R{0}-{1:D2}" -f $_, $RunsPerPlan }) -join "；"

        $ImplementationPlanInstruction = @"
3. Also read this implementation plan with explicit UTF-8 decoding:
   - $ImplementationPlanPath
"@

        if ([string]::IsNullOrWhiteSpace($ImplementationPlanPath)) {
            $ImplementationPlanInstruction = @"
3. If the repository does not already make the targeted architecture obvious, recover the exact plan details from repository history before editing.
"@
        }

        return @"
You are an autonomous ML researcher. Follow the protocol in program.md EXACTLY, except that this user-directed rerun campaign overrides the normal backlog priority.

YOUR TASK FOR THIS SESSION (do exactly ONE experiment):

1. Read backlog.md - understand current best results, agent state, and priorities
2. Read program.md - understand the full experiment protocol
$ImplementationPlanInstruction
4. Focus ONLY on this targeted rerun campaign: rerun $PlanList, exactly $RunsPerPlan proxy runs per plan.
5. This targeted rerun campaign supersedes the normal "pick the highest priority uncompleted experiment from backlog.md" rule. If backlog.md does not already contain a dedicated highest-priority rerun section for this campaign, insert one above the current stage-5 queue without deleting the existing history.
6. Use stable rerun IDs so fresh sessions can resume deterministically: $RerunIds
7. Work through the rerun queue in the order provided above. Do NOT invent additional experiment families.
8. Execute exactly ONE unfinished rerun from this campaign:
   a. Apply the correct plan-specific implementation for the selected rerun:
$PlanDefinitions
   b. Make code/config changes within the allowed scope (see program.md)
   c. Git commit the changes before training
   d. Run proxy training: .\.venv\Scripts\python.exe train.py --config configs/autoresearch_proxy.yaml > run.log 2>&1
   e. If crash: handle per program.md crash rules
   f. If success: run threshold evaluation
   g. For THIS campaign, use threshold_eval.json as the source of truth and keep a run iff no_miss_val_acc > $KeepThresholdText at the zero-miss threshold. Do NOT require beating the global best 0.787. If no_miss_val_acc <= $KeepThresholdText, mark discard.
   h. Record results in results.tsv
   i. Update backlog.md (rerun progress, keep/discard status, Agent 状态 table, next unfinished rerun)
9. If all planned reruns are already complete, do NOT invent new work. Output this exact final line and exit cleanly:
   CAMPAIGN_COMPLETE: rerun queue exhausted | no action taken
10. Otherwise output a final summary line:
   EXPERIMENT_DONE: <status> | <description> | no_miss_val_acc=<value>

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
"@
    }

    return @'
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
}

function Get-CodexExecArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WorkDir,
        [Parameter(Mandatory = $true)]
        [string]$LastMessageFile,
        [switch]$UseOpenAIProvider,
        [string]$ProviderName,
        [string]$BaseUrl,
        [string]$EnvKeyName,
        [string]$Model,
        [string]$WireApi
    )

    $Args = @(
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--color", "never",
        "-C", $WorkDir,
        "-o", $LastMessageFile
    )

    if ($UseOpenAIProvider) {
        $Args += @(
            "-c", ('model_provider="{0}"' -f $ProviderName),
            "-c", ('model="{0}"' -f $Model),
            "-c", ('model_providers.{0}.name="OpenAI Primary"' -f $ProviderName),
            "-c", ('model_providers.{0}.base_url="{1}"' -f $ProviderName, $BaseUrl),
            "-c", ('model_providers.{0}.env_key="{1}"' -f $ProviderName, $EnvKeyName),
            "-c", ('model_providers.{0}.wire_api="{1}"' -f $ProviderName, $WireApi)
        )
    }

    return $Args
}

function Test-OpenAIProviderFailure {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $false
    }

    $LogTail = Get-Content -Path $LogPath -Encoding UTF8 -Tail 200 -ErrorAction SilentlyContinue
    if (-not $LogTail) {
        return $false
    }

    $LogText = ($LogTail -join "`n").ToLowerInvariant()
    $FailurePatterns = @(
        '\bapi key\b',
        '\binvalid api key\b',
        '\bincorrect api key\b',
        '\bauthentication\b',
        '\bunauthorized\b',
        '\bforbidden\b',
        '\b(401|403|429)\b',
        '\brate limit\b',
        '\bconnection refused\b',
        '\btime(?:d)? out\b',
        '\beconnrefused\b',
        '\benotfound\b',
        '\bdns\b',
        '\btls\b',
        '\bssl\b',
        '\bcertificate\b',
        'failed to send request',
        '\bmodel provider\b',
        '\bprovider error\b',
        'stream disconnected',
        'could not resolve host'
    )

    foreach ($Pattern in $FailurePatterns) {
        if ([regex]::IsMatch($LogText, $Pattern)) {
            return $true
        }
    }

    return $false
}

function Invoke-CodexExec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CodexBinary,
        [Parameter(Mandatory = $true)]
        [string]$PromptText,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$LogPath,
        [string]$OpenAIEnvKeyName,
        [string]$OpenAIEnvValue
    )

    $PreviousApiKey = $null
    $HadPreviousApiKey = $false

    if ($OpenAIEnvKeyName) {
        $PreviousApiKey = [Environment]::GetEnvironmentVariable($OpenAIEnvKeyName, "Process")
        $HadPreviousApiKey = $null -ne $PreviousApiKey
        [Environment]::SetEnvironmentVariable($OpenAIEnvKeyName, $OpenAIEnvValue, "Process")
    }

    try {
        $PromptText | & $CodexBinary @Arguments *> $LogPath
        return $LASTEXITCODE
    } finally {
        if ($OpenAIEnvKeyName) {
            if ($HadPreviousApiKey) {
                [Environment]::SetEnvironmentVariable($OpenAIEnvKeyName, $PreviousApiKey, "Process")
            } else {
                [Environment]::SetEnvironmentVariable($OpenAIEnvKeyName, $null, "Process")
            }
        }
    }
}

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

$OpenAIInputs = @($OpenAIBaseUrl, $OpenAIApiKey, $OpenAIModel) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
if ($OpenAIInputs.Count -gt 0 -and $OpenAIInputs.Count -lt 3) {
    throw "OpenAI fallback mode requires OpenAIBaseUrl, OpenAIApiKey, and OpenAIModel together."
}

if ($OpenAIInputs.Count -eq 3) {
    if ($OpenAIFailureThreshold -lt 1) {
        throw "OpenAIFailureThreshold must be >= 1."
    }

    $ResolvedOpenAIWireApi = Resolve-CodexWireApi -ApiMode $OpenAIApi
    $UseOpenAIPrimary = $true
}

$RerunPlanNumbers = Get-UniquePlanNumbers -PlanNumbers $RerunPlanNumbers
$TargetedRerunMode = $RerunPlanNumbers.Count -gt 0

if (-not [string]::IsNullOrWhiteSpace($ImplementationPlanPath)) {
    $ResolvedImplementationPlanPath = (Resolve-Path -LiteralPath $ImplementationPlanPath -ErrorAction Stop).Path
}

if ($TargetedRerunMode -and -not $PSBoundParameters.ContainsKey("MaxIterations")) {
    $MaxIterations = $RerunPlanNumbers.Count * $RunsPerPlan
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

$Prompt = Get-SessionPrompt `
    -TargetedRerunMode:$TargetedRerunMode `
    -ImplementationPlanPath $ResolvedImplementationPlanPath `
    -PlanNumbers $RerunPlanNumbers `
    -RunsPerPlan $RunsPerPlan `
    -KeepThresholdText $KeepThresholdText

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Autoresearch Loop - Ankle CT Classifier" -ForegroundColor Cyan
Write-Host (" Max iterations: {0}" -f $MaxIterations) -ForegroundColor Cyan
Write-Host (" Cooldown: {0}s between experiments" -f $CooldownSeconds) -ForegroundColor Cyan
Write-Host (" Codex workdir: {0}" -f $CodexWorkDir) -ForegroundColor Cyan
Write-Host (" Codex command: {0}" -f $CodexCommand) -ForegroundColor Cyan
if ($UseOpenAIPrimary) {
    Write-Host (" OpenAI primary: {0} | model={1} | api={2} | fallback after {3} consecutive provider failures" -f $OpenAIBaseUrl, $OpenAIModel, $ResolvedOpenAIWireApi, $OpenAIFailureThreshold) -ForegroundColor Cyan
} else {
    Write-Host " OpenAI primary: disabled" -ForegroundColor Cyan
}
if ($TargetedRerunMode) {
    Write-Host (" Targeted rerun: {0}" -f (Format-PlanList -PlanNumbers $RerunPlanNumbers)) -ForegroundColor Cyan
    Write-Host (" Runs per plan: {0} | keep if no_miss_val_acc > {1}" -f $RunsPerPlan, $KeepThresholdText) -ForegroundColor Cyan
    if ($ResolvedImplementationPlanPath) {
        Write-Host (" Implementation plan: {0}" -f $ResolvedImplementationPlanPath) -ForegroundColor Cyan
    }
}
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
    $LastSummaryLine = $null
    $UsePrimaryRunnerThisIteration = $UseOpenAIPrimary -and -not $FallbackToCodex
    if ($UsePrimaryRunnerThisIteration) {
        Write-Host ("  Runner: OpenAI primary ({0}/{1} consecutive failures)" -f $OpenAIFailureStreak, $OpenAIFailureThreshold) -ForegroundColor Gray
    } elseif ($UseOpenAIPrimary) {
        Write-Host ("  Runner: Codex fallback (primary disabled after {0} consecutive provider failures)" -f $OpenAIFailureThreshold) -ForegroundColor Gray
    } else {
        Write-Host "  Runner: Codex" -ForegroundColor Gray
    }

    try {
        $CodexArgs = Get-CodexExecArguments `
            -WorkDir $CodexWorkDir `
            -LastMessageFile $LastMessageFile `
            -UseOpenAIProvider:$UsePrimaryRunnerThisIteration `
            -ProviderName $PrimaryProviderName `
            -BaseUrl $OpenAIBaseUrl `
            -EnvKeyName $PrimaryEnvKeyName `
            -Model $OpenAIModel `
            -WireApi $ResolvedOpenAIWireApi

        $InvokeParams = @{
            CodexBinary = $CodexCommand
            PromptText = $Prompt
            Arguments = $CodexArgs
            LogPath = $LogFile
        }

        if ($UsePrimaryRunnerThisIteration) {
            $InvokeParams.OpenAIEnvKeyName = $PrimaryEnvKeyName
            $InvokeParams.OpenAIEnvValue = $OpenAIApiKey
        }

        $ExitCode = Invoke-CodexExec @InvokeParams
    } catch {
        Write-Host ("  Codex invocation failed: {0}" -f $_) -ForegroundColor Red
        ("ERROR: {0}" -f $_) | Out-File -FilePath $LogFile -Append -Encoding utf8
        $ExitCode = 1
    }

    $CompletedIterations++
    $DurationMin = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)

    if ($UsePrimaryRunnerThisIteration) {
        if ($ExitCode -eq 0) {
            if ($OpenAIFailureStreak -gt 0) {
                Write-Host "  OpenAI primary recovered; resetting failure streak." -ForegroundColor Green
            }
            $OpenAIFailureStreak = 0
        } elseif (Test-OpenAIProviderFailure -LogPath $LogFile) {
            $OpenAIFailureStreak++
            Write-Host ("  OpenAI primary failure streak: {0}/{1}" -f $OpenAIFailureStreak, $OpenAIFailureThreshold) -ForegroundColor Yellow

            if ($OpenAIFailureStreak -ge $OpenAIFailureThreshold) {
                $FallbackToCodex = $true
                Write-Host ("  OpenAI primary reached {0} consecutive provider failures. Future iterations will fallback to Codex." -f $OpenAIFailureThreshold) -ForegroundColor Yellow
            }
        } else {
            Write-Host "  Primary runner failed, but it did not match API/provider failure patterns; failure streak unchanged." -ForegroundColor DarkYellow
        }
    }

    if ($ExitCode -eq 0) {
        Write-Host ("  Iteration {0} completed in {1} min" -f $i, $DurationMin) -ForegroundColor Green
        if (Test-Path -LiteralPath $LastMessageFile) {
            $LastSummaryLine = Get-Content -Path $LastMessageFile -Encoding UTF8 | Select-Object -Last 1
            if ($LastSummaryLine) {
                Write-Host ("  Final summary: {0}" -f $LastSummaryLine) -ForegroundColor Gray
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

    if ($TargetedRerunMode -and $LastSummaryLine -and $LastSummaryLine.StartsWith("CAMPAIGN_COMPLETE:")) {
        Write-Host "  Targeted rerun queue completed. Stopping loop." -ForegroundColor Green
        break
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
