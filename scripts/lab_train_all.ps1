<#
.SYNOPSIS
    Run all training steps on the lab PC.
.DESCRIPTION
    Runs centralized baselines, FL grid, export, and packs results.
    Each step is resumable and individually skippable via -Skip.
    On error, still packs whatever finished.
.PARAMETER Profile
    Override profile. Default: "lab".
.PARAMETER Skip
    Steps to skip. Valid: xgb, mlp, mlp_no_iat, fl, export, pack.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\lab_train_all.ps1
    powershell -ExecutionPolicy Bypass -File scripts\lab_train_all.ps1 -Skip xgb,mlp_no_iat
    powershell -ExecutionPolicy Bypass -File scripts\lab_train_all.ps1 -Profile laptop
#>
param(
    [string]$Profile = "lab",
    [string[]]$Skip = @()
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Start transcript
$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = Join-Path $LogDir "lab_$Timestamp.log"
Start-Transcript -Path $TranscriptPath -Append

Write-Host "=== NetShield-FL Lab Training ===" -ForegroundColor Cyan
Write-Host "Profile:    $Profile"
Write-Host "Skip:       $($Skip -join ', ')"
Write-Host "Transcript: $TranscriptPath"
Write-Host "Started:    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Set profile
$env:NETSHIELD_PROFILE = $Profile

# Activate venv
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment not found. Run lab_setup.ps1 first."
    Stop-Transcript
    exit 1
}
. $VenvActivate

$TotalStart = Get-Date
$StepTimes = @{}
$Errors = @()

function Run-Step {
    param(
        [string]$Name,
        [string]$Label,
        [scriptblock]$Action
    )
    if ($Skip -contains $Name) {
        Write-Host "`n--- [$Label] SKIPPED ---" -ForegroundColor DarkGray
        return
    }
    Write-Host "`n--- [$Label] Starting ---" -ForegroundColor Yellow
    $StepStart = Get-Date
    try {
        & $Action
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Command exited with code $LASTEXITCODE"
        }
    } catch {
        $Errors += "$Label : $_"
        Write-Host "ERROR in $Label : $_" -ForegroundColor Red
    }
    $Elapsed = (Get-Date) - $StepStart
    $StepTimes[$Name] = $Elapsed
    Write-Host "--- [$Label] Done in $($Elapsed.ToString('hh\:mm\:ss')) ---" -ForegroundColor Green
}

# (a) XGBoost full
Run-Step -Name "xgb" -Label "XGBoost Full" -Action {
    python -m netshield.models.train_centralized --model xgb --run-name xgb_full
}

# (b) MLP full
Run-Step -Name "mlp" -Label "MLP Full" -Action {
    python -m netshield.models.train_centralized --model mlp --run-name mlp_full
}

# (c) MLP without IAT
Run-Step -Name "mlp_no_iat" -Label "MLP No-IAT Full" -Action {
    python -m netshield.models.train_centralized --model mlp --drop-features iat --run-name mlp_no_iat_full
}

# (d) FL grid
Run-Step -Name "fl" -Label "FL Grid" -Action {
    python -m netshield.federated.run_experiment
}

# (e) Export production ONNX (FL already does select_best + export if profile=lab)
Run-Step -Name "export" -Label "Export Centralized MLP" -Action {
    python -m netshield.models.export_onnx --run-name mlp_full --onnx-name netshield_mlp_full --set-active production
}

# (f) Summary table
Write-Host "`n--- Summary ---" -ForegroundColor Cyan
$ResultsDir = Join-Path $RepoRoot "reports\tables"
if (Test-Path $ResultsDir) {
    Write-Host "`nCentralized results:" -ForegroundColor White
    foreach ($F in (Get-ChildItem (Join-Path $ResultsDir "centralized_*.json"))) {
        $R = Get-Content $F.FullName -Raw | ConvertFrom-Json
        Write-Host ("  {0,-30s} macro_f1={1:F4}  wall={2:F1}s  profile={3}" -f $R.run_name, $R.test_metrics.macro_f1, $R.wall_time_s, $R.profile)
    }

    $FlCsv = Join-Path $ResultsDir "fl_results.csv"
    if (Test-Path $FlCsv) {
        Write-Host "`nFL results:" -ForegroundColor White
        Import-Csv $FlCsv | ForEach-Object {
            Write-Host ("  {0,-40s} f1={1}  wall={2}s" -f $_.run_name, $_.test_macro_f1, $_.wall_time_s)
        }
    }
}

$ActivePath = Join-Path $RepoRoot "models\serving\active.json"
if (Test-Path $ActivePath) {
    $Active = Get-Content $ActivePath -Raw | ConvertFrom-Json
    Write-Host "`nActive model: $($Active.model_file) (kind=$($Active.kind))" -ForegroundColor Green
}

# Step timings
Write-Host "`nStep timings:" -ForegroundColor White
foreach ($Key in $StepTimes.Keys | Sort-Object) {
    Write-Host ("  {0,-20s} {1}" -f $Key, $StepTimes[$Key].ToString('hh\:mm\:ss'))
}
$TotalElapsed = (Get-Date) - $TotalStart
Write-Host ("Total elapsed: {0}" -f $TotalElapsed.ToString('hh\:mm\:ss'))

# (g) Pack results
Run-Step -Name "pack" -Label "Pack Results" -Action {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\pack_results.ps1")
}

# Report errors
if ($Errors.Count -gt 0) {
    Write-Host "`n=== ERRORS ===" -ForegroundColor Red
    foreach ($E in $Errors) {
        Write-Host "  $E" -ForegroundColor Red
    }
}

Write-Host "`n=== Lab Training Complete ===" -ForegroundColor Green
Stop-Transcript
