param(
    [string]$SlippiRoot = "$env:USERPROFILE\Documents\Slippi",
    [int]$Samples = 4,
    [int]$SegmentsPerGame = 8,
    [double]$MaxHours = 6
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "data\nightly"
$logPath = Join-Path $logDir "scheduled-task.log"
$entryPoint = Join-Path $repo ".venv\Scripts\slippi-review.exe"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $repo

$activePath = Join-Path $logDir "active_nightly.json"
if (Test-Path $activePath) {
    try {
        $active = Get-Content $activePath -Raw | ConvertFrom-Json
        if ($active.pid) {
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$active.pid)" -ErrorAction SilentlyContinue
            $commandLine = [string]$process.CommandLine
            $nightlyCommandPattern = "(?i)$([regex]::Escape($entryPoint))`"?\s+nightly(?:\s|$)"
            $nightlyModulePattern = "(?i)-m\s+slippi_ai_review\.cli\s+nightly(?:\s|$)"

            if ($process -and ($commandLine -match $nightlyCommandPattern -or $commandLine -match $nightlyModulePattern)) {
                "[$(Get-Date -Format o)] skipped: nightly run $($active.runId) is already active (PID $($active.pid))" |
                    Add-Content -Path $logPath
                exit 0
            }

            if ($process) {
                "[$(Get-Date -Format o)] ignored stale active-run metadata: PID $($active.pid) belongs to an unrelated process" |
                    Add-Content -Path $logPath
            } else {
                "[$(Get-Date -Format o)] ignored stale active-run metadata: PID $($active.pid) is no longer running" |
                    Add-Content -Path $logPath
            }
        }
    } catch {
        "[$(Get-Date -Format o)] ignored unreadable active-run metadata: $($_.Exception.Message)" |
            Add-Content -Path $logPath
    }
}

$arguments = @(
    "nightly",
    "--slippi-root", $SlippiRoot,
    "--samples", $Samples,
    "--segments-per-game", $SegmentsPerGame,
    "--max-hours", $MaxHours,
    "--gpu-duty-cycle", 1.0
)

"[$(Get-Date -Format o)] scheduled nightly run starting" | Add-Content -Path $logPath
if (Test-Path $entryPoint) {
    & $entryPoint @arguments *>> $logPath
} else {
    & python -m slippi_ai_review.cli @arguments *>> $logPath
}
$exitCode = $LASTEXITCODE
"[$(Get-Date -Format o)] scheduled nightly run exited $exitCode" | Add-Content -Path $logPath
exit $exitCode
