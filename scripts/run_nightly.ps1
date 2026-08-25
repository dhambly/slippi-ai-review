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
        if ($active.pid -and (Get-Process -Id $active.pid -ErrorAction SilentlyContinue)) {
            "[$(Get-Date -Format o)] skipped: nightly run $($active.runId) is already active (PID $($active.pid))" |
                Add-Content -Path $logPath
            exit 0
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
