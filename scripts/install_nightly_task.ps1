param(
    [string]$TaskName = "Slippi AI Nightly Review",
    [string]$At = "02:00",
    [string]$SlippiRoot = "$env:USERPROFILE\Documents\Slippi"
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_nightly.ps1"
if (-not (Test-Path $runner)) {
    throw "Nightly runner not found: $runner"
}

$time = [datetime]::ParseExact($At, "HH:mm", [Globalization.CultureInfo]::InvariantCulture)
$quotedRunner = $runner.Replace('"', '\"')
$quotedRoot = $SlippiRoot.Replace('"', '\"')
$arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$quotedRunner`" -SlippiRoot `"$quotedRoot`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments
$trigger = New-ScheduledTaskTrigger -Daily -At $time
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 7)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Analyze the newest Slippi session and publish a recurring Phillip practice report." `
    -Force | Out-Null

Write-Output "Installed '$TaskName' for $At daily."
Write-Output "Runner: $runner"
Write-Output "Log: $(Join-Path (Split-Path -Parent $PSScriptRoot) 'data\nightly\scheduled-task.log')"
