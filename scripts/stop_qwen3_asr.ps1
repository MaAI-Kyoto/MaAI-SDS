param(
    [string]$ComposeFile = "docker-compose.qwen-asr.yml"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Push-Location $projectRoot
try {
    Write-Host "[Qwen3-ASR] Stopping Docker service..."
    docker compose -f $ComposeFile down
    Write-Host "[Qwen3-ASR] Stopped."
} finally {
    Pop-Location
}
