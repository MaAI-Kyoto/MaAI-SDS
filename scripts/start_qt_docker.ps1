param(
    [int]$Port = 0,
    [string]$Python = "",
    [ValidateSet("auto", "gpu", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$configPath = Join-Path $repoRoot "controller_config.docker.json"
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Missing controller_config.docker.json. Copy controller_config.docker.example.json and edit it first."
}
# docker-compose.yml mounts enabled_engines.json as a file. It is not in Git, and if it is missing
# Docker creates a directory with that name instead, which breaks saving the engine list.
$enginesPath = Join-Path $repoRoot "enabled_engines.json"
if (Test-Path -LiteralPath $enginesPath -PathType Container) {
    throw "enabled_engines.json is a directory (compose was started without the file). Delete it and run again."
}
if (-not (Test-Path -LiteralPath $enginesPath)) {
    [System.IO.File]::WriteAllText($enginesPath, "[]`n")
}

if ($Port -le 0) {
    $configuredPort = [Environment]::GetEnvironmentVariable(
        "MAAI_SDS_QT_ORCHESTRATOR_PORT", "Process"
    )
    if (-not $configuredPort) {
        $envFile = Join-Path $repoRoot ".env"
        if (Test-Path -LiteralPath $envFile -PathType Leaf) {
            foreach ($line in Get-Content -LiteralPath $envFile) {
                if ($line -match '^\s*MAAI_SDS_QT_ORCHESTRATOR_PORT\s*=\s*["'']?(\d+)["'']?\s*(?:#.*)?$') {
                    $configuredPort = $Matches[1]
                    break
                }
            }
        }
    }
    $Port = if ($configuredPort) { [int]$configuredPort } else { 8765 }
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535."
}

if (-not $Python) {
    $Python = [Environment]::GetEnvironmentVariable("MAAI_SDS_QT_PYTHON", "Process")
}
if (-not $Python) {
    $candidate = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $Python = $candidate
    } else {
        $command = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            throw "Python was not found. Pass -Python or set MAAI_SDS_QT_PYTHON."
        }
        $Python = $command.Source
    }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable was not found: $Python"
}

$resolvedDevice = $Device.ToLowerInvariant()
if ($resolvedDevice -eq "auto") {
    $deviceMarker = Join-Path $repoRoot ".device_extra"
    $deviceExtra = ""
    if (Test-Path -LiteralPath $deviceMarker -PathType Leaf) {
        $deviceExtra = (Get-Content -LiteralPath $deviceMarker -Raw).Trim().ToLowerInvariant()
    }
    $resolvedDevice = if ($deviceExtra -eq "cpu") { "cpu" } else { "gpu" }
}

$composeArgs = @(
    "compose",
    "-f", (Join-Path $repoRoot "docker-compose.yml")
)
if ($resolvedDevice -eq "cpu") {
    $composeArgs += @("-f", (Join-Path $repoRoot "docker-compose.cpu.yml"))
} else {
    $composeArgs += @("-f", (Join-Path $repoRoot "docker-compose.gpu.yml"))
}
$composeFiles = @($composeArgs | Where-Object { $_ -like "*.yml" })
$composeArgs += @("--profile", "qt", "up", "-d", "orchestrator")

function Test-TcpPort {
    param([string]$HostName, [int]$PortNumber)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect($HostName, $PortNumber, $null, $null)
        if (-not $result.AsyncWaitHandle.WaitOne(500)) {
            return $false
        }
        $client.EndConnect($result)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$env:MAAI_SDS_QT_ORCHESTRATOR_PORT = [string]$Port
$env:MAAI_SDS_REMOTE_ORCHESTRATOR = "127.0.0.1:$Port"
$env:MAAI_SDS_CONTROLLER_CONFIG = $configPath
# GUI のモジュール状態の「対話サーバー」行から、同じ compose でコンテナを起動・停止できるようにする。
$env:MAAI_SDS_QT_COMPOSE_FILES = ($composeFiles -join ";")
$env:MAAI_SDS_QT_COMPOSE_DIR = $repoRoot
$env:PYQTGRAPH_QT_LIB = "PySide6"
$env:PYTHONIOENCODING = "utf-8"

# The source checkout is usable without an editable install. Development
# worktrees may also keep the PySide6 essentials wheels in a sibling folder.
$pythonPathParts = @()
$essentialsRoot = Join-Path (Split-Path -Parent $repoRoot) "pylib_essentials"
if (Test-Path -LiteralPath $essentialsRoot -PathType Container) {
    $pythonPathParts += $essentialsRoot
}
$pythonPathParts += (Join-Path $repoRoot "src")
$existingPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ($existingPythonPath) {
    $pythonPathParts += $existingPythonPath
}
$env:PYTHONPATH = ($pythonPathParts -join ";")

Push-Location $repoRoot
try {
    & docker @composeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }

    $deadline = [DateTime]::UtcNow.AddSeconds(120)
    while (-not (Test-TcpPort -HostName "127.0.0.1" -PortNumber $Port)) {
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "Timed out waiting for the orchestrator on 127.0.0.1:$Port"
        }
        Start-Sleep -Milliseconds 500
    }

    & $Python (Join-Path $repoRoot "main.py") `
        --remote-orchestrator "127.0.0.1:$Port" `
        --config $configPath
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
