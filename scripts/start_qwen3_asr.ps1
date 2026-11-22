param(
    [string]$ComposeFile = "docker-compose.qwen-asr.yml",
    [string]$Model = "Qwen/Qwen3-ASR-0.6B",
    [string]$ServedModelName = "Qwen3-ASR-0.6B",
    [string]$GpuMemoryUtilization = "0.8",
    [string]$MaxModelLen = "4096",
    [string]$HostPort = "8000",
    [string]$SessionTimeoutSec = "60",
    [string]$ChunkSec = "2.0",
    [string]$EnforceEager = "0"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# docker が PATH に無い / Docker Desktop が起動していない場合、`docker compose up` は
# CommandNotFoundException や接続エラーで落ちるが、そのままだと呼び出し元（GUI）には
# 「exit=1」としか伝わらず原因がわからない。先に明示チェックして分かりやすいメッセージにする。
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCmd) {
    throw "Docker が見つかりません。Docker Desktop をインストールし、'docker' コマンドが PATH に通っていることを確認してください: https://www.docker.com/products/docker-desktop/"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker デーモンに接続できません。Docker Desktop を起動してから再試行してください。"
}

Push-Location $projectRoot
try {
    $env:QWEN_ASR_MODEL = $Model
    $env:QWEN_ASR_SERVED_MODEL_NAME = $ServedModelName
    $env:QWEN_ASR_GPU_MEMORY_UTILIZATION = $GpuMemoryUtilization
    $env:QWEN_ASR_MAX_MODEL_LEN = $MaxModelLen
    $env:QWEN_ASR_HOST_PORT = $HostPort
    $env:QWEN_ASR_SESSION_TIMEOUT_SEC = $SessionTimeoutSec
    $env:QWEN_ASR_CHUNK_SEC = $ChunkSec
    if ($EnforceEager -eq "1") {
        $env:QWEN_ASR_ENFORCE_EAGER = "1"
    } else {
        $env:QWEN_ASR_ENFORCE_EAGER = ""
    }

    Write-Host "[Qwen3-ASR] Starting Docker service..."
    docker compose -f $ComposeFile up -d
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed (exit=$LASTEXITCODE)"
    }

    Write-Host "[Qwen3-ASR] Waiting for health endpoint (first boot can take several minutes)..."
    for ($i = 0; $i -lt 300; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri ("http://127.0.0.1:" + $HostPort + "/health") -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) {
                Write-Host "[Qwen3-ASR] Ready."
                exit 0
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    throw ("Timed out waiting for http://127.0.0.1:" + $HostPort + "/health")
} finally {
    Pop-Location
}
