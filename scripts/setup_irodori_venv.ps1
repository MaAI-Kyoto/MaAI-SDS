param(
    # Path of the dedicated venv (relative to project root, or absolute).
    # Must match the TTS default irodori_python = ".venv_irodori/Scripts/python.exe".
    [string]$VenvDir = ".venv_irodori",
    # Base Python used to create the venv. Defaults to "python" on PATH.
    [string]$BasePython = "python",
    # PyTorch wheel index for a CUDA-enabled torch (change to match your GPU).
    # Set to empty string to let Irodori-TTS resolve torch itself (may be CPU-only).
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

# Create the dedicated venv for Irodori-TTS inside the project.
# Irodori-TTS requires transformers>=5.12.1,<6, which conflicts with the main app
# venv's transformers==5.5.3, so it is isolated in a separate process and venv.
# tts/irodori_server.py is launched with this venv's Python.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Push-Location $projectRoot
try {
    if ([System.IO.Path]::IsPathRooted($VenvDir)) {
        $venvPath = $VenvDir
    } else {
        $venvPath = Join-Path $projectRoot $VenvDir
    }
    $venvPython = Join-Path $venvPath "Scripts\python.exe"

    if (Test-Path $venvPython) {
        Write-Host "[irodori] venv already exists: $venvPath"
    } else {
        Write-Host "[irodori] Creating venv at $venvPath (base: $BasePython)..."
        & $BasePython -m venv $venvPath
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit=$LASTEXITCODE)" }
    }

    Write-Host "[irodori] Upgrading pip..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit=$LASTEXITCODE)" }

    if ($TorchIndexUrl -ne "") {
        Write-Host "[irodori] Installing CUDA torch from $TorchIndexUrl ..."
        & $venvPython -m pip install "torch>=2.10.0" "torchaudio>=2.10.0" --index-url $TorchIndexUrl
        if ($LASTEXITCODE -ne 0) { throw "torch install failed (exit=$LASTEXITCODE)" }
    }

    # dacvae is not published on PyPI. Irodori-TTS declares only the bare package name
    # and relies on its repository's uv-specific [tool.uv.sources] mapping, which pip does
    # not understand. Installing the Git distribution first lets pip recognize the
    # requirement as satisfied when it resolves Irodori-TTS's remaining dependencies.
    $dacvaeSpec = "dacvae @ git+https://github.com/facebookresearch/dacvae"
    Write-Host "[irodori] Installing dacvae from Git..."
    & $venvPython -m pip install $dacvaeSpec
    if ($LASTEXITCODE -ne 0) { throw "dacvae install failed (exit=$LASTEXITCODE)" }

    $irodoriSpec = "irodori-tts @ git+https://github.com/Aratako/Irodori-TTS@8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
    Write-Host "[irodori] Installing pinned irodori-tts..."
    & $venvPython -m pip install $irodoriSpec
    if ($LASTEXITCODE -ne 0) { throw "irodori-tts install failed (exit=$LASTEXITCODE)" }

    Write-Host "[irodori] Verifying imports..."
    & $venvPython -c "import torch, irodori_tts; from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'irodori_tts', 'OK')"
    if ($LASTEXITCODE -ne 0) { throw "import verification failed (exit=$LASTEXITCODE)" }

    Write-Host ""
    Write-Host "[irodori] Done. Set TTS 'Irodori Python' to: $VenvDir/Scripts/python.exe"
} finally {
    Pop-Location
}
