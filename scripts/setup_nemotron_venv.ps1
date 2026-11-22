param(
    # Path of the dedicated venv (relative to project root, or absolute).
    # Must match the ASR default asr_nemotron_python = ".venv_nemotron/Scripts/python.exe".
    [string]$VenvDir = ".venv_nemotron",
    # Base Python used to create the venv. Defaults to "python" on PATH.
    [string]$BasePython = "python",
    # PyTorch wheel index for a CUDA-enabled torch (change to match your GPU).
    # Set to empty string to let nemo-toolkit resolve torch itself (may be CPU-only).
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    # nemo-toolkit version spec. **Pinned on purpose** - do not change to "" (latest).
    #
    # The multilingual checkpoint nvidia/nemotron-3.5-asr-streaming-0.6b declares
    #   target: nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt
    # in its model_config.yaml. That module does NOT exist in nemo-toolkit 2.7.3, so
    # ASRModel.from_pretrained() falls back to instantiating the abstract base class and dies with
    #   TypeError: Can't instantiate abstract class ASRModel with abstract methods
    #              setup_training_data, setup_validation_data
    # 3.0.0 is the first PyPI release that ships rnnt_bpe_models_prompt.
    # Verified 2026-08-18: model loads as EncDecRNNTBPEModelWithPrompt and recognition works.
    [string]$NemoVersion = "==3.0.0"
)

# Create the dedicated venv for nemotron-asr inside the project.
# nemo-toolkit conflicts with the main app venv (torch 2.7.0+cu128 / transformers 5.x),
# so it is isolated in a separate process and venv. asr/nemotron_asr_server.py is launched
# with this venv's Python.

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
        Write-Host "[nemotron] venv already exists: $venvPath"
    } else {
        Write-Host "[nemotron] Creating venv at $venvPath (base: $BasePython)..."
        & $BasePython -m venv $venvPath
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit=$LASTEXITCODE)" }
    }

    Write-Host "[nemotron] Upgrading pip..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit=$LASTEXITCODE)" }

    if ($TorchIndexUrl -ne "") {
        Write-Host "[nemotron] Installing CUDA torch from $TorchIndexUrl ..."
        & $venvPython -m pip install torch torchaudio --index-url $TorchIndexUrl
        if ($LASTEXITCODE -ne 0) { throw "torch install failed (exit=$LASTEXITCODE)" }
    }

    $nemoSpec = "nemo-toolkit[asr]$NemoVersion"
    Write-Host "[nemotron] Installing $nemoSpec ..."
    & $venvPython -m pip install $nemoSpec
    if ($LASTEXITCODE -ne 0) { throw "nemo-toolkit install failed (exit=$LASTEXITCODE)" }

    Write-Host "[nemotron] Verifying imports..."
    & $venvPython -c "import torch, nemo.collections.asr as a; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "import verification failed (exit=$LASTEXITCODE)" }

    Write-Host ""
    Write-Host "[nemotron] Done. Set ASR 'Nemotron Python' to: $VenvDir/Scripts/python.exe"
} finally {
    Pop-Location
}
