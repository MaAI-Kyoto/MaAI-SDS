param(
    # Official v1.13.5: prompt_index (#3671, in 1.13.3+) and NeMo greedy
    # decoder_out cache (#3785). Remaining MaAI patches: max_symbols=30, NeMo mel.
    [string]$SherpaTag = "v1.13.5",
    [string]$SherpaCommit = "3dc7c569f31ca2cd4a20ed6f7db780327e6714c5",
    [string]$FallbackPyPiVersion = "1.13.5",
    # GPU build: fetches onnxruntime-win-x64-gpu-1.24.4 and bundles the CUDA execution
    # provider. nemotron-onnx fp32 encoder on CPU is ~165ms/chunk vs ~27ms on GPU.
    # Requires an NVIDIA GPU + CUDA 12 / cuDNN 9 at runtime (resolved via
    # ort_windows_preload._add_cuda_dll_dirs from the venv nvidia-* / torch wheels).
    # Set -EnableGpu:$false for a CPU-only build.
    [bool]$EnableGpu = $true
)

# Upgrade sherpa-onnx in the project venv to official v1.13.5 plus MaAI quality patches.
# PyPI 1.13.5 already has prompt_index and the greedy double-feed fix; Git build is
# still required for max_symbols=30 and NeMo preprocessor/mel recompute.
#
# Windows: Git + CMake + MSVC Build Tools (C++).
# Do NOT override TEMP/TMP — uv build isolation needs a stable system temp for FindPython.
# Use local clone + venv python + --no-build-isolation to avoid ephemeral interpreter paths.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$gitUrl = "https://github.com/k2-fsa/sherpa-onnx.git"
$srcDir = Join-Path $projectRoot "vendor\sherpa-onnx"
$installLog = Join-Path $projectRoot "upgrade_sherpa_build.log"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$venvScripts = Join-Path $projectRoot ".venv\Scripts"
$venvPythonCmake = ($venvPython -replace '\\', '/')

if (-not (Test-Path $venvPython)) {
    throw "Project venv not found: $venvPython (run 'uv sync' first)"
}

# Prefer venv cmake/ninja (kaldi-native-fbank needs cmake>=4.3.2)
$env:PATH = "$venvScripts;$env:PATH"

# Short UV cache only (optional; keeps wheel/git cache paths shorter)
$shortCache = "D:\uv-bld\cache"
if (-not (Test-Path $shortCache)) {
    New-Item -ItemType Directory -Force -Path $shortCache | Out-Null
}
$env:UV_CACHE_DIR = $shortCache

# Skip CLI/TTS/etc. — Python extension only (MSVC C1083 workaround)
# OpenFST MSVC build is unstable with -m:2; cmake_extension.py reads this.
$env:SHERPA_ONNX_MSVC_JOBS = "1"
$cmakeArgsList = @(
    "-DSHERPA_ONNX_ENABLE_BINARY=OFF",
    "-DSHERPA_ONNX_ENABLE_TTS=OFF",
    "-DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF",
    "-DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF",
    "-DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF",
    "-DPYTHON_EXECUTABLE=$venvPythonCmake"
)
if ($EnableGpu) {
    # Selects cmake/onnxruntime-win-x64-gpu.cmake -> onnxruntime-win-x64-gpu-1.24.4 (CUDA 12 / cuDNN 9).
    $cmakeArgsList += "-DSHERPA_ONNX_ENABLE_GPU=ON"
}
$env:SHERPA_ONNX_CMAKE_ARGS = $cmakeArgsList -join " "

function Invoke-UvQuiet {
    param([Parameter(Mandatory = $true)][string[]]$Args)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & uv @Args 2>&1
        $code = [int]$LASTEXITCODE
        $null = $output
        return $code
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Invoke-UvWithLog {
    param(
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    # uv は進捗を stderr に出す。PowerShell が NativeCommandError として赤表示しないよう文字列化する。
    # パイプ末尾に Tee-Object を繋ぐと関数の戻り値がログ行になり exit 判定が壊れるため、
    # 出力を変数に受けてからログへ書き、uv の LASTEXITCODE だけを返す。
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if (Test-Path $LogPath) {
            Remove-Item -Force $LogPath
        }
        $output = & uv @Args 2>&1
        $code = [int]$LASTEXITCODE
        $lines = foreach ($item in $output) {
            if ($item -is [System.Management.Automation.ErrorRecord]) {
                [string]$item.Exception.Message
            } else {
                [string]$item
            }
        }
        if ($lines) {
            foreach ($line in $lines) { Write-Host $line }
            $lines | Set-Content -Path $LogPath -Encoding utf8
        } else {
            "" | Set-Content -Path $LogPath -Encoding utf8
        }
        return $code
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Test-UvInstallLogSuccess {
    param([Parameter(Mandatory = $true)][string]$LogPath)
    if (-not (Test-Path $LogPath)) { return $false }
    $logText = Get-Content -Path $LogPath -Raw -ErrorAction SilentlyContinue
    return ($logText -match "Built sherpa-onnx") -and ($logText -match "Installed 1 package")
}

function Sync-SherpaNativeLibs {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$SitePackagesDir
    )
    $builtLib = Get-ChildItem -Path (Join-Path $SourceRoot "build") -Directory -Filter "lib.win-*-cpython-*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $builtLib) {
        throw "Built sherpa lib dir not found under $SourceRoot\build\lib.win-*-cpython-*"
    }
    $from = Join-Path $builtLib.FullName "sherpa_onnx"
    if (-not (Test-Path $from)) {
        throw "Missing build output: $from"
    }
    New-Item -ItemType Directory -Force -Path $SitePackagesDir | Out-Null
    Copy-Item -Path (Join-Path $from "*") -Destination $SitePackagesDir -Recurse -Force
    Write-Host "[sherpa-onnx] Synced package tree from $from -> $SitePackagesDir"
}

function Restore-PyPiSherpaOnnx {
    param([string]$Version)
    Write-Host "[sherpa-onnx] Restoring PyPI sherpa-onnx==$Version ..."
    $code = Invoke-UvQuiet -Args @("pip", "install", "--no-cache", "sherpa-onnx==$Version")
    if ($code -ne 0) {
        throw "Failed to restore PyPI sherpa-onnx==$Version (exit=$code)"
    }
}

function Ensure-SherpaSource {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git not found. Install Git for Windows."
    }
    if (-not (Test-Path (Join-Path $srcDir ".git"))) {
        Write-Host "[sherpa-onnx] Cloning to $srcDir ..."
        git clone --filter=blob:none $gitUrl $srcDir
        if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit=$LASTEXITCODE)" }
    }
    Write-Host "[sherpa-onnx] Fetching $SherpaTag ..."
    git -C $srcDir fetch --depth 1 origin "refs/tags/${SherpaTag}:refs/tags/${SherpaTag}"
    if ($LASTEXITCODE -ne 0) {
        git -C $srcDir fetch --depth 1 origin $SherpaCommit
    }
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed (exit=$LASTEXITCODE)" }
    Write-Host "[sherpa-onnx] Checking out $SherpaTag ($SherpaCommit) ..."
    git -C $srcDir checkout -f $SherpaCommit
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed (exit=$LASTEXITCODE)" }

    $patchMaxSymbols = Join-Path $scriptDir "patch_sherpa_nemo_max_symbols.py"
    if (Test-Path $patchMaxSymbols) {
        Write-Host "[sherpa-onnx] Patching NeMo online decoder max_symbols_per_frame=30 (Japanese dropout fix)..."
        & $venvPython $patchMaxSymbols $srcDir --max-symbols 30
        if ($LASTEXITCODE -ne 0) { throw "patch_sherpa_nemo_max_symbols.py failed (exit=$LASTEXITCODE)" }
    }

    $patchDoubleFeed = Join-Path $scriptDir "patch_sherpa_nemo_greedy_doublefeed.py"
    if (Test-Path $patchDoubleFeed) {
        Write-Host "[sherpa-onnx] Checking NeMo greedy double-feed (skip if upstream #3785)..."
        & $venvPython $patchDoubleFeed $srcDir
        if ($LASTEXITCODE -ne 0) { throw "patch_sherpa_nemo_greedy_doublefeed.py failed (exit=$LASTEXITCODE)" }
    }

    $patchMel = Join-Path $scriptDir "patch_sherpa_nemo_mel_params.py"
    if (Test-Path $patchMel) {
        Write-Host "[sherpa-onnx] Patching NeMo mel fbank params (preemph/frame/hann/librosa)..."
        & $venvPython $patchMel $srcDir
        if ($LASTEXITCODE -ne 0) { throw "patch_sherpa_nemo_mel_params.py failed (exit=$LASTEXITCODE)" }
    }

    $patchNormalize = Join-Path $scriptDir "patch_sherpa_nemo_online_normalize.py"
    if (Test-Path $patchNormalize) {
        Write-Host "[sherpa-onnx] Patching NeMo online full-STFT recompute + per_feature normalize (15s audio window)..."
        & $venvPython $patchNormalize $srcDir
        if ($LASTEXITCODE -ne 0) { throw "patch_sherpa_nemo_online_normalize.py failed (exit=$LASTEXITCODE)" }
    }

    $patchPreprocessor = Join-Path $scriptDir "patch_sherpa_nemo_preprocessor.py"
    if (Test-Path $patchPreprocessor) {
        Write-Host "[sherpa-onnx] Patching NeMo preprocessor ONNX window-mel path..."
        & $venvPython $patchPreprocessor $srcDir
        if ($LASTEXITCODE -ne 0) { throw "patch_sherpa_nemo_preprocessor.py failed (exit=$LASTEXITCODE)" }
    }

    $patchMsvc = Join-Path $scriptDir "patch_sherpa_msvc_parallel.py"
    if (Test-Path $patchMsvc) {
        Write-Host "[sherpa-onnx] Patching MSVC parallel jobs (SHERPA_ONNX_MSVC_JOBS=1, OpenFST fix)..."
        & $venvPython $patchMsvc $srcDir
        if ($LASTEXITCODE -ne 0) { throw "patch_sherpa_msvc_parallel.py failed (exit=$LASTEXITCODE)" }
    }
}

Push-Location $projectRoot
try {
    Write-Host "[sherpa-onnx] Upgrading to k2-fsa/sherpa-onnx@$SherpaCommit ..."
    Write-Host "[sherpa-onnx] PYTHON=$venvPython"
    Write-Host "[sherpa-onnx] UV_CACHE_DIR=$($env:UV_CACHE_DIR)"
    Write-Host "[sherpa-onnx] SHERPA_ONNX_CMAKE_ARGS=$($env:SHERPA_ONNX_CMAKE_ARGS)"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found. Install uv or activate the project venv manually."
    }

    Write-Host "[sherpa-onnx] Removing old packages..."
    Invoke-UvQuiet -Args @("pip", "uninstall", "sherpa-onnx", "sherpa-onnx-core") | Out-Null

    Ensure-SherpaSource

    $buildDir = Join-Path $srcDir "build"
    if (Test-Path $buildDir) {
        Write-Host "[sherpa-onnx] Removing stale build tree (avoids pyd/__init__.py mismatch)..."
        Remove-Item -Recurse -Force $buildDir
    }

    $binDir = Join-Path $srcDir "build\sherpa_onnx\bin"
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null

    Write-Host "[sherpa-onnx] Ensuring build deps in venv (cmake>=4.3.2)..."
    Invoke-UvQuiet -Args @("pip", "install", "setuptools", "wheel", "cmake>=4.3.2", "ninja") | Out-Null
    $cmakeVer = & (Join-Path $venvScripts "cmake.exe") --version 2>$null | Select-Object -First 1
    Write-Host "[sherpa-onnx] cmake: $cmakeVer"

    Write-Host "[sherpa-onnx] Building from local clone via uv pip (may take 5-15 min)..."
    Write-Host "[sherpa-onnx] NOTE: uv progress (Resolved / Building) is normal, not an error."
    $installCode = Invoke-UvWithLog -LogPath $installLog -Args @(
        "pip", "install", "--python", $venvPython,
        "--no-build-isolation", "--no-cache", "--reinstall", $srcDir
    )

    if (($installCode -ne 0) -and -not (Test-UvInstallLogSuccess -LogPath $installLog)) {
        Write-Host "[sherpa-onnx] Git build failed (exit=$installCode). See $installLog"
        Restore-PyPiSherpaOnnx -Version $FallbackPyPiVersion
        throw @"
sherpa-onnx git build failed. PyPI $FallbackPyPiVersion restored.
PyPI $FallbackPyPiVersion has prompt_index (#3671) and greedy #3785, but not max_symbols=30 / NeMo mel.
See upgrade_sherpa_build.log
"@
    }
    if ($installCode -ne 0) {
        Write-Host "[sherpa-onnx] NOTE: uv exit code was $installCode but log shows success; continuing."
    }

    $sitePackages = Join-Path $projectRoot ".venv\Lib\site-packages\sherpa_onnx"
    Sync-SherpaNativeLibs -SourceRoot $srcDir -SitePackagesDir $sitePackages

    $fixInitScript = Join-Path $scriptDir "fix_sherpa_onnx_init.py"
    Write-Host "[sherpa-onnx] Aligning __init__.py with built extension (TTS-disabled builds omit symbols)..."
    & $venvPython $fixInitScript $sitePackages
    if ($LASTEXITCODE -ne 0) {
        throw "fix_sherpa_onnx_init.py failed (exit=$LASTEXITCODE)"
    }

    Write-Host "[sherpa-onnx] Verifying import..."
    $verifyScript = Join-Path $scriptDir "verify_sherpa_import.py"
    $prevPyPath = $env:PYTHONPATH
    $env:PYTHONPATH = $projectRoot
    try {
        & $venvPython $verifyScript
        $verifyCode = $LASTEXITCODE
    } finally {
        if ($null -eq $prevPyPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $prevPyPath
        }
    }
    if ($verifyCode -ne 0) {
        throw "sherpa-onnx import verification failed (exit=$verifyCode). Git build left in place; see upgrade_sherpa_build.log"
    }

    $markerPath = Join-Path $projectRoot ".sherpa_onnx_git_built"
    @(
        "commit=$SherpaCommit",
        "tag=$SherpaTag",
        "built_at=$(Get-Date -Format o)",
        "source=$srcDir",
        "enable_gpu=$EnableGpu",
        "nemo_max_symbols_per_frame=30",
        "nemo_greedy_doublefeed_fix=upstream_3785",
        "nemo_online_per_feature_window_sec=15",
        "nemo_online_full_stft_recompute=15s"
    ) | Set-Content -Path $markerPath -Encoding utf8
    Write-Host "[sherpa-onnx] Wrote marker: $markerPath"

    Write-Host "[sherpa-onnx] Running nemotron prompt_index smoke test..."
    $testScript = Join-Path $scriptDir "test_nemotron_prompt_index.py"
    if (Test-Path $testScript) {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $prevPyPath = $env:PYTHONPATH
        $env:PYTHONPATH = $projectRoot
        try {
            Push-Location $projectRoot
            & $venvPython $testScript 2>&1 | Tee-Object -FilePath (Join-Path $projectRoot "upgrade_sherpa_test.log")
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[sherpa-onnx] WARNING: smoke test failed (exit=$LASTEXITCODE). See upgrade_sherpa_test.log"
            }
        } finally {
            Pop-Location
            if ($null -eq $prevPyPath) {
                Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
            } else {
                $env:PYTHONPATH = $prevPyPath
            }
            $ErrorActionPreference = $prevEap
        }
    }

    Write-Host ""
    Write-Host "[sherpa-onnx] Done. Restart ASR (nemotron-onnx) from the GUI."
    Write-Host "[sherpa-onnx] Do NOT run 'uv sync' — it reinstalls unpatched PyPI (max_symbols=10, no NeMo mel)."
    Write-Host "[sherpa-onnx] After confirming nemotron-onnx works, add git pin to pyproject.toml then 'uv lock'."
    exit 0
} catch {
    Write-Host "[sherpa-onnx] ERROR: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
