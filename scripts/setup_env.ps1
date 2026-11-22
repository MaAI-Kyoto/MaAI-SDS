<#
.SYNOPSIS
    MaAI-SDS の環境を最初から作り直す（単一の入口）。

.DESCRIPTION
    このリポジトリの環境は「uv sync だけでは完成しない」4 点があり、順番も決まっている。
    そこを毎回手で思い出すのが事故の元だったので 1 本にまとめた。

      1. uv sync --all-extras     … 本体 venv（.venv）。PySide6 も各 ASR/TTS エンジンも
                                    optional extra なので、素の uv sync では入らない。
      2. カスタム sherpa-onnx      … PyPI 版では日本語 ASR の品質が出ない。
                                    **uv sync はこれを毎回 PyPI 版で上書きする**ので後に置く。
      3. .venv_nemotron            … -WithNemotron のときだけ。nemotron-asr は nemo-toolkit を
                                    別 venv に入れてサブプロセスで動かす（本体 venv とは依存が
                                    両立しない。約 8GB）。GUI の「エンジン管理」からも作れる。
      4. エンジン資産              … 既定の sherpa-onnx（ReazonSpeech）と VOICEVOX-Core の分だけ
                                    （setup_assets.ps1。冪等。無ければ入手、あれば触らない）。
                                    ほかのエンジンは GUI の「エンジン管理」で入れる。

    最後に検証まで走る（sherpa の版・ワーカーの -m 起動・GUI 構築・テスト）。

.PARAMETER SkipSherpa
    sherpa-onnx の復元/ビルドを行わない。**uv sync 直後は PyPI 版が入った状態になる**ので、
    日本語 ASR の品質検証をする前に必ず別途復元すること。

.PARAMETER WithNemotron
    .venv_nemotron も作る（nemotron-asr エンジン用。約 8GB）。既定では作らない。

.PARAMETER SkipAssets
    エンジン資産の取得（setup_assets.ps1）をしない。
    Tsukasa-HF は setup_assets.ps1 が公開リポジトリから取る（トークン不要）:
      powershell -File scripts\setup_env.ps1 -SkipAssets
      powershell -File scripts\setup_assets.ps1 -PreFetchWhisper

.PARAMETER ForceSherpaBuild
    ローカルのスナップショットがあってもソースからビルドする（新しいマシンと同じ経路を試す）。

.PARAMETER VerifyOnly
    何もインストールせず、検証だけ実行する。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -WithNemotron
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -VerifyOnly
#>
param(
    # GPU を使うか。cuda = torch cu128 + onnxruntime の CUDA ランタイム、
    # cpu = どちらも CPU 版。GPU の無いマシンでは cuda を選ぶと nvidia-* 一式
    # （cudnn 1.0GB / nvjitlink 1.0GB / cublas 0.8GB など計 7 パッケージ）が
    # 丸ごと無駄になるうえ、CUDA を要求したエンジンが例外で落ちる。
    # ここでの選択は .device_extra に記録され、以後のエンジン適用もそれに従う。
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$SkipSherpa,
    [switch]$WithNemotron,
    [switch]$ForceSherpaBuild,
    [switch]$SkipAssets,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$venvPython  = Join-Path $projectRoot ".venv\Scripts\python.exe"
$sitePkgs    = Join-Path $projectRoot ".venv\Lib\site-packages"
# カスタム sherpa の退避先。uv sync が上書きする前にここへ待避し、後で戻す。
# （新しいマシンにはこれが無いので、その場合はソースからビルドする経路に落ちる）
$sherpaSnap  = Join-Path $projectRoot "vendor\sherpa_onnx_prebuilt"

function Write-Step([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Warn([string]$msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-Ok([string]$msg)   { Write-Host "[ok] $msg" -ForegroundColor Green }

# ---------------------------------------------------------------- 前提チェック
function Test-Prereqs {
    Write-Step "前提コマンドの確認"
    $missing = @()
    foreach ($cmd in @("uv", "git")) {
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { $missing += $cmd }
    }
    if ($missing.Count -gt 0) {
        throw "必要なコマンドが見つからない: $($missing -join ', ')"
    }
    Write-Ok ("uv " + (& uv --version))
    if (-not $SkipSherpa) {
        # ソースビルドに落ちた場合だけ必要。スナップショットからの復元では要らない。
        if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
            Write-Warn "cmake が PATH に無い。sherpa-onnx をソースからビルドするには CMake + MSVC Build Tools の C++ ワークロードが必要。"
        }
    }
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia) {
        Write-Ok "NVIDIA GPU あり（sherpa/onnxruntime の CUDA 実行プロバイダを使う）"
    } else {
        Write-Warn "nvidia-smi が見つからない。CPU では nemotron-onnx の encoder が 1 チャンク約 165ms かかり（GPU なら約 27ms）、リアルタイムに追いつかない。"
    }
}

# PowerShell 5.1 では、ネイティブ exe の stderr を 2>&1 でリダイレクトすると 1 行ごとに
# ErrorRecord に包まれ、$ErrorActionPreference="Stop" のせいで**終了コード 0 でも中断する**
# （Qt の QFontDatabase 警告で実際に踏んだ）。ここだけ Continue にして文字列として受ける。
function Invoke-VenvPython([string]$code) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $venvPython -B -c $code 2>&1
        return (($out | ForEach-Object { "$_" }) -join "`n")
    } finally { $ErrorActionPreference = $prev }
}

# -------------------------------------------------- sherpa のスナップショット
function Get-SherpaVersion {
    if (-not (Test-Path $venvPython)) { return $null }
    $out = Invoke-VenvPython "import sherpa_onnx,sys; sys.stdout.write(sherpa_onnx.__version__)"
    if ($LASTEXITCODE -ne 0) { return $null }
    return $out.Trim()
}

function Save-SherpaSnapshot {
    # uv sync の前に呼ぶ。カスタムビルド（版に '+' が付く）のときだけ待避する。
    $ver = Get-SherpaVersion
    if (-not $ver) { Write-Warn "sherpa-onnx が未インストール（待避なし）"; return $false }
    if ($ver -notmatch '\+') {
        Write-Warn "入っているのは PyPI 版 sherpa-onnx $ver（カスタムビルドではないので待避しない）"
        return $false
    }
    Write-Host "[sherpa] カスタムビルド $ver を待避する（uv sync が上書きするため）"
    if (Test-Path $sherpaSnap) { Remove-Item -Recurse -Force $sherpaSnap }
    New-Item -ItemType Directory -Force -Path $sherpaSnap | Out-Null
    Copy-Item (Join-Path $sitePkgs "sherpa_onnx") $sherpaSnap -Recurse -Force
    Get-ChildItem $sitePkgs -Directory -Filter "sherpa_onnx*dist-info" |
        ForEach-Object { Copy-Item $_.FullName $sherpaSnap -Recurse -Force }
    Set-Content -Path (Join-Path $sherpaSnap "VERSION.txt") -Value $ver -Encoding utf8
    Write-Ok "待避先: $sherpaSnap"
    return $true
}

function Restore-SherpaSnapshot {
    if (-not (Test-Path (Join-Path $sherpaSnap "sherpa_onnx"))) { return $false }
    $ver = (Get-Content (Join-Path $sherpaSnap "VERSION.txt") -ErrorAction SilentlyContinue) -join ""
    Write-Host "[sherpa] スナップショット ($ver) から復元する"
    # PyPI 版を消してから戻す（混在すると DLL の版が食い違う）
    $target = Join-Path $sitePkgs "sherpa_onnx"
    if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    Get-ChildItem $sitePkgs -Directory -Filter "sherpa_onnx*dist-info" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
    # sherpa_onnx と *.dist-info だけ（engine_manager._restore_sherpa_snapshot と同じ）。
    # 配布 zip の licenses/ などは site-packages に入れない。
    Get-ChildItem $sherpaSnap -Directory |
        Where-Object { $_.Name -eq "sherpa_onnx" -or $_.Name -like "*.dist-info" } |
        ForEach-Object { Copy-Item $_.FullName $sitePkgs -Recurse -Force }
    Write-Ok "復元した"
    return $true
}

# ------------------------------------------------------------------ 各ステップ
function Invoke-UvSync {
    Write-Step "1. uv sync --all-extras（本体 venv）"
    # onnxruntime-gpu の wheel は 280MB 近くあり、uv の既定 30 秒では読み切れず
    #   Failed to download distribution due to network timeout
    # で落ちる（実機で発生）。uv は版によって見る変数名が違うので両方入れる。
    # 呼び出し側が明示していればそちらを尊重する。
    if (-not $env:UV_HTTP_TIMEOUT)    { $env:UV_HTTP_TIMEOUT = "300" }
    if (-not $env:UV_REQUEST_TIMEOUT) { $env:UV_REQUEST_TIMEOUT = "300" }
    Push-Location $projectRoot
    try {
        # maai は pyproject で **commit 固定**。branch 参照だと同じ pyproject でも
        # 環境が変わってしまい再現できない（実際に uv.lock と食い違った）。
        #
        # --all-extras が要る。PySide6（gui）と各 ASR/TTS エンジンの依存は optional
        # extra なので、素の `uv sync` では 1 つも入らず、GUI が起動せず、この
        # スクリプト自身の検証（GUI 構築）も落ちる。extras に何があるかは
        # pyproject.toml が唯一の情報源なので、ここに一覧を書き写さず uv に任せる。
        # 使わないエンジンは、後から GUI の「エンジン管理」で外せる。
        # cpu / cu128 は排他 extra（pyproject の [tool.uv] conflicts）なので、
        # --all-extras だけだと "Extras `cpu` and `cu128` are incompatible" で落ちる。
        # 使わない方を --no-extra で外す。
        $deviceExtra = if ($Device -eq "cpu") { "cpu" } else { "cu128" }
        $dropExtra   = if ($Device -eq "cpu") { "cu128" } else { "cpu" }
        Write-Host "[env] デバイス構成: $Device (extra=$deviceExtra)"
        Set-Content -Path (Join-Path $projectRoot ".device_extra") -Value $deviceExtra -Encoding ascii -NoNewline
        & uv sync --all-extras --no-extra $dropExtra
        if ($LASTEXITCODE -ne 0) { throw "uv sync が失敗した (exit $LASTEXITCODE)" }
    } finally { Pop-Location }
    Write-Ok "uv sync 完了"
}

function Invoke-SherpaSetup {
    Write-Step "2. カスタム sherpa-onnx"
    if (-not $ForceSherpaBuild) {
        if (Restore-SherpaSnapshot) { return }
        # スナップショットが無ければ配布プリビルドを取りに行く。ソースビルドは
        # Git + CMake + MSVC が要り 5〜15 分かかるので、クリーンクローンでは
        # まずこちらを試す。取れなければ従来どおりビルドへ落ちる。
        $assets = Join-Path $scriptDir "setup_assets.ps1"
        if (Test-Path $assets) {
            Write-Host "[env] 配布プリビルドの取得を試みます"
            & powershell -ExecutionPolicy Bypass -File $assets -Only SherpaPrebuilt
            if ($LASTEXITCODE -eq 0 -and (Restore-SherpaSnapshot)) { return }
        }
        Write-Warn "プリビルドを取得できないのでソースからビルドする（5〜15 分以上かかる）"
    }
    $up = Join-Path $scriptDir "upgrade_sherpa_onnx.ps1"
    if (-not (Test-Path $up)) {
        throw ("見つからない: $up`n" +
               "  ビルド一式（upgrade_sherpa_onnx.ps1 / patch_sherpa_*.py /`n" +
               "  fix_sherpa_onnx_init.py / sherpa_patches\）は追跡下にあるので、`n" +
               "  通常のクローンには入っています。消えている場合は git status で確認を。`n" +
               "  カスタムビルド無しで進めるなら -SkipSherpa（PyPI 版で動作しますが`n" +
               "  日本語 ASR の品質は落ちます）。")
    }
    # -Device cpu なら CUDA 実行プロバイダ無しでビルドする（GPU の無いマシンで CUDA 版を作らない）
    if ($Device -eq "cpu") {
        & powershell -ExecutionPolicy Bypass -File $up -CpuOnly
    } else {
        & powershell -ExecutionPolicy Bypass -File $up
    }
    if ($LASTEXITCODE -ne 0) { throw "upgrade_sherpa_onnx.ps1 が失敗した (exit $LASTEXITCODE)" }
    # 次回の uv sync 用に、今できたものを待避しておく
    Save-SherpaSnapshot | Out-Null
}

function Invoke-NemotronSetup {
    Write-Step "3. .venv_nemotron（nemotron-asr 用の別 venv）"
    $sn = Join-Path $scriptDir "setup_nemotron_venv.ps1"
    if (-not (Test-Path $sn)) { throw "見つからない: $sn" }
    # nemo-toolkit の版は setup_nemotron_venv.ps1 側で ==3.0.0 に固定してある
    # （2.7.3 には rnnt_bpe_models_prompt が無く、チェックポイントを読めない）。
    & powershell -ExecutionPolicy Bypass -File $sn
    if ($LASTEXITCODE -ne 0) { throw "setup_nemotron_venv.ps1 が失敗した (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------------- 検証
function Invoke-Verify {
    Write-Step "検証"
    $failures = @()

    # (a) sherpa の版。カスタムビルドなら '+cuda' などが付く。
    $ver = Get-SherpaVersion
    if (-not $ver) {
        $failures += "sherpa-onnx を import できない"
    } elseif ($ver -match '\+') {
        Write-Ok "sherpa-onnx $ver（カスタムビルド）"
    } else {
        Write-Warn "sherpa-onnx $ver は PyPI 版（max_symbols=30 と NeMo mel のパッチが無く日本語 ASR の品質が落ちる）。復元: powershell -File scripts\setup_env.ps1"
    }

    # (b) 7 ワーカー + orchestrator を import できるか。
    #     名前は d633d8c のサブパッケージ分割後のもの（`python -m` の起動名と同じ）。
    #     分割前の平坦名（maai_sds.asr_worker 等）はもう存在しない。
    $workers = @("asr.worker", "asr.compare_worker", "audio.input_worker", "audio.output_worker",
                 "llm.worker", "maai.worker", "tts.worker", "dialog.worker")
    foreach ($w in $workers) {
        $out = Invoke-VenvPython "import importlib,sys; importlib.import_module('maai_sds.$w'); sys.stdout.write('OK')"
        if ($out -match "OK") {
            Write-Ok "maai_sds.$w"
        } else {
            $failures += "maai_sds.$w を import できない: $out"
        }
    }

    # (c) GUI 構築（オフスクリーン）
    $env:QT_QPA_PLATFORM = "offscreen"
    $guiCode = "import maai_sds.gui.controller as c; from PySide6.QtWidgets import QApplication; a=QApplication([]); w=c.MMDAgentController(); print('CONSTRUCT_OK')"
    $gui = Invoke-VenvPython $guiCode
    if ($gui -match "CONSTRUCT_OK") { Write-Ok "GUI 構築" } else { $failures += "GUI 構築に失敗: $gui" }

    # (d) .venv_nemotron
    $nemoPy = Join-Path $projectRoot ".venv_nemotron\Scripts\python.exe"
    if (Test-Path $nemoPy) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $nv = (& $nemoPy -B -c "import nemo; print(nemo.__version__)" 2>&1 | ForEach-Object { "$_" }) -join "`n"
        $nemoRc = $LASTEXITCODE
        $ErrorActionPreference = $prev
        if ($nemoRc -eq 0) { Write-Ok "nemo-toolkit $($nv.Trim()) (.venv_nemotron)" }
        else { $failures += ".venv_nemotron で nemo を import できない: $nv" }
    } else {
        Write-Warn ".venv_nemotron が無い（nemotron-asr エンジンは使えない）"
    }

    # (e) テスト。-B で古い .pyc を使わない（同じ大きさの編集を同じ秒に行うと古いバイトコードが残るため）
    $testsDir = Join-Path $projectRoot "tests"
    if (Test-Path $testsDir) {
        Write-Host "[test] tests/ の unittest を実行する"
        Push-Location $projectRoot
        try { & $venvPython -B -m unittest discover -s tests -p "test_*.py" }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { $failures += "tests/ のテストが失敗した" }
        else { Write-Ok "テスト green" }
    } else {
        Write-Warn "tests/ が無いので飛ばす"
    }

    Write-Step "結果"
    if ($failures.Count -gt 0) {
        foreach ($f in $failures) { Write-Host "[FAIL] $f" -ForegroundColor Red }
        throw "$($failures.Count) 件失敗した"
    }
    Write-Ok "すべて通った"
}

# ------------------------------------------------------------------- 実行本体
Write-Host "MaAI-SDS 環境セットアップ ($projectRoot)"

if ($VerifyOnly) {
    Invoke-Verify
    return
}

Test-Prereqs

if (-not $SkipSherpa) {
    # uv sync の前に待避する。ここが順番の要点。
    Save-SherpaSnapshot | Out-Null
}
Invoke-UvSync
if (-not $SkipSherpa) { Invoke-SherpaSetup } else { Write-Warn "sherpa-onnx は PyPI 版のまま（-SkipSherpa）" }
if ($WithNemotron) { Invoke-NemotronSetup } else { Write-Host "[env] .venv_nemotron は作らない（nemotron-asr を使うなら -WithNemotron か、GUI の「エンジン管理」）" }

# ステップ 4: 各エンジンのモデル / EXE / 辞書（冪等。既にある環境では純粋な検証になる）
if (-not $VerifyOnly) {
    if (-not $SkipAssets) {
        $sa = Join-Path $scriptDir "setup_assets.ps1"
        if (-not (Test-Path $sa)) { throw "見つからない: $sa" }
        & powershell -ExecutionPolicy Bypass -File $sa
        if ($LASTEXITCODE -ne 0) { throw "setup_assets.ps1 が失敗した (exit $LASTEXITCODE)" }
    } else {
        Write-Warn "エンジン資産の取得はしない（-SkipAssets）"
    }
}

Invoke-Verify
