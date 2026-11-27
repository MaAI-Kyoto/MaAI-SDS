<#
.SYNOPSIS
    MaAI-SDS の環境を最初から作り直す（単一の入口）。

.DESCRIPTION
    このリポジトリの環境は「uv sync だけでは完成しない」4 点があり、順番も決まっている。
    そこを毎回手で思い出すのが事故の元だったので 1 本にまとめた。

      1. uv sync                  … 本体 venv（.venv）。PySide6 も各 ASR/TTS エンジンも
                                    optional extra なので、素の uv sync では入らない。
                                    既定の構成（gui・web・asr-sherpa・tts-voicevox）と、既に入っている
                                    エンジンの extra だけを入れる。全部入れるなら -AllExtras。
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

.PARAMETER AllExtras
    すべてのエンジンの依存（pyproject の optional extra 全部）を入れる。既定では、既定の構成で使う
    extra と、既に入っている（またはエンジン管理で有効にした）エンジンの extra だけを入れ、
    ほかは GUI の「エンジン管理」で後から入れる。tts-kokoro・tts-tsukasa は Windows では
    ソースからのビルドがあり、MSVC Build Tools と CMake が要る。

.PARAMETER InstallPrereqs
    足りない前提を winget で入れる（Python 3.11・Node.js、ビルドが要る構成なら VS 2022 Build Tools の C++ と CMake。
    無いものだけ）。Build Tools は数 GB あり、管理者の確認が出る。既定では入れず、足りないものと入れ方を示すだけ。

.PARAMETER SkipTests
    最後の検証で tests/ の unittest を実行しない（sherpa の版・ワーカーの import・GUI 構築は確かめる）。
    -VerifyOnly と組み合わせてもよい。

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
    [switch]$AllExtras,
    [switch]$InstallPrereqs,
    [switch]$SkipTests,
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
function Get-SmartAppControlState {
    # On / Off / Eval。Defender の無い環境などで取れなければ $null。
    try {
        $st = (Get-MpComputerStatus -ErrorAction Stop).SmartAppControlState
        if ($st) { return "$st" }
    } catch {}
    return $null
}

function Get-MsvcInstallPath {
    # MSVC の C++ ツールセット（VCTools）を含む VS / Build Tools の場所。無ければ $null。
    # vswhere は VS / Build Tools のインストーラーが置く。
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($path) { return "$path".Trim() }
    return $null
}

function Test-MsvcBuildTools { return [bool](Get-MsvcInstallPath) }

function Test-Python311 {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $found = (& uv python find 3.11 2>$null)
        if ($LASTEXITCODE -eq 0 -and $found) { return "$found".Trim() }
        return $null
    } finally { $ErrorActionPreference = $prevEap }
}

function Update-PathFromRegistry {
    # winget で入れたものは、今の PowerShell の PATH には入らない。レジストリから読み直す。
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ";"
}

function Install-WingetPackage([string]$id, [string[]]$extra = @()) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget が無いので $id を自動で入れられません。手で入れてから再実行してください。"
    }
    Write-Host "[install] winget install $id（管理者の確認が出たら許可してください）"
    $wingetArgs = @("install", "--id", $id, "--exact", "--source", "winget",
                    "--accept-package-agreements", "--accept-source-agreements") + $extra
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & winget @wingetArgs } finally { $ErrorActionPreference = $prev }
    $rc = $LASTEXITCODE
    Update-PathFromRegistry
    if ($rc -ne 0) { Write-Warn "winget install $id が終了コード $rc を返しました（下で入ったかを確かめます）。" }
}

# 既定で入れる extra（既定の設定：ASR は ReazonSpeech〔sherpa-onnx〕、TTS は VOICEVOX Core）。
# デバイスの extra（cu128 / cpu）は Invoke-UvSync で足す。
$baseExtras = @("gui", "web", "asr-sherpa", "tts-voicevox")
# Windows ではソースからのビルドがある extra（misaki[ja] → pyopenjtalk、monotonic-align）。
$compilerExtras = @("tts-kokoro", "tts-tsukasa")

function Get-SyncExtras {
    # uv sync に渡す extra（デバイスの extra は含まない）。-AllExtras なら $null（= --all-extras）。
    # uv sync は渡さなかった extra を消すので、既に入っているエンジンと、エンジン管理で有効にした
    # エンジンの extra は残す（以前の --all-extras で作った環境を再実行しても、何も消さない）。
    if ($AllExtras) { return $null }
    $extras = @($baseExtras)
    if (Test-Path $venvPython) {
        $code = "from maai_sds import engine_manager as em; k={e.id: e for e in em.list_engines()}; " +
                "x={e.extra for e in k.values() if e.installed and e.extra}; " +
                "x|={k[i].extra for i in em.get_enabled_engines() if i in k and k[i].extra}; " +
                "print(' '.join(sorted(x)))"
        $out = Invoke-VenvPython $code
        if ($LASTEXITCODE -eq 0) {
            $last = (($out -split "`n") | Where-Object { $_.Trim() } | Select-Object -Last 1)
            foreach ($e in ("$last".Trim() -split "\s+")) {
                if ($e -match '^[a-z0-9-]+$') { $extras += $e }
            }
        } else {
            Write-Warn "入っているエンジンを調べられなかった（既定の extra だけを入れる）: $out"
        }
    }
    return @($extras | Select-Object -Unique)
}

# -InstallPrereqs のときに winget で入れるもの（足りないものだけ）。
$wingetBuildTools = @("Microsoft.VisualStudio.2022.BuildTools",
    @("--override", "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"))

function Enter-MsvcEnvironment {
    # MSVC の環境（cl.exe・INCLUDE・LIB など）を、この PowerShell に読み込む。
    # これで Developer PowerShell for VS から実行しなくても、uv sync 中のソースビルドが MSVC を見つけられる
    # （今の PowerShell から起動する uv・cmake・子の powershell はこの環境を引き継ぐ）。
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { Write-Ok "MSVC の環境は読み込み済み"; return }
    $vs = Get-MsvcInstallPath
    if (-not $vs) { return }
    $dll = Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    if (-not (Test-Path $dll)) { Write-Warn "見つからない: $dll（MSVC の環境を読み込まずに続けます）"; return }
    # VsDevCmd の中で vswhere を PATH から呼ぶ箇所がある（無いと「認識されていません」と出る）
    $installerDir = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    if (($env:Path -split ";") -notcontains $installerDir) { $env:Path = "$installerDir;$env:Path" }
    try {
        Import-Module $dll
        Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null
        if (Get-Command cl.exe -ErrorAction SilentlyContinue) { Write-Ok "MSVC の環境を読み込んだ ($vs)" }
        else { Write-Warn "MSVC の環境を読み込んだが cl.exe が見つからない" }
    } catch {
        Write-Warn "MSVC の環境を読み込めなかった: $_（Developer PowerShell for VS 2022 から実行すると確実です）"
    }
}

function Test-BuildTools {
    # MSVC と CMake。Windows 向けのビルド済みパッケージが無く uv sync の中でソースからビルドするのは、
    # tts-kokoro（misaki[ja] → pyopenjtalk）と tts-tsukasa（monotonic-align）だけ。既定の構成では要らない。
    $needCompiler = [bool]($syncExtras | Where-Object { $compilerExtras -contains $_ })
    if ($AllExtras -or $ForceSherpaBuild) { $needCompiler = $true }
    if (-not $needCompiler) {
        $lack = @()
        if (-not (Test-MsvcBuildTools)) { $lack += "MSVC Build Tools" }
        if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { $lack += "CMake" }
        if ($lack.Count -gt 0) {
            Write-Host ("[info] $($lack -join ' と ') が無い（既定の構成では不要。Kokoro・Tsukasa を入れるとき、" +
                        "sherpa-onnx をソースからビルドするときに必要）")
        } else {
            Write-Ok "MSVC Build Tools と CMake"
        }
        return
    }
    if ($InstallPrereqs) {
        if (-not (Test-MsvcBuildTools)) { Install-WingetPackage $wingetBuildTools[0] $wingetBuildTools[1] }
        if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Install-WingetPackage "Kitware.CMake" }
    }
    $needBuild = @()
    if (-not (Test-MsvcBuildTools)) { $needBuild += "Visual Studio 2022 Build Tools（C++。winget install Microsoft.VisualStudio.2022.BuildTools --override `"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`"）" }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { $needBuild += "CMake（winget install Kitware.CMake）" }
    if ($needBuild.Count -gt 0) {
        $msg = "ソースからのビルドに必要なものが見つからない:`n  - " + ($needBuild -join "`n  - ") +
               "`n  -InstallPrereqs を付けて再実行すると winget で入れます（手で入れた場合は、新しい PowerShell を開いて再実行）。"
        if (-not (Test-Path $venvPython)) { throw $msg }
        Write-Warn $msg
    } else {
        Write-Ok "MSVC Build Tools と CMake"
    }
}

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

    # スマート アプリ コントロールがオンだと、署名の無い exe / DLL / .pyd が起動・読み込みを拒まれる。
    # uv が取ってくる python-build-standalone の python.exe（os error 4551）や、
    # ビルドした拡張モジュールの DLL load failed で、uv sync が途中で止まる（実機で発生）。
    $sac = Get-SmartAppControlState
    if ($sac -eq "On") {
        throw ("スマート アプリ コントロールがオンです。署名の無い Python や拡張モジュールが止められ、uv sync が失敗します。`n" +
               "  Windows セキュリティ → アプリとブラウザーの制御 → スマート アプリ コントロール でオフにしてから再実行してください。")
    } elseif ($sac -eq "Eval") {
        Write-Warn "スマート アプリ コントロールが評価モードです。途中でオンに切り替わると、署名の無い Python や拡張モジュールが止められます。"
    }

    # Python 3.11。.python-version で 3.11 を指定している（配布している sherpa-onnx のビルド済みパッケージが cp311 向け）。
    # 見つからなければ uv が python-build-standalone を取ってくる（署名が無い）。python.org の公式版を勧める。
    $py311 = Test-Python311
    if ((-not $py311) -and $InstallPrereqs) {
        Install-WingetPackage "Python.Python.3.11"
        $py311 = Test-Python311
    }
    if ($py311) {
        Write-Ok "Python 3.11: $py311"
    } else {
        Write-Warn ("Python 3.11 が見つからないので、uv が python-build-standalone を取ってきます。" +
                    "python.org の公式版を勧めます（-InstallPrereqs を付けるか、winget install Python.Python.3.11）。")
    }

    Test-BuildTools

    # Node.js。テストの一部（Web UI の JavaScript）だけが使う。無ければそのテストは飛ばす。
    if ((-not (Get-Command node -ErrorAction SilentlyContinue)) -and $InstallPrereqs) {
        Install-WingetPackage "OpenJS.NodeJS.LTS"
    }
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Write-Warn "node が無いので、Web UI の JavaScript のテストは飛ばします（-InstallPrereqs か winget install OpenJS.NodeJS.LTS）。"
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
    Write-Step "1. uv sync（本体 venv）"
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
        # extra を渡す必要がある。PySide6（gui）と各 ASR/TTS エンジンの依存は optional
        # extra なので、素の `uv sync` では 1 つも入らず、GUI が起動せず、この
        # スクリプト自身の検証（GUI 構築）も落ちる。
        # 既定は Get-SyncExtras の分だけ。以前は --all-extras だったが、Kokoro と Tsukasa の
        # 依存が Windows ではソースからのビルドになり、使わない人にも MSVC と CMake を要求していた。
        # cpu / cu128 は排他 extra（pyproject の [tool.uv] conflicts）なので、
        # --all-extras のときは "Extras `cpu` and `cu128` are incompatible" で落ちないよう、
        # 使わない方を --no-extra で外す。
        $deviceExtra = if ($Device -eq "cpu") { "cpu" } else { "cu128" }
        $dropExtra   = if ($Device -eq "cpu") { "cu128" } else { "cpu" }
        Write-Host "[env] デバイス構成: $Device (extra=$deviceExtra)"
        Set-Content -Path (Join-Path $projectRoot ".device_extra") -Value $deviceExtra -Encoding ascii -NoNewline
        if ($null -eq $syncExtras) {
            Write-Host "[env] すべての extra を入れる（-AllExtras）"
            & uv sync --all-extras --no-extra $dropExtra
        } else {
            $uvArgs = @("sync")
            foreach ($e in @($syncExtras) + $deviceExtra) { $uvArgs += @("--extra", $e) }
            Write-Host "[env] extra: $((@($syncExtras) + $deviceExtra) -join ', ')（ほかのエンジンは「エンジン管理」で入れる）"
            & uv @uvArgs
        }
        if ($LASTEXITCODE -ne 0) {
            throw ("uv sync が失敗した (exit $LASTEXITCODE)`n" +
                   "  ビルドに失敗したあとで MSVC や CMake を入れた場合は、失敗したときのビルド設定が uv のキャッシュに残っています。`n" +
                   "  uv cache clean <パッケージ名>（例: uv cache clean pyopenjtalk）で消してから再実行してください。`n" +
                   "  それでも失敗するときは、Developer PowerShell for VS 2022 から実行してください。")
        }
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
    if ($SkipTests) {
        Write-Warn "テストは実行しない（-SkipTests）"
    } elseif (Test-Path $testsDir) {
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

# uv sync に渡す extra を先に決める（ビルドツールが要るかの判断にも使う）
$syncExtras = Get-SyncExtras
Test-Prereqs
# Developer PowerShell でなくても、uv sync 中のソースビルドが MSVC を見つけられるようにする
Enter-MsvcEnvironment

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
