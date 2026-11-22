<#
.SYNOPSIS
    エンジンごとのモデル / バイナリ資産を自動取得する（uv sync では入らない部分）。

.DESCRIPTION
    pyproject.toml（uv sync）で入るのは Python パッケージまで。各エンジンが
    実行時に要求するファイルは別配布なので、このスクリプトがそれらを
    「あるか確認 → 無ければ入手」する。既存の環境には触れない（冪等）。

    既定で取るもの（既定の設定の sherpa-onnx と VOICEVOX-Core で対話できるところまで）:
      ASR   : sherpa-onnx 日本語 zipformer（ReazonSpeech）/ smart-turn ONNX /
              unidic 辞書（misaki[ja] が要求。unidic 再インストールで消える）/
              faster-whisper モデル（-PreFetchWhisper のときだけ）
      TTS   : VOICEVOX Core ランタイム（ORT DLL・OpenJTalk 辞書・VVM）
    ほかのエンジンの資産（Qwen3-ASR ONNX / Nemotron ONNX / SenseVoice などの ONNX モデル、
    Tsukasa-HF、AivisSpeech、Supertonic・Qwen3-TTS-CPP のサーバーと GGUF）は既定では取らない。
    GUI の「エンジン管理」でチェックして「適用」すると、-Only <ステップ> で取る。

    Irodori-TTS の PyTorch 版は専用 venv 方式（scripts/setup_irodori_venv.ps1）で、ここでは扱わない。

    カバーしないもの（他の経路で入れる）:
      - Python パッケージ全般        → uv sync（setup_env.ps1 のステップ 1）
      - カスタム sherpa-onnx ビルド  → setup_env.ps1 のステップ 2（Git 管理外スクリプト要）
      - .venv_nemotron               → setup_env.ps1 のステップ 3
      - qwen3-asr（vLLM サーバー）   → scripts/start_qwen3_asr.ps1（Docker）
      - chirp3-asr の GCP 認証       → GOOGLE_APPLICATION_CREDENTIALS 等を手動設定

.PARAMETER SkipSherpaModels
    sherpa-onnx 日本語 zipformer（既定の ASR）を取得しない。
.PARAMETER SkipSmartTurn
    smart-turn ONNX を取得しない。
.PARAMETER SkipVoicevoxRuntime
    VOICEVOX ランタイム（ORT DLL / 辞書 / VVM）を取得しない。
.PARAMETER PreFetchWhisper
    faster-whisper モデル（deepdml/faster-whisper-large-v3-turbo-ct2）を
    先行ダウンロードする（省略時は初回起動時に自動ダウンロードされる）。
.PARAMETER Qwen3OnnxUrl
    Qwen3-ASR 0.6B int8 の tar.bz2 直 URL。既定は k2-fsa 公開版。
    fp32 / 1.7B は GitHub に無いので ModelScope から別ステップで取る
    （Qwen3AsrOnnxFp32Model / Qwen3AsrOnnx17BModel）。
.PARAMETER Only
    指定した単一ステップだけを実行する。通常の連続実行と最終検証は行わない。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_assets.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_assets.ps1 -PreFetchWhisper
#>

param(
    [switch]$SkipSherpaModels,
    [switch]$SkipSmartTurn,
    [switch]$SkipVoicevoxRuntime,
    [switch]$PreFetchWhisper,
    [string]$Qwen3OnnxUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
    [string]$Only = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest の進捗表示が桁違いに遅いので止める

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$venvPython  = Join-Path $projectRoot ".venv\Scripts\python.exe"
$tmpRoot     = Join-Path $projectRoot "tmp\setup_assets"
# sherpa-onnx のビルド済みパッケージを置いている GitHub リポジトリ（Releases）。フォークでは差し替える
$releaseRepo = if ($env:MAAI_SDS_RELEASE_REPO) { $env:MAAI_SDS_RELEASE_REPO } else { "MaAI-Kyoto/MaAI-SDS" }

function Write-Step([string]$m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Warn([string]$m) { Write-Host "[warn] $m" -ForegroundColor Yellow }
function Write-Ok([string]$m)   { Write-Host "[ok] $m" -ForegroundColor Green }

# -------------------------------------------------------------------- 共通処理
function Invoke-Download([string]$url, [string]$dest) {
    $dir = Split-Path -Parent $dest
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Write-Host "[get] $url"
    & curl.exe -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 -o "$dest.part" $url
    if ($LASTEXITCODE -ne 0) { throw "download 失敗: $url" }
    Move-Item -Force "$dest.part" $dest
}

function Expand-Zip([string]$zip, [string]$dest) {
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Expand-Archive -Path $zip -DestinationPath $dest -Force
}

function Expand-Tar([string]$archive, [string]$dest) {
    # Windows 同梱の bsdtar は .tar.gz / .tar.bz2 / .tgz をそのまま開ける。
    # PATH 解決に頼らず System32 のものを直接呼ぶ — Git for Windows が入っている
    # と usr\bin\tar.exe（Unix tar）が先に解決され、"D:\..." を "D:" という
    # リモートホストと誤認して失敗する（実機で発生: "Cannot connect to D: resolve failed"）。
    $bsdtar = Join-Path $env:SystemRoot "System32\tar.exe"
    if (-not (Test-Path $bsdtar)) { $bsdtar = "tar.exe" }  # 無い環境向けの最後の手段
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    & $bsdtar -xf $archive -C $dest
    if ($LASTEXITCODE -ne 0) { throw "展開失敗: $archive" }
}

function Get-GitHubReleaseAsset([string]$repo, [string]$assetPattern) {
    # パターンに一致するアセットを **全リリース横断で** 探し、最も新しいものを返す。
    #
    # `releases/latest` だけを見ていると取り逃す。VOICEVOX/onnxruntime-builder は
    # latest が 1.17.3（2025-03）を指したままで、実際には 1.23.2（2026-07）が出ている。
    # 1.17.3 の CUDA ビルドは Blackwell（compute capability 12.0）のカーネルを持たず、
    #   CUDA error cudaErrorNoKernelImageForDevice
    # で推論できなかった（実機で発生。1.23.2 に上げると CPU 639ms → GPU 64ms）。
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases?per_page=100" `
        -Headers @{ "User-Agent" = "maai-sds-setup" } -TimeoutSec 30
    $hit = $releases |
        Where-Object { -not $_.draft } |
        Sort-Object -Property published_at -Descending |
        ForEach-Object {
            $a = $_.assets | Where-Object { $_.name -like $assetPattern } | Select-Object -First 1
            if ($a) { [pscustomobject]@{ Tag = $_.tag_name; Url = $a.browser_download_url } }
        } |
        Select-Object -First 1
    if (-not $hit) { throw "$repo に '$assetPattern' なアセットを持つリリースがありません" }
    Write-Host "[get] $repo $($hit.Tag)"
    return $hit.Url
}

# 展開先ディレクトリの中から条件に合うファイルを探し、見つかった実体の場所を返す
function Find-FileUnder([string]$root, [string]$fileName) {
    return Get-ChildItem -Path $root -Recurse -Filter $fileName -File -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

# ---------------------------------------------------------- ASR: ONNX モデル群
function Get-HostPython {
    if (Test-Path $venvPython) { return $venvPython }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "python が見つかりません（.venv か PATH 上の python）"
}

function Invoke-Qwen3OnnxModelScope([string]$Variant) {
    $py = Get-HostPython
    $script = Join-Path $scriptDir "download_qwen3_onnx.py"
    & $py -B $script --root $projectRoot --variant $Variant
    if ($LASTEXITCODE -ne 0) { throw "Qwen3 ONNX ($Variant) の取得に失敗した" }
}

function Invoke-Qwen3AsrOnnxModel {
    Write-Step "Qwen3-ASR 0.6B int8（qwen3-asr-onnx）"
    # GitHub 公開物。encoder/decoder は encoder.int8.onnx / decoder.int8.onnx。
    $tarName = [IO.Path]::GetFileName($Qwen3OnnxUrl)
    Get-SherpaTarball $tarName "conv_frontend.onnx"
}

function Invoke-Qwen3AsrOnnxFp32Model {
    Write-Step "Qwen3-ASR 0.6B fp32（ModelScope）"
    Invoke-Qwen3OnnxModelScope "0.6b-fp32"
}

function Invoke-Qwen3AsrOnnx17BModel {
    Write-Step "Qwen3-ASR 1.7B int8（ModelScope）"
    Invoke-Qwen3OnnxModelScope "1.7b-int8"
}

function Get-SherpaTarball([string]$tarName, [string]$markerFile) {
    $target = Join-Path $projectRoot ([IO.Path]::GetFileNameWithoutExtension($tarName))
    if ($target.EndsWith(".tar")) { $target = $target.Substring(0, $target.Length - 4) }
    if ((Test-Path (Join-Path $target $markerFile))) {
        Write-Ok "既にあるためスキップ: $target"
        return
    }
    $url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$tarName"
    $tarPath = Join-Path $tmpRoot $tarName
    Invoke-Download $url $tarPath
    # アーカイブ自体が単一のトップレベルフォルダ（$target と同名）を含むので、
    # $projectRoot に直接展開する（$tmpRoot に展開すると $target の場所と食い違う）。
    Expand-Tar $tarPath $projectRoot
    Remove-Item $tarPath -Force
    if (-not (Test-Path (Join-Path $target $markerFile))) {
        throw "展開したのに $markerFile が見つかりません: $target"
    }
    Write-Ok "配置完了: $target"
}

function Install-NemotronPreprocessor([string]$ModelDir) {
    $assetDir = Join-Path $projectRoot "assets\nemotron-3.5-asr-streaming-preprocessor"
    $expectedHashes = @{
        "preprocessor.onnx" = "A21D8788D37A71CEF45050988BD9BBD4B19112E60BD7958D52CE1CFE314963C1"
        "preprocessor.export.onnx.data" = "272F9ED95B43E15437DF20B7D124DAA6A8C831D8CDDE3F40E6E71EC3ACD06E38"
    }
    if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) {
        throw "Nemotron model directory が見つかりません: $ModelDir"
    }

    $updated = $false
    foreach ($fileName in @("preprocessor.onnx", "preprocessor.export.onnx.data")) {
        $source = Join-Path $assetDir $fileName
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Nemotron preprocessor asset が見つかりません: $source"
        }
        $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
        if ($sourceHash -ne $expectedHashes[$fileName]) {
            throw "Nemotron preprocessor asset の SHA-256 が不正です: $source"
        }

        $destination = Join-Path $ModelDir $fileName
        $destinationHash = if (Test-Path -LiteralPath $destination -PathType Leaf) {
            (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        } else {
            ""
        }
        if ($destinationHash -ne $sourceHash) {
            Copy-Item -LiteralPath $source -Destination $destination -Force
            $updated = $true
        }
    }
    if ($updated) { Write-Ok "Nemotron preprocessor 配置: $ModelDir" }
}

function Invoke-SherpaModels {
    Write-Step "sherpa-onnx 日本語 zipformer（ReazonSpeech。既定の ASR）"
    # Qwen3-ASR ONNX（Qwen3AsrOnnxModel）と Nemotron ONNX（NemotronOnnxModels）は、
    # エンジン管理で選んだときだけ取る。
    Get-SherpaTarball "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2" "tokens.txt"
}

function Invoke-NemotronOnnxModels {
    Write-Step "Nemotron 3.5 streaming ONNX（80/320/560/1120 ms int8）"
    # GUI は chunk に合わせてディレクトリ名の -320ms- を差し替えるので、4 つとも取る。
    # release tar の配布物は encoder.int8.onnx 等。preprocessor.onnx と external data は
    # release に無いため、リポジトリの assets から配置する。
    foreach ($ms in @(80, 320, 560, 1120)) {
        $dirName = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-${ms}ms-int8-2026-06-11"
        Get-SherpaTarball "$dirName.tar.bz2" "encoder.int8.onnx"
        Install-NemotronPreprocessor (Join-Path $projectRoot $dirName)
    }
}

# sherpa-onnx OfflineRecognizer 系 4 エンジンのモデル。どれも展開するだけで使える
# （中のファイル名は asr/engines_sherpa_offline._pick が解決する）。エンジンごとに
# 分けてあるのは、Engine Manager が選択されたものだけ取得できるようにするため。
function Invoke-ParakeetJaModel {
    Write-Step "Parakeet TDT-CTC 0.6B 日本語 ONNX（parakeet-onnx）"
    Get-SherpaTarball "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2" "model.int8.onnx"
}

function Invoke-SenseVoiceModel {
    Write-Step "SenseVoiceSmall ONNX（sensevoice-onnx）"
    # **2024-07-17 を使うこと**。同じ命名の 2025-09-09 は ASLP-lab/WSYue-ASR の広東語
    # ファインチューンで、日本語音声を簡体字で書き起こす（実測 CER 0.79）。
    Get-SherpaTarball "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2" "model.int8.onnx"
}

function Invoke-FunAsrNanoModel {
    Write-Step "Fun-ASR-Nano-2512 ONNX（funasr-nano-onnx）"
    Get-SherpaTarball "sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2" "llm.int8.onnx"
}

function Invoke-CohereAsrModel {
    Write-Step "Cohere Transcribe 14 言語 ONNX（cohere-asr-onnx）"
    # encoder は ONNX external data 形式で、展開後は約 2.8GB になる。
    Get-SherpaTarball "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2" "tokens.txt"
}

# --------------------------------------------------- speech.cpp（ASR / TTS の実行系）
# speech.cpp（MIT、https://github.com/nyosegawa/speech.cpp）。GPU は Vulkan で動くので CUDA は要らない。
# 公式のインストーラーはユーザーの PATH を書き換えるので使わず、版と SHA-256 を固定して zip を取る。
# 版を上げるときは src/maai_sds/speechcpp.py の VERSION と、setup_assets.sh の値もそろえる。
$SpeechCppVersion = "0.11.1"
$SpeechCppZipSha256 = "8a70018286813a06b72dc75af45a15f2b084e7e477915ed9d34f3cf6759cfd44"

function Invoke-SpeechCppRuntime {
    Write-Step "speech.cpp $SpeechCppVersion（Windows x64・Vulkan）"
    $dir = Join-Path $projectRoot "vendor\speechcpp"
    $exe = Join-Path $dir "speech.exe"
    if (Test-Path $exe) {
        $current = (& $exe --version 2>$null | Out-String)
        if ($current -match [regex]::Escape("speech.cpp $SpeechCppVersion")) {
            Write-Ok "既にあるためスキップ: $exe"
            return
        }
        Write-Warn "版が違うので入れ替えます: $($current.Trim())"
    }
    $url = "https://github.com/nyosegawa/speech.cpp/releases/download/v$SpeechCppVersion/speech-$SpeechCppVersion-windows-x64-vulkan.zip"
    $zip = Join-Path $tmpRoot "speechcpp-windows.zip"
    Invoke-Download $url $zip
    $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($hash -ne $SpeechCppZipSha256) {
        Remove-Item $zip -Force
        throw "speech.cpp の zip の SHA-256 が違います: $hash（期待値 $SpeechCppZipSha256）"
    }
    Expand-Zip $zip $dir
    Remove-Item $zip -Force
    if (-not (Test-Path $exe)) { throw "展開後も speech.exe がありません: $exe" }
    Write-Ok "配置完了: $exe"
}

# モデルは speech.cpp 自身が Hugging Face から取る（置き場所は speechcpp_models）。
# 置き場所を深くすると Windows の 260 文字の制限に掛かる（"Windows error 3"）ので、プロジェクト直下に置く。
function Invoke-SpeechCppPull([string]$model) {
    $exe = Join-Path $projectRoot "vendor\speechcpp\speech.exe"
    if (-not (Test-Path $exe)) { Invoke-SpeechCppRuntime }
    $models = Join-Path $projectRoot "speechcpp_models"
    if ($env:MAAI_SDS_SPEECHCPP_MODELS) { $models = $env:MAAI_SDS_SPEECHCPP_MODELS }
    if ($models.Length -gt 95) {
        Write-Warn ("モデルの置き場所のパスが長すぎます（{0} 文字）: {1}。Windows の 260 文字の制限で取得に失敗することがあります。" -f $models.Length, $models)
        Write-Warn "環境変数 MAAI_SDS_SPEECHCPP_MODELS に短い場所（例 C:\speechcpp_models）を指定するか、MaAI-SDS を浅いフォルダーに置いてください。"
    }
    New-Item -ItemType Directory -Path $models -Force | Out-Null
    $env:SPEECH_MODEL_DIR = $models
    & $exe pull $model
    if ($LASTEXITCODE -ne 0) { throw "speech.cpp のモデルを取得できません: $model" }
    Write-Ok "取得済み: $model（$models）"
}

function Invoke-SpeechCppAsrModel {
    Write-Step "speech.cpp の ASR モデル（Haas-ja v1.0）"
    Invoke-SpeechCppPull "haas-ja-v1.0"
}

function Invoke-SpeechCppIrodoriModel {
    Write-Step "speech.cpp の TTS モデル（Irodori-TTS v4.1 Small MF・q8_0）"
    Invoke-SpeechCppPull "irodori-tts-mf:q8_0"
}

function Invoke-SmartTurn {
    Write-Step "smart-turn ONNX"
    $target = Join-Path $projectRoot "smart-turn-v3.2-gpu.onnx"
    if (Test-Path $target) { Write-Ok "既にあるためスキップ: $target"; return }
    Invoke-Download "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx" $target
    Write-Ok "配置完了: $target"
}

function Invoke-WhisperPrefetch {
    if (-not $PreFetchWhisper) { return }
    Write-Step "faster-whisper モデルの先行ダウンロード"
    & $venvPython -B -c "from huggingface_hub import snapshot_download; p=snapshot_download('deepdml/faster-whisper-large-v3-turbo-ct2'); print('OK', p)"
    if ($LASTEXITCODE -ne 0) { throw "faster-whisper モデルのダウンロードに失敗した" }
    Write-Ok "HF キャッシュへダウンロード済み"
}

function Invoke-UnidicDict {
    # unidic パッケージは辞書を含まない。`python -m unidic download` が別途要る。
    # 辞書は site-packages 内へ展開されるため、unidic が再インストールされると消える
    # （uv sync の再実行後など）。
    Write-Step "unidic 辞書"
    $dicrc = Join-Path $projectRoot ".venv\Lib\site-packages\unidic\dicdir\dicrc"
    if (-not (Test-Path $venvPython)) {
        Write-Warn ".venv が無いためスキップ（先に uv sync すること）"
        return
    }
    if (Test-Path $dicrc) { Write-Ok "既にあるためスキップ: $dicrc"; return }
    & $venvPython -B -m unidic download
    if ($LASTEXITCODE -ne 0) { throw "unidic download に失敗した" }
    if (-not (Test-Path $dicrc)) { throw "download 後も dicrc が無い: $dicrc" }
    Write-Ok "ダウンロード完了"
}

# --------------------------------------------------- TTS: VOICEVOX ランタイム
function Invoke-VoicevoxRuntime {
    Write-Step "VOICEVOX ランタイム（ORT DLL / OpenJTalk 辞書 / VVM）"
    $rt = Join-Path $projectRoot "voicevox_runtime"

    # (a) voicevox_onnxruntime DLL。VOICEVOX 独自ビルドで、本家 onnxruntime では代用不可。
    #     設定は voicevox_onnxruntime_v.dll という名前を指しているので両名で置く。
    $dllTarget = Join-Path $rt "onnxruntime\lib\voicevox_onnxruntime_v.dll"
    # 「あればスキップ」だけでは古い版が残り続ける。1.17.3 の CUDA ビルドは
    # Blackwell（compute capability 12.0）のカーネルを持たず
    #   CUDA error cudaErrorNoKernelImageForDevice
    # で推論できない（実機で発生）。DLL は FileVersion を持っているので、その版を見る。
    $needsFetch = -not (Test-Path $dllTarget)
    if (-not $needsFetch) {
        $ver = (Get-Item $dllTarget).VersionInfo.FileVersion
        $parsed = $null
        if ($ver -and [version]::TryParse(($ver -split '-')[0], [ref]$parsed)) {
            if ($parsed -lt [version]"1.23.0") {
                Write-Warn "既存の ONNX Runtime が古い（$ver）。Blackwell 世代の GPU で"
                Write-Warn "カーネルが無く推論できないため取り直します。"
                $needsFetch = $true
            }
        } else {
            Write-Warn "既存 DLL の版を読めなかったので取り直します（$ver）"
            $needsFetch = $true
        }
    }
    if (-not $needsFetch) {
        Write-Ok "既にあるためスキップ: $dllTarget"
    } else {
        # 取り直す前に古い DLL を消す。種類を変えた場合（CUDA → DirectML 等）に
        # 使われないプロバイダが残ると、そちらが優先されて食い違う。
        Get-ChildItem (Join-Path $rt "onnxruntime\lib") -Filter "voicevox_onnxruntime*.dll" `
            -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        # どのランタイムを取るかは controller_config.json の voicevox_core_device に
        # 従う（GUI の VOICEVOX-Core 設定で選ぶ）。実測（RTX 5070 Ti Laptop, 10 回の
        # 定常値）: CUDA 52ms / DirectML 249ms / CPU 620ms。
        # CUDA は最速だが cudnn64_9.dll を外部依存として要する。DirectML は DLL 1 本で
        # 完結し GPU ベンダー非依存。設定が読めない場合は CUDA（従来の挙動）。
        $ortVariant = "cuda-"
        $cfgPath = Join-Path $projectRoot "controller_config.json"
        if (Test-Path $cfgPath) {
            try {
                $dev = [string]((Get-Content $cfgPath -Raw -Encoding utf8 | ConvertFrom-Json).voicevox_core_device)
                if ($dev -match 'DirectML') { $ortVariant = "dml-" }
                elseif ($dev -eq 'CPU')     { $ortVariant = "" }
            } catch {
                Write-Warn "controller_config.json を読めなかったので CUDA 版を取ります"
            }
        }
        $ortLabel = switch ($ortVariant) { "cuda-" { "CUDA" } "dml-" { "DirectML" } default { "CPU" } }
        Write-Host "[voicevox] ランタイム: $ortLabel"
        $url = Get-GitHubReleaseAsset "VOICEVOX/onnxruntime-builder" "voicevox_onnxruntime-win-x64-$ortVariant*.tgz"
        $tgz = Join-Path $tmpRoot ([IO.Path]::GetFileName($url))
        Invoke-Download $url $tgz
        $ex = Join-Path $tmpRoot "ort_extract"
        Expand-Tar $tgz $ex
        Remove-Item $tgz -Force
        $src = Find-FileUnder $ex "voicevox_onnxruntime.dll"
        if (-not $src) { throw "tgz 内に voicevox_onnxruntime.dll が見つかりません" }
        $libDir = Split-Path -Parent $dllTarget
        New-Item -ItemType Directory -Path $libDir -Force | Out-Null
        Copy-Item $src $dllTarget -Force
        Copy-Item $src (Join-Path $libDir "voicevox_onnxruntime.dll") -Force
        # **本体だけでは CUDA が使えない**。プロバイダは別 DLL で、
        # voicevox_onnxruntime_providers_cuda.dll（328MB）と ..._providers_shared.dll が
        # 同じディレクトリに無いと Synthesizer が GPU を要求した時点で
        # 「GPU機能をサポートすることができません」になり CPU に落ちる（実機で発生）。
        # onnxruntime 本家と同じ構造。
        Get-ChildItem -Path (Split-Path -Parent $src) -Filter "voicevox_onnxruntime_providers_*.dll" |
            ForEach-Object {
                Copy-Item $_.FullName (Join-Path $libDir $_.Name) -Force
                Write-Ok ("  プロバイダ配置: {0} ({1:N0} MB)" -f $_.Name, ($_.Length / 1MB))
            }
        Remove-Item $ex -Recurse -Force
        Write-Ok "配置完了: $dllTarget（+_v 無し名も同内容で配置）"
    }

    # (b) OpenJTalk 辞書。公式は SourceForge の tar.gz（ミラーは環境によって名前解決できる
    #     ホストが違うので、複数試す）。voicevox_core が読むのは sys.dic 等で dicrc は無くてよい。
    $dictTarget = Join-Path $rt "dict\open_jtalk_dic_utf_8-1.11"
    if (Test-Path (Join-Path $dictTarget "sys.dic")) {
        Write-Ok "既にあるためスキップ: $dictTarget"
    } else {
        $urls = @(
            "https://master.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz?viasf=1",
            "https://jaist.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz"
        )
        $tgz = Join-Path $tmpRoot "open_jtalk_dic_utf_8-1.11.tar.gz"
        $downloaded = $false
        foreach ($url in $urls) {
            try { Invoke-Download $url $tgz; $downloaded = $true; break }
            catch { Write-Warn "取得失敗、次のミラーを試す: $($_.Exception.Message)" }
        }
        if (-not $downloaded) { throw "OpenJTalk 辞書をどのミラーからも取得できませんでした" }
        Expand-Tar $tgz (Join-Path $rt "dict")
        Remove-Item $tgz -Force
        if (-not (Test-Path (Join-Path $dictTarget "sys.dic"))) { throw "辞書の展開に失敗した: $dictTarget" }
        Write-Ok "配置完了: $dictTarget"
    }

    # (c) VVM 音声モデル（speaker id 0）
    $vvmTarget = Join-Path $rt "models\vvms\0.vvm"
    if (Test-Path $vvmTarget) { Write-Ok "既にあるためスキップ: $vvmTarget"; return }
    Invoke-Download "https://github.com/VOICEVOX/voicevox_vvm/releases/download/0.16.4/0.vvm" $vvmTarget
    Write-Ok "配置完了: $vvmTarget"
}

# ------------------------------------------------------- TTS: ベンダーディレクトリ
function Invoke-TsukasaVendor([switch]$Nested, [switch]$ThrowOnFailure) {
    if (-not $Nested) { Write-Step "Tsukasa-HF ベンダーディレクトリ" }

    # Tsukasa-HF は公開リポジトリ Respair/Tsukasa_Speech から取る（トークン不要）。
    $t = Join-Path $projectRoot "Tsukasa-HF"
    if (Test-Path (Join-Path $t "importable.py")) {
        Write-Ok "既にあるためスキップ: Tsukasa-HF/"
    } elseif (-not (Test-Path $venvPython)) {
        $message = ".venv が無いため Tsukasa-HF をダウンロードできない（先に uv sync すること）"
        if ($ThrowOnFailure) { throw $message }
        Write-Warn $message
    } else {
        $tsukasaRepo = "Respair/Tsukasa_Speech"
        Write-Host "[get] HF snapshot: $tsukasaRepo"
        $code = @"
import sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], token=False)
    print('OK', p)
except Exception as ex:
    print('FAIL:', type(ex).__name__, str(ex)[:300])
    sys.exit(3)
"@
        $codePath = Join-Path $tmpRoot "tsukasa_dl.py"
        New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null
        Set-Content -Path $codePath -Value $code -Encoding utf8
        & $venvPython -B $codePath $tsukasaRepo $t
        if ($LASTEXITCODE -ne 0) {
            $message = "Tsukasa-HF を自動取得できませんでした（$tsukasaRepo）。ネットワークを確認して再実行してください。"
            Write-Warn $message
            if ($ThrowOnFailure) { throw "Tsukasa-HF のダウンロードに失敗した" }
        } else {
            Write-Ok "ダウンロード完了: Tsukasa-HF/"
        }
    }
}

function Invoke-VendorClones {
    Write-Step "ベンダーディレクトリ（Tsukasa-HF）"
    Invoke-TsukasaVendor -Nested
}

# ------------------------------------------------------------------ AivisSpeech
function Invoke-AivisSpeech {
    Write-Step "AivisSpeech（ポータブル ZIP をインストーラー標準の位置へ展開）"
    $engineExe = Join-Path $env:LOCALAPPDATA "Programs\AivisSpeech\AivisSpeech-Engine\run.exe"
    if (Test-Path $engineExe) { Write-Ok "既にあるためスキップ: $engineExe"; return }

    $url = Get-GitHubReleaseAsset "Aivis-Project/AivisSpeech" "AivisSpeech-Windows-x64-*.zip"
    $zip = Join-Path $tmpRoot ([IO.Path]::GetFileName($url))
    Invoke-Download $url $zip
    $ex = Join-Path $tmpRoot "aivis_extract"
    Expand-Zip $zip $ex
    Remove-Item $zip -Force

    # ZIP 内構造はバージョンで変わり得るので、run.exe の位置から実際のルートを求める
    $found = Find-FileUnder $ex "run.exe"
    if (-not $found) { throw "ZIP 内に AivisSpeech-Engine\run.exe が見つかりません" }
    $engineRoot = Split-Path -Parent $found                       # .../AivisSpeech-Engine
    $installSrc = Split-Path -Parent $engineRoot                  # Engine と Editor の親
    $destRoot   = Join-Path $env:LOCALAPPDATA "Programs\AivisSpeech"
    New-Item -ItemType Directory -Path $destRoot -Force | Out-Null
    # tmp はプロジェクト直下（D:）、インストール先は %LOCALAPPDATA%（C:）。
    # PS 5.1 の Move-Item はボリュームを跨げないので Copy + Remove で行う。
    Get-ChildItem $installSrc | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $destRoot $_.Name) -Recurse -Force
    }
    Remove-Item $installSrc -Recurse -Force
    Remove-Item $ex -Recurse -Force
    if (-not (Test-Path $engineExe)) { throw "展開後も run.exe が見つかりません: $engineExe" }
    Write-Ok "配置完了: $destRoot（GUI 既定パスと一致）"
}

# --------------------------------------------------------------------- audio.cpp
# ランタイム資産は Release ZIP / GGUF 直リンクで揃い、ソース clone には依存しない。
# 開発者が明示的にソース一式を必要とする場合の `-Only AudioCppVendor` 互換用にだけ残す。
function Invoke-AudioCppVendor([switch]$Nested) {
    if (-not $Nested) { Write-Step "audio.cpp ベンダーディレクトリ（ソース一式）" }

    # この明示ステップでは、完全なソース clone かどうかを model manager の有無で判定する。
    # Qwen3-TTS-CPP を含む通常のセットアップ経路からは呼ばれない。
    $audioCppRoot = Join-Path $projectRoot "audio.cpp"
    $modelManager = Join-Path $audioCppRoot "tools\model_manager_v2.py"
    if (Test-Path $modelManager) {
        Write-Ok "既にあるためスキップ: audio.cpp/"
    } else {
        # audio.cpp\ は AudioCppServer / SupertonicGguf が先に走って models\ 等を作っている
        # ことがあり得る（実測済み）。git clone は空でないディレクトリへ直接は使えないので、
        # 一時ディレクトリへ clone してから、まだ無いトップレベル項目だけをマージする。
        $tempClone = Join-Path $tmpRoot "audio_cpp_vendor_clone"
        if (Test-Path $tempClone) { Remove-Item -Recurse -Force $tempClone }
        & git clone --depth 1 https://github.com/0xShug0/audio.cpp.git $tempClone
        if ($LASTEXITCODE -ne 0) { throw "audio.cpp の clone に失敗した" }
        if (-not (Test-Path $audioCppRoot)) {
            New-Item -ItemType Directory -Path $audioCppRoot -Force | Out-Null
        }
        Get-ChildItem -Path $tempClone -Force | ForEach-Object {
            $dest = Join-Path $audioCppRoot $_.Name
            if (-not (Test-Path $dest)) {
                Move-Item -Path $_.FullName -Destination $dest
            }
        }
        Remove-Item -Recurse -Force $tempClone
        if (-not (Test-Path $modelManager)) { throw "clone 後も見つかりません: $modelManager" }
        Write-Ok "clone 完了: audio.cpp/"
    }
}

# ------------------------------------------------ VOICEVOX（アプリ版・ポータブル ZIP）
function Invoke-VoicevoxApp([switch]$Nested) {
    if (-not $Nested) { Write-Step "VOICEVOX アプリ（ポータブル ZIP をインストーラ相当の位置へ展開）" }

    # インストーラ版と同じ位置に置く。既にインストーラで入れている人はここで検出され、
    # そのままスキップになる（probe_paths も同じパスを見る）。
    $exe = Join-Path $env:LOCALAPPDATA "Programs\VOICEVOX\VOICEVOX.exe"
    if (Test-Path $exe) {
        Write-Ok "既にあるためスキップ: $exe"
        return
    }

    # CUDA 版は分割 7z + NSIS しか配布が無いので触らない。GPU 構成では DirectML 版を
    # 使う（DX12 ベースなので NVIDIA でも動く）。CPU 構成では cpu 版。
    $marker = Join-Path $projectRoot ".device_extra"
    $variant = "directml"
    if (Test-Path $marker) {
        if ((Get-Content $marker -Raw).Trim() -eq "cpu") { $variant = "cpu" }
    }
    Write-Host "[voicevox] 取得する版: $variant"

    $url = Get-GitHubReleaseAsset "VOICEVOX/voicevox" "voicevox-windows-$variant-*.zip"
    $zip = Join-Path $tmpRoot ([IO.Path]::GetFileName(($url -split '\?')[0]))
    Invoke-Download $url $zip
    $ex = Join-Path $tmpRoot "voicevox_app_extract"
    if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
    Expand-Zip $zip $ex
    Remove-Item $zip -Force

    # ZIP 内構造はバージョンで変わり得るので、VOICEVOX.exe の位置から実際のルートを求める
    $found = Find-FileUnder $ex "VOICEVOX.exe"
    if (-not $found) {
        Remove-Item -Recurse -Force $ex
        throw "ZIP 内に VOICEVOX.exe が見つかりません: $url"
    }
    $installSrc = Split-Path -Parent $found
    $destRoot   = Join-Path $env:LOCALAPPDATA "Programs\VOICEVOX"
    New-Item -ItemType Directory -Path $destRoot -Force | Out-Null
    # tmp はプロジェクト直下（D:）、インストール先は %LOCALAPPDATA%（C:）。
    # PS 5.1 の Move-Item はボリュームを跨げないので Copy + Remove で行う。
    Get-ChildItem $installSrc -Force | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $destRoot $_.Name) -Recurse -Force
    }
    Remove-Item -Recurse -Force $ex
    Write-Ok "配置完了: $destRoot"
}

# ------------------------------------------------ カスタム sherpa-onnx プリビルド
function Invoke-SherpaPrebuilt([switch]$Nested) {
    if (-not $Nested) { Write-Step "カスタム sherpa-onnx プリビルド" }

    # 公式 v1.13.5 に MaAI のパッチ（NeMo mel 再計算 / max_symbols=30）を当てた
    # ビルド。PyPI 版でも動くが日本語 ASR の品質が落ちるので、既定ではこれを使う。
    # 展開先は vendor/sherpa_onnx_prebuilt/。site-packages へ入れるのは
    # engine_manager の _restore_sherpa_snapshot で、`uv sync` の後に走る
    # （sync は毎回 PyPI 版で上書きするため、順序が決まっている）。
    $target = Join-Path $projectRoot "vendor\sherpa_onnx_prebuilt"
    if (Test-Path (Join-Path $target "sherpa_onnx")) {
        Write-Ok "既にあるためスキップ: $target"
        return
    }

    # 配布物は CPython 3.11 の win_amd64 向け拡張モジュール（_sherpa_onnx.cp311-win_amd64.pyd）。
    # pyproject の requires-python は ">=3.11" なので 3.12 以降でも venv は作れてしまうが、
    # その .pyd は読めない。合わない環境に置くと import 時に不可解に落ちるので、
    # 取得せずスキップして PyPI 版に任せる。
    if (Test-Path $venvPython) {
        $abi = & $venvPython -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and $abi.Trim() -ne "cp311") {
            Write-Warn "配布しているプリビルドは cp311 向けです（この venv は $($abi.Trim())）。取得せず PyPI 版を使います。自前ビルドは scripts\upgrade_sherpa_onnx.ps1。"
            return
        }
    }

    # 自前ビルドは Git + CMake + MSVC が要り 30 分以上かかるので、既定は配布物の取得。
    # 環境変数で差し替え可能にしておく（社内ミラーやローカル検証用）。
    $url = $env:MAAI_SDS_SHERPA_PREBUILT_URL
    if (-not $url) {
        $url = Get-GitHubReleaseAsset $releaseRepo "sherpa_onnx_prebuilt_*win_amd64*.zip"
    }
    $zip = Join-Path $tmpRoot ([IO.Path]::GetFileName(($url -split '\?')[0]))
    Invoke-Download $url $zip
    $ex = Join-Path $tmpRoot "sherpa_prebuilt_extract"
    if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
    Expand-Zip $zip $ex
    Remove-Item $zip -Force

    # ZIP 直下が sherpa_onnx/ か、1 段フォルダを噛んでいるかの両方を許容する。
    $root = $ex
    if (-not (Test-Path (Join-Path $root "sherpa_onnx"))) {
        $inner = Get-ChildItem -Path $ex -Directory | Where-Object {
            Test-Path (Join-Path $_.FullName "sherpa_onnx")
        } | Select-Object -First 1
        if ($inner) { $root = $inner.FullName }
    }
    if (-not (Test-Path (Join-Path $root "sherpa_onnx"))) {
        Remove-Item -Recurse -Force $ex
        throw "ZIP 内に sherpa_onnx/ が見つかりません: $url"
    }

    New-Item -ItemType Directory -Path $target -Force | Out-Null
    Get-ChildItem -Path $root -Force | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $target $_.Name) -Recurse -Force
    }
    Remove-Item -Recurse -Force $ex
    $ver = Join-Path $target "VERSION.txt"
    $label = if (Test-Path $ver) { (Get-Content $ver -Raw).Trim() } else { "(版不明)" }
    Write-Ok "配置完了: $target ($label)"
}

# audio.cpp（MIT、https://github.com/0xShug0/audio.cpp）。版を固定する。LFM2.5-Audio と Irodori の 500M v3 /
# 600M VoiceDesign は 0.9.x の新しい版でないと読めず（2026-08 の版には --version も無い）、最新を取る方式だと
# 版が日ごとに変わる。版を上げるときはここと、setup_assets.sh の AUDIOCPP_VERSION をそろえる。
$AudioCppVersion = "0.9.6"

function Invoke-AudioCppServer([switch]$Nested) {
    if (-not $Nested) { Write-Step "audio.cpp サーバー $AudioCppVersion（Windows x64・CUDA 12.4）" }

    # SupertonicEngine は「EXE の 4 つ上の親」を基準に動くため、audio.cpp\build\windows-cuda-release\bin\ という
    # 構造を再現する。
    $exeTarget = Join-Path $projectRoot "audio.cpp\build\windows-cuda-release\bin\audiocpp_server.exe"
    $binDir = Split-Path -Parent $exeTarget
    $current = ""
    if (Test-Path $exeTarget) {
        $current = (& $exeTarget --version 2>$null | Out-String)
    }
    if ($current -match [regex]::Escape("audio.cpp $AudioCppVersion")) {
        Write-Ok "既にあるためスキップ: $exeTarget"
    } else {
        if (Test-Path $exeTarget) { Write-Warn "版が違うので入れ替えます（$($current.Trim() -replace '\s+', ' ')）" }
        $base = "https://github.com/0xShug0/audio.cpp/releases/download/v$AudioCppVersion"
        $zip = Join-Path $tmpRoot "audiocpp-bin.zip"
        Invoke-Download "$base/audio-v$AudioCppVersion-bin-windows-x64-cuda12.4.zip" $zip
        $ex = Join-Path $tmpRoot "audiocpp_extract"
        if (Test-Path $ex) { Remove-Item $ex -Recurse -Force }
        Expand-Zip $zip $ex
        Remove-Item $zip -Force
        $found = Find-FileUnder $ex "audiocpp_server.exe"
        if (-not $found) { throw "ZIP 内に audiocpp_server.exe が見つかりません" }
        New-Item -ItemType Directory -Path $binDir -Force | Out-Null
        # EXE と同じフォルダーにある DLL（ggml-cuda.dll など）も一緒に置く
        Get-ChildItem (Split-Path -Parent $found) -File | ForEach-Object {
            Copy-Item $_.FullName (Join-Path $binDir $_.Name) -Force
        }
        Remove-Item $ex -Recurse -Force
        Write-Ok "配置完了: $exeTarget"
    }

    # CUDA のランタイム（cudart・cuBLAS。約 600 MB）。bin の ZIP には入っておらず、CUDA Toolkit の無い PC では
    # これが無いと起動できない。
    if (-not (Test-Path (Join-Path $binDir "cublas64_12.dll"))) {
        $base = "https://github.com/0xShug0/audio.cpp/releases/download/v$AudioCppVersion"
        $zip = Join-Path $tmpRoot "audiocpp-cudart.zip"
        Invoke-Download "$base/audio-v$AudioCppVersion-cudart-windows-x64-cuda12.4.zip" $zip
        $ex = Join-Path $tmpRoot "audiocpp_cudart"
        if (Test-Path $ex) { Remove-Item $ex -Recurse -Force }
        Expand-Zip $zip $ex
        Remove-Item $zip -Force
        Get-ChildItem $ex -Recurse -File -Filter "*.dll" | ForEach-Object {
            Copy-Item $_.FullName (Join-Path $binDir $_.Name) -Force
        }
        Remove-Item $ex -Recurse -Force
        Write-Ok "CUDA ランタイムを配置: $binDir"
    }
}

# audio.cpp の GGUF を 1 つ取る（あればスキップ）。
function Get-AudioCppModelFile([string]$url, [string]$relTarget) {
    $target = Join-Path $projectRoot $relTarget
    if ((Test-Path $target) -and ((Get-Item $target).Length -gt 0)) {
        Write-Ok "既にあるためスキップ: $target"
        return
    }
    Invoke-Download $url $target
    if (-not (Test-Path $target)) { throw "ダウンロード後もファイルがありません: $target" }
    Write-Ok "配置完了: $target"
}

# LFM2.5-Audio-1.5B-JP（Liquid AI。LFM Open License v1.0：年商 1000 万ドル未満なら商用可）。4 つのファイルを同じ
# フォルダーに置き、audio.cpp にはフォルダーを渡す。リビジョンを固定する。
function Invoke-Lfm2AudioGguf {
    Write-Step "LFM2.5-Audio-1.5B-JP GGUF（audio.cpp・約 1.8 GB）"
    $rev = "64b96718b341dbd5650f9e85627cecdcbd4ac61b"
    foreach ($f in @("LFM2.5-Audio-1.5B-JP-Q8_0.gguf", "mmproj-LFM2.5-Audio-1.5B-JP-Q8_0.gguf",
                     "vocoder-LFM2.5-Audio-1.5B-JP-Q8_0.gguf", "tokenizer-LFM2.5-Audio-1.5B-JP-Q8_0.gguf")) {
        Get-AudioCppModelFile "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF/resolve/$rev/$f" `
            "audio.cpp\models\LFM2.5-Audio-1.5B-JP-GGUF\$f"
    }
}

# Irodori-TTS の audio.cpp 版（500M v3 と、声を文章で指定する 600M v3 VoiceDesign。どちらも約 1.1〜1.3 GB）。
function Invoke-IrodoriAudioCppGguf {
    Write-Step "Irodori-TTS 500M v3 / 600M v3 VoiceDesign GGUF（audio.cpp）"
    $base = "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main"
    Get-AudioCppModelFile "$base/Irodori-TTS-500M-v3-GGUF/irodori-tts-500m-v3-q8_0.gguf" `
        "audio.cpp\models\Irodori-TTS-500M-v3-GGUF\irodori-tts-500m-v3-q8_0.gguf"
    Get-AudioCppModelFile "$base/Irodori-TTS-600M-v3-VoiceDesign-GGUF/irodori-tts-600m-v3-voicedesign-q8_0.gguf" `
        "audio.cpp\models\Irodori-TTS-600M-v3-VoiceDesign-GGUF\irodori-tts-600m-v3-voicedesign-q8_0.gguf"
}

function Invoke-SupertonicGguf([switch]$Nested) {
    if (-not $Nested) { Write-Step "Supertonic-3 GGUF" }

    # Supertonic-3 GGUF
    $ggufTarget = Join-Path $projectRoot "audio.cpp\models\Supertonic-3-GGUF\supertonic-3-orig.gguf"
    if (Test-Path $ggufTarget) { Write-Ok "既にあるためスキップ: $ggufTarget"; return }
    Invoke-Download "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Supertonic-3-GGUF/supertonic-3-orig.gguf" $ggufTarget
    Write-Ok "配置完了: $ggufTarget"
}

function Invoke-AudioCpp {
    Write-Step "audio.cpp サーバー EXE と Supertonic GGUF"
    Invoke-AudioCppServer -Nested
    Invoke-SupertonicGguf -Nested
}

function Invoke-Qwen3TTSCppGguf {
    Write-Step "Qwen3-TTS CustomVoice Q8 GGUF（audio.cpp）"
    $ggufTarget = Join-Path $projectRoot "audio.cpp\models\Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF\qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf"
    if (Test-Path $ggufTarget) {
        if ((Get-Item $ggufTarget).Length -gt 0) {
            Write-Ok "既にあるためスキップ: $ggufTarget"
            return
        }
        Write-Warn "既存ファイルのサイズが 0 のため再取得する: $ggufTarget"
    }

    Invoke-Download "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf" $ggufTarget
    if (-not (Test-Path $ggufTarget)) {
        throw "ダウンロード後も GGUF が見つかりません: $ggufTarget"
    }
    if ((Get-Item $ggufTarget).Length -le 0) {
        throw "ダウンロード後の GGUF が空です: $ggufTarget"
    }
    Write-Ok "配置完了: $ggufTarget"
}

# ------------------------------------------------------------------------ 検証
function Invoke-VerifyAssets {
    Write-Step "検証（配置確認）"
    # 既定のステップで取るものだけを確かめる（ほかのエンジンは、エンジン管理が入っているかを判定する）。
    # -Skip* で飛ばしたものは見ない。
    $checks = @()
    if (-not $SkipSherpaModels) {
        $checks += @{ Name = "sherpa zipformer ja"; Path = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01\tokens.txt" }
    }
    if (-not $SkipSmartTurn) {
        $checks += @{ Name = "smart-turn ONNX"; Path = "smart-turn-v3.2-gpu.onnx" }
    }
    if (-not $SkipVoicevoxRuntime) {
        $checks += @{ Name = "VOICEVOX ORT DLL"; Path = "voicevox_runtime\onnxruntime\lib\voicevox_onnxruntime_v.dll" }
        $checks += @{ Name = "OpenJTalk 辞書";   Path = "voicevox_runtime\dict\open_jtalk_dic_utf_8-1.11\sys.dic" }
        $checks += @{ Name = "VVM 0";            Path = "voicevox_runtime\models\vvms\0.vvm" }
    }
    $missing = @()
    foreach ($c in $checks) {
        $p = $c.Path
        $exists = Test-Path (Join-Path $projectRoot $p)
        if ($exists) {
            Write-Ok ("{0}: {1}" -f $c.Name, $p)
        } else {
            Write-Warn ("{0} が見つかりません: {1}" -f $c.Name, $p)
            $missing += $c.Name
        }
    }
    return $missing
}

# ---------------------------------------------------------------------- 実行本体
Write-Host "MaAI-SDS エンジン資産セットアップ ($projectRoot)"
New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null

if ($Only -ne "") {
    switch ($Only) {
        "SherpaModels"          { Invoke-SherpaModels }
        "Qwen3AsrOnnxModel"     { Invoke-Qwen3AsrOnnxModel }
        "Qwen3AsrOnnxFp32Model" { Invoke-Qwen3AsrOnnxFp32Model }
        "Qwen3AsrOnnx17BModel"  { Invoke-Qwen3AsrOnnx17BModel }
        "NemotronOnnxModels"    { Invoke-NemotronOnnxModels }
        "SherpaPrebuilt"        { Invoke-SherpaPrebuilt }
        "VoicevoxApp"      { Invoke-VoicevoxApp }
        "ParakeetJaModel"  { Invoke-ParakeetJaModel }
        "SenseVoiceModel"  { Invoke-SenseVoiceModel }
        "SpeechCppRuntime"      { Invoke-SpeechCppRuntime }
        "SpeechCppAsrModel"     { Invoke-SpeechCppAsrModel }
        "SpeechCppIrodoriModel" { Invoke-SpeechCppIrodoriModel }
        "FunAsrNanoModel"  { Invoke-FunAsrNanoModel }
        "CohereAsrModel"   { Invoke-CohereAsrModel }
        "VoicevoxRuntime"  { Invoke-VoicevoxRuntime }
        "UnidicDict"       { Invoke-UnidicDict }
        "TsukasaVendor"    { Invoke-TsukasaVendor -ThrowOnFailure }
        "AivisSpeech"      { Invoke-AivisSpeech }
        "AudioCppVendor"   { Invoke-AudioCppVendor }
        "AudioCppServer"   { Invoke-AudioCppServer }
        "SupertonicGguf"   { Invoke-SupertonicGguf }
        "Qwen3TTSCppGguf"  { Invoke-Qwen3TTSCppGguf }
        "Lfm2AudioGguf"    { Invoke-Lfm2AudioGguf }
        "IrodoriAudioCppGguf" { Invoke-IrodoriAudioCppGguf }
        default             { throw "Unknown -Only value: $Only" }
    }
    exit 0
}

if (-not $SkipSherpaModels)   { Invoke-SherpaModels }      else { Write-Warn "-SkipSherpaModels" }
if (-not $SkipSmartTurn)      { Invoke-SmartTurn }         else { Write-Warn "-SkipSmartTurn" }
Invoke-WhisperPrefetch
Invoke-UnidicDict
if (-not $SkipVoicevoxRuntime){ Invoke-VoicevoxRuntime }   else { Write-Warn "-SkipVoicevoxRuntime" }
# Tsukasa-HF（TsukasaVendor）・AivisSpeech・audio.cpp（AudioCpp）などは、エンジン管理から -Only で取る。

$missing = Invoke-VerifyAssets
if ($missing.Count -gt 0) {
    throw ("{0} 件の資産が揃っていません: {1}" -f $missing.Count, ($missing -join ", "))
}
Write-Ok "既定の資産が揃っています（ほかのエンジンは GUI の「エンジン管理」で入れる）"
