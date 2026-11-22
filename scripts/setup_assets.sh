#!/usr/bin/env bash
# エンジンごとのモデル資産を取得する（setup_assets.ps1 の Linux 版）。
#
# **対象は Linux（主に Docker）のみ**。engine_manager は Windows 以外で一律この
# スクリプトを呼ぶが、ここが取りに行くのは Linux 向けバイナリ
# （libvoicevox_onnxruntime.so / audio-*-bin-linux-x64-cuda*.zip）なので、
# macOS で走らせても動かないものが置かれる。macOS を対象にするなら
# uname で分岐して darwin 向けアセットを足すこと。
#
# Windows 版と同じステップ名を受け付ける。engine_manager が OS を見てどちらを呼ぶか
# 決めるので、ステップ名は両者で一致させること。
#
#   bash scripts/setup_assets.sh --only SenseVoiceModel
#   bash scripts/setup_assets.sh --only TsukasaVendor
set -euo pipefail

case "$(uname -s)" in
    Linux) ;;
    *) printf '[warn] このスクリプトは Linux 向けです（検出: %s）。取得するバイナリはこの OS では動きません。\n' "$(uname -s)" >&2 ;;
esac

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${MAAI_SDS_DATA_ROOT:-$APP_ROOT}"
TMP_ROOT="${PROJECT_ROOT}/tmp/setup_assets"
# sherpa-onnx のビルド済みパッケージを置いている GitHub リポジトリ（Releases）。フォークでは差し替える
RELEASE_REPO="${MAAI_SDS_RELEASE_REPO:-MaAI-Kyoto/MaAI-SDS}"
VENV_PYTHON="${MAAI_SDS_VENV_PYTHON:-$APP_ROOT/.venv/bin/python}"
VOICEVOX_RUNTIME="${MAAI_SDS_VOICEVOX_RUNTIME:-$PROJECT_ROOT/voicevox_runtime}"
ONLY=""
TSUKASA_REPO="Respair/Tsukasa_Speech"
QWEN3_ONNX_TAR="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '[ok] %s\n' "$1"; }
warn() { printf '[warn] %s\n' "$1"; }
die()  { printf '[error] %s\n' "$1" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${2:-}"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done

download() {
    local url="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"
    printf '[get] %s\n' "$url"
    curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 -o "${dest}.part" "$url"
    mv -f "${dest}.part" "$dest"
}

# GitHub Releases の latest から glob に合う資産 URL を 1 件出す。
github_release_asset_url() {
    local repo="$1" pattern="$2"
    local py="$VENV_PYTHON"
    [ -x "$py" ] || py="$(command -v python3 || true)"
    [ -n "$py" ] || return 1
    "$py" - "$repo" "$pattern" <<'PY'
import fnmatch, json, sys, urllib.request

# `releases/latest` は使えない。VOICEVOX/onnxruntime-builder では latest が
# 1.17.3（2025-03）を指したまま 1.23.2（2026-07）が出ており、古い方を掴む。
# 全リリースを published_at の新しい順に見て、最初に一致した資産を返す。
# draft は資産が揃っていないことがあるので除く。
repo, pattern = sys.argv[1], sys.argv[2]
url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
req = urllib.request.Request(url, headers={"User-Agent": "maai-sds-setup"})
with urllib.request.urlopen(req, timeout=30) as response:
    releases = json.load(response)
releases = [r for r in releases if not r.get("draft")]
releases.sort(key=lambda r: r.get("published_at") or "", reverse=True)
for release in releases:
    for asset in release.get("assets") or []:
        name = asset.get("name") or ""
        if fnmatch.fnmatch(name, pattern):
            print(asset["browser_download_url"])
            sys.exit(0)
sys.exit(1)
PY
}

# k2-fsa の release tarball を取得してデータルート直下へ展開する。
get_sherpa_tarball() {
    local tar_name="$1" marker="$2"
    local dir_name="${tar_name%.tar.bz2}"
    local target="${PROJECT_ROOT}/${dir_name}"
    if [ -e "${target}/${marker}" ]; then
        ok "既にあるためスキップ: ${dir_name}"
        return 0
    fi
    mkdir -p "${TMP_ROOT}"
    local archive="${TMP_ROOT}/${tar_name}"
    download "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${tar_name}" "$archive"
    tar -xf "${archive}" -C "${PROJECT_ROOT}"
    rm -f "${archive}"
    [ -e "${target}/${marker}" ] || die "展開後も ${marker} が見つかりません: ${target}"
    ok "配置完了: ${dir_name}"
}

install_nemotron_preprocessor() {
    local model_dir="$1"
    local asset_dir="${APP_ROOT}/assets/nemotron-3.5-asr-streaming-preprocessor"
    local file source destination expected source_hash destination_hash
    local updated=0
    [ -d "$model_dir" ] || die "Nemotron model directory が見つかりません: ${model_dir}"

    for file in preprocessor.onnx preprocessor.export.onnx.data; do
        case "$file" in
            preprocessor.onnx) expected="a21d8788d37a71cef45050988bd9bbd4b19112e60bd7958d52ce1cfe314963c1" ;;
            preprocessor.export.onnx.data) expected="272f9ed95b43e15437df20b7d124daa6a8c831d8cdde3f40e6e71ec3acd06e38" ;;
        esac
        source="${asset_dir}/${file}"
        destination="${model_dir}/${file}"
        [ -f "$source" ] || die "Nemotron preprocessor asset が見つかりません: ${source}"
        source_hash="$(sha256sum -- "$source" | awk '{print $1}')"
        [ "$source_hash" = "$expected" ] || die "Nemotron preprocessor asset の SHA-256 が不正です: ${source}"
        destination_hash=""
        if [ -f "$destination" ]; then
            destination_hash="$(sha256sum -- "$destination" | awk '{print $1}')"
        fi
        if [ "$destination_hash" != "$source_hash" ]; then
            cp -f -- "$source" "$destination"
            updated=1
        fi
    done
    if [ "$updated" -eq 1 ]; then
        ok "Nemotron preprocessor 配置: ${model_dir}"
    fi
}

invoke_sherpa_prebuilt() {
    step "カスタム sherpa-onnx プリビルド"

    # 公式 v1.13.5 + MaAI パッチ（NeMo mel 再計算 / max_symbols=30）。
    # 展開先は vendor/sherpa_onnx_prebuilt/。site-packages へ入れるのは
    # engine_manager の _restore_sherpa_snapshot（uv sync の後に走る）。
    # 生成は scripts/build_sherpa_prebuilt_linux.sh（CUDA devel コンテナ内）。
    # 展開先は PROJECT_ROOT（Docker ではホストのリポジトリ /host_repo）ではなく APP_ROOT。
    # _restore_sherpa_snapshot は REPO_ROOT（= APP_ROOT。Docker では /app）を見るので、
    # PROJECT_ROOT に置くと写し直されず、しかもホストの vendor/ に Linux 用のビルドが入り、
    # ホストの setup_env.ps1 がそれを Windows の venv へ写してしまう。
    local target="${APP_ROOT}/vendor/sherpa_onnx_prebuilt"
    if [ -d "${target}/sherpa_onnx" ]; then
        ok "既にあるためスキップ: ${target}"
        return 0
    fi

    if [ "$(uname -s)" != "Linux" ]; then
        ok "配布物は Linux x86_64 向けです（検出: $(uname -s)）。取得せず PyPI 版を使います。"
        return 0
    fi

    # 配布物は CPython 3.11 の拡張。他の版では読めないので取りに行かない。
    local abi=""
    if [ -x "${VENV_PYTHON:-}" ]; then
        abi="$("${VENV_PYTHON}" -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || true)"
    elif command -v python3 >/dev/null 2>&1; then
        abi="$(python3 -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || true)"
    fi
    if [ -n "$abi" ] && [ "$abi" != "cp311" ]; then
        ok "配布物は cp311 向けです（検出: ${abi}）。取得せず PyPI 版を使います。"
        return 0
    fi

    local url="${MAAI_SDS_SHERPA_PREBUILT_URL:-}"
    if [ -z "$url" ]; then
        url="$(github_release_asset_url "$RELEASE_REPO" "sherpa_onnx_prebuilt_*linux_x86_64*.zip" || true)"
    fi
    if [ -z "$url" ]; then
        ok "配布物が見つかりませんでした。PyPI 版を使います（品質は落ちます）。"
        return 0
    fi

    local zip="${TMP_ROOT}/$(basename "${url%%\?*}")"
    download "$url" "$zip"
    local ex="${TMP_ROOT}/sherpa_prebuilt_extract"
    rm -rf "$ex"; mkdir -p "$ex"
    unzip -q "$zip" -d "$ex"
    rm -f "$zip"

    # ZIP 直下が sherpa_onnx/ か、1 段フォルダを噛んでいるかの両方を許容する。
    local root="$ex"
    if [ ! -d "${root}/sherpa_onnx" ]; then
        root="$(find "$ex" -mindepth 1 -maxdepth 1 -type d -exec test -d '{}/sherpa_onnx' \; -print | head -1)"
    fi
    if [ -z "$root" ] || [ ! -d "${root}/sherpa_onnx" ]; then
        rm -rf "$ex"
        die "ZIP 内に sherpa_onnx/ が見つかりません: ${url}"
    fi

    mkdir -p "$target"
    cp -a "$root"/. "$target"/
    rm -rf "$ex"
    local label="(版不明)"
    [ -f "${target}/VERSION.txt" ] && label="$(cat "${target}/VERSION.txt")"
    ok "配置完了: ${target} (${label})"
}

host_python() {
    if [ -x "${VENV_PYTHON}" ]; then
        printf '%s' "${VENV_PYTHON}"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    die "python が見つかりません（.venv か PATH 上の python3）"
}

invoke_qwen3_onnx_modelscope() {
    local variant="$1"
    local py
    py="$(host_python)"
    "${py}" -B "${APP_ROOT}/scripts/download_qwen3_onnx.py" --root "${PROJECT_ROOT}" --variant "${variant}"
}

invoke_qwen3_asr_onnx_model() {
    step "Qwen3-ASR 0.6B int8（qwen3-asr-onnx）"
    get_sherpa_tarball "$QWEN3_ONNX_TAR" "conv_frontend.onnx"
}

invoke_qwen3_asr_onnx_fp32_model() {
    step "Qwen3-ASR 0.6B fp32（ModelScope）"
    invoke_qwen3_onnx_modelscope "0.6b-fp32"
}

invoke_qwen3_asr_onnx_17b_model() {
    step "Qwen3-ASR 1.7B int8（ModelScope）"
    invoke_qwen3_onnx_modelscope "1.7b-int8"
}

invoke_sherpa_models() {
    step "sherpa-onnx 日本語 zipformer（ReazonSpeech。既定の ASR）"
    # Qwen3-ASR ONNX と Nemotron ONNX は、エンジン管理で選んだときだけ取る。
    get_sherpa_tarball "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2" "tokens.txt"
}

invoke_nemotron_onnx_models() {
    step "Nemotron 3.5 streaming ONNX（80/320/560/1120 ms int8）"
    local ms dir_name
    for ms in 80 320 560 1120; do
        dir_name="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-${ms}ms-int8-2026-06-11"
        get_sherpa_tarball "${dir_name}.tar.bz2" "encoder.int8.onnx"
        install_nemotron_preprocessor "${PROJECT_ROOT}/${dir_name}"
    done
}

# speech.cpp（MIT）。GPU は Vulkan。libvulkan が無い環境（多くの Docker イメージ）では CPU 版を入れる。
# 版を上げるときは src/maai_sds/speechcpp.py の VERSION と setup_assets.ps1 もそろえる。
SPEECHCPP_VERSION="0.11.1"
SPEECHCPP_SHA256_VULKAN="fef3fb5fe5690a96958f7e28ff1a788fa03ec14981eb6d40e8e1dc28fb52b133"
SPEECHCPP_SHA256_CPU="6ee3cda355b6ac8a60dc9ed5c5e5f6eb5946b266821f741c1690c6759f331fb6"

invoke_speechcpp_runtime() {
    step "speech.cpp ${SPEECHCPP_VERSION}（Linux x64）"
    # 実行ファイルはアプリ側（Docker ではイメージ内の /app）に置く。モデルはデータの置き場に置く。
    local dir="${APP_ROOT}/vendor/speechcpp"
    local exe="${dir}/speech"
    if [ -x "$exe" ] && "$exe" --version 2>/dev/null | grep -q "speech.cpp ${SPEECHCPP_VERSION}"; then
        ok "既にあるためスキップ: ${exe}"
        return 0
    fi
    if [ "${MAAI_SDS_SPEECHCPP_FLAVOR:-}" = "cuda" ]; then
        build_speechcpp_cuda "$dir"
        return 0
    fi
    local flavor="cpu" sum="${SPEECHCPP_SHA256_CPU}"
    if [ "${MAAI_SDS_SPEECHCPP_FLAVOR:-}" = "vulkan" ] || { [ -z "${MAAI_SDS_SPEECHCPP_FLAVOR:-}" ] && ldconfig -p 2>/dev/null | grep -q "libvulkan.so.1"; }; then
        flavor="vulkan"; sum="${SPEECHCPP_SHA256_VULKAN}"
    fi
    local url="https://github.com/nyosegawa/speech.cpp/releases/download/v${SPEECHCPP_VERSION}/speech-${SPEECHCPP_VERSION}-linux-x64-${flavor}.zip"
    mkdir -p "${TMP_ROOT}"
    local zip="${TMP_ROOT}/speechcpp-linux.zip"
    download "$url" "$zip"
    local got
    got="$(sha256sum "$zip" | cut -d' ' -f1)"
    if [ "$got" != "$sum" ]; then
        rm -f "$zip"
        die "speech.cpp の zip の SHA-256 が違います: ${got}（期待値 ${sum}）"
    fi
    rm -rf "$dir"
    mkdir -p "$dir"
    unzip -q -o "$zip" -d "$dir"
    rm -f "$zip"
    chmod +x "$exe"
    "$exe" --version >/dev/null || die "speech.cpp を起動できません: ${exe}"
    ok "配置完了（${flavor}）: ${exe}"
}

# CUDA 版をソースから作る（GPU 版の Docker イメージ用。MAAI_SDS_SPEECHCPP_FLAVOR=cuda）。
# Docker Desktop（WSL2）のコンテナでは NVIDIA の Vulkan ドライバーが使えず、配布の Vulkan 版は CPU で
# 動いてしまう。CUDA はコンテナでも使えるので、speech.cpp を GGML_CUDA で作る。speech.cpp の公式では
# CUDA 版は「ビルドは通るが確認も配布もしていない」扱いなので、版を固定し、Vulkan 版と同じ文字起こしに
# なることを確かめて使っている。nvcc（CUDA の devel イメージ）と cmake・ninja・git が要る。
# GGML_NATIVE=OFF：ビルドした PC の CPU 専用の命令に寄せず、どの x86-64（AVX2）でも動くようにする。
SPEECHCPP_CUDA_ARCHS="${MAAI_SDS_SPEECHCPP_CUDA_ARCHS:-75-real;80-real;86-real;89-real;90-real;120-real;120-virtual}"

build_speechcpp_cuda() {
    local dir="$1"
    command -v nvcc >/dev/null || die "nvcc がありません（CUDA の devel イメージで実行してください）"
    local src="${TMP_ROOT}/speechcpp-src"
    rm -rf "$src"
    mkdir -p "${TMP_ROOT}"
    git clone -q --recurse-submodules --depth 1 --shallow-submodules --branch "v${SPEECHCPP_VERSION}" \
        https://github.com/nyosegawa/speech.cpp.git "$src"
    # ggml の CUDA の ssm_conv は、チャンネル数が 128 の倍数でないと GGML_ASSERT で止まる。speech.cpp の
    # FastConformer（Haas-ja など）がこれに当たるので、どんな形でも動く版へ回すパッチを当てる。
    local patch
    for patch in "${APP_ROOT}/scripts/speechcpp_patches/"*.patch; do
        git -C "$src/ggml" apply "$patch" || die "パッチを当てられません: ${patch}"
    done
    # ビルドする場（docker build）には GPU のドライバーが無く libcuda.so.1 が無い。リンクだけは CUDA の
    # スタブで通す（実行時はコンテナに渡されるドライバーの libcuda を使う）。確認用のプログラムは
    # 作らず、speech だけを作る。
    local stubs="${TMP_ROOT}/cuda-stubs"
    mkdir -p "$stubs"
    ln -sf "$(dirname "$(command -v nvcc)")/../lib64/stubs/libcuda.so" "$stubs/libcuda.so.1"
    cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${SPEECHCPP_CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,${stubs}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,${stubs}" >/dev/null
    cmake --build "$src/build" --target speech -j "${MAAI_SDS_BUILD_JOBS:-$(nproc)}"
    rm -rf "$dir"
    mkdir -p "$dir"
    # speech は libspeech と ggml を静的に取り込んでいるので、これ 1 つでよい
    cp -a "$src/build/speech" "$dir/"
    rm -rf "$src" "$stubs"
    # ドライバーの無いビルドの場では起動できない（libcuda.so.1 が要る）ので、ここではできたかだけを見る
    [ -x "$dir/speech" ] || die "speech.cpp（CUDA 版）ができていません: ${dir}/speech"
    ok "配置完了（cuda、ソースからビルド）: ${dir}/speech"
}

invoke_speechcpp_pull() {
    local model="$1"
    local exe="${APP_ROOT}/vendor/speechcpp/speech"
    [ -x "$exe" ] || invoke_speechcpp_runtime
    local models="${MAAI_SDS_SPEECHCPP_MODELS:-${PROJECT_ROOT}/speechcpp_models}"
    mkdir -p "$models"
    SPEECH_MODEL_DIR="$models" "$exe" pull "$model" || die "speech.cpp のモデルを取得できません: ${model}"
    ok "取得済み: ${model}"
}

invoke_speechcpp_asr_model() {
    step "speech.cpp の ASR モデル（Haas-ja v1.0）"
    invoke_speechcpp_pull "haas-ja-v1.0"
}

invoke_speechcpp_irodori_model() {
    step "speech.cpp の TTS モデル（Irodori-TTS v4.1 Small MF・q8_0）"
    invoke_speechcpp_pull "irodori-tts-mf:q8_0"
}

invoke_smart_turn() {
    step "smart-turn ONNX"
    local target="${PROJECT_ROOT}/smart-turn-v3.2-gpu.onnx"
    if [ -e "$target" ]; then
        ok "既にあるためスキップ: ${target}"
        return 0
    fi
    download "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx" "$target"
    ok "配置完了: ${target}"
}

invoke_sensevoice_model() {
    step "SenseVoiceSmall ONNX（sensevoice-onnx）"
    # **2024-07-17 を使うこと**。同名の 2025-09-09 は広東語ファインチューンで、
    # 日本語音声を簡体字で書き起こす（実測 CER 0.79）。
    get_sherpa_tarball "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2" "model.int8.onnx"
}

invoke_parakeet_ja_model() {
    step "Parakeet TDT-CTC 0.6B 日本語 ONNX（parakeet-onnx）"
    get_sherpa_tarball "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2" "model.int8.onnx"
}

invoke_funasr_nano_model() {
    step "Fun-ASR-Nano-2512 ONNX（funasr-nano-onnx）"
    get_sherpa_tarball "sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2" "llm.int8.onnx"
}

invoke_cohere_asr_model() {
    step "Cohere Transcribe 14 言語 ONNX（cohere-asr-onnx）"
    get_sherpa_tarball "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2" "tokens.txt"
}

invoke_unidic_dict() {
    step "unidic 辞書"
    [ -x "${VENV_PYTHON}" ] || { warn ".venv が無いためスキップ（先に uv sync すること）"; return 0; }
    if "${VENV_PYTHON}" -c "import unidic,os,sys; sys.exit(0 if os.path.exists(os.path.join(unidic.DICDIR,'dicrc')) else 1)" 2>/dev/null; then
        ok "既にあるためスキップ: unidic 辞書"
        return 0
    fi
    "${VENV_PYTHON}" -m unidic download
    ok "配置完了: unidic 辞書"
}

invoke_tsukasa_vendor() {
    step "Tsukasa-HF ベンダーディレクトリ"
    local target="${PROJECT_ROOT}/Tsukasa-HF"
    if [ -e "${target}/importable.py" ]; then ok "既にあるためスキップ: Tsukasa-HF/"; return 0; fi
    [ -x "${VENV_PYTHON}" ] || die ".venv が無いため Tsukasa-HF を取得できません（先に uv sync すること）"
    "${VENV_PYTHON}" - "$target" "$TSUKASA_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
target, repo = sys.argv[1], sys.argv[2]
print(snapshot_download(repo_id=repo, local_dir=target, token=False))
PY
    ok "ダウンロード完了: Tsukasa-HF/"
}

invoke_voicevox_runtime() {
    step "VOICEVOX ランタイム（ORT / OpenJTalk 辞書 / VVM）"
    local rt="$VOICEVOX_RUNTIME"
    mkdir -p "$rt"

    local so_target="${rt}/onnxruntime/lib/libvoicevox_onnxruntime.so"
    local so_alt="${rt}/onnxruntime/lib/libvoicevox_onnxruntime.so.1"
    # 「あればスキップ」だけでは古い版が残り続ける。1.17.3 の CUDA ビルドは
    # Blackwell（compute capability 12.0）のカーネルを持たず推論できない。
    # 版は libvoicevox_onnxruntime.so.<version> というファイル名が持っている。
    local needs_fetch=0
    if [ -e "$so_target" ] || [ -e "$so_alt" ]; then
        local found_ver
        found_ver="$(ls -1 "${rt}/onnxruntime/lib/" 2>/dev/null | sed -n -e 's/^libvoicevox_onnxruntime\.so\.\([0-9][0-9.]*\)$/\1/p' | sort -V | tail -n 1)"
        if [ -z "$found_ver" ]; then
            warn "既存 ORT の版をファイル名から読めなかったので取り直します"
            needs_fetch=1
        elif [ "$(printf '1.23.0\n%s\n' "$found_ver" | sort -V | head -n 1)" != "1.23.0" ]; then
            warn "既存の ONNX Runtime が古い（${found_ver}）。Blackwell 世代の GPU では"
            warn "カーネルが無く推論できないため取り直します。"
            needs_fetch=1
        fi
    else
        needs_fetch=1
    fi
    if [ "$needs_fetch" -eq 0 ]; then
        ok "既にあるためスキップ: ${rt}/onnxruntime/lib/ (${found_ver:-?})"
    else
        # 種類を変えた場合に使われないプロバイダが残らないよう、先に消す。
        rm -f "${rt}"/onnxruntime/lib/libvoicevox_onnxruntime*.so*
        local url=""
        url="$(github_release_asset_url "VOICEVOX/onnxruntime-builder" "voicevox_onnxruntime-linux-x64-cuda-*.tgz" || true)"
        if [ -z "$url" ]; then
            url="$(github_release_asset_url "VOICEVOX/onnxruntime-builder" "voicevox_onnxruntime-linux-x64-*.tgz" || true)"
        fi
        [ -n "$url" ] || die "VOICEVOX Linux onnxruntime のリリース資産が見つかりません"
        mkdir -p "${TMP_ROOT}"
        local tgz="${TMP_ROOT}/voicevox_onnxruntime.tgz"
        download "$url" "$tgz"
        local ex="${TMP_ROOT}/ort_extract"
        rm -rf "$ex"
        mkdir -p "$ex"
        tar -xzf "$tgz" -C "$ex"
        rm -f "$tgz"
        local src
        src="$(find "$ex" -name 'libvoicevox_onnxruntime.so*' -type f | head -n 1 || true)"
        [ -n "$src" ] || die "tgz 内に libvoicevox_onnxruntime.so が見つかりません"
        mkdir -p "$(dirname "$so_target")"
        cp -a "$(dirname "$src")/." "$(dirname "$so_target")/"
        rm -rf "$ex"
        ok "配置完了: $(dirname "$so_target")"
    fi

    local dict_target="${rt}/dict/open_jtalk_dic_utf_8-1.11"
    if [ -e "${dict_target}/sys.dic" ]; then
        ok "既にあるためスキップ: ${dict_target}"
    else
        mkdir -p "${TMP_ROOT}" "${rt}/dict"
        local tgz="${TMP_ROOT}/open_jtalk_dic_utf_8-1.11.tar.gz"
        local downloaded=0
        local url
        for url in \
            "https://master.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz?viasf=1" \
            "https://jaist.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz"
        do
            if download "$url" "$tgz"; then
                downloaded=1
                break
            fi
            warn "取得失敗、次のミラーを試す: $url"
        done
        [ "$downloaded" -eq 1 ] || die "OpenJTalk 辞書をどのミラーからも取得できませんでした"
        tar -xzf "$tgz" -C "${rt}/dict"
        rm -f "$tgz"
        [ -e "${dict_target}/sys.dic" ] || die "辞書の展開に失敗した: ${dict_target}"
        ok "配置完了: ${dict_target}"
    fi

    local vvm_target="${rt}/models/vvms/0.vvm"
    if [ -e "$vvm_target" ]; then
        ok "既にあるためスキップ: ${vvm_target}"
        return 0
    fi
    download "https://github.com/VOICEVOX/voicevox_vvm/releases/download/0.16.4/0.vvm" "$vvm_target"
    ok "配置完了: ${vvm_target}"
}

invoke_aivis_speech() {
    step "AivisSpeech（リモート HTTP）"
    ok "Linux / Docker では AivisSpeech 本体は入れません。ホストで起動し host.docker.internal:10104 へ接続してください。"
}

# ランタイムは Release / コンテナと GGUF だけで動く。ソース clone は通常不要だが、
# 開発用途の明示的な `--only AudioCppVendor` との互換性のため、このステップだけ残す。
invoke_audio_cpp_vendor() {
    step "audio.cpp ベンダーディレクトリ（ソース一式）"
    local root="${PROJECT_ROOT}/audio.cpp"
    local manager="${root}/tools/model_manager_v2.py"
    if [ -e "$manager" ]; then
        ok "既にあるためスキップ: audio.cpp/"
        return 0
    fi
    git clone --depth 1 https://github.com/0xShug0/audio.cpp.git "$root"
    [ -e "$manager" ] || die "clone 後も見つかりません: ${manager}"
    ok "clone 完了: audio.cpp/"
}

invoke_audio_cpp_server() {
    step "audio.cpp サーバー（Linux バイナリまたは公式 Docker イメージ）"
    local linux_bin="${PROJECT_ROOT}/audio.cpp/build/linux-cuda-release/bin/audiocpp_server"
    if [ -x "$linux_bin" ]; then
        ok "既にあるためスキップ: ${linux_bin}"
        return 0
    fi
    local url=""
    url="$(github_release_asset_url "0xShug0/audio.cpp" "audio-*-bin-linux-x64-cuda*.zip" || true)"
    if [ -z "$url" ]; then
        url="$(github_release_asset_url "0xShug0/audio.cpp" "*linux*cuda*.zip" || true)"
    fi
    if [ -n "$url" ]; then
        mkdir -p "${TMP_ROOT}"
        local zip="${TMP_ROOT}/audiocpp-linux.zip"
        download "$url" "$zip"
        local ex="${TMP_ROOT}/audiocpp_extract"
        rm -rf "$ex"
        mkdir -p "$ex"
        unzip -q "$zip" -d "$ex"
        rm -f "$zip"
        local found
        found="$(find "$ex" -name 'audiocpp_server' -type f | head -n 1 || true)"
        if [ -n "$found" ]; then
            mkdir -p "$(dirname "$linux_bin")"
            cp -a "$(dirname "$found")/." "$(dirname "$linux_bin")/"
            chmod +x "$linux_bin"
            rm -rf "$ex"
            ok "配置完了: ${linux_bin}"
            return 0
        fi
        rm -rf "$ex"
        warn "ZIP 内に audiocpp_server が見つかりませんでした"
    fi
    warn "Linux 向けプリビルドは公式リリースにありません。GGUF を置いたうえで docker compose --profile audiocpp up を使うか、ホストの audiocpp_server へ host.docker.internal で接続してください。"
}

invoke_supertonic_gguf() {
    step "Supertonic-3 GGUF"
    local target="${PROJECT_ROOT}/audio.cpp/models/Supertonic-3-GGUF/supertonic-3-orig.gguf"
    if [ -s "$target" ]; then
        ok "既にあるためスキップ: ${target}"
        return 0
    fi
    download "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Supertonic-3-GGUF/supertonic-3-orig.gguf" "$target"
    ok "配置完了: ${target}"
}

invoke_audio_cpp() {
    step "audio.cpp サーバーと Supertonic GGUF"
    invoke_audio_cpp_server
    invoke_supertonic_gguf
}

invoke_qwen3_tts_cpp_gguf() {
    step "Qwen3-TTS CustomVoice Q8 GGUF（audio.cpp）"
    local target="${PROJECT_ROOT}/audio.cpp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf"
    if [ -s "$target" ]; then
        ok "既にあるためスキップ: ${target}"
        return 0
    fi
    download "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf" "$target"
    ok "配置完了: ${target}"
}

# audio.cpp の GGUF を 1 つ取る（あればスキップ）。Docker の audiocpp サービスは audio.cpp/models をそのまま使う。
get_audiocpp_model_file() {
    local url="$1" target="${PROJECT_ROOT}/$2"
    if [ -s "$target" ]; then
        ok "既にあるためスキップ: ${target}"
        return 0
    fi
    download "$url" "$target"
    ok "配置完了: ${target}"
}

# LFM2.5-Audio-1.5B-JP（LFM Open License v1.0）。4 つのファイルを同じフォルダーに置く。リビジョンを固定する。
invoke_lfm2_audio_gguf() {
    step "LFM2.5-Audio-1.5B-JP GGUF（audio.cpp・約 1.8 GB）"
    local rev="64b96718b341dbd5650f9e85627cecdcbd4ac61b" f
    for f in LFM2.5-Audio-1.5B-JP-Q8_0.gguf mmproj-LFM2.5-Audio-1.5B-JP-Q8_0.gguf              vocoder-LFM2.5-Audio-1.5B-JP-Q8_0.gguf tokenizer-LFM2.5-Audio-1.5B-JP-Q8_0.gguf; do
        get_audiocpp_model_file "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF/resolve/${rev}/${f}"             "audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF/${f}"
    done
}

# Irodori-TTS の audio.cpp 版（500M v3 と 600M v3 VoiceDesign）。
invoke_irodori_audiocpp_gguf() {
    step "Irodori-TTS 500M v3 / 600M v3 VoiceDesign GGUF（audio.cpp）"
    local base="https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main"
    get_audiocpp_model_file "${base}/Irodori-TTS-500M-v3-GGUF/irodori-tts-500m-v3-q8_0.gguf"         "audio.cpp/models/Irodori-TTS-500M-v3-GGUF/irodori-tts-500m-v3-q8_0.gguf"
    get_audiocpp_model_file "${base}/Irodori-TTS-600M-v3-VoiceDesign-GGUF/irodori-tts-600m-v3-voicedesign-q8_0.gguf"         "audio.cpp/models/Irodori-TTS-600M-v3-VoiceDesign-GGUF/irodori-tts-600m-v3-voicedesign-q8_0.gguf"
}

invoke_verify_assets() {
    step "検証（Linux / Docker 配置確認）"
    local -a missing=()
    # 既定のステップで取るものだけを確かめる（ほかのエンジンは、エンジン管理が入っているかを判定する）。
    local -a checks=(
        "sherpa zipformer ja|${PROJECT_ROOT}/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt"
        "smart-turn ONNX|${PROJECT_ROOT}/smart-turn-v3.2-gpu.onnx"
        "OpenJTalk 辞書|${VOICEVOX_RUNTIME}/dict/open_jtalk_dic_utf_8-1.11/sys.dic"
        "VVM 0|${VOICEVOX_RUNTIME}/models/vvms/0.vvm"
    )
    local check name path
    for check in "${checks[@]}"; do
        name="${check%%|*}"
        path="${check#*|}"
        if [ -s "$path" ]; then
            ok "${name}: ${path}"
        else
            warn "${name} が見つからないか空です: ${path}"
            missing+=("$name")
        fi
    done

    local voicevox_so="${VOICEVOX_RUNTIME}/onnxruntime/lib/libvoicevox_onnxruntime.so"
    local voicevox_so_alt="${voicevox_so}.1"
    if [ -e "$voicevox_so" ] || [ -e "$voicevox_so_alt" ]; then
        ok "VOICEVOX ORT: ${VOICEVOX_RUNTIME}/onnxruntime/lib/"
    else
        warn "VOICEVOX ORT が見つかりません: ${VOICEVOX_RUNTIME}/onnxruntime/lib/"
        missing+=("VOICEVOX ORT")
    fi

    if ((${#missing[@]} > 0)); then
        die "${#missing[@]} 件の必須資産が揃っていません: ${missing[*]}"
    fi
    ok "既定の資産が揃っています（ほかのエンジンは GUI の「エンジン管理」で入れる）"
}

run_step() {
    case "$1" in
        SherpaModels)      invoke_sherpa_models ;;
        Qwen3AsrOnnxModel) invoke_qwen3_asr_onnx_model ;;
        Qwen3AsrOnnxFp32Model) invoke_qwen3_asr_onnx_fp32_model ;;
        Qwen3AsrOnnx17BModel) invoke_qwen3_asr_onnx_17b_model ;;
        NemotronOnnxModels) invoke_nemotron_onnx_models ;;
        SherpaPrebuilt)    invoke_sherpa_prebuilt ;;
        SmartTurn)         invoke_smart_turn ;;
        SenseVoiceModel)   invoke_sensevoice_model ;;
        SpeechCppRuntime)  invoke_speechcpp_runtime ;;
        SpeechCppAsrModel) invoke_speechcpp_asr_model ;;
        SpeechCppIrodoriModel) invoke_speechcpp_irodori_model ;;
        ParakeetJaModel)   invoke_parakeet_ja_model ;;
        FunAsrNanoModel)   invoke_funasr_nano_model ;;
        CohereAsrModel)    invoke_cohere_asr_model ;;
        UnidicDict)        invoke_unidic_dict ;;
        TsukasaVendor)     invoke_tsukasa_vendor ;;
        VoicevoxRuntime)   invoke_voicevox_runtime ;;
        AivisSpeech)       invoke_aivis_speech ;;
        AudioCppVendor)    invoke_audio_cpp_vendor ;;
        AudioCppServer)    invoke_audio_cpp_server ;;
        SupertonicGguf)    invoke_supertonic_gguf ;;
        AudioCpp)          invoke_audio_cpp ;;
        Qwen3TTSCppGguf)   invoke_qwen3_tts_cpp_gguf ;;
        Lfm2AudioGguf)     invoke_lfm2_audio_gguf ;;
        IrodoriAudioCppGguf) invoke_irodori_audiocpp_gguf ;;
        VerifyAssets)      invoke_verify_assets ;;
        *) die "Unknown --only value: $1" ;;
    esac
}

printf 'MaAI-SDS エンジン資産セットアップ (app=%s data=%s)\n' "${APP_ROOT}" "${PROJECT_ROOT}"
mkdir -p "${TMP_ROOT}"

if [ -n "${ONLY}" ]; then
    run_step "${ONLY}"
    exit 0
fi

# 既定は、既定の設定（sherpa-onnx の ReazonSpeech と VOICEVOX-Core）で対話できるところまで。
# ほかのエンジン（SenseVoice などの ONNX モデル・Tsukasa・audio.cpp の GGUF など）は、
# GUI の「エンジン管理」で選ぶと --only で取る。Windows 版（setup_assets.ps1）と同じ。
default_steps=(SherpaModels SmartTurn UnidicDict VoicevoxRuntime VerifyAssets)
for s in "${default_steps[@]}"; do run_step "$s"; done
ok "既定ステップが完了しました（個別取得は --only を使ってください）"
