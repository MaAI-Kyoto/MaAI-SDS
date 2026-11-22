# GPU 構成が既定。CPU だけのマシン（GPU パススルーの無い Docker Desktop for Mac を含む）
# では docker-compose.cpu.yml が BASE_IMAGE=ubuntu:22.04 と DEVICE_EXTRA=cpu を渡す。
# CUDA イメージは python3.11 を deadsnakes から足しているだけなので、素の ubuntu でも
# 同じ手順がそのまま通る。
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# cu128（torch cu128 + onnxruntime の CUDA ランタイム）か cpu（どちらも CPU 版）。
# pyproject の [tool.uv] conflicts で排他なので、必ずどちらか一方を渡す。
# 渡さないと torch も onnxruntime-gpu も入らないイメージができる。
ARG DEVICE_EXTRA=cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

# gnupg を明示するのは、`add-apt-repository` が deadsnakes PPA の鍵を取り込むのに
# gpg を使うため。CUDA ベースイメージには最初から入っているが、素の ubuntu:22.04
# には無く、CPU ビルドが
#   ShortcutException: Command '['gpg', ... '--import']' returned non-zero exit status 2
# で落ちた（実測）。
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common ca-certificates gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev git curl build-essential cmake ninja-build pkg-config \
    libsndfile1-dev libportaudio2 portaudio19-dev sox libsox-dev libsox-fmt-all \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends bzip2 unzip \
    && rm -rf /var/lib/apt/lists/*
# onnxruntime-gpu の wheel は 280MB 近くあり、uv の既定 30 秒では読み切れずに
# ネットワークタイムアウトでビルドが落ちる。版によって見る変数名が違うので両方。
ENV UV_HTTP_TIMEOUT=300 UV_REQUEST_TIMEOUT=300

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock ./
# `web` はブラウザ UI（maai_sds.web.server）に必要。`gui` extra は入れない
# ——PySide6 はこのイメージでは使わないので、意図的に外したままにする。
# ASR/TTS extra は Web のエンジン管理「適用」でも足せるが、`uv sync` は exact sync
# なのでイメージに入れた extra は MAAI_SDS_ALWAYS_EXTRAS と揃えて残す。
RUN uv sync --frozen --no-install-project \
    --extra ${DEVICE_EXTRA} \
    --extra asr-faster-whisper \
    --extra asr-sherpa \
    --extra asr-chirp3 \
    --extra tts-voicevox \
    --extra tts-kokoro \
    --extra tts-cartesia \
    --extra tts-tsukasa \
    --extra web
COPY . ./
# エンジン適用とデバイス選択がこのファイルを見る（paths.repo_path 基準 = /app）。
# .dockerignore が deny-all なのでホスト側のものは入らず、書かないと CPU イメージでも
# 既定の cu128 と判定され、CUDA を要求して落ちる。
RUN printf %s "${DEVICE_EXTRA}" > /app/.device_extra
RUN uv sync --frozen \
    --extra ${DEVICE_EXTRA} \
    --extra asr-faster-whisper \
    --extra asr-sherpa \
    --extra asr-chirp3 \
    --extra tts-voicevox \
    --extra tts-kokoro \
    --extra tts-cartesia \
    --extra tts-tsukasa \
    --extra web
# パッチを当てた sherpa-onnx（GitHub Releases のビルド済み。ホストの setup_env.ps1 が入れるものの
# Linux 版）を入れる。PyPI 版には nemotron-onnx の精度のパッチ（docs/sherpa-onnx-custom-build.md）が無い。
# CUDA 版のビルドだが、CUDA のプロバイダーは使うときにだけ読み込むので、CPU 版のイメージでも動く
# （ホストの setup_env.ps1 -Device cpu も同じものを入れている）。
# /app/vendor/sherpa_onnx_prebuilt/ に置くので、Web の「エンジン管理」が uv sync したあとも
# engine_manager がここから写し直す。取れなかったとき（フォークなど）は警告を出して PyPI 版のまま進む。
# MAAI_SDS_SHERPA_PREBUILT_URL で取ってくる先を差し替えられる（社内ミラーなど）。
ARG MAAI_SDS_SHERPA_PREBUILT_URL=""
RUN MAAI_SDS_VENV_PYTHON=/app/.venv/bin/python bash scripts/setup_assets.sh --only SherpaPrebuilt \
    && /app/.venv/bin/python -c "from maai_sds import engine_manager as m; ok, err = m._restore_sherpa_snapshot(); import sherpa_onnx; print('[sherpa] restored' if ok else '[sherpa][WARN] prebuilt not installed: ' + str(err or 'not found'), sherpa_onnx.__version__); assert err is None, err" \
    && rm -rf /app/tmp
# speech.cpp（ASR の speechcpp-asr と、Irodori の既定の動かし方）。実行ファイルはイメージの
# /app/vendor/speechcpp に置き、モデルはデータの置き場（/host_repo/speechcpp_models）をホストと共有する。
# GPU 版のイメージでは CUDA 版をソースから作る（Docker Desktop のコンテナでは NVIDIA の Vulkan が
# 使えず、配布の Vulkan 版は CPU で動いてしまうため。数十分かかる）。CPU 版のイメージは配布の CPU 版。
# MAAI_SDS_BUILD_JOBS で並列数を絞れる（WSL のメモリが足りないとき）。
ARG MAAI_SDS_BUILD_JOBS=""
RUN if [ "${DEVICE_EXTRA}" = "cu128" ]; then export MAAI_SDS_SPEECHCPP_FLAVOR=cuda; fi     && MAAI_SDS_BUILD_JOBS="${MAAI_SDS_BUILD_JOBS}" bash scripts/setup_assets.sh --only SpeechCppRuntime     && rm -rf /app/tmp
RUN cp -a /app/.venv /opt/venv-seed \
    && cp docker/web-entrypoint.sh /usr/local/bin/maai-web-entrypoint.sh \
    && chmod +x /usr/local/bin/maai-web-entrypoint.sh /app/scripts/*.sh

# Models and runtime assets are intentionally not copied; mount them as volumes.
# compose の web サービスは entrypoint を maai-web-entrypoint.sh へ差し替える。
ENTRYPOINT ["/app/.venv/bin/python", "-m", "maai_sds.headless_main"]
CMD ["voice_dialog"]
