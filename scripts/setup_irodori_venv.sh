#!/usr/bin/env bash
# Irodori-TTS 専用 venv。transformers>=5.12.1 が本アプリの 5.5.3 と衝突するため隔離する。
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-.venv_irodori}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ "${VENV_DIR}" = /* ]]; then
    VENV_PATH="${VENV_DIR}"
else
    VENV_PATH="${APP_ROOT}/${VENV_DIR}"
fi
VENV_PYTHON="${VENV_PATH}/bin/python"

pick_base_python() {
    if [ -n "${BASE_PYTHON:-}" ]; then
        printf '%s\n' "$BASE_PYTHON"
        return 0
    fi
    if [ -x "${APP_ROOT}/.venv/bin/python" ]; then
        printf '%s\n' "${APP_ROOT}/.venv/bin/python"
        return 0
    fi
    command -v python3.11 || command -v python3
}

BASE_PYTHON="$(pick_base_python)"
cd "$APP_ROOT"

if [ -x "$VENV_PYTHON" ]; then
    echo "[irodori] venv already exists: ${VENV_PATH}"
else
    echo "[irodori] Creating venv at ${VENV_PATH} (base: ${BASE_PYTHON})..."
    "$BASE_PYTHON" -m venv "$VENV_PATH"
fi

echo "[irodori] Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

if [ -n "$TORCH_INDEX_URL" ]; then
    echo "[irodori] Installing CUDA torch from ${TORCH_INDEX_URL} ..."
    "$VENV_PYTHON" -m pip install "torch>=2.10.0" "torchaudio>=2.10.0" --index-url "$TORCH_INDEX_URL"
fi

DACVAE_SPEC="dacvae @ git+https://github.com/facebookresearch/dacvae"
echo "[irodori] Installing dacvae from Git..."
"$VENV_PYTHON" -m pip install "$DACVAE_SPEC"

IRODORI_SPEC="irodori-tts @ git+https://github.com/Aratako/Irodori-TTS@8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
echo "[irodori] Installing pinned irodori-tts..."
"$VENV_PYTHON" -m pip install "$IRODORI_SPEC"

echo "[irodori] Verifying imports..."
"$VENV_PYTHON" -c "import torch, irodori_tts; from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'irodori_tts', 'OK')"

echo ""
echo "[irodori] Done. Set TTS 'Irodori Python' to: ${VENV_DIR}/bin/python"
