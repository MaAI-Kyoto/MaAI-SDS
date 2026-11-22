#!/usr/bin/env bash
# nemotron-asr 専用 venv。nemo-toolkit が本アプリの torch / transformers と衝突するため隔離する。
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-.venv_nemotron}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
NEMO_VERSION="${NEMO_VERSION:===3.0.0}"

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
    echo "[nemotron] venv already exists: ${VENV_PATH}"
else
    echo "[nemotron] Creating venv at ${VENV_PATH} (base: ${BASE_PYTHON})..."
    "$BASE_PYTHON" -m venv "$VENV_PATH"
fi

echo "[nemotron] Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

if [ -n "$TORCH_INDEX_URL" ]; then
    echo "[nemotron] Installing CUDA torch from ${TORCH_INDEX_URL} ..."
    "$VENV_PYTHON" -m pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
fi

NEMO_SPEC="nemo-toolkit[asr]${NEMO_VERSION}"
echo "[nemotron] Installing ${NEMO_SPEC} ..."
"$VENV_PYTHON" -m pip install "$NEMO_SPEC"

echo "[nemotron] Verifying imports..."
"$VENV_PYTHON" -c "import torch, nemo.collections.asr as a; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "[nemotron] Done. Set ASR 'Nemotron Python' to: ${VENV_DIR}/bin/python"
