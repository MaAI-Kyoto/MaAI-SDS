#!/usr/bin/env bash
# カスタム sherpa-onnx（v1.13.5 + MaAI の品質パッチ）をこの PC でビルドし、アプリの環境に入れる。
# upgrade_sherpa_onnx.ps1 の macOS・Linux 版。Nemotron ONNX（nemotron-onnx）はこれが無いと動かない。
#
#   bash scripts/upgrade_sherpa_onnx.sh
#
# ビルドは build_sherpa_prebuilt_linux.sh（別の venv で作る。macOS は CPU 版）。できたものを
#   1. vendor/sherpa_onnx_prebuilt/ に置く（エンジン管理の「適用」が uv sync のあとにここから戻す）
#   2. アプリの .venv に入れる
#   3. import を確かめ、マーカー .sherpa_onnx_git_built を書く（nemotron-onnx はこれを見る）
# 要るもの：git、CPython 3.11、C++ コンパイラー（macOS は Xcode Command Line Tools）。数分かかる。
# エンジン管理を通さずに uv sync（uv run を含む）すると PyPI 版に戻るので、そのあとは入れ直す。
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-${APP_ROOT}/tmp/sherpa_build}"
VENV_PYTHON="${MAAI_SDS_VENV_PYTHON:-${APP_ROOT}/.venv/bin/python}"
SNAPSHOT="${APP_ROOT}/vendor/sherpa_onnx_prebuilt"
MARKER="${APP_ROOT}/.sherpa_onnx_git_built"

ok()  { printf '[ok] %s\n' "$1"; }
die() { printf '[error] %s\n' "$1" >&2; exit 1; }

[ -x "$VENV_PYTHON" ] || die "アプリの環境が見つかりません: ${VENV_PYTHON}（先に uv sync を）"
abi="$("$VENV_PYTHON" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
[ "$abi" = "cp311" ] || die "アプリの環境の Python が 3.11 ではありません（${abi}）。ビルドは 3.11 用です"

REPO="$APP_ROOT" OUT="${WORK}/out" WORK="$WORK" bash "${APP_ROOT}/scripts/build_sherpa_prebuilt_linux.sh"

# build_sherpa_prebuilt_linux.sh がパッケージングに使った sherpa_onnx/・dist-info・VERSION.txt
STAGE="${WORK}/stage"
[ -d "${STAGE}/sherpa_onnx" ] || die "ビルドの成果物がありません: ${STAGE}/sherpa_onnx"
VERSION="$(cat "${STAGE}/VERSION.txt")"

printf '\n=== アプリの環境へ入れる ===\n'
rm -rf "$SNAPSHOT"
mkdir -p "$(dirname "$SNAPSHOT")"
cp -a "$STAGE" "$SNAPSHOT"
ok "スナップショット: ${SNAPSHOT}"

SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
rm -rf "${SITE}/sherpa_onnx"
# 本体の dist-info だけ（sherpa_onnx_core-* は別パッケージなので残す。engine_manager の復元と同じ）
for d in "${SITE}"/sherpa_onnx-*.dist-info; do
    [ -d "$d" ] && rm -rf "$d"
done
cp -a "${STAGE}/sherpa_onnx" "${SITE}/"
for d in "${STAGE}"/sherpa_onnx-*.dist-info; do
    [ -d "$d" ] && cp -a "$d" "${SITE}/"
done
ok "入れた先: ${SITE}/sherpa_onnx"

PYTHONPATH="$APP_ROOT" "$VENV_PYTHON" "${APP_ROOT}/scripts/verify_sherpa_import.py" || die "import できません（${SITE}/sherpa_onnx）"

commit="$(git -C "${WORK}/sherpa-onnx" rev-parse HEAD 2>/dev/null || echo unknown)"
{
    echo "commit=${commit}"
    echo "version=${VERSION}"
    echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source=${WORK}/sherpa-onnx"
    echo "platform=$(uname -s)-$(uname -m)"
    echo "nemo_max_symbols_per_frame=30"
    echo "nemo_greedy_doublefeed_fix=upstream_3785"
} > "$MARKER"
ok "マーカー: ${MARKER}"
echo ""
echo "完了。GUI の ASR（nemotron-onnx）を起動し直してください。"
echo "uv sync（uv run を含む）は PyPI 版に戻します。戻ったら、エンジン管理の「適用」かこのスクリプトで入れ直してください。"
