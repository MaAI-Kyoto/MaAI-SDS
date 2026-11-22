#!/usr/bin/env bash
# カスタム sherpa-onnx（公式 v1.13.5 + MaAI パッチ）の Linux 向けプリビルドを作る。
#
# upgrade_sherpa_onnx.ps1 の Linux 版。同じ commit・同じパッチ・同じ CMake フラグを使う。
# パッチは Python スクリプトなのでそのまま流用できる（プラットフォーム非依存）。
#
# **コンテナの中で走らせる前提**。CUDA devel と Python 3.11 が要るので、
# このリポジトリの maai-sds イメージ（nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 派生）が
# そのまま使える。ホストから叩く例:
#
#   docker run --rm \
#     -v "$PWD:/repo" -v "$PWD/dist:/out" \
#     maai-sds-web:latest bash /repo/scripts/build_sherpa_prebuilt_linux.sh
#
# 成果物: /out/sherpa_onnx_prebuilt_<版>_linux_x86_64_cp311.zip
# これを GitHub Release に上げると、setup_assets.sh の SherpaPrebuilt が拾える
# （拾う側の実装は現状 Windows のみ。Linux 版を配る際にあわせて実装すること）。
set -euo pipefail

SHERPA_TAG="${SHERPA_TAG:-v1.13.5}"
SHERPA_COMMIT="${SHERPA_COMMIT:-3dc7c569f31ca2cd4a20ed6f7db780327e6714c5}"
ENABLE_GPU="${ENABLE_GPU:-1}"

REPO="${REPO:-/repo}"
OUT="${OUT:-/out}"
WORK="${WORK:-/tmp/sherpa_build}"
SRC="${WORK}/sherpa-onnx"
VENV="${WORK}/venv"

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '[ok] %s\n' "$1"; }
die()  { printf '[error] %s\n' "$1" >&2; exit 1; }

[ -d "$REPO/scripts" ] || die "リポジトリが $REPO に見つかりません（-v \"\$PWD:/repo\" を渡すこと）"
mkdir -p "$OUT" "$WORK"

step "ビルド用 Python (3.11) と cmake/ninja"
PY=""
for c in python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        v="$("$c" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        if [ "$v" = "3.11" ]; then PY="$c"; break; fi
    fi
done
[ -n "$PY" ] || die "CPython 3.11 が見つかりません（配布物の ABI タグを cp311 に揃えるため必須）"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel "cmake>=4.3.2" ninja
ok "cmake: $("$VENV/bin/cmake" --version | head -1)"

step "sherpa-onnx ソース取得 (${SHERPA_TAG} / ${SHERPA_COMMIT:0:8})"
if [ ! -d "$SRC/.git" ]; then
    git clone --filter=blob:none https://github.com/k2-fsa/sherpa-onnx.git "$SRC"
fi
git -C "$SRC" fetch --all --tags --quiet
git -C "$SRC" checkout --quiet "$SHERPA_COMMIT"
git -C "$SRC" submodule update --init --recursive --quiet || true
ok "checkout: $(git -C "$SRC" rev-parse --short HEAD)"

step "MaAI パッチ適用"
# .ps1 版と同じ 3 本。upstream に取り込まれていない品質パッチだけが残っている
# （prompt_index #3671 と greedy double-feed #3785 は v1.13.5 本体に入っている）。
for p in patch_sherpa_nemo_max_symbols patch_sherpa_nemo_greedy_doublefeed patch_sherpa_nemo_mel_params; do
    script="$REPO/scripts/${p}.py"
    [ -f "$script" ] || die "パッチが見つかりません: $script"
done
"$VENV/bin/python" "$REPO/scripts/patch_sherpa_nemo_max_symbols.py" "$SRC" --max-symbols 30
"$VENV/bin/python" "$REPO/scripts/patch_sherpa_nemo_greedy_doublefeed.py" "$SRC"
"$VENV/bin/python" "$REPO/scripts/patch_sherpa_nemo_mel_params.py" "$SRC"
ok "3 パッチ適用済み"

step "ビルド"
# .ps1 と同じフラグ。GPU=ON にすると sherpa 側の cmake が
# onnxruntime-linux-x64-gpu を取ってきて CUDA EP を同梱する。
CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_TTS=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF"
[ "$ENABLE_GPU" = "1" ] && CMAKE_ARGS="$CMAKE_ARGS -DSHERPA_ONNX_ENABLE_GPU=ON"
export SHERPA_ONNX_CMAKE_ARGS="$CMAKE_ARGS"
echo "SHERPA_ONNX_CMAKE_ARGS=$SHERPA_ONNX_CMAKE_ARGS"
PATH="$VENV/bin:$PATH" "$VENV/bin/pip" install --no-build-isolation --no-cache-dir "$SRC"
ok "インストール完了"

step "__init__.py を実際の拡張に合わせる"
# TTS=OFF でビルドすると GenerationConfig 等が .so から落ちるのに、
# v1.13.5 の __init__.py はそれらを無条件 import するため、そのままでは
# `ImportError: cannot import name 'GenerationConfig'` で必ず落ちる
# （.ps1 版も同じ理由でこの工程を挟んでいる）。
# 渡すのは site-packages ではなく sherpa_onnx パッケージのディレクトリ（.ps1 と同じ）。
SITE_EARLY="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
"$VENV/bin/python" "$REPO/scripts/fix_sherpa_onnx_init.py" "$SITE_EARLY/sherpa_onnx"
ok "__init__.py 調整済み"

step "検証"
"$VENV/bin/python" - <<'PY'
import sherpa_onnx, pathlib
p = pathlib.Path(sherpa_onnx.__file__).parent
sos = sorted(f.name for f in p.rglob('*.so'))
print('  __version__:', getattr(sherpa_onnx, '__version__', '(なし)'))
print('  .so:', len(sos), '個')
for n in sos[:8]:
    print('   ', n)
assert any('onnxruntime' in n for n in sos), '同梱 onnxruntime が見つかりません（GPU ビルド失敗?）'
PY

step "パッケージング"
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
VER="$("$VENV/bin/python" -c 'import sherpa_onnx; print(sherpa_onnx.__version__)')"
STAGE="$WORK/stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -a "$SITE/sherpa_onnx" "$STAGE/"
# 本体の dist-info だけ。sherpa_onnx_core-* は別パッケージなので混ぜない
# （engine_manager 側の復元も sherpa_onnx-*.dist-info だけを対象にしている）。
for d in "$SITE"/sherpa_onnx-*.dist-info; do
    [ -d "$d" ] && cp -a "$d" "$STAGE/"
done
printf '%s' "$VER" > "$STAGE/VERSION.txt"

ZIP="$OUT/sherpa_onnx_prebuilt_${VER}_linux_x86_64_cp311.zip"
rm -f "$ZIP"
( cd "$STAGE" && "$VENV/bin/python" -c "
import shutil, sys
shutil.make_archive(sys.argv[1][:-4], 'zip', '.')
" "$ZIP" )
ok "生成: $ZIP ($(du -h "$ZIP" | cut -f1))"
