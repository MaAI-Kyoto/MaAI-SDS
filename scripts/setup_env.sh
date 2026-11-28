#!/usr/bin/env bash
# MaAI-SDS の環境を作る（macOS・Apple Silicon 用。setup_env.ps1 の macOS 版）。
#
#   bash scripts/setup_env.sh                  # 既定の構成（ReazonSpeech・VOICEVOX Core で対話できる最小限）
#   bash scripts/setup_env.sh --with-nemotron  # nemotron-asr 用の .venv_nemotron も作る
#   bash scripts/setup_env.sh --verify-only    # 何も入れずに検証だけ
#
# uv sync だけでは環境は完成しない。順番も決まっている（setup_env.ps1 と同じ）。
#   1. uv sync            … 本体の .venv。GUI も各エンジンも optional extra なので、素の uv sync では入らない。
#                           既定の構成（gui・web・asr-sherpa・tts-voicevox）と、既に入っている・エンジン管理で
#                           有効にしたエンジンの extra を入れる。macOS は CPU 構成（cpu）で、.device_extra に記録する。
#   2. sherpa-onnx        … uv sync は PyPI 版で上書きするので、前に待避して後で戻す
#                           （vendor/sherpa_onnx_prebuilt。エンジン管理の「適用」と同じ）。
#                           パッチ適用済みの版を初めて入れるときは --build-sherpa（nemotron-onnx に必要）。
#   3. .venv_nemotron     … --with-nemotron のときだけ。
#   4. エンジン資産       … setup_assets.sh（冪等。無ければ入手、あれば触らない）。
#                           ほかのエンジンは GUI の「エンジン管理」で入れる。
# 最後に検証する（sherpa-onnx の版・ワーカーの import・GUI の構築・テスト）。
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${MAAI_SDS_VENV_PYTHON:-${APP_ROOT}/.venv/bin/python}"
SHERPA_MARKER="${APP_ROOT}/.sherpa_onnx_git_built"
BASE_EXTRAS=(gui web asr-sherpa tts-voicevox)
DEVICE_EXTRA="cpu"   # macOS に CUDA は無い（cu128 の extra は入らない）

WITH_NEMOTRON=0
BUILD_SHERPA=0
SKIP_ASSETS=0
ALL_EXTRAS=0
SKIP_TESTS=0
VERIFY_ONLY=0

usage() {
    cat <<'EOF'
使い方: bash scripts/setup_env.sh [オプション]

  --with-nemotron  nemotron-asr 用の環境 .venv_nemotron も作る
  --build-sherpa   パッチ適用済みの sherpa-onnx をこの Mac でビルドして入れる（nemotron-onnx に必要。
                   Xcode Command Line Tools が要る。数分かかる）
  --skip-assets    エンジン資産を取得しない（setup_assets.sh を実行しない）
  --all-extras     すべてのエンジンの Python パッケージを入れる
  --skip-tests     最後の検証で tests/ のテストを実行しない
  --verify-only    何も入れずに検証だけ行う
  -h, --help       この説明を表示する
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-nemotron) WITH_NEMOTRON=1 ;;
        --build-sherpa)  BUILD_SHERPA=1 ;;
        --skip-assets)   SKIP_ASSETS=1 ;;
        --all-extras)    ALL_EXTRAS=1 ;;
        --skip-tests)    SKIP_TESTS=1 ;;
        --verify-only)   VERIFY_ONLY=1 ;;
        -h|--help)       usage; exit 0 ;;
        *) printf '[error] 知らないオプション: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '[ok] %s\n' "$1"; }
warn() { printf '[warn] %s\n' "$1"; }
info() { printf '[info] %s\n' "$1"; }
die()  { printf '[error] %s\n' "$1" >&2; exit 1; }

venv_py() {
    # 本体の .venv の Python でコードを実行する。-B は古い .pyc を使わないため。
    (cd "$APP_ROOT" && "$VENV_PYTHON" -B -c "$1")
}

# ---------------------------------------------------------------- 前提の確認
check_platform() {
    case "$(uname -s)" in
        Darwin) ;;
        *) die "このスクリプトは macOS 用です（検出: $(uname -s)）。Windows は scripts/setup_env.ps1、Linux は Docker（README の「Docker」）を使ってください" ;;
    esac
    # onnxruntime・VOICEVOX Core・speech.cpp・audio.cpp の macOS 版は Apple Silicon 用だけ
    [ "$(uname -m)" = "arm64" ] || die "Intel の Mac には対応していません（検出: $(uname -m)）。Apple Silicon の Mac で実行してください"
}

check_prereqs() {
    step "前提の確認"
    local cmd
    for cmd in uv git; do
        command -v "$cmd" >/dev/null 2>&1 || die "$cmd が見つかりません（uv: brew install uv、git: xcode-select --install）"
    done
    ok "$(uv --version)"

    # .python-version で 3.11 を指定している（sherpa-onnx のパッチ適用済みビルドなどが 3.11 向け）
    local py311
    py311="$(uv python find 3.11 2>/dev/null || true)"
    if [ -n "$py311" ]; then
        ok "Python 3.11: ${py311}"
    else
        info "Python 3.11 が見つからないので、uv が取ってきます（brew install python@3.11 で入れておいてもよい）"
    fi

    # Xcode Command Line Tools。パッチ適用済みの sherpa-onnx のビルド（--build-sherpa）に要る
    if xcode-select -p >/dev/null 2>&1; then
        ok "Xcode Command Line Tools"
    elif [ "$BUILD_SHERPA" = "1" ]; then
        die "--build-sherpa には Xcode Command Line Tools が要ります（xcode-select --install で入れてから再実行）"
    else
        info "Xcode Command Line Tools が無い（既定の構成では不要。--build-sherpa のときに要る）"
    fi

    # 7-Zip。エンジン管理で VOICEVOX（アプリ版）・AivisSpeech を入れるときに要る
    if command -v 7zz >/dev/null 2>&1 || command -v 7z >/dev/null 2>&1; then
        ok "7-Zip"
    else
        info "7-Zip が無い（既定の構成では不要。VOICEVOX〔アプリ版〕・AivisSpeech を入れるときに brew install sevenzip）"
    fi

    # Node.js。テストの一部（Web UI の JavaScript）だけが使う
    command -v node >/dev/null 2>&1 || warn "node が無いので、Web UI の JavaScript のテストは飛ばします（brew install node）"

    # 低電力モードでは合成・認識が 3〜5 倍遅くなる
    if pmset -g 2>/dev/null | awk '/lowpowermode/ { if ($2 == 1) found = 1 } END { exit found ? 0 : 1 }'; then
        warn "低電力モードがオンです。対話中は合成・認識が 3〜5 倍遅くなります（システム設定 → バッテリー）"
    fi
}

# ------------------------------------------------------------------ 各ステップ
sync_extras() {
    # uv sync に渡す extra。uv sync は渡さなかった extra を消すので、既に入っているエンジンと、
    # エンジン管理で有効にしたエンジンの extra は残す（setup_env.ps1 の Get-SyncExtras と同じ）。
    local extras=("${BASE_EXTRAS[@]}")
    if [ -x "$VENV_PYTHON" ]; then
        local out
        if out="$(venv_py "from maai_sds import engine_manager as em; k={e.id: e for e in em.list_engines()}; x={e.extra for e in k.values() if e.installed and e.extra}; x|={k[i].extra for i in em.get_enabled_engines() if i in k and k[i].extra}; print(' '.join(sorted(x)))" 2>/dev/null)"; then
            local extra
            for extra in $(printf '%s\n' "$out" | tail -n 1); do
                case "$extra" in
                    *[!a-z0-9-]*|"") ;;
                    *) extras+=("$extra") ;;
                esac
            done
        else
            warn "入っているエンジンを調べられなかった（既定の extra だけを入れる）"
        fi
    fi
    printf '%s\n' "${extras[@]}" | awk '!seen[$0]++'
}

save_sherpa_snapshot() {
    # uv sync の前に、入っているパッチ適用済みの sherpa-onnx を待避する（エンジン管理の「適用」と同じ処理）
    [ -x "$VENV_PYTHON" ] || return 0
    venv_py "from maai_sds import engine_manager as em; em._save_sherpa_snapshot_if_custom()" \
        || warn "sherpa-onnx を待避できなかった（パッチ適用済みの版が入っていれば、あとで --build-sherpa で入れ直す）"
}

restore_sherpa_snapshot() {
    step "2. sherpa-onnx"
    local out
    out="$(venv_py "from maai_sds import engine_manager as em; ok, err = em._restore_sherpa_snapshot(); print('restored' if ok else ('error: ' + err if err else 'none'))")" \
        || die "sherpa-onnx を戻せませんでした: ${out}"
    case "$out" in
        restored) ok "パッチ適用済みの sherpa-onnx を戻した（vendor/sherpa_onnx_prebuilt）" ;;
        none)     info "パッチ適用済みの sherpa-onnx は無い（PyPI 版のまま。nemotron-onnx を使うなら --build-sherpa）" ;;
        *)        die "sherpa-onnx を戻せませんでした: ${out}" ;;
    esac
}

run_uv_sync() {
    step "1. uv sync（本体の .venv・CPU 構成）"
    # 大きな wheel（torch など）で uv の既定 30 秒の待ち時間を超えることがある。uv は版によって見る変数が違う
    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
    export UV_REQUEST_TIMEOUT="${UV_REQUEST_TIMEOUT:-300}"
    # エンジン管理の「適用」もこれに従う（cpu を渡し続ける）
    printf '%s' "$DEVICE_EXTRA" > "${APP_ROOT}/.device_extra"
    local args=(sync)
    if [ "$ALL_EXTRAS" = "1" ]; then
        info "すべての extra を入れる（--all-extras）"
        args+=(--all-extras --no-extra cu128)
    else
        local extras=() extra
        while IFS= read -r extra; do extras+=("$extra"); done < <(sync_extras)
        extras+=("$DEVICE_EXTRA")
        info "extra: ${extras[*]}（ほかのエンジンは「エンジン管理」で入れる）"
        for extra in "${extras[@]}"; do args+=(--extra "$extra"); done
    fi
    (cd "$APP_ROOT" && uv "${args[@]}") || die "uv sync が失敗しました"
    ok "uv sync 完了"
}

build_sherpa() {
    step "2b. パッチ適用済みの sherpa-onnx をビルドする（数分かかる）"
    bash "${APP_ROOT}/scripts/upgrade_sherpa_onnx.sh" || die "upgrade_sherpa_onnx.sh が失敗しました"
}

setup_nemotron() {
    step "3. .venv_nemotron（nemotron-asr 用の別の環境）"
    bash "${APP_ROOT}/scripts/setup_nemotron_venv.sh" || die "setup_nemotron_venv.sh が失敗しました"
}

fetch_assets() {
    step "4. エンジン資産"
    bash "${APP_ROOT}/scripts/setup_assets.sh" || die "setup_assets.sh が失敗しました"
}

# ---------------------------------------------------------------------- 検証
verify() {
    step "検証"
    local failures=()
    [ -x "$VENV_PYTHON" ] || die "本体の環境がありません: ${VENV_PYTHON}（--verify-only を外して実行してください）"

    # (a) sherpa-onnx の版
    local ver
    if ver="$(venv_py "import sherpa_onnx, sys; sys.stdout.write(sherpa_onnx.__version__)" 2>/dev/null)"; then
        # upgrade_sherpa_onnx.sh はマーカーを書く。Windows・Linux の GPU 版は版に +cuda などが付く
        if [ -f "$SHERPA_MARKER" ] || [[ "$ver" == *+* ]]; then
            ok "sherpa-onnx ${ver}（パッチ適用済み）"
        else
            info "sherpa-onnx ${ver}（PyPI 版。既定の ReazonSpeech は動く。nemotron-onnx を使うなら --build-sherpa）"
        fi
    else
        failures+=("sherpa-onnx を import できない")
    fi

    # (b) ワーカーを import できるか（python -m で起動する名前）
    local w out
    for w in asr.worker asr.compare_worker audio.input_worker audio.output_worker llm.worker maai.worker tts.worker dialog.worker; do
        if out="$(venv_py "import importlib; importlib.import_module('maai_sds.${w}')" 2>&1)"; then
            ok "maai_sds.${w}"
        else
            failures+=("maai_sds.${w} を import できない: $(printf '%s' "$out" | tail -n 1)")
        fi
    done

    # (c) GUI の構築（画面には出さない）
    if out="$(QT_QPA_PLATFORM=offscreen venv_py "import maai_sds.gui.controller as c; from PySide6.QtWidgets import QApplication; a = QApplication([]); w = c.MMDAgentController(); print('CONSTRUCT_OK')" 2>&1)" \
            && printf '%s' "$out" | grep -q CONSTRUCT_OK; then
        ok "GUI の構築"
    else
        failures+=("GUI を構築できない: $(printf '%s' "$out" | tail -n 1)")
    fi

    # (d) .venv_nemotron
    local nemo_py="${APP_ROOT}/.venv_nemotron/bin/python"
    if [ -x "$nemo_py" ]; then
        if out="$("$nemo_py" -B -c "import nemo; print(nemo.__version__)" 2>&1)"; then
            ok "nemo-toolkit $(printf '%s' "$out" | tail -n 1)（.venv_nemotron）"
        else
            failures+=(".venv_nemotron で nemo を import できない: $(printf '%s' "$out" | tail -n 1)")
        fi
    else
        info ".venv_nemotron が無い（nemotron-asr を使うなら --with-nemotron か、GUI の「エンジン管理」）"
    fi

    # (e) テスト
    if [ "$SKIP_TESTS" = "1" ]; then
        warn "テストは実行しない（--skip-tests）"
    elif [ -d "${APP_ROOT}/tests" ]; then
        info "tests/ のテストを実行する"
        if (cd "$APP_ROOT" && QT_QPA_PLATFORM=offscreen "$VENV_PYTHON" -B -m unittest discover -s tests -p "test_*.py"); then
            ok "テストがすべて通った"
        else
            failures+=("tests/ のテストが失敗した")
        fi
    else
        warn "tests/ が無いので飛ばす"
    fi

    step "結果"
    if [ "${#failures[@]}" -gt 0 ]; then
        local f
        for f in "${failures[@]}"; do printf '[FAIL] %s\n' "$f" >&2; done
        die "${#failures[@]} 件失敗しました"
    fi
    ok "すべて通った。起動: .venv/bin/python main.py"
}

# ------------------------------------------------------------------- 実行本体
printf 'MaAI-SDS 環境セットアップ（macOS）(%s)\n' "$APP_ROOT"
check_platform

if [ "$VERIFY_ONLY" = "1" ]; then
    verify
    exit 0
fi

check_prereqs
save_sherpa_snapshot   # uv sync の前に待避する。ここが順番の要点
run_uv_sync
restore_sherpa_snapshot
if [ "$BUILD_SHERPA" = "1" ]; then
    build_sherpa
fi
if [ "$WITH_NEMOTRON" = "1" ]; then
    setup_nemotron
else
    info ".venv_nemotron は作らない（nemotron-asr を使うなら --with-nemotron か、GUI の「エンジン管理」）"
fi
if [ "$SKIP_ASSETS" = "1" ]; then
    warn "エンジン資産は取得しない（--skip-assets）"
else
    fetch_assets
fi
verify
