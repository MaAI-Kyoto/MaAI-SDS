"""PySide の UI から Web 版が描画するためのマニフェストを吸い出す。

## なぜ AST ではなく実物を歩くのか

`gui/app_config_spec.py` は「既存コードから AST で機械抽出して生成」されているが、
あれは `_load_app_config` / `_collect_app_config_payload` という**平坦な代入列**が相手
だったから成立した。`gui/ui.py` のレイアウトはそうではない。QFormLayout.addRow /
QVBoxLayout.addWidget / QGridLayout / QStackedWidget / QTabWidget が入り混じり、
ウィジェットは変数を経由して別の場所で組まれる。AST で追うと分岐だらけの近似にしかならない。

代わりに `MMDAgentController` をオフスクリーンで実際に構築し、出来上がった
ウィジェット木を歩く。構築後の姿がそのまま出るので、命令的に組まれた部分も
取りこぼさない。生成物 `src/maai_sds/web/ui_manifest.json` はリポジトリに入れて
Web サーバがそのまま配信する（実行時に PySide を必要としない＝ヘッドレスの
Docker イメージでも動く）。

## 使い方

    uv run python scripts/extract_web_ui_manifest.py            # 生成
    uv run python scripts/extract_web_ui_manifest.py --check    # 差分があれば非ゼロ終了

`--check` は PySide 側だけ変えて Web 側が取り残されるのを検出するためのもの。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QWidget,
)

from maai_sds.gui.app_config_spec import APP_CONFIG_FIELDS  # noqa: E402
from maai_sds.gui.collapsible import CollapsibleSection  # noqa: E402
from maai_sds.gui.engine_choices import HIDDEN_SUB_ATTRS, EngineChoiceCombo, choice_id  # noqa: E402
from maai_sds.gui.numeric_fields import manifest_hint  # noqa: E402
from maai_sds.maai.plot_window import MAAI_PLOT_VISIBILITY_SPECS  # noqa: E402

OUTPUT = REPO_ROOT / "src" / "maai_sds" / "web" / "ui_manifest.json"

# config_io のキー名がウィジェット属性の語幹と一致しないもの。
ATTR_CONFIG_OVERRIDES = {
    "voice_dialog_interrupt_check": "voice_dialog_interrupt_on_user_speech",
    "voice_dialog_interrupt_pnow_threshold_edit": "voice_dialog_interrupt_p_now_threshold",
    # 以下は `_config_key_index` の命名規則（<stem>_combo → <stem> 等）で当たらないもの。
    # config_key が付かないと app.js 側の `if (key)` ガードで拾われず、**編集できるのに
    # 何も保存されない無言の no-op** になる。いずれも実際にランタイムが読むキーなので、
    # 保存されないと Web からは設定を変えられない。対応は gui/config_io.py の
    # `_collect_app_config_payload` の実キーと 1 対 1 で確認済み。
    "asr_nemotron_chunk_combo": "asr_nemotron_chunk_size_ms",
    # ONNX 版は別ウィジェットだが同じキーを共有する（ui.py が currentTextChanged で
    # 相互に同期し、保存は上の combo の値。どちらから書いても等価）。
    "asr_nemotron_onnx_chunk_combo": "asr_nemotron_chunk_size_ms",
    "asr_nemotron_finalize_combo": "asr_nemotron_finalize_mode",
    "asr_nemotron_onnx_finalize_combo": "asr_nemotron_finalize_mode",
    "asr_qwen_stop_container_check": "asr_qwen_stop_with_compose_down",
    "asr_qwen_enforce_eager_check": "asr_qwen_vllm_enforce_eager",
    # AEC は wav 用と共通用の 2 キーに書かれるが（config_io.py の 840/844）、
    # dialog/settings.py が実際に読むのは audio_input_enable_aec の方。
    "audio_input_wav_aec_check": "audio_input_enable_aec",
    "audio_input_wav_aec_ref_combo": "audio_input_aec_ref",
    "audio_input_wav_channel_combo": "audio_input_wav_channel_index",
    "maai_backchannel_min_interval_edit": "maai_backchannel_min_interval_sec",
    "maai_plot_raw_draw_interval_edit": "maai_plot_raw_draw_interval_ms",
    # checkable QGroupBox のチェック状態（相槌 / 頷き / ターンテイキング VAP の有効無効）。
    # `_box` を落とすだけの汎用ルールにはしない: `tts_model_box` → `tts_model` のように
    # 保存キーと綴りが衝突するグループが実在し、そちらが checkable になった瞬間に
    # テキストキーへ bool を書き込む。数が少ないので明示する。
    "maai_enable_bc_box": "maai_enable_bc",
    "maai_enable_nod_box": "maai_enable_nod",
    "maai_enable_turn_vap_box": "maai_enable_turn_vap",
    # `audio_input_mic_device_combo` は**意図的に載せない**。PySide では
    # `audio_input_mic_device_index`（サーバ側のデバイス番号）だが、Web では
    # ブラウザのマイク列挙に転用していて意味が違う。紐づけるとブラウザ側の
    # デバイス番号がサーバ用のキーに保存され、PySide 側の選択が壊れる。
}

# `*_browse_btn` → 対象入力。規則で当たらないものだけ列挙する。
BROWSE_TARGET_OVERRIDES = {
    "exec_browse_btn": "exec_edit",
    "mdf_browse_btn": "mdf_edit",
    "voicevox_exe_browse_btn": "voicevox_exe_edit",
    "audio_input_wav_browse_btn": "audio_input_wav_path_edit",
    "asr_external_start_script_browse_btn": "asr_external_start_script_edit",
}

# ローカル変数のまま controller 属性に載らないボタン。
# 出現順キュー（同じ text が複数ある画面用）。
WINDOW_TEXT_ATTR_QUEUE = {
    "main": {
        "MMDAgent-EXを起動": ["mmdagent_launch_btn"],
        "停止": ["mmdagent_stop_btn"],
        "MMDAgent-EX起動設定": ["settings_toggle_btn"],
        "TTS": ["tts_test_toggle_btn"],
        "ASR": ["asr_settings_toggle_btn"],
        "LLM": ["llm_test_toggle_btn"],
        "MaAIテスト": ["maai_test_toggle_btn"],
        "OBS録画設定": ["obs_settings_toggle_btn"],
        "音声入力の設定": ["audio_input_settings_toggle_btn"],
        "音声出力の設定": ["audio_output_settings_toggle_btn"],
        "プロセスコンソール": ["console_toggle_btn"],
        "エンジン管理": ["engine_manager_toggle_btn"],
    },
    "settings_window": {
        "参照…": ["exec_browse_btn", "mdf_browse_btn"],
        "MDFから読込": ["reload_mdf_btn"],
        "MDFへ保存": ["save_mdf_btn"],
        "MDFを新規保存…": ["save_as_mdf_btn"],
        "デフォルトに戻す": ["reset_defaults_btn"],
    },
    "obs_settings_window": {
        "接続テスト": ["obs_remote_test_btn"],
        "録画開始（手動）": ["obs_manual_start_btn"],
        "録画停止（手動）": ["obs_manual_stop_btn"],
    },
    "maai_test_window": {
        "ログクリア": ["maai_test_clear_btn"],
        "MaAIモデル設定": ["maai_settings_open_btn"],
        "頷きモーション波形": ["maai_nod_chart_open_btn"],
    },
    "audio_input_settings_window": {
        "参照…": ["audio_input_wav_browse_btn"],
    },
}

# サブウィンドウは controller の属性として保持されている。Web では「画面」になる。
# 並び順がそのままサイドバーの並びになるので、PySide のメニュー順ではなく使用頻度順にする。
WINDOW_ATTRS = [
    ("main", None, "メイン"),
    ("voice_dialog_settings", "voice_dialog_settings_window", "音声対話の詳細設定"),
    ("asr_test_window", "asr_test_window", "ASR（音声認識）"),
    ("tts_test_window", "tts_test_window", "TTS（音声合成）"),
    ("llm_test_window", "llm_test_window", "LLM（応答生成）"),
    ("maai_test_window", "maai_test_window", "MaAIテスト"),
    ("maai_settings_window", "maai_settings_window", "MaAIモデル設定"),
    ("llm_s2s_test_window", "llm_s2s_test_window", "LLM分割テスト"),
    ("audio_input_settings_window", "audio_input_settings_window", "音声入力の設定"),
    ("audio_output_settings_window", "audio_output_settings_window", "音声出力の設定"),
    ("settings_window", "settings_window", "MMDAgent-EX起動設定"),
    # engine_manager_window は載せない。Web は app.js の SKIP_WINDOW_IDS で読み飛ばし、
    # /api/engines から自前で描いているので使われないうえ、この画面のチェックボックスは
    # 「そのマシンに何が入っているか」を値として持つ。生成物に混ざると、抽出した人の
    # インストール状態が全ユーザーへ配られ、--check も他マシンでは必ず落ちる。
    ("obs_settings_window", "obs_settings_window", "OBS録画設定"),
    ("tts_template_auth_window", "tts_template_auth_window", "TEMPLATE認証"),
    ("console_window", "console_window", "プロセスコンソール"),
    ("nod_motion_chart_window", "nod_motion_chart_window", "頷きモーション波形"),
]


def _attr_index(controller) -> dict[int, str]:
    """ウィジェットの id() -> 名前。木から名前を引くために使う。

    MDF の入力欄は `controller.<attr>` ではなく `controller.field_widgets[key]` に
    入っているので、`mdf:<key>` という合成名で拾う（保存先が controller_config では
    なく MDF ファイルなので、Web 側でも別扱いにする必要がある）。
    """
    index: dict[int, str] = {}
    for name, value in vars(controller).items():
        if isinstance(value, QWidget):
            index.setdefault(id(value), name)
    for key, widget in getattr(controller, "field_widgets", {}).items():
        if isinstance(widget, QWidget):
            index[id(widget)] = f"mdf:{key}"
    for key, btn in getattr(controller, "module_status_start_btns", {}).items():
        if isinstance(btn, QWidget):
            index.setdefault(id(btn), f"{key}_module_start_btn")
    for key, btn in getattr(controller, "module_status_stop_btns", {}).items():
        if isinstance(btn, QWidget):
            index.setdefault(id(btn), f"{key}_module_stop_btn")
    return index


def _config_key_index(controller) -> dict[str, str]:
    """ウィジェット属性名 -> controller_config のキー。

    APP_CONFIG_FIELDS が押さえているのはテキスト欄 167 個だけで、コンボと
    チェックボックスは `config_io` が明示コードで読み書きしている。そちらは
    宣言が無いので、実際に保存されるキー集合（`_collect_app_config_payload`）に対して
    命名規則で引き当てる。規則で当たらないものは None のまま残し、Web 側は
    「表示するが保存経路は無い」ものとして扱う。
    """
    index = {row[1]: row[0] for row in APP_CONFIG_FIELDS}
    index.update(ATTR_CONFIG_OVERRIDES)
    # MDF フィールド（保存先が MDF ファイル側）。
    for key in getattr(controller, "field_widgets", {}):
        index[f"mdf:{key}"] = key
    # MaAI プロットの表示パネル。宣言は MAAI_PLOT_VISIBILITY_SPECS 側にある。
    for panel_id, _label in MAAI_PLOT_VISIBILITY_SPECS:
        index[f"maai_plot_visibility_{panel_id}_check"] = f"maai_plot_show_{panel_id}"
    try:
        saved_keys = set(controller._collect_app_config_payload().keys())
    except Exception:  # noqa: BLE001
        return index
    for attr in list(vars(controller)):
        if attr in index:
            continue
        for suffix in ("_combo", "_check", "_edit"):
            if not attr.endswith(suffix):
                continue
            stem = attr[: -len(suffix)]
            for candidate in (stem, f"{stem}_enabled", f"{stem}_path"):
                if candidate in saved_keys:
                    index[attr] = candidate
                    break
            break
    return index


def _portable_value(value: str) -> str:
    """リポジトリ配下の絶対パスを、リポジトリ基準の相対パスへ落とす。

    `gui/config_io.py` は空欄のとき既定を導出する（`maai_root` 空 →
    `paths.assets_root()`、`maai_backchannel_dictionary_path` 空 →
    `repo_path("backchannel_forms.json")`）。設定を読ませずに構築してもこの導出は走るので、
    そのままだと抽出した人の `D:\\MaAI-SDS\\...` が生成物に焼き付き、
    `web/parity.py` の merge_app_ui_defaults 経由で全ユーザーの既定値になる。
    他マシンでは `--check` も必ず落ちる。

    アプリ側は `paths.resolve_under_repo` で相対パスをリポジトリ基準に戻すので、
    相対化しても解決先は変わらない。
    """
    if not value:
        return value
    try:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        pass
    return value


def _describe(widget: QWidget, attrs: dict[int, str], keys: dict[str, str]) -> dict | None:
    attr = attrs.get(id(widget))
    tooltip = (widget.toolTip() or "").strip()
    common = {
        "attr": attr,
        "config_key": keys.get(attr or ""),
        # "mdf" は MDF ファイルへ、"app" は controller_config.json へ保存される。
        "scope": "mdf" if (attr or "").startswith("mdf:") else "app",
        "tooltip": tooltip or None,
    }
    # エンジンの一覧で選ぶようになった欄（speech.cpp のモデル・Irodori の実行系）は Web でも出さない。
    if attr in HIDDEN_SUB_ATTRS:
        common["hidden"] = True
    if isinstance(widget, QLineEdit):
        # パスワード欄は値を一切載せない。build_manifest が設定を読ませずに構築するので
        # 本来ここは常に空だが、この生成物は git に commit され `/api/ui/manifest` から
        # ブラウザへも配信される。過去に API キー 3 件が実際に混入した経路なので、
        # 「値の出所が変わっても秘密だけは出ない」を構造で担保しておく。
        password = widget.echoMode() == QLineEdit.EchoMode.Password
        desc = {**common, "kind": "line",
                "value": "" if password else _portable_value(widget.text()),
                "password": password}
        # 数値の欄は範囲を載せる（Web は input type=number にして min/max を付ける）。
        numeric = manifest_hint(attr)
        if numeric is not None:
            desc["numeric"] = numeric
        return desc
    if isinstance(widget, QPlainTextEdit):
        # 読み取り専用の欄はログ/結果の表示先で、抽出中に届いたイベントが
        # そのまま焼き付いてしまう（実際に console_text_edit に
        # "[AudioOutput] ready" が入り、--check がドリフト検出として
        # 機能しなくなった）。中身は成果物に載せない。
        readonly = widget.isReadOnly()
        return {**common, "kind": "textarea",
                "value": "" if readonly else widget.toPlainText(),
                "readonly": readonly}
    if isinstance(widget, EngineChoiceCombo):
        # 1 行＝「モデル（実行系）」。保存するのはエンジンの値（config_key）と、行が決める隠した欄の値。
        # Web は選んだ行の subs をそれぞれの設定キーへ書く（app.js の engine_choice）。
        items = []
        for c in widget.choices():
            items.append({
                "value": choice_id(c),
                "label": c.label,
                "engine": c.engine,
                "subs": [{"config_key": keys.get(a), "value": v} for a, v in c.subs],
            })
        current = widget.current_choice()
        return {**common, "kind": "combo", "engine_choice": True, "items": items,
                "value": current.engine if current is not None else ""}
    if isinstance(widget, QComboBox):
        # マイク入力の選択肢は載せない。中身は **抽出したマシンに繋がっている
        # オーディオデバイス** で、Bluetooth の接続状況ひとつで変わるため、
        # `--check` がドリフト検出として機能しなくなる（実際に落ちた）。
        # Web 側は fillBrowserMics() が innerHTML を捨ててブラウザ側の
        # enumerateDevices() で埋め直すので、そもそも使っていない。
        if attr == "audio_input_mic_device_combo":
            return {**common, "kind": "combo", "items": [], "value": ""}
        # 表示名と保存値（itemData）が別の combo は {value, label} で載せる。
        # 表示名だけを載せると Web は表示名をそのまま保存してしまい、
        # Qt と違う値（"GPU (cuda)" など）がワーカーへ渡る。
        texts = [widget.itemText(i) for i in range(widget.count())]
        datas = [widget.itemData(i) for i in range(widget.count())]
        if attr == "audio_input_wav_aec_ref_combo":
            # Qt は入力モードで項目を作り直す（ステレオ WAV のときだけ「未選択ch」）。
            # 抽出時はマイクモードなので全項目を載せる。マイク/TCP では実行側が
            # system に固定する（dialog/settings.build_audio_input_worker_config）。
            texts, datas = ["未選択ch", "システム音声"], ["other_ch", "system"]
        if any(d is not None for d in datas):
            items = [
                {"value": str(t if d is None else d), "label": t}
                for t, d in zip(texts, datas)
            ]
            data = widget.currentData()
            value = str(widget.currentText() if data is None else data)
        else:
            items = texts
            value = widget.currentText()
        return {**common, "kind": "combo", "items": items, "value": value}
    if isinstance(widget, QCheckBox):
        return {**common, "kind": "check", "value": widget.isChecked(),
                "text": widget.text()}
    if isinstance(widget, QPushButton):
        desc = {**common, "kind": "button", "text": widget.text()}
        target = _browse_target(attr)
        if target:
            desc["browse_target"] = target
        return desc
    return None


def _browse_target(attr: str | None) -> str | None:
    if not attr:
        return None
    if attr in BROWSE_TARGET_OVERRIDES:
        return BROWSE_TARGET_OVERRIDES[attr]
    if attr.endswith("_browse_btn"):
        return f"{attr[: -len('_browse_btn')]}_edit"
    return None


def _assign_fallback_attrs(windows: list[dict]) -> None:
    """controller 属性に載らなかったボタンへ、画面とラベルから安定 ID を付ける。"""
    for win in windows:
        queues = {
            text: list(names)
            for text, names in WINDOW_TEXT_ATTR_QUEUE.get(win["id"], {}).items()
        }
        for group in win.get("groups") or []:
            for row in group.get("rows") or []:
                for w in row.get("widgets") or []:
                    if w.get("kind") != "button" or w.get("attr"):
                        continue
                    text = str(w.get("text") or "")
                    names = queues.get(text)
                    if not names:
                        continue
                    attr = names.pop(0)
                    w["attr"] = attr
                    target = _browse_target(attr)
                    if target:
                        w["browse_target"] = target
                    if attr in ATTR_CONFIG_OVERRIDES:
                        w["config_key"] = ATTR_CONFIG_OVERRIDES[attr]
    _infer_row_browse_targets(windows)


def _infer_row_browse_targets(windows: list[dict]) -> None:
    """同じ行の入力欄と「参照」ボタンを紐づける。"""
    for win in windows:
        for group in win.get("groups") or []:
            for row in group.get("rows") or []:
                widgets = row.get("widgets") or []
                edits = [w for w in widgets if w.get("kind") in {"line", "textarea"} and w.get("attr")]
                buttons = [
                    w for w in widgets
                    if w.get("kind") == "button" and "参照" in str(w.get("text") or "")
                ]
                if len(edits) != 1 or not buttons:
                    continue
                target = edits[0]["attr"]
                for btn in buttons:
                    if not btn.get("browse_target"):
                        btn["browse_target"] = target
                    if not btn.get("attr") and target.endswith("_edit"):
                        btn["attr"] = f"{target[: -len('_edit')]}_browse_btn"


def _label_text(item_widget: QWidget | None) -> str:
    if isinstance(item_widget, QLabel):
        return (item_widget.text() or "").strip()
    return ""


def _walk(node, attrs, keys, groups, current) -> None:
    """レイアウト木を歩いて groups を埋める。current は今書き込んでいるグループ。"""
    if isinstance(node, CollapsibleSection):
        # 「詳細設定」の中の行は、親のグループの行として載せ、detail に見出しを付ける。
        # Web はこの印の付いた連続する行を、たためる枠（<details>）にまとめる。
        if current is None:
            current = {"title": "", "rows": []}
            groups.append(current)
        start = len(current["rows"])
        content = node._content.layout()
        if content is not None:
            _walk(content, attrs, keys, groups, current)
        title = node._toggle.text()
        for row in current["rows"][start:]:
            row.setdefault("detail", title)
        return
    if isinstance(node, QScrollArea):
        # 中身は viewport の下にぶら下がるので layout() からは辿れない。
        # MaAI テスト / MaAI モデル設定 / 起動設定のタブがこの形。
        inner = node.widget()
        if inner is not None:
            _walk(inner, attrs, keys, groups, current)
        return

    if isinstance(node, QGroupBox):
        group = {"title": (node.title() or "").strip(), "rows": []}
        groups.append(group)
        # checkable な QGroupBox は「見出し」であると同時に機能の ON/OFF スイッチでもある。
        # PySide ではタイトルバーにチェックが付くが、Web はタイトルを別に描くので、
        # グループ先頭の 1 行として明示的に出す。これを拾わないと MaAI の相槌 /
        # 頷き / ターンテイキング VAP を Web から無効化できない（保存キーはあり、
        # dialog/settings.py が実際に読んでいる）。
        if node.isCheckable():
            attr = attrs.get(id(node))
            group["rows"].append({"label": "", "widgets": [{
                "attr": attr,
                "config_key": keys.get(attr or ""),
                "scope": "app",
                "tooltip": (node.toolTip() or "").strip() or None,
                "kind": "check",
                "value": node.isChecked(),
                "text": "有効",
            }]})
        if node.layout() is not None:
            _walk(node.layout(), attrs, keys, groups, group)
        return

    if isinstance(node, QTabWidget):
        for i in range(node.count()):
            group = {"title": node.tabText(i), "rows": []}
            groups.append(group)
            page = node.widget(i)
            if page is not None and page.layout() is not None:
                _walk(page.layout(), attrs, keys, groups, group)
        return

    if isinstance(node, QStackedWidget):
        # ASR/TTS のエンジン別ページ。どのページがどのエンジンかは controller 側の
        # 対応表でしか分からないので、index を残して Web 側で紐づける。
        for i in range(node.count()):
            group = {"title": f"__stack__{i}", "stack_index": i, "rows": []}
            groups.append(group)
            page = node.widget(i)
            if page is not None and page.layout() is not None:
                _walk(page.layout(), attrs, keys, groups, group)
        return

    if isinstance(node, QWidget):
        described = _describe(node, attrs, keys)
        if described is not None:
            if current is None:
                current = {"title": "", "rows": []}
                groups.append(current)
            current["rows"].append({"label": "", "widgets": [described]})
            return
        if node.layout() is not None:
            _walk(node.layout(), attrs, keys, groups, current)
        return

    if isinstance(node, QFormLayout):
        for row in range(node.rowCount()):
            label_item = node.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field_item = node.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if field_item is None:
                field_item = node.itemAt(row, QFormLayout.ItemRole.SpanningRole)
            # フォームの 1 行として置いた「詳細設定」は、中の行をそれぞれ行として拾う
            # （_collect_widgets に渡すと、中のフォームが 1 行に平たく詰め込まれてラベルが落ちる）。
            if field_item is not None and isinstance(field_item.widget(), CollapsibleSection):
                if current is None:
                    current = {"title": "", "rows": []}
                    groups.append(current)
                _walk(field_item.widget(), attrs, keys, groups, current)
                continue
            label = _label_text(label_item.widget() if label_item else None)
            widgets: list[dict] = []
            if field_item is not None:
                _collect_widgets(field_item, attrs, keys, widgets, groups)
            if not widgets:
                continue
            if current is None:
                current = {"title": "", "rows": []}
                groups.append(current)
            current["rows"].append({"label": label, "widgets": widgets})
        return

    if isinstance(node, QLayout):
        for i in range(node.count()):
            item = node.itemAt(i)
            if item is None:
                continue
            child_widget = item.widget()
            child_layout = item.layout()
            if child_widget is not None:
                _walk(child_widget, attrs, keys, groups, current)
            elif child_layout is not None:
                _walk(child_layout, attrs, keys, groups, current)


def _collect_widgets(item, attrs, keys, out: list[dict], groups) -> None:
    """QFormLayout の field 側。1 行に複数ウィジェット（入力＋参照ボタン等）が入る。"""
    widget = item.widget()
    layout = item.layout()
    if widget is not None:
        described = _describe(widget, attrs, keys)
        if described is not None:
            out.append(described)
        elif isinstance(widget, (QGroupBox, QTabWidget, QStackedWidget)):
            _walk(widget, attrs, keys, groups, None)
        elif widget.layout() is not None:
            for i in range(widget.layout().count()):
                sub = widget.layout().itemAt(i)
                if sub is not None:
                    _collect_widgets(sub, attrs, keys, out, groups)
        return
    if layout is not None:
        for i in range(layout.count()):
            sub = layout.itemAt(i)
            if sub is not None:
                _collect_widgets(sub, attrs, keys, out, groups)


def build_manifest() -> dict:
    import maai_sds.gui.controller as gc

    app = QApplication.instance() or QApplication([])
    # 設定を読み込ませずに構築する。`MMDAgentController.__init__` は `_load_app_config()`
    # を呼ぶので、素直に生成すると **抽出した人の controller_config.json の値** が
    # `widget.text()` に載り、それがそのまま manifest の "value" として commit される。
    # 実際に API キー 3 件と個人の絶対パスが混入していた（`llm_api_key` /
    # `cartesia_tts_api_key` / `obs_remote_password`）。さらにその値は
    # `web/parity.py` の `merge_app_ui_defaults` 経由で **Web の既定値** になるため、
    # 抽出した人の設定が全ユーザーに配られていた。
    #
    # `ConfigStore.load()` は存在しないパスなら `{}` を返す（config_store.py）。
    # 読み込み経路自体は通したうえでキーを空にすることで、「設定ファイルを持たない
    # 新規ユーザーが最初に見る画面」= ウィジェット構築時の既定値だけが残る。
    # 副作用として `--check` が「GUI で設定を変えるたびに差分を出す」ノイズ源でなくなり、
    # 生成物のドリフト検出として実際に使えるようになる。
    # デバイス構成も既定（GPU）に固定して構築する。CPU 構成のマシンでは
    # ui.apply_device_choice_limits が GPU 項目を無効化して現在値を CPU 側へ寄せるため、
    # そのまま抽出すると「抽出した人のマシンが CPU か」が生成物に入り、`--check` が
    # 他マシンで必ず落ちる。CPU 環境向けの無効化は配信時に server.py が
    # disabled_choices として添えるので、生成物側は既定のままでよい。
    # MMDAgent-EX の EXE / MDF の自動探索（C:\MMDAgent-EX）も止める。止めないと、抽出した人の PC に
    # ある MDF のパスと、その MDF から読んだ値（WebSocket のポート、ログのファイル名など）が生成物に入り、
    # merge_app_ui_defaults 経由で全ユーザーの Web の既定値になっていた。`--check` も他の PC では落ちる。
    # Web 版は読み込み時に同じ探索をする（web/parity.py の fill_default_mmdagent_paths）。
    from maai_sds import engine_manager as _em
    from maai_sds.mmd import default_paths as _dp

    _orig_device_extra = _em.device_extra
    _orig_detect = (_dp.detect_default_exec_path, _dp.detect_default_mdf_path)
    _em.device_extra = lambda: "cu128"
    _dp.detect_default_exec_path = lambda *_a, **_k: ""
    _dp.detect_default_mdf_path = lambda *_a, **_k: ""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            gc.MMDAgentController.APP_CONFIG_PATH = Path(tmp) / "no-such-config.json"
            controller = gc.MMDAgentController()
    finally:
        _em.device_extra = _orig_device_extra
        _dp.detect_default_exec_path, _dp.detect_default_mdf_path = _orig_detect
    attrs = _attr_index(controller)
    keys = _config_key_index(controller)

    windows = []
    for window_id, attr, title in WINDOW_ATTRS:
        widget = controller if attr is None else getattr(controller, attr, None)
        if widget is None:
            continue
        groups: list[dict] = []
        if widget.layout() is not None:
            _walk(widget.layout(), attrs, keys, groups, None)
        groups = [g for g in groups if g["rows"]]
        windows.append({
            "id": window_id,
            "title": title,
            "qt_title": widget.windowTitle() or "",
            "groups": groups,
        })
    _assign_fallback_attrs(windows)
    # QStackedWidget のどのページがどのエンジンかは controller 側の対応表にしかない。
    # Web はこれを見て、選択中のエンジンのページだけ出す。
    stack_maps = {
        "asr_test_window": dict(getattr(controller, "asr_engine_page_index", {}) or {}),
        "llm_test_window": {"LM Studio": 0, "OpenAI": 1, "Custom": 2},
        "audio_input_settings_window": {"mic": 0, "wav": 1, "tcp": 2},
    }
    del controller
    app.processEvents()
    return {
        "note": "generated by scripts/extract_web_ui_manifest.py from the PySide6 widget tree",
        "stack_maps": stack_maps,
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="生成せず、既存ファイルとの差分があれば非ゼロ終了")
    args = parser.parse_args()

    manifest = build_manifest()
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not OUTPUT.is_file():
            print(f"[check] {OUTPUT} がありません。生成してください。", file=sys.stderr)
            return 1
        on_disk = OUTPUT.read_text(encoding="utf-8")
        if on_disk != text:
            import difflib

            diff = list(difflib.unified_diff(
                on_disk.splitlines(), text.splitlines(),
                fromfile="ui_manifest.json", tofile="PySide6 から再抽出", lineterm="", n=1))
            print("[check] PySide6 の UI とマニフェストが食い違っています。"
                  " scripts/extract_web_ui_manifest.py を実行して更新してください。", file=sys.stderr)
            for line in diff[:40]:
                print(f"  {line}", file=sys.stderr)
            if len(diff) > 40:
                print(f"  ... 他 {len(diff) - 40} 行", file=sys.stderr)
            return 1
        print("[check] マニフェストは最新です。")
        return 0

    OUTPUT.write_text(text, encoding="utf-8")
    total_groups = sum(len(w["groups"]) for w in manifest["windows"])
    total_rows = sum(len(g["rows"]) for w in manifest["windows"] for g in w["groups"])
    total_widgets = sum(len(r["widgets"]) for w in manifest["windows"]
                        for g in w["groups"] for r in g["rows"])
    print(f"{OUTPUT.relative_to(REPO_ROOT)}: "
          f"{len(manifest['windows'])} windows / {total_groups} groups / "
          f"{total_rows} rows / {total_widgets} widgets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
