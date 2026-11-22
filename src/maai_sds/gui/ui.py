import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QFrame,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QPlainTextEdit,
    QSizePolicy,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .chat_view import ChatView
from .collapsible import CollapsibleSection
from .settings_hub import SettingsHub
from ..utils import to_bool
from .. import engine_manager
from .. import speechcpp

from ..maai.plot_window import MAAI_PLOT_VISIBILITY_SPECS
from ..mmd.blink import DEFAULT_BLINK_INTERVAL_MEAN_SEC

from ..maai.nod_motion_chart_widget import NodMotionChartWidget


# ASR エンジンの選択肢。メイン画面のコンボと比較モードの上下段コンボが同じ一覧を使う
# （以前は 3 箇所に同じリテラルが並んでいて、エンジンを足すと比較モードにだけ現れない、
# という取りこぼしが起きていた）。並び順は「よく使う順」で、設定ページの並びとは独立。
_ASR_ENGINE_CHOICES = [
    "sherpa-onnx", "qwen3-asr-onnx", "nemotron-onnx",
    "sensevoice-onnx", "parakeet-onnx", "cohere-asr-onnx", "funasr-nano-onnx",
    "speechcpp-asr", "faster-whisper", "qwen3-asr", "nemotron-asr", "external-asr-tcp",
]


class _CurrentPageSizeStack(QStackedWidget):
    def sizeHint(self):
        current = self.currentWidget()
        if current is not None:
            return current.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        current = self.currentWidget()
        if current is not None:
            return current.minimumSizeHint()
        return super().minimumSizeHint()


def _finish_main_layout(
    controller,
    root,
    settings_window_box,
    mmd_control_box,
    voice_dialog_start_btn,
    voice_dialog_stop_btn,
    llm_s2s_test_open_btn,
    llm_s2s_test_run_btn,
    maai_plot_open_btn,
) -> None:
    """build_ui の末尾（全体レイアウトの組み立てとウィンドウ配線）。

    `build_ui` から機械的に切り出した（Phase 3-3）。文の順序は変えていない。
    ここは前で作ったローカルを 8 個だけ使い、後ろへ何も渡さないので独立に切り出せた。
    """
    controller.asr_maai_plot_btn = maai_plot_open_btn
    controller._update_maai_test_button_states()
    controller.tts_input_streaming_label.setToolTip(
        "音声対話で、LLM応答を累積して文確定分を順次TTS送信します。TTS内部では分割しません。"
    )
    controller.tts_input_streaming_check.setToolTip(
        "有効: LLM側で文単位の確定分を逐次送信 / 無効: LLM完了後に全文を1回送信。"
    )
    controller.tts_segment_gap_silence_sec_label.setToolTip(
        "LLM側文分割が有効なとき、セグメントとセグメントのあいだに挿入する無音の秒数（0 で無効）。"
    )
    controller.tts_segment_gap_silence_sec_edit.setToolTip(
        "16kHz・モノラルの無音PCMを送出します。最大10秒です。"
    )
    controller.llm_test_tts_input_streaming_label.setToolTip(
        "LLMテストで、LLM側の累積文分割（確定分のみ送信）を切り替えます。"
    )
    controller.llm_test_tts_input_streaming_check.setToolTip(
        "TTS内分割は行わず、LLM側で確定文を切り出して送ります。"
    )
    controller.llm_split_custom_delimiters_label.setToolTip("トークナイザーで「Custom」を選択した場合の区切り文字です。")
    controller.llm_split_custom_strip_emoji_label.setToolTip(
        "トークナイザーで「Custom」を選択した場合、LLMストリーム分割前に絵文字を削除します（TTS向け）。"
    )
    controller.llm_split_custom_strip_parenthetical_label.setToolTip(
        "トークナイザーで「Custom」を選択した場合、括弧で囲まれたテキストを削除してから分割します。"
    )
    controller.llm_split_custom_strip_standalone_warai_label.setToolTip(
        "語頭の「笑」で、直後がい・う・わ・え・お・っのものは動詞語幹として残します。"
        "直後が「笑」以外の漢字なら compound（例: 笑顔）として残します。末尾や句読点直前の単独「笑」や「笑笑」は除去します。"
    )
    llm_s2s_test_open_btn.setToolTip("LLM側の入力文分割をテストするウィンドウを開きます。")
    llm_s2s_test_run_btn.setToolTip("現在の分割設定でs2s単体テストを実行します。")
    controller._sync_llm_test_tts_input_streaming_widgets()

    llm_s2s_form = QFormLayout()
    llm_s2s_form.addRow(controller.llm_test_tts_input_streaming_label, controller.llm_test_tts_input_streaming_check)
    llm_s2s_form.addRow(controller.llm_split_custom_delimiters_label, controller.llm_split_custom_delimiters_edit)
    llm_s2s_form.addRow(
        controller.llm_split_custom_strip_emoji_label,
        controller.llm_split_custom_strip_emoji_check,
    )
    llm_s2s_form.addRow(
        controller.llm_split_custom_strip_parenthetical_label,
        controller.llm_split_custom_strip_parenthetical_check,
    )
    llm_s2s_form.addRow(
        controller.llm_split_custom_strip_standalone_warai_label,
        controller.llm_split_custom_strip_standalone_warai_check,
    )
    controller.llm_s2s_test_chunk_size_edit = QLineEdit("2")
    controller.llm_s2s_test_input_edit = QPlainTextEdit()
    controller.llm_s2s_test_input_edit.setPlaceholderText("s2s単体テスト用テキスト")
    controller.llm_s2s_test_output_edit = QPlainTextEdit()
    controller.llm_s2s_test_output_edit.setReadOnly(True)
    llm_s2s_form.addRow(QLabel("テストチャンクサイズ"), controller.llm_s2s_test_chunk_size_edit)
    llm_s2s_form.addRow(QLabel("テスト入力"), controller.llm_s2s_test_input_edit)
    llm_s2s_form.addRow(QLabel("テスト結果"), controller.llm_s2s_test_output_edit)
    llm_s2s_action_row = QWidget()
    llm_s2s_action_layout = QHBoxLayout()
    llm_s2s_action_layout.setContentsMargins(0, 0, 0, 0)
    llm_s2s_action_layout.addWidget(llm_s2s_test_run_btn, 0)
    llm_s2s_action_layout.addStretch(1)
    llm_s2s_action_row.setLayout(llm_s2s_action_layout)
    llm_s2s_form.addRow(llm_s2s_action_row)
    controller.llm_s2s_test_window = QWidget(controller, Qt.WindowType.Window)
    controller.llm_s2s_test_window.setWindowTitle("LLM分割テスト")
    controller.llm_s2s_test_window.resize(760, 640)
    llm_s2s_layout = QVBoxLayout()
    llm_s2s_layout.addLayout(llm_s2s_form)
    controller.llm_s2s_test_window.setLayout(llm_s2s_layout)

    controller.obs_remote_enabled_check = QCheckBox("遠隔録画を有効にする（WebSocket）")
    controller.obs_remote_host_edit = QLineEdit("127.0.0.1")
    controller.obs_remote_port_edit = QLineEdit("4455")
    controller.obs_remote_password_edit = QLineEdit("")
    controller.obs_remote_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
    controller.obs_remote_link_maai_check = QCheckBox("MaAIテストと連動（開始で録画・終了で停止）")
    controller.obs_remote_link_maai_check.setToolTip(
        "MaAIテストの「起動／停止」に合わせて録画します。"
        " 音声対話で録画する場合は、別途「音声対話と連動」もオンにしてください（MaAIテストと連動のみでは音声対話では録画しません）。"
    )
    controller.obs_remote_link_voice_check = QCheckBox("音声対話と連動（開始で録画・終了で停止）")
    controller.obs_remote_link_voice_check.setToolTip(
        "音声対話の「開始／終了」に合わせて録画します。"
        " このチェックがオフのときは音声対話では録画しません（MaAIテストと連動だけをONにしていても同様です）。"
    )
    obs_remote_test_btn = QPushButton("接続テスト")
    obs_remote_hint = QLabel(
        "OBSの「ツール→WebSocketサーバー設定」でポートとパスワードを合わせてください。"
        " 複数セッションが重なる場合は参照カウントで1本の録画にまとめます。"
    )
    obs_remote_hint.setWordWrap(True)
    obs_form = QFormLayout()
    obs_form.addRow(controller.obs_remote_enabled_check)
    obs_form.addRow(QLabel("ホスト"), controller.obs_remote_host_edit)
    obs_form.addRow(QLabel("ポート"), controller.obs_remote_port_edit)
    obs_form.addRow(QLabel("WebSocketパスワード"), controller.obs_remote_password_edit)
    obs_form.addRow(controller.obs_remote_link_maai_check)
    obs_form.addRow(controller.obs_remote_link_voice_check)
    obs_manual_start_btn = QPushButton("録画開始（手動）")
    obs_manual_stop_btn = QPushButton("録画停止（手動）")
    obs_manual_start_btn.setToolTip("連動と同じ参照カウントで録画を開始します。録画中の場合は参照だけを追加します。")
    obs_manual_stop_btn.setToolTip("手動で追加した参照を1つ解放します。参照が0になると録画を停止します。")
    obs_manual_row = QHBoxLayout()
    obs_manual_row.setContentsMargins(0, 0, 0, 0)
    obs_manual_row.addWidget(obs_manual_start_btn, 0)
    obs_manual_row.addWidget(obs_manual_stop_btn, 0)
    obs_manual_row.addStretch(1)
    obs_manual_w = QWidget()
    obs_manual_w.setLayout(obs_manual_row)
    obs_form.addRow(QLabel("手動"), obs_manual_w)
    obs_test_row = QHBoxLayout()
    obs_test_row.setContentsMargins(0, 0, 0, 0)
    obs_test_row.addWidget(obs_remote_test_btn, 0)
    obs_test_row.addStretch(1)
    obs_test_w = QWidget()
    obs_test_w.setLayout(obs_test_row)
    obs_form.addRow(obs_test_w)
    obs_outer = QVBoxLayout()
    obs_outer.addLayout(obs_form)
    obs_outer.addWidget(obs_remote_hint)
    controller.obs_settings_window = QWidget(controller, Qt.WindowType.Window)
    controller.obs_settings_window.setWindowTitle("OBS録画設定")
    controller.obs_settings_window.resize(520, 320)
    controller.obs_settings_window.setLayout(obs_outer)
    controller.obs_remote_host_edit.editingFinished.connect(controller._save_app_config)
    controller.obs_remote_port_edit.editingFinished.connect(controller._save_app_config)
    controller.obs_remote_password_edit.editingFinished.connect(controller._save_app_config)
    controller.obs_remote_enabled_check.stateChanged.connect(lambda _s: controller._save_app_config())
    controller.obs_remote_link_maai_check.stateChanged.connect(lambda _s: controller._save_app_config())
    controller.obs_remote_link_voice_check.stateChanged.connect(lambda _s: controller._save_app_config())
    obs_remote_test_btn.clicked.connect(controller._on_obs_remote_test_clicked)
    obs_manual_start_btn.clicked.connect(controller._on_obs_remote_manual_start_clicked)
    obs_manual_stop_btn.clicked.connect(controller._on_obs_remote_manual_stop_clicked)

    # メインには操作（開始/終了）と状態だけを置き、細かい設定は「音声対話の詳細設定」画面へ。
    # Web UI の「詳細設定」と同じ分け方。
    voice_dialog_start_btn.setProperty("role", "primary")
    voice_dialog_start_btn.setProperty("size", "large")
    voice_dialog_start_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    voice_dialog_start_btn.setMaximumWidth(16777215)  # 上で一律に付けた 220px の上限を外す
    voice_dialog_settings_btn = QPushButton("詳細設定")
    controller.voice_dialog_settings_toggle_btn = voice_dialog_settings_btn
    # Web の「割り込み」と同じ。システム発話を止めてユーザーの番にする（dialog/worker の interrupt）。
    voice_dialog_interrupt_btn = QPushButton("割り込み")
    voice_dialog_interrupt_btn.setToolTip("話している途中のシステム発話を止めます。")
    voice_dialog_interrupt_btn.clicked.connect(
        lambda: controller._send_dialog_orchestrator_command({"type": "interrupt"})
    )
    controller.voice_dialog_interrupt_btn = voice_dialog_interrupt_btn
    voice_dialog_hint = QLabel("開始すると、ASR・LLM・TTS・MaAIを自動で起動します。")
    voice_dialog_hint.setProperty("role", "caption")
    voice_dialog_hint.setWordWrap(True)
    voice_dialog_sub_buttons = QHBoxLayout()
    voice_dialog_sub_buttons.addWidget(voice_dialog_stop_btn)
    voice_dialog_sub_buttons.addWidget(voice_dialog_interrupt_btn)
    voice_dialog_sub_buttons.addStretch()
    voice_dialog_sub_buttons.addWidget(voice_dialog_settings_btn)
    controller.voice_dialog_status_label.setProperty("role", "status")
    controller.voice_dialog_latency_label.setProperty("role", "status")
    voice_dialog_controls_layout = QVBoxLayout()
    # 右の列をスクロールなしで収めるため、説明文は開始ボタンのツールチップにし、状態と遅延は 1 行に並べる。
    voice_dialog_start_btn.setToolTip(voice_dialog_hint.text())
    voice_dialog_status_row = QHBoxLayout()
    voice_dialog_status_row.setSpacing(14)
    voice_dialog_status_row.addWidget(controller.voice_dialog_status_label)
    voice_dialog_status_row.addWidget(controller.voice_dialog_latency_label, 1)
    controller.voice_dialog_latency_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    voice_dialog_controls_layout.addWidget(voice_dialog_start_btn)
    voice_dialog_controls_layout.addLayout(voice_dialog_sub_buttons)
    voice_dialog_controls_layout.addLayout(voice_dialog_status_row)
    voice_dialog_controls_box = QGroupBox("音声対話")
    voice_dialog_controls_box.setLayout(voice_dialog_controls_layout)

    vd_utter_form = QFormLayout()
    vd_utter_form.addRow(QLabel("開始時のシステム発話"), controller.voice_dialog_opening_utterance_edit)
    vd_utter_form.addRow(QLabel("終了までの時間（秒）"), controller.voice_dialog_scheduled_exit_sec_edit)
    vd_utter_form.addRow(QLabel("終了時の発話"), controller.voice_dialog_scheduled_exit_utterance_edit)
    vd_utter_form.addRow(QLabel("ターン取得までの沈黙（秒）"), controller.voice_dialog_idle_take_turn_sec_edit)
    vd_utter_box = QGroupBox("発話とターン")
    vd_utter_box.setLayout(vd_utter_form)
    vd_interrupt_form = QFormLayout()
    vd_interrupt_form.addRow(controller.voice_dialog_interrupt_check)
    vd_interrupt_form.addRow(QLabel("割り込みVADしきい値"), controller.voice_dialog_interrupt_vad_threshold_edit)
    vd_interrupt_form.addRow(QLabel("割り込みp_nowしきい値"), controller.voice_dialog_interrupt_pnow_threshold_edit)
    vd_interrupt_box = QGroupBox("割り込み")
    vd_interrupt_box.setLayout(vd_interrupt_form)
    vd_stream_form = QFormLayout()
    vd_stream_form.addRow(controller.tts_input_streaming_label, controller.tts_input_streaming_check)
    vd_stream_form.addRow(controller.tts_segment_gap_silence_sec_label, controller.tts_segment_gap_silence_sec_edit)
    vd_stream_box = QGroupBox("LLM→TTS")
    vd_stream_box.setLayout(vd_stream_form)
    vd_settings_layout = QVBoxLayout()
    vd_settings_layout.addWidget(vd_utter_box)
    vd_settings_layout.addWidget(vd_interrupt_box)
    vd_settings_layout.addWidget(vd_stream_box)
    vd_settings_layout.addStretch(1)
    controller.voice_dialog_settings_window = QWidget(controller, Qt.WindowType.Window)
    controller.voice_dialog_settings_window.setWindowTitle("音声対話の詳細設定")
    controller.voice_dialog_settings_window.resize(560, 460)
    controller.voice_dialog_settings_window.setLayout(vd_settings_layout)

    def _open_voice_dialog_settings_window() -> None:
        win = controller.voice_dialog_settings_window
        win.show()
        win.raise_()
        win.activateWindow()

    voice_dialog_settings_btn.clicked.connect(_open_voice_dialog_settings_window)

    # モジュール状態パネル（ASR / LLM / TTS / MaAI の稼働ランプ + 設定要約）。
    # wav 入力時は同じ並びに WAV の再生/停止を出す。
    # ランプ色は controller._refresh_module_status_panel() が状態に応じて設定する。
    module_status_grid = QGridLayout()
    controller.module_status_lamps = {}
    controller.module_status_state_labels = {}
    controller.module_status_summary_labels = {}
    controller.module_status_metrics_labels = {}
    controller.module_status_start_btns = {}
    controller.module_status_stop_btns = {}
    for _row, (_module_key, _module_name) in enumerate(
        (("asr", "ASR"), ("llm", "LLM"), ("tts", "TTS"), ("maai", "MaAI"))
    ):
        _lamp = QLabel()
        _lamp.setFixedSize(12, 12)
        _name_label = QLabel(_module_name)
        _name_label.setMinimumWidth(40)
        _state_label = QLabel("停止")
        _state_label.setMinimumWidth(44)
        _summary_label = QLabel("-")
        # 設定の要約は長いので、列の幅を押し広げないようにする（収まらない分は切れて、全文はツールチップ）。
        _summary_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        _metrics_label = QLabel("")
        _metrics_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if _module_key == "maai":
            _metrics_label.setToolTip("MaAIコンソールの平均処理時間とRTFから表示します")
        _start_btn = QPushButton("▶")
        _start_btn.setProperty("role", "icon")
        _start_btn.setFixedSize(26, 24)
        _start_btn.setToolTip(f"{_module_name}起動")
        _stop_btn = QPushButton("■")
        _stop_btn.setProperty("role", "icon")
        _stop_btn.setFixedSize(26, 24)
        _stop_btn.setToolTip(f"{_module_name}停止")
        _stop_btn.setEnabled(False)
        _mk = _module_key
        _start_btn.clicked.connect(lambda checked, m=_mk: controller._module_panel_start(m))
        _stop_btn.clicked.connect(lambda checked, m=_mk: controller._module_panel_stop(m))
        module_status_grid.addWidget(_lamp, _row, 0)
        module_status_grid.addWidget(_name_label, _row, 1)
        module_status_grid.addWidget(_state_label, _row, 2)
        module_status_grid.addWidget(_start_btn, _row, 3)
        module_status_grid.addWidget(_stop_btn, _row, 4)
        module_status_grid.addWidget(_summary_label, _row, 5)
        module_status_grid.addWidget(_metrics_label, _row, 6)
        controller.module_status_lamps[_module_key] = _lamp
        controller.module_status_state_labels[_module_key] = _state_label
        controller.module_status_metrics_labels[_module_key] = _metrics_label
        controller.module_status_summary_labels[_module_key] = _summary_label
        controller.module_status_start_btns[_module_key] = _start_btn
        controller.module_status_stop_btns[_module_key] = _stop_btn
    _wav_lamp = QLabel()
    _wav_lamp.setFixedSize(12, 12)
    _wav_name = QLabel("WAV")
    _wav_name.setMinimumWidth(40)
    _wav_state = QLabel("停止")
    _wav_state.setMinimumWidth(44)
    _wav_summary = QLabel("-")
    _wav_summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    _wav_metrics = QLabel("")
    _wav_start = QPushButton("▶")
    _wav_start.setProperty("role", "icon")
    _wav_start.setFixedSize(26, 24)
    _wav_start.setToolTip("WAV再生")
    _wav_stop = QPushButton("■")
    _wav_stop.setProperty("role", "icon")
    _wav_stop.setFixedSize(26, 24)
    _wav_stop.setToolTip("WAV停止")
    _wav_stop.setEnabled(False)
    _wav_start.clicked.connect(controller._audio_input_wav_panel_play)
    _wav_stop.clicked.connect(controller._audio_input_wav_panel_stop)
    _wav_row = 4
    module_status_grid.addWidget(_wav_lamp, _wav_row, 0)
    module_status_grid.addWidget(_wav_name, _wav_row, 1)
    module_status_grid.addWidget(_wav_state, _wav_row, 2)
    module_status_grid.addWidget(_wav_start, _wav_row, 3)
    module_status_grid.addWidget(_wav_stop, _wav_row, 4)
    module_status_grid.addWidget(_wav_summary, _wav_row, 5)
    module_status_grid.addWidget(_wav_metrics, _wav_row, 6)
    controller.module_status_wav_lamp = _wav_lamp
    controller.module_status_wav_state_label = _wav_state
    controller.module_status_wav_summary_label = _wav_summary
    controller.module_status_wav_start_btn = _wav_start
    controller.module_status_wav_stop_btn = _wav_stop
    controller.module_status_wav_widgets = [
        _wav_lamp,
        _wav_name,
        _wav_state,
        _wav_start,
        _wav_stop,
        _wav_summary,
        _wav_metrics,
    ]
    for _w in controller.module_status_wav_widgets:
        _w.setVisible(False)
    module_status_grid.setColumnStretch(5, 1)
    controller.module_status_start_all_btn = QPushButton("一括起動")
    controller.module_status_start_all_btn.setToolTip(
        "現在の設定でASR・LLM・TTS・MaAIを起動します。稼働中・起動中のモジュールは再利用します。"
    )
    controller.module_status_start_all_btn.clicked.connect(controller._module_panel_start_all)
    controller.module_status_stop_all_btn = QPushButton("一括停止")
    controller.module_status_stop_all_btn.setToolTip(
        "動いているASR・LLM・TTS・MaAIをすべて停止します。音声対話中なら、先に音声対話を終了します。"
    )
    controller.module_status_stop_all_btn.clicked.connect(controller._module_panel_stop_all)
    module_status_head = QHBoxLayout()
    module_status_head.addStretch(1)
    module_status_head.addWidget(controller.module_status_start_all_btn)
    module_status_head.addWidget(controller.module_status_stop_all_btn)
    # 対話サーバー（ASR・LLM・TTS・MaAI を動かす裏のプロセス）。名前が長いので下のモジュールとは別の行にする。
    server_row = QHBoxLayout()
    server_row.setSpacing(6)
    controller.dialog_server_lamp = QLabel()
    controller.dialog_server_lamp.setFixedSize(12, 12)
    _server_name = QLabel("対話サーバー")
    _server_name.setToolTip("ASR・LLM・TTS・MaAI を動かして音声対話を進める裏のプロセスです。")
    controller.dialog_server_state_label = QLabel("接続待ち")
    controller.dialog_server_state_label.setMinimumWidth(52)
    controller.dialog_server_start_btn = QPushButton("▶")
    controller.dialog_server_start_btn.setProperty("role", "icon")
    controller.dialog_server_start_btn.setFixedSize(26, 24)
    controller.dialog_server_start_btn.setToolTip("対話サーバー起動")
    controller.dialog_server_start_btn.clicked.connect(lambda: controller._dialog_server_start())
    controller.dialog_server_stop_btn = QPushButton("■")
    controller.dialog_server_stop_btn.setProperty("role", "icon")
    controller.dialog_server_stop_btn.setFixedSize(26, 24)
    controller.dialog_server_stop_btn.setToolTip("対話サーバー停止（動いているモジュールと音声対話も止まります）")
    controller.dialog_server_stop_btn.clicked.connect(lambda: controller._dialog_server_stop())
    # どちらの対話サーバーを使うか。選ぶと GUI を開き直す（maai_sds/dialog_server_mode.py）。
    from ..dialog_server_mode import MODE_LABELS, docker_unavailable_reason
    from .. import paths as _paths

    controller.dialog_server_mode_combo = QComboBox()
    for _mode_key in ("local", "docker"):
        controller.dialog_server_mode_combo.addItem(MODE_LABELS[_mode_key], _mode_key)
    _docker_reason = "" if controller.remote_backend_enabled else docker_unavailable_reason(_paths.REPO_ROOT)
    if _docker_reason:
        _docker_item = controller.dialog_server_mode_combo.model().item(1)
        _docker_item.setEnabled(False)
        _docker_item.setToolTip(_docker_reason)
    controller.dialog_server_mode_combo.setCurrentIndex(1 if controller.remote_backend_enabled else 0)
    controller.dialog_server_mode_combo.activated.connect(controller._on_dialog_server_mode_activated)
    for _w in (
        controller.dialog_server_lamp,
        _server_name,
        controller.dialog_server_state_label,
        controller.dialog_server_start_btn,
        controller.dialog_server_stop_btn,
    ):
        server_row.addWidget(_w)
    server_row.addStretch(1)
    server_row.addWidget(controller.dialog_server_mode_combo)
    module_status_layout = QVBoxLayout()
    module_status_layout.addLayout(module_status_head)
    module_status_layout.addLayout(server_row)
    module_status_layout.addLayout(module_status_grid)
    module_status_box = QGroupBox("モジュール状態")
    module_status_box.setLayout(module_status_layout)

    # 左ペイン: 会話。履歴（新しいものが下に増える）の下に生成中の応答を置き、上から下へ読めるようにする。
    # voice_dialog_llm_stream_view（生成中の文の欄）は会話の吹き出しに統合したので画面には置かない
    # （中身の更新は従来どおり続く）。
    conversation_layout = QVBoxLayout()
    conversation_layout.setContentsMargins(4, 8, 4, 4)
    conversation_layout.addWidget(controller.voice_dialog_history_view, 1)
    # 進行中（まだ確定していないもの）: 認識中の発話と生成中の応答。会話の欄には確定だけを出す。
    live_panel = QFrame()
    live_panel.setProperty("role", "live-panel")
    live_grid = QGridLayout(live_panel)
    live_grid.setContentsMargins(12, 8, 12, 8)
    live_grid.setHorizontalSpacing(12)
    live_grid.setVerticalSpacing(4)
    for _row, (_name, _attr) in enumerate(
        (("認識中", "voice_dialog_live_user_label"), ("生成中", "voice_dialog_live_assistant_label"))
    ):
        _caption = QLabel(_name)
        _caption.setProperty("role", "caption")
        _text = QLabel("—")
        _text.setWordWrap(True)
        _text.setProperty("role", "live-text")
        _text.setProperty("empty", True)
        _text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        live_grid.addWidget(_caption, _row, 0, Qt.AlignmentFlag.AlignTop)
        live_grid.addWidget(_text, _row, 1)
        setattr(controller, _attr, _text)
    live_grid.setColumnStretch(1, 1)
    conversation_layout.addWidget(live_panel)
    conversation_box = QGroupBox("会話")
    conversation_box.setLayout(conversation_layout)
    left_panel = QVBoxLayout()
    left_panel.addWidget(conversation_box, 1)

    # 右ペイン: 操作ボタン系すべて（MMDAgent 起動・各ウィンドウ・音声対話操作）＋状態。
    # 使う順（MMDAgent-EX を起動 → 音声対話 → 状態の確認 → たまに設定）に上から並べる。
    right_panel = QVBoxLayout()
    right_panel.setSpacing(10)
    right_panel.addWidget(mmd_control_box)
    right_panel.addWidget(voice_dialog_controls_box)
    right_panel.addWidget(module_status_box)
    right_panel.addWidget(settings_window_box)
    right_panel.addStretch()
    right_panel.setContentsMargins(0, 0, 0, 0)
    right_widget = QWidget()
    right_widget.setLayout(right_panel)
    # 右の列はスクロールさせない。幅は 420px 固定で、ウィンドウの最小の高さは中身から決まる。
    right_widget.setFixedWidth(420)

    split = QHBoxLayout()
    split.setSpacing(12)
    split.addLayout(left_panel, 1)
    split.addWidget(right_widget, 0)
    root.addWidget(_build_header_bar(controller))
    root.addLayout(split, 1)
    status_timer = QTimer(controller)
    status_timer.setInterval(500)
    status_timer.timeout.connect(controller._refresh_header_status)
    status_timer.start()
    controller._header_status_timer = status_timer
    controller.setLayout(root)
    # 設定・テストの各画面を 1 枚のウィンドウにまとめる（全画面を作り終えたここで行う）。
    controller.settings_hub_window = SettingsHub(controller)
    controller._on_asr_engine_changed(controller.asr_engine_combo.currentText())
    controller._update_asr_compare_result_labels()
    controller._on_asr_compare_mode_changed()


def _build_header_bar(controller) -> QWidget:
    """上の帯: アプリ名と、対話サーバー・MMDAgent-EX・音声対話の状態（色付きの印）。右端に状態メッセージ。

    中身は controller._refresh_header_status() が 0.5 秒ごとに更新する。
    """
    bar = QWidget()
    bar.setProperty("role", "header")
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(4, 0, 4, 6)
    layout.setSpacing(14)
    title = QLabel("MaAI-SDS")
    title.setProperty("role", "app-title")
    layout.addWidget(title)
    controller.header_chips = {}
    chip_tips = {"backend": "ASR・LLM・TTS・MaAI を動かして音声対話を進める裏のプロセスです。この画面はそこにつないで操作します。"}
    for key in ("backend", "mmd", "voice"):
        chip = QWidget()
        if key in chip_tips:
            chip.setToolTip(chip_tips[key])
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(0, 0, 0, 0)
        chip_layout.setSpacing(6)
        lamp = QLabel()
        lamp.setFixedSize(10, 10)
        text = QLabel("")
        chip_layout.addWidget(lamp)
        chip_layout.addWidget(text)
        layout.addWidget(chip)
        controller.header_chips[key] = (lamp, text)
    layout.addStretch(1)
    controller.status_label.setProperty("role", "status")
    layout.addWidget(controller.status_label)
    return bar


def _build_initial_widgets(controller):
    """build_ui の先頭（ファイル設定・TTS・ASR などのウィジェット生成）。

    `build_ui` から機械的に切り出した（Phase 3-3）。文の順序は変えていない。
    前から何も受け取らず、後ろへ渡すのは `root` だけなので独立に切り出せた。
    """
    controller.setWindowTitle("MaAI-SDS")
    # 会話の欄が縦長（幅およそ 510px × 高さ 760px）になる大きさ。右の列は 420px 固定で、
    # スクロールなしで全部見える高さにする。
    controller.resize(980, 850)

    root = QVBoxLayout()
    controller.exec_edit = QLineEdit()
    controller.mdf_edit = QLineEdit()
    controller.tts_engine_combo = QComboBox()
    controller.tts_engine_combo.addItems(
        [
            "VOICEVOX",
            "VOICEVOX-Core",
            "Kokoro-RealtimeTTS",
            "Qwen3-TTS-CPP",
            "LFM2.5-Audio",
            "Cartesia-Realtime",
            "AivisSpeech",
            "Supertonic-3",
            "Irodori-v4",
            "Tsukasa",
            "TEMPLATE",
        ]
    )

    controller.engine_install_checks = {}
    controller.engine_install_status_labels = {}
    engine_install_layout = QVBoxLayout()
    enabled_engine_ids = set(engine_manager.get_enabled_engines())
    for engine_info in engine_manager.list_engines():
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        check = QCheckBox(engine_info.label)
        # extra=None used to mean "nothing Engine Manager can do here, always available"
        # (asr-qwen3/asr-external-tcp/tts-cartesia/tts-template: genuinely no setup action).
        # That stopped being true once dedicated-venv/asset engines with setup_venv_script or
        # setup_steps but no uv extra (asr-nemotron, tts-aivis, tts-supertonic,
        # tts-qwen3-cpp) were added — those DO have a real Apply action and must stay
        # toggleable, not forced-checked-and-disabled like the truly-external ones.
        has_setup_action = bool(engine_info.setup_steps or engine_info.setup_venv_script)
        always_available = engine_info.extra is None and not has_setup_action
        check.setChecked(True if always_available else (engine_info.id in enabled_engine_ids or engine_info.installed))
        check.setEnabled(not always_available)
        status = QLabel("インストール済み" if engine_info.installed else "未インストール")
        row_layout.addWidget(check, 1)
        row_layout.addWidget(status, 0)
        engine_install_layout.addWidget(row)
        controller.engine_install_checks[engine_info.id] = check
        controller.engine_install_status_labels[engine_info.id] = status
    controller.engine_install_apply_btn = QPushButton("適用")
    # --- GPU / CPU の選択 ---------------------------------------------------
    # torch と onnxruntime の CUDA ランタイムは cpu / cu128 の排他 extra に分かれており、
    # 適用のたびにどちらか一方が入る。setup_env.ps1 -Device を通らずに
    # `uv sync --extra gui` だけで入れた人（CPU 環境）が GPU へ移る唯一の場所。
    controller.device_extra_group = QGroupBox("デバイス構成")
    _device_row = QHBoxLayout()
    controller.device_extra_gpu_radio = QRadioButton("GPU（CUDA）")
    controller.device_extra_cpu_radio = QRadioButton("CPUのみ")
    controller.device_extra_gpu_radio.setToolTip(
        "torch cu128とonnxruntimeのCUDAランタイムをインストールします。NVIDIA GPUが必要です。"
    )
    controller.device_extra_cpu_radio.setToolTip(
        "CPU版のtorchとonnxruntimeをインストールします。nvidia-*一式（約3GB）は不要ですが、"
        "推論は遅くなります。"
    )
    # 操作した時点で明示的な選択として .device_extra に書く。書くまでは適用が
    # デバイス extra を渡さないので、`uv sync --extra gui` で作った環境の torch を
    # 勝手に入れ替えない。
    controller.device_extra_gpu_radio.toggled.connect(
        controller._on_device_extra_radio_toggled)
    controller.device_extra_cpu_radio.toggled.connect(
        controller._on_device_extra_radio_toggled)
    _device_row.addWidget(controller.device_extra_gpu_radio)
    _device_row.addWidget(controller.device_extra_cpu_radio)
    _device_row.addStretch(1)
    controller.device_extra_note = QLabel("")
    controller.device_extra_note.setWordWrap(True)
    _device_col = QVBoxLayout()
    _device_col.addLayout(_device_row)
    _device_col.addWidget(controller.device_extra_note)
    controller.device_extra_group.setLayout(_device_col)
    engine_install_layout.addWidget(controller.device_extra_group)

    controller.engine_install_apply_btn.clicked.connect(controller._apply_engine_installation)
    engine_install_layout.addWidget(controller.engine_install_apply_btn)
    # 断った理由・進捗をこの画面の中に出す。コンソールウィンドウ（別ウィンドウ）だけに
    # 出していると、閉じている人には「押しても何も起きない」ように見える。
    controller.engine_install_notice = QLabel("")
    controller.engine_install_notice.setWordWrap(True)
    controller.engine_install_notice.setVisible(False)
    engine_install_layout.addWidget(controller.engine_install_notice)
    controller.engine_install_box = QGroupBox("エンジンのインストール管理")
    controller.engine_install_box.setLayout(engine_install_layout)
    controller.voicevox_exe_edit = QLineEdit("")
    controller.voicevox_core_onnxruntime_edit = QLineEdit(
        r"voicevox_runtime/onnxruntime/lib/voicevox_onnxruntime.dll"
    )
    controller.voicevox_core_openjtalk_dict_edit = QLineEdit(
        r"voicevox_runtime/dict/open_jtalk_dic_utf_8-1.11"
    )
    controller.voicevox_core_vvm_edit = QLineEdit(r"voicevox_runtime/models/vvms/0.vvm")
    controller.voicevox_core_device_combo = QComboBox()
    # ランタイムの種類まで含めて 1 本で選ばせる。どれを選んだかで
    # setup_assets.ps1 の VoicevoxRuntime が取得するアーカイブが変わる
    # （cuda / dml / 素）。実測: CUDA 52ms / DirectML 249ms / CPU 620ms。
    # CUDA は cudnn64_9.dll を要する分、DirectML は GPU ベンダー非依存。
    controller.voicevox_core_device_combo.addItems(
        ["AUTO", "CPU", "GPU (CUDA)", "GPU (DirectML)"])
    controller.voicevox_core_device_combo.setCurrentText("GPU")
    controller.voicevox_core_cpu_threads_edit = QLineEdit("16")
    controller.voicevox_core_speed_scale_edit = QLineEdit("1.0")
    controller.voicevox_core_speed_scale_edit.setValidator(
        QDoubleValidator(0.5, 2.0, 2, controller.voicevox_core_speed_scale_edit)
    )
    controller.voicevox_core_speed_scale_edit.setToolTip(
        "0.50〜2.00。1.00が標準で、大きくすると速くなります。"
    )
    controller.tts_host_edit = QLineEdit("127.0.0.1")
    controller.tts_port_edit = QLineEdit("14500")
    controller.tts_base_path_edit = QLineEdit("")
    controller.tts_voice_edit = QLineEdit("1")
    controller.tts_model_edit = QLineEdit("kokoro")
    controller.tts_input_streaming_check = QCheckBox("有効")
    controller.tts_input_streaming_check.setChecked(True)
    controller.tts_segment_gap_silence_sec_edit = QLineEdit("0")
    controller.mmd_burst_snd_frames_edit = QLineEdit("20")
    controller.mmd_blink_enable_check = QCheckBox("自動まばたき")
    controller.mmd_blink_enable_check.setChecked(True)
    controller.mmd_blink_interval_mean_sec_edit = QLineEdit(str(DEFAULT_BLINK_INTERVAL_MEAN_SEC))
    controller.mmd_blink_interval_mean_sec_edit.setValidator(
        QDoubleValidator(1.0, 30.0, 2, controller.mmd_blink_interval_mean_sec_edit)
    )
    controller.mmd_blink_morph_edit = QLineEdit("まばたき")
    controller.mmd_blink_mode_combo = QComboBox()
    controller.mmd_blink_mode_combo.addItem("常に", "always")
    controller.mmd_blink_mode_combo.addItem("対話中のみ", "dialog")
    controller.audio_output_mode_combo = QComboBox()
    controller.audio_output_mode_combo.addItem("MMD（Remote）", "mmd")
    controller.audio_output_mode_combo.addItem("ローカルスピーカー", "local")
    controller.audio_output_mode_combo.addItem("TCP送出", "tcp")
    controller.audio_output_tcp_host_edit = QLineEdit("127.0.0.1")
    controller.audio_output_tcp_port_edit = QLineEdit("15601")
    controller.qwen3_tts_cpp_server_exe_edit = QLineEdit(
        r"audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe"
    )
    controller.qwen3_tts_cpp_model_path_edit = QLineEdit(
        r"audio.cpp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf"
    )
    controller.qwen3_tts_cpp_host_edit = QLineEdit("127.0.0.1")
    controller.qwen3_tts_cpp_port_edit = QLineEdit("18081")
    controller.qwen3_tts_cpp_model_id_edit = QLineEdit("qwen3-tts-customvoice")
    controller.qwen3_tts_cpp_speaker_edit = QLineEdit("Vivian")
    controller.qwen3_tts_cpp_instruct_edit = QLineEdit("")
    controller.qwen3_tts_cpp_language_edit = QLineEdit("japanese")
    controller.qwen3_tts_cpp_do_sample_check = QCheckBox("有効")
    controller.qwen3_tts_cpp_do_sample_check.setChecked(True)
    controller.qwen3_tts_cpp_temperature_edit = QLineEdit("0.9")
    controller.qwen3_tts_cpp_top_k_edit = QLineEdit("50")
    controller.qwen3_tts_cpp_top_p_edit = QLineEdit("1.0")
    controller.qwen3_tts_cpp_repetition_penalty_edit = QLineEdit("1.05")
    controller.qwen3_tts_cpp_seed_edit = QLineEdit("")
    # LFM2.5-Audio（audio.cpp）。声は選べない（モデルの声 1 つ）。
    controller.lfm2_audio_host_edit = QLineEdit("127.0.0.1")
    controller.lfm2_audio_port_edit = QLineEdit("18082")
    controller.lfm2_audio_model_id_edit = QLineEdit("lfm2.5-audio-jp")
    controller.lfm2_audio_model_path_edit = QLineEdit(r"audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF")
    controller.lfm2_audio_model_path_edit.setToolTip(
        "LFM2.5-Audio-1.5B-JP-GGUF のフォルダーを指定します（中に GGUF が複数入っています）。"
    )
    controller.lfm2_audio_server_exe_edit = QLineEdit(
        r"audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe"
    )
    controller.cartesia_tts_model_id_edit = QLineEdit("sonic-3")
    controller.cartesia_tts_sample_rate_edit = QLineEdit("16000")
    controller.cartesia_tts_api_key_edit = QLineEdit("")
    controller.cartesia_tts_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
    controller.cartesia_tts_api_key_env_edit = QLineEdit("CARTESIA_API_KEY")
    controller.aivisspeech_host_edit = QLineEdit("127.0.0.1")
    controller.aivisspeech_port_edit = QLineEdit("10104")
    # 既定値に %LOCALAPPDATA% を使うのは、ここが生成した人のユーザー名を含んだまま
    # ui_manifest.json に入り、parity.merge_app_ui_defaults 経由で全ユーザーの
    # 既定値として配られていたため。engine_manager の probe_paths も同じ表記。
    controller.aivisspeech_exe_edit = QLineEdit(r"%LOCALAPPDATA%\Programs\AivisSpeech\AivisSpeech-Engine\run.exe")
    controller.supertonic_host_edit = QLineEdit("127.0.0.1")
    controller.supertonic_port_edit = QLineEdit("18080")
    controller.supertonic_model_id_edit = QLineEdit("supertonic-3")
    controller.supertonic_server_exe_edit = QLineEdit(r"audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe")
    controller.supertonic_model_path_edit = QLineEdit(r"audio.cpp/models/Supertonic-3-GGUF/supertonic-3-orig.gguf")
    controller.supertonic_steps_edit = QLineEdit("8")
    controller.irodori_python_edit = QLineEdit(".venv_irodori/Scripts/python.exe")
    controller.irodori_checkpoint_edit = QLineEdit("Aratako/Irodori-TTS-v4.1-Small")
    controller.irodori_reference_audio_edit = QLineEdit("")
    controller.irodori_steps_edit = QLineEdit("16")
    controller.irodori_backend_combo = QComboBox()
    controller.irodori_backend_combo.addItem("speech.cpp（速い・専用の環境が要らない）", "speechcpp")
    controller.irodori_backend_combo.addItem("audio.cpp（500M v3 / 600M VoiceDesign）", "audiocpp")
    controller.irodori_backend_combo.addItem("PyTorch（専用の環境 .venv_irodori）", "pytorch")
    controller.irodori_backend_combo.setToolTip(
        "speech.cppはIrodori-TTS v4.1 Small MFで1文0.15〜0.3秒・最初の声まで約70ms（GPU）。"
        "audio.cppはIrodori-TTS 500M v3 / 600M v3 VoiceDesignで1文約0.35秒（読みのCER 0.004）。"
        "PyTorch版は1文1秒前後で、チェックポイントを選べます。"
    )
    controller.irodori_audiocpp_model_combo = QComboBox()
    controller.irodori_audiocpp_model_combo.addItem("Irodori-TTS 500M v3", "500m-v3")
    controller.irodori_audiocpp_model_combo.addItem(
        "Irodori-TTS 600M v3 VoiceDesign（声を文章で指定）", "600m-v3-voicedesign"
    )
    controller.irodori_audiocpp_model_combo.setToolTip(
        "VoiceDesignは声質キャプション（話者欄の文）で声を指定します。参照WAVがあればそちらを優先します。"
    )
    controller.irodori_audiocpp_host_edit = QLineEdit("127.0.0.1")
    controller.irodori_audiocpp_port_edit = QLineEdit("18083")
    controller.irodori_speechcpp_device_combo = QComboBox()
    controller.irodori_speechcpp_device_combo.addItem("GPU（外付け。無ければCPU）", "gpu")
    controller.irodori_speechcpp_device_combo.addItem("CPU", "cpu")
    controller.irodori_speechcpp_device_combo.setToolTip("CPUでは1文4秒前後かかり、対話には向きません。")
    controller.irodori_speechcpp_steps_edit = QLineEdit("4")
    controller.irodori_speechcpp_steps_edit.setToolTip("speech.cpp版の生成ステップ数です。MFモデルは4で作られています。")
    controller.irodori_seed_edit = QLineEdit("1")
    controller.irodori_seed_edit.setToolTip(
        "参照WAVが無いときの声はseedで決まります。固定しておくと文ごとに声が変わりません（speech.cpp版）。"
    )
    for _irodori_combo in (
        controller.irodori_backend_combo,
        controller.irodori_speechcpp_device_combo,
        controller.irodori_audiocpp_model_combo,
    ):
        _irodori_combo.currentIndexChanged.connect(lambda _index: controller._save_app_config())
    controller.tsukasa_root_edit = QLineEdit("Tsukasa-HF")
    controller.tsukasa_reference_audio_edit = QLineEdit("")
    controller.tsukasa_steps_edit = QLineEdit("5")
    controller.tsukasa_alpha_edit = QLineEdit("0.3")
    controller.tsukasa_beta_edit = QLineEdit("0.7")
    controller.irodori_reference_audio_browse_btn = QPushButton("参照…")
    controller.irodori_reference_audio_browse_btn.clicked.connect(
        lambda: controller._browse_tts_reference_audio(
            "irodori_reference_audio_edit", "Irodori の参照音声を選択"
        )
    )
    controller.tsukasa_reference_audio_browse_btn = QPushButton("参照…")
    controller.tsukasa_reference_audio_browse_btn.clicked.connect(
        lambda: controller._browse_tts_reference_audio(
            "tsukasa_reference_audio_edit", "Tsukasa の参照音声を選択"
        )
    )
    controller.voicevox_core_onnxruntime_browse_btn = QPushButton("参照…")
    controller.voicevox_core_onnxruntime_browse_btn.clicked.connect(controller._browse_voicevox_core_onnxruntime)
    controller.voicevox_core_openjtalk_dict_browse_btn = QPushButton("参照…")
    controller.voicevox_core_openjtalk_dict_browse_btn.clicked.connect(controller._browse_voicevox_core_openjtalk_dict)
    controller.voicevox_core_vvm_browse_btn = QPushButton("参照…")
    controller.voicevox_core_vvm_browse_btn.clicked.connect(controller._browse_voicevox_core_vvm)
    controller.aivisspeech_exe_browse_btn = QPushButton("参照…")
    controller.aivisspeech_exe_browse_btn.clicked.connect(controller._browse_aivisspeech_exe)
    controller.supertonic_server_exe_browse_btn = QPushButton("参照…")
    controller.supertonic_server_exe_browse_btn.clicked.connect(controller._browse_supertonic_server_exe)
    controller.supertonic_model_path_browse_btn = QPushButton("参照…")
    controller.supertonic_model_path_browse_btn.clicked.connect(controller._browse_supertonic_model_path)
    # 編集が controller_config.json へ保存されるよう _save_app_config に接続する（従来は未接続で編集が消えていた）。
    for _cartesia_edit in (
        controller.cartesia_tts_model_id_edit,
        controller.cartesia_tts_sample_rate_edit,
        controller.cartesia_tts_api_key_edit,
        controller.cartesia_tts_api_key_env_edit,
    ):
        _cartesia_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_auth_mode_combo = QComboBox()
    controller.tts_auth_mode_combo.addItems(["none", "bearer_env", "header_env"])
    controller.tts_auth_env_edit = QLineEdit("")
    controller.tts_auth_header_edit = QLineEdit("X-API-Key")
    controller.tts_auth_prefix_edit = QLineEdit("Bearer ")
    controller.tts_template_edit = QPlainTextEdit()
    controller.tts_template_edit.setPlaceholderText("TTSテンプレートJSONを入力")
    controller.tts_template_edit.setMinimumHeight(180)
    controller.tts_test_text_edit = QLineEdit("TTSテストです。")
    controller.tts_latency_value_label = QLabel("-")
    controller.llm_backend_combo = QComboBox()
    controller.llm_backend_combo.addItems(["LM Studio", "OpenAI", "Custom"])
    controller.llm_host_edit = QLineEdit("http://127.0.0.1:1234")
    controller.llm_custom_base_url_edit = QLineEdit("http://127.0.0.1:8000")
    controller.llm_custom_api_key_edit = QLineEdit("")
    controller.llm_custom_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
    controller.llm_custom_api_key_env_edit = QLineEdit("CUSTOM_LLM_API_KEY")
    controller.llm_model_lmstudio_combo = QComboBox()
    controller.llm_model_lmstudio_combo.setEditable(True)
    controller.llm_model_openai_combo = QComboBox()
    controller.llm_model_openai_combo.setEditable(True)
    controller.llm_model_openai_combo.addItem("gpt-4o-mini")
    controller.llm_model_openai_combo.setCurrentText("gpt-4o-mini")
    controller.llm_model_custom_combo = QComboBox()
    controller.llm_model_custom_combo.setEditable(True)
    controller.llm_model_custom_combo.addItem("Qwen/Qwen3.5-35B-A3B")
    controller.llm_model_custom_combo.setCurrentText("Qwen/Qwen3.5-35B-A3B")
    controller.llm_api_key_edit = QLineEdit("")
    controller.llm_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
    controller.llm_api_key_env_edit = QLineEdit("OPENAI_API_KEY")
    controller.llm_system_prompt_edit = QPlainTextEdit()
    controller.llm_system_prompt_edit.setPlainText("あなたは簡潔で丁寧なアシスタントです。")
    controller.llm_system_prompt_edit.setMinimumHeight(48)
    controller.llm_system_prompt_edit.setMaximumHeight(110)
    controller.llm_context_len_edit = QLineEdit("10")
    controller.llm_max_tokens_edit = QLineEdit("128")
    controller.llm_custom_send_temperature_check = QCheckBox("リクエストにtemperatureを含める")
    controller.llm_custom_send_temperature_check.setChecked(True)
    controller.llm_custom_send_max_tokens_check = QCheckBox("リクエストにmax_tokensを含める")
    controller.llm_custom_send_max_tokens_check.setChecked(True)
    controller.llm_custom_temperature_edit = QLineEdit("0.7")
    controller.llm_custom_extra_body_thinking_check = QCheckBox(
        "extra_body：chat_template_kwargs.enable_thinking=False（Qwen3など）"
    )
    controller.llm_custom_extra_body_thinking_check.setChecked(False)
    controller.llm_tts_stream_check = QCheckBox("有効")
    controller.llm_tts_stream_check.setChecked(False)
    controller.llm_test_tts_input_streaming_check = QCheckBox("有効")
    controller.llm_test_tts_input_streaming_check.setChecked(True)
    controller.llm_split_custom_delimiters_edit = QLineEdit("。．！？!?\\n")
    controller.llm_split_custom_strip_emoji_check = QCheckBox("「Custom」選択時に入力から絵文字を除去する")
    controller.llm_split_custom_strip_emoji_check.setChecked(True)
    controller.llm_split_custom_strip_parenthetical_check = QCheckBox(
        "「Custom」選択時に括弧で囲まれた部分を除去する"
    )
    controller.llm_split_custom_strip_parenthetical_check.setChecked(False)
    controller.llm_split_custom_strip_standalone_warai_check = QCheckBox(
        "「Custom」選択時に単独の「笑」を除去する（笑い・笑う・笑わ・笑え・笑お・笑っ…・笑顔は残す）"
    )
    controller.llm_split_custom_strip_standalone_warai_check.setChecked(False)
    controller.llm_user_input_edit = QLineEdit("")
    controller.llm_chat_view = QPlainTextEdit()
    controller.llm_chat_view.setReadOnly(True)
    controller.llm_chat_view.setPlaceholderText("LLMとの対話ログがここに表示されます")
    controller.llm_status_label = QLabel("状態：待機")
    controller.asr_engine_combo = QComboBox()
    controller.asr_engine_combo.addItems(_ASR_ENGINE_CHOICES)
    controller.asr_engine_combo_settings = QComboBox()
    controller.asr_engine_combo_settings.addItems(_ASR_ENGINE_CHOICES)
    controller.asr_vad_backend_combo = QComboBox()
    controller.asr_vad_backend_combo.addItems(["maai", "silero"])
    controller.asr_turn_backend_combo = QComboBox()
    controller.asr_turn_backend_combo.addItems(["maai", "smart-turn"])
    controller.asr_device_combo = QComboBox()
    controller.asr_device_combo.addItems(["cuda", "cpu"])
    controller.asr_compute_type_combo = QComboBox()
    controller.asr_compute_type_combo.addItems(["int8_float16", "float16", "int8", "float32"])
    controller.asr_whisper_model_edit = QLineEdit("large-v3")
    controller.asr_whisper_model_edit.setToolTip(
        "faster-whisperのサイズ名（tiny／base／small／medium／large-v3／turboなど）。"
        "HF名もローカルパスも空のときだけ使います。"
    )
    controller.asr_whisper_hf_edit = QLineEdit("deepdml/faster-whisper-large-v3-turbo-ct2")
    controller.asr_whisper_hf_edit.setToolTip(
        "Hugging FaceのモデルID（例：deepdml/faster-whisper-large-v3-turbo-ct2）。"
        "ローカルパスが空のときに使います。"
    )
    controller.asr_whisper_model_path_edit = QLineEdit("")
    controller.asr_whisper_model_path_edit.setToolTip(
        "展開済みのCTranslate2ディレクトリです。指定するとHF名より優先します。"
    )
    controller.asr_sherpa_tokens_edit = QLineEdit("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt")
    controller.asr_sherpa_encoder_edit = QLineEdit("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.int8.onnx")
    controller.asr_sherpa_decoder_edit = QLineEdit("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.onnx")
    controller.asr_sherpa_joiner_edit = QLineEdit("sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.int8.onnx")
    controller.asr_sherpa_num_threads_edit = QLineEdit("2")
    controller.asr_sherpa_provider_combo = QComboBox()
    controller.asr_sherpa_provider_combo.addItem("CPU", "cpu")
    controller.asr_sherpa_provider_combo.addItem("GPU (cuda)", "cuda")
    controller.asr_sherpa_provider_combo.setToolTip(
        "既定はCPUです。int8のままGPUにすると遅くなります（実測122ms→203ms。"
        "dynamic量子化にCUDAカーネルがないため）。GPUを使う場合は上のEncoder／Joinerも"
        "fp32（*-epoch-99-avg-1.onnx、int8と同じフォルダーに同梱）へ変更してください。"
        "その組み合わせが最速かつ最良で、実測102ms／CER 0.1367（int8＋CPU・2スレッドは152ms／0.1414）です。"
    )
    # qwen3-asr-onnx (sherpa-onnx OfflineRecognizer.from_qwen3_asr)
    # GitHub 公開物は int8 のみ。エンジン管理の「Qwen3-ASR ONNX」適用でこのディレクトリが来る。
    controller.asr_qwen3_onnx_conv_frontend_edit = QLineEdit(
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx")
    controller.asr_qwen3_onnx_encoder_edit = QLineEdit(
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx")
    controller.asr_qwen3_onnx_decoder_edit = QLineEdit(
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx")
    controller.asr_qwen3_onnx_tokenizer_edit = QLineEdit(
        "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer")
    controller.asr_qwen3_onnx_language_edit = QLineEdit("Japanese")
    controller.asr_qwen3_onnx_language_edit.setToolTip("言語を強制します（フル名：Japanese／English／Chineseなど。空欄＝自動判定）。")
    controller.asr_qwen3_onnx_provider_combo = QComboBox()
    controller.asr_qwen3_onnx_provider_combo.addItem("CPU", "cpu")
    controller.asr_qwen3_onnx_provider_combo.addItem("GPU (cuda)", "cuda")
    controller.asr_qwen3_onnx_provider_combo.setCurrentText("CPU")
    controller.asr_qwen3_onnx_provider_combo.setToolTip(
        "公開int8（*.int8.onnx）はCPU＋多スレッドの方が高速です（MatMulIntegerがGPUに乗らないため）。"
        "fp32（encoder.onnx／decoder.onnx）を用意した場合のみGPU (cuda)が高速です。"
    )
    controller.asr_qwen3_onnx_num_threads_edit = QLineEdit("8")
    controller.asr_qwen3_onnx_num_threads_edit.setToolTip(
        "ONNXのCPUスレッド数です。int8＋CPUは8、fp32＋GPUは2を推奨します。"
    )
    # sherpa-onnx OfflineRecognizer 系（parakeet / sensevoice / funasr-nano / cohere）。
    # 4 つとも k2-fsa の release tarball を展開した**ディレクトリ**を指す（中のファイル名は
    # asr/engines_sherpa_offline._pick が解決するので int8/fp32 で設定を変えなくてよい）。
    _SHERPA_OFFLINE_PROVIDER_TIP = (
        "配布されているint8モデルはdynamic量子化でCUDAカーネルが効かないため、"
        "実測ではCPUの方が速く、最悪遅延も安定します（既定はCPU）。fp32に差し替えた場合のみGPUを検討してください。"
    )
    controller.asr_parakeet_onnx_dir_edit = QLineEdit("sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8")
    controller.asr_parakeet_onnx_dir_edit.setToolTip(
        "Parakeet TDT-CTC 0.6B日本語モデルの展開先ディレクトリです。k2-fsaの配布物はCTCブランチのみで、"
        "TDTデコーダー版より精度は落ちますが、非自己回帰のため遅延は安定します。"
    )
    controller.asr_parakeet_onnx_provider_combo = QComboBox()
    controller.asr_parakeet_onnx_provider_combo.addItem("CPU", "cpu")
    controller.asr_parakeet_onnx_provider_combo.addItem("GPU (cuda)", "cuda")
    controller.asr_parakeet_onnx_provider_combo.setToolTip(_SHERPA_OFFLINE_PROVIDER_TIP)
    controller.asr_parakeet_onnx_num_threads_edit = QLineEdit("8")

    controller.asr_sensevoice_onnx_dir_edit = QLineEdit("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
    controller.asr_sensevoice_onnx_dir_edit.setToolTip(
        "SenseVoiceSmallの展開先ディレクトリです。2024-07-17（本家FunAudioLLM由来）を使用してください。"
        "同名の2025-09-09は広東語ファインチューンで、日本語を簡体字で書き起こします。"
    )
    controller.asr_sensevoice_onnx_provider_combo = QComboBox()
    controller.asr_sensevoice_onnx_provider_combo.addItem("CPU", "cpu")
    controller.asr_sensevoice_onnx_provider_combo.addItem("GPU (cuda)", "cuda")
    controller.asr_sensevoice_onnx_provider_combo.setToolTip(_SHERPA_OFFLINE_PROVIDER_TIP)
    controller.asr_sensevoice_onnx_num_threads_edit = QLineEdit("8")
    controller.asr_sensevoice_onnx_language_edit = QLineEdit("ja")
    controller.asr_sensevoice_onnx_language_edit.setToolTip("ja／en／zh／ko／yue。空欄＝自動判定。")
    controller.asr_sensevoice_onnx_itn_check = QCheckBox("有効")
    controller.asr_sensevoice_onnx_itn_check.setChecked(True)
    controller.asr_sensevoice_onnx_itn_check.setToolTip(
        "逆テキスト正規化です。句読点と数字表記が付きます。実測では有効時にCERが下がりました（0.153→0.090）。"
    )

    controller.asr_funasr_nano_onnx_dir_edit = QLineEdit("sherpa-onnx-funasr-nano-int8-2025-12-30")
    controller.asr_funasr_nano_onnx_provider_combo = QComboBox()
    controller.asr_funasr_nano_onnx_provider_combo.addItem("CPU", "cpu")
    controller.asr_funasr_nano_onnx_provider_combo.addItem("GPU (cuda)", "cuda")
    controller.asr_funasr_nano_onnx_provider_combo.setToolTip(_SHERPA_OFFLINE_PROVIDER_TIP)
    controller.asr_funasr_nano_onnx_num_threads_edit = QLineEdit("8")
    controller.asr_funasr_nano_onnx_language_edit = QLineEdit("ja")
    controller.asr_funasr_nano_onnx_itn_check = QCheckBox("有効")
    controller.asr_funasr_nano_onnx_itn_check.setChecked(True)
    controller.asr_funasr_nano_onnx_max_new_tokens_edit = QLineEdit("128")
    controller.asr_funasr_nano_onnx_max_new_tokens_edit.setToolTip(
        "LLMデコーダーの生成上限です。最悪遅延の上限になるため、対話用途では小さくします。"
        "1発話なら128で十分です（長い読み上げを扱う場合のみ増やします）。"
    )

    controller.asr_cohere_onnx_dir_edit = QLineEdit("sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01")
    controller.asr_cohere_onnx_dir_edit.setToolTip(
        "Cohere Transcribe（日本語を含む14言語）の展開先ディレクトリです。"
        "encoderはONNX external data形式のため、encoder.int8.onnx.dataも同じ場所に置いてください。"
    )
    controller.asr_cohere_onnx_provider_combo = QComboBox()
    controller.asr_cohere_onnx_provider_combo.addItem("CPU", "cpu")
    controller.asr_cohere_onnx_provider_combo.addItem("GPU (cuda)", "cuda")
    controller.asr_cohere_onnx_provider_combo.setToolTip(_SHERPA_OFFLINE_PROVIDER_TIP)
    controller.asr_cohere_onnx_num_threads_edit = QLineEdit("8")
    controller.asr_cohere_onnx_language_edit = QLineEdit("ja")
    controller.asr_cohere_onnx_punct_check = QCheckBox("有効")
    controller.asr_cohere_onnx_punct_check.setChecked(True)
    controller.asr_cohere_onnx_itn_check = QCheckBox("有効")
    controller.asr_cohere_onnx_itn_check.setChecked(True)
    # speech.cpp（ggml）。モデルは初回に speech.cpp が Hugging Face から取ってくる。
    controller.asr_speechcpp_model_combo = QComboBox()
    for _name, _label in speechcpp.ASR_MODELS:
        controller.asr_speechcpp_model_combo.addItem(_label, _name)
    controller.asr_speechcpp_model_combo.setToolTip(
        "speech.cppで動かすモデルです。初めて選んだモデルは起動時に取得します（0.7〜2.2GB）。"
        "時間はRTX 5070 Ti LaptopのGPUで1発話あたり。"
    )
    controller.asr_speechcpp_device_combo = QComboBox()
    controller.asr_speechcpp_device_combo.addItem("GPU（外付け。無ければCPU）", "gpu")
    controller.asr_speechcpp_device_combo.addItem("CPU", "cpu")
    controller.asr_speechcpp_device_combo.setToolTip(
        "GPUはVulkanで動きます（CUDAは不要）。内蔵GPUはCPUより遅かったので選びません。"
        "CPUではHaas-jaで1発話0.2〜0.3秒かかるため、疑似ストリーミングより発話の終わりの1回の認識向きです。"
    )
    controller.asr_speechcpp_threads_edit = QLineEdit("0")
    controller.asr_speechcpp_threads_edit.setToolTip("CPUのスレッド数です。0＝speech.cppに任せる（性能コアの数）。")
    # nemotron-onnx (sherpa-onnx OnlineRecognizer.from_transducer, multilingual)
    controller.asr_nemotron_onnx_tokens_edit = QLineEdit("")
    controller.asr_nemotron_onnx_encoder_edit = QLineEdit("")
    controller.asr_nemotron_onnx_decoder_edit = QLineEdit("")
    controller.asr_nemotron_onnx_joiner_edit = QLineEdit("")
    controller.asr_nemotron_onnx_language_edit = QLineEdit("auto")
    controller.asr_nemotron_onnx_language_edit.setToolTip(
        "言語ヒント（空欄／auto＝自動、ja／en／ja-JPなど）。多言語Nemotron ONNXにはsherpa-onnx v1.13.3以降（PR #3671）が必要です。"
    )
    controller.asr_smart_turn_model_edit = QLineEdit("smart-turn-v3.2-gpu.onnx")
    controller.asr_qwen_host_edit = QLineEdit("127.0.0.1")
    controller.asr_qwen_port_edit = QLineEdit("8000")
    controller.asr_external_host_edit = QLineEdit("127.0.0.1")
    controller.asr_external_port_edit = QLineEdit("19090")
    controller.asr_external_audio_host_edit = QLineEdit("127.0.0.1")
    controller.asr_external_audio_port_edit = QLineEdit("14600")
    controller.asr_external_start_script_edit = QLineEdit("")
    controller.asr_qwen_model_edit = QComboBox()
    controller.asr_qwen_model_edit.addItems(["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B"])
    controller.asr_qwen_model_edit.setCurrentText("Qwen/Qwen3-ASR-1.7B")
    controller.asr_qwen_model_edit.setToolTip("Hugging FaceのモデルIDです（カスタムモデルは下のパスで指定）。")
    controller.asr_qwen_custom_model_path_edit = QLineEdit("")
    controller.asr_qwen_custom_model_path_edit.setPlaceholderText(
        "空欄では上記HFモデルを使用します。入力すると優先されます（プロジェクト配下はDockerで/data/shared/Qwen3-ASR/...へ変換）。"
    )
    controller.asr_qwen_custom_model_path_edit.setToolTip(
        "ローカルチェックポイント等。参照でフォルダを選ぶとホストパスが入ります。"
    )
    controller.asr_qwen_gpu_util_edit = QLineEdit("0.8")
    controller.asr_qwen_max_model_len_edit = QLineEdit("4096")
    controller.asr_qwen_chunk_sec_edit = QLineEdit("2.0")
    controller.asr_qwen_timeout_sec_edit = QLineEdit("30")
    controller.asr_qwen_ws_path_edit = QLineEdit("/ws")
    controller.asr_qwen_session_timeout_edit = QLineEdit("60")
    controller.asr_qwen_no_text_flush_sec_edit = QLineEdit("0")
    controller.asr_qwen_min_return_audio_sec_edit = QLineEdit("0")
    controller.asr_nemotron_model_edit = QLineEdit("nvidia/nemotron-3.5-asr-streaming-0.6b")
    controller.asr_nemotron_model_edit.setToolTip(
        "Hugging FaceのモデルIDまたはローカル.nemoパス"
    )
    controller.asr_nemotron_chunk_combo = QComboBox()
    controller.asr_nemotron_chunk_combo.addItems(["80", "320", "560", "1120"])
    controller.asr_nemotron_chunk_combo.setCurrentText("320")
    controller.asr_nemotron_chunk_combo.setToolTip(
        "コンテキストサイズ（cache-aware att_contextのチャンク、ms）です。"
        "80＝最低遅延～1120＝最高精度（多言語NeMoの既定は320ms）。"
        "nemotron-onnxはチャンクごとに別のONNXを使います（パス内の-560ms-などを自動置換）。"
    )
    controller.asr_nemotron_onnx_chunk_combo = QComboBox()
    controller.asr_nemotron_onnx_chunk_combo.addItems(["80", "320", "560", "1120"])
    controller.asr_nemotron_onnx_chunk_combo.setCurrentText("320")
    controller.asr_nemotron_onnx_chunk_combo.setToolTip(controller.asr_nemotron_chunk_combo.toolTip())
    controller.asr_nemotron_onnx_compute_combo = QComboBox()
    controller.asr_nemotron_onnx_compute_combo.addItem("GPU (fp32)", "gpu-fp32")
    controller.asr_nemotron_onnx_compute_combo.addItem("CPU (int8)", "cpu-int8")
    controller.asr_nemotron_onnx_compute_combo.setToolTip(
        "実行モードです。GPU (fp32)：CUDAでfp32 encoderを使用（最低遅延約0.4秒・最高精度、NVIDIA GPUとGPUビルドのsherpaが必要）。"
        "CPU (int8)：GPU不要でint8を使用（遅延約1.3秒・精度はほぼ同等）。"
        "int8はGPUで高速化されないため、GPU使用時はfp32を推奨します。"
    )
    controller.asr_nemotron_window_sec_edit = QLineEdit("15.0")
    controller.asr_nemotron_window_sec_edit.setToolTip(
        "前処理に使う直近音声の窓（秒）です。大きいほど正規化が安定しますが、前処理コストが増えます。"
    )
    controller.asr_nemotron_max_symbols_edit = QLineEdit("30")
    controller.asr_nemotron_max_symbols_edit.setToolTip(
        "RNNT greedyの1フレームあたり最大トークン数です。小さいと日本語を取りこぼします（既定：30）。"
    )
    controller.asr_nemotron_lang_edit = QLineEdit("ja-JP")
    controller.asr_nemotron_lang_edit.setToolTip('言語コード（例：ja-JP、en-US）または"auto"')
    controller.asr_nemotron_device_combo = QComboBox()
    controller.asr_nemotron_device_combo.addItems(["cuda", "cpu"])
    controller.asr_nemotron_finalize_combo = QComboBox()
    controller.asr_nemotron_finalize_combo.addItems(["vad", "continuous"])
    controller.asr_nemotron_finalize_combo.setToolTip(
        "vad：VADで発話を区切り、partial→finalを確定します。continuous：区切らず連続で文字起こしします。"
    )
    controller.asr_nemotron_onnx_finalize_combo = QComboBox()
    controller.asr_nemotron_onnx_finalize_combo.addItems(["vad", "continuous"])
    controller.asr_nemotron_onnx_finalize_combo.setToolTip(
        controller.asr_nemotron_finalize_combo.toolTip()
    )
    controller.asr_nemotron_python_edit = QLineEdit(".venv_nemotron/Scripts/python.exe")
    controller.asr_nemotron_python_edit.setToolTip(
        "Nemotron推論を実行する専用venvのpython.exeです。アプリ環境とは別プロセスです。"
        "scripts/setup_nemotron_venv.ps1が.venv_nemotronを作成します（nemo-toolkit==3.0.0固定）。"
        "相対パスはリポジトリルート基準で解決される。"
    )
    controller.audio_input_mode_combo = QComboBox()
    controller.audio_input_mode_combo.addItems(["mic", "wav", "tcp"])
    controller.audio_input_mic_device_combo = QComboBox()
    controller.audio_input_mic_device_combo.addItem("default", -1)
    controller.audio_input_wav_path_edit = QLineEdit("")
    controller.audio_input_wav_start_sec_edit = QLineEdit("0")
    controller.audio_input_wav_end_sec_edit = QLineEdit("")
    controller.audio_input_wav_channel_combo = QComboBox()
    controller.audio_input_wav_channel_combo.addItem("ch1", 0)
    return root


def build_ui(controller) -> None:
    root = _build_initial_widgets(controller)
    controller.audio_input_wav_channel_combo.addItem("ch2", 1)
    controller.audio_input_wav_channel_combo.setToolTip(
        "ステレオWAVのどちらのチャンネルをモノラル入力に使うか選びます。モノラルファイルでは無視されます。"
    )
    controller.audio_input_wav_local_playback_check = QCheckBox("ローカル再生（スピーカー）")
    controller.audio_input_wav_local_playback_check.setToolTip(
        "WAVをパイプラインへ送るのと同時に、PCの既定出力でモニター再生します。"
    )
    controller.audio_input_wav_other_ch_to_mmd_check = QCheckBox("逆chをMMDAgentへ送信")
    controller.audio_input_wav_other_ch_to_mmd_check.setToolTip(
        "ステレオWAVの入力チャンネルと逆のチャンネルの音声をMMDAgent-EXへSND送信します。"
        "モノラルファイルでは無視されます。"
    )
    controller.audio_input_wav_other_ch_to_maai_ch2_check = QCheckBox("逆chをMaAI ch2へ注入")
    controller.audio_input_wav_other_ch_to_maai_ch2_check.setToolTip(
        "ステレオWAVの逆チャンネルをMaAIのシステム音声入力（ch2）として注入します。"
        "モノラルファイルでは無視されます。"
    )
    controller.audio_input_wav_aec_check = QCheckBox("AEC（エコーキャンセル）")
    controller.audio_input_wav_aec_check.setToolTip(
        "マイク・WAV・TCPのいずれの入力モードでも利用できます。"
        "参照信号は、マイク／TCPではシステム音声（TTSなど）、ステレオWAVでは未選択チャンネルも選べます。"
    )
    controller.audio_input_wav_aec_ref_combo = QComboBox()
    controller.audio_input_wav_aec_ref_combo.addItem("未選択ch", "other_ch")
    controller.audio_input_wav_aec_ref_combo.addItem("システム音声", "system")
    controller.audio_input_wav_aec_ref_combo.setToolTip(
        "WAVがステレオのときのみ「未選択ch」が有効です。"
        "マイク／TCPではシステム音声（TTSなど）のみ参照できます。\n"
        "システム音声：TTSなどのシステム出力を参照信号に使います。"
    )
    audio_input_wav_browse_btn = QPushButton("参照…")
    audio_input_wav_play_pause_btn = QPushButton("再生")
    audio_input_wav_end_btn = QPushButton("終了")
    controller.audio_input_wav_play_pause_btn = audio_input_wav_play_pause_btn
    controller.audio_input_wav_end_btn = audio_input_wav_end_btn
    audio_input_wav_play_pause_btn.setToolTip("再生中は一時停止、停止中は再生")
    audio_input_wav_end_btn.setToolTip("再生を止め、ファイル先頭に戻します（ワーカーは起動したまま）")
    controller.audio_input_tcp_host_edit = QLineEdit("127.0.0.1")
    controller.audio_input_tcp_port_edit = QLineEdit("15600")
    controller.audio_input_gain_edit = QLineEdit("1.0")
    controller.audio_input_gain_label = QLabel("入力ゲイン（倍率）")
    controller.audio_input_gain_label.setToolTip(
        "マイク／WAV／TCPのユーザー入力に乗算します（-1～1にクリップ）。反映には音声入力ワーカーの再起動が必要です。"
    )
    controller.audio_input_gain_edit.setToolTip(
        "1.0で変更なし、2.0で約2倍、0で無音です。最大20倍まで指定できます。"
    )
    controller.asr_qwen_finish_on_vad_end_check = QCheckBox("有効")
    controller.asr_qwen_finish_on_vad_end_check.setChecked(True)
    controller.asr_qwen_stop_container_check = QCheckBox("有効")
    controller.asr_qwen_stop_container_check.setChecked(False)
    controller.asr_qwen_enforce_eager_check = QCheckBox("有効")
    controller.asr_qwen_enforce_eager_check.setChecked(os.name == "nt")
    controller.asr_qwen_enforce_eager_check.setToolTip(
        "vLLMのtorch.compile／CUDA graphを無効化します。WSL2でcudaErrorIllegalAddressが出る場合に有効にします。"
    )
    controller.asr_chirp_project_id_edit = QLineEdit("")
    controller.asr_chirp_location_edit = QLineEdit("global")
    controller.asr_chirp_recognizer_edit = QLineEdit("_")
    controller.asr_chirp_model_edit = QLineEdit("chirp_3")
    controller.asr_vad_on_threshold_edit = QLineEdit("0.1")
    controller.asr_vad_on_sec_edit = QLineEdit("0.1")
    controller.asr_vad_off_threshold_edit = QLineEdit("0.2")
    controller.asr_vad_off_sec_edit = QLineEdit("0.1")
    controller.asr_min_segment_sec_edit = QLineEdit("0.5")
    controller.asr_pre_roll_buffer_sec_edit = QLineEdit("1.0")
    controller.asr_nonstream_pseudo_enabled_check = QCheckBox("有効")
    controller.asr_nonstream_pseudo_enabled_check.setChecked(True)
    controller.asr_nonstream_pseudo_chunk_sec_edit = QLineEdit("0.25")
    controller.asr_maai_turn_threshold_edit = QLineEdit("0.5")
    controller.asr_smart_turn_threshold_edit = QLineEdit("0.5")
    controller.asr_smart_turn_num_threads_edit = QLineEdit("2")
    controller.asr_torch_cpu_intra_threads_edit = QLineEdit("2")
    controller.asr_torch_cpu_intra_threads_edit.setToolTip(
        "ASRプロセスのTorchスレッド上限（既定：2）です。turn_backend=maaiのときにターン判定MaAIが使用します。"
        "上げるとMaAIワーカーやTTSとCPUコアを奪い合います。"
    )
    controller.asr_smart_turn_num_threads_edit.setToolTip(
        "SmartTurn ONNXの演算内スレッド数（既定：2）です。未指定ではORTがCPUコア数－1本まで使用し、MaAIやTTSと奪い合います。"
    )
    controller.asr_result_edit = QPlainTextEdit()
    controller.asr_result_edit.setReadOnly(True)
    controller.asr_result_edit.setPlaceholderText("ASR認識結果がここに表示されます")
    controller.asr_compare_mode_check = QCheckBox("有効")
    controller.asr_compare_top_engine_combo = QComboBox()
    controller.asr_compare_top_engine_combo.addItems(_ASR_ENGINE_CHOICES)
    controller.asr_compare_bottom_engine_combo = QComboBox()
    controller.asr_compare_bottom_engine_combo.addItems(_ASR_ENGINE_CHOICES)
    controller.asr_compare_bottom_engine_combo.setCurrentText("faster-whisper")
    controller.asr_compare_top_result_edit = QPlainTextEdit()
    controller.asr_compare_top_result_edit.setReadOnly(True)
    controller.asr_compare_top_result_edit.setPlaceholderText("比較結果（上段）がここに表示されます")
    controller.asr_compare_bottom_result_edit = QPlainTextEdit()
    controller.asr_compare_bottom_result_edit.setReadOnly(True)
    controller.asr_compare_bottom_result_edit.setPlaceholderText("比較結果（下段）がここに表示されます")
    controller.status_label = QLabel("状態：未起動")
    controller.voice_dialog_status_label = QLabel("音声対話：停止")
    controller.voice_dialog_latency_label = QLabel("遅延　ASR：- / LLM：- / TTS：-")
    controller.voice_dialog_opening_utterance_edit = QLineEdit()
    controller.voice_dialog_opening_utterance_edit.setPlaceholderText(
        "開始時にシステムが話す内容（空欄なら開始時の発話なし）"
    )
    controller.voice_dialog_scheduled_exit_sec_edit = QLineEdit("0")
    controller.voice_dialog_scheduled_exit_sec_edit.setPlaceholderText("開始からの経過秒（0で無効）")
    controller.voice_dialog_scheduled_exit_utterance_edit = QLineEdit()
    controller.voice_dialog_scheduled_exit_utterance_edit.setPlaceholderText(
        "経過後、次のターン検知時に話してから終了（空欄なら無効）"
    )
    controller.voice_dialog_idle_take_turn_sec_edit = QLineEdit("0")
    controller.voice_dialog_idle_take_turn_sec_edit.setPlaceholderText("0で無効")
    controller.voice_dialog_idle_take_turn_sec_edit.setToolTip(
        "ターンロック解除後、最後のユーザー発話検知からこの秒数経過したら"
        "MaAIのターン検知（crossed_up）と同様にターン取得します。"
    )
    controller.voice_dialog_interrupt_check = QCheckBox("ユーザー発話でシステム発話を割り込み停止")
    controller.voice_dialog_interrupt_check.setChecked(True)
    controller.voice_dialog_interrupt_vad_threshold_edit = QLineEdit("0.1")
    controller.voice_dialog_interrupt_pnow_threshold_edit = QLineEdit("0.5")
    controller.voice_dialog_llm_stream_label = QLabel("LLMストリーミング（生成中）")
    controller.voice_dialog_llm_stream_view = QPlainTextEdit()
    controller.voice_dialog_llm_stream_view.setReadOnly(True)
    controller.voice_dialog_llm_stream_view.setPlaceholderText(
        "LLMのストリーミングとTTS分割（次の応答が始まるまで表示を保持）"
    )
    controller.voice_dialog_llm_stream_view.setMinimumHeight(28)
    controller.voice_dialog_llm_stream_view.setMaximumHeight(72)
    # 会話の欄は吹き出し（gui/chat_view.py）。生成中の応答も同じ欄の一番下に出す。
    controller.voice_dialog_history_view = ChatView()

    exec_browse_btn = QPushButton("参照…")
    mdf_browse_btn = QPushButton("参照…")
    reload_mdf_btn = QPushButton("MDFから読込")
    save_mdf_btn = QPushButton("MDFへ保存")
    save_as_mdf_btn = QPushButton("MDFを新規保存…")
    reset_defaults_btn = QPushButton("デフォルトに戻す")
    settings_toggle_btn = QPushButton("MMDAgent-EX起動設定")
    tts_test_toggle_btn = QPushButton("TTSテスト")
    asr_settings_toggle_btn = QPushButton("ASRテスト")
    llm_test_toggle_btn = QPushButton("LLMテスト")
    maai_test_toggle_btn = QPushButton("MaAIテスト")
    obs_settings_toggle_btn = QPushButton("OBS録画設定")
    audio_input_settings_toggle_btn = QPushButton("音声入力の設定")
    audio_output_settings_toggle_btn = QPushButton("音声出力の設定")
    console_toggle_btn = QPushButton("プロセスコンソール")
    engine_manager_toggle_btn = QPushButton("エンジン管理")
    def _new_start_stop_pair() -> tuple[QPushButton, QPushButton]:
        start = QPushButton("起動")
        start.setProperty("role", "primary")  # その画面でいちばん使う操作（docs/ui-guidelines.md）
        return start, QPushButton("停止")

    voicevox_launch_btn, voicevox_stop_btn = _new_start_stop_pair()
    voicevox_core_launch_btn, voicevox_core_stop_btn = _new_start_stop_pair()
    kokoro_realtimetts_launch_btn, kokoro_realtimetts_stop_btn = _new_start_stop_pair()
    qwen_cpp_launch_btn, qwen_cpp_stop_btn = _new_start_stop_pair()
    lfm2_audio_launch_btn, lfm2_audio_stop_btn = _new_start_stop_pair()
    aivis_launch_btn, aivis_stop_btn = _new_start_stop_pair()
    supertonic_launch_btn, supertonic_stop_btn = _new_start_stop_pair()
    irodori_launch_btn, irodori_stop_btn = _new_start_stop_pair()
    tsukasa_launch_btn, tsukasa_stop_btn = _new_start_stop_pair()
    voicevox_exe_browse_btn = QPushButton("VOICEVOX実行ファイルを参照…")
    tts_test_send_btn = QPushButton("TTSテスト")
    tts_test_send_btn.setProperty("role", "primary")
    llm_chat_send_btn = QPushButton("送信")
    llm_chat_send_btn.setProperty("role", "primary")
    llm_chat_reset_btn = QPushButton("履歴リセット")
    llm_s2s_test_open_btn = QPushButton("LLM分割テスト")
    llm_s2s_test_run_btn = QPushButton("分割テストを実行")
    llm_model_lmstudio_refresh_btn = QPushButton("取得")
    llm_model_openai_refresh_btn = QPushButton("取得")
    llm_model_unload_btn = QPushButton("現在以外をアンロード")
    llm_warmup_btn = QPushButton("ウォームアップ")
    template_load_btn = QPushButton("テンプレートJSONを読み込む…")
    template_save_btn = QPushButton("テンプレートJSONを保存…")
    launch_btn = QPushButton("MMDAgent-EXを起動")
    stop_btn = QPushButton("停止")
    voice_dialog_start_btn = QPushButton("音声対話開始")
    voice_dialog_stop_btn = QPushButton("音声対話終了")
    asr_launch_btn, asr_stop_btn = _new_start_stop_pair()
    asr_clear_btn = QPushButton("認識ログをクリア")
    asr_whisper_model_path_browse_btn = QPushButton("参照…")
    asr_sherpa_tokens_browse_btn = QPushButton("参照…")
    asr_sherpa_encoder_browse_btn = QPushButton("参照…")
    asr_sherpa_decoder_browse_btn = QPushButton("参照…")
    asr_sherpa_joiner_browse_btn = QPushButton("参照…")
    asr_qwen3_onnx_conv_frontend_browse_btn = QPushButton("参照…")
    asr_qwen3_onnx_encoder_browse_btn = QPushButton("参照…")
    asr_qwen3_onnx_decoder_browse_btn = QPushButton("参照…")
    asr_qwen3_onnx_tokenizer_browse_btn = QPushButton("参照…")
    asr_nemotron_onnx_tokens_browse_btn = QPushButton("参照…")
    asr_nemotron_onnx_encoder_browse_btn = QPushButton("参照…")
    asr_nemotron_onnx_decoder_browse_btn = QPushButton("参照…")
    asr_nemotron_onnx_joiner_browse_btn = QPushButton("参照…")
    asr_smart_turn_model_browse_btn = QPushButton("参照…")
    asr_qwen_custom_model_path_browse_btn = QPushButton("参照…")
    asr_nemotron_python_browse_btn = QPushButton("参照…")
    asr_external_start_script_browse_btn = QPushButton("参照…")

    for btn in (
        voicevox_exe_browse_btn,
        voicevox_launch_btn,
        voicevox_stop_btn,
        voicevox_core_launch_btn,
        voicevox_core_stop_btn,
        kokoro_realtimetts_launch_btn,
        kokoro_realtimetts_stop_btn,
        qwen_cpp_launch_btn,
        qwen_cpp_stop_btn,
        lfm2_audio_launch_btn,
        lfm2_audio_stop_btn,
        aivis_launch_btn,
        aivis_stop_btn,
        supertonic_launch_btn,
        supertonic_stop_btn,
        irodori_launch_btn,
        irodori_stop_btn,
        tsukasa_launch_btn,
        tsukasa_stop_btn,
        tts_test_send_btn,
        llm_s2s_test_open_btn,
        template_load_btn,
        template_save_btn,
        asr_launch_btn,
        asr_stop_btn,
        asr_clear_btn,
        voice_dialog_start_btn,
        voice_dialog_stop_btn,
    ):
        btn.setMaximumWidth(220)

    exec_browse_btn.clicked.connect(controller._browse_exec)
    mdf_browse_btn.clicked.connect(controller._browse_mdf)
    reload_mdf_btn.clicked.connect(controller.load_mdf_settings)
    save_mdf_btn.clicked.connect(controller.save_mdf_settings)
    save_as_mdf_btn.clicked.connect(controller.save_mdf_as)
    reset_defaults_btn.clicked.connect(controller.reset_to_defaults)
    settings_toggle_btn.clicked.connect(controller._open_settings_window)
    tts_test_toggle_btn.clicked.connect(controller._open_tts_test_window)
    asr_settings_toggle_btn.clicked.connect(controller._open_asr_test_window)
    llm_test_toggle_btn.clicked.connect(controller._open_llm_test_window)
    maai_test_toggle_btn.clicked.connect(controller._open_maai_test_window)
    obs_settings_toggle_btn.clicked.connect(controller._open_obs_settings_window)
    audio_input_settings_toggle_btn.clicked.connect(controller._open_audio_input_settings_window)
    audio_output_settings_toggle_btn.clicked.connect(controller._open_audio_output_settings_window)
    console_toggle_btn.clicked.connect(controller._open_console_window)
    engine_manager_toggle_btn.clicked.connect(controller._open_engine_manager_window)
    audio_input_wav_browse_btn.clicked.connect(controller._browse_audio_input_wav_file)
    audio_input_wav_play_pause_btn.clicked.connect(controller._audio_input_wav_play_pause_toggle)
    audio_input_wav_end_btn.clicked.connect(controller._audio_input_wav_end)
    controller.audio_input_wav_path_edit.textChanged.connect(
        lambda _text: controller._sync_wav_module_panel()
    )
    controller.audio_input_wav_local_playback_check.stateChanged.connect(lambda _s: controller._save_app_config())
    controller.audio_input_wav_other_ch_to_mmd_check.stateChanged.connect(lambda _s: controller._save_app_config())
    controller.audio_input_wav_other_ch_to_maai_ch2_check.stateChanged.connect(lambda _s: controller._save_app_config())
    controller.audio_input_wav_aec_check.stateChanged.connect(lambda _s: controller._save_app_config())
    controller.audio_input_wav_aec_ref_combo.currentIndexChanged.connect(lambda _i: controller._save_app_config())
    voicevox_launch_btn.clicked.connect(controller.launch_voicevox)
    voicevox_stop_btn.clicked.connect(controller.stop_voicevox)
    voicevox_core_launch_btn.clicked.connect(controller.launch_voicevox_core)
    voicevox_core_stop_btn.clicked.connect(controller.stop_voicevox_core)
    kokoro_realtimetts_launch_btn.clicked.connect(controller.launch_kokoro_realtimetts)
    kokoro_realtimetts_stop_btn.clicked.connect(controller.stop_kokoro_realtimetts)
    qwen_cpp_launch_btn.clicked.connect(lambda: controller.launch_worker_tts("Qwen3-TTS-CPP"))
    qwen_cpp_stop_btn.clicked.connect(lambda: controller.stop_worker_tts("Qwen3-TTS-CPP"))
    lfm2_audio_launch_btn.clicked.connect(lambda: controller.launch_worker_tts("LFM2.5-Audio"))
    lfm2_audio_stop_btn.clicked.connect(lambda: controller.stop_worker_tts("LFM2.5-Audio"))
    aivis_launch_btn.clicked.connect(controller.launch_aivisspeech)
    aivis_stop_btn.clicked.connect(controller.stop_aivisspeech)
    supertonic_launch_btn.clicked.connect(lambda: controller.launch_worker_tts("Supertonic-3"))
    supertonic_stop_btn.clicked.connect(lambda: controller.stop_worker_tts("Supertonic-3"))
    irodori_launch_btn.clicked.connect(lambda: controller.launch_worker_tts("Irodori-v4"))
    irodori_stop_btn.clicked.connect(lambda: controller.stop_worker_tts("Irodori-v4"))
    tsukasa_launch_btn.clicked.connect(lambda: controller.launch_worker_tts("Tsukasa"))
    tsukasa_stop_btn.clicked.connect(lambda: controller.stop_worker_tts("Tsukasa"))
    voicevox_exe_browse_btn.clicked.connect(controller._browse_voicevox_exe)
    tts_test_send_btn.clicked.connect(controller.send_mmdagent_tts_test)
    llm_chat_send_btn.clicked.connect(controller.send_llm_test_chat)
    llm_chat_reset_btn.clicked.connect(controller.reset_llm_test_history)
    llm_s2s_test_open_btn.clicked.connect(controller._open_llm_s2s_test_window)
    llm_s2s_test_run_btn.clicked.connect(controller.run_llm_s2s_test)
    llm_model_lmstudio_refresh_btn.clicked.connect(controller.refresh_lmstudio_models)
    llm_model_openai_refresh_btn.clicked.connect(controller.refresh_openai_models)
    llm_model_unload_btn.clicked.connect(controller.unload_lmstudio_non_current_models)
    llm_warmup_btn.clicked.connect(controller.run_llm_warmup_now)
    controller.llm_user_input_edit.returnPressed.connect(controller.send_llm_test_chat)
    template_load_btn.clicked.connect(controller._load_tts_template_from_json)
    template_save_btn.clicked.connect(controller._save_tts_template_to_json)
    launch_btn.clicked.connect(controller.launch_mmdagent)
    stop_btn.clicked.connect(controller.stop_mmdagent)
    asr_launch_btn.clicked.connect(controller.launch_asr)
    asr_stop_btn.clicked.connect(controller.stop_asr)
    voice_dialog_start_btn.clicked.connect(controller._start_voice_dialog)
    voice_dialog_stop_btn.clicked.connect(controller._stop_voice_dialog)
    asr_clear_btn.clicked.connect(controller.clear_asr_results)
    asr_whisper_model_path_browse_btn.clicked.connect(controller._browse_asr_whisper_model_path)
    asr_sherpa_tokens_browse_btn.clicked.connect(controller._browse_asr_sherpa_tokens)
    asr_sherpa_encoder_browse_btn.clicked.connect(controller._browse_asr_sherpa_encoder)
    asr_sherpa_decoder_browse_btn.clicked.connect(controller._browse_asr_sherpa_decoder)
    asr_sherpa_joiner_browse_btn.clicked.connect(controller._browse_asr_sherpa_joiner)
    asr_qwen3_onnx_conv_frontend_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_qwen3_onnx_conv_frontend_edit, "Qwen3 conv_frontend.onnx を選択")
    )
    asr_qwen3_onnx_encoder_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_qwen3_onnx_encoder_edit, "Qwen3 encoder.onnx を選択")
    )
    asr_qwen3_onnx_decoder_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_qwen3_onnx_decoder_edit, "Qwen3 decoder.onnx を選択")
    )
    asr_qwen3_onnx_tokenizer_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_qwen3_onnx_tokenizer_edit, "Qwen3 tokenizer を選択", "All Files (*.*)")
    )
    asr_nemotron_onnx_tokens_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_nemotron_onnx_tokens_edit, "Nemotron tokens.txt を選択", "Text (*.txt);;All Files (*.*)")
    )
    asr_nemotron_onnx_encoder_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_nemotron_onnx_encoder_edit, "Nemotron encoder.onnx を選択")
    )
    asr_nemotron_onnx_decoder_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_nemotron_onnx_decoder_edit, "Nemotron decoder.onnx を選択")
    )
    asr_nemotron_onnx_joiner_browse_btn.clicked.connect(
        lambda: controller._browse_asr_model_file(controller.asr_nemotron_onnx_joiner_edit, "Nemotron joiner.onnx を選択")
    )
    asr_smart_turn_model_browse_btn.clicked.connect(controller._browse_asr_smart_turn_model)
    asr_qwen_custom_model_path_browse_btn.clicked.connect(controller._browse_asr_qwen_custom_model_path)
    asr_nemotron_python_browse_btn.clicked.connect(controller._browse_asr_nemotron_python)
    asr_external_start_script_browse_btn.clicked.connect(controller._browse_asr_external_start_script)
    controller.tts_engine_combo.currentTextChanged.connect(controller._on_tts_engine_changed)
    controller.tts_engine_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.voicevox_exe_edit.editingFinished.connect(controller._save_app_config)
    controller.voicevox_core_onnxruntime_edit.editingFinished.connect(controller._save_app_config)
    controller.voicevox_core_openjtalk_dict_edit.editingFinished.connect(controller._save_app_config)
    controller.voicevox_core_vvm_edit.editingFinished.connect(controller._save_app_config)
    controller.voicevox_core_device_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.voicevox_core_cpu_threads_edit.editingFinished.connect(controller._save_app_config)
    controller.voicevox_core_speed_scale_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_host_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_port_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_base_path_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_voice_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_model_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_input_streaming_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.mmd_burst_snd_frames_edit.editingFinished.connect(controller._save_app_config)
    controller.mmd_blink_enable_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.mmd_blink_interval_mean_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.mmd_blink_morph_edit.editingFinished.connect(controller._save_app_config)
    controller.mmd_blink_mode_combo.currentIndexChanged.connect(lambda _i: controller._save_app_config())
    controller.tts_segment_gap_silence_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_output_mode_combo.currentIndexChanged.connect(lambda _i: controller._save_app_config())
    controller.audio_output_tcp_host_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_output_tcp_port_edit.editingFinished.connect(controller._save_app_config)
    for _qwen_cpp_edit in (
        controller.qwen3_tts_cpp_server_exe_edit,
        controller.qwen3_tts_cpp_model_path_edit,
        controller.qwen3_tts_cpp_host_edit,
        controller.qwen3_tts_cpp_port_edit,
        controller.qwen3_tts_cpp_model_id_edit,
        controller.qwen3_tts_cpp_speaker_edit,
        controller.qwen3_tts_cpp_instruct_edit,
        controller.qwen3_tts_cpp_language_edit,
        controller.qwen3_tts_cpp_temperature_edit,
        controller.qwen3_tts_cpp_top_k_edit,
        controller.qwen3_tts_cpp_top_p_edit,
        controller.qwen3_tts_cpp_repetition_penalty_edit,
        controller.qwen3_tts_cpp_seed_edit,
    ):
        _qwen_cpp_edit.editingFinished.connect(controller._save_app_config)
    controller.qwen3_tts_cpp_do_sample_check.stateChanged.connect(
        lambda _state: controller._save_app_config()
    )
    # Engine-specific TTS fields are persisted on editing completion just like
    # the existing VOICEVOX/Qwen fields above.
    for _tts_engine_edit in (
        controller.aivisspeech_host_edit,
        controller.aivisspeech_port_edit,
        controller.aivisspeech_exe_edit,
        controller.supertonic_host_edit,
        controller.supertonic_port_edit,
        controller.supertonic_model_id_edit,
        controller.supertonic_server_exe_edit,
        controller.supertonic_model_path_edit,
        controller.supertonic_steps_edit,
        controller.lfm2_audio_host_edit,
        controller.lfm2_audio_port_edit,
        controller.lfm2_audio_model_id_edit,
        controller.lfm2_audio_model_path_edit,
        controller.lfm2_audio_server_exe_edit,
        controller.irodori_python_edit,
        controller.irodori_checkpoint_edit,
        controller.irodori_reference_audio_edit,
        controller.irodori_steps_edit,
        controller.irodori_speechcpp_steps_edit,
        controller.irodori_seed_edit,
        controller.irodori_audiocpp_host_edit,
        controller.irodori_audiocpp_port_edit,
        controller.tsukasa_root_edit,
        controller.tsukasa_reference_audio_edit,
        controller.tsukasa_steps_edit,
        controller.tsukasa_alpha_edit,
        controller.tsukasa_beta_edit,
    ):
        _tts_engine_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_auth_mode_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.tts_auth_env_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_auth_header_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_auth_prefix_edit.editingFinished.connect(controller._save_app_config)
    controller.tts_template_edit.textChanged.connect(controller._save_app_config)
    controller.tts_test_text_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_backend_combo.currentTextChanged.connect(controller._on_llm_backend_changed)
    controller.llm_backend_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.llm_host_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_model_lmstudio_combo.currentTextChanged.connect(controller._on_llm_model_text_changed)
    controller.llm_model_lmstudio_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.llm_model_lmstudio_combo.activated.connect(lambda _index: controller._on_llm_model_selection_committed())
    llm_model_lmstudio_line_edit = controller.llm_model_lmstudio_combo.lineEdit()
    if llm_model_lmstudio_line_edit is not None:
        llm_model_lmstudio_line_edit.editingFinished.connect(controller._on_llm_model_selection_committed)
    controller.llm_model_openai_combo.currentTextChanged.connect(controller._on_llm_model_text_changed)
    controller.llm_model_openai_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.llm_api_key_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_api_key_env_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_custom_base_url_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_custom_api_key_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_custom_api_key_env_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_model_custom_combo.currentTextChanged.connect(controller._on_llm_model_text_changed)
    controller.llm_model_custom_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.llm_custom_send_temperature_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.llm_custom_send_max_tokens_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.llm_custom_temperature_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_custom_extra_body_thinking_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.llm_system_prompt_edit.textChanged.connect(controller._save_app_config)
    controller.llm_context_len_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_max_tokens_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_tts_stream_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.llm_tts_stream_check.stateChanged.connect(
        lambda _state: controller._sync_llm_test_tts_input_streaming_widgets()
    )
    controller.llm_test_tts_input_streaming_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.llm_split_custom_delimiters_edit.editingFinished.connect(controller._save_app_config)
    controller.llm_split_custom_strip_emoji_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.llm_split_custom_strip_parenthetical_check.stateChanged.connect(
        lambda _state: controller._save_app_config()
    )
    controller.llm_split_custom_strip_standalone_warai_check.stateChanged.connect(
        lambda _state: controller._save_app_config()
    )
    controller.voice_dialog_interrupt_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.voice_dialog_interrupt_vad_threshold_edit.editingFinished.connect(controller._save_app_config)
    controller.voice_dialog_interrupt_pnow_threshold_edit.editingFinished.connect(controller._save_app_config)
    controller.voice_dialog_interrupt_vad_threshold_edit.editingFinished.connect(controller._sync_maai_plot_thresholds)
    controller.voice_dialog_interrupt_pnow_threshold_edit.editingFinished.connect(controller._sync_maai_plot_thresholds)
    controller.voice_dialog_opening_utterance_edit.editingFinished.connect(controller._save_app_config)
    controller.voice_dialog_scheduled_exit_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.voice_dialog_scheduled_exit_utterance_edit.editingFinished.connect(controller._save_app_config)
    controller.voice_dialog_idle_take_turn_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_engine_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_engine_combo.currentTextChanged.connect(controller._on_asr_engine_changed)
    controller.asr_engine_combo_settings.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_engine_combo_settings.currentTextChanged.connect(controller._on_asr_engine_changed)
    controller.asr_vad_backend_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_turn_backend_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_device_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_compute_type_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_whisper_model_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_whisper_hf_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_whisper_model_path_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_sherpa_tokens_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_sherpa_encoder_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_sherpa_decoder_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_sherpa_joiner_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_sherpa_num_threads_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_sherpa_provider_combo.currentTextChanged.connect(
        lambda _text: controller._save_app_config()
    )
    controller.asr_qwen3_onnx_conv_frontend_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen3_onnx_encoder_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen3_onnx_decoder_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen3_onnx_tokenizer_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen3_onnx_language_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen3_onnx_provider_combo.currentTextChanged.connect(
        lambda _text: controller._save_app_config()
    )
    controller.asr_qwen3_onnx_num_threads_edit.editingFinished.connect(controller._save_app_config)
    for _sherpa_offline_edit in (
        controller.asr_parakeet_onnx_dir_edit,
        controller.asr_parakeet_onnx_num_threads_edit,
        controller.asr_sensevoice_onnx_dir_edit,
        controller.asr_sensevoice_onnx_num_threads_edit,
        controller.asr_sensevoice_onnx_language_edit,
        controller.asr_funasr_nano_onnx_dir_edit,
        controller.asr_funasr_nano_onnx_num_threads_edit,
        controller.asr_funasr_nano_onnx_language_edit,
        controller.asr_funasr_nano_onnx_max_new_tokens_edit,
        controller.asr_cohere_onnx_dir_edit,
        controller.asr_cohere_onnx_num_threads_edit,
        controller.asr_cohere_onnx_language_edit,
    ):
        _sherpa_offline_edit.editingFinished.connect(controller._save_app_config)
    for _sherpa_offline_combo in (
        controller.asr_parakeet_onnx_provider_combo,
        controller.asr_sensevoice_onnx_provider_combo,
        controller.asr_funasr_nano_onnx_provider_combo,
        controller.asr_cohere_onnx_provider_combo,
    ):
        _sherpa_offline_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    for _sherpa_offline_check in (
        controller.asr_sensevoice_onnx_itn_check,
        controller.asr_funasr_nano_onnx_itn_check,
        controller.asr_cohere_onnx_punct_check,
        controller.asr_cohere_onnx_itn_check,
    ):
        _sherpa_offline_check.stateChanged.connect(lambda _state: controller._save_app_config())
    for _speechcpp_combo in (controller.asr_speechcpp_model_combo, controller.asr_speechcpp_device_combo):
        _speechcpp_combo.currentIndexChanged.connect(lambda _index: controller._save_app_config())
    controller.asr_speechcpp_threads_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_onnx_tokens_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_onnx_encoder_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_onnx_decoder_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_onnx_joiner_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_onnx_language_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_host_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_port_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_external_host_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_external_port_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_external_audio_host_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_external_audio_port_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_external_start_script_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_model_edit.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_qwen_custom_model_path_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_gpu_util_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_max_model_len_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_chunk_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_timeout_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_ws_path_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_session_timeout_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_no_text_flush_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_min_return_audio_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_model_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_chunk_combo.currentTextChanged.connect(
        lambda text: controller._on_nemotron_chunk_combo_changed(text, source="asr")
    )
    controller.asr_nemotron_onnx_chunk_combo.currentTextChanged.connect(
        lambda text: controller._on_nemotron_chunk_combo_changed(text, source="onnx")
    )
    controller.asr_nemotron_finalize_combo.currentTextChanged.connect(
        controller._on_nemotron_finalize_combo_changed
    )
    controller.asr_nemotron_onnx_finalize_combo.currentTextChanged.connect(
        controller._on_nemotron_finalize_combo_changed
    )
    controller.asr_nemotron_onnx_compute_combo.currentTextChanged.connect(
        lambda _text: controller._save_app_config()
    )
    controller.asr_nemotron_window_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_max_symbols_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_lang_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nemotron_device_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_nemotron_python_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_input_mode_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.audio_input_mic_device_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.audio_input_wav_path_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_input_wav_start_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_input_wav_end_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_input_wav_channel_combo.currentIndexChanged.connect(lambda _i: controller._save_app_config())
    controller.audio_input_tcp_host_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_input_tcp_port_edit.editingFinished.connect(controller._save_app_config)
    controller.audio_input_gain_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_qwen_finish_on_vad_end_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.asr_qwen_stop_container_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.asr_qwen_enforce_eager_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.asr_chirp_project_id_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_chirp_location_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_chirp_recognizer_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_chirp_model_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_smart_turn_model_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_vad_on_threshold_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_vad_on_threshold_edit.editingFinished.connect(controller._sync_maai_plot_thresholds)
    controller.asr_vad_on_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_vad_off_threshold_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_vad_off_threshold_edit.editingFinished.connect(controller._sync_maai_plot_thresholds)
    controller.asr_vad_off_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_min_segment_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_pre_roll_buffer_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_nonstream_pseudo_enabled_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.asr_nonstream_pseudo_chunk_sec_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_maai_turn_threshold_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_smart_turn_threshold_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_smart_turn_num_threads_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_torch_cpu_intra_threads_edit.editingFinished.connect(controller._save_app_config)
    controller.asr_maai_turn_threshold_edit.editingFinished.connect(controller._sync_maai_plot_thresholds)
    controller.asr_compare_mode_check.stateChanged.connect(controller._on_asr_compare_mode_changed)
    controller.asr_compare_mode_check.stateChanged.connect(lambda _state: controller._save_app_config())
    controller.asr_compare_top_engine_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_compare_bottom_engine_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    controller.asr_compare_top_engine_combo.currentTextChanged.connect(controller._update_asr_compare_result_labels)
    controller.asr_compare_bottom_engine_combo.currentTextChanged.connect(controller._update_asr_compare_result_labels)

    settings_file_form = QGridLayout()
    settings_file_form.addWidget(QLabel("MMDAgent-EX実行ファイル"), 0, 0)
    settings_file_form.addWidget(controller.exec_edit, 0, 1)
    settings_file_form.addWidget(exec_browse_btn, 0, 2)
    settings_file_form.addWidget(QLabel("MDFファイル"), 1, 0)
    settings_file_form.addWidget(controller.mdf_edit, 1, 1)
    settings_file_form.addWidget(mdf_browse_btn, 1, 2)
    settings_file_box = QGroupBox("ファイル設定")
    settings_file_box.setLayout(settings_file_form)

    settings_mdf_buttons = QHBoxLayout()
    settings_mdf_buttons.addWidget(reload_mdf_btn)
    settings_mdf_buttons.addWidget(save_mdf_btn)
    settings_mdf_buttons.addWidget(save_as_mdf_btn)
    settings_mdf_buttons.addWidget(reset_defaults_btn)
    settings_mdf_buttons.addStretch()
    settings_mdf_box = QGroupBox("MDF操作")
    settings_mdf_box.setLayout(settings_mdf_buttons)

    launch_btn.setProperty("role", "primary")
    mmd_control_buttons = QHBoxLayout()
    mmd_control_buttons.addWidget(launch_btn)
    mmd_control_buttons.addWidget(stop_btn)
    mmd_control_buttons.addStretch()
    # 接続状態は controller._refresh_header_status() が更新する（Web の MMDAgent-EX カードと同じ情報）。
    controller.mmd_status_label = QLabel("プロセス：停止　接続：未接続")
    controller.mmd_status_label.setProperty("role", "status")
    mmd_control_layout = QVBoxLayout()
    mmd_control_layout.addLayout(mmd_control_buttons)
    mmd_control_layout.addWidget(controller.mmd_status_label)
    mmd_control_box = QGroupBox("MMDAgent-EX")
    mmd_control_box.setLayout(mmd_control_layout)

    # 開く画面を「テスト / 設定 / ツール」に分け、等幅のグリッドに並べる（Web と同じ並び）。
    main_plot_open_btn = QPushButton("MaAIプロット")
    main_plot_open_btn.clicked.connect(controller._open_maai_plot_window)
    controller.main_maai_plot_open_btn = main_plot_open_btn
    main_maai_settings_open_btn = QPushButton("MaAIモデル設定")
    # maai_settings_window はこの後で作るので、押した時点で参照する。
    main_maai_settings_open_btn.clicked.connect(
        lambda: (controller.maai_settings_window.show(), controller.maai_settings_window.raise_())
    )
    controller.main_maai_settings_open_btn = main_maai_settings_open_btn
    settings_window_buttons = QVBoxLayout()
    settings_window_buttons.setSpacing(4)
    # (見出し, 列数, ボタン)。右の列をスクロールなしで収めるため、名前の短いグループは 1 行に並べる。
    _settings_rows = (
        ("テスト", 4, (asr_settings_toggle_btn, llm_test_toggle_btn, tts_test_toggle_btn, maai_test_toggle_btn)),
        ("設定", 2, (settings_toggle_btn, audio_input_settings_toggle_btn, audio_output_settings_toggle_btn,
                     main_maai_settings_open_btn, obs_settings_toggle_btn)),
        ("ツール", 3, (console_toggle_btn, engine_manager_toggle_btn, main_plot_open_btn)),
    )
    for _caption, _cols, _buttons in _settings_rows:
        _cap = QLabel(_caption)
        _cap.setProperty("role", "caption")
        settings_window_buttons.addWidget(_cap)
        _group_grid = QGridLayout()
        _group_grid.setHorizontalSpacing(6)
        _group_grid.setVerticalSpacing(5)
        for _i, _btn in enumerate(_buttons):
            _btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            _btn.setMinimumWidth(0)
            _group_grid.addWidget(_btn, _i // _cols, _i % _cols)
        for _c in range(_cols):
            _group_grid.setColumnStretch(_c, 1)
        settings_window_buttons.addLayout(_group_grid)
    settings_window_box = QGroupBox("設定・テスト")
    settings_window_box.setLayout(settings_window_buttons)

    tts_control_form = QFormLayout()
    tts_control_form.addRow(QLabel("TTSエンジン"), controller.tts_engine_combo)
    tts_control_buttons = QHBoxLayout()
    tts_control_buttons.addWidget(voicevox_launch_btn, 0)
    tts_control_buttons.addWidget(voicevox_stop_btn, 0)
    tts_control_buttons.addWidget(voicevox_core_launch_btn, 0)
    tts_control_buttons.addWidget(voicevox_core_stop_btn, 0)
    tts_control_buttons.addWidget(kokoro_realtimetts_launch_btn, 0)
    tts_control_buttons.addWidget(kokoro_realtimetts_stop_btn, 0)
    tts_control_buttons.addWidget(qwen_cpp_launch_btn, 0)
    tts_control_buttons.addWidget(qwen_cpp_stop_btn, 0)
    tts_control_buttons.addWidget(lfm2_audio_launch_btn, 0)
    tts_control_buttons.addWidget(lfm2_audio_stop_btn, 0)
    tts_control_buttons.addWidget(aivis_launch_btn, 0)
    tts_control_buttons.addWidget(aivis_stop_btn, 0)
    tts_control_buttons.addWidget(supertonic_launch_btn, 0)
    tts_control_buttons.addWidget(supertonic_stop_btn, 0)
    tts_control_buttons.addWidget(irodori_launch_btn, 0)
    tts_control_buttons.addWidget(irodori_stop_btn, 0)
    tts_control_buttons.addWidget(tsukasa_launch_btn, 0)
    tts_control_buttons.addWidget(tsukasa_stop_btn, 0)
    tts_control_buttons.addStretch(1)
    tts_control_buttons_row = QWidget()
    tts_control_buttons_row.setLayout(tts_control_buttons)
    tts_control_form.addRow(tts_control_buttons_row)
    tts_control_box = QGroupBox("TTS制御")
    tts_control_box.setLayout(tts_control_form)
    tts_control_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    asr_common_form = QFormLayout()
    asr_common_form.addRow(QLabel("VADバックエンド"), controller.asr_vad_backend_combo)
    asr_common_form.addRow(QLabel("ターン判定バックエンド"), controller.asr_turn_backend_combo)
    asr_whisper_form = QFormLayout()
    asr_sherpa_form = QFormLayout()
    asr_qwen_form = QFormLayout()
    asr_external_form = QFormLayout()
    controller.asr_device_label = QLabel("実行デバイス")
    controller.asr_compute_type_label = QLabel("計算精度")
    controller.asr_whisper_model_label = QLabel("モデルサイズ")
    controller.asr_whisper_hf_label = QLabel("モデル（Hugging Face）")
    controller.asr_whisper_path_label = QLabel("モデルのパス")
    controller.asr_sherpa_tokens_label = QLabel("Tokens")
    controller.asr_sherpa_encoder_label = QLabel("Encoder")
    controller.asr_sherpa_decoder_label = QLabel("Decoder")
    controller.asr_sherpa_joiner_label = QLabel("Joiner")
    controller.asr_qwen3_onnx_conv_frontend_label = QLabel("Conv frontend")
    controller.asr_qwen3_onnx_encoder_label = QLabel("Encoder")
    controller.asr_qwen3_onnx_decoder_label = QLabel("Decoder")
    controller.asr_qwen3_onnx_tokenizer_label = QLabel("Tokenizer")
    controller.asr_qwen3_onnx_language_label = QLabel("言語")
    controller.asr_qwen3_onnx_provider_label = QLabel("実行デバイス")
    controller.asr_qwen3_onnx_num_threads_label = QLabel("スレッド数")
    controller.asr_nemotron_onnx_tokens_label = QLabel("Tokens")
    controller.asr_nemotron_onnx_encoder_label = QLabel("Encoder")
    controller.asr_nemotron_onnx_decoder_label = QLabel("Decoder")
    controller.asr_nemotron_onnx_joiner_label = QLabel("Joiner")
    controller.asr_nemotron_onnx_language_label = QLabel("言語")
    controller.asr_nemotron_onnx_chunk_label = QLabel("コンテキストサイズ（チャンク、ms）")
    controller.asr_nemotron_onnx_finalize_label = QLabel("確定モード")
    controller.asr_qwen_host_label = QLabel("ホスト")
    controller.asr_qwen_port_label = QLabel("ポート")
    controller.asr_external_host_label = QLabel("外部ASRホスト")
    controller.asr_external_port_label = QLabel("外部ASRポート")
    controller.asr_external_audio_host_label = QLabel("外部音声ホスト")
    controller.asr_external_audio_port_label = QLabel("外部音声ポート")
    controller.asr_external_start_script_label = QLabel("外部ASR起動スクリプト")
    controller.asr_qwen_model_label = QLabel("モデル（Hugging Face）")
    controller.asr_qwen_custom_model_path_label = QLabel("カスタムモデルのパス")
    controller.asr_qwen_gpu_util_label = QLabel("GPU使用率")
    controller.asr_qwen_max_model_len_label = QLabel("最大モデル長")
    controller.asr_qwen_chunk_sec_label = QLabel("チャンク長（秒）")
    controller.asr_qwen_timeout_sec_label = QLabel("タイムアウト（秒）")
    controller.asr_qwen_ws_path_label = QLabel("WebSocketパス")
    controller.asr_qwen_session_timeout_label = QLabel("セッションのタイムアウト（秒）")
    controller.asr_qwen_no_text_flush_sec_label = QLabel("無応答で確定するまで（秒）")
    controller.asr_qwen_min_return_audio_sec_label = QLabel("最小返却音声長（秒）")
    controller.asr_qwen_finish_on_vad_end_label = QLabel("VAD終了時にQwen認識を確定")
    controller.asr_qwen_stop_container_label = QLabel("停止時にQwenコンテナも停止")
    controller.asr_qwen_enforce_eager_label = QLabel("vLLMのenforce-eager（WSL向け）")
    controller.asr_nemotron_model_label = QLabel("モデル")
    controller.asr_nemotron_chunk_label = QLabel("コンテキストサイズ（チャンク、ms）")
    controller.asr_nemotron_window_sec_label = QLabel("音声窓（秒）")
    controller.asr_nemotron_max_symbols_label = QLabel("最大シンボル数")
    controller.asr_nemotron_lang_label = QLabel("言語")
    controller.asr_nemotron_device_label = QLabel("実行デバイス")
    controller.asr_nemotron_finalize_label = QLabel("確定モード")
    controller.asr_nemotron_python_label = QLabel("Python実行ファイル")
    controller.asr_chirp_project_id_label = QLabel("ChirpプロジェクトID")
    controller.asr_chirp_location_label = QLabel("Chirpロケーション")
    controller.asr_chirp_recognizer_label = QLabel("Chirp認識子")
    controller.asr_chirp_model_label = QLabel("Chirpモデル")
    asr_whisper_form.addRow(controller.asr_device_label, controller.asr_device_combo)
    asr_whisper_form.addRow(controller.asr_compute_type_label, controller.asr_compute_type_combo)
    asr_whisper_form.addRow(controller.asr_whisper_hf_label, controller.asr_whisper_hf_edit)
    asr_whisper_form.addRow(controller.asr_whisper_model_label, controller.asr_whisper_model_edit)

    asr_whisper_row = QWidget()
    asr_whisper_row_layout = QHBoxLayout()
    asr_whisper_row_layout.setContentsMargins(0, 0, 0, 0)
    asr_whisper_row_layout.addWidget(controller.asr_whisper_model_path_edit, 1)
    asr_whisper_row_layout.addWidget(asr_whisper_model_path_browse_btn, 0)
    asr_whisper_row.setLayout(asr_whisper_row_layout)
    asr_whisper_form.addRow(controller.asr_whisper_path_label, asr_whisper_row)

    asr_sherpa_tokens_row = QWidget()
    asr_sherpa_tokens_layout = QHBoxLayout()
    asr_sherpa_tokens_layout.setContentsMargins(0, 0, 0, 0)
    asr_sherpa_tokens_layout.addWidget(controller.asr_sherpa_tokens_edit, 1)
    asr_sherpa_tokens_layout.addWidget(asr_sherpa_tokens_browse_btn, 0)
    asr_sherpa_tokens_row.setLayout(asr_sherpa_tokens_layout)
    asr_sherpa_form.addRow(controller.asr_sherpa_tokens_label, asr_sherpa_tokens_row)

    asr_sherpa_encoder_row = QWidget()
    asr_sherpa_encoder_layout = QHBoxLayout()
    asr_sherpa_encoder_layout.setContentsMargins(0, 0, 0, 0)
    asr_sherpa_encoder_layout.addWidget(controller.asr_sherpa_encoder_edit, 1)
    asr_sherpa_encoder_layout.addWidget(asr_sherpa_encoder_browse_btn, 0)
    asr_sherpa_encoder_row.setLayout(asr_sherpa_encoder_layout)
    asr_sherpa_form.addRow(controller.asr_sherpa_encoder_label, asr_sherpa_encoder_row)

    asr_sherpa_decoder_row = QWidget()
    asr_sherpa_decoder_layout = QHBoxLayout()
    asr_sherpa_decoder_layout.setContentsMargins(0, 0, 0, 0)
    asr_sherpa_decoder_layout.addWidget(controller.asr_sherpa_decoder_edit, 1)
    asr_sherpa_decoder_layout.addWidget(asr_sherpa_decoder_browse_btn, 0)
    asr_sherpa_decoder_row.setLayout(asr_sherpa_decoder_layout)
    asr_sherpa_form.addRow(controller.asr_sherpa_decoder_label, asr_sherpa_decoder_row)

    asr_sherpa_joiner_row = QWidget()
    asr_sherpa_joiner_layout = QHBoxLayout()
    asr_sherpa_joiner_layout.setContentsMargins(0, 0, 0, 0)
    asr_sherpa_joiner_layout.addWidget(controller.asr_sherpa_joiner_edit, 1)
    asr_sherpa_joiner_layout.addWidget(asr_sherpa_joiner_browse_btn, 0)
    asr_sherpa_joiner_row.setLayout(asr_sherpa_joiner_layout)
    asr_sherpa_form.addRow(controller.asr_sherpa_joiner_label, asr_sherpa_joiner_row)
    asr_sherpa_form.addRow(QLabel("実行デバイス"), controller.asr_sherpa_provider_combo)
    asr_sherpa_form.addRow(QLabel("CPUスレッド数"), controller.asr_sherpa_num_threads_edit)

    def _file_browse_row(edit, browse_btn):
        row = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(browse_btn, 0)
        row.setLayout(layout)
        return row

    asr_qwen3_onnx_form = QFormLayout()
    asr_qwen3_onnx_form.addRow(controller.asr_qwen3_onnx_conv_frontend_label, _file_browse_row(controller.asr_qwen3_onnx_conv_frontend_edit, asr_qwen3_onnx_conv_frontend_browse_btn))
    asr_qwen3_onnx_form.addRow(controller.asr_qwen3_onnx_encoder_label, _file_browse_row(controller.asr_qwen3_onnx_encoder_edit, asr_qwen3_onnx_encoder_browse_btn))
    asr_qwen3_onnx_form.addRow(controller.asr_qwen3_onnx_decoder_label, _file_browse_row(controller.asr_qwen3_onnx_decoder_edit, asr_qwen3_onnx_decoder_browse_btn))
    asr_qwen3_onnx_form.addRow(controller.asr_qwen3_onnx_tokenizer_label, _file_browse_row(controller.asr_qwen3_onnx_tokenizer_edit, asr_qwen3_onnx_tokenizer_browse_btn))
    asr_qwen3_onnx_form.addRow(controller.asr_qwen3_onnx_language_label, controller.asr_qwen3_onnx_language_edit)
    asr_qwen3_onnx_form.addRow(controller.asr_qwen3_onnx_provider_label, controller.asr_qwen3_onnx_provider_combo)
    asr_qwen3_onnx_form.addRow(controller.asr_qwen3_onnx_num_threads_label, controller.asr_qwen3_onnx_num_threads_edit)

    asr_nemotron_onnx_form = QFormLayout()
    asr_nemotron_onnx_form.addRow(controller.asr_nemotron_onnx_tokens_label, _file_browse_row(controller.asr_nemotron_onnx_tokens_edit, asr_nemotron_onnx_tokens_browse_btn))
    asr_nemotron_onnx_form.addRow(controller.asr_nemotron_onnx_encoder_label, _file_browse_row(controller.asr_nemotron_onnx_encoder_edit, asr_nemotron_onnx_encoder_browse_btn))
    asr_nemotron_onnx_form.addRow(controller.asr_nemotron_onnx_decoder_label, _file_browse_row(controller.asr_nemotron_onnx_decoder_edit, asr_nemotron_onnx_decoder_browse_btn))
    asr_nemotron_onnx_form.addRow(controller.asr_nemotron_onnx_joiner_label, _file_browse_row(controller.asr_nemotron_onnx_joiner_edit, asr_nemotron_onnx_joiner_browse_btn))
    asr_nemotron_onnx_form.addRow(controller.asr_nemotron_onnx_language_label, controller.asr_nemotron_onnx_language_edit)
    asr_nemotron_onnx_form.addRow(
        controller.asr_nemotron_onnx_chunk_label, controller.asr_nemotron_onnx_chunk_combo
    )
    asr_nemotron_onnx_form.addRow("実行モード", controller.asr_nemotron_onnx_compute_combo)
    asr_nemotron_onnx_form.addRow(
        controller.asr_nemotron_onnx_finalize_label, controller.asr_nemotron_onnx_finalize_combo
    )

    asr_qwen_form.addRow(controller.asr_qwen_host_label, controller.asr_qwen_host_edit)
    asr_qwen_form.addRow(controller.asr_qwen_port_label, controller.asr_qwen_port_edit)
    asr_qwen_form.addRow(controller.asr_qwen_model_label, controller.asr_qwen_model_edit)
    asr_qwen_custom_model_path_row = QWidget()
    asr_qwen_custom_model_path_layout = QHBoxLayout()
    asr_qwen_custom_model_path_layout.setContentsMargins(0, 0, 0, 0)
    asr_qwen_custom_model_path_layout.addWidget(controller.asr_qwen_custom_model_path_edit, 1)
    asr_qwen_custom_model_path_layout.addWidget(asr_qwen_custom_model_path_browse_btn, 0)
    asr_qwen_custom_model_path_row.setLayout(asr_qwen_custom_model_path_layout)
    asr_qwen_form.addRow(controller.asr_qwen_custom_model_path_label, asr_qwen_custom_model_path_row)
    asr_qwen_form.addRow(controller.asr_qwen_gpu_util_label, controller.asr_qwen_gpu_util_edit)
    asr_qwen_form.addRow(controller.asr_qwen_max_model_len_label, controller.asr_qwen_max_model_len_edit)
    asr_qwen_form.addRow(controller.asr_qwen_chunk_sec_label, controller.asr_qwen_chunk_sec_edit)
    asr_qwen_form.addRow(controller.asr_qwen_timeout_sec_label, controller.asr_qwen_timeout_sec_edit)
    asr_qwen_form.addRow(controller.asr_qwen_ws_path_label, controller.asr_qwen_ws_path_edit)
    asr_qwen_form.addRow(controller.asr_qwen_session_timeout_label, controller.asr_qwen_session_timeout_edit)
    asr_qwen_form.addRow(controller.asr_qwen_no_text_flush_sec_label, controller.asr_qwen_no_text_flush_sec_edit)
    asr_qwen_form.addRow(controller.asr_qwen_min_return_audio_sec_label, controller.asr_qwen_min_return_audio_sec_edit)
    asr_qwen_form.addRow(controller.asr_qwen_finish_on_vad_end_label, controller.asr_qwen_finish_on_vad_end_check)
    asr_qwen_form.addRow(controller.asr_qwen_stop_container_label, controller.asr_qwen_stop_container_check)
    asr_qwen_form.addRow(controller.asr_qwen_enforce_eager_label, controller.asr_qwen_enforce_eager_check)

    asr_nemotron_form = QFormLayout()
    asr_nemotron_form.addRow(controller.asr_nemotron_model_label, controller.asr_nemotron_model_edit)
    asr_nemotron_form.addRow(controller.asr_nemotron_chunk_label, controller.asr_nemotron_chunk_combo)
    asr_nemotron_form.addRow(controller.asr_nemotron_finalize_label, controller.asr_nemotron_finalize_combo)
    asr_nemotron_form.addRow(controller.asr_nemotron_window_sec_label, controller.asr_nemotron_window_sec_edit)
    asr_nemotron_form.addRow(controller.asr_nemotron_max_symbols_label, controller.asr_nemotron_max_symbols_edit)
    asr_nemotron_form.addRow(controller.asr_nemotron_lang_label, controller.asr_nemotron_lang_edit)
    asr_nemotron_form.addRow(controller.asr_nemotron_device_label, controller.asr_nemotron_device_combo)
    asr_nemotron_python_row = QWidget()
    asr_nemotron_python_layout = QHBoxLayout()
    asr_nemotron_python_layout.setContentsMargins(0, 0, 0, 0)
    asr_nemotron_python_layout.addWidget(controller.asr_nemotron_python_edit, 1)
    asr_nemotron_python_layout.addWidget(asr_nemotron_python_browse_btn, 0)
    asr_nemotron_python_row.setLayout(asr_nemotron_python_layout)
    controller.asr_nemotron_python_row = asr_nemotron_python_row
    asr_nemotron_form.addRow(controller.asr_nemotron_python_label, asr_nemotron_python_row)

    asr_external_form.addRow(controller.asr_external_host_label, controller.asr_external_host_edit)
    asr_external_form.addRow(controller.asr_external_port_label, controller.asr_external_port_edit)
    asr_external_form.addRow(controller.asr_external_audio_host_label, controller.asr_external_audio_host_edit)
    asr_external_form.addRow(controller.asr_external_audio_port_label, controller.asr_external_audio_port_edit)
    asr_external_start_script_row = QWidget()
    asr_external_start_script_layout = QHBoxLayout()
    asr_external_start_script_layout.setContentsMargins(0, 0, 0, 0)
    asr_external_start_script_layout.addWidget(controller.asr_external_start_script_edit, 1)
    asr_external_start_script_layout.addWidget(asr_external_start_script_browse_btn, 0)
    asr_external_start_script_row.setLayout(asr_external_start_script_layout)
    asr_external_form.addRow(controller.asr_external_start_script_label, asr_external_start_script_row)
    # Chirp3 は内部実装のみ残し、GUI からは非表示にする。

    asr_smart_turn_row = QWidget()
    asr_smart_turn_layout = QHBoxLayout()
    asr_smart_turn_layout.setContentsMargins(0, 0, 0, 0)
    asr_smart_turn_layout.addWidget(controller.asr_smart_turn_model_edit, 1)
    asr_smart_turn_layout.addWidget(asr_smart_turn_model_browse_btn, 0)
    asr_smart_turn_row.setLayout(asr_smart_turn_layout)
    asr_common_form.addRow(QLabel("疑似ストリーミング"), controller.asr_nonstream_pseudo_enabled_check)
    asr_common_detail_form = QFormLayout()
    asr_common_detail_form.addRow(QLabel("SmartTurn ONNXモデル"), asr_smart_turn_row)
    asr_common_detail_form.addRow(QLabel("VAD開始しきい値"), controller.asr_vad_on_threshold_edit)
    asr_common_detail_form.addRow(QLabel("VAD開始継続時間（秒）"), controller.asr_vad_on_sec_edit)
    asr_common_detail_form.addRow(QLabel("VAD終了しきい値"), controller.asr_vad_off_threshold_edit)
    asr_common_detail_form.addRow(QLabel("VAD終了継続時間（秒）"), controller.asr_vad_off_sec_edit)
    asr_common_detail_form.addRow(QLabel("最小発話区間（秒）"), controller.asr_min_segment_sec_edit)
    asr_common_detail_form.addRow(QLabel("先読みバッファ（秒）"), controller.asr_pre_roll_buffer_sec_edit)
    asr_common_detail_form.addRow(QLabel("疑似チャンク長（秒）"), controller.asr_nonstream_pseudo_chunk_sec_edit)
    asr_common_detail_form.addRow(QLabel("SmartTurnしきい値"), controller.asr_smart_turn_threshold_edit)
    asr_common_detail_form.addRow(QLabel("SmartTurnスレッド数"), controller.asr_smart_turn_num_threads_edit)
    asr_common_detail_form.addRow(QLabel("Torch演算内スレッド数"), controller.asr_torch_cpu_intra_threads_edit)
    asr_common_detail = CollapsibleSection("詳細設定（しきい値・秒数・スレッド数）")
    asr_common_detail.setContentLayout(asr_common_detail_form)
    controller.asr_common_detail_section = asr_common_detail
    asr_common_layout = QVBoxLayout()
    asr_common_layout.addLayout(asr_common_form)
    asr_common_layout.addWidget(asr_common_detail)
    asr_common_box = QGroupBox("共通設定")
    asr_common_box.setLayout(asr_common_layout)
    asr_common_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.asr_common_settings_box = asr_common_box
    asr_whisper_page = QWidget()
    asr_whisper_page.setLayout(asr_whisper_form)
    controller.asr_whisper_settings_box = asr_whisper_page

    asr_sherpa_page = QWidget()
    asr_sherpa_page.setLayout(asr_sherpa_form)
    controller.asr_sherpa_settings_box = asr_sherpa_page

    asr_qwen_page = QWidget()
    asr_qwen_page.setLayout(asr_qwen_form)
    controller.asr_qwen_settings_box = asr_qwen_page

    asr_nemotron_page = QWidget()
    asr_nemotron_page.setLayout(asr_nemotron_form)
    controller.asr_nemotron_settings_box = asr_nemotron_page

    asr_external_page = QWidget()
    asr_external_page.setLayout(asr_external_form)
    controller.asr_external_settings_box = asr_external_page

    asr_qwen3_onnx_page = QWidget()
    asr_qwen3_onnx_page.setLayout(asr_qwen3_onnx_form)
    controller.asr_qwen3_onnx_settings_box = asr_qwen3_onnx_page

    asr_nemotron_onnx_page = QWidget()
    asr_nemotron_onnx_page.setLayout(asr_nemotron_onnx_form)
    controller.asr_nemotron_onnx_settings_box = asr_nemotron_onnx_page

    # sherpa-onnx OfflineRecognizer 系の 4 ページ。項目構成がほぼ同じなので組み立ても共通化する
    # （個別に書くと 4 倍の near-duplicate になるだけで、読むときの手掛かりも増えない）。
    def _add_sherpa_offline_page(dir_edit, provider_combo, threads_edit, extra_rows):
        form = QFormLayout()
        form.addRow(QLabel("モデルディレクトリ"), dir_edit)
        form.addRow(QLabel("実行デバイス"), provider_combo)
        form.addRow(QLabel("スレッド数"), threads_edit)
        for label, widget in extra_rows:
            form.addRow(QLabel(label), widget)
        page = QWidget()
        page.setLayout(form)
        return page

    asr_parakeet_onnx_page = _add_sherpa_offline_page(
        controller.asr_parakeet_onnx_dir_edit,
        controller.asr_parakeet_onnx_provider_combo,
        controller.asr_parakeet_onnx_num_threads_edit,
        [],
    )
    asr_sensevoice_onnx_page = _add_sherpa_offline_page(
        controller.asr_sensevoice_onnx_dir_edit,
        controller.asr_sensevoice_onnx_provider_combo,
        controller.asr_sensevoice_onnx_num_threads_edit,
        [("言語", controller.asr_sensevoice_onnx_language_edit),
         ("ITN（句読点・数字）", controller.asr_sensevoice_onnx_itn_check)],
    )
    asr_funasr_nano_onnx_page = _add_sherpa_offline_page(
        controller.asr_funasr_nano_onnx_dir_edit,
        controller.asr_funasr_nano_onnx_provider_combo,
        controller.asr_funasr_nano_onnx_num_threads_edit,
        [("言語", controller.asr_funasr_nano_onnx_language_edit),
         ("ITN（句読点・数字）", controller.asr_funasr_nano_onnx_itn_check),
         ("最大生成トークン数", controller.asr_funasr_nano_onnx_max_new_tokens_edit)],
    )
    asr_cohere_onnx_page = _add_sherpa_offline_page(
        controller.asr_cohere_onnx_dir_edit,
        controller.asr_cohere_onnx_provider_combo,
        controller.asr_cohere_onnx_num_threads_edit,
        [("言語", controller.asr_cohere_onnx_language_edit),
         ("句読点", controller.asr_cohere_onnx_punct_check),
         ("ITN（数字表記）", controller.asr_cohere_onnx_itn_check)],
    )
    controller.asr_parakeet_onnx_settings_box = asr_parakeet_onnx_page
    asr_speechcpp_form = QFormLayout()
    asr_speechcpp_form.addRow(QLabel("モデル"), controller.asr_speechcpp_model_combo)
    asr_speechcpp_form.addRow(QLabel("実行デバイス"), controller.asr_speechcpp_device_combo)
    asr_speechcpp_form.addRow(QLabel("スレッド数"), controller.asr_speechcpp_threads_edit)
    asr_speechcpp_page = QWidget()
    asr_speechcpp_page.setLayout(asr_speechcpp_form)
    controller.asr_speechcpp_settings_box = asr_speechcpp_page
    controller.asr_sensevoice_onnx_settings_box = asr_sensevoice_onnx_page
    controller.asr_funasr_nano_onnx_settings_box = asr_funasr_nano_onnx_page
    controller.asr_cohere_onnx_settings_box = asr_cohere_onnx_page

    asr_engine_stack = QStackedWidget()
    # エンジン名 → ページの対応をここで 1 度だけ決める。以前は controller 側に
    # setCurrentIndex(0..6) の数字が直接書かれていて、ページを足すたびに両方を
    # 突き合わせる必要があった。
    asr_engine_pages = [
        ("faster-whisper", asr_whisper_page),
        ("sherpa-onnx", asr_sherpa_page),
        ("qwen3-asr", asr_qwen_page),
        ("nemotron-asr", asr_nemotron_page),
        ("external-asr-tcp", asr_external_page),
        ("qwen3-asr-onnx", asr_qwen3_onnx_page),
        ("nemotron-onnx", asr_nemotron_onnx_page),
        ("parakeet-onnx", asr_parakeet_onnx_page),
        ("sensevoice-onnx", asr_sensevoice_onnx_page),
        ("funasr-nano-onnx", asr_funasr_nano_onnx_page),
        ("cohere-asr-onnx", asr_cohere_onnx_page),
        ("speechcpp-asr", asr_speechcpp_page),
    ]
    for _name, _page in asr_engine_pages:
        asr_engine_stack.addWidget(_page)
    controller.asr_engine_page_index = {name: i for i, (name, _) in enumerate(asr_engine_pages)}
    asr_engine_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.asr_engine_stack = asr_engine_stack

    asr_engine_layout = QVBoxLayout()
    asr_engine_layout.setContentsMargins(0, 0, 0, 0)
    asr_engine_layout.setSpacing(0)
    asr_engine_layout.addWidget(asr_engine_stack)
    asr_engine_box = QGroupBox("エンジン設定")
    asr_engine_box.setLayout(asr_engine_layout)
    asr_engine_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.asr_engine_settings_box = asr_engine_box
    asr_settings_layout = QVBoxLayout()
    asr_settings_layout.setContentsMargins(0, 0, 0, 0)
    asr_settings_layout.setSpacing(8)
    asr_settings_layout.addWidget(asr_common_box)
    asr_settings_layout.addWidget(asr_engine_box)
    asr_box = QWidget()
    asr_box.setLayout(asr_settings_layout)
    asr_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.asr_box = asr_box

    controller.asr_whisper_row = asr_whisper_row
    controller.asr_sherpa_tokens_row = asr_sherpa_tokens_row
    controller.asr_sherpa_encoder_row = asr_sherpa_encoder_row
    controller.asr_sherpa_decoder_row = asr_sherpa_decoder_row
    controller.asr_sherpa_joiner_row = asr_sherpa_joiner_row

    asr_runtime_form = QFormLayout()
    controller.asr_runtime_form = asr_runtime_form
    controller.asr_runtime_engine_label = QLabel("ASRエンジン")
    asr_runtime_form.addRow(controller.asr_runtime_engine_label, controller.asr_engine_combo)
    asr_button_row = QWidget()
    asr_button_layout = QHBoxLayout()
    asr_button_layout.setContentsMargins(0, 0, 0, 0)
    asr_button_layout.addWidget(asr_launch_btn, 0)
    asr_button_layout.addWidget(asr_stop_btn, 0)
    asr_button_layout.addWidget(asr_clear_btn, 0)
    asr_button_layout.addStretch(1)
    asr_button_row.setLayout(asr_button_layout)
    asr_runtime_form.addRow(asr_button_row)
    controller.asr_compare_mode_label = QLabel("比較モード")
    controller.asr_runtime_result_label = QLabel("認識結果")
    asr_runtime_form.addRow(controller.asr_compare_mode_label, controller.asr_compare_mode_check)
    asr_compare_form = QFormLayout()
    controller.asr_compare_top_engine_label = QLabel("上段")
    controller.asr_compare_bottom_engine_label = QLabel("下段")
    controller.asr_compare_top_result_label = QLabel("結果（上段）")
    controller.asr_compare_bottom_result_label = QLabel("結果（下段）")
    asr_compare_form.addRow(controller.asr_compare_top_engine_label, controller.asr_compare_top_engine_combo)
    asr_compare_form.addRow(controller.asr_compare_bottom_engine_label, controller.asr_compare_bottom_engine_combo)
    asr_compare_form.addRow(controller.asr_compare_top_result_label, controller.asr_compare_top_result_edit)
    asr_compare_form.addRow(controller.asr_compare_bottom_result_label, controller.asr_compare_bottom_result_edit)
    asr_compare_panel = QWidget()
    asr_compare_panel.setLayout(asr_compare_form)
    controller.asr_compare_panel = asr_compare_panel
    asr_runtime_form.addRow(asr_compare_panel)
    asr_runtime_form.addRow(controller.asr_runtime_result_label, controller.asr_result_edit)
    asr_runtime_box = QGroupBox("ASR制御")
    asr_runtime_box.setLayout(asr_runtime_form)
    asr_runtime_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    controller.asr_runtime_box = asr_runtime_box

    controller.voicevox_exe_label = QLabel("実行ファイル")
    controller.voicevox_core_onnxruntime_label = QLabel("ONNX Runtime")
    controller.voicevox_core_openjtalk_dict_label = QLabel("Open JTalk dictionary")
    controller.voicevox_core_vvm_label = QLabel("VVM")
    controller.voicevox_core_device_label = QLabel("実行デバイス")
    tts_settings_layout = QVBoxLayout()
    controller.tts_host_label = QLabel("TTSサーバーホスト")
    controller.tts_port_label = QLabel("TTSサーバーポート")
    controller.tts_base_path_label = QLabel("ベースパス")
    controller.tts_voice_label = QLabel("話者／音声")
    controller.tts_model_label = QLabel("モデル")
    controller.tts_input_streaming_label = QLabel("LLM→TTSストリーミング")
    controller.tts_segment_gap_silence_sec_label = QLabel("セグメント間の無音（秒）")
    controller.tts_auth_mode_label = QLabel("認証方式")
    controller.tts_auth_env_label = QLabel("認証情報の環境変数")
    controller.tts_auth_header_label = QLabel("認証ヘッダー")
    controller.tts_auth_prefix_label = QLabel("認証プレフィックス")
    controller.qwen3_tts_cpp_server_exe_label = QLabel("audio.cppサーバー実行ファイル")
    controller.qwen3_tts_cpp_model_path_label = QLabel("GGUFモデル")
    controller.qwen3_tts_cpp_host_label = QLabel("サーバーホスト")
    controller.qwen3_tts_cpp_port_label = QLabel("サーバーポート")
    controller.qwen3_tts_cpp_model_id_label = QLabel("モデルID")
    controller.qwen3_tts_cpp_speaker_label = QLabel("話者")
    controller.qwen3_tts_cpp_instruct_label = QLabel("指示")
    controller.qwen3_tts_cpp_language_label = QLabel("言語")
    controller.qwen3_tts_cpp_do_sample_label = QLabel("サンプリング")
    controller.qwen3_tts_cpp_temperature_label = QLabel("温度")
    controller.qwen3_tts_cpp_top_k_label = QLabel("Top K")
    controller.qwen3_tts_cpp_top_p_label = QLabel("Top P")
    controller.qwen3_tts_cpp_repetition_penalty_label = QLabel("反復ペナルティ")
    controller.qwen3_tts_cpp_seed_label = QLabel("シード（空欄＝ランダム）")
    controller.cartesia_tts_model_id_label = QLabel("モデルID")
    controller.cartesia_tts_sample_rate_label = QLabel("サンプルレート（Hz）")
    controller.cartesia_tts_api_key_label = QLabel("APIキー")
    controller.cartesia_tts_api_key_env_label = QLabel("APIキーの環境変数")
    controller.tts_template_label = QLabel("TTSテンプレート（JSON・直接編集）")
    controller.template_load_btn = template_load_btn
    controller.template_save_btn = template_save_btn
    controller.voicevox_exe_browse_btn = voicevox_exe_browse_btn
    controller.voicevox_launch_btn = voicevox_launch_btn
    controller.voicevox_stop_btn = voicevox_stop_btn
    controller.voicevox_core_launch_btn = voicevox_core_launch_btn
    controller.voicevox_core_stop_btn = voicevox_core_stop_btn
    controller.kokoro_realtimetts_launch_btn = kokoro_realtimetts_launch_btn
    controller.kokoro_realtimetts_stop_btn = kokoro_realtimetts_stop_btn
    controller.qwen_cpp_launch_btn = qwen_cpp_launch_btn
    controller.qwen_cpp_stop_btn = qwen_cpp_stop_btn
    controller.lfm2_audio_launch_btn = lfm2_audio_launch_btn
    controller.lfm2_audio_stop_btn = lfm2_audio_stop_btn
    controller.aivis_launch_btn = aivis_launch_btn
    controller.aivis_stop_btn = aivis_stop_btn
    controller.supertonic_launch_btn = supertonic_launch_btn
    controller.supertonic_stop_btn = supertonic_stop_btn
    controller.irodori_launch_btn = irodori_launch_btn
    controller.irodori_stop_btn = irodori_stop_btn
    controller.tsukasa_launch_btn = tsukasa_launch_btn
    controller.tsukasa_stop_btn = tsukasa_stop_btn
    controller.asr_external_start_script_browse_btn = asr_external_start_script_browse_btn
    controller.asr_qwen_custom_model_path_browse_btn = asr_qwen_custom_model_path_browse_btn
    controller.tts_test_send_btn = tts_test_send_btn
    controller.asr_launch_btn = asr_launch_btn
    controller.asr_stop_btn = asr_stop_btn
    controller.asr_clear_btn = asr_clear_btn
    controller.voice_dialog_start_btn = voice_dialog_start_btn
    controller.voice_dialog_stop_btn = voice_dialog_stop_btn
    voice_dialog_stop_btn.setEnabled(False)
    controller.llm_chat_send_btn = llm_chat_send_btn
    controller.llm_chat_reset_btn = llm_chat_reset_btn
    controller.llm_s2s_test_open_btn = llm_s2s_test_open_btn
    controller.llm_s2s_test_run_btn = llm_s2s_test_run_btn
    controller.llm_model_lmstudio_refresh_btn = llm_model_lmstudio_refresh_btn
    controller.llm_model_openai_refresh_btn = llm_model_openai_refresh_btn
    controller.llm_model_unload_btn = llm_model_unload_btn
    controller.llm_warmup_btn = llm_warmup_btn
    voicevox_exe_row = QWidget()
    voicevox_exe_row_layout = QHBoxLayout()
    voicevox_exe_row_layout.setContentsMargins(0, 0, 0, 0)
    voicevox_exe_row_layout.addWidget(controller.voicevox_exe_edit, 1)
    voicevox_exe_row_layout.addWidget(controller.voicevox_exe_browse_btn, 0)
    voicevox_exe_row.setLayout(voicevox_exe_row_layout)

    voicevox_exe_form = QFormLayout()
    voicevox_exe_form.addRow(controller.voicevox_exe_label, voicevox_exe_row)
    voicevox_exe_box = QGroupBox("VOICEVOX実行ファイル")
    voicevox_exe_box.setLayout(voicevox_exe_form)
    voicevox_exe_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.voicevox_exe_box = voicevox_exe_box

    tts_exe_layout = QVBoxLayout()
    tts_exe_layout.setContentsMargins(0, 0, 0, 0)
    tts_exe_layout.setSpacing(8)
    tts_exe_layout.addWidget(voicevox_exe_box)
    tts_exe_container = QWidget()
    tts_exe_container.setLayout(tts_exe_layout)
    tts_exe_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.tts_exe_box = tts_exe_container

    tts_connection_form = QFormLayout()
    tts_connection_form.addRow(controller.tts_host_label, controller.tts_host_edit)
    tts_connection_form.addRow(controller.tts_port_label, controller.tts_port_edit)
    tts_connection_form.addRow(controller.tts_base_path_label, controller.tts_base_path_edit)
    tts_connection_box = QGroupBox("TTS接続設定")
    tts_connection_box.setLayout(tts_connection_form)
    tts_connection_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.tts_connection_box = tts_connection_box

    tts_model_form = QFormLayout()
    tts_model_form.addRow(controller.tts_model_label, controller.tts_model_edit)
    tts_model_box = QGroupBox("モデル設定（対応エンジンのみ）")
    tts_model_box.setLayout(tts_model_form)
    tts_model_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.tts_model_box = tts_model_box

    qwen_cpp_settings_form = QFormLayout()
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_server_exe_label, controller.qwen3_tts_cpp_server_exe_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_model_path_label, controller.qwen3_tts_cpp_model_path_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_host_label, controller.qwen3_tts_cpp_host_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_port_label, controller.qwen3_tts_cpp_port_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_model_id_label, controller.qwen3_tts_cpp_model_id_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_speaker_label, controller.qwen3_tts_cpp_speaker_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_instruct_label, controller.qwen3_tts_cpp_instruct_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_language_label, controller.qwen3_tts_cpp_language_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_do_sample_label, controller.qwen3_tts_cpp_do_sample_check
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_temperature_label, controller.qwen3_tts_cpp_temperature_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_top_k_label, controller.qwen3_tts_cpp_top_k_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_top_p_label, controller.qwen3_tts_cpp_top_p_edit
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_repetition_penalty_label,
        controller.qwen3_tts_cpp_repetition_penalty_edit,
    )
    qwen_cpp_settings_form.addRow(
        controller.qwen3_tts_cpp_seed_label, controller.qwen3_tts_cpp_seed_edit
    )
    qwen_cpp_settings_box = QGroupBox("Qwen3-TTS C++設定（audio.cpp常駐サーバー）")
    qwen_cpp_settings_box.setLayout(qwen_cpp_settings_form)
    qwen_cpp_settings_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.qwen3_tts_cpp_settings_box = qwen_cpp_settings_box

    lfm2_audio_form = QFormLayout()
    lfm2_audio_form.addRow("サーバーホスト", controller.lfm2_audio_host_edit)
    lfm2_audio_form.addRow("サーバーポート", controller.lfm2_audio_port_edit)
    lfm2_audio_form.addRow("モデルID", controller.lfm2_audio_model_id_edit)
    lfm2_audio_form.addRow("モデルフォルダー", controller.lfm2_audio_model_path_edit)
    lfm2_audio_form.addRow("audio.cppサーバー実行ファイル", controller.lfm2_audio_server_exe_edit)
    lfm2_audio_box = QGroupBox("LFM2.5-Audio設定（audio.cppサーバー）")
    lfm2_audio_box.setLayout(lfm2_audio_form)
    lfm2_audio_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.lfm2_audio_settings_box = lfm2_audio_box

    voicevox_core_settings_form = QFormLayout()
    voicevox_core_onnxruntime_row = QHBoxLayout()
    voicevox_core_onnxruntime_row.addWidget(controller.voicevox_core_onnxruntime_edit)
    voicevox_core_onnxruntime_row.addWidget(controller.voicevox_core_onnxruntime_browse_btn)
    voicevox_core_settings_form.addRow(
        controller.voicevox_core_onnxruntime_label, voicevox_core_onnxruntime_row
    )
    voicevox_core_openjtalk_dict_row = QHBoxLayout()
    voicevox_core_openjtalk_dict_row.addWidget(controller.voicevox_core_openjtalk_dict_edit)
    voicevox_core_openjtalk_dict_row.addWidget(controller.voicevox_core_openjtalk_dict_browse_btn)
    voicevox_core_settings_form.addRow(
        controller.voicevox_core_openjtalk_dict_label, voicevox_core_openjtalk_dict_row
    )
    voicevox_core_vvm_row = QHBoxLayout()
    voicevox_core_vvm_row.addWidget(controller.voicevox_core_vvm_edit)
    voicevox_core_vvm_row.addWidget(controller.voicevox_core_vvm_browse_btn)
    voicevox_core_settings_form.addRow(controller.voicevox_core_vvm_label, voicevox_core_vvm_row)
    voicevox_core_settings_form.addRow(controller.voicevox_core_device_label, controller.voicevox_core_device_combo)
    voicevox_core_settings_form.addRow(QLabel("CPUスレッド数"), controller.voicevox_core_cpu_threads_edit)
    voicevox_core_settings_form.addRow(
        QLabel("話速（1.00が標準）"), controller.voicevox_core_speed_scale_edit
    )
    voicevox_core_settings_box = QGroupBox("VOICEVOX-Core設定")
    voicevox_core_settings_box.setLayout(voicevox_core_settings_form)
    voicevox_core_settings_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.voicevox_core_settings_box = voicevox_core_settings_box

    cartesia_settings_form = QFormLayout()
    cartesia_settings_form.addRow(controller.cartesia_tts_model_id_label, controller.cartesia_tts_model_id_edit)
    cartesia_settings_form.addRow(controller.cartesia_tts_sample_rate_label, controller.cartesia_tts_sample_rate_edit)
    cartesia_settings_form.addRow(controller.cartesia_tts_api_key_label, controller.cartesia_tts_api_key_edit)
    cartesia_settings_form.addRow(controller.cartesia_tts_api_key_env_label, controller.cartesia_tts_api_key_env_edit)
    cartesia_tts_settings_box = QGroupBox("Cartesia Realtime設定")
    cartesia_tts_settings_box.setLayout(cartesia_settings_form)
    cartesia_tts_settings_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.cartesia_tts_settings_box = cartesia_tts_settings_box

    aivis_form = QFormLayout()
    aivisspeech_exe_row = QHBoxLayout()
    aivisspeech_exe_row.addWidget(controller.aivisspeech_exe_edit)
    aivisspeech_exe_row.addWidget(controller.aivisspeech_exe_browse_btn)
    aivis_form.addRow("エンジン実行ファイル", aivisspeech_exe_row)
    aivis_form.addRow("ホスト", controller.aivisspeech_host_edit)
    aivis_form.addRow("ポート", controller.aivisspeech_port_edit)
    aivis_box = QGroupBox("AivisSpeechエンジン設定")
    aivis_box.setLayout(aivis_form)
    controller.aivisspeech_settings_box = aivis_box

    supertonic_form = QFormLayout()
    supertonic_form.addRow("サーバーホスト", controller.supertonic_host_edit)
    supertonic_form.addRow("サーバーポート", controller.supertonic_port_edit)
    supertonic_form.addRow("モデルID", controller.supertonic_model_id_edit)
    supertonic_server_exe_row = QHBoxLayout()
    supertonic_server_exe_row.addWidget(controller.supertonic_server_exe_edit)
    supertonic_server_exe_row.addWidget(controller.supertonic_server_exe_browse_btn)
    supertonic_form.addRow("サーバー実行ファイル", supertonic_server_exe_row)
    supertonic_model_path_row = QHBoxLayout()
    supertonic_model_path_row.addWidget(controller.supertonic_model_path_edit)
    supertonic_model_path_row.addWidget(controller.supertonic_model_path_browse_btn)
    supertonic_form.addRow("GGUFモデル", supertonic_model_path_row)
    supertonic_form.addRow("生成ステップ数（既定＝8）", controller.supertonic_steps_edit)
    supertonic_box = QGroupBox("Supertonic 3設定（audio.cppサーバーは事前起動）")
    supertonic_box.setLayout(supertonic_form)
    controller.supertonic_settings_box = supertonic_box

    irodori_form = QFormLayout()
    irodori_form.addRow("動かし方", controller.irodori_backend_combo)
    irodori_form.addRow("モデル（audio.cpp）", controller.irodori_audiocpp_model_combo)
    irodori_form.addRow("サーバーホスト（audio.cpp）", controller.irodori_audiocpp_host_edit)
    irodori_form.addRow("サーバーポート（audio.cpp）", controller.irodori_audiocpp_port_edit)
    irodori_form.addRow("実行デバイス（speech.cpp）", controller.irodori_speechcpp_device_combo)
    irodori_form.addRow("生成ステップ数（speech.cpp・既定＝4）", controller.irodori_speechcpp_steps_edit)
    irodori_form.addRow("seed（speech.cpp / audio.cpp）", controller.irodori_seed_edit)
    irodori_form.addRow("Irodori Python実行ファイル", controller.irodori_python_edit)
    irodori_form.addRow("チェックポイント", controller.irodori_checkpoint_edit)
    irodori_reference_row = QHBoxLayout()
    irodori_reference_row.addWidget(controller.irodori_reference_audio_edit)
    irodori_reference_row.addWidget(controller.irodori_reference_audio_browse_btn)
    irodori_form.addRow("参照WAV（任意・クローニング）", irodori_reference_row)
    irodori_form.addRow("生成ステップ数（PyTorch・既定＝16）", controller.irodori_steps_edit)
    irodori_box = QGroupBox("Irodori v4設定")
    irodori_box.setLayout(irodori_form)
    controller.irodori_settings_box = irodori_box

    tsukasa_form = QFormLayout()
    tsukasa_form.addRow("モデルルート", controller.tsukasa_root_edit)
    tsukasa_reference_row = QHBoxLayout()
    tsukasa_reference_row.addWidget(controller.tsukasa_reference_audio_edit)
    tsukasa_reference_row.addWidget(controller.tsukasa_reference_audio_browse_btn)
    tsukasa_form.addRow("参照WAV（必須・本人の音声のみ）", tsukasa_reference_row)
    tsukasa_form.addRow("生成ステップ数", controller.tsukasa_steps_edit)
    tsukasa_form.addRow("Alpha", controller.tsukasa_alpha_edit)
    tsukasa_form.addRow("Beta", controller.tsukasa_beta_edit)
    tsukasa_box = QGroupBox("Tsukasa設定")
    tsukasa_box.setLayout(tsukasa_form)
    controller.tsukasa_settings_box = tsukasa_box

    template_settings_form = QFormLayout()
    template_auth_open_btn = QPushButton("TEMPLATE認証")
    template_auth_open_btn.setToolTip(
        "TEMPLATEのHTTP認証（テンプレート内の{{input.auth_mode}}など）を編集します。"
    )
    template_auth_open_btn.clicked.connect(controller._open_tts_template_auth_window)
    controller.template_auth_open_btn = template_auth_open_btn
    template_auth_row = QWidget()
    template_auth_row_layout = QHBoxLayout()
    template_auth_row_layout.setContentsMargins(0, 0, 0, 0)
    template_auth_row_layout.addWidget(QLabel("HTTP認証"), 0)
    template_auth_row_layout.addWidget(template_auth_open_btn, 0)
    template_auth_row_layout.addStretch(1)
    template_auth_row.setLayout(template_auth_row_layout)
    template_settings_form.addRow(template_auth_row)
    template_file_buttons = QWidget()
    template_file_buttons_layout = QHBoxLayout()
    template_file_buttons_layout.setContentsMargins(0, 0, 0, 0)
    template_file_buttons_layout.addWidget(template_load_btn, 0)
    template_file_buttons_layout.addWidget(template_save_btn, 0)
    template_file_buttons_layout.addStretch(1)
    template_file_buttons.setLayout(template_file_buttons_layout)
    template_settings_form.addRow(template_file_buttons)
    template_settings_form.addRow(controller.tts_template_label, controller.tts_template_edit)
    template_tts_settings_box = QGroupBox("TEMPLATE（HTTPテンプレート・JSONのみ）")
    template_tts_settings_box.setLayout(template_settings_form)
    template_tts_settings_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    controller.template_tts_settings_box = template_tts_settings_box

    tts_template_auth_form = QFormLayout()
    tts_template_auth_form.addRow(controller.tts_auth_mode_label, controller.tts_auth_mode_combo)
    tts_template_auth_form.addRow(controller.tts_auth_env_label, controller.tts_auth_env_edit)
    tts_template_auth_form.addRow(controller.tts_auth_header_label, controller.tts_auth_header_edit)
    tts_template_auth_form.addRow(controller.tts_auth_prefix_label, controller.tts_auth_prefix_edit)
    tts_template_auth_hint = QLabel(
        "テンプレートJSONの認証設定やヘッダーで{{input.auth_mode}}／{{input.auth_env}}／"
        "{{input.auth_header}} / {{input.auth_prefix}} を参照できます。"
    )
    tts_template_auth_hint.setWordWrap(True)
    tts_template_auth_outer = QVBoxLayout()
    tts_template_auth_outer.addWidget(tts_template_auth_hint)
    tts_template_auth_outer.addLayout(tts_template_auth_form)
    tts_template_auth_outer.addStretch(1)
    controller.tts_template_auth_window = QWidget(controller, Qt.WindowType.Window)
    controller.tts_template_auth_window.setWindowTitle("TEMPLATE認証")
    controller.tts_template_auth_window.resize(480, 260)
    controller.tts_template_auth_window.setLayout(tts_template_auth_outer)

    tts_settings_layout.addWidget(tts_connection_box)
    # engine_install_box は engine_manager_window へ移動
    tts_settings_layout.addWidget(tts_model_box)
    tts_settings_layout.addWidget(voicevox_core_settings_box)
    tts_settings_layout.addWidget(qwen_cpp_settings_box)
    tts_settings_layout.addWidget(lfm2_audio_box)
    tts_settings_layout.addWidget(cartesia_tts_settings_box)
    tts_settings_layout.addWidget(aivis_box)
    tts_settings_layout.addWidget(supertonic_box)
    tts_settings_layout.addWidget(irodori_box)
    tts_settings_layout.addWidget(tsukasa_box)
    tts_settings_layout.addWidget(template_tts_settings_box)
    tts_settings_container = QWidget()
    tts_settings_layout.setContentsMargins(0, 0, 0, 0)
    tts_settings_layout.setSpacing(8)
    tts_settings_container.setLayout(tts_settings_layout)
    tts_settings_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.tts_settings_box = tts_settings_container

    tts_test_form = QFormLayout()
    tts_test_form.addRow(controller.tts_voice_label, controller.tts_voice_edit)
    tts_test_form.addRow(QLabel("テスト発話"), controller.tts_test_text_edit)
    tts_test_form.addRow(QLabel("再生開始まで"), controller.tts_latency_value_label)
    tts_test_action_row = QWidget()
    tts_test_action_layout = QHBoxLayout()
    tts_test_action_layout.setContentsMargins(0, 0, 0, 0)
    tts_test_action_layout.addWidget(tts_test_send_btn, 0)
    tts_test_action_layout.addStretch(1)
    tts_test_action_row.setLayout(tts_test_action_layout)
    tts_test_form.addRow(tts_test_action_row)
    tts_test_box = QGroupBox("TTS直接通信テスト")
    tts_test_box.setLayout(tts_test_form)
    tts_test_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    tabs = QTabWidget()
    for category, specs in controller.FIELD_SPECS.items():
        tab = QWidget()
        tab_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        for key, label, default_value in specs:
            row_label = QLabel(label)
            tooltip = controller._tooltip_for_field(key, default_value)
            row_label.setToolTip(tooltip)
            if key in controller.BOOL_KEYS:
                checkbox = QCheckBox()
                checkbox.setChecked(to_bool(default_value))
                checkbox.stateChanged.connect(lambda state, k=key: controller._on_checkbox_changed(k, state))
                checkbox.setToolTip(tooltip)
                controller.field_widgets[key] = checkbox
                form.addRow(row_label, checkbox)
            else:
                edit = QLineEdit(default_value)
                edit.textEdited.connect(lambda _text, k=key: controller.touched_keys.add(k))
                edit.setToolTip(tooltip)
                controller.field_widgets[key] = edit
                form.addRow(row_label, edit)

        inner.setLayout(form)
        scroll.setWidget(inner)
        tab_layout.addWidget(scroll)
        tab.setLayout(tab_layout)
        tabs.addTab(tab, category)

    controller.settings_tabs = tabs
    controller.settings_window = QWidget(controller, Qt.WindowType.Window)
    controller.settings_window.setWindowTitle("MMDAgent-EX起動設定")
    controller.settings_window.resize(780, 860)
    settings_layout = QVBoxLayout()
    settings_layout.addWidget(settings_file_box)
    settings_layout.addWidget(settings_mdf_box)
    settings_layout.addWidget(controller.settings_tabs)
    controller.settings_window.setLayout(settings_layout)

    audio_input_mode_form = QFormLayout()
    audio_input_mode_form.addRow(QLabel("入力モード"), controller.audio_input_mode_combo)
    audio_input_mode_form.addRow(controller.audio_input_gain_label, controller.audio_input_gain_edit)
    audio_input_mode_box = QGroupBox("音声入力")
    audio_input_mode_box.setLayout(audio_input_mode_form)

    aec_row = QHBoxLayout()
    aec_row.addWidget(controller.audio_input_wav_aec_check)
    aec_row.addWidget(controller.audio_input_wav_aec_ref_combo)
    aec_row.addStretch(1)
    aec_row_widget = QWidget()
    aec_row_widget.setLayout(aec_row)
    audio_input_aec_form = QFormLayout()
    audio_input_aec_form.addRow(QLabel("AEC"), aec_row_widget)
    audio_input_aec_page = QWidget()
    audio_input_aec_page.setLayout(audio_input_aec_form)

    audio_input_mic_form = QFormLayout()
    audio_input_mic_form.addRow(QLabel("マイクデバイス"), controller.audio_input_mic_device_combo)
    audio_input_mic_page = QWidget()
    audio_input_mic_page.setLayout(audio_input_mic_form)

    audio_input_wav_form = QFormLayout()
    audio_input_wav_row = QHBoxLayout()
    audio_input_wav_row.addWidget(controller.audio_input_wav_path_edit, 1)
    audio_input_wav_row.addWidget(audio_input_wav_browse_btn)
    audio_input_wav_widget = QWidget()
    audio_input_wav_widget.setLayout(audio_input_wav_row)
    audio_input_wav_form.addRow(QLabel("WAVファイル"), audio_input_wav_widget)
    controller.audio_input_wav_start_sec_edit.setToolTip(
        "再生開始位置（秒・ファイル先頭から）です。変更後は音声入力ワーカーの再起動が必要です。"
    )
    controller.audio_input_wav_end_sec_edit.setToolTip(
        "再生終了位置（秒・ファイル先頭基準）です。空欄ではファイル末尾まで再生します。"
        "指定範囲の最後まで再生するとMaAIテスト／音声対話を自動終了します（wavモード時）。"
    )
    audio_input_wav_form.addRow(QLabel("開始位置（秒）"), controller.audio_input_wav_start_sec_edit)
    audio_input_wav_form.addRow(QLabel("終了位置（秒、空欄＝最後まで）"), controller.audio_input_wav_end_sec_edit)
    audio_input_wav_form.addRow(QLabel("入力チャンネル"), controller.audio_input_wav_channel_combo)
    audio_input_wav_form.addRow(QLabel("モニター"), controller.audio_input_wav_local_playback_check)
    audio_input_wav_form.addRow(QLabel("反対側のチャンネル→MMD"), controller.audio_input_wav_other_ch_to_mmd_check)
    audio_input_wav_form.addRow(QLabel("反対側のチャンネル→MaAI ch2"), controller.audio_input_wav_other_ch_to_maai_ch2_check)
    audio_input_wav_buttons = QHBoxLayout()
    audio_input_wav_buttons.addWidget(audio_input_wav_play_pause_btn)
    audio_input_wav_buttons.addWidget(audio_input_wav_end_btn)
    audio_input_wav_buttons.addStretch(1)
    audio_input_wav_buttons_widget = QWidget()
    audio_input_wav_buttons_widget.setLayout(audio_input_wav_buttons)
    audio_input_wav_form.addRow(QLabel("操作"), audio_input_wav_buttons_widget)
    audio_input_wav_page = QWidget()
    audio_input_wav_page.setLayout(audio_input_wav_form)

    audio_input_tcp_form = QFormLayout()
    audio_input_tcp_form.addRow(QLabel("受信アドレス"), controller.audio_input_tcp_host_edit)
    audio_input_tcp_form.addRow(QLabel("受信ポート"), controller.audio_input_tcp_port_edit)
    audio_input_tcp_page = QWidget()
    audio_input_tcp_page.setLayout(audio_input_tcp_form)

    audio_input_mode_stack = QStackedWidget()
    audio_input_mode_stack.addWidget(audio_input_mic_page)  # mic
    audio_input_mode_stack.addWidget(audio_input_wav_page)  # wav
    audio_input_mode_stack.addWidget(audio_input_tcp_page)  # tcp

    def _sync_audio_input_aec_ref_combo() -> None:
        combo = getattr(controller, "audio_input_wav_aec_ref_combo", None)
        if combo is None:
            return
        mode = controller.audio_input_mode_combo.currentText().strip().lower()
        combo.blockSignals(True)
        prev_data = combo.currentData()
        combo.clear()
        if mode == "wav":
            combo.addItem("未選択ch", "other_ch")
            combo.addItem("システム音声", "system")
            target = prev_data if prev_data in ("other_ch", "system") else "other_ch"
        else:
            combo.addItem("システム音声", "system")
            target = "system"
        ri = combo.findData(target)
        if ri < 0:
            ri = 0
        combo.setCurrentIndex(ri)
        combo.blockSignals(False)

    controller._sync_audio_input_aec_ref_combo = _sync_audio_input_aec_ref_combo

    def _sync_audio_input_mode_ui() -> None:
        mode = controller.audio_input_mode_combo.currentText().strip().lower()
        idx = 0 if mode == "mic" else (1 if mode == "wav" else 2)
        audio_input_mode_stack.setCurrentIndex(idx)
        _sync_audio_input_aec_ref_combo()
        controller._sync_audio_input_wav_transport_button()

    controller.audio_input_mode_combo.currentTextChanged.connect(lambda _text: _sync_audio_input_mode_ui())
    _sync_audio_input_mode_ui()

    audio_input_settings_layout = QVBoxLayout()
    audio_input_settings_layout.addWidget(audio_input_mode_box)
    audio_input_settings_layout.addWidget(audio_input_aec_page)
    audio_input_settings_layout.addWidget(audio_input_mode_stack)
    audio_input_settings_layout.addStretch(1)
    controller.audio_input_settings_window = QWidget(controller, Qt.WindowType.Window)
    controller.audio_input_settings_window.setWindowTitle("音声入力の設定")
    controller.audio_input_settings_window.resize(520, 340)
    controller.audio_input_settings_window.setLayout(audio_input_settings_layout)

    controller.mmd_burst_snd_frames_label = QLabel("MMD先頭バースト（SND枚数）")
    controller.mmd_burst_snd_frames_label.setToolTip(
        "MMD（Remote）出力のときのみ有効です。先頭のSNDフレーム数（各640B）をバースト送信し、その後は実時間ペースに切り替えます。"
    )
    controller.mmd_burst_snd_frames_edit.setToolTip(
        "出力先が「MMD」のときのみ適用します。0では常に実時間のみです。ローカル／TCP出力では先頭バーストを使いません。"
    )
    controller.mmd_blink_enable_check.setToolTip(
        "会話や口パクとは独立して、MMDモデルのまばたきモーフを自動で動かします。"
    )
    controller.mmd_blink_interval_mean_sec_label = QLabel("まばたき平均間隔（秒）")
    controller.mmd_blink_interval_mean_sec_label.setToolTip(
        "まばたき開始間隔の平均値です。1.0～30.0秒の範囲で指定します。"
    )
    controller.mmd_blink_interval_mean_sec_edit.setToolTip(
        "人の自発性瞬目に合わせ、間隔は短いことが多くたまに長く空くようにばらつき、"
        "約6%は2回、約1.5%は3回続けてまばたきします。"
    )
    controller.mmd_blink_morph_label = QLabel("まばたきモーフ名")
    controller.mmd_blink_morph_label.setToolTip(
        "モデルで目を閉じるモーフ名です。既定は「まばたき」です。"
    )
    controller.mmd_blink_morph_edit.setToolTip(
        "カンマ、等号、改行を含む名前は使えず、その場合は「まばたき」に戻ります。"
    )
    controller.mmd_blink_mode_label = QLabel("まばたきする時間")
    controller.mmd_blink_mode_label.setToolTip(
        "「常に」はMMDへ接続している間ずっと、「対話中のみ」は音声対話を開始してから停止するまでまばたきします。"
    )
    controller.mmd_blink_mode_combo.setToolTip(
        "「対話中のみ」では、音声対話を停止すると目を開けた状態に戻して止まります。"
    )
    controller.audio_output_mode_label = QLabel("音声出力先")
    controller.audio_output_mode_label.setToolTip(
        "TTS再生の送出先です（いずれも音声出力ワーカー経由）。"
        "MMDはプラグイン設定と同じRemoteブリッジ、ローカルはワーカー内のsounddevice、TCPは送信先へ640Bフレームを順送りします。"
    )
    controller.audio_output_tcp_host_label = QLabel("TCP出力ホスト")
    controller.audio_output_tcp_port_label = QLabel("TCP出力ポート")
    audio_output_settings_form = QFormLayout()
    audio_output_settings_form.addRow(controller.mmd_burst_snd_frames_label, controller.mmd_burst_snd_frames_edit)
    audio_output_settings_form.addRow(controller.mmd_blink_enable_check)
    audio_output_settings_form.addRow(controller.mmd_blink_mode_label, controller.mmd_blink_mode_combo)
    audio_output_settings_form.addRow(
        controller.mmd_blink_interval_mean_sec_label, controller.mmd_blink_interval_mean_sec_edit
    )
    audio_output_settings_form.addRow(controller.mmd_blink_morph_label, controller.mmd_blink_morph_edit)
    audio_output_settings_form.addRow(controller.audio_output_mode_label, controller.audio_output_mode_combo)
    audio_output_settings_form.addRow(controller.audio_output_tcp_host_label, controller.audio_output_tcp_host_edit)
    audio_output_settings_form.addRow(controller.audio_output_tcp_port_label, controller.audio_output_tcp_port_edit)
    audio_output_settings_layout = QVBoxLayout()
    audio_output_settings_layout.addLayout(audio_output_settings_form)
    audio_output_settings_layout.addStretch(1)
    controller.audio_output_settings_window = QWidget(controller, Qt.WindowType.Window)
    controller.audio_output_settings_window.setWindowTitle("音声出力の設定")
    controller.audio_output_settings_window.resize(480, 390)
    controller.audio_output_settings_window.setLayout(audio_output_settings_layout)

    controller.tts_test_window = QWidget(controller, Qt.WindowType.Window)
    controller.tts_test_window.setWindowTitle("TTSテスト")
    # エンジン切替で中身の高さは大きく変わる。ウィンドウ自体は固定気味にして
    # ScrollArea で逃がさないと、Windows の mintrack より小さい setGeometry が
    # 拒否され QWindowsWindow::setGeometry 警告が連発する。
    controller.tts_test_window.resize(600, 720)
    controller.tts_test_window.setMinimumSize(420, 360)
    tts_test_scroll_inner = QWidget()
    tts_test_layout = QVBoxLayout(tts_test_scroll_inner)
    tts_test_layout.setContentsMargins(8, 8, 8, 8)
    tts_test_main_hint = QLabel(
        "テスト再生時に選択中のTTSエンジンを自動起動します。"
        "TEMPLATE は接続先の外部HTTPサービスをあらかじめ起動してください。"
    )
    tts_test_main_hint.setWordWrap(True)
    # メイン画面から外したTTS制御UIを、このウィンドウで所有・表示する。
    tts_test_layout.addWidget(tts_test_main_hint)
    tts_test_layout.addWidget(tts_control_box)
    tts_test_layout.addWidget(tts_exe_container)
    tts_test_layout.addWidget(tts_settings_container)
    tts_test_layout.addWidget(tts_test_box)
    tts_test_layout.addStretch(1)
    tts_test_scroll = QScrollArea()
    tts_test_scroll.setWidgetResizable(True)
    tts_test_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    tts_test_scroll.setWidget(tts_test_scroll_inner)
    tts_test_win_layout = QVBoxLayout()
    tts_test_win_layout.setContentsMargins(4, 4, 4, 4)
    tts_test_win_layout.addWidget(tts_test_scroll)
    controller.tts_test_window.setLayout(tts_test_win_layout)
    controller._on_tts_engine_changed(controller.tts_engine_combo.currentText())

    controller.console_window = QWidget(controller, Qt.WindowType.Window)
    controller.console_window.setWindowTitle("プロセスコンソール")
    controller.console_window.resize(760, 460)
    controller.console_text_edit = QPlainTextEdit()
    controller.console_text_edit.setReadOnly(True)
    console_layout = QVBoxLayout()
    console_layout.addWidget(controller.console_text_edit)
    controller.console_window.setLayout(console_layout)

    controller.engine_manager_window = QWidget(controller, Qt.WindowType.Window)
    controller.engine_manager_window.setWindowTitle("エンジン管理")
    controller.engine_manager_window.resize(480, 520)
    engine_manager_layout = QVBoxLayout()
    engine_manager_layout.addWidget(controller.engine_install_box)
    controller.engine_manager_window.setLayout(engine_manager_layout)

    controller.asr_test_window = QWidget(controller, Qt.WindowType.Window)
    controller.asr_test_window.setWindowTitle("ASRテスト")
    controller.asr_test_window.resize(720, 860)
    asr_test_layout = QVBoxLayout()
    asr_test_layout.addWidget(asr_runtime_box)
    asr_test_layout.addWidget(asr_box)
    asr_test_layout.addStretch(1)
    controller.asr_test_window.setLayout(asr_test_layout)

    llm_connection_form = QFormLayout()
    controller.llm_host_label = QLabel("LM Studioホスト")
    controller.llm_model_lmstudio_label = QLabel("LM Studioモデル")
    controller.llm_model_openai_label = QLabel("OpenAIモデル")
    controller.llm_api_key_label = QLabel("OpenAI APIキー")
    controller.llm_api_key_env_label = QLabel("OpenAI APIキー環境変数")
    controller.llm_custom_base_url_label = QLabel("CustomベースURL")
    controller.llm_custom_api_key_label = QLabel("Custom APIキー")
    controller.llm_custom_api_key_env_label = QLabel("Custom APIキー環境変数")
    controller.llm_model_custom_label = QLabel("Customモデル")
    controller.llm_custom_send_temperature_label = QLabel("temperatureを送信")
    controller.llm_custom_send_max_tokens_label = QLabel("max_tokensを送信")
    controller.llm_custom_temperature_value_label = QLabel("temperature値")
    controller.llm_custom_extra_body_thinking_label = QLabel("extra_body")
    controller.llm_system_prompt_label = QLabel("システムプロンプト")
    controller.llm_context_len_label = QLabel("コンテキスト長")
    controller.llm_max_tokens_label = QLabel("最大トークン数")
    controller.llm_tts_stream_label = QLabel("LLM→TTSストリーミング")
    controller.llm_test_tts_input_streaming_label = QLabel("TTS入力ストリーミング（s2s）")
    controller.llm_split_custom_delimiters_label = QLabel("LLM分割の区切り文字（Custom）")
    controller.llm_split_custom_strip_emoji_label = QLabel("LLM分割の絵文字除去（Custom）")
    controller.llm_split_custom_strip_parenthetical_label = QLabel("LLM分割の括弧除去（Custom）")
    controller.llm_split_custom_strip_standalone_warai_label = QLabel("LLM分割の単独「笑」除去（Custom）")
    llm_model_lmstudio_row = QWidget()
    llm_model_lmstudio_layout = QHBoxLayout()
    llm_model_lmstudio_layout.setContentsMargins(0, 0, 0, 0)
    llm_model_lmstudio_layout.addWidget(controller.llm_model_lmstudio_combo, 1)
    llm_model_lmstudio_layout.addWidget(llm_model_lmstudio_refresh_btn, 0)
    llm_model_lmstudio_row.setLayout(llm_model_lmstudio_layout)
    controller.llm_model_lmstudio_row = llm_model_lmstudio_row
    # 「現在以外をアンロード」「ウォームアップ」はモデル行の一段下にまとめる
    llm_model_lmstudio_actions_row = QWidget()
    llm_model_lmstudio_actions_layout = QHBoxLayout()
    llm_model_lmstudio_actions_layout.setContentsMargins(0, 0, 0, 0)
    llm_model_lmstudio_actions_layout.addWidget(llm_model_unload_btn, 0)
    llm_model_lmstudio_actions_layout.addWidget(llm_warmup_btn, 0)
    llm_model_lmstudio_actions_layout.addStretch(1)
    llm_model_lmstudio_actions_row.setLayout(llm_model_lmstudio_actions_layout)
    controller.llm_model_lmstudio_actions_row = llm_model_lmstudio_actions_row
    llm_model_openai_row = QWidget()
    llm_model_openai_layout = QHBoxLayout()
    llm_model_openai_layout.setContentsMargins(0, 0, 0, 0)
    llm_model_openai_layout.addWidget(controller.llm_model_openai_combo, 1)
    llm_model_openai_layout.addWidget(llm_model_openai_refresh_btn, 0)
    llm_model_openai_row.setLayout(llm_model_openai_layout)
    controller.llm_model_openai_row = llm_model_openai_row
    llm_model_custom_row = QWidget()
    llm_model_custom_layout = QHBoxLayout()
    llm_model_custom_layout.setContentsMargins(0, 0, 0, 0)
    llm_model_custom_layout.addWidget(controller.llm_model_custom_combo, 1)
    llm_model_custom_row.setLayout(llm_model_custom_layout)
    controller.llm_model_custom_row = llm_model_custom_row
    llm_model_lmstudio_row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    llm_lmstudio_form = QFormLayout()
    llm_lmstudio_form.setContentsMargins(0, 0, 0, 0)
    llm_lmstudio_form.addRow(controller.llm_host_label, controller.llm_host_edit)
    llm_lmstudio_form.addRow(controller.llm_model_lmstudio_label, llm_model_lmstudio_row)
    llm_lmstudio_form.addRow("", llm_model_lmstudio_actions_row)
    llm_lmstudio_page = QWidget()
    llm_lmstudio_page.setLayout(llm_lmstudio_form)
    llm_lmstudio_page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.llm_lmstudio_settings_page = llm_lmstudio_page
    llm_openai_form = QFormLayout()
    llm_openai_form.setContentsMargins(0, 0, 0, 0)
    llm_openai_form.addRow(controller.llm_model_openai_label, llm_model_openai_row)
    llm_openai_form.addRow(controller.llm_api_key_label, controller.llm_api_key_edit)
    llm_openai_form.addRow(controller.llm_api_key_env_label, controller.llm_api_key_env_edit)
    llm_openai_page = QWidget()
    llm_openai_page.setLayout(llm_openai_form)
    llm_openai_page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.llm_openai_settings_page = llm_openai_page
    llm_custom_form = QFormLayout()
    llm_custom_form.setContentsMargins(0, 0, 0, 0)
    llm_custom_form.addRow(controller.llm_custom_base_url_label, controller.llm_custom_base_url_edit)
    llm_custom_form.addRow(controller.llm_custom_api_key_label, controller.llm_custom_api_key_edit)
    llm_custom_form.addRow(controller.llm_custom_api_key_env_label, controller.llm_custom_api_key_env_edit)
    llm_custom_form.addRow(controller.llm_model_custom_label, llm_model_custom_row)
    llm_custom_form.addRow(controller.llm_custom_send_temperature_label, controller.llm_custom_send_temperature_check)
    llm_custom_form.addRow(controller.llm_custom_send_max_tokens_label, controller.llm_custom_send_max_tokens_check)
    llm_custom_form.addRow(controller.llm_custom_temperature_value_label, controller.llm_custom_temperature_edit)
    llm_custom_form.addRow(
        controller.llm_custom_extra_body_thinking_label, controller.llm_custom_extra_body_thinking_check
    )
    llm_custom_page = QWidget()
    llm_custom_page.setLayout(llm_custom_form)
    llm_custom_page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.llm_custom_settings_page = llm_custom_page
    llm_backend_stack = _CurrentPageSizeStack()
    llm_backend_stack.addWidget(llm_lmstudio_page)
    llm_backend_stack.addWidget(llm_openai_page)
    llm_backend_stack.addWidget(llm_custom_page)
    llm_backend_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    controller.llm_backend_stack = llm_backend_stack
    llm_connection_form.addRow(QLabel("バックエンド"), controller.llm_backend_combo)
    llm_connection_form.addRow(llm_backend_stack)
    llm_connection_form.addRow(controller.llm_system_prompt_label, controller.llm_system_prompt_edit)
    llm_connection_form.addRow(controller.llm_context_len_label, controller.llm_context_len_edit)
    llm_connection_form.addRow(controller.llm_max_tokens_label, controller.llm_max_tokens_edit)
    llm_connection_form.addRow(controller.llm_tts_stream_label, controller.llm_tts_stream_check)
    llm_s2s_test_open_row = QWidget()
    llm_s2s_test_open_layout = QHBoxLayout()
    llm_s2s_test_open_layout.setContentsMargins(0, 0, 0, 0)
    llm_s2s_test_open_layout.addWidget(llm_s2s_test_open_btn, 0)
    llm_s2s_test_open_layout.addStretch(1)
    llm_s2s_test_open_row.setLayout(llm_s2s_test_open_layout)
    llm_s2s_test_open_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    llm_connection_form.addRow(QLabel("入力文分割"), llm_s2s_test_open_row)
    llm_connection_box = QGroupBox("LLM接続設定")
    llm_connection_box.setLayout(llm_connection_form)
    controller.llm_connection_box = llm_connection_box

    llm_chat_form = QFormLayout()
    llm_chat_form.addRow(QLabel("入力"), controller.llm_user_input_edit)
    llm_chat_action_row = QWidget()
    llm_chat_action_layout = QHBoxLayout()
    llm_chat_action_layout.setContentsMargins(0, 0, 0, 0)
    llm_chat_action_layout.addWidget(llm_chat_send_btn, 0)
    llm_chat_action_layout.addWidget(llm_chat_reset_btn, 0)
    llm_chat_action_layout.addStretch(1)
    llm_chat_action_row.setLayout(llm_chat_action_layout)
    llm_chat_form.addRow(llm_chat_action_row)
    llm_chat_form.addRow(QLabel("実行状態"), controller.llm_status_label)
    llm_chat_form.addRow(QLabel("対話ログ"), controller.llm_chat_view)
    llm_chat_box = QGroupBox("LLMテストチャット")
    llm_chat_box.setLayout(llm_chat_form)

    controller.llm_test_window = QWidget(controller, Qt.WindowType.Window)
    controller.llm_test_window.setWindowTitle("LLMテスト")
    controller.llm_test_window.resize(760, 760)
    llm_test_layout = QVBoxLayout()
    llm_test_layout.addWidget(llm_connection_box)
    llm_test_layout.addWidget(llm_chat_box, 1)
    controller.llm_test_window.setLayout(llm_test_layout)

    def _refresh_llm_connection_layout(_index: int = -1) -> None:
        llm_backend_stack.updateGeometry()
        llm_connection_box.updateGeometry()
        controller.llm_test_window.updateGeometry()

    llm_backend_stack.currentChanged.connect(_refresh_llm_connection_layout)
    controller._on_llm_backend_changed(controller.llm_backend_combo.currentText())
    _refresh_llm_connection_layout()

    maai_launch_btn = QPushButton("起動")
    maai_launch_btn.setProperty("role", "primary")
    maai_stop_btn = QPushButton("停止")
    maai_clear_btn = QPushButton("ログクリア")
    maai_device_combo = QComboBox()
    maai_device_combo.addItems(["cpu", "cuda"])
    maai_device_combo.setCurrentText("cuda")
    maai_test_turn_vap_frame_rate_edit = QLineEdit("20")
    maai_test_turn_vap_mode_edit = QLineEdit("vap_mc")
    maai_test_turn_vap_lang_edit = QLineEdit("jp_kyoto")
    maai_test_turn_vap_model_type_edit = QComboBox()
    maai_test_turn_vap_model_type_edit.addItems(["normal", "normal-ver2"])
    maai_test_turn_vap_local_model_edit = QLineEdit("")
    maai_test_turn_vap_local_model_edit.setPlaceholderText("未指定＝公式モデル")
    maai_test_turn_vap_local_model_browse_btn = QPushButton("参照…")
    maai_test_bc_frame_rate_edit = QLineEdit("10")
    maai_test_bc_lang_edit = QLineEdit("jp")
    maai_test_bc_model_type_edit = QComboBox()
    maai_test_bc_model_type_edit.addItems(["normal", "normal-ver2"])
    maai_test_bc_local_model_edit = QLineEdit("")
    maai_test_bc_local_model_edit.setPlaceholderText("未指定＝公式モデル")
    maai_test_bc_local_model_browse_btn = QPushButton("参照…")
    maai_test_nod_frame_rate_edit = QLineEdit("12.5")
    maai_test_nod_lang_edit = QLineEdit("jp")
    maai_test_nod_model_type_edit = QComboBox()
    maai_test_nod_model_type_edit.addItems(["normal", "normal-ver2"])
    maai_test_nod_model_type_edit.setCurrentText("normal-ver2")
    maai_multi_enabled_check = QCheckBox("マルチモード（エンコーダー共有）")
    maai_multi_frame_rate_edit = QLineEdit("10")
    maai_multi_model_type_edit = QComboBox()
    maai_multi_model_type_edit.addItems(["normal", "normal-ver2"])
    maai_multi_model_type_edit.setCurrentText("normal-ver2")
    maai_use_mimi_onnx_check = QCheckBox("Mimiエンコーダー（ONNX）")
    maai_use_mimi_onnx_check.setChecked(True)
    maai_use_mimi_onnx_check.setToolTip(
        "オン：MimiエンコーダーをONNX Runtimeで推論します。オフ：Torchを使います。"
        "相槌・頷き・VAPのTransformerは常にTorchを使います。"
    )
    maai_use_local_mimi_onnx_check = QCheckBox("ローカルファイルを使う")
    maai_use_local_mimi_onnx_check.setChecked(False)
    maai_use_local_mimi_onnx_check.setToolTip(
        "オン：下のMaAIルートディレクトリ・ONNXモデル・メタデータを使います。"
        "オフ：Hugging Face（maai-kyoto/continuous-mimi-onnx）から取得してキャッシュします。"
    )
    maai_onnx_row = QWidget()
    maai_onnx_row_layout = QHBoxLayout(maai_onnx_row)
    maai_onnx_row_layout.setContentsMargins(0, 0, 0, 0)
    maai_onnx_row_layout.addWidget(maai_use_mimi_onnx_check)
    maai_onnx_row_layout.addWidget(maai_use_local_mimi_onnx_check)
    maai_onnx_row_layout.addStretch(1)

    def _maai_onnx_path_row(placeholder: str):
        edit = QLineEdit("")
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(controller._save_app_config)
        browse = QPushButton("参照…")
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit, 1)
        lay.addWidget(browse, 0)
        return edit, browse, row

    maai_root_edit, maai_root_browse_btn, maai_root_row = _maai_onnx_path_row(
        "空欄＝SDS/maai_assets"
    )
    maai_mimi_onnx_path_edit, maai_mimi_onnx_path_browse_btn, maai_mimi_onnx_path_row = _maai_onnx_path_row(
        "空欄＝onnx/onnx_v5/continuous_mimi_fp32.onnx"
    )
    maai_mimi_onnx_meta_path_edit, maai_mimi_onnx_meta_path_browse_btn, maai_mimi_onnx_meta_path_row = _maai_onnx_path_row(
        "空欄＝.onnxに対応する.jsonを自動選択"
    )
    maai_mimi_onnx_intra_threads_edit = QLineEdit("8")
    maai_mimi_onnx_inter_threads_edit = QLineEdit("1")
    maai_torch_intra_threads_edit = QLineEdit("1")
    maai_torch_interop_threads_edit = QLineEdit("1")
    maai_torch_intra_threads_edit.setToolTip(
        "Torchの演算内並列数（既定1）。MaAIマルチモードは相槌・頷き・VAPを同時に実行するため、"
        "各モデル内をさらに分割すると競合します（Windows実測：1が最速、8は2～3割低速）。"
    )
    maai_torch_interop_threads_edit.setToolTip(
        "Torchの演算間並列数（既定：1）です。MaAIは相槌・頷き・VAPを同時実行するため、増やすと高速化する場合がありますが、他のワーカーとCPUコアを奪い合います。"
    )
    maai_inference_chunk_frames_edit = QLineEdit("1")
    maai_mimi_onnx_intra_threads_edit.setPlaceholderText("1–64（既定 8）")
    maai_mimi_onnx_inter_threads_edit.setPlaceholderText("1–64（既定 1）")
    maai_inference_chunk_frames_edit.setPlaceholderText("1–8（既定 1）")
    maai_inference_chunk_frames_edit.setToolTip(
        "Mimiエンコーダーが1回のONNX呼び出しで処理するフレーム数（キー：maai_inference_chunk_frames）。"
        "2以上にするとNフレームをまとめて1回で推論するため、ONNX呼び出し回数が減ります。"
        "Nが2以上の場合は、N用のMimi ONNX（continuous_mimi_{fp32|int8}_Nf.onnx）が別途必要です。"
        "ローカルファイルがない場合はエラーになり、Hugging Faceから取得する場合はN用モデルを自動選択します。"
    )
    maai_inference_chunk_frames_edit.editingFinished.connect(controller._save_app_config)
    _mimi_thread_tip = (
        "ORTのCPUスレッド数です。device=cpuのときだけ有効です。"
        "cudaではMaAIがintra/inter=0に固定します。"
        "SDSではASRと同居するため、既定はintra=2／inter=1です。"
        "単体処理を高速化する場合は4～8を指定します。"
    )
    maai_mimi_onnx_intra_threads_edit.setToolTip(_mimi_thread_tip)
    maai_mimi_onnx_inter_threads_edit.setToolTip(_mimi_thread_tip)
    maai_mimi_onnx_intra_threads_edit.editingFinished.connect(controller._save_app_config)
    maai_mimi_onnx_inter_threads_edit.editingFinished.connect(controller._save_app_config)
    maai_torch_intra_threads_edit.editingFinished.connect(controller._save_app_config)
    maai_torch_interop_threads_edit.editingFinished.connect(controller._save_app_config)
    maai_root_label = QLabel("MaAIルートディレクトリ")
    maai_mimi_onnx_path_label = QLabel("Mimi ONNXモデル")
    maai_mimi_onnx_meta_path_label = QLabel("Mimiメタデータ")
    maai_onnx_params_box = QGroupBox("ONNX推論（Mimiエンコーダー）")
    maai_onnx_params_form = QFormLayout()
    maai_onnx_params_form.addRow(QLabel("有効"), maai_onnx_row)
    maai_onnx_params_form.addRow(maai_root_label, maai_root_row)
    maai_onnx_params_form.addRow(maai_mimi_onnx_path_label, maai_mimi_onnx_path_row)
    maai_onnx_params_form.addRow(maai_mimi_onnx_meta_path_label, maai_mimi_onnx_meta_path_row)
    maai_onnx_detail_form = QFormLayout()
    maai_onnx_detail_form.addRow(QLabel("演算内スレッド数"), maai_mimi_onnx_intra_threads_edit)
    maai_onnx_detail_form.addRow(QLabel("演算間スレッド数"), maai_mimi_onnx_inter_threads_edit)
    maai_onnx_detail_form.addRow(QLabel("Torch演算内スレッド数"), maai_torch_intra_threads_edit)
    maai_onnx_detail_form.addRow(QLabel("Torch演算間スレッド数"), maai_torch_interop_threads_edit)
    maai_onnx_detail_form.addRow(QLabel("推論チャンクのフレーム数"), maai_inference_chunk_frames_edit)
    maai_onnx_detail = CollapsibleSection("詳細設定（スレッド数・推論チャンク）")
    maai_onnx_detail.setContentLayout(maai_onnx_detail_form)
    controller.maai_onnx_detail_section = maai_onnx_detail
    maai_onnx_params_layout = QVBoxLayout()
    maai_onnx_params_layout.addLayout(maai_onnx_params_form)
    maai_onnx_params_layout.addWidget(maai_onnx_detail)
    maai_onnx_params_box.setLayout(maai_onnx_params_layout)
    maai_onnx_params_box.setToolTip(
        "MimiエンコーダーのONNX切替です。"
        "「ローカルファイルを使う」が有効な場合のみ、maai_root・.onnx・メタデータを指定できます"
        "（空欄ではmaai_root/onnx/onnx_v5の既定値）。"
        "無効な場合はHugging Faceから取得します。"
        "相槌・頷き・VAPのTransformerはTorchを使います。演算内・演算間スレッド数はCPU EPでのみ有効です。"
    )
    maai_root_browse_btn.clicked.connect(controller._browse_maai_root_dir)
    maai_mimi_onnx_path_browse_btn.clicked.connect(
        lambda: controller._browse_maai_onnx_file("maai_mimi_onnx_path_edit", "maai_mimi_onnx_meta_path_edit", "Mimi ONNX")
    )
    maai_mimi_onnx_meta_path_browse_btn.clicked.connect(
        lambda: controller._browse_maai_onnx_meta_file("maai_mimi_onnx_meta_path_edit", "Mimi ONNX meta (.json)")
    )
    maai_use_mimi_onnx_check.stateChanged.connect(lambda _s: controller._on_maai_mimi_onnx_options_changed())
    maai_use_local_mimi_onnx_check.stateChanged.connect(lambda _s: controller._on_maai_mimi_onnx_options_changed())
    maai_test_nod_count_t0_edit = QLineEdit("1.0")
    maai_test_nod_count_t1_edit = QLineEdit("1.0")
    maai_test_nod_count_t2_edit = QLineEdit("1.0")
    maai_test_nod_swing_threshold_edit = QLineEdit("0.5")
    maai_test_nod_count_t0_default_label = QLabel("既定：—")
    maai_test_nod_count_t1_default_label = QLabel("既定：—")
    maai_test_nod_count_t2_default_label = QLabel("既定：—")
    maai_test_nod_swing_default_label = QLabel("既定：—")
    for _dl in (
        maai_test_nod_count_t0_default_label,
        maai_test_nod_count_t1_default_label,
        maai_test_nod_count_t2_default_label,
        maai_test_nod_swing_default_label,
    ):
        _dl.setMinimumWidth(140)
        _dl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    # nod_para motion generation UI
    nod_para_enabled_check = QCheckBox("有効")
    nod_para_prob_threshold_edit = QLineEdit("0.6")
    nod_para_min_consecutive_frames_edit = QLineEdit("3")
    nod_para_min_interval_ms_edit = QLineEdit("1500")
    nod_para_decay_rate_edit = QLineEdit("0.8")
    nod_para_range_clip_min_edit = QLineEdit("0.0")
    nod_para_range_clip_max_edit = QLineEdit("1.0")
    nod_para_speed_clip_min_edit = QLineEdit("0.0")
    nod_para_speed_clip_max_edit = QLineEdit("1.0")
    nod_para_fps_edit = QLineEdit("30")
    nod_para_interpolation_combo = QComboBox()
    nod_para_interpolation_combo.addItems(["cosine", "cubic_spline", "cubic_hermite", "hermite_extrema"])
    nod_para_pre_rise_ratio_edit = QLineEdit("0.8")
    nod_para_scale_edit = QLineEdit("3.0")
    nod_para_speed_scale_edit = QLineEdit("1.2")
    nod_para_eye_scale_edit = QLineEdit("1.0")
    nod_para_bone_edit = QLineEdit("頭")
    for _npe in (
        nod_para_prob_threshold_edit, nod_para_min_consecutive_frames_edit,
        nod_para_min_interval_ms_edit, nod_para_decay_rate_edit,
        nod_para_range_clip_min_edit, nod_para_range_clip_max_edit,
        nod_para_speed_clip_min_edit, nod_para_speed_clip_max_edit,
        nod_para_fps_edit, nod_para_pre_rise_ratio_edit,
        nod_para_scale_edit, nod_para_speed_scale_edit,
        nod_para_eye_scale_edit, nod_para_bone_edit,
    ):
        _npe.editingFinished.connect(controller._save_app_config)
    nod_para_enabled_check.stateChanged.connect(controller._save_app_config)
    nod_para_interpolation_combo.currentTextChanged.connect(controller._save_app_config)
    maai_launch_btn.clicked.connect(controller.launch_maai_test)
    maai_stop_btn.clicked.connect(controller.stop_maai_test)
    maai_device_combo.currentTextChanged.connect(lambda _text: controller._save_app_config())
    for _e in (
        maai_test_turn_vap_frame_rate_edit,
        maai_test_turn_vap_mode_edit,
        maai_test_turn_vap_lang_edit,
        maai_test_turn_vap_local_model_edit,
        maai_test_bc_frame_rate_edit,
        maai_test_bc_lang_edit,
        maai_test_bc_local_model_edit,
        maai_test_nod_frame_rate_edit,
        maai_test_nod_lang_edit,
        maai_test_nod_count_t0_edit,
        maai_test_nod_count_t1_edit,
        maai_test_nod_count_t2_edit,
        maai_test_nod_swing_threshold_edit,
        maai_multi_frame_rate_edit,
    ):
        _e.editingFinished.connect(controller._save_app_config)
    for _c in (
        maai_test_turn_vap_model_type_edit,
        maai_test_bc_model_type_edit,
        maai_test_nod_model_type_edit,
        maai_multi_model_type_edit,
    ):
        _c.currentTextChanged.connect(lambda _text: controller._save_app_config())
    maai_multi_enabled_check.stateChanged.connect(lambda _: controller._save_app_config())
    maai_backchannel_enabled_check = QCheckBox()
    maai_backchannel_enabled_check.stateChanged.connect(lambda _state: controller._save_app_config())
    maai_backchannel_prob_threshold_edit = QLineEdit("0.5")
    maai_backchannel_prob_threshold_edit.editingFinished.connect(controller._save_app_config)
    maai_backchannel_min_interval_edit = QLineEdit("1.0")
    maai_backchannel_min_interval_edit.editingFinished.connect(controller._save_app_config)
    maai_backchannel_consecutive_frames_edit = QLineEdit("1")
    maai_backchannel_consecutive_frames_edit.editingFinished.connect(controller._save_app_config)
    maai_backchannel_dictionary_path_edit = QLineEdit()
    maai_backchannel_dictionary_path_edit.editingFinished.connect(controller._save_app_config)
    maai_backchannel_dictionary_path_browse_btn = QPushButton("参照")
    maai_backchannel_dictionary_path_browse_btn.clicked.connect(controller._browse_maai_backchannel_dictionary_path)
    maai_status_label = QLabel("状態：MaAI未起動")
    maai_log_edit = QPlainTextEdit()
    maai_log_edit.setReadOnly(True)
    maai_log_edit.setPlaceholderText("MaAI推論結果ログ（相槌確率／頷き確率）がここに表示されます")
    maai_clear_btn.clicked.connect(maai_log_edit.clear)
    maai_plot_open_btn = QPushButton("MaAIプロット")
    maai_plot_open_btn.setToolTip("MaAIプロットウィンドウを開きます")
    maai_plot_open_btn.setMaximumWidth(220)
    maai_plot_open_btn.clicked.connect(controller._open_maai_plot_window)
    maai_nod_chart_open_btn = QPushButton("頷きモーション波形")
    maai_nod_chart_open_btn.setToolTip("頷きモーション波形ウィンドウを開きます")
    maai_nod_chart_open_btn.setMaximumWidth(220)
    maai_nod_chart_open_btn.clicked.connect(controller._open_nod_motion_chart_window)
    controller.asr_maai_gui_plot_check = QCheckBox("有効")
    controller.asr_maai_gui_plot_check.setChecked(False)
    controller.asr_maai_gui_plot_check.setToolTip("MaAI結果・生波形・ASRセグメントをプロットに反映します（音声対話・ASRテスト共通）")
    controller.maai_plot_flush_ms_edit = QLineEdit("200")
    controller.maai_plot_flush_ms_edit.setToolTip("MaAI結果のバッチ描画間隔（ms）。約1000÷値Hzです（例：200→5Hz）")
    controller.maai_plot_asr_scroll_ms_edit = QLineEdit("200")
    controller.maai_plot_asr_scroll_ms_edit.setToolTip(
        "認識が止まっている間も、ASRセグメント／レイテンシ帯を時刻軸に合わせて左へ流すタイマー間隔（ms）です。"
        "描画更新間隔（ms）とは別です。認識イベント時の反映は間引きしません。"
    )
    _maai_plot_raw_default = f"{(1000.0 / 15.0):.6g}"
    controller.maai_plot_raw_draw_interval_edit = QLineEdit(_maai_plot_raw_default)
    controller.maai_plot_raw_draw_interval_edit.setToolTip("入力波形（x1/x2）の再描画の最小間隔（ms）")
    controller.asr_segment_arrival_align_check = QCheckBox("有効")
    controller.asr_segment_arrival_align_check.setChecked(False)
    controller.asr_segment_arrival_align_check.setToolTip("ASR確定セグメントの右端を到着時刻に合わせて表示します")
    controller.asr_maai_gui_plot_check.stateChanged.connect(
        lambda _state: controller._on_asr_maai_gui_plot_toggled()
    )
    controller.maai_plot_flush_ms_edit.editingFinished.connect(controller._on_maai_plot_timing_ui_changed)
    controller.maai_plot_asr_scroll_ms_edit.editingFinished.connect(controller._on_maai_plot_timing_ui_changed)
    controller.maai_plot_raw_draw_interval_edit.editingFinished.connect(controller._on_maai_plot_timing_ui_changed)
    controller.asr_segment_arrival_align_check.stateChanged.connect(controller._on_asr_segment_arrival_align_changed)
    controller.asr_segment_arrival_align_check.stateChanged.connect(lambda _state: controller._save_app_config())
    maai_settings_open_btn = QPushButton("MaAIモデル設定")
    maai_settings_open_btn.setToolTip("ターンテイキング・相槌・頷きモデルの設定ウィンドウを開きます")
    maai_settings_open_btn.setMaximumWidth(120)
    maai_action_row = QWidget()
    maai_action_layout = QHBoxLayout()
    maai_action_layout.setContentsMargins(0, 0, 0, 0)
    maai_action_layout.addWidget(maai_launch_btn, 0)
    maai_action_layout.addWidget(maai_stop_btn, 0)
    maai_action_layout.addWidget(maai_clear_btn, 0)
    maai_action_layout.addWidget(maai_settings_open_btn, 0)
    maai_action_layout.addWidget(maai_plot_open_btn, 0)
    maai_action_layout.addWidget(maai_nod_chart_open_btn, 0)
    maai_action_layout.addStretch(1)
    maai_action_row.setLayout(maai_action_layout)
    maai_backchannel_dictionary_path_row = QWidget()
    maai_backchannel_dictionary_path_layout = QHBoxLayout()
    maai_backchannel_dictionary_path_layout.setContentsMargins(0, 0, 0, 0)
    maai_backchannel_dictionary_path_layout.addWidget(maai_backchannel_dictionary_path_edit, 1)
    maai_backchannel_dictionary_path_layout.addWidget(maai_backchannel_dictionary_path_browse_btn, 0)
    maai_backchannel_dictionary_path_row.setLayout(maai_backchannel_dictionary_path_layout)
    maai_turn_vap_params_box = QGroupBox("ターンテイキングVAP・ASRのMaAI VAD")
    maai_turn_vap_params_box.setCheckable(True)
    maai_turn_vap_params_box.setChecked(True)
    controller.maai_enable_turn_vap_box = maai_turn_vap_params_box
    maai_turn_vap_params_form = QFormLayout()
    maai_turn_vap_params_form.addRow(QLabel("フレームレート（Hz）"), maai_test_turn_vap_frame_rate_edit)
    maai_turn_vap_params_form.addRow(QLabel("モード"), maai_test_turn_vap_mode_edit)
    maai_turn_vap_params_form.addRow(QLabel("言語"), maai_test_turn_vap_lang_edit)
    maai_turn_vap_params_form.addRow(QLabel("モデル種別"), maai_test_turn_vap_model_type_edit)
    maai_turn_vap_local_model_row = QWidget()
    maai_turn_vap_local_model_row_layout = QHBoxLayout()
    maai_turn_vap_local_model_row_layout.setContentsMargins(0, 0, 0, 0)
    maai_turn_vap_local_model_row_layout.addWidget(maai_test_turn_vap_local_model_edit, 1)
    maai_turn_vap_local_model_row_layout.addWidget(maai_test_turn_vap_local_model_browse_btn, 0)
    maai_turn_vap_local_model_row.setLayout(maai_turn_vap_local_model_row_layout)
    maai_turn_vap_params_form.addRow(QLabel("ローカルモデル（.pt）"), maai_turn_vap_local_model_row)
    maai_test_turn_vap_local_model_browse_btn.clicked.connect(controller._browse_maai_turn_vap_local_model)
    maai_turn_vap_params_form.addRow(QLabel("MaAIターンしきい値"), controller.asr_maai_turn_threshold_edit)
    maai_turn_vap_params_box.setLayout(maai_turn_vap_params_form)
    maai_test_turn_vap_frame_rate_edit.setToolTip(
        "MaAI VAPのframe_hz（1秒あたりのフレーム数、小数可）です。チャンク長≒sample_rate／frame_rate。"
        "MaAIワーカーのturn_vapと、vad_backend=maaiのASRのVADで共通して使われます。"
    )
    maai_test_turn_vap_mode_edit.setToolTip("ターンテイキング／MaAI VADのモード（例：vap_mc、vap）")
    maai_test_turn_vap_lang_edit.setToolTip("ターンテイキング／MaAI VADの言語（例：jp_kyoto、jp）")
    maai_test_turn_vap_model_type_edit.setToolTip(
        "model_type（normal＝CPC、normal-ver2＝Mimi＋ONNX）です。ver2は起動時のCUDA初期化に時間がかかることがあります。"
    )
    maai_bc_params_box = QGroupBox("相槌モデル")
    maai_bc_params_box.setCheckable(True)
    maai_bc_params_box.setChecked(True)
    controller.maai_enable_bc_box = maai_bc_params_box
    maai_bc_params_form = QFormLayout()
    maai_bc_params_form.addRow(QLabel("フレームレート（Hz）"), maai_test_bc_frame_rate_edit)
    maai_bc_params_form.addRow(QLabel("言語"), maai_test_bc_lang_edit)
    maai_bc_params_form.addRow(QLabel("モデル種別"), maai_test_bc_model_type_edit)
    maai_bc_local_model_row = QWidget()
    maai_bc_local_model_row_layout = QHBoxLayout()
    maai_bc_local_model_row_layout.setContentsMargins(0, 0, 0, 0)
    maai_bc_local_model_row_layout.addWidget(maai_test_bc_local_model_edit, 1)
    maai_bc_local_model_row_layout.addWidget(maai_test_bc_local_model_browse_btn, 0)
    maai_bc_local_model_row.setLayout(maai_bc_local_model_row_layout)
    maai_bc_params_form.addRow(QLabel("ローカルモデル（.pt）"), maai_bc_local_model_row)
    maai_test_bc_local_model_browse_btn.clicked.connect(controller._browse_maai_bc_local_model)
    maai_bc_params_form.addRow(QLabel("相槌有効"), maai_backchannel_enabled_check)
    maai_backchannel_playback_combo = QComboBox()
    maai_backchannel_playback_combo.addItem("再生する", "play")
    maai_backchannel_playback_combo.addItem("再生する（MaAI注入しない）", "play_no_inject")
    maai_backchannel_playback_combo.addItem("再生しない（MaAI注入のみ）", "silent")
    maai_backchannel_playback_combo.setToolTip(
        "相槌の予測・TTS合成は共通です。\n"
        "・再生する：スピーカー／MMDへ再生し、設定に応じてMaAIへも注入\n"
        "・再生する（MaAI注入しない）：スピーカー／MMDへは出すがMaAI入力へは流さない\n"
        "・再生しない（MaAI注入のみ）：スピーカー／MMDには出さずMaAI入力へだけ流す"
    )
    maai_backchannel_playback_combo.currentIndexChanged.connect(lambda _i: controller._save_app_config())
    controller.maai_backchannel_playback_combo = maai_backchannel_playback_combo
    maai_bc_params_form.addRow(QLabel("相槌再生"), maai_backchannel_playback_combo)
    maai_bc_params_form.addRow(QLabel("相槌確率のしきい値"), maai_backchannel_prob_threshold_edit)
    maai_bc_params_form.addRow(QLabel("相槌最小間隔（秒）"), maai_backchannel_min_interval_edit)
    maai_bc_params_form.addRow(QLabel("相槌連続フレーム数"), maai_backchannel_consecutive_frames_edit)
    maai_bc_params_form.addRow(QLabel("相槌辞書（JSON）"), maai_backchannel_dictionary_path_row)
    maai_bc_params_box.setLayout(maai_bc_params_form)
    maai_test_bc_frame_rate_edit.setToolTip("bcのframe_hz（1秒あたりフレーム数、小数可）")
    maai_test_bc_lang_edit.setToolTip("bcの言語タグ（例：jp、en、jp_kyoto）")
    maai_test_bc_model_type_edit.setToolTip("bcのmodel_type（例：normal、normal-ver2）")
    maai_nod_params_box = QGroupBox("頷きモデル")
    maai_nod_params_box.setCheckable(True)
    maai_nod_params_box.setChecked(True)
    controller.maai_enable_nod_box = maai_nod_params_box
    maai_nod_params_form = QFormLayout()
    maai_nod_params_form.addRow(QLabel("フレームレート（Hz）"), maai_test_nod_frame_rate_edit)
    maai_nod_params_form.addRow(QLabel("言語"), maai_test_nod_lang_edit)
    maai_nod_params_form.addRow(QLabel("モデル種別"), maai_test_nod_model_type_edit)

    def _maai_nod_thr_edit_row(edit: QLineEdit, default_lbl: QLabel) -> QWidget:
        row_w = QWidget()
        row_lay = QHBoxLayout(row_w)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.addWidget(edit, 1)
        row_lay.addWidget(default_lbl, 0)
        return row_w

    maai_nod_params_form.addRow(
        QLabel("頷き回数しきい値（t0）"),
        _maai_nod_thr_edit_row(maai_test_nod_count_t0_edit, maai_test_nod_count_t0_default_label),
    )
    maai_nod_params_form.addRow(
        QLabel("頷き回数しきい値（t1）"),
        _maai_nod_thr_edit_row(maai_test_nod_count_t1_edit, maai_test_nod_count_t1_default_label),
    )
    maai_nod_params_form.addRow(
        QLabel("頷き回数しきい値（t2）"),
        _maai_nod_thr_edit_row(maai_test_nod_count_t2_edit, maai_test_nod_count_t2_default_label),
    )
    maai_nod_params_form.addRow(
        QLabel("頷き上振り確率のしきい値"),
        _maai_nod_thr_edit_row(maai_test_nod_swing_threshold_edit, maai_test_nod_swing_default_label),
    )
    # nod_para motion settings sub-group
    nod_para_motion_box = QGroupBox("頷きモーション送信")
    nod_para_motion_form = QFormLayout()
    nod_para_motion_form.addRow(QLabel("MMDへ送信"), nod_para_enabled_check)
    nod_para_motion_form.addRow(QLabel("頷き確率のしきい値"), nod_para_prob_threshold_edit)
    nod_para_motion_form.addRow(QLabel("連続フレーム数"), nod_para_min_consecutive_frames_edit)
    nod_para_motion_form.addRow(QLabel("クールダウン（ms）"), nod_para_min_interval_ms_edit)
    nod_para_motion_form.addRow(QLabel("減衰率"), nod_para_decay_rate_edit)
    nod_para_motion_form.addRow(QLabel("振れ幅クリップ下限"), nod_para_range_clip_min_edit)
    nod_para_motion_form.addRow(QLabel("振れ幅クリップ上限"), nod_para_range_clip_max_edit)
    nod_para_motion_form.addRow(QLabel("速度クリップ下限"), nod_para_speed_clip_min_edit)
    nod_para_motion_form.addRow(QLabel("速度クリップ上限"), nod_para_speed_clip_max_edit)
    nod_para_motion_form.addRow(QLabel("FPS"), nod_para_fps_edit)
    nod_para_motion_form.addRow(QLabel("補間"), nod_para_interpolation_combo)
    nod_para_motion_form.addRow(QLabel("振り上げ倍率"), nod_para_pre_rise_ratio_edit)
    nod_para_motion_form.addRow(QLabel("振幅倍率"), nod_para_scale_edit)
    nod_para_motion_form.addRow(QLabel("再生速度倍率"), nod_para_speed_scale_edit)
    nod_para_motion_form.addRow(QLabel("目の回転倍率"), nod_para_eye_scale_edit)
    nod_para_motion_form.addRow(QLabel("頭ボーン名"), nod_para_bone_edit)
    nod_para_motion_box.setLayout(nod_para_motion_form)
    controller.nod_para_motion_form = nod_para_motion_form

    nod_para_prob_threshold_edit.setToolTip("頷きフレームを蓄積するp_nodしきい値です。")
    nod_para_min_consecutive_frames_edit.setToolTip("頷きを開始するまでに必要な連続フレーム数です。")
    nod_para_min_interval_ms_edit.setToolTip("モデルが頷きを開始する最小間隔（ms）です。")
    nod_para_decay_rate_edit.setToolTip("2回目以降の頷きの振幅減衰率（0～1）です。")
    nod_para_range_clip_min_edit.setToolTip(
        "モデル予測nod_range（振れ幅、rad）のクリップ下限です。既定：0.0。"
    )
    nod_para_range_clip_max_edit.setToolTip(
        "モデル予測nod_range（振れ幅、rad）のクリップ上限です。既定：1.0。"
    )
    nod_para_speed_clip_min_edit.setToolTip(
        "モデル予測nod_speed（速度、rad/s）のクリップ下限です。既定：0.0。"
    )
    nod_para_speed_clip_max_edit.setToolTip(
        "モデル予測nod_speed（速度、rad/s）のクリップ上限です。既定：1.0。"
    )
    nod_para_fps_edit.setToolTip("モーション生成フレームレート")
    nod_para_interpolation_combo.setToolTip("キーフレーム間の補間方式です（cosineはscipy不要）。")
    nod_para_pre_rise_ratio_edit.setToolTip("振り上げの大きさです（range_radに対する倍率）。")
    nod_para_scale_edit.setToolTip("最終ピッチに掛ける倍率")
    nod_para_speed_scale_edit.setToolTip("再生速度スケールです（1より大きいと高速化）。")
    nod_para_eye_scale_edit.setToolTip("目の逆回転倍率（0＝目は動かない）")
    nod_para_bone_edit.setToolTip("MMDAgent-EXの頭ボーン名（例：頭、首）")
    maai_nod_params_form.addRow(nod_para_motion_box)
    maai_nod_params_box.setLayout(maai_nod_params_form)
    maai_test_nod_frame_rate_edit.setToolTip("nod_paraのframe_hz（1秒あたりフレーム数、小数可）")
    maai_test_nod_lang_edit.setToolTip("nod_paraの言語タグ（例：jp、jp_kyoto）")
    maai_test_nod_model_type_edit.setToolTip("nod_paraのmodel_type（例：normal-ver2）")
    _nod_thr_tip = (
        "nod_paraの推論用しきい値です。.pt読込後のnodインスタンス値がワーカー準備完了時に既定値として同期されます。"
        "UIの値が既定値と一致する場合はVAP内部のpredをそのまま使い、異なる場合は"
        "nod_count（確率）とnod_swing_up（確率）にこのしきい値を再適用した結果を表示します。"
    )
    maai_test_nod_count_t0_edit.setToolTip(_nod_thr_tip)
    maai_test_nod_count_t1_edit.setToolTip(_nod_thr_tip)
    maai_test_nod_count_t2_edit.setToolTip(_nod_thr_tip)
    maai_test_nod_swing_threshold_edit.setToolTip(_nod_thr_tip)
    _nod_def_tip = "ワーカーの準備完了時にnodインスタンスから取得した既定しきい値（.ptなど）です。"
    maai_test_nod_count_t0_default_label.setToolTip(_nod_def_tip)
    maai_test_nod_count_t1_default_label.setToolTip(_nod_def_tip)
    maai_test_nod_count_t2_default_label.setToolTip(_nod_def_tip)
    maai_test_nod_swing_default_label.setToolTip(_nod_def_tip)
    maai_turn_vap_params_box.toggled.connect(lambda _: controller._save_app_config())
    maai_bc_params_box.toggled.connect(lambda _: controller._save_app_config())
    maai_nod_params_box.toggled.connect(lambda _: controller._save_app_config())
    maai_multi_params_box = QGroupBox("マルチモード")
    maai_multi_params_form = QFormLayout()
    maai_multi_params_form.addRow(QLabel("有効"), maai_multi_enabled_check)
    maai_multi_params_form.addRow(QLabel("フレームレート（Hz）"), maai_multi_frame_rate_edit)
    maai_multi_params_form.addRow(QLabel("モデル種別"), maai_multi_model_type_edit)
    maai_multi_params_box.setLayout(maai_multi_params_form)
    maai_multi_enabled_check.setToolTip(
        "有効にするとMaAIマルチモードを使い、ターンテイキングVAP・相槌・頷きの音声エンコーダーを共有します。"
        "全サブモデルが同じフレームレートとモデル種別を使います。"
    )
    maai_multi_frame_rate_edit.setToolTip("マルチモード時に共有するframe_hzです（相槌・頷き・ターンテイキングVAP共通）。")
    maai_multi_model_type_edit.setToolTip("マルチモード時に共有するmodel_typeです（相槌・頷き・ターンテイキングVAP共通）。")
    maai_plot_params_box = QGroupBox("MaAIプロット")
    maai_plot_params_form = QFormLayout()
    maai_plot_params_form.addRow(QLabel("プロット表示"), controller.asr_maai_gui_plot_check)
    maai_plot_params_form.addRow(QLabel("描画更新間隔（ms）"), controller.maai_plot_flush_ms_edit)
    maai_plot_params_form.addRow(QLabel("ASR時刻軸更新間隔（ms）"), controller.maai_plot_asr_scroll_ms_edit)
    maai_plot_params_form.addRow(QLabel("生波形再描画間隔（ms）"), controller.maai_plot_raw_draw_interval_edit)
    maai_plot_params_form.addRow(QLabel("ASRセグメントを到着時刻に揃える"), controller.asr_segment_arrival_align_check)
    maai_plot_visibility_box = QGroupBox("表示パネル")
    maai_plot_visibility_grid = QGridLayout()
    maai_plot_visibility_grid.setContentsMargins(4, 4, 4, 4)
    for _vi, (_panel_id, _panel_lbl) in enumerate(MAAI_PLOT_VISIBILITY_SPECS):
        _vis_cb = QCheckBox(_panel_lbl)
        _vis_cb.setChecked(True)
        _vr, _vc = divmod(_vi, 2)
        maai_plot_visibility_grid.addWidget(_vis_cb, _vr, _vc)
        setattr(controller, f"maai_plot_visibility_{_panel_id}_check", _vis_cb)
        _vis_cb.stateChanged.connect(controller._on_maai_plot_visibility_check_changed)
    maai_plot_visibility_box.setLayout(maai_plot_visibility_grid)
    maai_plot_visibility_box.setToolTip(
        "MaAIプロット上の各グラフパネルの表示を切り替えます。オフにすると描画負荷が下がります。"
    )
    maai_plot_params_form.addRow(maai_plot_visibility_box)
    maai_plot_params_box.setLayout(maai_plot_params_form)
    maai_test_box = QGroupBox("MaAIテスト")
    maai_test_form = QFormLayout()
    maai_test_form.addRow(QLabel("操作"), maai_action_row)
    maai_test_form.addRow(maai_plot_params_box)
    maai_test_form.addRow(QLabel("状態"), maai_status_label)
    maai_test_form.addRow(QLabel("ログ"), maai_log_edit)
    maai_test_box.setLayout(maai_test_form)
    controller.maai_test_window = QWidget(controller, Qt.WindowType.Window)
    controller.maai_test_window.setWindowTitle("MaAIテスト")
    controller.maai_test_window.resize(680, 520)
    maai_test_scroll = QScrollArea()
    maai_test_scroll.setWidgetResizable(True)
    maai_test_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    maai_test_scroll_inner = QWidget()
    maai_test_scroll_layout = QVBoxLayout(maai_test_scroll_inner)
    maai_test_scroll_layout.setContentsMargins(0, 0, 0, 0)
    maai_test_scroll_layout.addWidget(maai_test_box)
    maai_test_scroll.setWidget(maai_test_scroll_inner)
    maai_test_win_layout = QVBoxLayout()
    maai_test_win_layout.setContentsMargins(4, 4, 4, 4)
    maai_test_win_layout.addWidget(maai_test_scroll)
    controller.maai_test_window.setLayout(maai_test_win_layout)
    # MaAI モデル設定ウィンドウ（ターンテイキング・相槌・頷き設定）
    maai_device_combo.setToolTip(
        "bc・nod_para・ターンテイキングVAP・ASR内MaAI VADで共通の実行デバイス（cpuまたはcuda）です。"
        "ONNXでも実行デバイスに合わせてCUDA EP／CPU EPを選択します"
        "（INT8 MatMulはCUDA非対応になりやすいため、CUDAではfp32 ONNXを推奨）。"
    )
    maai_settings_box = QGroupBox("MaAIモデル設定")
    maai_settings_form = QFormLayout()
    maai_settings_form.addRow(QLabel("実行デバイス（MaAI共通）"), maai_device_combo)
    maai_settings_form.addRow(maai_onnx_params_box)
    maai_settings_form.addRow(maai_multi_params_box)
    maai_settings_form.addRow(maai_turn_vap_params_box)
    maai_settings_form.addRow(maai_bc_params_box)
    maai_settings_form.addRow(maai_nod_params_box)
    maai_settings_box.setLayout(maai_settings_form)
    controller.maai_settings_window = QWidget(controller, Qt.WindowType.Window)
    controller.maai_settings_window.setWindowTitle("MaAIモデル設定")
    controller.maai_settings_window.resize(680, 800)
    maai_settings_scroll = QScrollArea()
    maai_settings_scroll.setWidgetResizable(True)
    maai_settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    maai_settings_scroll_inner = QWidget()
    maai_settings_scroll_layout = QVBoxLayout(maai_settings_scroll_inner)
    maai_settings_scroll_layout.setContentsMargins(0, 0, 0, 0)
    maai_settings_scroll_layout.addWidget(maai_settings_box)
    maai_settings_scroll.setWidget(maai_settings_scroll_inner)
    maai_settings_win_layout = QVBoxLayout()
    maai_settings_win_layout.setContentsMargins(4, 4, 4, 4)
    maai_settings_win_layout.addWidget(maai_settings_scroll)
    controller.maai_settings_window.setLayout(maai_settings_win_layout)
    maai_settings_open_btn.clicked.connect(
        lambda: (controller.maai_settings_window.show(), controller.maai_settings_window.raise_())
    )

    maai_nod_motion_chart = NodMotionChartWidget()
    maai_nod_motion_chart.setMinimumHeight(300)
    maai_nod_motion_chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    controller.nod_motion_chart_window = QWidget(controller, Qt.WindowType.Window)
    controller.nod_motion_chart_window.setWindowTitle("頷きモーション波形")
    controller.nod_motion_chart_window.resize(720, 400)
    _nod_chart_win_layout = QVBoxLayout()
    _nod_chart_win_layout.addWidget(
        QLabel("最新1回分（音声対話で頷きモーションが生成されたときに更新されます）"),
    )
    _nod_chart_win_layout.addWidget(maai_nod_motion_chart, 1)
    controller.nod_motion_chart_window.setLayout(_nod_chart_win_layout)
    controller.maai_nod_motion_chart = maai_nod_motion_chart

    controller.maai_test_status_label = maai_status_label
    controller.maai_test_log_edit = maai_log_edit
    controller.maai_test_launch_btn = maai_launch_btn
    controller.maai_test_stop_btn = maai_stop_btn
    controller.maai_device_combo = maai_device_combo
    controller.maai_test_turn_vap_frame_rate_edit = maai_test_turn_vap_frame_rate_edit
    controller.maai_test_turn_vap_mode_edit = maai_test_turn_vap_mode_edit
    controller.maai_test_turn_vap_lang_edit = maai_test_turn_vap_lang_edit
    controller.maai_test_turn_vap_model_type_edit = maai_test_turn_vap_model_type_edit
    controller.maai_test_turn_vap_local_model_edit = maai_test_turn_vap_local_model_edit
    controller.maai_test_bc_frame_rate_edit = maai_test_bc_frame_rate_edit
    controller.maai_test_bc_lang_edit = maai_test_bc_lang_edit
    controller.maai_test_bc_model_type_edit = maai_test_bc_model_type_edit
    controller.maai_test_bc_local_model_edit = maai_test_bc_local_model_edit
    controller.maai_test_nod_frame_rate_edit = maai_test_nod_frame_rate_edit
    controller.maai_test_nod_lang_edit = maai_test_nod_lang_edit
    controller.maai_test_nod_model_type_edit = maai_test_nod_model_type_edit
    controller.maai_multi_enabled_check = maai_multi_enabled_check
    controller.maai_multi_frame_rate_edit = maai_multi_frame_rate_edit
    controller.maai_multi_model_type_edit = maai_multi_model_type_edit
    controller.maai_use_mimi_onnx_check = maai_use_mimi_onnx_check
    controller.maai_use_local_mimi_onnx_check = maai_use_local_mimi_onnx_check
    controller.maai_root_edit = maai_root_edit
    controller.maai_root_browse_btn = maai_root_browse_btn
    controller.maai_root_label = maai_root_label
    controller.maai_mimi_onnx_path_edit = maai_mimi_onnx_path_edit
    controller.maai_mimi_onnx_path_browse_btn = maai_mimi_onnx_path_browse_btn
    controller.maai_mimi_onnx_path_label = maai_mimi_onnx_path_label
    controller.maai_mimi_onnx_meta_path_edit = maai_mimi_onnx_meta_path_edit
    controller.maai_mimi_onnx_meta_path_browse_btn = maai_mimi_onnx_meta_path_browse_btn
    controller.maai_mimi_onnx_meta_path_label = maai_mimi_onnx_meta_path_label
    controller.maai_mimi_onnx_intra_threads_edit = maai_mimi_onnx_intra_threads_edit
    controller.maai_mimi_onnx_inter_threads_edit = maai_mimi_onnx_inter_threads_edit
    controller.maai_torch_intra_threads_edit = maai_torch_intra_threads_edit
    controller.maai_torch_interop_threads_edit = maai_torch_interop_threads_edit
    controller.maai_inference_chunk_frames_edit = maai_inference_chunk_frames_edit
    controller.maai_onnx_params_box = maai_onnx_params_box
    controller._sync_maai_local_mimi_onnx_controls()
    controller.maai_test_nod_count_t0_edit = maai_test_nod_count_t0_edit
    controller.maai_test_nod_count_t1_edit = maai_test_nod_count_t1_edit
    controller.maai_test_nod_count_t2_edit = maai_test_nod_count_t2_edit
    controller.maai_test_nod_swing_threshold_edit = maai_test_nod_swing_threshold_edit
    controller.maai_test_nod_count_t0_default_label = maai_test_nod_count_t0_default_label
    controller.maai_test_nod_count_t1_default_label = maai_test_nod_count_t1_default_label
    controller.maai_test_nod_count_t2_default_label = maai_test_nod_count_t2_default_label
    controller.maai_test_nod_swing_default_label = maai_test_nod_swing_default_label
    controller.nod_para_enabled_check = nod_para_enabled_check
    controller.nod_para_prob_threshold_edit = nod_para_prob_threshold_edit
    controller.nod_para_min_consecutive_frames_edit = nod_para_min_consecutive_frames_edit
    controller.nod_para_min_interval_ms_edit = nod_para_min_interval_ms_edit
    controller.nod_para_decay_rate_edit = nod_para_decay_rate_edit
    controller.nod_para_range_clip_min_edit = nod_para_range_clip_min_edit
    controller.nod_para_range_clip_max_edit = nod_para_range_clip_max_edit
    controller.nod_para_speed_clip_min_edit = nod_para_speed_clip_min_edit
    controller.nod_para_speed_clip_max_edit = nod_para_speed_clip_max_edit
    controller.nod_para_fps_edit = nod_para_fps_edit
    controller.nod_para_interpolation_combo = nod_para_interpolation_combo
    controller.nod_para_pre_rise_ratio_edit = nod_para_pre_rise_ratio_edit
    controller.nod_para_scale_edit = nod_para_scale_edit
    controller.nod_para_speed_scale_edit = nod_para_speed_scale_edit
    controller.nod_para_eye_scale_edit = nod_para_eye_scale_edit
    controller.nod_para_bone_edit = nod_para_bone_edit
    controller.maai_backchannel_enabled_check = maai_backchannel_enabled_check
    controller.maai_backchannel_prob_threshold_edit = maai_backchannel_prob_threshold_edit
    controller.maai_backchannel_min_interval_edit = maai_backchannel_min_interval_edit
    controller.maai_backchannel_consecutive_frames_edit = maai_backchannel_consecutive_frames_edit
    controller.maai_backchannel_dictionary_path_edit = maai_backchannel_dictionary_path_edit
    _finish_main_layout(
        controller,
        root,
        settings_window_box,
        mmd_control_box,
        voice_dialog_start_btn,
        voice_dialog_stop_btn,
        llm_s2s_test_open_btn,
        llm_s2s_test_run_btn,
        maai_plot_open_btn,
    )

def apply_device_choice_limits(controller) -> None:
    """デバイス構成に合わせて GPU 専用の選択肢を無効化 / 再有効化する（項目は残す）。

    黙って cpu へ落とす処理は dialog/settings.py にあるが、それだけだと
    「選べたのに効いていない」状態になる。エンジン管理のチェックボックスと同じく、
    出したまま押せなくして理由をツールチップに出す。

    **掛け直せることが必要**。エンジン管理で CPU → GPU に切り替えたあと、これを
    もう一度呼べば GPU 項目が戻る。以前は「無効化するものが無ければ即 return」して
    いたため、切り替えてもアプリを再起動するまで無効のままだった。
    そのため空振りでも GPU_ONLY_CHOICES の全項目を走査して状態を設定する。

    現在値が無効項目だった場合（GPU 機で書いた設定を CPU 機で開いた等）は、
    無効項目が選択されたままになるのを避けて選べる項目へ寄せる。
    """
    from .. import device as _device_rules

    disabled = _device_rules.disabled_choices()
    for attr, gpu_values in _device_rules.GPU_ONLY_CHOICES.items():
        combo = getattr(controller, attr, None)
        if combo is None or not hasattr(combo, "model"):
            continue
        # この構成で塞ぐ値。GPU 構成なら空集合になり、全項目が戻る。
        blocked_values = set(disabled.get(attr, ()))
        model = combo.model()
        blocked: list[int] = []
        for index in range(combo.count()):
            if combo.itemText(index) not in gpu_values:
                continue
            item = model.item(index) if hasattr(model, "item") else None
            if item is None:
                continue
            is_blocked = combo.itemText(index) in blocked_values
            item.setEnabled(not is_blocked)
            combo.setItemData(
                index,
                _device_rules.GPU_ONLY_REASON if is_blocked else None,
                Qt.ItemDataRole.ToolTipRole,
            )
            if is_blocked:
                blocked.append(index)
        if blocked and combo.currentIndex() in blocked:
            for index in range(combo.count()):
                if index not in blocked:
                    combo.setCurrentIndex(index)
                    break
