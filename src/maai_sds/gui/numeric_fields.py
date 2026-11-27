"""数値の入力欄の範囲。Qt（入力の制限と範囲への丸め）と Web（input type=number の min/max）が使う。

欄は QLineEdit のまま。設定の読み書き（config_io・app_config_spec）は text() / setText() を前提に
しているので、QSpinBox などへ差し替えるとそちらを全部書き換えることになる。ここでは
- 数字・小数点・符号以外を打てなくする
- 入力を終えたとき（フォーカスが外れた・Enter）に範囲の外なら端に寄せる
- ツールチップに範囲と、0 や空欄の意味を書く
だけを足す。範囲は実行時に dialog/settings.py が丸めている値に合わせてある（そちらに無いものは
実用上の範囲）。空欄はそのまま残す（空欄に意味がある欄があり、読み込み時に既定値へ落ちる）。

各要素: 属性名 -> (最小, 最大, 小数を許すか, 補足)
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject, QRegularExpression, Qt
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import QLineEdit, QToolTip

_PORT = (1, 65535, False, "")
_PROB = (0.0, 1.0, True, "")

NUMERIC_FIELDS: dict[str, tuple[float, float, bool, str]] = {
    # ポート
    "aivisspeech_port_edit": _PORT,
    "asr_external_audio_port_edit": _PORT,
    "asr_external_port_edit": _PORT,
    "asr_qwen_port_edit": _PORT,
    "audio_input_tcp_port_edit": _PORT,
    "audio_output_tcp_port_edit": _PORT,
    "irodori_audiocpp_port_edit": _PORT,
    "lfm2_audio_port_edit": _PORT,
    "obs_remote_port_edit": _PORT,
    "qwen3_tts_cpp_port_edit": _PORT,
    "supertonic_port_edit": _PORT,
    "tts_port_edit": _PORT,
    # スレッド数
    "asr_cohere_onnx_num_threads_edit": (1, 64, False, ""),
    "asr_funasr_nano_onnx_num_threads_edit": (1, 64, False, ""),
    "asr_parakeet_onnx_num_threads_edit": (1, 64, False, ""),
    "asr_qwen3_onnx_num_threads_edit": (1, 64, False, ""),
    "asr_sensevoice_onnx_num_threads_edit": (1, 64, False, ""),
    "asr_sherpa_num_threads_edit": (1, 32, False, ""),
    "asr_smart_turn_num_threads_edit": (1, 32, False, ""),
    "asr_speechcpp_threads_edit": (0, 64, False, "0＝speech.cppに任せる"),
    "asr_torch_cpu_intra_threads_edit": (1, 32, False, ""),
    "maai_mimi_onnx_inter_threads_edit": (1, 64, False, ""),
    "maai_mimi_onnx_intra_threads_edit": (1, 64, False, ""),
    "maai_torch_interop_threads_edit": (1, 24, False, ""),
    "maai_torch_intra_threads_edit": (1, 24, False, ""),
    "voicevox_core_cpu_threads_edit": (1, 32, False, ""),
    # しきい値・確率
    "asr_maai_turn_threshold_edit": _PROB,
    "asr_maai_turn_hold_frames_edit": (1, 25, False, "フレーム"),
    "asr_smart_turn_threshold_edit": _PROB,
    "asr_vad_off_threshold_edit": _PROB,
    "asr_vad_on_threshold_edit": _PROB,
    "maai_backchannel_prob_threshold_edit": _PROB,
    "maai_test_nod_count_t0_edit": (0.0, 1.0, True, "3つとも数値のときだけ使う。どれかが空欄ならモデルの値"),
    "maai_test_nod_count_t1_edit": (0.0, 1.0, True, "3つとも数値のときだけ使う。どれかが空欄ならモデルの値"),
    "maai_test_nod_count_t2_edit": (0.0, 1.0, True, "3つとも数値のときだけ使う。どれかが空欄ならモデルの値"),
    "maai_test_nod_swing_threshold_edit": (0.0, 1.0, True, "空欄＝モデルの値"),
    "nod_para_prob_threshold_edit": _PROB,
    "nod_para_decay_rate_edit": _PROB,
    "nod_para_range_clip_min_edit": _PROB,
    "nod_para_range_clip_max_edit": _PROB,
    "nod_para_speed_clip_min_edit": _PROB,
    "nod_para_speed_clip_max_edit": _PROB,
    "voice_dialog_interrupt_pnow_threshold_edit": _PROB,
    "voice_dialog_interrupt_vad_threshold_edit": _PROB,
    "tsukasa_alpha_edit": _PROB,
    "tsukasa_beta_edit": _PROB,
    "qwen3_tts_cpp_top_p_edit": _PROB,
    "asr_qwen_gpu_util_edit": (0.05, 1.0, True, ""),
    # 秒
    "asr_min_segment_sec_edit": (0.0, 10.0, True, ""),
    "asr_nemotron_window_sec_edit": (1.0, 60.0, True, ""),
    "asr_nonstream_pseudo_chunk_sec_edit": (0.05, 5.0, True, ""),
    "asr_pre_roll_buffer_sec_edit": (0.0, 5.0, True, ""),
    "asr_qwen_chunk_sec_edit": (0.02, 5.0, True, ""),
    "asr_qwen_min_return_audio_sec_edit": (0.0, 5.0, True, ""),
    "asr_qwen_no_text_flush_sec_edit": (0.0, 30.0, True, ""),
    "asr_qwen_session_timeout_edit": (1, 3600, False, ""),
    "asr_qwen_timeout_sec_edit": (1, 600, False, ""),
    "asr_vad_off_sec_edit": (0.0, 5.0, True, ""),
    "asr_vad_on_sec_edit": (0.0, 5.0, True, ""),
    "audio_input_wav_end_sec_edit": (0.0, 86400.0, True, "空欄＝最後まで"),
    "audio_input_wav_start_sec_edit": (0.0, 86400.0, True, "空欄＝最初から"),
    "maai_backchannel_min_interval_edit": (0.0, 60.0, True, ""),
    "tts_segment_gap_silence_sec_edit": (0.0, 10.0, True, ""),
    "voice_dialog_idle_take_turn_sec_edit": (0.0, 30.0, True, ""),
    "voice_dialog_scheduled_exit_sec_edit": (0.0, 86400.0, True, "0＝時間では終了しない"),
    # ミリ秒
    "maai_plot_asr_scroll_ms_edit": (10, 10000, True, ""),
    "maai_plot_flush_ms_edit": (10, 10000, True, ""),
    "maai_plot_raw_draw_interval_edit": (10, 10000, True, ""),
    "nod_para_min_interval_ms_edit": (0, 60000, True, ""),
    # フレーム数・フレームレート
    "maai_backchannel_consecutive_frames_edit": (1, 100, False, ""),
    "maai_inference_chunk_frames_edit": (1, 8, False, ""),
    "maai_multi_frame_rate_edit": (0.1, 500.0, True, ""),
    "maai_test_bc_frame_rate_edit": (0.1, 500.0, True, ""),
    "maai_test_nod_frame_rate_edit": (0.1, 500.0, True, ""),
    "maai_test_turn_vap_frame_rate_edit": (0.1, 500.0, True, ""),
    "mmd_burst_snd_frames_edit": (0, 500, False, ""),
    "nod_para_fps_edit": (1, 240, False, ""),
    "nod_para_min_consecutive_frames_edit": (1, 100, False, ""),
    # 倍率・その他
    "audio_input_gain_edit": (0.0, 20.0, True, ""),
    "nod_para_eye_scale_edit": (0.0, 10.0, True, "0＝目は動かない"),
    "nod_para_pre_rise_ratio_edit": (0.0, 2.0, True, ""),
    "nod_para_scale_edit": (0.0, 10.0, True, ""),
    "nod_para_speed_scale_edit": (0.1, 10.0, True, ""),
    "cartesia_tts_sample_rate_edit": (8000, 48000, False, ""),
    "irodori_seed_edit": (0, 2147483647, False, ""),
    "irodori_speechcpp_steps_edit": (1, 40, False, ""),
    "irodori_steps_edit": (1, 40, False, ""),
    "supertonic_steps_edit": (1, 8, False, ""),
    "supertonic_threads_edit": (0, 64, False, ""),
    "tsukasa_steps_edit": (3, 15, False, ""),
    "qwen3_tts_cpp_repetition_penalty_edit": (1.0, 2.0, True, ""),
    "qwen3_tts_cpp_seed_edit": (0, 2147483647, False, "空欄＝ランダム"),
    "qwen3_tts_cpp_temperature_edit": (0.0, 2.0, True, ""),
    "qwen3_tts_cpp_top_k_edit": (1, 1000, False, ""),
    "llm_context_len_edit": (0, 200, False, ""),
    "llm_custom_temperature_edit": (0.0, 2.0, True, ""),
    "llm_max_tokens_edit": (1, 32768, False, ""),
    "llm_s2s_test_chunk_size_edit": (1, 100, False, ""),
    "asr_funasr_nano_onnx_max_new_tokens_edit": (1, 4096, False, ""),
    "asr_nemotron_max_symbols_edit": (1, 100, False, ""),
    "asr_qwen_max_model_len_edit": (128, 32768, False, ""),
}


def _fmt(value: float) -> str:
    return f"{value:g}"


def range_text(spec: tuple[float, float, bool, str]) -> str:
    lo, hi, _decimals, note = spec
    text = f"範囲：{_fmt(lo)}〜{_fmt(hi)}"
    return f"{text}（{note}）" if note else text


def manifest_hint(attr: str | None) -> dict[str, Any] | None:
    """Web の画面定義に載せる範囲（scripts/extract_web_ui_manifest.py が使う）。"""
    spec = NUMERIC_FIELDS.get(attr or "")
    if spec is None:
        return None
    lo, hi, decimals, note = spec
    return {"min": lo, "max": hi, "decimals": decimals, "note": note or None}


class _ClampOnFinish(QObject):
    """フォーカスが外れる・Enter のとき、editingFinished より先に範囲へ寄せる。

    editingFinished に後からつなぐと、先につながっている保存処理が範囲外の値を保存してしまう。
    イベントフィルターはウィジェット自身の処理（editingFinished を出す）より前に呼ばれる。
    """

    def __init__(self, edit: QLineEdit, spec: tuple[float, float, bool, str]) -> None:
        super().__init__(edit)
        self._edit = edit
        self._spec = spec

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt の名前
        etype = event.type()
        if etype == QEvent.Type.FocusOut or (
            etype == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            self.clamp()
        return False

    def clamp(self) -> None:
        text = self._edit.text().strip()
        if not text:
            return
        lo, hi, decimals, _note = self._spec
        try:
            value = float(text)
        except ValueError:
            return
        fixed = min(max(value, lo), hi)
        if not decimals:
            fixed = round(fixed)
        if fixed != value:
            self._edit.setText(_fmt(fixed))
            QToolTip.showText(
                self._edit.mapToGlobal(self._edit.rect().bottomLeft()),
                f"{range_text(self._spec)}の数値にしてください（{_fmt(fixed)}にしました）",
                self._edit,
            )


def apply_numeric_fields(controller: Any) -> None:
    """controller の数値の欄に、入力の制限・範囲への丸め・ツールチップを付ける。"""
    for attr, spec in NUMERIC_FIELDS.items():
        edit = getattr(controller, attr, None)
        if not isinstance(edit, QLineEdit):
            continue
        lo, _hi, decimals, _note = spec
        sign = "-?" if lo < 0 else ""
        pattern = rf"{sign}\d*\.?\d*" if decimals else rf"{sign}\d*"
        # 既に QDoubleValidator が付いている欄（範囲つき）はそのまま使う。
        if edit.validator() is None:
            edit.setValidator(QRegularExpressionValidator(QRegularExpression(pattern), edit))
        clamp = _ClampOnFinish(edit, spec)
        edit.installEventFilter(clamp)
        tip = (edit.toolTip() or "").strip()
        rng = range_text(spec)
        edit.setToolTip(f"{tip}\n{rng}" if tip else rng)
