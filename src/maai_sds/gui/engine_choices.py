"""ASR・TTS のエンジンの選択欄。一覧の 1 行は「モデル（実行系）」の組み合わせ 1 つ。

以前の一覧は、行によって実行系（sherpa-onnx・speechcpp-asr）だったりモデル（Irodori-v4）だったりで、
speech.cpp のモデルと Irodori の実行系は下の別の欄で選んでいた。同じモデル（Qwen3-ASR など）が
一覧の別々の場所にもあった。

保存する値は変えない。エンジンの値（asr_engine / tts_engine）は今までどおりで、speech.cpp のモデル
（asr_speechcpp_model）と Irodori の実行系・モデル（irodori_backend / irodori_audiocpp_model）は、
一覧で行を選ぶと隠した下の欄が合わせて切り替わる。そのため設定ファイルの形も古い設定も
そのまま使える。コードが読む `currentText()` もエンジンの値を返す（表示名ではない）。

Web の画面定義（scripts/extract_web_ui_manifest.py）は同じ一覧と、連動する設定キーを載せる。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QWidget


@dataclass(frozen=True)
class EngineChoice:
    label: str
    engine: str
    # 行を選んだときに合わせる隠した欄（controller の属性名, その欄の itemData）
    subs: tuple[tuple[str, str], ...] = ()
    # 一覧の見出し（実行系）。_grouped が埋める。
    runtime: str = ""


_SPEECHCPP = "asr_speechcpp_model_combo"
_IRODORI_BACKEND = "irodori_backend_combo"
_IRODORI_AUDIOCPP_MODEL = "irodori_audiocpp_model_combo"


def _grouped(groups: tuple[tuple[str, tuple[EngineChoice, ...]], ...]) -> tuple[EngineChoice, ...]:
    return tuple(replace(c, runtime=runtime) for runtime, rows in groups for c in rows)


# 実行系ごとに見出しを付けて並べる（選ぶときに実行系で探せるように）。行の名前にも実行系を残すのは、
# 一覧を閉じたとき（選んだ 1 行だけが見えるとき）にも実行系が分かるようにするため。
ASR_ENGINE_CHOICES: tuple[EngineChoice, ...] = _grouped((
    ("sherpa-onnx", (
        EngineChoice("ReazonSpeech k2（sherpa-onnx）", "sherpa-onnx"),
        EngineChoice("Qwen3-ASR（sherpa-onnx）", "qwen3-asr-onnx"),
        EngineChoice("Parakeet TDT-CTC ja（sherpa-onnx）", "parakeet-onnx"),
        EngineChoice("Nemotron 3.5 streaming（sherpa-onnx）", "nemotron-onnx"),
        EngineChoice("SenseVoiceSmall（sherpa-onnx）", "sensevoice-onnx"),
        EngineChoice("Cohere Transcribe（sherpa-onnx）", "cohere-asr-onnx"),
        EngineChoice("Fun-ASR-Nano（sherpa-onnx）", "funasr-nano-onnx"),
    )),
    ("speech.cpp", (
        EngineChoice("Haas-ja v1.0（speech.cpp）", "speechcpp-asr", ((_SPEECHCPP, "haas-ja-v1.0"),)),
        EngineChoice("ReazonSpeech NeMo v2（speech.cpp）", "speechcpp-asr", ((_SPEECHCPP, "reazonspeech-v2"),)),
        EngineChoice("Parakeet TDT-CTC ja（speech.cpp）", "speechcpp-asr", ((_SPEECHCPP, "parakeet-tdt_ctc-0.6b-ja"),)),
        EngineChoice("Qwen3-ASR 0.6B（speech.cpp）", "speechcpp-asr", ((_SPEECHCPP, "qwen3-asr-0.6b"),)),
        EngineChoice("Qwen3-ASR 1.7B（speech.cpp）", "speechcpp-asr", ((_SPEECHCPP, "qwen3-asr-1.7b"),)),
    )),
    ("faster-whisper", (
        EngineChoice("Whisper（faster-whisper）", "faster-whisper"),
    )),
    ("NeMo", (
        EngineChoice("Nemotron 3.5 streaming（NeMo）", "nemotron-asr"),
    )),
    ("vLLM（Docker）", (
        EngineChoice("Qwen3-ASR（vLLM・Docker）", "qwen3-asr"),
    )),
    ("外部", (
        EngineChoice("外部ASR（TCP）", "external-asr-tcp"),
    )),
))

TTS_ENGINE_CHOICES: tuple[EngineChoice, ...] = _grouped((
    ("VOICEVOX Core", (
        EngineChoice("VOICEVOX（VOICEVOX Core）", "VOICEVOX-Core"),
    )),
    ("speech.cpp", (
        EngineChoice("Irodori-TTS v4.1 Small MF（speech.cpp）", "Irodori-v4", ((_IRODORI_BACKEND, "speechcpp"),)),
    )),
    ("audio.cpp", (
        EngineChoice("Irodori-TTS 500M v3（audio.cpp）", "Irodori-v4",
                     ((_IRODORI_BACKEND, "audiocpp"), (_IRODORI_AUDIOCPP_MODEL, "500m-v3"))),
        EngineChoice("Irodori-TTS 600M v3 VoiceDesign（audio.cpp）", "Irodori-v4",
                     ((_IRODORI_BACKEND, "audiocpp"), (_IRODORI_AUDIOCPP_MODEL, "600m-v3-voicedesign"))),
        EngineChoice("Qwen3-TTS 1.7B CustomVoice（audio.cpp）", "Qwen3-TTS-CPP"),
        EngineChoice("LFM2.5-Audio（audio.cpp）", "LFM2.5-Audio"),
        EngineChoice("Supertonic 3（audio.cpp）", "Supertonic-3"),
    )),
    ("PyTorch", (
        EngineChoice("Irodori-TTS（PyTorch）", "Irodori-v4", ((_IRODORI_BACKEND, "pytorch"),)),
        EngineChoice("Tsukasa Speech（PyTorch）", "Tsukasa"),
    )),
    ("RealtimeTTS", (
        EngineChoice("Kokoro（RealtimeTTS）", "Kokoro-RealtimeTTS"),
    )),
    ("アプリ（HTTP）", (
        EngineChoice("VOICEVOX（アプリ）", "VOICEVOX"),
        EngineChoice("AivisSpeech（アプリ）", "AivisSpeech"),
    )),
    ("クラウド", (
        EngineChoice("Cartesia（クラウド）", "Cartesia-Realtime"),
    )),
    ("HTTP", (
        EngineChoice("任意のHTTP API（TEMPLATE）", "TEMPLATE"),
    )),
))

# 一覧で選ぶようになったので画面から隠す欄（Web の画面定義でも隠す）。
HIDDEN_SUB_ATTRS: frozenset[str] = frozenset(
    attr for choices in (ASR_ENGINE_CHOICES, TTS_ENGINE_CHOICES) for c in choices for attr, _v in c.subs
)


def unique_engines(choices: tuple[EngineChoice, ...]) -> list[str]:
    """エンジンの値の一覧（並び順を保って重複を除く）。比較モードの欄などが使う。"""
    return list(dict.fromkeys(c.engine for c in choices))


def choice_id(choice: EngineChoice) -> str:
    """Web の選択肢の値（行ごとに一意）。"""
    if not choice.subs:
        return choice.engine
    return choice.engine + "::" + "&".join(f"{a}={v}" for a, v in choice.subs)


class EngineChoiceCombo(QComboBox):
    """「モデル（実行系）」の一覧。currentText() / setCurrentText() はエンジンの値で読み書きする。

    エンジンが変わったときは engineChanged を出す（currentTextChanged は表示名なので使わない）。
    """

    engineChanged = Signal(str)

    def __init__(self, controller: Any, choices: tuple[EngineChoice, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._choices = choices
        self._applying = False
        # 実行系ごとに選べない見出し行を挟む。itemData は選択肢の番号（見出しは None）。
        self._rows: list[int] = []
        runtime = None
        header_font = self.font()
        header_font.setBold(True)
        for i, c in enumerate(choices):
            if c.runtime and c.runtime != runtime:
                runtime = c.runtime
                self.addItem(c.runtime, None)
                item = self.model().item(self.count() - 1)
                if item is not None:
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                    item.setFont(header_font)
            self.addItem(c.label, i)
            self._rows.append(self.count() - 1)
        self.setCurrentIndex(self._rows[0] if self._rows else -1)
        self._last_engine = self.currentText()
        self.currentIndexChanged.connect(self._on_index_changed)
        self.activated.connect(self._on_activated)

    # --- エンジンの値で読み書きする（既存のコードは QComboBox と同じ名前で呼ぶ） ---
    def currentText(self) -> str:  # noqa: N802 - Qt と同じ名前
        c = self.current_choice()
        return c.engine if c is not None else ""

    def current_choice(self) -> EngineChoice | None:
        i = self.currentData()
        return self._choices[i] if isinstance(i, int) and 0 <= i < len(self._choices) else None

    def current_label(self) -> str:
        c = self.current_choice()
        return c.label if c is not None else ""

    def row_of(self, label: str) -> int:
        """表示名の行の位置（見出しを含めた QComboBox の番号）。"""
        for i, c in enumerate(self._choices):
            if c.label == label:
                return self._rows[i]
        return -1

    def choices(self) -> tuple[EngineChoice, ...]:
        return self._choices

    def setCurrentText(self, engine: str) -> None:  # noqa: N802
        index = self._best_index(str(engine or "").strip())
        if index is not None:
            self.setCurrentIndex(index)

    # --- 隠した欄との連動 ---
    def _sub_value(self, attr: str) -> Any:
        combo = getattr(self._controller, attr, None)
        return combo.currentData() if isinstance(combo, QComboBox) else None

    def _matches(self, c: EngineChoice) -> bool:
        return all(self._sub_value(attr) == value for attr, value in c.subs)

    def _best_index(self, engine: str) -> int | None:
        """engine の行のうち、隠した欄も合う行の位置（QComboBox の番号）。"""
        candidates = [i for i, c in enumerate(self._choices) if c.engine == engine]
        if not candidates:
            return None
        for i in candidates:
            if self._matches(self._choices[i]):
                return self._rows[i]
        return self._rows[candidates[0]]

    def _on_activated(self, index: int) -> None:
        """ユーザーが行を選んだら、隠した欄をその行のモデル・実行系に合わせる。"""
        i = self.itemData(index)
        if not (isinstance(i, int) and 0 <= i < len(self._choices)):
            return
        self._applying = True
        try:
            for attr, value in self._choices[i].subs:
                combo = getattr(self._controller, attr, None)
                if isinstance(combo, QComboBox):
                    j = combo.findData(value)
                    if j >= 0 and j != combo.currentIndex():
                        combo.setCurrentIndex(j)
        finally:
            self._applying = False

    def resync(self) -> None:
        """隠した欄が設定の読み込みなどで変わったら、一覧の行をそれに合わせる（エンジンは変えない）。"""
        if self._applying:
            return
        c = self.current_choice()
        if c is None or self._matches(c):
            return
        index = self._best_index(c.engine)
        if index is not None and index != self.currentIndex():
            self.setCurrentIndex(index)

    def _on_index_changed(self, _index: int) -> None:
        engine = self.currentText()
        if engine != self._last_engine:
            self._last_engine = engine
            self.engineChanged.emit(engine)


def link_engine_choice_combos(controller: Any) -> None:
    """隠した欄の変化を一覧に返し、その欄の行を画面から隠す。build_ui の後で 1 回呼ぶ。"""
    combos = [c for c in (getattr(controller, "asr_engine_combo", None), getattr(controller, "tts_engine_combo", None))
              if isinstance(c, EngineChoiceCombo)]
    for combo in combos:
        for attr in {a for ch in combo.choices() for a, _v in ch.subs}:
            sub = getattr(controller, attr, None)
            if isinstance(sub, QComboBox):
                sub.currentIndexChanged.connect(lambda _i, c=combo: c.resync())
        combo.resync()
    hidden = [getattr(controller, a, None) for a in HIDDEN_SUB_ATTRS]
    for form in controller.findChildren(QFormLayout):
        for w in hidden:
            if isinstance(w, QWidget):
                row, _role = form.getWidgetPosition(w)
                if row >= 0:
                    form.setRowVisible(row, False)


def resync_engine_choice_combos(controller: Any) -> None:
    for attr in ("asr_engine_combo", "tts_engine_combo"):
        combo = getattr(controller, attr, None)
        if isinstance(combo, EngineChoiceCombo):
            combo.resync()
