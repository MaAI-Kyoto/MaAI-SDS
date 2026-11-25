"""ASR・TTS のエンジンの一覧（1 行＝モデル（実行系））。保存する値は変えずに、隠した欄と連動する。"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from maai_sds.gui.engine_choices import (  # noqa: E402
    ASR_ENGINE_CHOICES,
    HIDDEN_SUB_ATTRS,
    TTS_ENGINE_CHOICES,
    EngineChoiceCombo,
)

try:
    from PySide6.QtWidgets import QApplication, QComboBox
except ImportError:  # pragma: no cover
    QApplication = None

_REPO = Path(__file__).resolve().parents[1]
# 以前の一覧（保存値）。どれも新しい一覧から選べること。
_OLD_ASR = {"sherpa-onnx", "qwen3-asr-onnx", "nemotron-onnx", "sensevoice-onnx", "parakeet-onnx",
            "cohere-asr-onnx", "funasr-nano-onnx", "speechcpp-asr", "faster-whisper", "qwen3-asr",
            "nemotron-asr", "external-asr-tcp"}
_OLD_TTS = {"VOICEVOX", "VOICEVOX-Core", "Kokoro-RealtimeTTS", "Qwen3-TTS-CPP", "LFM2.5-Audio",
            "Cartesia-Realtime", "AivisSpeech", "Supertonic-3", "Irodori-v4", "Tsukasa", "TEMPLATE"}


class ChoiceTableTest(unittest.TestCase):
    def test_every_saved_engine_value_is_still_selectable(self) -> None:
        self.assertEqual({c.engine for c in ASR_ENGINE_CHOICES}, _OLD_ASR)
        self.assertEqual({c.engine for c in TTS_ENGINE_CHOICES}, _OLD_TTS)

    def test_every_row_is_model_and_runtime_and_unique(self) -> None:
        for choices in (ASR_ENGINE_CHOICES, TTS_ENGINE_CHOICES):
            labels = [c.label for c in choices]
            self.assertEqual(len(labels), len(set(labels)))
            for label in labels:
                self.assertRegex(label, r"^.+（.+）$", label)

    def test_sub_values_exist_in_the_hidden_combos(self) -> None:
        from maai_sds import speechcpp

        speech_models = {name for name, _ in speechcpp.ASR_MODELS}
        for c in ASR_ENGINE_CHOICES:
            for attr, value in c.subs:
                self.assertEqual(attr, "asr_speechcpp_model_combo")
                self.assertIn(value, speech_models)
        # speech.cpp のモデルは全部一覧に出ている
        self.assertEqual({v for c in ASR_ENGINE_CHOICES for _a, v in c.subs}, speech_models)


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class ComboTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _controller(self):
        backend = QComboBox()
        for v in ("speechcpp", "audiocpp", "pytorch"):
            backend.addItem(v, v)
        model = QComboBox()
        for v in ("500m-v3", "600m-v3-voicedesign"):
            model.addItem(v, v)
        ctrl = SimpleNamespace(irodori_backend_combo=backend, irodori_audiocpp_model_combo=model)
        ctrl.tts_engine_combo = EngineChoiceCombo(ctrl, TTS_ENGINE_CHOICES)
        backend.currentIndexChanged.connect(lambda _i: ctrl.tts_engine_combo.resync())
        model.currentIndexChanged.connect(lambda _i: ctrl.tts_engine_combo.resync())
        return ctrl

    def test_current_text_is_the_engine_value(self) -> None:
        ctrl = self._controller()
        self.assertEqual(ctrl.tts_engine_combo.currentText(), "VOICEVOX-Core")
        ctrl.tts_engine_combo.setCurrentText("Qwen3-TTS-CPP")
        self.assertEqual(ctrl.tts_engine_combo.currentText(), "Qwen3-TTS-CPP")
        self.assertEqual(ctrl.tts_engine_combo.current_label(), "Qwen3-TTS 1.7B CustomVoice（audio.cpp）")

    def test_picking_a_row_sets_the_hidden_runtime_and_model(self) -> None:
        ctrl = self._controller()
        combo = ctrl.tts_engine_combo
        row = next(i for i, c in enumerate(TTS_ENGINE_CHOICES) if "600M" in c.label)
        emitted: list[str] = []
        combo.engineChanged.connect(emitted.append)
        combo.setCurrentIndex(row)
        combo.activated.emit(row)  # ユーザーが選んだとき
        self.assertEqual(ctrl.irodori_backend_combo.currentData(), "audiocpp")
        self.assertEqual(ctrl.irodori_audiocpp_model_combo.currentData(), "600m-v3-voicedesign")
        self.assertEqual(combo.current_label(), "Irodori-TTS 600M v3 VoiceDesign（audio.cpp）")
        self.assertEqual(emitted, ["Irodori-v4"])
        # 同じエンジンの別の行へ移っても engineChanged は出ない（実行系だけ変わる）
        row2 = next(i for i, c in enumerate(TTS_ENGINE_CHOICES) if "PyTorch" in c.label)
        combo.setCurrentIndex(row2)
        combo.activated.emit(row2)
        self.assertEqual(emitted, ["Irodori-v4"])
        self.assertEqual(ctrl.irodori_backend_combo.currentData(), "pytorch")

    def test_loading_an_old_config_selects_the_matching_row(self) -> None:
        ctrl = self._controller()
        combo = ctrl.tts_engine_combo
        # 設定の読み込み順：エンジン → 隠した欄
        combo.setCurrentText("Irodori-v4")
        ctrl.irodori_backend_combo.setCurrentIndex(ctrl.irodori_backend_combo.findData("audiocpp"))
        self.assertEqual(combo.current_label(), "Irodori-TTS 500M v3（audio.cpp）")
        ctrl.irodori_audiocpp_model_combo.setCurrentIndex(1)
        self.assertEqual(combo.current_label(), "Irodori-TTS 600M v3 VoiceDesign（audio.cpp）")


class ManifestTest(unittest.TestCase):
    def test_web_gets_the_same_rows_and_hides_the_linked_fields(self) -> None:
        manifest = json.loads((_REPO / "src/maai_sds/web/ui_manifest.json").read_text(encoding="utf-8"))
        found: dict[str, dict] = {}
        hidden: set[str] = set()

        def walk(o):
            if isinstance(o, dict):
                if o.get("engine_choice"):
                    found[o["attr"]] = o
                if o.get("hidden") and o.get("attr"):
                    hidden.add(o["attr"])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(manifest)
        self.assertEqual([i["label"] for i in found["asr_engine_combo"]["items"]],
                         [c.label for c in ASR_ENGINE_CHOICES])
        self.assertEqual([i["label"] for i in found["tts_engine_combo"]["items"]],
                         [c.label for c in TTS_ENGINE_CHOICES])
        values = [i["value"] for i in found["tts_engine_combo"]["items"]]
        self.assertEqual(len(values), len(set(values)))
        for item in found["asr_engine_combo"]["items"] + found["tts_engine_combo"]["items"]:
            for sub in item["subs"]:
                self.assertTrue(sub["config_key"], item["label"])
        self.assertEqual(hidden, set(HIDDEN_SUB_ATTRS))


if __name__ == "__main__":
    unittest.main()
