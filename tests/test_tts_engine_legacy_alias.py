"""削除した TTS エンジン（PyTorch 版 "Qwen3-TTS"）の旧名が、保存済みの設定で落ちないこと。"""

from __future__ import annotations

import unittest

from maai_sds.dialog import settings as dialog_settings
from maai_sds.tts.worker import ENGINE_FACTORIES, TEXT_STREAM_ENGINES
from maai_sds.web import parity


class LegacyTtsEngineAliasTest(unittest.TestCase):
    def test_removed_engine_is_not_offered(self) -> None:
        self.assertNotIn("Qwen3-TTS", ENGINE_FACTORIES)
        self.assertNotIn("Qwen3-TTS", TEXT_STREAM_ENGINES)
        self.assertIn("Qwen3-TTS-CPP", ENGINE_FACTORIES)

    def test_normalize(self) -> None:
        self.assertEqual(dialog_settings.normalize_tts_engine(" Qwen3-TTS "), "Qwen3-TTS-CPP")
        self.assertEqual(dialog_settings.normalize_tts_engine("Qwen3-TTS-CPP"), "Qwen3-TTS-CPP")
        self.assertEqual(dialog_settings.normalize_tts_engine("VOICEVOX-Core"), "VOICEVOX-Core")
        self.assertEqual(dialog_settings.cfg_tts_engine({}), "VOICEVOX")
        self.assertEqual(dialog_settings.cfg_tts_engine({"tts_engine": "Qwen3-TTS"}), "Qwen3-TTS-CPP")

    def test_dialog_settings_use_replacement_engine(self) -> None:
        cfg = {"tts_engine": "Qwen3-TTS"}
        payload = dialog_settings.build_start_voice_dialog_payload(cfg)
        self.assertEqual(payload["tts_engine"], "Qwen3-TTS-CPP")
        worker = dialog_settings.build_tts_worker_config(cfg, "Qwen3-TTS")
        self.assertIn("model_id", worker)  # Qwen3-TTS-CPP の設定が組み立てられる

    def test_web_config_is_canonicalized(self) -> None:
        fixed = parity.canonical_combo_values({"tts_engine": "Qwen3-TTS"})
        self.assertEqual(fixed["tts_engine"], "Qwen3-TTS-CPP")
        kept = parity.canonical_combo_values({"tts_engine": "Qwen3-TTS-CPP"})
        self.assertEqual(kept["tts_engine"], "Qwen3-TTS-CPP")


if __name__ == "__main__":
    unittest.main()
