"""LFM2.5-Audio と Irodori の audio.cpp 版が、設定・エンジン管理・画面の定義につながっていること。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from maai_sds import engine_manager
from maai_sds.dialog import settings as dialog_settings
from maai_sds.gui.app_config_spec import APP_CONFIG_FIELD_BY_KEY
from maai_sds.web import parity

MANIFEST = Path(parity.__file__).with_name("ui_manifest.json")


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _combo_values(attr: str) -> list[str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in _walk_dicts(manifest):
        if item.get("attr") == attr:
            return [
                str(entry.get("value", "")) if isinstance(entry, dict) else str(entry)
                for entry in item.get("items", [])
            ]
    raise AssertionError(f"widget not found: {attr}")


class Lfm2AudioSettingsTest(unittest.TestCase):
    def test_worker_config_defaults(self) -> None:
        worker = dialog_settings.build_tts_worker_config({}, "LFM2.5-Audio")
        self.assertEqual(set(worker), {"host", "port", "model_id", "model_path", "server_exe"})
        self.assertEqual(worker["host"], "127.0.0.1")
        self.assertEqual(worker["port"], 18082)
        self.assertEqual(worker["model_id"], "lfm2.5-audio-jp")
        self.assertEqual(worker["model_path"], "audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF")
        self.assertTrue(str(worker["server_exe"]).strip())

    def test_worker_config_reads_keys(self) -> None:
        cfg = {
            "lfm2_audio_host": "10.0.0.5",
            "lfm2_audio_port": "9000",
            "lfm2_audio_model_id": "custom-id",
            "lfm2_audio_model_path": "models/lfm",
            "lfm2_audio_server_exe": "bin/server.exe",
        }
        worker = dialog_settings.build_tts_worker_config(cfg, "LFM2.5-Audio")
        self.assertEqual(worker["host"], "10.0.0.5")
        self.assertEqual(worker["port"], 9000)
        self.assertEqual(worker["model_id"], "custom-id")
        self.assertEqual(worker["model_path"], "models/lfm")
        self.assertEqual(worker["server_exe"], "bin/server.exe")

    def test_tts_inputs_and_web_worker(self) -> None:
        inputs = dialog_settings.build_tts_inputs({}, engine="LFM2.5-Audio", utterance="こんにちは")
        self.assertEqual(inputs["port"], 0)
        self.assertIn("LFM2.5-Audio", dialog_settings._WEB_WORKER_TTS)

    def test_config_fields_and_ui(self) -> None:
        for key in ("lfm2_audio_host", "lfm2_audio_port", "lfm2_audio_model_id",
                    "lfm2_audio_model_path", "lfm2_audio_server_exe"):
            self.assertIn(key, APP_CONFIG_FIELD_BY_KEY)
        self.assertIn("LFM2.5-Audio", _combo_values("tts_engine_combo"))


class IrodoriAudioCppSettingsTest(unittest.TestCase):
    def test_worker_config_has_audiocpp_keys(self) -> None:
        worker = dialog_settings.build_tts_worker_config({"irodori_backend": "audiocpp"}, "Irodori-v4")
        self.assertEqual(worker["backend"], "audiocpp")
        self.assertEqual(worker["audiocpp_model"], "500m-v3")
        self.assertEqual(worker["audiocpp_port"], 18083)
        self.assertEqual(worker["host"], "127.0.0.1")
        self.assertTrue(str(worker["server_exe"]).strip())
        # 既存のキーは残る
        for key in ("speechcpp_model", "speechcpp_device", "speechcpp_steps", "seed",
                    "irodori_python", "checkpoint", "voice", "reference_audio", "steps"):
            self.assertIn(key, worker)

    def test_worker_config_reads_keys(self) -> None:
        cfg = {
            "irodori_backend": "audiocpp",
            "irodori_audiocpp_model": "600m-v3-voicedesign",
            "irodori_audiocpp_host": "10.0.0.6",
            "irodori_audiocpp_port": "8080",
        }
        worker = dialog_settings.build_tts_worker_config(cfg, "Irodori-v4")
        self.assertEqual(worker["audiocpp_model"], "600m-v3-voicedesign")
        self.assertEqual(worker["audiocpp_port"], 8080)
        self.assertEqual(worker["host"], "10.0.0.6")

    def test_config_fields_and_ui(self) -> None:
        self.assertEqual(APP_CONFIG_FIELD_BY_KEY["irodori_audiocpp_model"][2], "combo")
        self.assertIn("irodori_audiocpp_host", APP_CONFIG_FIELD_BY_KEY)
        self.assertIn("irodori_audiocpp_port", APP_CONFIG_FIELD_BY_KEY)
        self.assertEqual(_combo_values("irodori_backend_combo"), ["speechcpp", "audiocpp", "pytorch"])
        self.assertEqual(
            _combo_values("irodori_audiocpp_model_combo"), ["500m-v3", "600m-v3-voicedesign"]
        )


class EngineManagerEntriesTest(unittest.TestCase):
    def test_new_engine_ids(self) -> None:
        ids = {info.id: info for info in engine_manager._ENGINE_DEFINITIONS}
        self.assertIn("tts-lfm2-audio", {info.id for info in engine_manager.list_engines()})
        lfm = ids["tts-lfm2-audio"]
        self.assertEqual(lfm.setup_steps, ("AudioCppServer", "Lfm2AudioGguf"))
        self.assertIsNone(lfm.extra)
        irodori = ids["tts-irodori-audiocpp"]
        self.assertEqual(irodori.setup_steps, ("AudioCppServer", "IrodoriAudioCppGguf"))
        self.assertEqual(len(irodori.probe_paths), 3)


if __name__ == "__main__":
    unittest.main()
