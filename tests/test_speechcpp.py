"""speech.cpp の部品（speechcpp.py・asr/engines_speechcpp.py・Irodori の切り替え）のテスト。

本物の speech.cpp は使わない。`speech worker` の JSON Lines の受け答えをまねる小さな
スクリプトを Python で動かして、要求の組み立てと応答の読み方を確かめる。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from maai_sds import engine_manager, speechcpp
from maai_sds.asr.engines_speechcpp import SpeechCppRecognizer, language_for
from maai_sds.gui.controller import MMDAgentController
from maai_sds.tts.engines import IrodoriSpeechCppEngine, IrodoriTTSEngine, make_irodori_engine

DEVICES = """\
Vulkan0    igpu  Intel(R) Graphics, 19.2 GB, 18.4 GB free
Vulkan1    gpu   NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12.5 GB, 11.7 GB free
CPU        cpu   Intel(R) Core(TM) Ultra 9 275HX, 33.8 GB, 16.6 GB free
"""

# speech worker の受け答えをまねる。認識は受け取ったサンプル数と言語を文字で返し、
# 合成は 3 チャンク（各 100 サンプル）を返す。add_voice と fatal も試せるようにする。
FAKE_WORKER = textwrap.dedent('''
    import base64, json, sys
    args = sys.argv[1:]
    if "fatal-model" in args:
        print(json.dumps({"type": "fatal", "error": {"code": "model_file", "message": "no such model"}}), flush=True)
        sys.exit(1)
    sys.stderr.write("speech worker: ready\\n"); sys.stderr.flush()
    print(json.dumps({"type": "ready", "protocol": 4, "model": {"device": args[args.index("--device") + 1],
                      "sample_rate": 48000, "args": args}}), flush=True)
    pending = {}
    for line in sys.stdin:
        m = json.loads(line)
        if m["type"] == "chunk":
            pending.setdefault(m["id"], []).append(len(base64.b64decode(m["pcm"])) // 2)
        elif m["type"] == "transcribe":
            n = sum(pending.pop(m["id"], []))
            print(json.dumps({"type": "progress", "id": m["id"], "done": 0.5}), flush=True)
            print(json.dumps({"type": "end", "id": m["id"], "text": f"n={n} sr={m['sample_rate']} lang={m.get('language', '')}",
                              "stop": "complete"}), flush=True)
        elif m["type"] == "synthesize":
            if m["text"] == "bad":
                print(json.dumps({"type": "error", "id": m["id"], "error": {"code": "invalid_argument", "option": "text",
                                  "message": "bad text"}}), flush=True)
                continue
            for seq in range(3):
                pcm = base64.b64encode(bytes(200)).decode()
                print(json.dumps({"type": "chunk", "id": m["id"], "seq": seq, "pcm": pcm}), flush=True)
            opts = {k: v for k, v in m.items() if k not in ("type", "id", "text")}
            print(json.dumps({"type": "end", "id": m["id"], "samples": 300, "stop": "complete", "opts": opts}), flush=True)
        elif m["type"] == "add_voice":
            print(json.dumps({"type": "end", "id": m["id"]}), flush=True)
''')


class _FakeExe:
    """`speech` の代わりに Python のスクリプトを動かす（Popen の先頭を差し替える）。"""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.script = root / "fake_worker.py"
        self.script.write_text(FAKE_WORKER, encoding="utf-8")
        self.exe = root / ("speech.exe" if os.name == "nt" else "speech")
        self.exe.write_bytes(b"")  # require_exe がファイルの有無だけを見る
        self.calls: list[list[str]] = []
        real_popen = subprocess.Popen

        def popen(args, **kw):
            self.calls.append(list(args))
            return real_popen([sys.executable, str(self.script), *args[2:]], **kw)

        self.patches = [
            mock.patch.object(speechcpp.subprocess, "Popen", popen),
            mock.patch.object(speechcpp, "list_devices", lambda _exe: speechcpp.parse_devices(DEVICES)),
            mock.patch.dict(os.environ, {"MAAI_SDS_SPEECHCPP_MODELS": str(root / "models")}),
        ]

    def __enter__(self) -> "_FakeExe":
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc) -> None:
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()


class DeviceChoiceTest(unittest.TestCase):
    def test_gpu_means_the_discrete_gpu_not_the_integrated_one(self) -> None:
        devices = speechcpp.parse_devices(DEVICES)
        self.assertEqual([d[0] for d in devices], ["Vulkan0", "Vulkan1", "CPU"])
        self.assertEqual(speechcpp.choose_device("gpu", devices), "Vulkan1")
        self.assertEqual(speechcpp.choose_device("", devices), "Vulkan1")

    def test_falls_back_to_cpu_without_a_discrete_gpu(self) -> None:
        only_igpu = [d for d in speechcpp.parse_devices(DEVICES) if d[1] != "gpu"]
        self.assertEqual(speechcpp.choose_device("gpu", only_igpu), "cpu")
        self.assertEqual(speechcpp.choose_device("gpu", []), "cpu")

    def test_explicit_choices_pass_through(self) -> None:
        devices = speechcpp.parse_devices(DEVICES)
        self.assertEqual(speechcpp.choose_device("cpu", devices), "cpu")
        self.assertEqual(speechcpp.choose_device("Vulkan0", devices), "Vulkan0")


class WorkerProtocolTest(unittest.TestCase):
    def test_transcribe_sends_int16_chunks_and_reads_the_text(self) -> None:
        with _FakeExe() as fake:
            worker = speechcpp.SpeechCppWorker("haas-ja-v1.0", device="gpu", exe=str(fake.exe)).start()
            try:
                self.assertEqual(worker.device, "Vulkan1")
                self.assertEqual(fake.calls[0][1:5], ["worker", "haas-ja-v1.0", "--device", "Vulkan1"])
                text = worker.transcribe(np.zeros(1600, dtype=np.float32), 16000, language="ja")
                self.assertEqual(text, "n=1600 sr=16000 lang=ja")
                # 2 回目も同じプロセスで通る（要求の id が重ならない）
                self.assertEqual(worker.transcribe(np.zeros(800, dtype=np.int16), 16000), "n=800 sr=16000 lang=")
            finally:
                worker.close()
            self.assertFalse(worker.alive)

    def test_synthesize_collects_chunks_and_raises_on_error(self) -> None:
        with _FakeExe() as fake:
            worker = speechcpp.SpeechCppWorker("irodori-tts-mf:q8_0", device="cpu", exe=str(fake.exe),
                                               options={"steps": 4}).start()
            try:
                self.assertIn("--steps", fake.calls[0])
                audio, rate = worker.synthesize("こんにちは", voice="none", seed=1)
                self.assertEqual((audio.dtype, audio.size, rate), (np.dtype(np.int16), 300, 48000))
                with self.assertRaises(speechcpp.SpeechCppError):
                    worker.synthesize("bad")
            finally:
                worker.close()

    def test_fatal_at_start_is_reported(self) -> None:
        with _FakeExe() as fake:
            with self.assertRaisesRegex(speechcpp.SpeechCppError, "no such model"):
                speechcpp.SpeechCppWorker("fatal-model", exe=str(fake.exe)).start()

    def test_missing_exe_points_to_the_engine_manager(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "エンジン管理"):
            speechcpp.SpeechCppWorker("haas-ja-v1.0", exe=str(Path(tempfile.gettempdir()) / "no-such-speech.exe"))


class RecognizerTest(unittest.TestCase):
    def test_sherpa_shaped_wrapper(self) -> None:
        worker = mock.Mock()
        worker.transcribe.return_value = "こんにちは"
        rec = SpeechCppRecognizer(worker, language="ja")
        stream = rec.create_stream()
        stream.set_option("language", "Japanese")  # sherpa の stream と同じ呼び方を受け付ける
        stream.accept_waveform(16000, np.zeros(160, dtype=np.float32))
        stream.accept_waveform(16000, np.zeros(160, dtype=np.float32))
        rec.decode_stream(stream)
        self.assertEqual(stream.result.text, "こんにちは")
        audio, rate = worker.transcribe.call_args.args
        self.assertEqual((audio.size, rate), (320, 16000))
        self.assertEqual(worker.transcribe.call_args.kwargs, {"language": "ja"})

    def test_language_only_for_multilingual_models(self) -> None:
        self.assertEqual(language_for("haas-ja-v1.0", ""), "")
        self.assertEqual(language_for("qwen3-asr-1.7b", ""), "ja")
        self.assertEqual(language_for("qwen3-asr-1.7b", "en"), "en")


class IrodoriBackendTest(unittest.TestCase):
    def test_backend_switch(self) -> None:
        self.assertIsInstance(make_irodori_engine({}), IrodoriSpeechCppEngine)
        self.assertIsInstance(make_irodori_engine({"backend": "speechcpp"}), IrodoriSpeechCppEngine)
        self.assertIsInstance(make_irodori_engine({"backend": "pytorch"}), IrodoriTTSEngine)

    def test_caption_and_fixed_seed_are_sent(self) -> None:
        with _FakeExe() as fake:
            engine = IrodoriSpeechCppEngine({"speechcpp_exe": str(fake.exe), "voice": "明るい声", "seed": 7})
            engine.load()
            try:
                pcm, voice = engine.synthesize({"text": "テスト"})
                self.assertEqual(voice, "明るい声")
                self.assertGreater(len(pcm), 0)
                # 48 kHz の 300 サンプル → 16 kHz の 100 サンプル（16 bit）
                self.assertEqual(len(pcm), 200)
            finally:
                engine.close()


class EngineManagerTest(unittest.TestCase):
    def test_speechcpp_entries(self) -> None:
        ids = {e.id: e for e in engine_manager._ENGINE_DEFINITIONS}
        self.assertEqual(ids["asr-speechcpp"].setup_steps, ("SpeechCppRuntime", "SpeechCppAsrModel"))
        self.assertEqual(ids["tts-irodori"].setup_steps, ("SpeechCppRuntime", "SpeechCppIrodoriModel"))
        self.assertEqual(ids["tts-irodori-pytorch"].setup_venv_script, "scripts/setup_irodori_venv.ps1")

    def test_linux_probe_drops_exe(self) -> None:
        with mock.patch.object(engine_manager.os, "name", "posix"):
            self.assertEqual(engine_manager._posixify_probe_path("vendor/speechcpp/speech.exe"), "vendor/speechcpp/speech")


class ComboFieldTest(unittest.TestCase):
    """設定の宣言テーブルの combo（保存値は itemData）。"""

    def _combo(self):
        from PySide6.QtWidgets import QApplication, QComboBox

        QApplication.instance() or QApplication([])
        combo = QComboBox()
        for value, label in speechcpp.ASR_MODELS:
            combo.addItem(label, value)
        return combo

    def test_roundtrip(self) -> None:
        holder = types.SimpleNamespace(asr_speechcpp_model_combo=self._combo())
        MMDAgentController._apply_app_config_field(holder, {"asr_speechcpp_model": "reazonspeech-v2"}, "asr_speechcpp_model")
        self.assertEqual(MMDAgentController._read_app_config_field(holder, "asr_speechcpp_model"), "reazonspeech-v2")
        # 知らない値は既定へ
        MMDAgentController._apply_app_config_field(holder, {"asr_speechcpp_model": "no-such"}, "asr_speechcpp_model")
        self.assertEqual(MMDAgentController._read_app_config_field(holder, "asr_speechcpp_model"), "haas-ja-v1.0")


if __name__ == "__main__":
    unittest.main()
