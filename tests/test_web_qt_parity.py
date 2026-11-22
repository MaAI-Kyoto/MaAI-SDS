"""Web UI と PySide 版の動作をそろえるための回帰テスト（Qt/Web 突き合わせ監査の結果）。

PARITY-xx は監査で見つかった違いの番号。どれも「Qt と同じ値・同じ命令になること」を確かめる。
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from maai_sds.config_store import ConfigStore
from maai_sds.dialog import settings as dialog_settings
from maai_sds.gui.app_config_spec import APP_CONFIG_FIELDS
from maai_sds.mmd import default_paths
from maai_sds.web import parity
from maai_sds.web.obs_link import ObsRecordingSessions
from maai_sds.web.server import OrchestratorBridge, create_app


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_JS = REPO_ROOT / "web" / "app" / "app.js"
MANIFEST = Path(parity.__file__).with_name("ui_manifest.json")

# Qt でも画面のどこにも置かれていない（作るだけで使われない）部品。Web に無くても差は出ない。
QT_WIDGETS_NOT_ON_SCREEN = {
    "asr_chirp_location_edit",
    "asr_chirp_model_edit",
    "asr_chirp_project_id_edit",
    "asr_chirp_recognizer_edit",
}


class _FakeProcessService:
    def is_running(self) -> bool:
        return False

    def start(self, exec_path: Path, mdf_path: Path, env: dict):  # pragma: no cover
        raise AssertionError("parity tests must not start MMDAgent")

    def stop(self, timeout_sec: int = 3) -> None:
        return None


class _FakeBridge:
    def __init__(self) -> None:
        self.commands: list[dict] = []
        self.module_states: dict[str, str] = {}
        self.voice_state = "stopped"
        self.mmd_remote_connected = False
        self.mmd_remote_mode = ""

    def send(self, command: dict) -> None:
        self.commands.append(dict(command))


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _find_widget(manifest: dict, attr: str) -> dict:
    for item in _walk_dicts(manifest):
        if item.get("attr") == attr:
            return item
    raise AssertionError(f"widget not found: {attr}")


def _item_values(widget: dict) -> list[str]:
    return [
        str(item.get("value", "")) if isinstance(item, dict) else str(item)
        for item in widget.get("items", [])
    ]


def _js_function(source: str, name: str) -> str:
    match = re.search(rf"function {name}\(.*?\n\}}\n", source.replace("\r\n", "\n"), re.DOTALL)
    if match is None:
        raise AssertionError(f"function not found in app.js: {name}")
    return match.group(0)


class WebQtParityTests(unittest.TestCase):
    def _manifest(self) -> dict:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    def _make_client(self, initial: dict):
        temp = tempfile.TemporaryDirectory()
        config_path = Path(temp.name) / "controller_config.json"
        config_path.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")
        bridge = _FakeBridge()
        app = create_app(
            config_store=ConfigStore(config_path),
            bridge_override=bridge,
            mmd_process_service=_FakeProcessService(),
            platform="win32",
            docker=False,
            watch_mmd_process=False,
        )
        # lifespan に入らない（入るとダイアログワーカーが起動する）。
        client = TestClient(app)
        self.addCleanup(temp.cleanup)
        self.addCleanup(client.close)
        return config_path, bridge, client, app

    # ---- settings (manifest / load / save) --------------------------------

    def test_manifest_visible_labels_do_not_use_legacy_wording(self) -> None:
        manifest = self._manifest()
        visible_keys = ("title", "qt_title", "label", "text")
        visible_texts = [
            str(item[key])
            for item in _walk_dicts(manifest)
            for key in visible_keys
            if isinstance(item.get(key), str)
        ]
        denied = (
            "VAD backend",
            "Turn backend",
            "Pseudo streaming",
            "Torch intra threads",
            "SmartTurn threads",
            "CPU threads",
            "AudioInput設定",
            "AudioOutput設定",
            "frame_rate(Hz)",
            "frame_rate (Hz)",
            "inference_chunk_frames",
            "intra threads",
            "MAAI",
        )
        offenders = sorted(
            {text for text in visible_texts if any(old in text for old in denied)}
        )
        self.assertEqual(offenders, [])

    def test_parity_01_combo_items_expose_qt_item_data(self) -> None:
        """PARITY-01: Web の選択肢の値は Qt が保存する値（itemData）。"""
        manifest = self._manifest()
        expected = {
            "asr_sherpa_provider_combo": ["cpu", "cuda"],
            "asr_qwen3_onnx_provider_combo": ["cpu", "cuda"],
            "asr_parakeet_onnx_provider_combo": ["cpu", "cuda"],
            "asr_sensevoice_onnx_provider_combo": ["cpu", "cuda"],
            "asr_funasr_nano_onnx_provider_combo": ["cpu", "cuda"],
            "asr_cohere_onnx_provider_combo": ["cpu", "cuda"],
            "asr_nemotron_onnx_compute_combo": ["gpu-fp32", "cpu-int8"],
            "maai_backchannel_playback_combo": ["play", "play_no_inject", "silent"],
            "audio_input_wav_channel_combo": ["0", "1"],
            "audio_input_wav_aec_ref_combo": ["other_ch", "system"],
            "audio_output_mode_combo": ["mmd", "local", "tcp"],
            "mmd_blink_mode_combo": ["always", "dialog"],
        }
        actual = {attr: _item_values(_find_widget(manifest, attr)) for attr in expected}
        self.assertEqual(actual, expected)

    def test_parity_02_every_qt_config_widget_is_in_manifest(self) -> None:
        """PARITY-02: Qt の設定欄は（画面に無い部品を除き）すべて Web にもある。"""
        by_attr = {str(item.get("attr")) for item in _walk_dicts(self._manifest()) if item.get("attr")}
        missing = [
            (config_key, attr)
            for config_key, attr, *_rest in APP_CONFIG_FIELDS
            if attr not in by_attr and attr not in QT_WIDGETS_NOT_ON_SCREEN
        ]
        self.assertEqual(missing, [])

    def test_parity_03_empty_config_round_trip_does_not_crash(self) -> None:
        """PARITY-03: キーが欠けた設定でも、Web から保存できる。"""
        _path, _bridge, client, _app = self._make_client({})
        payload = client.get("/api/config").json()
        response = client.post("/api/config", json=payload)
        self.assertEqual(response.status_code, 200, response.text)

    def test_parity_04_default_provider_is_canonical(self) -> None:
        """PARITY-04: 既定値も実行時に使える値。"""
        merged = parity.merge_app_ui_defaults({})
        self.assertEqual(merged["asr_sherpa_provider"], "cpu")
        self.assertEqual(merged["asr_nemotron_onnx_compute"], "gpu-fp32")
        self.assertEqual(merged["maai_backchannel_playback"], "play")

    def test_parity_04b_label_values_are_repaired_on_load_and_save(self) -> None:
        """以前の Web UI が表示名のまま保存した値は、読み込み・保存で Qt の値に直る。"""
        initial = {
            "asr_qwen3_onnx_provider": "GPU (cuda)",
            "asr_nemotron_onnx_compute": "CPU (int8)",
            "maai_backchannel_playback": "再生しない（MaAI注入のみ）",
            "audio_input_wav_channel_index": "ch2",
            # 削除した「口パク方式」の古い保存値が残っていても読み込み・保存できる。
            "mmd_lipsync_mode": "RMS振幅 (簡易)",
        }
        path, _bridge, client, _app = self._make_client(initial)
        shown = client.get("/api/config").json()
        self.assertEqual(shown["asr_qwen3_onnx_provider"], "cuda")
        self.assertEqual(shown["maai_backchannel_playback"], "silent")
        self.assertEqual(client.post("/api/config", json=shown).status_code, 200)
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(stored["asr_qwen3_onnx_provider"], "cuda")
        self.assertEqual(stored["asr_nemotron_onnx_compute"], "cpu-int8")
        self.assertEqual(stored["maai_backchannel_playback"], "silent")
        self.assertEqual(stored["audio_input_wav_channel_index"], "1")
        settings = dialog_settings.build_start_voice_dialog_payload(stored)
        self.assertTrue(settings["maai_backchannel_inject_only"])
        self.assertNotIn("mmd_lipsync_mode", settings)
        self.assertNotIn("mmd_lipsync_mode", settings["worker_configs"]["audio_output"])

    def test_parity_04c_shipped_examples_store_canonical_values(self) -> None:
        for name in ("controller_config.example.json", "controller_config.docker.example.json"):
            with self.subTest(name=name):
                cfg = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
                self.assertEqual(parity.canonical_combo_values(cfg), cfg)

    def test_parity_07_tts_voice_save_updates_engine_alias(self) -> None:
        """PARITY-07: 話者を変えると、実行時に優先されるエンジン別キーも変わる。"""
        initial = {"tts_engine": "VOICEVOX-Core", "tts_voice": "8", "tts_voice_voicevox_core": "8"}
        path, _bridge, client, _app = self._make_client(initial)
        payload = client.get("/api/config").json()
        payload["tts_voice"] = "9"
        self.assertEqual(client.post("/api/config", json=payload).status_code, 200)
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(stored["tts_voice_voicevox_core"], "9")
        worker = dialog_settings.build_tts_worker_config(stored, "VOICEVOX-Core")
        self.assertEqual(worker["voice"], "9")

    def test_parity_08_numeric_fields_keep_qt_json_types(self) -> None:
        """PARITY-08: Qt が数値で保存するキーは数値のまま。"""
        initial = {
            "mmd_blink_interval_mean_sec": 4.0,
            "mmd_burst_snd_frames": 13,
            "voice_dialog_idle_take_turn_sec": 2.0,
            "voice_dialog_scheduled_exit_sec": 0.0,
        }
        path, _bridge, client, _app = self._make_client(initial)
        payload = client.get("/api/config").json()
        payload.update(
            mmd_blink_interval_mean_sec="5.5",
            mmd_burst_snd_frames="17",
            voice_dialog_idle_take_turn_sec="3.25",
            voice_dialog_scheduled_exit_sec="60",
        )
        self.assertEqual(client.post("/api/config", json=payload).status_code, 200)
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(stored["mmd_blink_interval_mean_sec"], 5.5)
        self.assertIsInstance(stored["mmd_burst_snd_frames"], int)
        self.assertEqual(stored["mmd_burst_snd_frames"], 17)
        self.assertIsInstance(stored["voice_dialog_idle_take_turn_sec"], float)
        self.assertIsInstance(stored["voice_dialog_scheduled_exit_sec"], float)

    def test_parity_09_legacy_string_false_renders_unchecked(self) -> None:
        """PARITY-09: "false" などの文字列を Qt と同じく偽として表示する。"""
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("Boolean(currentValue(w))", source)
        body = _js_function(source, "configBool")
        self.assertIn("'1', 'true', 'yes', 'on'", body)

    def test_parity_13_mmdagent_paths_detected_like_qt(self) -> None:
        """PARITY-13: EXE / MDF が未設定なら Qt と同じ場所を探して埋める。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "Release").mkdir()
            exe = base / "Release" / "MMDAgent-EX.exe"
            mdf = base / "Release" / "MMDAgent-EX_gene.mdf"
            exe.write_text("", encoding="utf-8")
            mdf.write_text("", encoding="utf-8")
            filled = default_paths.fill_default_mmdagent_paths({"exec_path": "", "mdf_path": "Z:/missing.mdf"}, base)
            self.assertEqual(filled["exec_path"], str(exe))
            self.assertEqual(filled["mdf_path"], str(mdf))
            kept = default_paths.fill_default_mmdagent_paths({"exec_path": str(mdf)}, base)
            self.assertEqual(kept["exec_path"], str(mdf))  # 存在するパスは置き換えない
        nothing = default_paths.fill_default_mmdagent_paths({"exec_path": ""}, Path("Z:/no/such/dir"))
        self.assertEqual(nothing["exec_path"], "")

    # ---- commands --------------------------------------------------------

    def test_parity_05_web_voice_stop_keeps_preexisting_modules(self) -> None:
        """PARITY-05: 対話開始前から動いていた ASR / MaAI は停止時に残す（Qt と同じ）。"""
        _path, bridge, client, _app = self._make_client({})
        bridge.module_states.update(asr="running", maai="stopped")
        self.assertEqual(client.post("/api/command", json={"type": "start_voice_dialog"}).status_code, 200)
        bridge.module_states.update(maai="running")  # 対話が MaAI を起動した
        self.assertEqual(client.post("/api/command", json={"type": "stop_voice_dialog"}).status_code, 200)
        stop = next(c for c in bridge.commands if c.get("type") == "stop_voice_dialog")
        self.assertIs(stop.get("keep_asr"), True)
        self.assertIs(stop.get("keep_maai"), False)
        self.assertEqual(bridge.commands[-1], {"type": "tts_done", "playback_aborted": True})

    def test_parity_06_asr_compare_starts_compare_worker(self) -> None:
        """PARITY-06: 比較モードでは比較ワーカー（asr_compare）を起動する。"""
        cfg = {
            "asr_compare_mode_enabled": True,
            "asr_compare_top_engine": "sherpa-onnx",
            "asr_compare_bottom_engine": "faster-whisper",
        }
        commands = dialog_settings.expand_web_ui_commands(
            cfg, {"type": "start_module", "module": "asr"},
            assume_mmd_running=False, force_browser_mic=False,
        )
        modules = [cmd.get("module") for cmd in commands]
        self.assertIn("asr_compare", modules)
        self.assertNotIn("asr", modules)
        compare = next(c for c in commands if c.get("module") == "asr_compare")
        self.assertEqual(compare["config"]["compare_bottom_engine"], "faster-whisper")
        same = {**cfg, "asr_compare_bottom_engine": "sherpa-onnx"}
        with self.assertRaises(ValueError):
            dialog_settings.expand_web_ui_commands(same, {"type": "start_module", "module": "asr"})

    def _bridge_ops(self, commands: list[dict]) -> list[str]:
        return [
            str(cmd.get("payload", {}).get("op"))
            for cmd in commands
            if cmd.get("type") == "send_to_tts_bridge"
        ]

    def test_parity_12_aivis_start_manages_external_engine(self) -> None:
        """PARITY-12: AivisSpeech の起動は Engine 本体も起動する。"""
        cfg = {"tts_engine": "AivisSpeech", "aivisspeech_port": "10104", "aivisspeech_exe_path": "C:/a/run.exe"}
        commands = dialog_settings.expand_web_ui_commands(
            cfg, {"type": "start_module", "module": "tts", "engine": "AivisSpeech"},
            assume_mmd_running=False, force_browser_mic=False,
        )
        self.assertEqual(self._bridge_ops(commands), ["configure", "launch_aivisspeech"])
        self.assertEqual(commands[0]["payload"]["aivisspeech_exe_path"], "C:/a/run.exe")
        self.assertEqual(commands[1]["payload"]["port"], 10104)

    def test_parity_12_external_engines_stop_with_module(self) -> None:
        """PARITY-12: VOICEVOX / AivisSpeech の停止は外部プロセスも止める。"""
        for engine, op in (("VOICEVOX", "stop_voicevox"), ("AivisSpeech", "stop_aivisspeech")):
            with self.subTest(engine=engine):
                commands = dialog_settings.expand_web_ui_commands(
                    {"tts_engine": engine}, {"type": "stop_module", "module": "tts", "engine": engine},
                    assume_mmd_running=False, force_browser_mic=False,
                )
                self.assertIn(op, self._bridge_ops(commands))
        commands = dialog_settings.expand_web_ui_commands(
            {}, {"type": "stop_module", "module": "tts", "engine": "VOICEVOX-Core"},
        )
        self.assertEqual(self._bridge_ops(commands), [])

    # ---- OBS -------------------------------------------------------------

    def test_parity_14_obs_recording_refcount_like_qt(self) -> None:
        """PARITY-14: 録画は手動・音声対話・MaAI の参照カウント（Qt の obs_mixin と同じ）。"""
        calls: list[str] = []
        cfg = {"obs_remote_enabled": True, "obs_remote_link_voice": True, "obs_remote_link_maai": False}
        sessions = ObsRecordingSessions(
            lambda: cfg, lambda _msg: None,
            start=lambda _c: (calls.append("start") or True, ""),
            stop=lambda _c: (calls.append("stop") or True, ""),
        )
        sessions.acquire_if_linked("voice")
        sessions.acquire_if_linked("maai")  # 連動オフなので取らない
        sessions.acquire("manual")
        sessions.release("voice")
        self.assertEqual(calls, ["start"])  # 手動が残っているので止めない
        sessions.release("manual")
        self.assertEqual(calls, ["start", "stop"])
        sessions.release("manual")  # 持っていない参照は何もしない
        self.assertEqual(calls, ["start", "stop"])

    def test_parity_14_voice_dialog_links_obs_and_unmasks_password(self) -> None:
        initial = {"obs_remote_enabled": True, "obs_remote_link_voice": True, "obs_remote_password": "secret"}
        _path, _bridge, client, _app = self._make_client(initial)
        seen: list[tuple[str, str]] = []

        def fake_start(cfg):
            seen.append(("start", str(cfg.get("obs_remote_password"))))
            return True, ""

        def fake_stop(cfg):
            seen.append(("stop", str(cfg.get("obs_remote_password"))))
            return True, ""

        with mock.patch.object(parity, "obs_start", fake_start), mock.patch.object(parity, "obs_stop", fake_stop), \
                mock.patch("maai_sds.web.obs_link.web_parity.obs_start", fake_start), \
                mock.patch("maai_sds.web.obs_link.web_parity.obs_stop", fake_stop):
            sessions = client.app.state.obs_sessions
            sessions._start, sessions._stop = fake_start, fake_stop
            client.post("/api/command", json={"type": "start_voice_dialog"})
            client.post("/api/command", json={"type": "stop_voice_dialog"})
            res = client.post("/api/obs/start", json={"obs_remote_password": "********"}).json()
            self.assertTrue(res["ok"], res)
        self.assertEqual(seen, [("start", "secret"), ("stop", "secret"), ("start", "secret")])

    # ---- events / display -----------------------------------------------

    def test_parity_10_bridge_turns_qt_events_into_web_events(self) -> None:
        """PARITY-10: Qt が表示するイベントは、Web へ届く形に変えて流す。"""
        bridge = OrchestratorBridge(ConfigStore(Path(tempfile.gettempdir()) / "unused_parity.json"))
        relayed: list[dict] = []
        bridge.loop = None
        with mock.patch.object(OrchestratorBridge, "_note_runtime_state", lambda self, ev: None):
            original = bridge._on_event

            def capture(ev):
                relayed.append(dict(ev))
                return original(ev)

            bridge._on_event = capture  # type: ignore[method-assign]
            bridge.module_states["tts"] = "running"
            capture({"type": "worker_disconnected", "worker": "tts"})
        self.assertEqual(bridge.module_states["tts"], "error")
        self.assertIn("切断", bridge.module_errors["tts"])
        self.assertIn("error", OrchestratorBridge._CONSOLE_IPC_TYPES)
        self.assertIn("interrupt_condition", OrchestratorBridge._CONSOLE_IPC_TYPES)
        handled = set(re.findall(r"case\s+'([^']+)'\s*:", APP_JS.read_text(encoding="utf-8")))
        self.assertIn("llm_test_tts_segment", handled)

    def test_parity_10b_voice_and_maai_stop_release_obs(self) -> None:
        bridge = OrchestratorBridge(ConfigStore(Path(tempfile.gettempdir()) / "unused_parity.json"))
        stopped: list[str] = []
        bridge.on_voice_stopped = lambda: stopped.append("voice")
        bridge.on_module_stopped = lambda m: stopped.append(m)
        bridge._on_event({"type": "status", "state": "running"})
        bridge._on_event({"type": "status", "state": "stopped"})
        bridge._on_event({"type": "module_started", "module": "maai"})
        bridge._on_event({"type": "module_stopped", "module": "maai"})
        self.assertEqual(stopped, ["voice", "maai"])

    def test_parity_11_asr_event_preserves_results_and_compare_panes(self) -> None:
        """PARITY-11: ASR 結果は Qt と同じく追記・partial 置換・比較欄振り分け。"""
        body = _js_function(APP_JS.read_text(encoding="utf-8"), "appendAsrResult")
        self.assertIn("asr_compare_top_result_edit", body)
        self.assertIn("asr_compare_bottom_result_edit", body)
        self.assertIn("-partial", body)
        self.assertIn("latency=", body)


if __name__ == "__main__":
    unittest.main()
