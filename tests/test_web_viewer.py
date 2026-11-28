"""ブラウザ表示の「MMDAgent-EX起動」：MMDAgent-EX Web のページとサンプル集を配り、シーンを指定して開く。"""

from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

from maai_sds.config_store import ConfigStore
from maai_sds.dialog import settings as vds
from maai_sds.mmd import web_viewer
from maai_sds.web.mmdagent import MmdAgentConfigurationError, WebMmdAgentController, WebRuntimePolicy


def _make_web(root: Path, *, with_launch: bool = True) -> Path:
    web = root / "mmdagent-ex-web"
    (web / "src").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>MMDAgent-EX Web</title>", encoding="utf-8")
    (web / "src" / "app.js").write_text("export const x = 1;\n", encoding="utf-8")
    if with_launch:
        (web / "src" / "launch.js").write_text("export const y = 2;\n", encoding="utf-8")
    return web


def _make_example(root: Path) -> Path:
    """README の置き方：example/ にサンプル集、example/maai_sds/ に MaAI-SDS の MDF と FST。"""
    example = root / "MMDAgent-EX" / "example"
    (example / "maai_sds").mkdir(parents=True)
    (example / "gene").mkdir()
    (example / ".git").mkdir()
    (example / "maai_sds" / "maai_sds.mdf").write_text("Plugin_Remote_EnableClient=true\n", encoding="utf-8")
    (example / "maai_sds" / "maai_sds_web.mdf").write_text("Plugin_Remote_Websocket_Port=9001\n", encoding="utf-8")
    (example / "maai_sds" / "maai_sds_web.fst").write_text("0 0 <eps> MODEL_ADD|gene|../gene/Gene.pmd\n", encoding="utf-8")
    (example / "gene" / "Gene.pmd").write_bytes(b"PMD-bytes")
    (example / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return example


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as res:  # noqa: S310 - 127.0.0.1 のテスト用サーバー
            return res.status, res.headers.get("Content-Type", ""), res.read()
    except urllib.error.HTTPError as ex:
        return ex.code, ex.headers.get("Content-Type", ""), b""


class PathsTest(unittest.TestCase):
    def test_default_web_dir_follows_the_readme(self) -> None:
        self.assertEqual(web_viewer.default_web_dir("win32"), Path(r"C:\mmdagent-ex-web"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Path, "home", lambda: Path(tmp)):
            self.assertEqual(web_viewer.default_web_dir("darwin"), Path(tmp) / "mmdagent-ex-web")

    def test_missing_or_old_web_folder_is_explained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(web_viewer.WebViewerError, "見つかりません"):
                web_viewer.resolve_web_dir(Path(tmp) / "none")
            old = _make_web(Path(tmp), with_launch=False)
            with self.assertRaisesRegex(web_viewer.WebViewerError, "git pull"):
                web_viewer.resolve_web_dir(old)
            (old / "src" / "launch.js").write_text("", encoding="utf-8")
            self.assertEqual(web_viewer.resolve_web_dir(str(old)), old)

    def test_the_web_mdf_next_to_the_tcp_one_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            example = _make_example(Path(tmp))
            web_mdf = example / "maai_sds" / "maai_sds_web.mdf"
            self.assertEqual(web_viewer.browser_mdf_for(example / "maai_sds" / "maai_sds.mdf"), web_mdf)
            self.assertEqual(web_viewer.browser_mdf_for(web_mdf), web_mdf)
            self.assertEqual(web_viewer.content_root_for(web_mdf), example.resolve())
            with self.assertRaisesRegex(web_viewer.WebViewerError, "設定されていません"):
                web_viewer.browser_mdf_for("")
            with self.assertRaisesRegex(web_viewer.WebViewerError, "見つかりません"):
                web_viewer.browser_mdf_for(example / "none.mdf")

    def test_file_list_skips_hidden_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            example = _make_example(Path(tmp))
            self.assertEqual(
                web_viewer.list_content_files(example),
                ["gene/Gene.pmd", "maai_sds/maai_sds.mdf", "maai_sds/maai_sds_web.fst", "maai_sds/maai_sds_web.mdf"],
            )

    def test_launch_url_has_the_scene_and_the_connection(self) -> None:
        url = web_viewer.launch_url(8010, "maai_sds/maai_sds_web.mdf", "ws://localhost:9001/chat")
        parsed = urllib.parse.urlsplit(url)
        self.assertEqual((parsed.netloc, parsed.path), ("localhost:8010", "/index.html"))
        query = dict(urllib.parse.parse_qsl(parsed.query))
        self.assertEqual(query, {
            "content": "/content/", "mdf": "maai_sds/maai_sds_web.mdf", "ws": "ws://localhost:9001/chat", "connect": "1",
        })

    def test_settings_defaults(self) -> None:
        self.assertEqual(vds.avatar_browser_http_port({}), 8010)
        self.assertEqual(vds.avatar_browser_http_port({"avatar_browser_http_port": "8123"}), 8123)
        self.assertEqual(vds.avatar_browser_http_port({"avatar_browser_http_port": "0"}), 8010)
        self.assertEqual(vds.avatar_browser_web_dir({"avatar_browser_web_dir": "  /x/web  "}), "/x/web")


class ServerTest(unittest.TestCase):
    def test_serves_the_page_the_content_and_its_file_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web = _make_web(Path(tmp))
            example = _make_example(Path(tmp))
            server = web_viewer.WebViewerServer(web, example, port=0)
            server.start()
            try:
                base = f"http://127.0.0.1:{server.port}"
                status, kind, body = _get(base + "/index.html?content=/content/")
                self.assertEqual(status, 200)
                self.assertTrue(kind.startswith("text/html"), kind)
                self.assertIn(b"MMDAgent-EX Web", body)
                # モジュールのスクリプトは JavaScript の種類でないと読まれない（Windows のレジストリに左右されない）
                status, kind, _ = _get(base + "/src/app.js")
                self.assertEqual((status, kind.split(";")[0]), (200, "text/javascript"))
                status, kind, body = _get(base + "/content/files.json")
                self.assertEqual(status, 200)
                self.assertIn("gene/Gene.pmd", json.loads(body)["files"])
                self.assertNotIn(".git/HEAD", json.loads(body)["files"])
                self.assertEqual(_get(base + "/content/gene/Gene.pmd")[2], b"PMD-bytes")
                # /content/ の下は、.. を付けてもコンテンツの外（ページのフォルダーやその上）へ出ない
                (Path(tmp) / "secret.txt").write_text("secret", encoding="utf-8")
                self.assertEqual(_get(base + "/content/../index.html")[0], 404)
                self.assertEqual(_get(base + "/content/%2e%2e/%2e%2e/secret.txt")[0], 404)
                self.assertEqual(_get(base + "/../secret.txt")[0], 404)
            finally:
                server.stop()
            self.assertFalse(server.is_running())

    def test_busy_port_is_explained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web = _make_web(Path(tmp))
            first = web_viewer.WebViewerServer(web, Path(tmp), port=0)
            first.start()
            try:
                second = web_viewer.WebViewerServer(web, Path(tmp), port=first.port)
                with self.assertRaisesRegex(web_viewer.WebViewerError, "ポート"):
                    second.start()
            finally:
                first.stop()


class LauncherTest(unittest.TestCase):
    def test_launch_reuses_the_server_and_stop_ends_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web = _make_web(Path(tmp))
            example = _make_example(Path(tmp))
            launcher = web_viewer.WebViewerLauncher()
            try:
                url = launcher.launch(web_dir=str(web), mdf_path=str(example / "maai_sds" / "maai_sds.mdf"),
                                      http_port=0, ws_port=19001)
                query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
                self.assertEqual(query["mdf"], "maai_sds/maai_sds_web.mdf")
                self.assertEqual(query["ws"], "ws://localhost:19001/chat")
                port = urllib.parse.urlsplit(url).port
                self.assertEqual(_get(f"http://127.0.0.1:{port}/content/maai_sds/maai_sds_web.mdf")[0], 200)
                again = launcher.launch(web_dir=str(web), mdf_path=str(example / "maai_sds" / "maai_sds_web.mdf"),
                                        http_port=0, ws_port=19001)
                self.assertEqual(urllib.parse.urlsplit(again).port, port)
            finally:
                self.assertTrue(launcher.stop())
            self.assertFalse(launcher.stop())


class _FakeProcessService:
    process = None

    def is_running(self) -> bool:
        return False

    def stop(self, timeout_sec: int = 3) -> None:  # pragma: no cover - 呼ばれない
        raise AssertionError("ブラウザ表示では MMDAgent-EX を止めない")


class _FakeBridge:
    def __init__(self) -> None:
        self.commands: list[dict] = []
        self.events: list[dict] = []
        self.module_states: dict = {}
        self.voice_state = "stopped"
        self.mmd_remote_connected = False
        self.mmd_remote_mode = ""

    def send(self, command: dict) -> None:
        self.commands.append(dict(command))

    def emit_event(self, event: dict) -> None:
        self.events.append(dict(event))


class WebUiTest(unittest.TestCase):
    def test_web_ui_start_returns_the_url_on_a_mac_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web = _make_web(Path(tmp))
            example = _make_example(Path(tmp))
            store = ConfigStore(Path(tmp) / "controller_config.json")
            store.save_raising({
                "avatar_display": "browser",
                "avatar_browser_web_dir": str(web),
                "avatar_browser_http_port": "0",
                "mdf_path": str(example / "maai_sds" / "maai_sds.mdf"),
            })
            # macOS のネイティブの Web UI：MMDAgent-EX は起動できないが、ブラウザ表示なら起動ボタンを使える
            policy = WebRuntimePolicy.detect(_FakeProcessService(), platform="darwin", docker=False)
            controller = WebMmdAgentController(store, _FakeBridge(), policy, watch_process=False)
            with mock.patch.object(vds, "avatar_browser_http_port", lambda _cfg: 0):
                self.assertTrue(controller.status()["launch_available"])
                result = controller.start()
            try:
                self.assertIn("browser_url", result)
                port = urllib.parse.urlsplit(result["browser_url"]).port
                self.assertEqual(_get(f"http://127.0.0.1:{port}/content/files.json")[0], 200)
            finally:
                stopped = controller.stop()
            self.assertIn("配信を止めました", stopped["message"])

            # MMDAgent-EX 本体の表示に戻すと、macOS では今までどおり起動できない
            store.save_raising({**store.load_file_only(), "avatar_display": "mmdagent"})
            self.assertFalse(controller.status()["launch_available"])

    def test_web_ui_start_reports_a_missing_web_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            example = _make_example(Path(tmp))
            store = ConfigStore(Path(tmp) / "controller_config.json")
            store.save_raising({
                "avatar_display": "browser",
                "avatar_browser_web_dir": str(Path(tmp) / "none"),
                "mdf_path": str(example / "maai_sds" / "maai_sds.mdf"),
            })
            policy = WebRuntimePolicy.detect(_FakeProcessService(), platform="win32", docker=False)
            controller = WebMmdAgentController(store, _FakeBridge(), policy, watch_process=False)
            with self.assertRaisesRegex(MmdAgentConfigurationError, "見つかりません"):
                controller.start()


class GuiLaunchTest(unittest.TestCase):
    """メイン画面の「MMDAgent-EX起動」：ブラウザ表示ならページを配ってブラウザで開き、「停止」で止める。

    画面全体は作らず（作る試験は test_example_config_is_the_default などにある）、起動と停止の処理だけを
    最小限の self で動かす。
    """

    def _fake_window(self, cfg: dict, mdf: Path):
        import types

        from maai_sds.gui.controller import MMDAgentController

        class _Label:
            def __init__(self) -> None:
                self.value = ""

            def setText(self, text: str) -> None:  # noqa: N802 - Qt と同じ名前
                self.value = text

            def text(self) -> str:
                return self.value

        win = types.SimpleNamespace(
            status_label=_Label(),
            process_service=types.SimpleNamespace(is_running=lambda: False),
            remote_stops=[],
        )
        win._widget_config_snapshot = lambda: dict(cfg)
        win._current_mdf_host_path = lambda warn=False: mdf
        win._avatar_in_browser = lambda: True
        win._send_remote_plugin_stop = lambda: win.remote_stops.append(True)
        for name in ("launch_mmdagent", "_launch_browser_avatar", "_stop_browser_avatar", "stop_mmdagent"):
            setattr(win, name, types.MethodType(getattr(MMDAgentController, name), win))
        return win

    def test_start_button_opens_the_browser_and_stop_ends_serving(self) -> None:
        from maai_sds.gui import controller as controller_module

        with tempfile.TemporaryDirectory() as tmp:
            web = _make_web(Path(tmp))
            example = _make_example(Path(tmp))
            cfg = {"avatar_display": "browser", "avatar_browser_web_dir": str(web)}
            win = self._fake_window(cfg, example / "maai_sds" / "maai_sds.mdf")
            opened: list[str] = []
            with mock.patch.object(vds, "avatar_browser_http_port", lambda _cfg: 0), \
                    mock.patch.object(controller_module.webbrowser, "open", lambda url: opened.append(url) or True):
                win.launch_mmdagent()
            try:
                self.assertEqual(len(opened), 1)
                query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(opened[0]).query))
                self.assertEqual((query["mdf"], query["ws"], query["connect"]),
                                 ("maai_sds/maai_sds_web.mdf", "ws://localhost:9001/chat", "1"))
                self.assertIn("開きました", win.status_label.text())
                port = urllib.parse.urlsplit(opened[0]).port
                self.assertEqual(_get(f"http://127.0.0.1:{port}/content/files.json")[0], 200)
            finally:
                win.stop_mmdagent()
            self.assertIn("配信を止めました", win.status_label.text())
            self.assertEqual(win.remote_stops, [True])
            with self.assertRaises(urllib.error.URLError):
                urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=2)  # noqa: S310

    def test_missing_web_folder_is_shown_instead_of_opening(self) -> None:
        from maai_sds.gui import controller as controller_module

        with tempfile.TemporaryDirectory() as tmp:
            example = _make_example(Path(tmp))
            cfg = {"avatar_display": "browser", "avatar_browser_web_dir": str(Path(tmp) / "none")}
            win = self._fake_window(cfg, example / "maai_sds" / "maai_sds.mdf")
            warnings: list[str] = []
            box = mock.MagicMock()
            box.warning.side_effect = lambda _parent, _title, text: warnings.append(text)
            with mock.patch.object(controller_module, "QMessageBox", box), \
                    mock.patch.object(controller_module.webbrowser, "open", side_effect=AssertionError("開かない")):
                win.launch_mmdagent()
            self.assertEqual(len(warnings), 1)
            self.assertIn("見つかりません", warnings[0])


if __name__ == "__main__":
    unittest.main()
