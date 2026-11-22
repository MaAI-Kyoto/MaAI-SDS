"""アプリのアイコン（Qt の窓・Web の favicon）のテスト。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from maai_sds.config_store import ConfigStore  # noqa: E402
from maai_sds.gui import app_icon  # noqa: E402
from maai_sds.web.server import WEB_APP_DIR, create_app  # noqa: E402

APP = QApplication.instance() or QApplication([])


class _FakeProcessService:
    def find_running(self, *_args, **_kwargs):
        return None


class _FakeBridge:
    module_states: dict = {}
    voice_state = "stopped"
    mmd_remote_connected = False
    mmd_remote_mode = ""

    def send(self, command: dict) -> None:
        pass

    def emit_event(self, event: dict) -> None:
        pass


class AppIconTest(unittest.TestCase):
    def test_qt_icon_has_every_size(self) -> None:
        icon = app_icon.app_icon()
        self.assertFalse(icon.isNull())
        sizes = {s.width() for s in icon.availableSizes()}
        self.assertEqual(sizes, set(app_icon.ICON_SIZES))

    def test_missing_files_give_an_empty_icon_instead_of_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(app_icon.app_icon(Path(tmp)).isNull())

    def test_web_serves_the_same_icon_and_pages_link_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller_config.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            app = create_app(config_store=ConfigStore(path), bridge_override=_FakeBridge(),
                             mmd_process_service=_FakeProcessService(), platform="linux", docker=False,
                             watch_mmd_process=False, remote_orchestrator="")
            client = TestClient(app)
            try:
                ico = client.get("/favicon.ico")
                svg = client.get("/icon.svg")
            finally:
                client.close()
        self.assertEqual(ico.status_code, 200)
        self.assertEqual(ico.headers["content-type"], "image/x-icon")
        self.assertEqual(svg.status_code, 200)
        self.assertIn("image/svg+xml", svg.headers["content-type"])
        for page in ("index.html", "plot.html"):
            html = (WEB_APP_DIR / page).read_text(encoding="utf-8")
            self.assertIn('href="/icon.svg"', html, page)
            self.assertIn('href="/favicon.ico"', html, page)


if __name__ == "__main__":
    unittest.main()
