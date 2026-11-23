"""設定画面の使い勝手の直し（エンジンの画面の並び・エンジン管理・選択式の欄・数値の欄）。"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from maai_sds import engine_manager  # noqa: E402
from maai_sds.engine_manager import EngineInfo  # noqa: E402

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLineEdit
except ImportError:  # pragma: no cover - GUI extra が無い環境
    QApplication = None

_REPO = Path(__file__).resolve().parents[1]


def _eng(eid, extra=None, installed=False, steps=(), gb=None, runtime=None):
    return EngineInfo(eid, eid, extra, None, installed=installed, setup_steps=steps,
                      download_gb=gb, shared_runtime=runtime)


class EnginePlanTest(unittest.TestCase):
    def test_checked_and_missing_is_installed_unchecked_and_present_is_removed(self) -> None:
        engines = [
            _eng("asr-a", extra="asr-x", installed=False, gb=0.7),
            _eng("asr-b", extra="asr-y", installed=True),
            _eng("tts-c", installed=True, steps=("C",)),  # extra が無いものは外しても何も消えない
        ]
        plan = engine_manager.plan_engine_changes(engines, ["asr-a"])
        self.assertEqual(plan["install"], ["asr-a"])
        self.assertEqual(plan["remove"], ["asr-b"])
        self.assertEqual(plan["download_gb"], 0.7)

    def test_shared_extra_is_not_removed_while_another_checked_engine_uses_it(self) -> None:
        engines = [_eng("asr-sherpa", extra="asr-sherpa", installed=True),
                   _eng("asr-qwen3-onnx", extra="asr-sherpa", installed=True)]
        plan = engine_manager.plan_engine_changes(engines, ["asr-sherpa"])
        self.assertEqual(plan["remove"], [])

    def test_shared_runtime_is_counted_once_and_only_when_missing(self) -> None:
        a = _eng("tts-a", installed=False, steps=("A",), gb=1.0, runtime="audiocpp")
        b = _eng("tts-b", installed=False, steps=("B",), gb=2.0, runtime="audiocpp")
        plan = engine_manager.plan_engine_changes([a, b], ["tts-a", "tts-b"])
        self.assertEqual(plan["download_gb"], round(3.0 + engine_manager.SHARED_RUNTIME_GB["audiocpp"], 1))
        present = _eng("tts-c", installed=True, steps=("C",), runtime="audiocpp")
        plan = engine_manager.plan_engine_changes([a, present], ["tts-a", "tts-c"])
        self.assertEqual(plan["download_gb"], 1.0)

    def test_every_engine_is_asr_or_tts(self) -> None:
        kinds = {engine_manager.engine_kind(e) for e in engine_manager.list_engines()}
        self.assertEqual(kinds, {"ASR", "TTS"})

    def test_web_api_returns_the_same_plan_as_qt(self) -> None:
        import tempfile

        from fastapi.testclient import TestClient

        from test_app_icon import ConfigStore, _FakeBridge, _FakeProcessService, create_app

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller_config.json"
            path.write_text("{}", encoding="utf-8")
            app = create_app(config_store=ConfigStore(path), bridge_override=_FakeBridge(),
                             mmd_process_service=_FakeProcessService(), platform="linux", docker=False,
                             watch_mmd_process=False, remote_orchestrator="")
            client = TestClient(app)
            try:
                listing = client.get("/api/engines").json()
                self.assertTrue(all(e["kind"] in ("ASR", "TTS") for e in listing["engines"]))
                self.assertIn("download_gb", listing["engines"][0])
                ids = [e["id"] for e in listing["engines"] if e["installed"]][:1]
                got = client.post("/api/engines/plan", json={"ids": ids}).json()
                want = engine_manager.plan_engine_changes(engine_manager.list_engines(), ids)
                self.assertEqual(got, want)
                self.assertEqual(client.post("/api/engines/plan", json={"ids": "x"}).status_code, 422)
            finally:
                client.close()


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class WidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_choice_combo_keeps_unknown_values_from_old_configs(self) -> None:
        from maai_sds.gui.choice_combo import ChoiceCombo

        combo = ChoiceCombo(["jp", "en"], "jp")
        self.assertEqual(combo.text(), "jp")
        combo.setText("jp_custom")  # 一覧に無い値でも失わない
        self.assertEqual(combo.text(), "jp_custom")
        combo.setText("")  # 空は無視（既定のまま）
        self.assertEqual(combo.text(), "jp_custom")

    def test_numeric_field_is_clamped_before_it_is_saved(self) -> None:
        from maai_sds.gui.numeric_fields import apply_numeric_fields

        edit = QLineEdit("0.5")
        saved: list[str] = []
        edit.editingFinished.connect(lambda: saved.append(edit.text()))  # 保存処理は先につながっている
        apply_numeric_fields(SimpleNamespace(asr_vad_on_threshold_edit=edit))
        edit.show()
        edit.setFocus()
        edit.clear()
        QTest.keyClicks(edit, "2.5abc")  # 文字は打てない
        self.assertEqual(edit.text(), "2.5")
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.assertEqual(edit.text(), "1")
        self.assertEqual(saved[-1], "1")
        self.assertIn("範囲：0〜1", edit.toolTip())


class ManifestTest(unittest.TestCase):
    def _manifest(self) -> dict:
        return json.loads((_REPO / "src/maai_sds/web/ui_manifest.json").read_text(encoding="utf-8"))

    def test_every_numeric_field_exists_and_reaches_the_web(self) -> None:
        from maai_sds.gui.numeric_fields import NUMERIC_FIELDS

        found: dict[str, object] = {}

        def walk(o):
            if isinstance(o, dict):
                if o.get("kind") == "line" and o.get("attr"):
                    found[o["attr"]] = o.get("numeric")
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(self._manifest())
        missing = [a for a in NUMERIC_FIELDS if not found.get(a)]
        self.assertEqual(missing, [])

    def test_maai_model_details_are_folded_on_the_web_too(self) -> None:
        win = next(w for w in self._manifest()["windows"] if w["id"] == "maai_settings_window")
        labels = {row["label"]: row.get("detail") for g in win["groups"] for row in g["rows"]}
        self.assertTrue(labels.get("モード"))
        self.assertTrue(labels.get("FPS"))
        self.assertIsNone(labels.get("MaAIターンしきい値"))

    def test_engine_pages_come_first_in_the_settings_list(self) -> None:
        from maai_sds.gui.settings_hub import HUB_PAGES

        section, items = HUB_PAGES[0]
        self.assertEqual(section, "エンジン")
        self.assertEqual([a for a, _ in items],
                         ["asr_test_window", "llm_test_window", "tts_test_window", "maai_settings_window"])
        app_js = (_REPO / "web/app/app.js").read_text(encoding="utf-8")
        self.assertIn("['エンジン', ['asr_test_window', 'llm_test_window', 'tts_test_window', 'maai_settings_window']]",
                      app_js)


if __name__ == "__main__":
    unittest.main()
