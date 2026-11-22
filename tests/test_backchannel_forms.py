"""相槌の言い方の一覧（backchannel_forms.json）がリポジトリに入っていて、そこから選ばれる。

以前は .gitignore で除外されていて、clone した環境では一覧が無く、相槌がいつも「はい。」になっていた。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from maai_sds import paths
from maai_sds.dialog.orchestrator import DialogOrchestrator


class BackchannelFormsTest(unittest.TestCase):
    def test_forms_file_ships_with_the_repository(self) -> None:
        data = json.loads(paths.repo_path("backchannel_forms.json").read_text(encoding="utf-8"))
        texts = [e["text"] for e in data["entries"]]
        self.assertIn("うん。", texts)

    def test_backchannel_text_is_picked_from_the_forms(self) -> None:
        orch = DialogOrchestrator.__new__(DialogOrchestrator)
        orch.settings = {"maai_backchannel_dictionary_path": str(paths.repo_path("backchannel_forms.json"))}
        picked = {orch._select_backchannel_text() for _ in range(200)}
        self.assertTrue(picked <= {"うん。", "うんうん。", "うんうんうん。"}, picked)
        self.assertIn("うん。", picked)

    def test_relative_or_empty_path_is_read_from_the_repository(self) -> None:
        from maai_sds.dialog import settings as vds

        for value in ("backchannel_forms.json", ""):
            out = vds.build_dialog_orchestrator_settings({"maai_backchannel_dictionary_path": value})
            resolved = Path(out["maai_backchannel_dictionary_path"])
            self.assertTrue(resolved.is_absolute(), resolved)
            self.assertEqual(resolved, paths.repo_path("backchannel_forms.json"))


if __name__ == "__main__":
    unittest.main()
