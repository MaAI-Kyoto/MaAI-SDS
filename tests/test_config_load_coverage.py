"""設定の宣言テーブル（app_config_spec）に載せた欄は、保存だけでなく読み込みもされること。

保存（_collect_app_config_payload）は宣言テーブルを全部回すが、読み込み（_load_app_config）は欄ごとに
`_apply_app_config_field(cfg, "キー")` を書く作りなので、書き忘れると画面がいつも既定値で始まり、
保存のたびに設定ファイルの値が既定値で上書きされる（ターンを取る条件・雑音抑制で起きた）。
"""

from __future__ import annotations

import inspect
import re
import unittest

try:
    from maai_sds.gui.app_config_spec import APP_CONFIG_FIELDS
    from maai_sds.gui.config_io import AppConfigIOMixin
except ImportError:  # pragma: no cover - PySide6 が無い環境
    APP_CONFIG_FIELDS = None


@unittest.skipIf(APP_CONFIG_FIELDS is None, "PySide6 is not installed")
class ConfigLoadCoverageTest(unittest.TestCase):
    def test_every_declared_field_is_loaded(self) -> None:
        src = inspect.getsource(AppConfigIOMixin._load_app_config) + inspect.getsource(
            AppConfigIOMixin._apply_whisper_fields_from_cfg
        )
        literals = set(re.findall(r'"([a-z0-9_]+)"', src))
        missing = [row[0] for row in APP_CONFIG_FIELDS if row[0] not in literals]
        self.assertEqual(missing, [], "読み込み（_load_app_config）で読んでいない設定")


if __name__ == "__main__":
    unittest.main()
