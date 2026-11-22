"""`launch_asr`（ASR 単体・一括起動）は、音声対話の開始と同じ `build_asr_config_dict` で設定を組むこと。

以前は `launch_asr` がウィジェットから手で辞書を組んでいて、共通の関数にある設定
（torch_cpu_intra_threads、smart_turn_num_threads、nemotron_remote_*、SenseVoice などの ONNX 版の設定）
が抜けていた。先に一括起動した ASR はそのまま対話でも使われるので、これらの設定が効かなかった（2026-11-03）。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent / "src" / "maai_sds"


class TestLaunchAsrUsesSharedBuilder(unittest.TestCase):
    def test_launch_asr_builds_with_shared_function(self) -> None:
        ctrl = ast.parse((_PKG / "gui/controller.py").read_text(encoding="utf-8"))
        launch = next(n for n in ast.walk(ctrl) if isinstance(n, ast.FunctionDef) and n.name == "launch_asr")
        calls = {n.func.attr for n in ast.walk(launch) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn("_build_asr_config_dict", calls)
        hand_built = [n.lineno for n in ast.walk(launch) if isinstance(n, ast.Dict)
                      and any(isinstance(k, ast.Constant) and k.value == "asr_engine" for k in n.keys)]
        self.assertEqual(hand_built, [], f"launch_asr が ASR 設定を手で組んでいる（行 {hand_built}）")
        builder = next(n for n in ast.walk(ctrl) if isinstance(n, ast.FunctionDef) and n.name == "_build_asr_config_dict")
        self.assertIn("build_asr_config_dict", ast.unparse(builder))

    def test_launch_asr_has_no_unused_locals(self) -> None:
        ctrl = ast.parse((_PKG / "gui/controller.py").read_text(encoding="utf-8"))
        launch = next(n for n in ast.walk(ctrl) if isinstance(n, ast.FunctionDef) and n.name == "launch_asr")
        stored, used = set(), set()
        for n in ast.walk(launch):
            if isinstance(n, ast.Name):
                (stored if isinstance(n.ctx, ast.Store) else used).add(n.id)
        self.assertEqual(sorted(stored - used), [])


if __name__ == "__main__":
    unittest.main()
