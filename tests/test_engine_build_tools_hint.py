"""Kokoro・Tsukasa を入れるときのビルドツールの案内（engine_manager.build_tools_hint）。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from maai_sds import engine_manager as em


class BuildToolsHintTest(unittest.TestCase):
    def test_not_windows(self) -> None:
        self.assertIsNone(em.build_tools_hint(["tts-kokoro"], platform="posix"))

    def test_default_extras_need_no_compiler(self) -> None:
        with patch.object(em, "_msvc_available", return_value=False), patch.object(em.shutil, "which", return_value=None):
            self.assertIsNone(em.build_tools_hint(["gui", "asr-sherpa", "tts-voicevox", "cu128"], platform="nt"))

    def test_missing_tools_are_named(self) -> None:
        with patch.object(em, "_msvc_available", return_value=False), patch.object(em.shutil, "which", return_value=None):
            hint = em.build_tools_hint(["gui", "tts-kokoro", "tts-tsukasa"], platform="nt")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("tts-kokoro, tts-tsukasa", hint)
        self.assertIn("Build Tools", hint)
        self.assertIn("CMake", hint)

    def test_only_cmake_missing(self) -> None:
        with patch.object(em, "_msvc_available", return_value=True), patch.object(em.shutil, "which", return_value=None):
            hint = em.build_tools_hint(["tts-tsukasa"], platform="nt")
        assert hint is not None
        self.assertIn("CMake", hint)
        self.assertNotIn("Build Tools", hint)

    def test_tools_present(self) -> None:
        with patch.object(em, "_msvc_available", return_value=True), patch.object(em.shutil, "which", return_value="cmake"):
            self.assertIsNone(em.build_tools_hint(["tts-kokoro"], platform="nt"))

    def test_compiler_extras_exist_in_pyproject(self) -> None:
        import tomllib
        from maai_sds import paths

        data = tomllib.loads(paths.repo_path("pyproject.toml").read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        for name in em.COMPILER_EXTRAS + ("gui", "web", "asr-sherpa", "tts-voicevox"):
            self.assertIn(name, extras)


if __name__ == "__main__":
    unittest.main()
