"""対話サーバーが音声フレームを ASR / MaAI へ転送するときに送出時刻を落とさないことと、compose が全ポートを loopback に出すこと。"""

from __future__ import annotations

import queue
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from maai_sds.dialog.runtime import VoiceDialogRuntime

_REPO = Path(__file__).resolve().parents[1]


class AudioFrameForwardingTest(unittest.TestCase):
    def test_emitted_at_reaches_asr_and_maai_queues(self) -> None:
        # ASR は emitted_at を端から端までの遅延（latency_sec）の起点にする。転送で落とすと受信時刻が起点になる。
        fake = SimpleNamespace(
            _asr_audio_q=queue.Queue(),
            _asr_compare_audio_q=queue.Queue(),
            _maai_audio_q=queue.Queue(),
            _emit=lambda payload: None,
        )
        payload = {"audio": [0.0] * 320, "sample_start": 0, "sample_end": 320, "ts_start": 1.0, "ts_end": 1.02, "emitted_at": 123.5}
        VoiceDialogRuntime.handle_audio_frame(fake, "audio_frame", payload)  # type: ignore[arg-type]
        for q in (fake._asr_audio_q, fake._asr_compare_audio_q, fake._maai_audio_q):
            self.assertEqual(q.get_nowait()["emitted_at"], 123.5)


class ComposePortsTest(unittest.TestCase):
    def test_every_published_port_is_bound_to_loopback(self) -> None:
        # 各サーバーには認証が無いので、ホストの全アドレス（0.0.0.0）には出さない
        for name in ("docker-compose.yml", "docker-compose.qwen-asr.yml"):
            published = self._ports_entries((_REPO / name).read_text(encoding="utf-8"))
            self.assertTrue(published, name)
            for p in published:
                self.assertTrue(p.startswith("127.0.0.1:"), f"{name}: {p}")

    @staticmethod
    def _ports_entries(text: str) -> list[str]:
        """`ports:` の直下に並ぶ `- "..."` だけを集める（command などの引数は拾わない）。"""
        out: list[str] = []
        ports_indent: int | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if stripped == "ports:":
                ports_indent = indent
                continue
            if ports_indent is not None:
                if indent <= ports_indent:
                    ports_indent = None
                elif stripped.startswith("-"):
                    m = re.match(r'-\s*"?([^"#\s]+)"?', stripped)
                    if m:
                        out.append(m.group(1))
        return out


if __name__ == "__main__":
    unittest.main()
