"""LLM ワーカーの要求処理の回帰テスト。

先読みを取り消した直後に通常の要求を出すと、新しい要求が、取り消した要求の最初の
トークン待ちが終わるまで始まらなかった（実測で最大約 300 ms の上乗せ）。要求ごとに
スレッドで処理し、新しい要求はすぐ始まることを確かめる。
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import maai_sds.llm.worker as llm_worker  # noqa: E402


class _FakeClient:
    def __init__(self, *_a, **_k) -> None:
        self.sent: list[dict] = []
        self.lock = threading.Lock()
        self.on_event = None
        self.on_disconnected = None

    def start_reader(self, on_event, on_disconnected=None) -> None:
        self.on_event = on_event
        self.on_disconnected = on_disconnected

    def send(self, payload: dict) -> None:
        with self.lock:
            self.sent.append(dict(payload, _t=time.perf_counter()))

    def events(self, etype: str, rid: str) -> list[dict]:
        with self.lock:
            return [p for p in self.sent if p.get("type") == etype and p.get("request_id") == rid]


class _SlowFirstTokenService:
    """"slow" の要求は最初のトークンまで 1 秒かかる。その他はすぐ返す。"""

    def __init__(self, *_a, **_k) -> None:
        pass

    def run_chat(self, *, messages, on_chunk, is_cancelled, **_k):
        slow = any("slow" in m.get("content", "") for m in messages)
        time.sleep(1.0 if slow else 0.0)
        if not is_cancelled():
            on_chunk("はい。")
        return SimpleNamespace(assistant_text="はい。", ttft_sec=0.0, total_sec=0.0)


def _start(rid: str, text: str) -> dict:
    return {"type": "start", "request_id": rid, "backend": "LM Studio", "model": "m",
            "messages": [{"role": "user", "content": text}], "stream": True}


class TestRequestsDoNotWaitForEachOther(unittest.TestCase):
    def test_new_request_starts_while_cancelled_one_waits_for_first_token(self) -> None:
        client = _FakeClient()
        stop = threading.Event()
        with patch.object(llm_worker, "JsonLineDuplexClient", lambda *a, **k: client), \
             patch.object(llm_worker, "connect_worker_client", lambda *a, **k: True), \
             patch.object(llm_worker, "finish_worker", lambda *a, **k: None), \
             patch.object(llm_worker, "LLMService", _SlowFirstTokenService), \
             patch.object(sys, "argv", ["llm_worker", "--host", "127.0.0.1", "--port", "1"]):
            # stdin が空だとワーカーは止まるので、stop まで待つイテレータに差し替える
            with patch.object(sys, "stdin", _Blocking(stop)):
                t = threading.Thread(target=llm_worker.main, daemon=True)
                t.start()
                deadline = time.time() + 5
                while client.on_event is None and time.time() < deadline:
                    time.sleep(0.01)
                t0 = time.perf_counter()
                client.on_event(_start("prefetch", "slow"))
                time.sleep(0.05)
                client.on_event({"type": "cancel", "request_id": "prefetch"})
                client.on_event(_start("normal", "fast"))
                while not client.events("done", "normal") and time.time() < deadline:
                    time.sleep(0.01)
                first_chunk = client.events("chunk", "normal")
                stop.set()
                if client.on_disconnected:
                    client.on_disconnected()
                t.join(timeout=3)

        self.assertTrue(first_chunk, "通常の要求に応答がない")
        self.assertLess(first_chunk[0]["_t"] - t0, 0.5, "取り消した要求の最初のトークン待ちを待っている")


class _Blocking:
    """for raw in sys.stdin: が stop まで待つようにする。"""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop

    def __iter__(self):
        self.stop.wait()
        return iter(())


if __name__ == "__main__":
    unittest.main()
