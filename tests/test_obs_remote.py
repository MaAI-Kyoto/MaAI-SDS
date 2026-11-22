"""obs/remote.py（OBS WebSocket v5 の自前クライアント）を、偽の OBS サーバ相手に確かめる。"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from maai_sds.obs import remote  # noqa: E402

_SALT = "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI="
_CHALLENGE = "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY="


class _FakeObs:
    """Hello → Identify → Identified → Request → RequestResponse を返す最小の OBS。"""

    def __init__(self, password: str | None, *, fail_request: bool = False) -> None:
        self.password = password
        self.fail_request = fail_request
        self.requests: list[str] = []
        self.identify: dict | None = None
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5)

    def _run(self) -> None:
        import websockets

        async def handler(ws) -> None:
            hello: dict = {"obsWebSocketVersion": "5.5.0", "rpcVersion": 1}
            if self.password is not None:
                hello["authentication"] = {"salt": _SALT, "challenge": _CHALLENGE}
            await ws.send(json.dumps({"op": 0, "d": hello}))
            ident = json.loads(await ws.recv())
            self.identify = ident["d"]
            if self.password is not None:
                want = remote.auth_string(self.password, _SALT, _CHALLENGE)
                if ident["d"].get("authentication") != want:
                    await ws.close(code=4009, reason="Authentication failed.")
                    return
            await ws.send(json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}}))
            req = json.loads(await ws.recv())["d"]
            self.requests.append(req["requestType"])
            # 無関係なイベントを先に挟んでも読み飛ばせること
            await ws.send(json.dumps({"op": 5, "d": {"eventType": "Noise"}}))
            status = {"result": not self.fail_request, "code": 100 if not self.fail_request else 501}
            if self.fail_request:
                status["comment"] = "Output already active."
            await ws.send(json.dumps({"op": 7, "d": {
                "requestType": req["requestType"], "requestId": req["requestId"],
                "requestStatus": status, "responseData": {"obsVersion": "31.0.0"},
            }}))
            try:
                await ws.recv()
            except Exception:
                pass

        async def main() -> None:
            self._stop = asyncio.Event()
            async with websockets.serve(handler, "127.0.0.1", self.port):
                self._ready.set()
                await self._stop.wait()

        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        finally:
            self._loop.close()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)


class AuthStringTest(unittest.TestCase):
    def test_matches_obs_websocket_spec_example(self) -> None:
        # obs-websocket の protocol.md にある例（password=supersecretpassword）
        self.assertEqual(remote.auth_string("supersecretpassword", _SALT, _CHALLENGE),
                         "1Ct943GAT+6YQUUX47Ia/ncufilbe6+oD6lY+5kaCu4=")


class ObsRemoteTest(unittest.TestCase):
    def _server(self, *args, **kw) -> _FakeObs:
        srv = _FakeObs(*args, **kw)
        self.addCleanup(srv.close)
        return srv

    def test_no_auth_start_stop_and_version(self) -> None:
        srv = self._server(None)
        self.assertEqual(remote.start_record("127.0.0.1", srv.port, ""), (True, None))
        self.assertEqual(remote.stop_record("127.0.0.1", srv.port, ""), (True, None))
        self.assertEqual(remote.test_connection("127.0.0.1", srv.port, ""), (True, None))
        self.assertEqual(srv.requests, ["StartRecord", "StopRecord", "GetVersion"])
        self.assertNotIn("authentication", srv.identify)
        self.assertEqual(srv.identify["eventSubscriptions"], 0)

    def test_correct_password(self) -> None:
        srv = self._server("pw123")
        self.assertEqual(remote.start_record("127.0.0.1", srv.port, "pw123"), (True, None))

    def test_wrong_password_reports_auth_failure(self) -> None:
        srv = self._server("pw123")
        ok, err = remote.start_record("127.0.0.1", srv.port, "nope", timeout=5)
        self.assertFalse(ok)
        self.assertIn("認証に失敗", err)

    def test_missing_password_when_required(self) -> None:
        srv = self._server("pw123")
        ok, err = remote.test_connection("127.0.0.1", srv.port, "")
        self.assertFalse(ok)
        self.assertIn("認証が必要", err)

    def test_failed_request_status(self) -> None:
        srv = self._server(None, fail_request=True)
        ok, err = remote.start_record("127.0.0.1", srv.port, "")
        self.assertFalse(ok)
        self.assertIn("StartRecord", err)
        self.assertIn("already active", err)

    def test_connection_refused(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ok, err = remote.test_connection("127.0.0.1", port, "", timeout=2)
        self.assertFalse(ok)
        self.assertTrue(err)


if __name__ == "__main__":
    unittest.main()
