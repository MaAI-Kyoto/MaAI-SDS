"""OBS WebSocket（v5）経由の遠隔録画。

以前は obsws-python を使っていたが、GPL-3.0-only なので MIT の本体からは外した。
使うのは「接続して 1 つ要求を送って閉じる」だけなので、プロトコルを直接話す。
手順は OBS WebSocket 5.x の仕様どおり:
Hello（op 0）を受ける → Identify（op 1、パスワードがあれば認証文字列）→ Identified（op 2）
→ Request（op 6）→ RequestResponse（op 7）。
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import uuid
from typing import Any

_RPC_VERSION = 1
_OP_HELLO = 0
_OP_IDENTIFY = 1
_OP_IDENTIFIED = 2
_OP_REQUEST = 6
_OP_REQUEST_RESPONSE = 7

# OBS が接続を閉じるときの理由コード（WebSocketCloseCode）のうち、利用者に意味があるもの。
_CLOSE_REASONS = {
    4008: "認証が必要です（OBS のパスワードを設定してください）",
    4009: "認証に失敗しました（パスワードが違います）",
    4010: "対応していない RPC バージョンです",
}


class ObsError(Exception):
    pass


def auth_string(password: str, salt: str, challenge: str) -> str:
    """OBS WebSocket 5.x の認証文字列: base64(sha256(base64(sha256(password + salt)) + challenge))。"""
    secret = base64.b64encode(hashlib.sha256((password + salt).encode("utf-8")).digest()).decode("ascii")
    return base64.b64encode(hashlib.sha256((secret + challenge).encode("utf-8")).digest()).decode("ascii")


def _closed_error(code: int | None) -> ObsError:
    return ObsError(_CLOSE_REASONS.get(code, f"OBS が接続を閉じました（code={code}）"))


def _recv_op(ws: Any, want_op: int) -> dict:
    import websocket  # websocket-client

    while True:
        try:
            # recv() は close フレームを空文字にしてしまい理由コードが取れないので recv_data を使う。
            opcode, data = ws.recv_data()
        except websocket.WebSocketConnectionClosedException as ex:
            raise _closed_error(None) from ex
        if opcode == websocket.ABNF.OPCODE_CLOSE:
            code = struct.unpack("!H", data[:2])[0] if len(data) >= 2 else None
            raise _closed_error(code)
        if opcode not in (websocket.ABNF.OPCODE_TEXT, websocket.ABNF.OPCODE_BINARY):
            continue
        msg = json.loads(data)
        if msg.get("op") == want_op:
            return msg.get("d") or {}
        # Event（op 5）などは読み捨てる。eventSubscriptions=0 なので通常は来ない。


def request(host: str, port: int, password: str, request_type: str, *, timeout: float = 15.0) -> dict:
    """OBS に 1 つ要求を送り、responseData を返す。失敗は ObsError。"""
    import websocket  # websocket-client

    ws = websocket.create_connection(f"ws://{host}:{int(port)}", timeout=timeout)
    try:
        hello = _recv_op(ws, _OP_HELLO)
        identify: dict[str, Any] = {"rpcVersion": _RPC_VERSION, "eventSubscriptions": 0}
        auth = hello.get("authentication")
        if auth:
            if not password:
                raise ObsError(_CLOSE_REASONS[4008])
            identify["authentication"] = auth_string(password, auth["salt"], auth["challenge"])
        ws.send(json.dumps({"op": _OP_IDENTIFY, "d": identify}))
        _recv_op(ws, _OP_IDENTIFIED)

        request_id = uuid.uuid4().hex
        ws.send(json.dumps({"op": _OP_REQUEST, "d": {"requestType": request_type, "requestId": request_id}}))
        while True:
            resp = _recv_op(ws, _OP_REQUEST_RESPONSE)
            if resp.get("requestId") == request_id:
                break
        status = resp.get("requestStatus") or {}
        if not status.get("result"):
            comment = status.get("comment") or ""
            raise ObsError(f"{request_type} が失敗しました（code={status.get('code')}）{comment}")
        return resp.get("responseData") or {}
    finally:
        try:
            ws.close()
        except Exception:
            pass


def _run(host: str, port: int, password: str, request_type: str, timeout: float) -> tuple[bool, str | None]:
    try:
        request(host, port, password or "", request_type, timeout=timeout)
        return True, None
    except Exception as ex:
        return False, str(ex)


def start_record(host: str, port: int, password: str, *, timeout: float = 15.0) -> tuple[bool, str | None]:
    return _run(host, port, password, "StartRecord", timeout)


def stop_record(host: str, port: int, password: str, *, timeout: float = 15.0) -> tuple[bool, str | None]:
    return _run(host, port, password, "StopRecord", timeout)


def test_connection(host: str, port: int, password: str, *, timeout: float = 5.0) -> tuple[bool, str | None]:
    return _run(host, port, password, "GetVersion", timeout)
