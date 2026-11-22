"""Shared RPC protocol between the app venv and the dedicated-venv TTS servers.

Used by irodori_server.py (.venv_irodori), which is launched by absolute path and imports
this module by bare name after putting its own directory on sys.path — so this module must
stay importable standalone and must not import anything from maai_sds.

Length-prefixed JSON frames (4-byte big-endian length + UTF-8 JSON), same framing as
asr/nemotron_asr.py's client/server bridge. PCM audio bytes travel base64-encoded inside
the JSON payload (synthesis is a single request/response, not streaming, so this is
simpler than the ASR bridge — no feed/finalize/reset session state).
"""

from __future__ import annotations

import base64
import json
import socket
import struct


def send_frame(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    payload = _recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def encode_pcm(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_pcm(b64: str) -> bytes:
    return base64.b64decode(b64.encode("ascii"))


class DedicatedVenvTTSClient:
    """Thin client for a TTS server running in a dedicated venv."""

    def __init__(self, host: str, port: int, timeout: float = 30.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self._sock = socket.create_connection((host, int(port)), timeout=self.timeout)
        self._sock.settimeout(self.timeout)

    def _rpc(self, payload: dict) -> dict:
        send_frame(self._sock, payload)
        resp = recv_frame(self._sock)
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(str(resp.get("error")))
        return resp

    def synthesize(self, req: dict) -> tuple[bytes, str]:
        """RPC {"op": "synthesize", **req} and return PCM bytes plus the server's
        resolved voice string (Irodori reports its caption in the same field)."""
        resp = self._rpc({"op": "synthesize", **req})
        return decode_pcm(str(resp.get("pcm_b64", ""))), str(resp.get("voice", ""))

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:  # noqa: BLE001
            pass
