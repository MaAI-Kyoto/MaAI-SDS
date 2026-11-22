"""audio.cpp（ggml の音声合成サーバー）を立ち上げ、OpenAI 互換の `/v1/audio/speech` で合成する部品。

LFM2.5-Audio・Irodori（audio.cpp 版）・Qwen3-TTS-CPP が使う。Supertonic は従来どおり自前で立ち上げている。

- サーバーは 1 つのエンジンにつき 1 つ（ポートで分ける）。設定のホスト・ポートで既にサーバーが応答していれば
  それを使う（Docker では compose の `audiocpp` サービス）。
- 少しずつ受け取るとき（ストリーミング）は、モデルを `"mode": "streaming"` で登録し、要求に
  `"stream": true, "stream_format": "audio", "response_format": "pcm"` を付ける。16 bit PCM（LE）がチャンクで届く。
- 返ってくる PCM のサンプリング周波数はヘッダーに無い。まとめて受け取ったときの
  `X-AudioCPP-Audio-Duration-Ms`（音声の長さ）とサンプル数から求める（`probe_sample_rate`）。
"""

from __future__ import annotations

import http.client
import json
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .. import paths as _paths


class AudioCppServer:
    """audio.cpp のサーバー 1 つ。`ensure()` で応答を確かめ、無ければ立ち上げる。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        models: list[dict[str, Any]],
        server_exe: str = "",
        backend: str = "cuda",
        threads: int = 8,
        name: str = "audiocpp",
        start_timeout_sec: float = 240.0,
    ) -> None:
        self.host = (host or "127.0.0.1").strip() or "127.0.0.1"
        self.port = int(port)
        self.models = models
        self.server_exe = _paths.resolve_audiocpp_server(server_exe or _paths.default_audiocpp_server())
        self.backend = backend
        self.threads = int(threads)
        self.name = name
        self.start_timeout_sec = float(start_timeout_sec)
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def healthy(self, timeout: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=timeout) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    def ensure(self) -> bool:
        """サーバーを使える状態にする。自分で立ち上げたら True、既にあったら False。"""
        if self.healthy(timeout=5.0):
            return False
        missing = [str(m["path"]) for m in self.models if not Path(str(m["path"])).exists()]
        if not self.server_exe.is_file() or missing:
            raise RuntimeError(
                f"{self.name}: audio.cpp のサーバーに接続できず、自分で立ち上げることもできません"
                f"（{self.base_url}）。GUI の「エンジン管理」で該当エンジンにチェックを入れて「適用」してください。"
                + (f" 実行ファイルがありません: {self.server_exe}。" if not self.server_exe.is_file() else "")
                + (f" モデルがありません: {', '.join(missing)}。" if missing else "")
            )
        config = {
            "host": self.host, "port": self.port, "backend": self.backend, "device": 0,
            "threads": self.threads, "lazy_load": False,
            "models": [{"task": "tts", "mode": "streaming", **m} for m in self.models],
        }
        config_file = Path(tempfile.gettempdir()) / f"maai-{self.name}-{self.port}.json"
        config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        self._proc = subprocess.Popen(
            [str(self.server_exe), "--config", str(config_file)],
            cwd=str(self.server_exe.parent),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + self.start_timeout_sec
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"{self.name}: audio.cpp のサーバーが終了しました (exit {self._proc.returncode})")
            if self.healthy():
                return True
            time.sleep(0.25)
        self.close()
        raise RuntimeError(f"{self.name}: audio.cpp のサーバーの起動がタイムアウトしました")

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except Exception:  # noqa: BLE001
                proc.kill()


class AudioCppSpeechClient:
    """`/v1/audio/speech` を呼ぶ。同じサーバーへの要求は 1 つずつ送る。"""

    def __init__(self, server: AudioCppServer, *, timeout_sec: float = 120.0) -> None:
        self.server = server
        self.timeout_sec = float(timeout_sec)
        self._lock = threading.Lock()

    def speech(
        self,
        body: dict[str, Any],
        *,
        on_chunk: Callable[[bytes], None] | None = None,
    ) -> tuple[bytes, float]:
        """合成して (16 bit PCM の LE, 音声の長さ ms) を返す。`on_chunk` を渡すと少しずつ受け取り、届くたびに呼ぶ。

        音声の長さはまとめて受け取ったときだけヘッダーにある（ストリーミングでは 0）。
        """
        payload = dict(body)
        if on_chunk is not None:
            payload.update({"stream": True, "stream_format": "audio", "response_format": "pcm"})
        else:
            payload["response_format"] = "pcm"
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with self._lock:
            conn = http.client.HTTPConnection(self.server.host, self.server.port, timeout=self.timeout_sec)
            try:
                conn.request("POST", "/v1/audio/speech", raw, {"Content-Type": "application/json"})
                response = conn.getresponse()
                if response.status != 200:
                    detail = response.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"audio.cpp error {response.status}: {detail[:400]}")
                pcm = bytearray()
                odd = b""
                while True:
                    data = response.read1(65536)
                    if not data:
                        break
                    pcm += data
                    if on_chunk is not None:
                        data = odd + data
                        cut = len(data) - (len(data) % 2)
                        data, odd = data[:cut], data[cut:]
                        if data:
                            on_chunk(bytes(data))
                duration_ms = float(response.getheader("X-AudioCPP-Audio-Duration-Ms") or 0.0)
            finally:
                conn.close()
        return bytes(pcm[: len(pcm) - (len(pcm) % 2)]), duration_ms

    def probe_sample_rate(self, body: dict[str, Any], fallback: int) -> int:
        """短い文をまとめて合成し、サンプル数と音声の長さからサンプリング周波数を求める（ウォームアップも兼ねる）。"""
        try:
            pcm, duration_ms = self.speech(body)
        except Exception:  # noqa: BLE001
            return int(fallback)
        samples = len(pcm) // 2
        if samples <= 0 or duration_ms <= 0:
            return int(fallback)
        rate = samples / (duration_ms / 1000.0)
        # 22050 / 24000 / 44100 / 48000 などの標準の値へ丸める
        for standard in (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000):
            if abs(rate - standard) / standard < 0.02:
                return standard
        return int(round(rate))
