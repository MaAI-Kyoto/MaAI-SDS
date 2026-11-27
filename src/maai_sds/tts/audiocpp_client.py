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
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .. import paths as _paths

# --- 自分で立ち上げたサーバーの後始末 -------------------------------------------------------------
# 以前は、アプリを閉じても audio.cpp のサーバーが残り、次に起動したときに（設定が変わっていても）
# 残ったサーバーをそのまま使っていた。スレッド数を直しても効かない、という形で実機で踏んだ。
# - Windows では Job Object に入れ、立ち上げたプロセス（TTS ワーカー）が終われば、強制終了でもサーバーも終わる。
# - 立ち上げたときの設定と PID を記録し、設定が違うサーバーが残っていれば止めてから立ち上げ直す。
#   記録の無いサーバー（Docker の compose や手で立てたもの）は今までどおりそのまま使う。

_job_handle: Any = None


def _bind_to_this_process(proc: subprocess.Popen) -> None:
    """Windows: proc を、このプロセスが終わると一緒に終わる Job Object に入れる。失敗しても続ける。"""
    global _job_handle
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        if _job_handle is None:
            class _IoCounters(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class _BasicLimit(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class _ExtendedLimit(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return
            info = _ExtendedLimit()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            # 9 = JobObjectExtendedLimitInformation
            if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                return
            _job_handle = job  # 閉じない（このプロセスが終わると閉じ、中のプロセスも終わる）
        kernel32.AssignProcessToJobObject(_job_handle, wintypes.HANDLE(int(proc._handle)))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _pid_file(config_file: Path) -> Path:
    return config_file.with_suffix(".pid")


def _is_audiocpp_process(pid: int) -> bool:
    """pid が audio.cpp のサーバーか（PID の使い回しで別のプロセスを止めないため）。"""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            return "audiocpp" in out.lower()
        cmdline = Path(f"/proc/{pid}/cmdline")
        return cmdline.is_file() and b"audiocpp" in cmdline.read_bytes().lower()
    except Exception:  # noqa: BLE001
        return False


def _stop_pid(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal

            os.kill(pid, signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass


def reuse_running_server(*, base_url: str, config_file: Path, config: dict[str, Any], name: str) -> bool:
    """応答しているサーバーをそのまま使ってよいか。

    自分が前に立ち上げたもの（設定と PID の記録がある）で設定が違えば止めて False を返す（立ち上げ直す）。
    記録が無い・読めないもの（Docker の compose や手で立てたもの）はそのまま使う。
    """
    pid_path = _pid_file(config_file)
    try:
        stored = json.loads(config_file.read_text(encoding="utf-8"))
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:  # noqa: BLE001
        return True
    if stored == config:
        return True
    if not _is_audiocpp_process(pid):
        return True
    print(f"[{name}] 設定の違う audio.cpp のサーバー（PID {pid}）が残っているので止めて立ち上げ直します", flush=True)
    _stop_pid(pid)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=1.0):
                pass
        except Exception:  # noqa: BLE001
            break
        time.sleep(0.25)
    try:
        pid_path.unlink()
    except OSError:
        pass
    return False


def launch_server(*, server_exe: Path, config_file: Path, config: dict[str, Any], cwd: Path) -> subprocess.Popen:
    """設定を書いてサーバーを立ち上げ、このプロセスと一緒に終わるようにし、PID を記録する。"""
    config_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.Popen(
        [str(server_exe), "--config", str(config_file)],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _bind_to_this_process(proc)
    try:
        _pid_file(config_file).write_text(str(proc.pid), encoding="utf-8")
    except OSError:
        pass
    return proc


def backend_for_server(server_exe: Path) -> str:
    """置いてある audio.cpp サーバーの種類から backend（cuda / vulkan / cpu）を決める。

    audio.cpp は Windows 用に CUDA・Vulkan・CPU 版を配っている。NVIDIA の GPU が無い PC（内蔵 GPU だけ）では
    Vulkan 版が速い（Iris Xe で Supertonic 3 の 23 文字が 0.64 秒。CPU 版 4 スレッドは 2.5 秒）。
    フォルダー名に vulkan / cpu があればそれ、隣の FLAVOR.txt が cpu / vulkan / cuda で始まればそれ、無ければ cuda。
    セットアップ（setup_assets.ps1）は、どの種類でも audio.cpp/build/windows-cuda-release/bin/ に置き、
    FLAVOR.txt に種類を書く（既定のパスを 1 つにしておくため）。
    """
    parts = [p.lower() for p in server_exe.parts]
    for name in ("vulkan", "cpu"):
        if any(name in p for p in parts[-4:]):
            return name
    flavor = server_exe.parent / "FLAVOR.txt"
    try:
        head = flavor.read_text(encoding="utf-8-sig").strip().lower()
    except OSError:
        head = ""
    for name in ("vulkan", "cpu", "cuda"):
        if head.startswith(name):
            return name
    return "cuda"


def default_threads(backend: str) -> int:
    """audio.cpp サーバーの CPU スレッド数の既定。

    Vulkan・CPU 版で 8 にすると、4 コアの PC では MaAI などが CPU を使っている間にスレッドの待ち合わせで止まり、
    6 文字の合成が 0.6 秒 → 20 秒以上になった（Iris Xe・i5-1135G7 で実測）。Vulkan は計算が GPU なので 2 で足りる。
    """
    if backend == "cuda":
        return 8
    if backend == "vulkan":
        return 2
    return max(1, min(4, (os.cpu_count() or 4) // 2))


class AudioCppServer:
    """audio.cpp のサーバー 1 つ。`ensure()` で応答を確かめ、無ければ立ち上げる。

    backend が空なら置いてあるサーバーの種類から、threads が 0 なら backend から決める。
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        models: list[dict[str, Any]],
        server_exe: str = "",
        backend: str = "",
        threads: int = 0,
        name: str = "audiocpp",
        start_timeout_sec: float = 240.0,
    ) -> None:
        self.host = (host or "127.0.0.1").strip() or "127.0.0.1"
        self.port = int(port)
        self.models = models
        self.server_exe = _paths.resolve_audiocpp_server(server_exe or _paths.default_audiocpp_server())
        self.backend = (backend or "").strip().lower() or backend_for_server(self.server_exe)
        self.threads = int(threads or 0) if int(threads or 0) > 0 else default_threads(self.backend)
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
        config = {
            "host": self.host, "port": self.port, "backend": self.backend, "device": 0,
            "threads": self.threads, "lazy_load": False,
            "models": [{"task": "tts", "mode": "streaming", **m} for m in self.models],
        }
        config_file = Path(tempfile.gettempdir()) / f"maai-{self.name}-{self.port}.json"
        if self.healthy(timeout=5.0) and reuse_running_server(
            base_url=self.base_url, config_file=config_file, config=config, name=self.name
        ):
            return False
        missing = [str(m["path"]) for m in self.models if not Path(str(m["path"])).exists()]
        if not self.server_exe.is_file() or missing:
            raise RuntimeError(
                f"{self.name}: audio.cpp のサーバーに接続できず、自分で立ち上げることもできません"
                f"（{self.base_url}）。GUI の「エンジン管理」で該当エンジンにチェックを入れて「適用」してください。"
                + (f" 実行ファイルがありません: {self.server_exe}。" if not self.server_exe.is_file() else "")
                + (f" モデルがありません: {', '.join(missing)}。" if missing else "")
            )
        self._proc = launch_server(
            server_exe=self.server_exe, config_file=config_file, config=config, cwd=self.server_exe.parent
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
