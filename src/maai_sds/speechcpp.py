"""speech.cpp（ggml の音声認識・音声合成の実行系）を `speech worker` で動かす。

speech.cpp は MIT ライセンスの C/C++ 実装（https://github.com/nyosegawa/speech.cpp）。
配布の実行ファイルを `vendor/speechcpp/` に置き（`setup_assets` の `SpeechCppRuntime`）、
モデルは `speech` が初回に Hugging Face から `speechcpp_models/` へ取ってくる。

`speech worker MODEL` は 1 つのモデルを読み込み、標準入出力で 1 行 1 つの JSON をやり取りする
（worker protocol 4。speech.cpp の docs/worker.md）。ポートを取らず、プロセスの寿命も
エンジンと同じにできるので、HTTP サーバー（`speech serve`）ではなくこちらを使う。

- 認識：`chunk`（16 bit PCM の base64）を送り、`transcribe` で締める → `end` に `text`
- 合成：`synthesize` を送る → `chunk` が届き、`end` で終わる

**GPU の選び方**：`--device gpu` や `auto` は「最初の GPU」で、内蔵 GPU のある PC ではそちらが
選ばれる（実測で Haas-ja が NVIDIA 30 ms・内蔵 GPU 240 ms・CPU 180 ms と、内蔵 GPU は CPU より遅い）。
そこで `speech devices` の一覧から種類が `gpu`（外付け）のものを選び、無ければ CPU にする。

**Docker**：Docker Desktop のコンテナでは NVIDIA の Vulkan ドライバーが使えないので、GPU 版のイメージには
CUDA 版をソースから作って入れる（`setup_assets.sh` の `build_speechcpp_cuda`）。CUDA の GPU も `speech devices` の
種類は `gpu` なので、選び方は同じ。

**Windows のパスの長さ**：モデルは `<置き場所>/<組織>--<リポジトリ>/<64 桁のハッシュ>/<ファイル>` に
置かれ、置き場所が深いと 260 文字を超えて取得に失敗する（`Windows error 3`）。置き場所は
データの置き場の直下（`speechcpp_models`）にしてある。
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import paths as _paths

VERSION = "0.11.1"
MODEL_DIR_NAME = "speechcpp_models"

# 選べるモデル。名前は `speech models` の NAME[:TYPE]。数値は awesome-japanese-speech の実測
# （RTX 5070 Ti Laptop。ASR は 1 発話、TTS は 1 文）。
ASR_MODELS: tuple[tuple[str, str], ...] = (
    ("haas-ja-v1.0", "Haas-ja v1.0（日本語。GPU 約 30〜40 ms・最も正確）"),
    ("reazonspeech-v2", "ReazonSpeech NeMo v2（日本語。GPU 約 60〜100 ms・放送の話し言葉に強い）"),
    ("parakeet-tdt_ctc-0.6b-ja", "Parakeet TDT-CTC 0.6B ja（日本語。GPU 約 40 ms）"),
    ("qwen3-asr-0.6b", "Qwen3-ASR 0.6B（多言語。GPU 約 95 ms・1.7B より速い）"),
    ("qwen3-asr-1.7b", "Qwen3-ASR 1.7B（多言語。GPU 約 170 ms・くだけた会話に強い）"),
)
DEFAULT_ASR_MODEL = "haas-ja-v1.0"
DEFAULT_IRODORI_MODEL = "irodori-tts-mf:q8_0"


def exe_name() -> str:
    return "speech.exe" if os.name == "nt" else "speech"


def default_exe() -> str:
    return f"vendor/speechcpp/{exe_name()}"


def resolve_exe(raw: str = "") -> Path:
    """設定の実行ファイルを解決する。空なら `vendor/speechcpp/` の中（コンテナではイメージ内が先）。"""
    text = (raw or "").strip()
    if text:
        path = Path(os.path.expandvars(text))
        return path if path.is_absolute() else _paths.resolve_under_repo(path)
    return _paths.resolve_under_repo(default_exe())


def model_dir() -> Path:
    """モデルの置き場所。`MAAI_SDS_SPEECHCPP_MODELS` で変えられる。"""
    env = os.environ.get("MAAI_SDS_SPEECHCPP_MODELS", "").strip()
    return Path(env) if env else _paths.data_root() / MODEL_DIR_NAME


# モデルのファイルは <置き場所>\<組織>--<リポジトリ>\<64 桁>\<ファイル>.lock まで約 160 文字伸びる。
# speech.cpp は \\?\ 付きのパスを受け付けない（試した）ので、置き場所を短くするしかない。
WINDOWS_MODEL_DIR_MAX = 95


def long_path_warning(root: Path | None = None) -> str:
    """Windows で置き場所が深すぎてモデルを取れないおそれがあれば、その案内を返す。"""
    root = root or model_dir()
    if os.name != "nt" or len(str(root)) <= WINDOWS_MODEL_DIR_MAX:
        return ""
    return (
        f"speech.cpp のモデルの置き場所のパスが長すぎます（{len(str(root))} 文字）: {root}。"
        "Windows の 260 文字の制限でモデルを取得できないことがあります。環境変数 MAAI_SDS_SPEECHCPP_MODELS に"
        " 短い場所（例 C:\\speechcpp_models）を指定するか、MaAI-SDS を浅いフォルダーに置いてください。"
    )


def _env() -> dict[str, str]:
    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    return dict(os.environ, SPEECH_MODEL_DIR=str(root))


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def require_exe(raw: str = "") -> Path:
    exe = resolve_exe(raw)
    if not exe.is_file():
        raise RuntimeError(
            f"speech.cpp の実行ファイルがありません: {exe}。GUI の「エンジン管理」で speech.cpp を使うエンジン"
            "（speech.cpp ASR / Irodori）にチェックを入れて「適用」してください。"
        )
    return exe


def parse_devices(text: str) -> list[tuple[str, str, str]]:
    """`speech devices` の出力を (名前, 種類, 説明) の並びにする。種類は gpu / igpu / cpu。"""
    devices: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[1] in ("gpu", "igpu", "cpu"):
            devices.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return devices


def list_devices(exe: Path) -> list[tuple[str, str, str]]:
    try:
        out = subprocess.run([str(exe), "devices"], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60, env=_env(), creationflags=_creationflags())
    except Exception:  # noqa: BLE001
        return []
    return parse_devices(out.stdout)


def choose_device(want: str, devices: list[tuple[str, str, str]]) -> str:
    """設定（gpu / cpu / デバイス名）から `--device` に渡す名前を決める。

    gpu は外付けの GPU（種類 gpu）の最初のもの。無ければ CPU（内蔵 GPU は CPU より遅かったので選ばない）。
    """
    key = (want or "gpu").strip()
    low = key.lower()
    if low == "cpu":
        return "cpu"
    if low in ("gpu", "auto", "cuda", ""):
        for name, kind, _desc in devices:
            if kind == "gpu":
                return name
        return "cpu"
    return key


class SpeechCppError(RuntimeError):
    pass


class SpeechCppWorker:
    """`speech worker MODEL` を 1 つ動かし、要求を 1 つずつ投げる。スレッドから呼んでよい。"""

    def __init__(
        self,
        model: str,
        *,
        device: str = "gpu",
        threads: int = 0,
        options: dict[str, object] | None = None,
        exe: str = "",
        on_log: Callable[[str], None] | None = None,
        ready_timeout_sec: float = 1800.0,
    ) -> None:
        self.model = (model or "").strip()
        if not self.model:
            raise SpeechCppError("speech.cpp のモデル名が空です")
        self.want_device = device
        self.threads = int(threads or 0)
        self.options = {k: v for k, v in (options or {}).items() if v is not None and v != ""}
        self.exe = require_exe(exe)
        self.on_log = on_log or (lambda _m: None)
        self.ready_timeout_sec = ready_timeout_sec
        self.device = ""
        self.info: dict = {}
        self._proc: subprocess.Popen | None = None
        self._out: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._stderr_tail: list[str] = []

    # ------------------------------------------------------------------ 起動・終了
    def start(self) -> "SpeechCppWorker":
        self.device = choose_device(self.want_device, list_devices(self.exe))
        args = [str(self.exe), "worker", self.model, "--device", self.device]
        if self.threads > 0:
            args += ["--threads", str(self.threads)]
        for key, value in self.options.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                args.append(f"{flag}={'true' if value else 'false'}")
            else:
                args += [flag, str(value)]
        self.on_log(f"speech.cpp {VERSION}: {self.model} を {self.device} で起動します（モデル: {model_dir()}）")
        warning = long_path_warning()
        if warning:
            self.on_log(warning)
        t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_env(), creationflags=_creationflags(),
        )
        threading.Thread(target=self._read_stdout, daemon=True, name="speechcpp-out").start()
        threading.Thread(target=self._read_stderr, daemon=True, name="speechcpp-err").start()
        msg = self._next(self.ready_timeout_sec, waiting="ready")
        if msg.get("type") == "fatal":
            self.close()
            raise SpeechCppError(f"speech.cpp が起動できません: {_error_text(msg)}")
        if msg.get("type") != "ready":
            self.close()
            raise SpeechCppError(f"speech.cpp の応答が想定外です: {msg}")
        self.info = msg.get("model") or {}
        self.on_log(
            f"speech.cpp: {self.model} ready in {time.perf_counter() - t0:.1f}s "
            f"device={self.info.get('device', self.device)}"
        )
        return self

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(5)
        except Exception:  # noqa: BLE001
            proc.kill()
            try:
                proc.wait(5)
            except Exception:  # noqa: BLE001
                pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def sample_rate(self) -> int:
        return int(self.info.get("sample_rate") or 0)

    # ------------------------------------------------------------------ 要求
    def transcribe(self, audio: np.ndarray, sample_rate: int, **options: object) -> str:
        """音声（float32 は -1〜1、int16 はそのまま）を書き起こす。"""
        pcm = _to_int16(audio)
        with self._lock:
            rid = self._new_id()
            self._send({"type": "chunk", "id": rid, "seq": 0, "pcm": base64.b64encode(pcm.tobytes()).decode("ascii")})
            self._send({"type": "transcribe", "id": rid, "sample_rate": int(sample_rate), **_clean(options)})
            msg = self._until_end(rid)
        return str(msg.get("text") or "").strip()

    def synthesize(
        self, text: str, *, on_chunk: Callable[[np.ndarray], None] | None = None, **options: object
    ) -> tuple[np.ndarray, int]:
        """文を合成し、(int16 の音声, サンプリング周波数) を返す。

        `on_chunk` を渡すと、届いた音声（int16）をその都度渡す（少しずつ再生するため）。
        """
        with self._lock:
            rid = self._new_id()
            self._send({"type": "synthesize", "id": rid, "text": text, **_clean(options)})
            chunks: list[np.ndarray] = []
            self._until_end(rid, chunks, on_chunk)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return audio, self.sample_rate

    def add_voice(self, name: str, path: str) -> None:
        with self._lock:
            rid = self._new_id()
            self._send({"type": "add_voice", "id": rid, "name": name, "path": path})
            self._until_end(rid)

    # ------------------------------------------------------------------ 内部
    def _new_id(self) -> str:
        return f"m{next(self._ids)}"

    def _send(self, obj: dict) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise SpeechCppError(f"speech.cpp が終了しています{self._tail()}")
        try:
            proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            proc.stdin.flush()
        except OSError as ex:
            raise SpeechCppError(f"speech.cpp へ送れません: {ex}{self._tail()}") from ex

    def _until_end(
        self,
        rid: str,
        chunks: list[np.ndarray] | None = None,
        on_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> dict:
        while True:
            msg = self._next(600.0, waiting=rid)
            if msg.get("id") != rid:
                if msg.get("type") == "error" and msg.get("id") is None:
                    raise SpeechCppError(f"speech.cpp: {_error_text(msg)}")
                continue
            kind = msg.get("type")
            if kind == "chunk" and chunks is not None:
                chunk = np.frombuffer(base64.b64decode(msg.get("pcm") or ""), dtype="<i2").astype(np.int16)
                chunks.append(chunk)
                if on_chunk is not None and chunk.size:
                    on_chunk(chunk)
            elif kind == "end":
                return msg
            elif kind == "error":
                raise SpeechCppError(f"speech.cpp: {_error_text(msg)}")
            elif kind == "cancelled":
                raise SpeechCppError("speech.cpp: 要求が取り消されました")

    def _next(self, timeout: float, *, waiting: str) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SpeechCppError(f"speech.cpp の応答がありません（{waiting}、{timeout:.0f} 秒）{self._tail()}")
            try:
                msg = self._out.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if self._proc is None or self._proc.poll() is not None:
                    raise SpeechCppError(f"speech.cpp が終了しました{self._tail()}") from None
                continue
            if msg is None:
                raise SpeechCppError(f"speech.cpp が終了しました{self._tail()}")
            return msg

    def _read_stdout(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.on_log(f"speech.cpp: 解釈できない出力: {line[:200]}")
                continue
            if isinstance(msg, dict) and msg.get("type") != "progress":
                self._out.put(msg)
        self._out.put(None)

    def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            self._stderr_tail = (self._stderr_tail + [line])[-20:]
            # 初回のモデル取得（数百 MB〜数 GB）の進み具合が出るので、ログへ流す
            self.on_log(f"speech.cpp: {line}")

    def _tail(self) -> str:
        return ("\n" + "\n".join(self._stderr_tail[-8:])) if self._stderr_tail else ""


def _to_int16(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio).reshape(-1)
    if x.dtype == np.int16:
        return np.ascontiguousarray(x)
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    return np.round(x * 32767.0).astype("<i2")


def _clean(options: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in options.items() if v is not None and v != ""}


def _error_text(msg: dict) -> str:
    err = msg.get("error") or {}
    if isinstance(err, dict):
        option = f"（{err.get('option')}）" if err.get("option") else ""
        return f"{err.get('code', 'error')}{option}: {err.get('message', '')}"
    return str(err or msg)
