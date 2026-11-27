"""TTS エンジンの実装（tts/worker.py から移設）。

エンジンはどれも `synthesize` 系のインタフェースを持ち、
`tts/worker.ENGINE_FACTORIES` から選ばれる。
本文は 1 文字も変えずに移してある。
"""

from __future__ import annotations

import asyncio
import sys as _sys
import json
import math
import multiprocessing
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable
from .. import engine_manager, paths as _paths, speechcpp
from . import audiocpp_client
from .playback_service import BINARY_STREAM_ENGINES
from .venv_rpc import DedicatedVenvTTSClient
from pathlib import Path

import numpy as np
from ..audio.convert import (
    Pcm16BeLeadingSilenceGate,
    StreamingPcm16Resampler,
    pcm16le_to_pcm16be_16k,
    swap_pcm16_endian,
    pcmf32le_to_pcm16le,
    strip_leading_silence_pcm16be,
    wav_bytes_to_pcm16be_16k_mono,
)

_DEFAULT_VOICEVOX_ORT_FILE = (
    "voicevox_runtime/onnxruntime/lib/libvoicevox_onnxruntime.so"
    if _sys.platform != "win32"
    else "voicevox_runtime/onnxruntime/lib/voicevox_onnxruntime.dll"
)


# VOICEVOX-Core に渡す CPU スレッド数の上限。
#
# 以前は `max(multiprocessing.cpu_count(), 2)`（24 コア機で 24）だった。
# TTS / ASR / MaAI / GUI が**別プロセスで同時に動く**構成なので、
# TTS だけが全コアを要求すると oversubscription になる。
#
# 24 コア機での実測（CPU モード / 15 字 = 音声 2.18 秒 / 合成時間の中央値。
# 「負荷あり」は他に 10 プロセスが CPU を回している状態）:
#
#   threads   負荷なし    負荷あり   負荷時の最悪
#         4    0.931s     0.705s      1.187s
#         8    0.508s     0.416s      0.432s
#        12    0.457s     0.362s      0.530s
#        16    0.242s     0.356s      0.390s   <- 採用
#        20    0.246s     0.498s      1.775s
#        24    0.259s     0.495s      1.726s   <- 以前の設定
#
# 20 以上は負荷が掛かると 1.7 秒台へ跳ねる（「ときどき 200ms、ときどき 2 秒」の原因）。
# 16 は負荷なしで最速かつ負荷時も 0.39 秒に収まる。
#
# 16 という数字はこの 24 コア機での実測値。コア数がこれ以下の機械では
# 全コアを使う（そもそも譲る余地が無く、4 スレッドまで絞ると逆に 2 倍遅い）。
_VOICEVOX_CPU_THREADS_CAP = 16


def _pin_voicevox_cpu_away_from_maai() -> None:
    """Linux hybrid CPU で MaAI（P-core 0-7）と取り合うのを避ける。"""
    if not hasattr(os, "sched_setaffinity"):
        return
    from ..maai.cpu_affinity import linux_cpus_are_virtual

    if linux_cpus_are_virtual():
        return  # 仮想 CPU では MaAI 側も固定しないので、譲る相手がいない。
    try:
        ncpu = os.cpu_count() or 0
        if ncpu >= 16:
            e_cores = set(range(8, ncpu))
            os.sched_setaffinity(0, e_cores)
            print(
                f"[VOICEVOX-Core] CPU affinity={sorted(e_cores)} (0-7 は MaAI 用)",
                file=sys.stderr,
                flush=True,
            )
    except OSError:
        pass


def _resolve_voicevox_cpu_threads(raw: Any) -> int:
    """`cpu_num_threads` を決める。未指定なら上限とコア数の小さい方。"""
    cores = max(int(multiprocessing.cpu_count() or 2), 2)
    if raw is None or str(raw).strip() == "":
        return max(1, min(_VOICEVOX_CPU_THREADS_CAP, cores))
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return max(1, min(_VOICEVOX_CPU_THREADS_CAP, cores))
    return max(1, min(n, cores))


class VoicevoxCoreEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.default_voice = str(cfg.get("voice", "8")).strip() or "8"
        self.onnxruntime_file = str(
            cfg.get("onnxruntime_file", _DEFAULT_VOICEVOX_ORT_FILE)
        ).strip() or _DEFAULT_VOICEVOX_ORT_FILE
        self.openjtalk_dict = str(cfg.get("openjtalk_dict", "voicevox_runtime/dict/open_jtalk_dic_utf_8-1.11")).strip() or "voicevox_runtime/dict/open_jtalk_dic_utf_8-1.11"
        self.vvm_path = str(cfg.get("vvm_path", "voicevox_runtime/models/vvms/0.vvm")).strip() or "voicevox_runtime/models/vvms/0.vvm"
        # 設定は "AUTO" / "CPU" / "GPU (CUDA)" / "GPU (DirectML)"。voicevox_core が
        # 受けるのは AUTO / CPU / GPU の 3 つだけなので、ここで畳む。どちらの GPU に
        # なるかは、どのランタイム DLL が置かれているかで決まる（setup_assets.ps1 の
        # VoicevoxRuntime が同じ設定を見て取得する）。
        # 旧値 "GPU" もそのまま通す（CUDA 版を置いていた環境との互換）。
        raw_mode = str(cfg.get("acceleration_mode", "CPU")).strip() or "CPU"
        self.acceleration_mode = "GPU" if raw_mode.upper().startswith("GPU") else raw_mode.upper()
        self.timeout_sec = float(cfg.get("timeout_sec", 120.0) or 120.0)
        self.cpu_num_threads = _resolve_voicevox_cpu_threads(cfg.get("cpu_num_threads"))
        self._loop: asyncio.AbstractEventLoop | None = None
        self._synthesizer = None
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_path(path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str(_paths.resolve_under_repo(p).resolve())

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_runner, daemon=True, name="voicevox-core-loop")
        t.start()
        self._loop = loop
        return loop

    async def _load_async(self) -> None:
        engine_manager.require_module_installed("voicevox_core", "VOICEVOX-Core")
        from voicevox_core.asyncio import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile  # noqa: PLC0415

        # CUDA/cuDNN の DLL 探索パスを先に通す。新しい voicevox_onnxruntime（1.23.2 以降）の
        # CUDA プロバイダは cudnn64_9.dll を **外部依存** として要求する（1.17.3 は同梱
        # していた: プロバイダが 328MB → 137MB に減っている）。実体は venv の
        # nvidia-cudnn wheel にあるが、パスを通さないと
        #   Error loading "...providers_cuda.dll" which depends on
        #   "cudnn64_9.dll" which is missing. (Error 126)
        # で GPU が使えない。ASR 側は同じ仕組みを worker の先頭で通している。
        try:
            from ..asr import ort_windows_preload  # noqa: PLC0415

            ort_windows_preload._add_cuda_dll_dirs()
        except Exception:  # noqa: BLE001
            pass

        ort = await Onnxruntime.load_once(filename=self._resolve_path(self.onnxruntime_file))
        ojt = await OpenJtalk.new(self._resolve_path(self.openjtalk_dict))
        mode = self.acceleration_mode
        try:
            self._synthesizer = Synthesizer(
                ort,
                ojt,
                acceleration_mode=mode,
                cpu_num_threads=self.cpu_num_threads,
            )
        except Exception as ex:
            # CPU 専用の ORT（古い Linux runtime など）で GPU/AUTO を指定するとこの文言で落ちる。
            # setup_assets.sh は CUDA 版（libvoicevox_onnxruntime_providers_cuda.so 同梱）を
            # 取るので、GPU イメージの Docker ではそのまま CUDA で動く。
            if mode != "CPU" and "GPU機能をサポートすることができません" in str(ex):
                print(
                    "[VOICEVOX-Core] 同梱 ONNX Runtime に CUDA/DirectML が無いため CPU で起動します。",
                    file=sys.stderr,
                    flush=True,
                )
                self.acceleration_mode = "CPU"
                self._synthesizer = Synthesizer(
                    ort,
                    ojt,
                    acceleration_mode="CPU",
                    cpu_num_threads=self.cpu_num_threads,
                )
            else:
                raise
        async with await VoiceModelFile.open(self._resolve_path(self.vvm_path)) as model:
            await self._synthesizer.load_voice_model(model)

    def load(self) -> None:
        if self._synthesizer is not None:
            return
        _pin_voicevox_cpu_away_from_maai()
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(self._load_async(), loop)
        fut.result(timeout=max(30.0, self.timeout_sec))

    @staticmethod
    def _parse_speaker_id(raw: Any) -> int:
        text = str(raw).strip()
        if not text:
            raise ValueError("VOICEVOX-Core の speaker id が空です。")
        try:
            return int(text)
        except Exception as ex:
            raise ValueError("VOICEVOX-Core の speaker id は整数で指定してください。") from ex

    async def _synthesize_async(self, text: str, speaker_id: int, speed_scale: float = 1.0) -> bytes:
        if self._synthesizer is None:
            raise RuntimeError("VOICEVOX-Core synthesizer is not loaded")
        query = await self._synthesizer.create_audio_query(text, speaker_id)
        query.speed_scale = speed_scale
        # 既定の pre_phoneme_length は 0.1 秒。合成 WAV の先頭に無音が乗り、
        # 再生開始から発声までその分遅れる。
        query.pre_phoneme_length = 0.0
        wav = await self._synthesizer.synthesis(query, speaker_id)
        return bytes(wav)

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        if self._synthesizer is None:
            raise RuntimeError("VOICEVOX-Core model is not loaded")
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        speaker_id = self._parse_speaker_id(req.get("voice", self.default_voice))
        speed_scale = float(req.get("speed_scale", 1.0))
        if not math.isfinite(speed_scale) or not 0.5 <= speed_scale <= 2.0:
            raise ValueError("VOICEVOX-Core の話速は 0.5〜2.0 で指定してください。")
        loop = self._ensure_loop()
        with self._lock:
            fut = asyncio.run_coroutine_threadsafe(
                self._synthesize_async(text=text, speaker_id=speaker_id, speed_scale=speed_scale), loop
            )
            wav_bytes = fut.result(timeout=max(10.0, self.timeout_sec))
        audio_l16be = wav_bytes_to_pcm16be_16k_mono(wav_bytes)
        if not audio_l16be:
            raise RuntimeError("VOICEVOX-Core returned empty audio")
        return audio_l16be, str(speaker_id)

class CartesiaRealtimeEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model_id = str(cfg.get("model_id", "sonic-3")).strip() or "sonic-3"
        self.voice_id = str(cfg.get("voice_id", "f786b574-daa5-4673-aa0c-cbe3e8534c02")).strip()
        self.sample_rate = int(float(cfg.get("sample_rate", 16000) or 16000))
        self.sample_rate = max(8000, min(48000, self.sample_rate))
        self.api_key = str(cfg.get("api_key", "")).strip()
        self.api_key_env = str(cfg.get("api_key_env", "CARTESIA_API_KEY")).strip() or "CARTESIA_API_KEY"
        self.timeout_sec = float(cfg.get("timeout_sec", 30.0) or 30.0)
        self._lock = threading.Lock()
        self._cartesia_client = None

    def _resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        return str(os.environ.get(self.api_key_env, "")).strip()

    def load(self) -> None:
        resolved = self._resolve_api_key()
        if not resolved:
            raise RuntimeError(
                f"Cartesia APIキーが未設定です。UI入力または環境変数 {self.api_key_env} を設定してください。"
            )
        engine_manager.require_module_installed("cartesia", "Cartesia Realtime")
        from cartesia import Cartesia  # noqa: PLC0415 # pyright: ignore[reportMissingImports]

        self._cartesia_client = Cartesia(api_key=resolved)

    def synthesize_stream(self, req: dict[str, Any], on_chunk: Any) -> tuple[int, str]:
        if self._cartesia_client is None:
            raise RuntimeError("Cartesia client is not initialized")
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        model_id = str(req.get("model_id", self.model_id)).strip() or self.model_id
        voice_id = str(req.get("voice", "")).strip() or str(req.get("voice_id", self.voice_id)).strip() or self.voice_id
        if not voice_id:
            raise ValueError("Cartesia voice_id が未設定です。")
        sample_rate = int(float(req.get("sample_rate", self.sample_rate) or self.sample_rate))
        sample_rate = max(8000, min(48000, sample_rate))
        total_bytes = 0

        with self._lock:
            with self._cartesia_client.tts.websocket_connect() as connection:
                ctx = connection.context(
                    model_id=model_id,
                    voice={"mode": "id", "id": voice_id},
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": sample_rate,
                    },
                )
                ctx.push(text)
                ctx.no_more_inputs()
                for event in ctx.receive():
                    event_type = str(getattr(event, "type", "")).strip().lower()
                    if event_type == "chunk":
                        audio = getattr(event, "audio", None)
                        if not isinstance(audio, (bytes, bytearray)):
                            continue
                        pcm16le = pcmf32le_to_pcm16le(bytes(audio))
                        chunk = pcm16le_to_pcm16be_16k(pcm16le, sample_rate=sample_rate)
                        if not chunk:
                            continue
                        on_chunk(chunk)
                        total_bytes += len(chunk)
                    elif event_type == "done":
                        break
        if total_bytes <= 0:
            raise RuntimeError("Cartesia Realtime TTS returned empty audio")
        return total_bytes, voice_id

class RealtimeTTSKokoroSession:
    """KokoroEngine に対し push されたテキストをそのまま1回の synthesize に渡す（TTS内で文分割しない）。"""

    def __init__(
        self,
        *,
        request_id: str,
        kokoro_engine: Any,
        voice: str,
        on_chunk: Callable[[bytes], None],
        on_done: Callable[[int, str], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.request_id = request_id
        self.kokoro_engine = kokoro_engine
        self.voice = voice
        self.on_chunk = on_chunk
        self.on_done = on_done
        self.on_error = on_error
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        self._done_sent = False
        self._done_lock = threading.Lock()
        self._total_bytes = 0
        self._lead = Pcm16BeLeadingSilenceGate()

    def _emit_done_once(self) -> None:
        with self._done_lock:
            if self._done_sent:
                return
            self._done_sent = True
        self.on_done(self._total_bytes, self.voice)

    def _emit_error_once(self, message: str) -> None:
        with self._done_lock:
            if self._done_sent:
                return
            self._done_sent = True
        self.on_error(message)

    def _on_audio_chunk(self, chunk_pcm16le: bytes) -> None:
        if not isinstance(chunk_pcm16le, (bytes, bytearray)):
            return
        chunk_l16be = pcm16le_to_pcm16be_16k(bytes(chunk_pcm16le), sample_rate=24000)
        audible = self._lead.feed(chunk_l16be)
        if not audible:
            return
        self._total_bytes += len(audible)
        self.on_chunk(audible)

    def _drain_engine_queues(self) -> None:
        while True:
            try:
                chunk = self.kokoro_engine.queue.get_nowait()
            except queue.Empty:
                break
            self._on_audio_chunk(chunk)
        while True:
            try:
                self.kokoro_engine.timings.get_nowait()
            except queue.Empty:
                break

    def _worker_loop(self) -> None:
        try:
            while True:
                item = self._text_queue.get()
                if item is None:
                    break
                text = str(item or "").strip()
                if not text:
                    continue
                self._drain_engine_queues()
                self._lead.reset()
                ok = bool(self.kokoro_engine.synthesize(text))
                if not ok:
                    self._emit_error_once("KokoroEngine.synthesize が失敗しました")
                    return
                self._drain_engine_queues()
                tail = self._lead.flush()
                if tail:
                    self._total_bytes += len(tail)
                    self.on_chunk(tail)
        except Exception as ex:
            self._emit_error_once(str(ex))
            return
        self._emit_done_once()

    def start(self) -> None:
        threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"kokoro-realtimetts-{self.request_id}",
        ).start()

    def push_text(self, text: str) -> None:
        if not text:
            return
        with self._close_lock:
            if self._closed:
                raise RuntimeError("stream is already closed")
        self._text_queue.put(text)

    def end_input(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._text_queue.put(None)

    def cancel(self) -> None:
        # キャンセル時は session 側の done を抑制する。worker の cancel_stream ハンドラが
        # 明示的に done を emit するため、両方出ると done が二重になる（#24）。
        with self._done_lock:
            self._done_sent = True
        self.end_input()

class RealtimeTTSKokoroEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.default_voice = str(cfg.get("voice", "jf_alpha")).strip() or "jf_alpha"
        try:
            self.default_speed = float(cfg.get("speed", 1.0) or 1.0)
        except Exception:
            self.default_speed = 1.0
        self._kokoro_engine = None
        self._sessions: dict[str, RealtimeTTSKokoroSession] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        # インポート可否をロード時点で検証して、UI側へ早期に失敗を返す。
        engine_manager.require_module_installed("RealtimeTTS", "Kokoro (RealtimeTTS)")
        from RealtimeTTS import KokoroEngine  # noqa: PLC0415

        # worker起動中は同一KokoroEngineを使い回して再初期化コストを抑える。
        if self._kokoro_engine is None:
            self._kokoro_engine = KokoroEngine(voice=self.default_voice, default_speed=self.default_speed)

    def _resolve_session(self, request_id: str) -> RealtimeTTSKokoroSession:
        with self._lock:
            session = self._sessions.get(request_id)
        if session is None:
            raise RuntimeError("stream session not found")
        return session

    def start_stream(
        self,
        *,
        request_id: str,
        req: dict[str, Any],
        on_chunk: Callable[[bytes], None],
        on_done: Callable[[int, str], None],
        on_error: Callable[[str], None],
    ) -> str:
        voice = str(req.get("voice", "")).strip() or self.default_voice
        try:
            speed = float(req.get("speed", self.default_speed) or self.default_speed)
        except Exception:
            speed = self.default_speed
        speed = min(3.0, max(0.5, speed))
        print("[TTSWorker][INFO] Kokoro-RealtimeTTS プッシュ単位合成（TTS内分割なし）", flush=True)
        if self._kokoro_engine is None:
            self.load()
        kokoro_engine = self._kokoro_engine
        if kokoro_engine is None:
            raise RuntimeError("KokoroEngine の初期化に失敗しました。")
        try:
            if hasattr(kokoro_engine, "set_voice"):
                kokoro_engine.set_voice(voice)
        except Exception:
            pass
        try:
            if hasattr(kokoro_engine, "set_speed"):
                kokoro_engine.set_speed(speed)
        except Exception:
            pass

        def _on_done_internal(total_bytes: int, resolved_voice: str) -> None:
            with self._lock:
                self._sessions.pop(request_id, None)
            on_done(total_bytes, resolved_voice)

        def _on_error_internal(message: str) -> None:
            with self._lock:
                self._sessions.pop(request_id, None)
            on_error(message)

        session = RealtimeTTSKokoroSession(
            request_id=request_id,
            kokoro_engine=kokoro_engine,
            voice=voice,
            on_chunk=on_chunk,
            on_done=_on_done_internal,
            on_error=_on_error_internal,
        )
        with self._lock:
            if request_id in self._sessions:
                raise RuntimeError("stream session already exists")
            self._sessions[request_id] = session
        try:
            session.start()
        except Exception:
            with self._lock:
                self._sessions.pop(request_id, None)
            raise
        return voice

    def push_text(self, *, request_id: str, text: str) -> None:
        self._resolve_session(request_id).push_text(text)

    def end_stream(self, *, request_id: str) -> None:
        self._resolve_session(request_id).end_input()

    def cancel_stream(self, *, request_id: str) -> None:
        session = self._resolve_session(request_id)
        session.cancel()

    def synthesize_stream(self, req: dict[str, Any], on_chunk: Callable[[bytes], None]) -> tuple[int, str]:
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        done_event = threading.Event()
        error_holder: dict[str, str] = {"message": ""}
        voice_holder: dict[str, str] = {"voice": ""}
        total_holder: dict[str, int] = {"bytes": 0}
        request_id = f"sync-{int(time.time() * 1000)}-{threading.get_ident()}"

        def _on_chunk(chunk: bytes) -> None:
            if not chunk:
                return
            on_chunk(chunk)
            total_holder["bytes"] += len(chunk)

        def _on_done(total_bytes: int, resolved_voice: str) -> None:
            voice_holder["voice"] = resolved_voice
            total_holder["bytes"] = max(total_holder["bytes"], int(total_bytes))
            done_event.set()

        def _on_error(message: str) -> None:
            error_holder["message"] = message
            done_event.set()

        voice = self.start_stream(
            request_id=request_id,
            req=req,
            on_chunk=_on_chunk,
            on_done=_on_done,
            on_error=_on_error,
        )
        self.push_text(request_id=request_id, text=text)
        self.end_stream(request_id=request_id)
        if not done_event.wait(timeout=180.0):
            self.cancel_stream(request_id=request_id)
            raise RuntimeError("Kokoro-RealtimeTTS timed out")
        if error_holder["message"]:
            raise RuntimeError(error_holder["message"])
        if total_holder["bytes"] <= 0:
            raise RuntimeError("Kokoro-RealtimeTTS returned empty audio")
        return total_holder["bytes"], (voice_holder["voice"] or voice)

class TextStreamAdapterSession:
    def __init__(
        self,
        *,
        engine_name: str,
        engine: Any,
        request_id: str,
        base_inputs: dict[str, Any],
        emit_audio_chunk: Callable[[bytes], None],
        emit_done: Callable[[int, str], None],
        emit_error: Callable[[str], None],
    ) -> None:
        self.engine_name = engine_name
        self.engine = engine
        self.request_id = request_id
        self.base_inputs = dict(base_inputs)
        self.emit_audio_chunk = emit_audio_chunk
        self.emit_done = emit_done
        self.emit_error = emit_error
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self._cancelled = False
        self._lock = threading.Lock()
        self._done_sent = False
        self._done_lock = threading.Lock()
        self._total_bytes = 0
        self._voice = ""

    def _emit_done_once(self) -> None:
        with self._done_lock:
            if self._done_sent:
                return
            self._done_sent = True
        self.emit_done(int(self._total_bytes), str(self._voice))

    def _emit_error_once(self, message: str) -> None:
        with self._done_lock:
            if self._done_sent:
                return
            self._done_sent = True
        self.emit_error(str(message).strip() or "不明なエラー")

    def _process_one_text(self, text: str) -> None:
        src_inputs = dict(self.base_inputs)
        src_inputs["text"] = text
        if self.engine_name in BINARY_STREAM_ENGINES:
            gate = Pcm16BeLeadingSilenceGate()

            def _emit_without_lead(chunk: bytes, _gate: Pcm16BeLeadingSilenceGate = gate) -> None:
                audible = _gate.feed(chunk)
                if audible:
                    self.emit_audio_chunk(audible)
                    self._total_bytes += len(audible)

            _chunk_bytes, chunk_voice = self.engine.synthesize_stream(src_inputs, _emit_without_lead)
            tail = gate.flush()
            if tail:
                self.emit_audio_chunk(tail)
                self._total_bytes += len(tail)
            if (not self._voice) and chunk_voice:
                self._voice = str(chunk_voice)
            return
        audio_l16be, chunk_voice = self.engine.synthesize(src_inputs)
        audio_l16be = strip_leading_silence_pcm16be(audio_l16be)
        chunk_size = 24 * 1024
        for pos in range(0, len(audio_l16be), chunk_size):
            self.emit_audio_chunk(audio_l16be[pos : pos + chunk_size])
        self._total_bytes += int(len(audio_l16be))
        if (not self._voice) and chunk_voice:
            self._voice = str(chunk_voice)

    def _run(self) -> None:
        try:
            while True:
                text = self._text_queue.get()
                if text is None:
                    break
                with self._lock:
                    cancelled = self._cancelled
                if cancelled:
                    break
                if not str(text).strip():
                    continue
                self._process_one_text(str(text))
            self._emit_done_once()
        except Exception as ex:
            self._emit_error_once(str(ex))

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name=f"tts-adapter-{self.request_id}").start()

    def push_text(self, text: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("stream is already closed")
        self._text_queue.put(str(text or ""))

    def end_input(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._text_queue.put(None)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
        self.end_input()


class AivisSpeechEngine:
    """AivisSpeech Engine の VOICEVOX 互換 HTTP API を常駐 worker から利用する。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.host = str(cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        self.port = int(cfg.get("port", 10104))
        self.default_voice = str(cfg.get("voice", "888753761")).strip() or "888753761"
        self._lock = threading.Lock()

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _request(self, path: str, *, method: str = "GET", body: bytes | None = None, content_type: str | None = None) -> bytes:
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(self._base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as ex:
            detail = ex.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AivisSpeech API error {ex.code}: {detail}") from ex
        except urllib.error.URLError as ex:
            raise RuntimeError(f"AivisSpeech Engine に接続できません ({self._base_url}): {ex.reason}") from ex

    def load(self) -> None:
        self._request("/version")

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        voice = str(req.get("voice", "")).strip() or self.default_voice
        try:
            speaker = int(voice)
        except ValueError as ex:
            raise ValueError("AivisSpeech の話者IDは数値で指定してください") from ex
        query = urllib.parse.urlencode({"text": text, "speaker": speaker})
        with self._lock:
            query_json = self._request(f"/audio_query?{query}", method="POST")
            # VOICEVOX 互換 API も既定で先頭に 0.1 秒の無音（prePhonemeLength）を付ける。
            query_obj = json.loads(query_json.decode("utf-8"))
            query_obj["prePhonemeLength"] = 0.0
            wav = self._request(
                f"/synthesis?speaker={speaker}",
                method="POST",
                body=json.dumps(query_obj).encode("utf-8"),
                content_type="application/json",
            )
        return wav_bytes_to_pcm16be_16k_mono(wav), voice


# 置いてある audio.cpp サーバーの種類から backend を決める（設定の supertonic_backend が空のとき）。
# LFM2.5-Audio・Irodori（engines_audiocpp）と共通（audiocpp_client.backend_for_server）。
_audiocpp_backend_for = audiocpp_client.backend_for_server


class SupertonicEngine:
    """audio.cpp 常駐 server 経由の Supertonic 3。CLI 再起動を避ける。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.host = str(cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        self.port = int(cfg.get("port", 18080))
        self.model_id = str(cfg.get("model_id", "supertonic-3")).strip() or "supertonic-3"
        self.default_voice = str(cfg.get("voice", "F1")).strip() or "F1"
        self.steps = max(1, min(int(cfg.get("steps", 8)), 8))
        self.server_exe = _paths.resolve_audiocpp_server(str(cfg.get("server_exe", "")))
        raw_model = str(cfg.get("model_path", "")).strip() or (
            "audio.cpp/models/Supertonic-3-GGUF/supertonic-3-orig.gguf"
        )
        self.model_path = _paths.resolve_under_repo(raw_model)
        self.backend = str(cfg.get("backend", "") or "").strip().lower() or _audiocpp_backend_for(self.server_exe)
        # audio.cpp サーバーの CPU スレッド数。0 / 空なら backend で決める（audiocpp_client.default_threads）。
        raw_threads = int(cfg.get("threads", 0) or 0)
        self.threads = raw_threads if raw_threads > 0 else audiocpp_client.default_threads(self.backend)
        self._server_process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def load(self) -> None:
        config = {
            "host": self.host, "port": self.port, "backend": self.backend,
            "device": audiocpp_client.pick_device(self.server_exe, self.backend), "threads": self.threads,
            "lazy_load": False,
            "models": [{"id": self.model_id, "family": "supertonic", "path": str(self.model_path), "task": "tts", "mode": "offline",
                "default_request_options": {"voice_id": self.default_voice, "num_inference_steps": self.steps, "language": "ja", "speaking_rate": 1.0}}],
        }
        config_file = Path(tempfile.gettempdir()) / f"maai-supertonic-{self.port}.json"
        health_error: Exception | None = None
        try:
            with urllib.request.urlopen(self._base_url + "/health", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
            # 前に立ち上げたサーバーが設定の違うまま残っていれば、止めて立ち上げ直す
            if audiocpp_client.reuse_running_server(
                base_url=self._base_url, config_file=config_file, config=config, name="Supertonic"
            ):
                return
        except Exception as ex:
            health_error = ex
        if not (self.server_exe.is_file() and self.model_path.is_file()):
            raise RuntimeError(
                "Supertonic server に接続できません。audiocpp_server.exe と GGUF モデルのパスを設定するか、"
                f"{self._base_url} に事前起動してください: {health_error}"
            ) from health_error
        self._server_process = audiocpp_client.launch_server(
            server_exe=self.server_exe, config_file=config_file, config=config,
            cwd=self.server_exe.parent.parent.parent.parent,
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self._server_process.poll() is not None:
                raise RuntimeError(f"Supertonic server が終了しました (exit {self._server_process.returncode})")
            try:
                with urllib.request.urlopen(self._base_url + "/health", timeout=2):
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("Supertonic server の起動がタイムアウトしました")

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        voice = str(req.get("voice", "")).strip() or self.default_voice
        payload = json.dumps({
            "model": self.model_id,
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "options": {"voice_id": voice, "num_inference_steps": self.steps, "language": "ja", "speaking_rate": 1.0},
        }).encode("utf-8")
        request = urllib.request.Request(
            self._base_url + "/v1/audio/speech", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self._lock:
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    wav = response.read()
            except urllib.error.HTTPError as ex:
                detail = ex.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Supertonic server error {ex.code}: {detail}") from ex
        return wav_bytes_to_pcm16be_16k_mono(wav), voice

    def close(self) -> None:
        if self._server_process is not None and self._server_process.poll() is None:
            self._server_process.terminate()
        self._server_process = None


def _normalize_audiocpp_qwen_language(value: str) -> str:
    """audio.cpp Qwen3 が受け取る小文字の言語名へ正規化する。"""
    text = (value or "").strip().lower()
    if not text:
        return "japanese"
    aliases = {
        "ja": "japanese",
        "jp": "japanese",
        "japanese": "japanese",
        "en": "english",
        "english": "english",
        "zh": "chinese",
        "cn": "chinese",
        "chinese": "chinese",
    }
    return aliases.get(text, text)


class Qwen3TTSCppEngine:
    """audio.cpp 常駐 server 経由の Qwen3-TTS CustomVoice。

    Q8 量子化モデルで実測 4.51 倍速（以前あった PyTorch 版の専用 venv サブプロセス
    方式は約 0.44 倍速で、削除済み）。CLI 再起動を避ける常駐サーバー方式は
    SupertonicEngine と同一。
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.host = str(cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        self.port = int(cfg.get("port", 18081))
        self.model_id = str(cfg.get("model_id", "qwen3-tts-customvoice")).strip() or "qwen3-tts-customvoice"
        self.default_voice = str(cfg.get("voice", "Vivian")).strip() or "Vivian"
        self.default_instruct = str(cfg.get("instruct", "")).strip()
        self.default_language = str(cfg.get("language", "japanese")).strip() or "japanese"
        raw_do_sample = cfg.get("do_sample", True)
        if isinstance(raw_do_sample, bool):
            self.do_sample = raw_do_sample
        elif raw_do_sample is None:
            self.do_sample = True
        else:
            self.do_sample = str(raw_do_sample).strip().lower() not in {"0", "false", "no", "off"}
        self.temperature = float(cfg.get("temperature", 0.9))
        self.top_k = int(cfg.get("top_k", 50))
        self.top_p = float(cfg.get("top_p", 1.0))
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.05))
        raw_seed = cfg.get("seed")
        self.seed: int | None = None if raw_seed is None or str(raw_seed).strip() == "" else int(raw_seed)
        self.server_exe = _paths.resolve_audiocpp_server(
            str(cfg.get("server_exe", _paths.default_audiocpp_server()))
        )
        self.model_path = _paths.resolve_under_repo(
            str(
                cfg.get(
                    "model_path",
                    "audio.cpp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/"
                    "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf",
                )
            )
        )
        # 実行デバイスとスレッド数は LFM2.5-Audio・Irodori（engines_audiocpp）と同じ決め方
        # （以前は cuda・8 固定で、NVIDIA の GPU が無い PC では動かなかった）
        self.backend = (
            str(cfg.get("audiocpp_backend", "") or "").strip().lower()
            or audiocpp_client.backend_for_server(self.server_exe)
        )
        raw_threads = int(cfg.get("audiocpp_threads", 0) or 0)
        self.threads = raw_threads if raw_threads > 0 else audiocpp_client.default_threads(self.backend)
        self._server_process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _request_options(self, *, voice: str, instruct: str, language: str) -> dict[str, Any]:
        options: dict[str, Any] = {
            "speaker": voice,
            "instruct": instruct,
            "language": _normalize_audiocpp_qwen_language(language),
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def load(self) -> None:
        default_request_options = self._request_options(
            voice=self.default_voice,
            instruct=self.default_instruct,
            language=self.default_language,
        )
        config = {
            "host": self.host, "port": self.port, "backend": self.backend,
            "device": audiocpp_client.pick_device(self.server_exe, self.backend), "threads": self.threads,
            "lazy_load": False,
            # streaming で登録すると、まとめて返す要求と少しずつ返す要求の両方を受け付ける
            "models": [{"id": self.model_id, "family": "qwen3_tts", "path": str(self.model_path), "task": "tts", "mode": "streaming",
                "default_request_options": default_request_options}],
        }
        config_file = Path(tempfile.gettempdir()) / f"maai-qwen3-tts-cpp-{self.port}.json"
        health_error: Exception | None = None
        try:
            with urllib.request.urlopen(self._base_url + "/health", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
            # 前に立ち上げたサーバーが設定の違うまま残っていれば、止めて立ち上げ直す
            if audiocpp_client.reuse_running_server(
                base_url=self._base_url, config_file=config_file, config=config, name="Qwen3-TTS"
            ):
                return
        except Exception as ex:
            health_error = ex
        if not (self.server_exe.is_file() and self.model_path.is_file()):
            raise RuntimeError(
                "Qwen3-TTS C++ server に接続できません。audiocpp_server.exe と GGUF モデルのパスを設定するか、"
                f"{self._base_url} に事前起動してください: {health_error}"
            ) from health_error
        self._server_process = audiocpp_client.launch_server(
            server_exe=self.server_exe, config_file=config_file, config=config,
            cwd=self.server_exe.parent.parent.parent.parent,
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self._server_process.poll() is not None:
                raise RuntimeError(f"Qwen3-TTS server が終了しました (exit {self._server_process.returncode})")
            try:
                with urllib.request.urlopen(self._base_url + "/health", timeout=2):
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("Qwen3-TTS server の起動がタイムアウトしました")

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        voice = str(req.get("voice", "")).strip() or self.default_voice
        instruct = str(req.get("instruct", "")).strip() or self.default_instruct
        language = str(req.get("language", "")).strip() or self.default_language
        payload = json.dumps({
            "model": self.model_id,
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "options": self._request_options(voice=voice, instruct=instruct, language=language),
        }).encode("utf-8")
        request = urllib.request.Request(
            self._base_url + "/v1/audio/speech", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self._lock:
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    wav = response.read()
            except urllib.error.HTTPError as ex:
                detail = ex.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Qwen3-TTS C++ server error {ex.code}: {detail}") from ex
        return wav_bytes_to_pcm16be_16k_mono(wav), voice

    # Qwen3-TTS は 24 kHz で音を少しずつ作る（最初の声まで 約 70 ms、1 文を作り終えるまでは 0.6 秒前後）。
    SAMPLE_RATE = 24000

    def synthesize_stream(self, req: dict[str, Any], on_chunk: Callable[[bytes], None]) -> tuple[int, str]:
        """届いた音から少しずつ渡す（audio.cpp の `"stream": true`）。"""
        from .audiocpp_client import AudioCppServer, AudioCppSpeechClient

        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        voice = str(req.get("voice", "")).strip() or self.default_voice
        instruct = str(req.get("instruct", "")).strip() or self.default_instruct
        language = str(req.get("language", "")).strip() or self.default_language
        body = {
            "model": self.model_id,
            "input": text,
            "voice": voice,
            "options": self._request_options(voice=voice, instruct=instruct, language=language),
        }
        client = AudioCppSpeechClient(AudioCppServer(host=self.host, port=self.port, models=[]), timeout_sec=60.0)
        resampler = StreamingPcm16Resampler(self.SAMPLE_RATE, 16000)
        total = [0]

        def _emit(raw_le: bytes) -> None:
            out = resampler.process(raw_le)
            if out:
                total[0] += len(out)
                on_chunk(swap_pcm16_endian(out))

        with self._lock:
            client.speech(body, on_chunk=_emit)
        if total[0] <= 0:
            raise RuntimeError("Qwen3-TTS C++ server returned empty audio")
        return total[0], voice

    def close(self) -> None:
        if self._server_process is not None and self._server_process.poll() is None:
            self._server_process.terminate()
        self._server_process = None


class IrodoriTTSEngine:
    """Thin RPC client for irodori_server.py, run out-of-process in the dedicated
    .venv_irodori environment. Irodori-TTS's own pyproject.toml requires
    transformers>=5.12.1,<6, which conflicts with this app's transformers==5.5.3 pin —
    same dedicated-venv pattern as asr/nemotron_asr.py."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.checkpoint = str(cfg.get("checkpoint", "Aratako/Irodori-TTS-v4.1-Small"))
        self.default_caption = str(cfg.get("voice", "明るく自然な日本語の声"))
        self.reference_audio = str(cfg.get("reference_audio", "")).strip()
        self.steps = max(1, min(int(cfg.get("steps", 4)), 40))
        self.irodori_python = str(cfg.get("irodori_python", ".venv_irodori/Scripts/python.exe"))
        self._client: DedicatedVenvTTSClient | None = None
        self._proc: subprocess.Popen | None = None
        self._log_fp = None

    def load(self) -> None:
        if self._client is not None:
            return
        py = str(_paths.resolve_venv_python(self.irodori_python, ".venv_irodori"))
        if not py or not Path(py).exists():
            raise RuntimeError(
                "Irodori-TTS 用 Python（専用 venv）が見つかりません。"
                "エンジン管理で Irodori を適用するか、"
                f"TTS設定の「Irodori Python」にパスを設定してください: {py!r}"
            )
        server_script = str(_paths.package_path("tts", "irodori_server.py"))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        host = "127.0.0.1"

        cmd = [
            py,
            server_script,
            "--host", host,
            "--port", str(port),
            "--checkpoint", self.checkpoint,
            "--default-caption", self.default_caption,
            "--reference-audio", self.reference_audio,
            "--steps", str(self.steps),
        ]

        log_dir = _paths.logs_dir()
        try:
            log_dir.mkdir(exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
        log_path = log_dir / f"irodori_server_{port}.log"
        try:
            self._log_fp = open(log_path, "w", encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            self._log_fp = None
        print(
            f"[irodori] starting: checkpoint={self.checkpoint} port={port}",
            file=sys.stderr,
            flush=True,
        )

        child_env = {
            k: v
            for k, v in os.environ.items()
            if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"}
        }
        child_env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            cwd=str(_paths.REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        self._proc = proc

        def _drain(stream) -> None:
            try:
                for raw in iter(stream.readline, ""):
                    line = (raw or "").rstrip()
                    if not line:
                        continue
                    print(line, file=sys.stderr, flush=True)
                    fp = self._log_fp
                    if fp is not None:
                        try:
                            fp.write(line + "\n")
                            fp.flush()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_drain, args=(proc.stdout,), daemon=True, name="tts-irodori-log").start()

        deadline = time.time() + 240.0
        last_err: Exception | None = None
        try:
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Irodori-TTS サブプロセスが起動に失敗しました（exit={proc.returncode}）。"
                        "ログ logs/irodori_server_*.log を確認してください。"
                    )
                try:
                    self._client = DedicatedVenvTTSClient(host, port, timeout=240.0)
                    break
                except (ConnectionError, OSError) as ex:
                    last_err = ex
                    time.sleep(0.5)
            if self._client is None:
                raise RuntimeError(f"Irodori-TTS サブプロセスへ接続できませんでした: {last_err}")
        except Exception:
            self.close()
            raise
        print(
            f"[irodori] ready: connected to {host}:{port}",
            file=sys.stderr,
            flush=True,
        )

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        if self._client is None:
            raise RuntimeError("Irodori-TTS is not loaded")
        return self._client.synthesize(req)

    def synthesize_stream(self, req: dict[str, Any], on_chunk: Callable[[bytes], None]) -> tuple[int, str]:
        """PyTorch 版は少しずつ返せないので、1 文を作り終えてからまとめて渡す（Irodori-v4 は
        ストリーミングのエンジンとして扱うので、どの動かし方でもこの口が要る）。"""
        pcm, voice = self.synthesize(req)
        if pcm:
            on_chunk(pcm)
        return len(pcm), voice

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            client.close()

        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except Exception:  # noqa: BLE001
                        proc.kill()
            except Exception:  # noqa: BLE001
                pass

        fp = self._log_fp
        self._log_fp = None
        if fp is not None:
            try:
                fp.close()
            except Exception:  # noqa: BLE001
                pass


class IrodoriSpeechCppEngine:
    """speech.cpp（ggml）で動かす Irodori-TTS。既定の動かし方（`irodori_backend` = speechcpp）。

    専用の Python 環境が要らず、Irodori-TTS v4.1 Small MF（q8_0）で 1 文 0.15〜0.3 秒・最初の声まで
    約 70 ms・GPU メモリ 約 2 GB（RTX 5070 Ti Laptop・Vulkan の実測）。PyTorch 版（`IrodoriTTSEngine`）は
    1 文 1 秒前後かかる。

    - 声：参照 WAV があればそれを真似る（speech.cpp の add_voice）。無ければ `voice` の文
      （キャプション。例「明るく自然な日本語の声」）を `instructions` として渡す。
    - **seed を固定する**。参照 WAV もキャプションも無いときの声は seed で決まり、seed を変えると
      文ごとに別の人の声になる（実測で seed 1 と 2 で声の高さが 286 Hz と 142 Hz に分かれた）。
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model = str(cfg.get("speechcpp_model", "") or "").strip() or speechcpp.DEFAULT_IRODORI_MODEL
        self.device = str(cfg.get("speechcpp_device", "gpu") or "gpu").strip() or "gpu"
        self.steps = max(1, min(int(cfg.get("speechcpp_steps", 4) or 4), 40))
        self.seed = int(cfg.get("seed", 1) or 0)
        self.default_caption = str(cfg.get("voice", "") or "").strip()
        self.reference_audio = str(cfg.get("reference_audio", "") or "").strip()
        self.exe = str(cfg.get("speechcpp_exe", "") or "").strip()
        self._worker: speechcpp.SpeechCppWorker | None = None
        self._voice = "none"

    def load(self) -> None:
        if self._worker is not None:
            return
        worker = speechcpp.SpeechCppWorker(
            self.model,
            device=self.device,
            options={"steps": self.steps},
            exe=self.exe,
            on_log=lambda m: print(f"[irodori] {m}", file=sys.stderr, flush=True),
        ).start()
        try:
            ref = _paths.resolve_docker_local_file(self.reference_audio) if self.reference_audio else ""
            if ref:
                ref_path = Path(ref)
                if not ref_path.is_absolute():
                    ref_path = _paths.resolve_under_repo(ref_path)
                if not ref_path.is_file():
                    raise RuntimeError(f"Irodori の参照 WAV がありません: {ref_path}")
                worker.add_voice("ref", str(ref_path))
                self._voice = "ref"
            else:
                self._voice = "none"
        except Exception:
            worker.close()
            raise
        self._worker = worker

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        if self._worker is None:
            raise RuntimeError("Irodori-TTS (speech.cpp) is not loaded")
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        caption = str(req.get("voice", "") or "").strip() or self.default_caption
        options: dict[str, Any] = {"voice": self._voice, "seed": self.seed}
        if self._voice == "none" and caption:
            options["instructions"] = caption
        audio, rate = self._worker.synthesize(text, **options)
        return pcm16le_to_pcm16be_16k(audio.astype("<i2").tobytes(), rate), caption or self._voice

    def synthesize_stream(self, req: dict[str, Any], on_chunk: Callable[[bytes], None]) -> tuple[int, str]:
        """届いた音から少しずつ渡す。MF 版は最初の音まで 約 60 ms（1 文を作り終えるまでは 0.15〜0.3 秒）。"""
        if self._worker is None:
            raise RuntimeError("Irodori-TTS (speech.cpp) is not loaded")
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        caption = str(req.get("voice", "") or "").strip() or self.default_caption
        options: dict[str, Any] = {"voice": self._voice, "seed": self.seed}
        if self._voice == "none" and caption:
            options["instructions"] = caption
        resampler = StreamingPcm16Resampler(self._worker.sample_rate or 48000, 16000)
        total = [0]

        def _emit(chunk: np.ndarray) -> None:
            out = resampler.process(chunk.astype("<i2").tobytes())
            if out:
                total[0] += len(out)
                on_chunk(swap_pcm16_endian(out))

        self._worker.synthesize(text, on_chunk=_emit, **options)
        return total[0], caption or self._voice

    def close(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()


def make_irodori_engine(cfg: dict[str, Any]):
    """`irodori_backend` に合わせて speech.cpp 版・audio.cpp 版・PyTorch 版（専用の環境）を作る。"""
    backend = str(cfg.get("backend", "speechcpp") or "speechcpp").strip().lower()
    if backend == "pytorch":
        return IrodoriTTSEngine(cfg)
    if backend == "audiocpp":
        from .engines_audiocpp import IrodoriAudioCppEngine

        return IrodoriAudioCppEngine(cfg)
    return IrodoriSpeechCppEngine(cfg)


_TSUKASA_IMPORT_LOCK = threading.Lock()


class TsukasaEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.root = _paths.resolve_under_repo(str(cfg.get("root", "Tsukasa-HF")))
        self.reference_audio = str(cfg.get("reference_audio", "")).strip()
        self.steps = max(3, min(int(cfg.get("steps", 5)), 15))
        self.alpha = float(cfg.get("alpha", 0.3))
        self.beta = float(cfg.get("beta", 0.7))
        self._module: Any = None
        self._phonemizer: Any = None
        self._styles: dict[str, Any] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._module is not None:
            return
        if not self.root.is_dir():
            raise RuntimeError(f"Tsukasa モデルディレクトリがありません: {self.root}")
        engine_manager.require_module_installed("munch", "Tsukasa")
        # importable.py reads Configs/config_kanade.yml relative to the current
        # working directory.  The app worker starts in the app root instead.
        with _TSUKASA_IMPORT_LOCK:
            old_cwd = Path.cwd()
            sys.path.insert(0, str(self.root))
            try:
                os.chdir(self.root)
                import transformers
                if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
                    transformers.PreTrainedModel.all_tied_weights_keys = {}
                import importable
                from Utils.phonemize import mixed_phon
            except Exception as ex:
                raise RuntimeError(f"Tsukasa の import に失敗しました: {ex}") from ex
            finally:
                os.chdir(old_cwd)
        self._module = importable
        self._phonemizer = mixed_phon

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        if self._module is None:
            raise RuntimeError("Tsukasa is not loaded")
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        ref = str(req.get("reference_audio", "")).strip() or self.reference_audio
        if not ref:
            raise RuntimeError("Tsukasa は本人の参照音声を指定してください（同梱サンプルは使用しません）")
        path = Path(ref).resolve()
        if not path.is_file():
            raise RuntimeError(f"Tsukasa の参照音声が見つかりません: {path}")
        try:
            header = path.read_bytes()[:32]
        except OSError as ex:
            raise RuntimeError(f"Tsukasa の参照音声を読めません: {path} ({ex})") from ex
        if header.startswith(b"version https://git-lfs.github"):
            raise RuntimeError(
                "Tsukasa の参照音声が Git LFS ポインタのままです。"
                "実際の WAV ファイルを選択してください: " + str(path)
            )
        key = str(path)
        with self._lock:
            style = self._styles.get(key)
            if style is None:
                try:
                    style = self._module.compute_style_through_clip(key)
                except Exception as ex:
                    detail = str(ex).strip() or type(ex).__name__
                    raise RuntimeError(
                        f"Tsukasa の参照音声を読めません: {path} ({detail})"
                    ) from ex
                self._styles[key] = style
            # Tsukasa's inference expects its custom phoneme representation;
            # supplying raw Japanese produces unstable, clipped waveforms.
            phonemes = self._phonemizer.smart_phonemize(text)
            wav = self._module.inference(
                phonemes, style, alpha=self.alpha, beta=self.beta,
                diffusion_steps=self.steps, embedding_scale=1, rate_of_speech=1.0,
            )
        samples = np.nan_to_num(
            np.asarray(wav, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0
        )
        pcm16le = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return pcm16le_to_pcm16be_16k(pcm16le, 24000), path.name
