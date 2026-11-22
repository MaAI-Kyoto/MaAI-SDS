"""TTS 合成・再生まわりの GUI 非依存ヘルパ。

`controller.py` の `_voice_dialog_synthesize_l16be`、`_send_tts_play_*`、
`_voice_dialog_speak_nonstream_async` 等で重複していた処理のうち、
**Qt や IPC client に依存しない純関数**を集めるモジュール。

- worker（`dialog/worker.py`）と controller（`controller.py`）の双方が共有する。
- IPC を直接叩くわけではなく、呼び出し側が用意したコールバック (`tts_rpc`,
  `route_pcm`, `route_break` など) を介して動作する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..audio.convert import swap_pcm16_endian  # type: ignore


# ----- TTS エンジン分類 ---------------------------------------------------------

WORKER_BACKED_ENGINES = frozenset(
    {
        "Qwen3-TTS-CPP",
        "VOICEVOX-Core",
        "Kokoro-RealtimeTTS",
        "Cartesia-Realtime",
        "AivisSpeech",
        "Supertonic-3",
        "Irodori-v4",
        "Tsukasa",
        "LFM2.5-Audio",
    }
)

BRIDGE_BACKED_ENGINES = frozenset(
    {
        "VOICEVOX",
        "TEMPLATE",
    }
)

# 合成しながら音を少しずつ返すエンジン（`synthesize_stream` を持つ）。最初の声までの待ち時間が短くなる。
# Irodori-v4 は speech.cpp 版・audio.cpp 版が少しずつ返し、PyTorch 版は 1 文をまとめて 1 回で渡す。
BINARY_STREAM_ENGINES = frozenset(
    {
        "Cartesia-Realtime",
        "Kokoro-RealtimeTTS",
        "Qwen3-TTS-CPP",
        "Irodori-v4",
        "LFM2.5-Audio",
    }
)


def is_worker_engine(engine: str) -> bool:
    """ワーカープロセス（`tts/worker.py`）経由で合成するエンジンか。"""
    return (engine or "").strip() in WORKER_BACKED_ENGINES


def is_bridge_engine(engine: str) -> bool:
    """`tts/service.py` ブリッジ経由で合成するエンジンか。"""
    return (engine or "").strip() in BRIDGE_BACKED_ENGINES


def is_binary_stream_engine(engine: str) -> bool:
    """合成しながら随時 audio チャンクを emit するエンジンか（`tts/worker.py` の BINARY_STREAM_ENGINES と対応）。"""
    return (engine or "").strip() in BINARY_STREAM_ENGINES


def default_synth_timeout_sec(engine: str) -> float:
    return float(
        {
            "VOICEVOX": 30.0,
            "VOICEVOX-Core": 120.0,
            "Kokoro-RealtimeTTS": 60.0,
            "Cartesia-Realtime": 120.0,
            "Qwen3-TTS-CPP": 60.0,
            "AivisSpeech": 60.0,
            "Supertonic-3": 60.0,
            "Irodori-v4": 180.0,
            "LFM2.5-Audio": 60.0,
            "Tsukasa": 120.0,
        }.get((engine or "").strip(), 120.0),
    )


# ----- PCM ヘルパ -------------------------------------------------------------

L16_SAMPLE_RATE = 16000  # MMD/MaAI の標準
L16_FRAME_BYTES = 640    # 10ms @ 16kHz / 16bit / 1ch


def split_pcm_into_frames(
    pcm: bytes, *, frame_bytes: int = L16_FRAME_BYTES
) -> Iterable[bytes]:
    """PCM を 640B (10ms@16kHz) フレームに分割するイテレータ。"""
    if not pcm:
        return
    pos = 0
    n = len(pcm)
    fb = max(1, int(frame_bytes))
    while pos < n:
        chunk = pcm[pos : pos + fb]
        pos += len(chunk)
        if chunk:
            yield chunk


# ----- 合成リクエスト用 dict 組み立て --------------------------------------------

@dataclass(frozen=True)
class SynthRequest:
    request_id: str
    engine: str
    inputs: dict


def build_voicevox_bridge_command(req: SynthRequest, *, timeout: float = 30.0) -> dict:
    inputs = dict(req.inputs or {})
    vv = str(inputs.get("voice", "1")).strip() or "1"
    try:
        speaker = int(vv)
    except ValueError as ex:
        raise RuntimeError("VOICEVOX の話者IDは整数で指定してください。") from ex
    return {
        "op": "synthesize",
        "request_id": req.request_id,
        "method": "voicevox",
        "host": str(inputs.get("host", "127.0.0.1")),
        "port": int(inputs.get("port", 50021)),
        "speaker": speaker,
        "utterance": str(inputs.get("text", "")),
        "voice_text": vv,
        "timeout": float(max(5.0, min(600.0, timeout))),
    }


def build_template_bridge_command(req: SynthRequest, *, timeout: float = 30.0) -> dict:
    inputs = dict(req.inputs or {})
    template = inputs.get("template")
    if not isinstance(template, dict):
        raise RuntimeError("TEMPLATE は template 設定が orchestrator に渡されていません。")
    return {
        "op": "synthesize",
        "request_id": req.request_id,
        "method": "template",
        "timeout": float(max(5.0, min(600.0, timeout))),
        "template": template,
        "inputs": inputs,
        "voice": str(inputs.get("voice", "")),
    }


# ----- 安全な PCM 変換 ----------------------------------------------------------

def to_l16le(pcm: bytes, *, source_endian: str = "be") -> bytes:
    """合成結果（L16 BE 既定）→ 出力用の L16 LE に変換。"""
    if not pcm:
        return b""
    if source_endian == "le":
        return bytes(pcm)
    return swap_pcm16_endian(pcm)


def gap_silence_pcm_l16le(gap_sec: float) -> bytes:
    """セグメント間ギャップ用ゼロ PCM (L16 LE, 16kHz mono) を返す。最大 10 秒。"""
    if gap_sec <= 0.0:
        return b""
    samples = int(L16_SAMPLE_RATE * min(float(gap_sec), 10.0))
    return bytes(samples * 2)


__all__ = [
    "WORKER_BACKED_ENGINES",
    "BRIDGE_BACKED_ENGINES",
    "BINARY_STREAM_ENGINES",
    "is_worker_engine",
    "is_bridge_engine",
    "is_binary_stream_engine",
    "default_synth_timeout_sec",
    "L16_SAMPLE_RATE",
    "L16_FRAME_BYTES",
    "split_pcm_into_frames",
    "SynthRequest",
    "build_voicevox_bridge_command",
    "build_template_bridge_command",
    "to_l16le",
    "gap_silence_pcm_l16le",
]
