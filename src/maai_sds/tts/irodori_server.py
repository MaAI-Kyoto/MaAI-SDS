#!/usr/bin/env python
"""Irodori-TTS subprocess server for the dedicated .venv_irodori environment.

The model is loaded before the listening socket is bound, so a successful client
connection is the readiness signal. One app-side client is accepted, and each synthesis
request receives exactly one length-prefixed JSON response.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

# This file is launched by absolute path. Its directory contains venv_rpc.py and is
# inserted explicitly so the dedicated venv does not need maai_sds installed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from venv_rpc import encode_pcm, recv_frame, send_frame  # noqa: E402


DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"
DEFAULT_CAPTION = "明るく自然な日本語の声"


def _resample_pcm16le(raw_pcm16le: bytes, source_rate: int, target_rate: int = 16000) -> bytes:
    pcm = raw_pcm16le if (len(raw_pcm16le) % 2 == 0) else raw_pcm16le[:-1]
    if not pcm:
        return b""
    src_rate = int(source_rate)
    dst_rate = int(target_rate)
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return pcm
    src = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if src.size == 0:
        return b""
    target_len = max(1, int(round(src.size * (float(dst_rate) / float(src_rate)))))
    x_old = np.linspace(0.0, 1.0, num=src.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    dst = np.interp(x_new, x_old, src).astype(np.float32)
    dst = np.clip(dst, -32768.0, 32767.0).astype("<i2")
    return dst.tobytes()


def _swap_pcm16_endian(raw_pcm16: bytes) -> bytes:
    pcm = raw_pcm16 if (len(raw_pcm16) % 2 == 0) else raw_pcm16[:-1]
    if not pcm:
        return b""
    swapped = bytearray(len(pcm))
    swapped[0::2] = pcm[1::2]
    swapped[1::2] = pcm[0::2]
    return bytes(swapped)


def _pcm16le_to_pcm16be_16k(raw_pcm16le: bytes, sample_rate: int) -> bytes:
    pcm16le = _resample_pcm16le(raw_pcm16le, source_rate=int(sample_rate), target_rate=16000)
    return _swap_pcm16_endian(pcm16le)


def pick_device_and_precision(torch: Any) -> tuple[str, str]:
    """モデルとコーデックを載せる装置と精度。

    CUDA があれば bf16 で CUDA。無ければ Apple Silicon の MPS、それも無ければ CPU。
    irodori_tts は bf16 を CUDA・XPU にしか許さないので、MPS・CPU は fp32
    （以前は cuda・bf16 固定で、Mac では「CUDA device requested but torch.cuda.is_available() is False」で落ちた）。
    """
    if torch.cuda.is_available():
        return "cuda", "bf16"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "fp32"
    return "cpu", "fp32"


class IrodoriTTSServerModel:
    def __init__(
        self,
        *,
        checkpoint: str,
        default_caption: str,
        reference_audio: str,
        steps: int,
    ) -> None:
        import torch
        from irodori_tts.inference_runtime import (
            InferenceRuntime,
            RuntimeKey,
            SamplingRequest,
            download_hf_checkpoint,
        )

        self.default_caption = default_caption
        self.reference_audio = reference_audio.strip()
        self.steps = max(1, min(int(steps), 40))
        self._sampling_request = SamplingRequest
        self._lock = threading.Lock()

        # Windows + torch 2.11 has no Triton by default, so keep the existing eager path.
        # TF32 remains useful on RTX-class GPUs.
        device, precision = pick_device_and_precision(torch)
        if device == "cuda":
            torch.set_float32_matmul_precision("high")
        checkpoint_path = download_hf_checkpoint(checkpoint)
        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=device,
                codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                model_precision=precision,
                codec_device=device,
                codec_precision=precision,
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
                compile_model=False,
                compile_dynamic=False,
            )
        )

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        text = str(req.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must not be empty")
        caption = str(req.get("voice", "")).strip() or self.default_caption
        ref_wav = str(req.get("reference_audio", "")).strip() or self.reference_audio
        kwargs: dict[str, Any] = {
            "text": text,
            "caption": caption,
            "num_steps": self.steps,
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 3.0,
            "cfg_scale_speaker": 5.0,
            "seed": 42,
        }
        if ref_wav:
            path = Path(ref_wav)
            if not path.is_file():
                raise RuntimeError(f"Irodori の参照音声が見つかりません: {path}")
            kwargs["ref_wav"] = str(path.resolve())
        else:
            kwargs["no_ref"] = True
        with self._lock:
            result = self._runtime.synthesize(self._sampling_request(**kwargs), log_fn=None)
        wav = result.audio.detach().float().cpu().numpy().reshape(-1)
        pcm16le = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return _pcm16le_to_pcm16be_16k(pcm16le, int(result.sample_rate)), caption


def _log(message: str) -> None:
    print(f"[irodori] {message}", flush=True)


def _serve_client(conn: socket.socket, model: IrodoriTTSServerModel) -> None:
    conn.settimeout(None)
    while True:
        try:
            req = recv_frame(conn)
        except (ConnectionError, OSError):
            _log("client disconnected")
            return
        except (ValueError, UnicodeDecodeError):
            _log("malformed frame; closing client")
            return
        if not isinstance(req, dict):
            _log("malformed frame (non-dict); closing client")
            return
        op = str(req.get("op", "")).strip().lower()
        try:
            if op == "synthesize":
                pcm, caption = model.synthesize(req)
                send_frame(conn, {"pcm_b64": encode_pcm(pcm), "voice": caption})
            else:
                send_frame(conn, {"error": f"unknown op: {op}"})
        except Exception as ex:  # noqa: BLE001
            try:
                send_frame(conn, {"error": str(ex)})
            except Exception:  # noqa: BLE001
                return


def main() -> int:
    parser = argparse.ArgumentParser(description="Irodori-TTS subprocess server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--default-caption", default=DEFAULT_CAPTION)
    parser.add_argument("--reference-audio", default="")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    _log(f"Loading model: {args.checkpoint}")
    model = IrodoriTTSServerModel(
        checkpoint=args.checkpoint,
        default_caption=args.default_caption,
        reference_audio=args.reference_audio,
        steps=args.steps,
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    _log(f"ready, listening on {args.host}:{args.port}")

    try:
        conn, addr = server.accept()
        _log(f"client connected: {addr}")
        try:
            _serve_client(conn, model)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            server.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
