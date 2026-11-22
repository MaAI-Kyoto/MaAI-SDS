from __future__ import annotations

import threading
from collections import deque
from functools import wraps
from typing import Any


def project_captured_frames(
    captured: list[dict[str, Any]] | None,
    public_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep the model's public output contract for every captured frame.

    A nod_para forward also returns internal VAP/VAD/BC heads, while its
    public result exposes only nod outputs. Forwarding those internal heads
    interleaves two models in the same GUI probability series for N > 1.
    """
    if not captured:
        return [public_result]
    return [
        {key: value for key, value in frame.items() if key in public_result}
        for frame in captured
        if isinstance(frame, dict)
    ]


class ChunkFrameCapture:
    """Capture every causally evaluated frame from a chunked MaAI VAP forward.

    MaAI evaluates all ``inference_chunk_frames`` causally, but its public
    result queue intentionally retains only the newest frame.  SDS decision
    rules (turn / BC / nod consecutive counts, mean_prob) are defined per
    audio frame, so the worker needs the complete list without changing
    MaAI's public return contract.
    """

    def __init__(self) -> None:
        self._batches: deque[list[dict[str, Any]]] = deque()
        self._lock = threading.Lock()

    def push(self, frames: list[dict[str, Any]]) -> None:
        with self._lock:
            self._batches.append([dict(frame) for frame in frames])

    def pop(self) -> list[dict[str, Any]] | None:
        with self._lock:
            return self._batches.popleft() if self._batches else None

    def clear(self) -> None:
        with self._lock:
            self._batches.clear()


def install_chunk_frame_capture(
    maai_or_vap: Any,
    *,
    inference_chunk_frames: int,
) -> ChunkFrameCapture | None:
    """Install a transparent full-frame capture hook on a MaAI VAP.

    Returns ``None`` for one-frame inference because the public MaAI result is
    already per-frame.  For MaaiMultiple, whose caller does not request
    ``return_all_frames``, the wrapper requests all frames internally and still
    returns only the last one to preserve the package's public behavior.
    """

    if int(inference_chunk_frames) <= 1:
        return None
    vap = getattr(maai_or_vap, "vap", maai_or_vap)
    existing = getattr(vap, "_maai_sds_chunk_frame_capture", None)
    if isinstance(existing, ChunkFrameCapture):
        return existing

    original_forward = getattr(vap, "forward")
    capture = ChunkFrameCapture()

    @wraps(original_forward)
    def _forward_all_frames(*args: Any, **kwargs: Any):
        call_args = list(args)
        if len(call_args) >= 4:
            caller_requested_all = bool(call_args[3])
            call_args[3] = True
        else:
            caller_requested_all = bool(kwargs.get("return_all_frames", False))
            kwargs["return_all_frames"] = True
        out, cache = original_forward(*call_args, **kwargs)
        frames_obj = out if isinstance(out, list) else [out]
        frames = [dict(frame) for frame in frames_obj if isinstance(frame, dict)]
        frames = frames[-max(1, int(inference_chunk_frames)) :]
        if frames:
            capture.push(frames)
        public_out = out if caller_requested_all else (frames_obj[-1] if frames_obj else out)
        return public_out, cache

    vap.forward = _forward_all_frames
    setattr(vap, "_maai_sds_chunk_frame_capture", capture)
    return capture
