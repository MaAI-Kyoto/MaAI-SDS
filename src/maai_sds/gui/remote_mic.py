"""Host microphone capture for a Docker-hosted dialog orchestrator."""

from __future__ import annotations

import base64
import queue
import threading
from typing import Callable, Protocol

from ..audio.convert import resample_pcm16le


class _SoundDeviceLike(Protocol):
    def query_devices(self, *args, **kwargs): ...
    def RawInputStream(self, *args, **kwargs): ...  # noqa: N802


class RemoteMicSender:
    """Capture host PCM and forward exact 20 ms/16 kHz mono chunks.

    The PortAudio callback only copies into a bounded queue.  Resampling,
    base64 encoding and socket writes happen on a normal Python thread so a
    slow remote connection cannot block the real-time callback.
    """

    TARGET_RATE = 16000
    CHUNK_SAMPLES = 320
    CHUNK_BYTES = CHUNK_SAMPLES * 2

    def __init__(
        self,
        send_payload: Callable[[dict[str, object]], bool | None],
        *,
        sounddevice_module: _SoundDeviceLike | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if sounddevice_module is None:
            import sounddevice as sounddevice_module  # type: ignore[no-redef]

        self._sd = sounddevice_module
        self._send_payload = send_payload
        self._log = log or (lambda _message: None)
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._source_rate = self.TARGET_RATE
        self._device_index: int | None = None
        self._lock = threading.RLock()
        self._last_status = ""
        self._dropped = 0

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._stream is not None

    @property
    def device_index(self) -> int | None:
        return self._device_index

    def _device_sample_rate(self, device_index: int | None) -> int:
        try:
            if device_index is None:
                info = self._sd.query_devices(kind="input")
            else:
                info = self._sd.query_devices(device_index, kind="input")
            rate = int(round(float(info.get("default_samplerate", self.TARGET_RATE))))
            if rate > 0:
                return rate
        except Exception:
            pass
        return self.TARGET_RATE

    def _choose_source_rate(self, device_index: int | None) -> int:
        """ネイティブの AudioInputWorker と同じく、まず 16 kHz で開く。

        16 kHz で開けば OS（PortAudio / WASAPI）がフィルタ付きで変換する。デバイス既定の
        48 kHz で開いて resample_pcm16le（線形補間・ローパス無し）で間引くと折り返し雑音が
        乗り、ASR の精度がネイティブより落ちる。16 kHz を受け付けないデバイスだけ既定値で開く。
        """
        check = getattr(self._sd, "check_input_settings", None)
        if callable(check):
            try:
                check(device=device_index, samplerate=self.TARGET_RATE, channels=1, dtype="int16")
                return self.TARGET_RATE
            except Exception:
                pass
        return self._device_sample_rate(device_index)

    def start(self, device_index: int | None = None) -> None:
        normalized_device = None if device_index is None or int(device_index) < 0 else int(device_index)
        with self._lock:
            if self._stream is not None and normalized_device == self._device_index:
                return
        self.stop()
        source_rate = self._choose_source_rate(normalized_device)
        blocksize = max(1, int(round(source_rate * 0.020)))
        self._source_rate = source_rate
        self._device_index = normalized_device
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._last_status = ""
        self._dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        thread = threading.Thread(
            target=self._sender_loop,
            args=(stop_event, source_rate),
            daemon=True,
            name="qt-remote-mic-sender",
        )
        self._thread = thread
        thread.start()
        stream = None
        try:
            stream = self._sd.RawInputStream(
                samplerate=source_rate,
                blocksize=blocksize,
                device=normalized_device,
                channels=1,
                dtype="int16",
                latency="low",
                callback=self._audio_callback,
            )
            # Publish the stream before start().  Some PortAudio backends can
            # invoke the callback immediately, and a concurrent stop must not
            # miss an already-created stream in that interval.
            with self._lock:
                self._stream = stream
            stream.start()
        except Exception:
            with self._lock:
                if self._stream is stream:
                    self._stream = None
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            stop_event.set()
            thread.join(timeout=1.0)
            self._thread = None
            self._device_index = None
            raise
        label = "default" if normalized_device is None else str(normalized_device)
        self._log(f"host microphone started: device={label}, rate={source_rate}Hz")

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        self._device_index = None
        if stream is not None:
            self._log("host microphone stopped")

    def _audio_callback(self, indata, _frames, _time_info, status) -> None:  # noqa: ANN001
        if status:
            self._last_status = str(status)
        try:
            raw = bytes(indata)
        except Exception:
            return
        if not raw:
            return
        if len(raw) % 2:
            raw = raw[:-1]
        try:
            self._queue.put_nowait(raw)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(raw)
            except (queue.Empty, queue.Full):
                pass
            self._dropped += 1

    def _sender_loop(self, stop_event: threading.Event, source_rate: int) -> None:
        pending = bytearray()
        reported_drops = 0
        while not stop_event.is_set():
            status = self._last_status
            if status:
                self._last_status = ""
                self._log(f"host microphone callback status: {status}")
            if self._dropped != reported_drops:
                reported_drops = self._dropped
                self._log(f"host microphone frames dropped: {reported_drops}")
            try:
                raw = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            converted = resample_pcm16le(
                raw,
                source_rate=source_rate,
                target_rate=self.TARGET_RATE,
            )
            if not converted:
                continue
            pending.extend(converted)
            while len(pending) >= self.CHUNK_BYTES:
                chunk = bytes(pending[: self.CHUNK_BYTES])
                del pending[: self.CHUNK_BYTES]
                payload: dict[str, object] = {
                    "type": "inject_user_audio",
                    "audio_b64": base64.b64encode(chunk).decode("ascii"),
                }
                try:
                    self._send_payload(payload)
                except Exception as ex:
                    self._log(f"host microphone send failed: {ex}")
