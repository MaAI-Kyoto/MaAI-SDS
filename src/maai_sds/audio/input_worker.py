from __future__ import annotations

import argparse
import base64
import json
import queue
import socket
import sys
import threading
import time
import wave

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

from ..ipc.jsonline import JsonLineEventClient
from ..ipc.worker_runtime import connect_worker_client, finish_worker, parse_json_object, safe_emit, to_jsonable as _to_jsonable
from .convert import f32_from_pcm16_array, f32_from_pcm16_bytes, pcm16_from_f32_full, pcm16_from_f32_unit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio input worker process")
    parser.add_argument("--host", required=True, type=str)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--config-json", required=True, type=str)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        config = parse_json_object(args.config_json, worker_name="AudioInputWorker")
    except ValueError as ex:
        print(str(ex), file=sys.stderr, flush=True)
        return 2

    sample_rate = int(config.get("sample_rate", 16000) or 16000)
    input_mode = str(config.get("input_mode", "mic")).strip().lower() or "mic"
    # 共通入力ハブの送信単位は常に320サンプルに固定する。
    # 各モジュール（ASR/MAAI）側で必要なサイズへ再チャンク化して消費する。
    block_size = 320
    mic_device = config.get("mic_device_index")
    try:
        mic_device_index = int(mic_device) if mic_device is not None else None
    except Exception:
        mic_device_index = None

    client = JsonLineEventClient(args.host, args.port)
    if not connect_worker_client(client, worker_name="AudioInputWorker"):
        return 2

    def emit(event_type: str, **data: object) -> None:
        safe_emit(client.send, event_type, **data)

    # 廃止: 別ソケットへの audio_forward。Orchestrator 経路は primary IPC のみ。
    config.pop("audio_forward_host", None)
    config.pop("audio_forward_port", None)

    def emit_audio(event_type: str, **data: object) -> None:
        safe_emit(client.send, event_type, **data)

    # WAV ローカル再生の出力先（名前の部分一致）。空なら従来どおり既定デバイス。
    # 既定（例: Bluetooth イヤホン）以外へ出したいとき用。
    wav_local_playback_device = str(config.get("wav_local_playback_device", "") or "").strip()
    wav_want_speaker = bool(config.get("wav_local_playback", False)) and input_mode == "wav"
    wav_browser_monitor = bool(config.get("wav_browser_monitor", False)) and input_mode == "wav"
    wav_local_playback = wav_want_speaker and sd is not None
    if wav_browser_monitor:
        emit("log", message="wav_browser_monitor: 読んだ PCM を UI へ送ります")
    if wav_want_speaker and sd is None:
        emit(
            "log",
            message=(
                "wav_local_playback: sounddevice でスピーカーへ出せないため、"
                "モニター音声を UI へ送ります" if wav_browser_monitor
                else "wav_local_playback: sounddevice が無いためローカル再生をスキップします"
            ),
        )
    try:
        wav_channel_index = int(config.get("wav_channel_index", 0))
    except Exception:
        wav_channel_index = 0
    wav_channel_index = max(0, int(wav_channel_index))
    wav_send_other_ch_to_mmd = bool(config.get("wav_send_other_ch_to_mmd", False)) and input_mode == "wav"
    wav_other_ch_to_maai_ch2 = bool(config.get("wav_other_ch_to_maai_ch2", False)) and input_mode == "wav"

    # --- AEC state (initialised when enable_aec; all input modes) ---
    aec_apm: object | None = None
    aec_ref_mode = str(config.get("aec_ref", config.get("wav_aec_ref", "other_ch"))).strip() or "other_ch"
    if input_mode != "wav" and aec_ref_mode == "other_ch":
        aec_ref_mode = "system"
    aec_near_buf = np.empty(0, dtype=np.float32)
    aec_far_buf = np.empty(0, dtype=np.float32)
    aec_frame_samples = sample_rate // 100  # 10 ms
    aec_sys_audio_buf = np.empty(0, dtype=np.float32)
    aec_sys_lock = threading.Lock()

    local_play_buffer = np.empty(0, dtype=np.float32)
    local_play_lock = threading.Lock()
    local_play_stream = None

    def _emit_wav_monitor(audio_f32: np.ndarray) -> None:
        if not wav_browser_monitor:
            return
        x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if x.size <= 0:
            return
        pcm_i16 = pcm16_from_f32_full(x)
        emit_audio(
            "wav_monitor_audio",
            audio_b64=base64.b64encode(pcm_i16.tobytes()).decode("ascii"),
            sample_rate=int(sample_rate),
        )

    def _append_local_speaker(audio_f32: np.ndarray) -> None:
        nonlocal local_play_buffer
        if not wav_local_playback:
            return
        x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if x.size <= 0:
            return
        with local_play_lock:
            if local_play_buffer.size <= 0:
                local_play_buffer = x.copy()
            else:
                local_play_buffer = np.concatenate([local_play_buffer, x])

    def _feed_monitors(audio_f32: np.ndarray) -> None:
        if wav_local_playback:
            _append_local_speaker(audio_f32)
        if wav_browser_monitor:
            _emit_wav_monitor(audio_f32)

    def _clear_local_speaker_buffer() -> None:
        nonlocal local_play_buffer
        with local_play_lock:
            local_play_buffer = np.empty(0, dtype=np.float32)

    def _resolve_local_playback_device(name: str) -> int | None:
        """出力デバイスを名前の部分一致で選ぶ。見つからなければ既定（None）。

        同じ名前のデバイスは MME / DirectSound / WASAPI ごとに別番号で並ぶ。既定の
        出力と同じホスト API のものを優先し、遅延やサンプルレート変換の性質を
        これまでと揃える（WASAPI 共有モードは 16 kHz を受けないことがある）。
        """
        if not name or sd is None:
            return None
        want = name.lower()
        try:
            default_api = sd.query_devices(kind="output")["hostapi"]
            candidates = [
                (index, device)
                for index, device in enumerate(sd.query_devices())
                if device["max_output_channels"] > 0
                and want in str(device["name"]).lower()
            ]
        except Exception:
            return None
        if not candidates:
            emit("log", message=f"wav_local_playback: 出力デバイス「{name}」が無いため既定を使います")
            return None
        same_api = [item for item in candidates if item[1]["hostapi"] == default_api]
        return (same_api or candidates)[0][0]

    def _start_local_speaker() -> None:
        nonlocal local_play_stream, local_play_buffer, wav_local_playback
        if (not wav_local_playback) or sd is None:
            return
        if local_play_stream is not None:
            return

        def _local_out_callback(outdata, frames, _time_info, _status):  # noqa: ANN001
            nonlocal local_play_buffer
            with local_play_lock:
                if local_play_buffer.size >= frames:
                    chunk = local_play_buffer[:frames].copy()
                    local_play_buffer = local_play_buffer[frames:]
                else:
                    chunk = np.zeros(frames, dtype=np.float32)
                    if local_play_buffer.size > 0:
                        chunk[: local_play_buffer.size] = local_play_buffer
                        local_play_buffer = np.empty(0, dtype=np.float32)
            outdata[:, 0] = chunk

        device = _resolve_local_playback_device(wav_local_playback_device)
        try:
            local_play_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                blocksize=block_size,
                dtype="float32",
                latency="low",
                device=device,
                callback=_local_out_callback,
            )
            local_play_stream.start()
            if device is not None:
                emit("log", message=f"wav_local_playback: 出力先 = {sd.query_devices(device)['name']}")
        except Exception as ex:
            emit(
                "log",
                message=f"wav_local_playback: スピーカーを開けないためスキップします ({ex})",
            )
            local_play_stream = None
            wav_local_playback = False

    def _stop_local_speaker() -> None:
        nonlocal local_play_stream
        if not wav_local_playback or sd is None:
            return
        _clear_local_speaker_buffer()
        if local_play_stream is not None:
            try:
                local_play_stream.stop()
            except Exception:
                pass
            try:
                local_play_stream.close()
            except Exception:
                pass
            local_play_stream = None
        try:
            sd.stop()
        except Exception:
            pass

    stop_event = threading.Event()
    browser_audio_frames = {"n": 0}
    # mic コールバックはリアルタイムスレッドで動くため、ブロッキング送信・AEC・
    # アロケーションを直接行わずフレームをここへ積むだけにする（C1 対策）。
    # 重い処理と client.send は _mic_processor_loop（別スレッド）が担当する。
    mic_frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
    mic_dropped_frames = {"count": 0}
    mic_last_status = {"text": ""}
    processing_enabled = threading.Event()
    start_paused = bool(config.get("start_paused", False))
    if not start_paused:
        processing_enabled.set()
    wav_play_event = threading.Event()
    wav_rewind_request = threading.Event()
    user_sample_cursor = 0
    user_clock_base_ts: float | None = None
    user_lock = threading.Lock()
    system_audio_buffer = np.empty(0, dtype=np.float32)
    system_lock = threading.Lock()

    try:
        input_gain = float(config.get("input_gain", 1.0) or 1.0)
    except Exception:
        input_gain = 1.0
    input_gain = max(0.0, min(20.0, float(input_gain)))

    # 雑音抑制（audio/denoise.py）。AEC の設定の後で作る。入力ゲインの後にかけ、長さは変えない。
    noise_suppressor = None

    def _denoise(mono_f32: np.ndarray) -> np.ndarray:
        if noise_suppressor is None or mono_f32.size <= 0:
            return mono_f32
        return noise_suppressor(mono_f32)

    def _apply_input_gain(mono_f32: np.ndarray) -> np.ndarray:
        if input_gain == 1.0:
            return mono_f32
        return np.clip(mono_f32 * np.float32(input_gain), -1.0, 1.0)

    def _reset_audio_timeline() -> None:
        nonlocal user_sample_cursor, user_clock_base_ts
        with user_lock:
            user_sample_cursor = 0
            user_clock_base_ts = None

    def _append_system_audio(audio_f32: np.ndarray) -> None:
        nonlocal system_audio_buffer
        x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if x.size <= 0:
            return
        with system_lock:
            if system_audio_buffer.size <= 0:
                system_audio_buffer = x.copy()
            else:
                system_audio_buffer = np.concatenate([system_audio_buffer, x])
            # 処理停止中などドレインされない間の無制限蓄積を防ぐ（直近のみ保持）。
            # 再開時に巨大な古バックログを抱えないよう asr/service.inject_system_audio と同じ 2 秒上限。
            _max_keep_sys = int(sample_rate * 2)
            if system_audio_buffer.size > _max_keep_sys:
                system_audio_buffer = system_audio_buffer[-_max_keep_sys:]

    def _append_aec_system_audio(audio_f32: np.ndarray) -> None:
        nonlocal aec_sys_audio_buf
        x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if x.size <= 0:
            return
        with aec_sys_lock:
            if aec_sys_audio_buf.size <= 0:
                aec_sys_audio_buf = x.copy()
            else:
                aec_sys_audio_buf = np.concatenate([aec_sys_audio_buf, x])
            # 同上: ドレインされない間の無制限蓄積を防ぐ（直近 2 秒のみ保持）。
            _max_keep_aec = int(sample_rate * 2)
            if aec_sys_audio_buf.size > _max_keep_aec:
                aec_sys_audio_buf = aec_sys_audio_buf[-_max_keep_aec:]

    def _take_aec_system_audio(n_samples: int) -> np.ndarray:
        nonlocal aec_sys_audio_buf
        n = max(1, int(n_samples))
        with aec_sys_lock:
            if aec_sys_audio_buf.size >= n:
                out = aec_sys_audio_buf[:n].copy()
                aec_sys_audio_buf = aec_sys_audio_buf[n:]
                return out
            out = np.zeros(n, dtype=np.float32)
            if aec_sys_audio_buf.size > 0:
                out[: aec_sys_audio_buf.size] = aec_sys_audio_buf
                aec_sys_audio_buf = np.empty(0, dtype=np.float32)
            return out

    def _take_system_audio(n_samples: int) -> np.ndarray:
        nonlocal system_audio_buffer
        n = max(1, int(n_samples))
        with system_lock:
            if system_audio_buffer.size >= n:
                out = system_audio_buffer[:n].copy()
                system_audio_buffer = system_audio_buffer[n:]
                return out
            out = np.zeros(n, dtype=np.float32)
            if system_audio_buffer.size > 0:
                out[: system_audio_buffer.size] = system_audio_buffer
                system_audio_buffer = np.empty(0, dtype=np.float32)
            return out

    def _stdin_watcher() -> None:
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if line.lower() == "stop":
                    stop_event.set()
                    return
                if line == "start_processing":
                    processing_enabled.set()
                    if input_mode == "wav":
                        wav_play_event.set()
                    emit("processing_started")
                    continue
                if line == "stop_processing":
                    processing_enabled.clear()
                    if input_mode == "wav":
                        wav_play_event.clear()
                        _clear_local_speaker_buffer()
                    emit("processing_stopped")
                    continue
                if line.startswith("system_audio:"):
                    try:
                        audio_bytes = base64.b64decode(line[13:])
                        audio_f32 = f32_from_pcm16_bytes(audio_bytes)
                        _append_system_audio(audio_f32)
                        if aec_ref_mode == "system":
                            _append_aec_system_audio(audio_f32)
                    except Exception:
                        pass
                    continue
                if line.startswith("user_audio:"):
                    try:
                        audio_bytes = base64.b64decode(line[len("user_audio:") :])
                        n_even = len(audio_bytes) - (len(audio_bytes) % 2)
                        if n_even > 0:
                            _emit_pcm_bytes(audio_bytes[:n_even], src_rate=sample_rate)
                            if browser_audio_frames["n"] == 0:
                                emit("log", message="browser mic: 最初の音声フレームを受信")
                            browser_audio_frames["n"] += 1
                    except Exception:
                        pass
                    continue
                if line == "wav_play":
                    processing_enabled.set()
                    _reset_audio_timeline()
                    wav_play_event.set()
                    emit("wav_command_done", command="wav_play")
                    continue
                if line == "wav_stop":
                    wav_play_event.clear()
                    _clear_local_speaker_buffer()
                    emit("wav_command_done", command="wav_stop")
                    continue
                if line == "wav_end":
                    wav_play_event.clear()
                    _clear_local_speaker_buffer()
                    wav_rewind_request.set()
                    continue
                if line.startswith("system_audio_frame:"):
                    try:
                        obj = json.loads(line[len("system_audio_frame:") :])
                        audio_f32 = np.asarray(obj.get("audio", []), dtype=np.float32).reshape(-1)
                        _append_system_audio(audio_f32)
                        if aec_ref_mode == "system":
                            _append_aec_system_audio(audio_f32)
                    except Exception:
                        pass
                    continue
            stop_event.set()
        except Exception:
            stop_event.set()

    def _emit_user_and_system(mono: np.ndarray) -> None:
        nonlocal user_sample_cursor, user_clock_base_ts
        n = int(mono.size)
        if n <= 0:
            return
        system = _take_system_audio(n)
        with user_lock:
            if user_clock_base_ts is None:
                duration = float(n) / float(max(1, sample_rate))
                user_clock_base_ts = time.time() - duration
            sample_origin = int(user_sample_cursor)
            user_sample_cursor = sample_origin + n
            clock_base = float(user_clock_base_ts)
        # ハブの送信単位は 320 サンプル（20ms @ 16kHz）。ブラウザ ScriptProcessor の
        # 4096 フレームを 1 本の JSON にすると IPC が一瞬で膨らみ、対話ワーカーの
        # UI emit が詰まって MaAI への供給が実時間から外れる。
        offset = 0
        while offset < n:
            end = min(n, offset + block_size)
            piece = mono[offset:end]
            sys_piece = system[offset:end]
            pn = int(piece.size)
            sample_start = sample_origin + offset
            sample_end = sample_start + pn
            ts_start = clock_base + (sample_start / float(sample_rate))
            ts_end = clock_base + (sample_end / float(sample_rate))
            emitted_at = time.time()
            emit_audio(
                "audio_frame",
                channel="user",
                compare_slot="top",
                sample_rate=int(sample_rate),
                audio=_to_jsonable(piece),
                sample_start=sample_start,
                sample_end=sample_end,
                ts_start=ts_start,
                ts_end=ts_end,
                emitted_at=emitted_at,
            )
            emit_audio(
                "system_audio_frame",
                channel="system",
                compare_slot="top",
                sample_rate=int(sample_rate),
                audio=_to_jsonable(sys_piece),
                sample_start=sample_start,
                sample_end=sample_end,
                ts_start=ts_start,
                ts_end=ts_end,
                emitted_at=emitted_at,
            )
            offset = end

    def _audio_callback(indata, _frames, _time_info, status) -> None:  # noqa: ANN001
        # リアルタイムオーディオコールバック。ブロッキング I/O・重い処理・大きな
        # アロケーションは禁止。indata は再利用バッファなのでコピーしてキューへ積み、
        # 即座に戻る。status の emit（ブロッキング送信）も処理スレッドへ委譲する。
        if status:
            mic_last_status["text"] = str(status)
        try:
            mono = np.array(indata[:, 0], dtype=np.float32)  # 再利用バッファから退避
        except Exception:
            return
        try:
            mic_frame_queue.put_nowait(mono)
        except queue.Full:
            # 処理が実時間に追いつかない場合は最古フレームを捨ててレイテンシを抑える。
            try:
                mic_frame_queue.get_nowait()
                mic_frame_queue.put_nowait(mono)
            except (queue.Empty, queue.Full):
                pass
            mic_dropped_frames["count"] += 1

    def _mic_processor_loop() -> None:
        """mic フレームキューを排出し、AEC・変換・送信（ブロッキング可）を行う。"""
        last_drop_log = 0.0
        while not stop_event.is_set():
            status_text = mic_last_status.get("text", "")
            if status_text:
                mic_last_status["text"] = ""
                emit("log", message=f"audio callback status: {status_text}")
            try:
                mono = mic_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                pcm = pcm16_from_f32_unit(mono).tobytes()
                _emit_pcm_bytes(
                    pcm,
                    src_rate=sample_rate,
                    channels=1,
                    sample_width=2,
                    pcm_channel_index=0,
                )
            except Exception:
                pass
            dropped = mic_dropped_frames["count"]
            if dropped:
                now = time.time()
                if now - last_drop_log >= 1.0:
                    last_drop_log = now
                    mic_dropped_frames["count"] = 0
                    emit("log", message=f"mic frames dropped (processing slower than realtime): {dropped}")

    def _resample_linear(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate <= 0 or dst_rate <= 0 or mono.size <= 1 or src_rate == dst_rate:
            return mono
        new_len = max(1, int(round((mono.size * dst_rate) / src_rate)))
        src_x = np.linspace(0.0, 1.0, mono.size, endpoint=True)
        dst_x = np.linspace(0.0, 1.0, new_len, endpoint=True)
        return np.interp(dst_x, src_x, mono).astype(np.float32)

    def _pcm_bytes_to_mono_f32(
        raw: bytes, *, channels: int, sample_width: int, channel_index: int = 0
    ) -> np.ndarray:
        if not raw:
            return np.empty(0, dtype=np.float32)
        ch = max(1, int(channels))
        sw = int(sample_width)
        ci = max(0, int(channel_index))
        if sw == 1:
            arr_u8 = np.frombuffer(raw, dtype=np.uint8)
            mono = (arr_u8.astype(np.float32) - 128.0) / 128.0
        elif sw == 2:
            n_even = len(raw) - (len(raw) % 2)
            if n_even <= 0:
                return np.empty(0, dtype=np.float32)
            mono = f32_from_pcm16_bytes(raw[:n_even])
        elif sw == 3:
            n_triplets = len(raw) // 3
            if n_triplets <= 0:
                return np.empty(0, dtype=np.float32)
            b = np.frombuffer(raw[: n_triplets * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            sign = 1 << 23
            v = (v ^ sign) - sign
            mono = v.astype(np.float32) / float(1 << 23)
        elif sw == 4:
            n4 = len(raw) - (len(raw) % 4)
            if n4 <= 0:
                return np.empty(0, dtype=np.float32)
            mono = np.frombuffer(raw[:n4], dtype="<i4").astype(np.float32) / float(1 << 31)
        else:
            raise RuntimeError(f"unsupported wav sample width: {sw} bytes")
        if ch > 1:
            n = (mono.size // ch) * ch
            if n <= 0:
                return np.empty(0, dtype=np.float32)
            pick = min(ci, ch - 1)
            mono = mono[:n].reshape(-1, ch)[:, pick]
        return mono.astype(np.float32, copy=False)

    def _run_aec(near_f32: np.ndarray, far_f32: np.ndarray) -> np.ndarray:
        """10 ms 固定フレームで LiveKit APM の AEC を適用する。"""
        nonlocal aec_near_buf, aec_far_buf
        if aec_apm is None:
            return near_f32
        from livekit.rtc import AudioFrame as _AecFrame

        near_all = np.concatenate([aec_near_buf, near_f32]) if aec_near_buf.size > 0 else near_f32
        far_all = np.concatenate([aec_far_buf, far_f32]) if aec_far_buf.size > 0 else far_f32
        fsz = aec_frame_samples
        n = min(near_all.size, far_all.size) // fsz
        if n <= 0:
            aec_near_buf = near_all.copy()
            aec_far_buf = far_all.copy()
            return np.empty(0, dtype=np.float32)
        parts: list[np.ndarray] = []
        for i in range(n):
            s, e = i * fsz, (i + 1) * fsz
            near_i16 = pcm16_from_f32_full(near_all[s:e])
            far_i16 = pcm16_from_f32_full(far_all[s:e])
            far_frame = _AecFrame(
                data=far_i16.tobytes(), sample_rate=sample_rate,
                num_channels=1, samples_per_channel=fsz,
            )
            near_frame = _AecFrame(
                data=near_i16.tobytes(), sample_rate=sample_rate,
                num_channels=1, samples_per_channel=fsz,
            )
            aec_apm.process_reverse_stream(far_frame)
            aec_apm.process_stream(near_frame)
            parts.append(f32_from_pcm16_array(near_frame.data))
        consumed = n * fsz
        aec_near_buf = near_all[consumed:].copy() if consumed < near_all.size else np.empty(0, dtype=np.float32)
        aec_far_buf = far_all[consumed:].copy() if consumed < far_all.size else np.empty(0, dtype=np.float32)
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def _prepare_other_ch(
        raw: bytes, *, src_rate: int, channels: int, sample_width: int, pcm_channel_index: int,
    ) -> np.ndarray | None:
        if channels < 2:
            return None
        if not wav_send_other_ch_to_mmd and not wav_other_ch_to_maai_ch2:
            return None
        other_ch = 1 - min(pcm_channel_index, 1)
        other = _pcm_bytes_to_mono_f32(raw, channels=channels, sample_width=sample_width, channel_index=other_ch)
        if other.size <= 0:
            return None
        other = _resample_linear(other, int(src_rate), int(sample_rate))
        if wav_other_ch_to_maai_ch2:
            _append_system_audio(other)
        return other

    def _emit_other_ch_to_mmd(other: np.ndarray | None) -> None:
        if wav_send_other_ch_to_mmd and other is not None:
            pcm_i16 = pcm16_from_f32_full(other)
            emit_audio("wav_other_ch_audio", audio_b64=base64.b64encode(pcm_i16.tobytes()).decode("ascii"))

    def _emit_pcm_bytes(
        raw: bytes,
        *,
        src_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        pcm_channel_index: int = 0,
    ) -> None:
        if not raw:
            return
        if not processing_enabled.is_set():
            return
        # AEC path: other_ch — stereo WAV, unselected channel as reference
        if aec_apm is not None and aec_ref_mode == "other_ch" and channels == 2:
            near_ch = pcm_channel_index
            far_ch = 1 - near_ch
            near = _pcm_bytes_to_mono_f32(raw, channels=channels, sample_width=sample_width, channel_index=near_ch)
            far = _pcm_bytes_to_mono_f32(raw, channels=channels, sample_width=sample_width, channel_index=far_ch)
            if near.size <= 0:
                return
            near = _resample_linear(near, int(src_rate), int(sample_rate))
            far = _resample_linear(far, int(src_rate), int(sample_rate))
            mono = _run_aec(near, far)
            if mono.size <= 0:
                return
            mono = _denoise(_apply_input_gain(mono))
            _feed_monitors(mono)
            # Append the current WAV other-channel block before taking the
            # paired system frame. The old order emitted one block of zeros,
            # then kept ch2 320 samples behind ch1 for the whole file.
            other = _prepare_other_ch(
                raw,
                src_rate=src_rate,
                channels=channels,
                sample_width=sample_width,
                pcm_channel_index=pcm_channel_index,
            )
            _emit_user_and_system(mono)
            _emit_other_ch_to_mmd(other)
            return
        # AEC path: system — use system audio (TTS output etc.) as reference
        if aec_apm is not None and aec_ref_mode == "system":
            mono = _pcm_bytes_to_mono_f32(
                raw, channels=channels, sample_width=sample_width, channel_index=pcm_channel_index
            )
            if mono.size <= 0:
                return
            mono = _resample_linear(mono, int(src_rate), int(sample_rate))
            far = _take_aec_system_audio(mono.size)
            mono = _run_aec(mono, far)
            if mono.size <= 0:
                return
            mono = _denoise(_apply_input_gain(mono))
            _feed_monitors(mono)
            other = _prepare_other_ch(
                raw,
                src_rate=src_rate,
                channels=channels,
                sample_width=sample_width,
                pcm_channel_index=pcm_channel_index,
            )
            _emit_user_and_system(mono)
            _emit_other_ch_to_mmd(other)
            return
        # Non-AEC path (mono WAV / mic / tcp / APM unavailable)
        mono = _pcm_bytes_to_mono_f32(
            raw, channels=channels, sample_width=sample_width, channel_index=pcm_channel_index
        )
        if mono.size <= 0:
            return
        mono = _resample_linear(mono, int(src_rate), int(sample_rate))
        mono = _denoise(_apply_input_gain(mono))
        _feed_monitors(mono)
        other = _prepare_other_ch(
            raw,
            src_rate=src_rate,
            channels=channels,
            sample_width=sample_width,
            pcm_channel_index=pcm_channel_index,
        )
        _emit_user_and_system(mono)
        _emit_other_ch_to_mmd(other)

    threading.Thread(target=_stdin_watcher, daemon=True, name="audio-input-worker-stdin").start()
    enable_aec = bool(config.get("enable_aec", config.get("wav_enable_aec", False)))
    if enable_aec:
        try:
            from livekit.rtc import AudioProcessingModule as _AecAPM

            aec_apm = _AecAPM(
                echo_cancellation=True,
                noise_suppression=False,
                high_pass_filter=False,
                auto_gain_control=False,
            )
            emit("log", message="AEC enabled (livekit.rtc)")
        except Exception as ex:
            emit("log", message=f"AEC not available, proceeding without: {ex}")
    else:
        emit("log", message="AEC disabled by config")
    from .denoise import make_noise_suppressor

    noise_suppressor = make_noise_suppressor(
        str(config.get("noise_suppression", "off")), sample_rate, lambda m: emit("log", message=m),
    )
    stream = None
    feeder_thread: threading.Thread | None = None
    tcp_server_sock: socket.socket | None = None
    try:
        if input_mode == "mic":
            if sd is None:
                raise RuntimeError("sounddevice is not available for mic mode")
            stream = sd.InputStream(
                channels=1,
                samplerate=sample_rate,
                blocksize=block_size,
                dtype="float32",
                latency="low",
                callback=_audio_callback,
                device=mic_device_index,
            )
            threading.Thread(
                target=_mic_processor_loop,
                daemon=True,
                name="audio-input-mic-processor",
            ).start()
            stream.start()
        elif input_mode == "wav":
            wav_path = str(config.get("wav_path", "")).strip()
            if not wav_path:
                raise RuntimeError("wav_path is empty")
            wav_loop = bool(config.get("wav_loop", False))

            def _parse_wav_sec(key: str) -> float | None:
                raw = config.get(key)
                if raw is None:
                    return None
                s = str(raw).strip()
                if s == "":
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

            _ws = _parse_wav_sec("wav_start_sec")
            wav_start_sec = 0.0 if _ws is None else max(0.0, float(_ws))
            wav_end_sec = _parse_wav_sec("wav_end_sec")

            if not start_paused:
                wav_play_event.set()
            _start_local_speaker()

            def _wav_loop() -> None:
                try:
                    with wave.open(wav_path, "rb") as wf:
                        src_rate = int(wf.getframerate() or sample_rate)
                        channels = int(wf.getnchannels() or 1)
                        sample_width = int(wf.getsampwidth() or 2)
                        n_file_frames = int(wf.getnframes() or 0)
                        chunk_frames = max(1, int(round(block_size * src_rate / float(max(1, sample_rate)))))
                        start_frame = int(round(wav_start_sec * float(src_rate)))
                        start_frame = max(0, min(start_frame, max(0, n_file_frames)))
                        end_frame = n_file_frames
                        if wav_end_sec is not None:
                            end_frame = int(round(float(wav_end_sec) * float(src_rate)))
                        end_frame = max(start_frame, min(end_frame, n_file_frames))
                        if start_frame >= end_frame:
                            emit(
                                "log",
                                message=f"wav range empty: start={start_frame} end={end_frame} (skip playback)",
                            )
                            emit("wav_playback_finished", reason="empty_range")
                            return
                        try:
                            wf.setpos(start_frame)
                        except Exception:
                            wf.rewind()
                            remain = int(start_frame)
                            while remain > 0 and not stop_event.is_set():
                                blk = min(remain, max(1, chunk_frames * 8))
                                wf.readframes(blk)
                                remain -= blk

                        if aec_apm is not None and aec_ref_mode == "other_ch" and channels == 2:
                            emit("log", message=f"AEC active (other_ch): near=ch{wav_channel_index+1}, far=ch{2-wav_channel_index}")
                        elif aec_apm is not None and aec_ref_mode == "other_ch" and channels != 2:
                            emit("log", message=f"AEC skipped: wav is {channels}ch (stereo required for other_ch mode)")
                        elif aec_apm is not None and aec_ref_mode == "system":
                            emit("log", message=f"AEC active (system): near=ch{wav_channel_index+1}, far=system audio")
                        emit(
                            "log",
                            message=f"wav range: frames [{start_frame},{end_frame}) / {n_file_frames} @ {src_rate}Hz",
                        )
                        # Realtime pacing (deadline-based, with a small lead so the
                        # local OutputStream never underruns from sleep jitter).
                        # ソース系列サンプルで累積目標時刻を管理し、周回ごとの作業時間で
                        # ドリフトしないようにする。pace_t0=None は「再アンカー要求」。
                        WAV_PACE_LEAD_SEC = 0.08
                        pace_t0: float | None = None
                        pace_samples = 0
                        silence_block = np.zeros(block_size, dtype=np.float32)
                        silence_pace_t0: float | None = None
                        silence_pace_samples = 0
                        while not stop_event.is_set():
                            if wav_rewind_request.is_set():
                                wav_rewind_request.clear()
                                try:
                                    wf.setpos(start_frame)
                                except Exception:
                                    wf.rewind()
                                _reset_audio_timeline()
                                pace_t0 = None
                                emit("wav_command_done", command="wav_end")
                            if not wav_play_event.is_set():
                                if silence_pace_t0 is None:
                                    silence_pace_t0 = time.perf_counter()
                                    silence_pace_samples = 0
                                _emit_user_and_system(silence_block)
                                silence_pace_samples += block_size
                                dt = silence_pace_t0 + silence_pace_samples / float(sample_rate) - time.perf_counter()
                                if dt > 0:
                                    time.sleep(dt)
                                pace_t0 = None
                                continue
                            silence_pace_t0 = None
                            if not processing_enabled.is_set():
                                time.sleep(0.01)
                                pace_t0 = None
                                continue
                            pos = wf.tell()
                            if pos >= end_frame:
                                if wav_loop:
                                    try:
                                        wf.setpos(start_frame)
                                    except Exception:
                                        wf.rewind()
                                    continue
                                emit("log", message="wav playback finished (range end)")
                                emit("wav_playback_finished", reason="range_end")
                                wav_play_event.clear()
                                try:
                                    wf.setpos(start_frame)
                                except Exception:
                                    wf.rewind()
                                _reset_audio_timeline()
                                continue
                            n_read = min(chunk_frames, end_frame - pos)
                            if n_read <= 0:
                                if wav_loop:
                                    try:
                                        wf.setpos(start_frame)
                                    except Exception:
                                        wf.rewind()
                                    continue
                                emit("wav_playback_finished", reason="range_end")
                                wav_play_event.clear()
                                try:
                                    wf.setpos(start_frame)
                                except Exception:
                                    wf.rewind()
                                _reset_audio_timeline()
                                continue
                            raw = wf.readframes(n_read)
                            if not raw:
                                if wav_loop:
                                    try:
                                        wf.setpos(start_frame)
                                    except Exception:
                                        wf.rewind()
                                    continue
                                emit("log", message="wav playback finished")
                                emit("wav_playback_finished", reason="eof")
                                wav_play_event.clear()
                                try:
                                    wf.setpos(start_frame)
                                except Exception:
                                    wf.rewind()
                                _reset_audio_timeline()
                                continue
                            _emit_pcm_bytes(
                                raw,
                                src_rate=src_rate,
                                channels=channels,
                                sample_width=sample_width,
                                pcm_channel_index=wav_channel_index,
                            )
                            # Deadline-based realtime pacing. 供給の作業時間ぶんドリフト
                            # しないよう、累積ソースサンプル数から次の目標時刻を計算する。
                            # LEAD ぶん先行して供給し、OutputStream のアンダーラン(=ぶつぶつ)
                            # と sleep 分解能ジッタを先読みバッファで吸収する。
                            frames_read = len(raw) // max(1, channels * sample_width)
                            if pace_t0 is None:
                                pace_t0 = time.perf_counter()
                                pace_samples = 0
                            pace_samples += frames_read
                            target = pace_t0 + pace_samples / float(max(1, src_rate)) - WAV_PACE_LEAD_SEC
                            dt = target - time.perf_counter()
                            if dt > 0:
                                time.sleep(dt)
                            elif dt < -0.5:
                                # 大きく遅延した(長いストール等)場合は再アンカーして
                                # まとめ供給のバースト(バッファ/IPC スパイク)を避ける。
                                pace_t0 = time.perf_counter()
                                pace_samples = 0
                        if wav_send_other_ch_to_mmd and channels >= 2:
                            emit("wav_other_ch_audio_end")
                except Exception as ex:
                    emit("fatal", message=f"wav mode error: {ex}")
                    stop_event.set()

            feeder_thread = threading.Thread(target=_wav_loop, daemon=True, name="audio-input-wav")
            feeder_thread.start()
        elif input_mode == "browser":
            emit("log", message="browser mic: Web UI からの 16kHz PCM を待ちます")
        elif input_mode == "tcp":
            host = str(config.get("tcp_host", "127.0.0.1")).strip() or "127.0.0.1"
            port = int(config.get("tcp_port", 15600) or 15600)
            tcp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcp_server_sock.bind((host, port))
            tcp_server_sock.listen(1)
            tcp_server_sock.settimeout(0.3)

            def _tcp_loop() -> None:
                while not stop_event.is_set():
                    conn = None
                    try:
                        conn, addr = tcp_server_sock.accept()
                        emit("log", message=f"tcp audio connected: {addr}")
                        conn.settimeout(0.3)
                        carry = b""
                        while not stop_event.is_set():
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            buf = carry + chunk
                            n_even = len(buf) - (len(buf) % 2)
                            if n_even > 0:
                                _emit_pcm_bytes(buf[:n_even], src_rate=sample_rate)
                            carry = buf[n_even:]
                    except socket.timeout:
                        continue
                    except Exception as ex:
                        emit("log", message=f"tcp audio error: {ex}")
                        time.sleep(0.1)
                    finally:
                        if conn is not None:
                            try:
                                conn.close()
                            except Exception:
                                pass

            feeder_thread = threading.Thread(target=_tcp_loop, daemon=True, name="audio-input-tcp")
            feeder_thread.start()
            emit("log", message=f"tcp audio listening: {host}:{port}")
        else:
            raise RuntimeError(f"unknown input_mode: {input_mode}")

        emit("ready")
        if processing_enabled.is_set():
            emit("processing_started")
        while not stop_event.is_set():
            time.sleep(0.1)
    except Exception as ex:
        emit("fatal", message=str(ex))
        return 1
    finally:
        _stop_local_speaker()
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if tcp_server_sock is not None:
            try:
                tcp_server_sock.close()
            except Exception:
                pass
        finish_worker(client, lambda: emit("stopped"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

