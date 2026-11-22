"""MMD/Nod 関連ランタイム。

`MmdRemoteBridge` / `MmdAudioPacer` は audio/output_worker にある。このクラスは頷きモーションの
生成と送出だけを担い、MMDAgent-EX への送出は `send_audio_output` IPC（mmd_send_text /
mmd_bind_head）に任せる。

頷きは公式 MMDAgent-EX の仕組みで動かす。頭（と目）のボーンを KeyValue に結び付け
（`MODEL_BINDBONE`、mmd/avatar_keys.py）、角度を `KEYVALUE_SET` で毎フレーム送る。
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from ..maai.nod_motion import generate_natural_nod
from .avatar_keys import head_pitch_command


class MmdRuntime:
    """MMD ブリッジを保持しない `MmdRuntime`（音声・Remote は別プロセス側）。"""

    def __init__(
        self,
        *,
        emit: Callable[[dict[str, object]], None],
        send_audio_output: Callable[[dict], bool],
        log_fn: Callable[[str], None],
    ) -> None:
        self._emit = emit
        self._send_audio_output = send_audio_output
        self._log = log_fn

        self._remote_connected: bool = False

        self._nod_cancel_event = threading.Event()
        self._nod_thread: threading.Thread | None = None

    # -- remote status（audio/output_worker の bridge からの IPC） ---------------

    def on_remote_status_event(self, payload: dict) -> None:
        self._remote_connected = bool(payload.get("connected", False))

    # -- nod ---------------------------------------------------------------------

    def cancel_nod_motion(self) -> None:
        self._nod_cancel_event.set()
        t = self._nod_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._nod_thread = None

    def _send_head_pitch(self, pitch: float) -> None:
        self._send_audio_output({"type": "mmd_send_text", "text": head_pitch_command(pitch)})

    def _nod_motion_thread_inner(self, params: dict) -> None:
        try:
            motion, time_axis = generate_natural_nod(
                range_rad=float(params.get("range_rad", 0.1)),
                count=int(params.get("count", 1)),
                use_pre_rise=bool(params.get("use_pre_rise", False)),
                velocity=float(params.get("velocity", 0.5)),
                fps=int(params.get("fps", 30)),
                decay_rate=float(params.get("decay_rate", 0.6)),
                interpolation=str(params.get("interpolation", "cosine")),
                pre_rise_ratio=float(params.get("pre_rise_ratio", 0.8)),
            )
        except Exception as ex:
            self._emit({"type": "log", "message": f"[NodMotion] generate error: {ex}"})
            return
        if len(motion) == 0:
            return

        motion_list = [float(x) for x in motion.tolist()]
        time_list = [float(x) for x in time_axis.tolist()]
        self._emit({
            "type": "nod_motion_chart",
            "motion": motion_list,
            "time_axis": time_list,
            "interpolation": str(params.get("interpolation", "")),
            "velocity": float(params.get("velocity", 0.0) or 0.0),
            "nod_count": int(params.get("count", 1) or 1),
            "p_nod": float(params.get("chart_p_nod", 0.0) or 0.0),
            "range_rad": float(params.get("range_rad", 0.0) or 0.0),
            "swing_up_prob": float(params.get("chart_swing_up_prob", 0.0) or 0.0),
            "use_pre_rise": bool(params.get("use_pre_rise", False)),
            "buffer_frames": int(params.get("buffer_frames", 0) or 0),
            "raw_range": float(params.get("chart_raw_range", 0.0) or 0.0),
            "raw_speed": float(params.get("chart_raw_speed", 0.0) or 0.0),
        })

        scale = float(params.get("scale", 1.0))
        speed_scale = max(0.01, float(params.get("speed_scale", 1.0)))
        eye_scale = float(params.get("eye_scale", 1.0))
        bone = str(params.get("bone", "頭"))

        if not self._remote_connected:
            self._emit({"type": "log", "message": "[NodMotion] bridge not connected"})
        else:
            try:
                # 頭（と目）を頷きの KeyValue に結び付ける（変わっていなければ bridge 側で送らない）
                self._send_audio_output({"type": "mmd_bind_head", "bone": bone, "eye_scale": eye_scale})
            except Exception:
                pass

        self._emit({
            "type": "log",
            "message": (
                f"[NodMotion] start frames={len(motion)} "
                f"count={params.get('count')} pre_rise={params.get('use_pre_rise')} "
                f"range={float(params.get('range_rad', 0.0) or 0.0):.3f} "
                f"buf={params.get('buffer_frames')}"
            ),
        })

        start_time = time.perf_counter()
        last_pitch = 0.0
        for i in range(len(motion)):
            if self._nod_cancel_event.is_set():
                self._emit({"type": "log", "message": "[NodMotion] cancelled"})
                break
            t = float(time_axis[i]) / speed_scale
            pitch = float(motion[i]) * scale
            target = start_time + t
            sleep_dur = target - time.perf_counter()
            if sleep_dur > 0:
                if self._nod_cancel_event.wait(timeout=sleep_dur):
                    self._emit({"type": "log", "message": "[NodMotion] cancelled"})
                    break
            try:
                self._send_head_pitch(pitch)
                last_pitch = pitch
            except Exception as ex:
                self._emit({"type": "log", "message": f"[NodMotion] send error: {ex}"})
                break
        if last_pitch != 0.0:
            # 途中で止めたときも頭を正面へ戻す（結び付けたボーンは最後の値のまま止まるため）
            try:
                self._send_head_pitch(0.0)
            except Exception:
                pass

    def _nod_motion_thread(self, params: dict) -> None:
        self._nod_motion_thread_inner(params)

    def start_nod_motion(self, params: dict) -> None:
        self.cancel_nod_motion()
        self._nod_cancel_event.clear()
        t = threading.Thread(
            target=self._nod_motion_thread, args=(params,),
            daemon=True, name="nod-motion",
        )
        self._nod_thread = t
        t.start()
