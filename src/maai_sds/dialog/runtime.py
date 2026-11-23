"""音声対話ランタイム本体。

`dialog/worker.py` の closure 群を 1 クラスに集約したもの。
- latency 計測
- VAD 割り込み判定
- TTS セグメント再生 / nonstream TTS 再生
- LLM ワーカーイベント処理
- GUI テスト dispatch (`llm_gui_test` / `tts_gui_test`)
- 音声対話ワーカー起動 (`ensure_voice_dialog_workers`)
- ASR / MaAI からの ingress イベント転送
- モジュール起動の async wrapper

挙動・スレッド名・ログ文言・公開イベント (`type` / `_src` / フィールド) は
移管前と完全互換。状態 dict は属性として保持し、closure 内の `nonlocal` を
廃した平坦移動のみ行っている。
"""
from __future__ import annotations

import audioop
import base64
import hashlib
import json
import queue
import threading
import time
from typing import Any, Callable

from ..tts import playback_service as tps
from ..audio.convert import swap_pcm16_endian
from ..mmd.blink import (
    DEFAULT_BLINK_INTERVAL_MEAN_SEC,
    normalize_blink_enable,
    normalize_blink_interval_mean_sec,
    normalize_blink_mode,
    sanitize_blink_morph,
)

def _tts_config_digest(config: object) -> str:
    """TTS ワーカー設定の指紋。GUI が「起動し直すか」を判断するために使う。

    `ensure_tts` が同一視に使う json.dumps(sort_keys=True) と同じ正規化にそろえる。
    設定が無い（VOICEVOX / TEMPLATE のような worker 型でないエンジン）ときは空。
    """
    if not isinstance(config, dict) or not config:
        return ""
    blob = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class VoiceDialogRuntime:
    """音声対話ランタイム。"""

    def __init__(
        self,
        *,
        emit: Callable[[dict[str, object]], None],
        event_queue: "queue.Queue[dict[str, object]]",
        supervisor: "Any",
        mmd: "Any",
        audio_input_server: "Any",
        asr_audio_server: "Any",
        asr_compare_audio_server: "Any",
        maai_audio_server: "Any",
    ) -> None:
        self._emit = emit
        self._event_queue = event_queue
        # 音声対話ライフサイクルの重い部分（ensure_* / ワーカー停止）を専用の単一スレッドで
        # 逐次実行し、イベントループ（on_tick / イベント排出）を塞がない。state 変更は
        # event_queue 経由でループ上に戻して実行する（orchestrator.state の競合回避）。
        # start/stop の整合は epoch で取る（進行中 start を stop/新 start が無効化）。
        self._vd_lifecycle_queue: "queue.Queue" = queue.Queue()
        self._vd_lifecycle_thread: "threading.Thread | None" = None
        self._vd_lifecycle_lock = threading.Lock()
        self._vd_start_epoch = 0
        self._supervisor = supervisor
        self._mmd = mmd
        # tts_router は循環参照（runtime.vd_state を tts_router が必要とする）を避けるため、
        # 構築後に set_tts_router で注入する。
        self._tts_router: Any | None = None
        self._audio_input_server = audio_input_server
        self._asr_audio_server = asr_audio_server
        self._asr_compare_audio_server = asr_compare_audio_server
        self._maai_audio_server = maai_audio_server
        self._orchestrator: Any | None = None

        # 音声対話ストリーミングTTSの補助（ペーサ completion → on_tts_done 等）
        self._orch_vd_tts_state: dict[str, object] = {
            "segments": [],
            "tail_gen": 0,
            "tail_pending": False,
            "first_audible": False,
            "playback_pcm_bytes": 0,
            "suppress_orch": False,
        }
        # Silero VAD（最後の値）と割り込み判定の最小実装（worker 内完結）
        self._orch_vd_vad_state: dict[str, object] = {
            "last_silero_vad": None,
            "last_p_now": None,
            "last_update_ts": 0.0,
        }
        # Controller の voice_dialog_latency_signal と同等のタイミング
        self._vd_latency_lock = threading.Lock()
        # llm_/tts_latency_band_emitted: 1 ターンテイキングにつき LLM 帯・TTS 帯は各 1 本のみ（複数 tts_segment で二重計測しない）
        self._vd_turn_latency_state: dict[str, float | bool | None] = {
            "llm_perf": None,
            "llm_wall": None,
            "tts_req_perf": None,
            "tts_req_wall": None,
            "llm_latency_band_emitted": False,
            "tts_latency_band_emitted": False,
        }
        self._vd_tts_jobs: dict[str, int] = {"inflight": 0}
        # inflight の内訳（request_id ごと）。割り込み判定で未確定の先読みの合成を除くために使う。
        self._vd_tts_inflight_by_rid: dict[str, int] = {}
        self._vd_tts_jobs_lock = threading.Lock()
        self._vd_tts_segments_done = threading.Condition(self._vd_tts_jobs_lock)
        self._vd_tts_seg_lock = threading.Lock()  # セグメント合成・ルーティングの順序保証
        # セグメント TTS は FIFO キュー＋単一ワーカーで逐次処理し発話順を保証する
        # （旧: セグメント毎にスレッド生成 → Lock 取得順が非FIFOで音声が前後しうる）。
        self._vd_tts_seg_queue: "queue.Queue" = queue.Queue()
        self._vd_tts_seg_worker: "threading.Thread | None" = None
        self._vd_nonstream_refs: dict[str, int] = {"n": 0}
        self._vd_interrupt_stop = threading.Event()

        self._llm_pending: dict[str, dict] = {}
        self._llm_pending_lock = threading.Lock()
        self._llm_terminal_ids: dict[str, None] = {}
        self._llm_settings: dict[str, object] = {}
        self._llm_warmup_error = ""

        self._vd_nonstream_stop_flag = threading.Event()
        self._vd_scheduled_exit_pending: dict[str, bool] = {"flag": False}
        self._tail_done_timer: threading.Timer | None = None
        self._tail_done_timer_lock = threading.Lock()
        # _orch_vd_tts_state の tail_gen / tail_pending は _after_llm_done スレッドと
        # on_mmd_send_complete (メインループ) が同時アクセスするため専用ロックで保護する。
        self._vd_tts_tail_lock = threading.Lock()

        # TTS→MaAI system 注入の無音ギャップ中、プロット ch2 を連続スクロールさせるゼロ送出（旧 Controller QTimer 相当）
        self._vd_voice_dialog_running_for_maai_tick = threading.Event()
        self._vd_maai_last_real_system_audio_ts = 0.0
        threading.Thread(
            target=self._run_maai_system_audio_silence_tick_forever,
            daemon=True,
            name="vd-maai-silence-tick",
        ).start()

        # 音声ファンアウト用 per-target キュー。
        # handle_audio_frame はキューに入れるだけ（ノンブロッキング）、
        # 各スレッドが独立して srv.send() を担当するため、
        # 1 つのワーカーが遅延しても他のパスが止まらない。
        self._asr_audio_q: queue.Queue = queue.Queue(maxsize=200)
        self._asr_compare_audio_q: queue.Queue = queue.Queue(maxsize=200)
        self._maai_audio_q: queue.Queue = queue.Queue(maxsize=200)
        for _name, _srv, _q in (
            ("asr", asr_audio_server, self._asr_audio_q),
            ("compare", asr_compare_audio_server, self._asr_compare_audio_q),
            ("maai", maai_audio_server, self._maai_audio_q),
        ):
            threading.Thread(
                target=self._run_audio_send_loop,
                args=(_name, _srv, _q),
                daemon=True,
                name=f"vd-audio-send-{_name}",
            ).start()

    # -- wiring helpers -------------------------------------------------------

    def set_orchestrator(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator

    def set_tts_router(self, tts_router: Any) -> None:
        self._tts_router = tts_router

    def as_orchestrator_callbacks(self) -> dict[str, Callable]:
        """`DialogOrchestrator` のコンストラクタへ展開するコールバック群。

        worker は `DialogOrchestrator(**runtime.as_orchestrator_callbacks())` の
        ように使い、コールバック実装の本体は VoiceDialogRuntime / MmdRuntime に閉じる。
        """

        emit = self._emit
        mmd = self._mmd
        supervisor = self._supervisor
        event_queue = self._event_queue

        def _on_emit(ev: dict) -> None:
            emit(ev)

        def _on_nod_motion(ev: dict) -> None:
            emit(dict(ev))
            mmd.start_nod_motion(ev)

        def _on_trigger_llm(user_text: str, *, prefetch: bool = False) -> None:
            if self.settings.get("wav_send_other_ch_to_mmd"):
                return
            emit(
                {
                    "type": "trigger_llm",
                    "user_text": user_text,
                    "prefetch": bool(prefetch),
                    "ts": time.time(),
                }
            )
            self.llm_dispatch_thread(prefetch=bool(prefetch))

        def _on_request_force_finalize() -> None:
            supervisor.asr.send_command("force_finalize")

        def _on_request_tts_done() -> None:
            event_queue.put({"type": "tts_done", "playback_aborted": False})

        def _on_request_opening_tts(text: str) -> None:
            emit({"type": "request_opening_tts", "assistant_text": text})
            if self.settings.get("wav_send_other_ch_to_mmd"):
                return
            if text:
                def _on_done(aborted: bool) -> None:
                    self.fire_tts_done_safe(playback_aborted=aborted)
                self.speak_nonstream_async(
                    text, allow_inactive=False, silent=False, source="opening", on_done=_on_done,
                )

        def _on_request_nonstream_tts(text: str, source: str) -> None:
            emit({"type": "request_nonstream_tts", "assistant_text": text, "source": source})
            if self.settings.get("wav_send_other_ch_to_mmd"):
                return
            allow_inactive = source == "maai_backchannel"
            silent = bool(allow_inactive and bool(self.settings.get("maai_backchannel_silent", False)))
            inject_only = bool(allow_inactive and bool(self.settings.get("maai_backchannel_inject_only", False)))
            no_inject = bool(allow_inactive and bool(self.settings.get("maai_backchannel_no_inject", False)))
            if text:
                def _on_done(aborted: bool, _src: str = source) -> None:
                    self.fire_tts_done_safe(playback_aborted=aborted)
                self.speak_nonstream_async(
                    text,
                    allow_inactive=allow_inactive,
                    silent=silent,
                    inject_only=inject_only,
                    no_inject=no_inject,
                    source=source or "nonstream",
                    on_done=_on_done,
                )

        def _on_request_interrupt(reason: str) -> None:
            # 条件成立中は毎フレーム request_interrupt が来る。停止フラグ済みの再実行では
            # プロット用 INT マーカーを重ねない（1 回の割り込みバーストにつき 1 emit）。
            already = self._vd_interrupt_stop.is_set()
            if self.execute_user_interrupt(reason):
                if not already:
                    emit({"type": "latency", "kind": "interrupt", "end_ts": time.time()})

        def _on_request_scheduled_exit_tts(text: str) -> None:
            emit({"type": "request_scheduled_exit_tts", "assistant_text": text})
            self.set_scheduled_exit_pending(True)
            if self.settings.get("wav_send_other_ch_to_mmd"):
                if self.consume_scheduled_exit_pending():
                    event_queue.put({"type": "stop_voice_dialog"})
                return
            if text:
                def _on_done(aborted: bool) -> None:
                    self.fire_tts_done_safe(playback_aborted=aborted)
                    if self.consume_scheduled_exit_pending():
                        event_queue.put({"type": "stop_voice_dialog"})
                self.speak_nonstream_async(
                    text, allow_inactive=False, silent=False, source="scheduled_exit", on_done=_on_done,
                )

        def _on_commit_prefetch() -> None:
            self.commit_prefetch()

        def _on_cancel_prefetch() -> None:
            self.cancel_prefetch()

        return {
            "on_emit": _on_emit,
            "on_nod_motion": _on_nod_motion,
            "on_trigger_llm": _on_trigger_llm,
            "on_request_force_finalize": _on_request_force_finalize,
            "on_request_tts_done": _on_request_tts_done,
            "on_request_opening_tts": _on_request_opening_tts,
            "on_request_nonstream_tts": _on_request_nonstream_tts,
            "on_request_interrupt": _on_request_interrupt,
            "on_request_scheduled_exit_tts": _on_request_scheduled_exit_tts,
            "on_commit_prefetch": _on_commit_prefetch,
            "on_cancel_prefetch": _on_cancel_prefetch,
        }

    @property
    def vd_state(self) -> dict[str, object]:
        return self._orch_vd_tts_state

    @property
    def settings(self) -> dict[str, object]:
        return self._llm_settings

    def update_settings(self, settings: dict) -> None:
        self._llm_settings.update(settings)

    def notify_maai_system_inject_activity(self) -> None:
        """TTS/MaAI 向け system_audio 注入があった直後に呼ぶ（無音ティックの抑制用）。"""
        self._vd_maai_last_real_system_audio_ts = time.perf_counter()

    def _run_maai_system_audio_silence_tick_forever(self) -> None:
        _interval_sec = 0.02
        _gap_sec = 0.08
        _n_samples = max(1, 320)
        zero_chunk = b"\x00\x00" * _n_samples
        cmd = f"system_audio:{base64.b64encode(zero_chunk).decode('ascii')}"
        _err_count = 0
        while True:
            time.sleep(_interval_sec)
            if not self._vd_voice_dialog_running_for_maai_tick.is_set():
                _err_count = 0
                continue
            if (time.perf_counter() - float(self._vd_maai_last_real_system_audio_ts)) < _gap_sec:
                _err_count = 0
                continue
            try:
                self._supervisor.send_audio_input_command(cmd)
                _err_count = 0
            except Exception:
                _err_count += 1
                if _err_count % 100 == 1:
                    try:
                        self._emit({"type": "log", "message": f"[MaAI-tick] system_audio 送信失敗 (連続{_err_count}回)"})
                    except Exception:
                        pass

    def _vd_start_epoch_current(self, epoch: "int | None") -> bool:
        """この start の epoch がまだ最新か（stop / 新 start に上書きされていないか）。"""
        if epoch is None:
            return True
        with self._vd_lifecycle_lock:
            return epoch == self._vd_start_epoch

    def clear_interrupt_stop(self) -> None:
        self._vd_interrupt_stop.clear()

    def clear_nonstream_stop_flag(self) -> None:
        self._vd_nonstream_stop_flag.clear()

    def set_scheduled_exit_pending(self, flag: bool) -> None:
        self._vd_scheduled_exit_pending["flag"] = bool(flag)

    def consume_scheduled_exit_pending(self) -> bool:
        if self._vd_scheduled_exit_pending.get("flag"):
            self._vd_scheduled_exit_pending["flag"] = False
            return True
        return False

    def update_vad_from_maai_result(self, result_obj: dict) -> None:
        try:
            sv = result_obj.get("silero_vad_score")
            if sv is not None:
                self._orch_vd_vad_state["last_silero_vad"] = float(sv)
            pn = result_obj.get("p_now")
            if pn is not None:
                self._orch_vd_vad_state["last_p_now"] = float(pn)
            self._orch_vd_vad_state["last_update_ts"] = time.time()
        except (TypeError, ValueError):
            pass

    # -- latency --------------------------------------------------------------

    def reset_turn_latency(self) -> None:
        with self._vd_latency_lock:
            self._vd_turn_latency_state["llm_perf"] = None
            self._vd_turn_latency_state["llm_wall"] = None
            self._vd_turn_latency_state["tts_req_perf"] = None
            self._vd_turn_latency_state["tts_req_wall"] = None
            self._vd_turn_latency_state["llm_latency_band_emitted"] = False
            self._vd_turn_latency_state["tts_latency_band_emitted"] = False
        # 進行中の緑帯をプロットから消す。
        self._emit({"type": "latency", "kind": "llm", "value": None})

    def begin_llm_turn_for_latency(self) -> None:
        with self._vd_latency_lock:
            nowp = time.perf_counter()
            noww = time.time()
            self._vd_turn_latency_state["llm_perf"] = nowp
            self._vd_turn_latency_state["llm_wall"] = noww
            self._vd_turn_latency_state["tts_req_perf"] = None
            self._vd_turn_latency_state["tts_req_wall"] = None
            self._vd_turn_latency_state["llm_latency_band_emitted"] = False
            self._vd_turn_latency_state["tts_latency_band_emitted"] = False
        # start_ts 付き value=None でプロット側に「進行中の緑帯」を開く。
        self._emit({"type": "latency", "kind": "llm", "value": None, "start_ts": noww})
        self._emit({"type": "latency", "kind": "tts", "value": None})

    def llm_latency_already_armed(self) -> bool:
        with self._vd_latency_lock:
            return self._vd_turn_latency_state.get("llm_wall") is not None

    def close_llm_latency_band_only(self) -> None:
        """進行中の緑帯だけ閉じる（TTS 帯は開始しない）。先読みの最初の文を TTS に渡したとき用。"""
        with self._vd_latency_lock:
            if bool(self._vd_turn_latency_state.get("llm_latency_band_emitted")):
                return
            lp = self._vd_turn_latency_state["llm_perf"]
            lw = self._vd_turn_latency_state["llm_wall"]
            if lp is None or lw is None:
                return
            nowp = time.perf_counter()
            noww = time.time()
            self._vd_turn_latency_state["llm_latency_band_emitted"] = True
        self._emit(
            {
                "type": "latency",
                "kind": "llm",
                "value": max(0.0, nowp - float(lp)),
                "start_ts": float(lw),
                "end_ts": noww,
            }
        )

    def start_tts_latency_at_prefetch_handoff(self) -> None:
        """先読みの最初の文を TTS に渡した時点からマゼンタ（TTS）帯を測り始める。

        終わりは、確定前にその文の合成が終われば合成完了の時点（close_tts_latency_band_at_prefetch_synth）、
        合成中に確定されれば最初の音が出た時点（emit_tts_latency_on_first_audible）。
        先読みの帯は緑・マゼンタとも「ユーザーの発話中に裏で使った処理時間」を先読みの時点に描く。
        """
        with self._vd_latency_lock:
            if bool(self._vd_turn_latency_state.get("tts_latency_band_emitted")):
                return
            if self._vd_turn_latency_state["tts_req_perf"] is not None:
                return
            self._vd_turn_latency_state["tts_req_perf"] = time.perf_counter()
            self._vd_turn_latency_state["tts_req_wall"] = time.time()

    def close_tts_latency_band_at_prefetch_synth(self) -> None:
        """先読みの最初の文の合成が（確定前に）終わった時点で TTS 帯を閉じる。"""
        self.emit_tts_latency_on_first_audible()

    def begin_tts_latency_for_playback(self) -> None:
        """再生開始時点からマゼンタ（TTS）帯を測る。先読み確定後の再生用。"""
        with self._vd_latency_lock:
            if bool(self._vd_turn_latency_state.get("tts_latency_band_emitted")):
                return
            if self._vd_turn_latency_state["tts_req_perf"] is not None:
                return
            self._vd_turn_latency_state["tts_req_perf"] = time.perf_counter()
            self._vd_turn_latency_state["tts_req_wall"] = time.time()

    def note_assistant_tts_request_for_latency(self) -> None:
        with self._vd_latency_lock:
            if bool(self._vd_turn_latency_state.get("llm_latency_band_emitted")):
                # LLM 帯は先読みで既に閉じていることがある。TTS 開始だけ行う。
                if self._vd_turn_latency_state["tts_req_perf"] is None and not bool(
                    self._vd_turn_latency_state.get("tts_latency_band_emitted")
                ):
                    self._vd_turn_latency_state["tts_req_perf"] = time.perf_counter()
                    self._vd_turn_latency_state["tts_req_wall"] = time.time()
                return
            if self._vd_turn_latency_state["tts_req_perf"] is not None:
                return
            lp = self._vd_turn_latency_state["llm_perf"]
            lw = self._vd_turn_latency_state["llm_wall"]
            nowp = time.perf_counter()
            noww = time.time()
            self._vd_turn_latency_state["tts_req_perf"] = nowp
            self._vd_turn_latency_state["tts_req_wall"] = noww
            will_emit_llm_band = lp is not None and lw is not None
            if will_emit_llm_band:
                self._vd_turn_latency_state["llm_latency_band_emitted"] = True
        if lp is not None and lw is not None:
            self._emit(
                {
                    "type": "latency",
                    "kind": "llm",
                    "value": max(0.0, nowp - float(lp)),
                    "start_ts": float(lw),
                    "end_ts": noww,
                }
            )

    def emit_tts_latency_on_first_audible(self) -> None:
        with self._vd_latency_lock:
            if bool(self._vd_turn_latency_state.get("tts_latency_band_emitted")):
                return
            trp = self._vd_turn_latency_state["tts_req_perf"]
            trw = self._vd_turn_latency_state["tts_req_wall"]
            if trp is None or trw is None:
                return
            nowp = time.perf_counter()
            noww = time.time()
            self._vd_turn_latency_state["tts_req_perf"] = None
            self._vd_turn_latency_state["tts_req_wall"] = None
            self._vd_turn_latency_state["tts_latency_band_emitted"] = True
        self._emit(
            {
                "type": "latency",
                "kind": "tts",
                "value": max(0.0, nowp - float(trp)),
                "start_ts": float(trw),
                "end_ts": noww,
            }
        )

    # -- nonstream ref / VAD interrupt ----------------------------------------

    def silero_vad_at_or_above_interrupt_threshold(self) -> bool:
        v = self._orch_vd_vad_state.get("last_silero_vad")
        if v is None:
            return False
        try:
            thr = float(self._llm_settings.get("interrupt_vad_threshold", 0.0) or 0.0)
        except (TypeError, ValueError):
            thr = 0.0
        if thr <= 0.0:
            return False
        try:
            return float(v) >= thr
        except (TypeError, ValueError):
            return False

    def nonstream_begin(self, *, backchannel: bool = False) -> None:
        self._vd_nonstream_refs["n"] = int(self._vd_nonstream_refs.get("n", 0) or 0) + 1
        if backchannel:
            self._vd_nonstream_refs["bc"] = int(self._vd_nonstream_refs.get("bc", 0) or 0) + 1

    def nonstream_end(self, *, backchannel: bool = False) -> None:
        self._vd_nonstream_refs["n"] = max(0, int(self._vd_nonstream_refs.get("n", 0) or 0) - 1)
        if backchannel:
            self._vd_nonstream_refs["bc"] = max(0, int(self._vd_nonstream_refs.get("bc", 0) or 0) - 1)

    def execute_user_interrupt(self, reason: str) -> bool:
        """Orchestrator の request_interrupt を worker 側で処理する（音声・LLM・キューを止める）。

        Controller の _voice_dialog_interrupt_system_speech に相当する。GUI 側は _llm_pending を
        持たないため、イベントを上流 emit だけにしても割り込みは効かない。

        未確定の先読みだけがある場合は actionable にしない（ユーザー発話中の p_future
        先読みを毎フレームの interrupt で潰さない）。実際にシステム発話側を止めるときだけ
        先読みも一緒に破棄する。
        """
        _ = reason
        if not bool(self._llm_settings.get("interrupt_on_user_speech", True)):
            return False
        # 条件成立中は毎フレーム来る。既に停止済みなら音声/LLM 再処理しない。
        if self._vd_interrupt_stop.is_set():
            return False
        normal_snaps: list[tuple[str, dict]] = []
        prefetch_snaps: list[tuple[str, dict]] = []
        with self._llm_pending_lock:
            for rid in list(self._llm_pending.keys()):
                raw = self._llm_pending.get(rid)
                if not isinstance(raw, dict):
                    self._llm_pending.pop(rid, None)
                    continue
                is_uncommitted_prefetch = bool(raw.get("prefetch")) and not bool(raw.get("committed"))
                if is_uncommitted_prefetch:
                    prefetch_snaps.append((str(rid), dict(raw)))
                else:
                    normal_snaps.append((str(rid), dict(raw)))

        # 未確定の先読みは再生せずに保持する合成なので、システム発話として数えない。
        prefetch_ids = {rid for rid, _st in prefetch_snaps}
        with self._vd_tts_jobs_lock:
            held = sum(self._vd_tts_inflight_by_rid.get(rid, 0) for rid in prefetch_ids)
            inflight_seg = max(0, int(self._vd_tts_jobs.get("inflight") or 0) - held)
        tail_p = bool(self._orch_vd_tts_state.get("tail_pending"))
        # 相槌はユーザー発話に重ねて出すものなので、ユーザー発話による割り込みの対象にしない。
        nons = int(self._vd_nonstream_refs.get("n") or 0) - int(self._vd_nonstream_refs.get("bc") or 0) > 0

        actionable = bool(normal_snaps) or inflight_seg > 0 or tail_p or nons
        if not actionable:
            return False

        snaps = normal_snaps + prefetch_snaps
        with self._llm_pending_lock:
            for rid, _st in snaps:
                self._remember_llm_terminal_locked(rid)
                self._pop_llm_pending_locked(rid)

        self._vd_interrupt_stop.set()
        self._vd_nonstream_stop_flag.set()
        self._discard_pending_tts_segments()
        with self._tail_done_timer_lock:
            if self._tail_done_timer is not None:
                self._tail_done_timer.cancel()
                self._tail_done_timer = None
        try:
            self._supervisor.send_audio_output({"type": "interrupt"})
        except Exception:
            pass

        try:
            if self._orchestrator is not None:
                self._orchestrator.clear_prefetch()
                self._orchestrator.on_llm_state(False)
        except Exception:
            pass

        for rid, _st in snaps:
            try:
                self._supervisor.send_llm({"type": "cancel", "request_id": str(rid).strip()})
            except Exception:
                pass

        if snaps:
            self.reset_turn_latency()
            merged_text = str(snaps[-1][1].get("assistant_text", "") or "").strip()
            try:
                if self._orchestrator is not None:
                    self._orchestrator.on_llm_finished(
                        ok=False,
                        assistant_text=merged_text,
                        error="interrupt",
                    )
            except Exception:
                pass

        try:
            self.fire_tts_done_safe(True)
        except Exception:
            pass

        self._emit(
            {
                "type": "log",
                "message": "[VoiceDialog] ユーザー発話検知(VAD/p_now): システム発話パイプラインを中断しました。",
            }
        )
        return True

    def cancel_prefetch(self) -> None:
        """未確定の先読み LLM/保持 PCM を破棄する（p_future 下降・失敗フォールバック）。"""
        snaps: list[tuple[str, dict]] = []
        with self._llm_pending_lock:
            for rid in list(self._llm_pending.keys()):
                raw = self._llm_pending.get(rid)
                if not isinstance(raw, dict):
                    continue
                if bool(raw.get("prefetch")) and not bool(raw.get("committed")):
                    self._remember_llm_terminal_locked(str(rid))
                    popped = self._pop_llm_pending_locked(rid)
                    if isinstance(popped, dict):
                        snaps.append((str(rid), popped))
        for rid, _st in snaps:
            try:
                self._supervisor.send_llm({"type": "cancel", "request_id": str(rid).strip()})
            except Exception:
                pass
        if snaps:
            self.reset_turn_latency()
            self._emit(
                {
                    "type": "log",
                    "message": f"[LLM][PREFETCH] discarded request_id={snaps[-1][0]}",
                }
            )

    def commit_prefetch(self) -> None:
        """p_now ターン取得: 先読みを確定し、保持 PCM を再生する。

        遅延の緑帯は p_future 先読み開始時に既に開始済み。ここではリセットしない。
        合成がまだなら二重生成せず、完了した PCM から順に再生する。
        """
        request_id = ""
        llm_done = False
        llm_done_ok = False
        assistant_text = ""
        synth_complete = False
        with self._llm_pending_lock:
            for rid, st in self._llm_pending.items():
                if not isinstance(st, dict):
                    continue
                if bool(st.get("prefetch")):
                    st["committed"] = True
                    request_id = str(rid)
                    llm_done = bool(st.get("llm_done"))
                    llm_done_ok = bool(st.get("llm_done_ok", True))
                    assistant_text = str(st.get("assistant_text", "") or "")
                    synth_complete = bool(st.get("synth_complete"))
                    break
        if not request_id:
            # 先読みが既に消えていた（失敗・cancel レース）。ターンは取得済みなので通常 LLM を起動する。
            self._emit(
                {
                    "type": "log",
                    "message": "[LLM][PREFETCH] commit missed pending; starting normal LLM",
                }
            )
            self.llm_dispatch_thread(prefetch=False)
            return

        self._vd_interrupt_stop.clear()
        # 先読み開始で既に緑帯を開いているときは上書きしない。
        if not self.llm_latency_already_armed():
            self.begin_llm_turn_for_latency()

        if llm_done:
            try:
                if self._orchestrator is not None:
                    self._orchestrator.on_llm_finished(
                        ok=bool(llm_done_ok),
                        assistant_text=assistant_text,
                        error="" if llm_done_ok else "prefetch_llm_failed",
                    )
            except Exception:
                pass

        self._flush_held_prefetch_pcm(request_id)

        if llm_done and synth_complete:
            self._arm_prefetch_tail_if_ready(request_id)

    # -- tts done callbacks ---------------------------------------------------

    def fire_tts_done_safe(self, playback_aborted: bool = False) -> None:
        try:
            if self._orchestrator is not None:
                self._orchestrator.on_tts_done(playback_aborted=bool(playback_aborted))
        except Exception:
            pass
        with self._vd_tts_tail_lock:
            self._orch_vd_tts_state["tail_pending"] = False

    def llm_gui_test_fire_tail_done(self) -> None:
        """GUI LLMテスト: tail 再生完了のみ（DialogOrchestrator.on_tts_done は呼ばない）。"""
        with self._vd_tts_tail_lock:
            self._orch_vd_tts_state["tail_pending"] = False
        self._orch_vd_tts_state["suppress_orch"] = False

    # -- segment playback -----------------------------------------------------

    def _ensure_vd_tts_seg_worker(self) -> None:
        if self._vd_tts_seg_worker is not None and self._vd_tts_seg_worker.is_alive():
            return
        t = threading.Thread(
            target=self._vd_tts_seg_worker_loop, daemon=True, name="vd-tts-seg-worker"
        )
        self._vd_tts_seg_worker = t
        t.start()

    def _discard_pending_tts_segments(self) -> None:
        """割り込み/停止時に保留中セグメント合成ジョブを破棄し inflight を 0 に戻す。
        放置すると次ターンが _vd_interrupt_stop をクリアした瞬間、旧ターンのキュー残りが
        合成・再生されうる(#5)。また _after_llm_done の inflight 待ちが旧ジョブを数えてしまう。"""
        try:
            while True:
                self._vd_tts_seg_queue.get_nowait()
        except queue.Empty:
            pass
        with self._vd_tts_jobs_lock:
            self._vd_tts_jobs["inflight"] = 0
            self._vd_tts_inflight_by_rid.clear()
            self._vd_tts_segments_done.notify_all()

    def _vd_tts_seg_worker_loop(self) -> None:
        while True:
            item = self._vd_tts_seg_queue.get()
            if item is None:
                break
            request_id, segment_text, is_final = item
            try:
                if not self._vd_interrupt_stop.is_set():
                    self._llm_play_tts_segment(request_id, segment_text, is_final=is_final)
            except Exception:
                pass
            finally:
                with self._vd_tts_jobs_lock:
                    self._vd_tts_jobs["inflight"] = max(
                        0, int(self._vd_tts_jobs.get("inflight", 0) or 0) - 1,
                    )
                    left = self._vd_tts_inflight_by_rid.get(request_id, 0) - 1
                    if left > 0:
                        self._vd_tts_inflight_by_rid[request_id] = left
                    else:
                        self._vd_tts_inflight_by_rid.pop(request_id, None)
                    self._vd_tts_segments_done.notify_all()

    def _synthesize_segment_pcm_l16le(
        self, engine: str, tin: dict, txt: str, *, hold: bool, inject: bool
    ) -> bytes | None:
        """セグメントを合成して L16LE PCM を返す。hold=False のストリームエンジンは再生しながら集約する。"""
        if tps.is_binary_stream_engine(engine):
            parts: list[bytes] = []
            _seg_audible = [False]

            def _on_seg_chunk(chunk_l16be: bytes, _meta: dict) -> None:
                if self._vd_interrupt_stop.is_set():
                    return
                chunk_l16le = tps.to_l16le(chunk_l16be)
                parts.append(chunk_l16le)
                if hold:
                    return
                if not _seg_audible[0]:
                    try:
                        chunk_max = int(audioop.max(chunk_l16le, 2))
                    except Exception:
                        chunk_max = 0
                    if chunk_max > 120:
                        _seg_audible[0] = True
                        self._orch_vd_tts_state["first_audible"] = True
                        suppress_orch = bool(self._orch_vd_tts_state.get("suppress_orch"))
                        try:
                            if not suppress_orch:
                                self._event_queue.put({"type": "tts_audible"})
                                self._emit({"type": "tts_audible"})
                                self.emit_tts_latency_on_first_audible()
                        except Exception:
                            pass
                for sub in tps.split_pcm_into_frames(chunk_l16le):
                    self._tts_router.route_pcm_l16le(sub, inject_maai=inject)

            try:
                self._tts_router.synthesize_l16be(engine, tin, txt, on_chunk=_on_seg_chunk)
            except Exception as ex:
                self._emit({"type": "log", "message": f"[VD-TTS] synthesize failed: {ex}"})
                return None
            return b"".join(parts)

        try:
            audio_l16be, _v = self._tts_router.synthesize_l16be(engine, tin, txt)
        except Exception as ex:
            self._emit({"type": "log", "message": f"[VD-TTS] synthesize failed: {ex}"})
            return None
        return swap_pcm16_endian(audio_l16be)

    def _route_pcm_l16le_with_audible(self, chunk_le: bytes, *, inject: bool) -> None:
        """保持 PCM / 非ストリーム合成結果をペーサへ送り、初回可聴でレイテンシを確定する。"""
        sub_bytes = 640
        pos = 0
        seg_announced_audible = False
        while pos < len(chunk_le):
            if self._vd_interrupt_stop.is_set():
                break
            sub = chunk_le[pos : pos + sub_bytes]
            pos += len(sub)
            if not sub:
                continue
            try:
                sub_max = int(audioop.max(sub, 2))
            except Exception:
                sub_max = 0
            if (not seg_announced_audible) and sub_max > 120:
                seg_announced_audible = True
                self._orch_vd_tts_state["first_audible"] = True
                suppress_orch = bool(self._orch_vd_tts_state.get("suppress_orch"))
                try:
                    if not suppress_orch:
                        self._event_queue.put({"type": "tts_audible"})
                        self._emit({"type": "tts_audible"})
                        self.emit_tts_latency_on_first_audible()
                except Exception:
                    pass
            self._tts_router.route_pcm_l16le(sub, inject_maai=inject)

    def _append_segment_and_break(
        self,
        request_id: str,
        segment_text: str,
        *,
        is_final: bool,
        tin: dict,
        inject: bool,
        br_gen: int,
    ) -> None:
        """合成成功後の履歴セグメント追記と segment break。"""
        seg_index = -1
        with self._llm_pending_lock:
            if request_id not in self._llm_pending or self._vd_interrupt_stop.is_set():
                return
            with self._vd_tts_seg_lock:
                lst = self._orch_vd_tts_state.get("segments")
                if not isinstance(lst, list):
                    lst = []
                lst.append(str(segment_text or ""))
                self._orch_vd_tts_state["segments"] = lst
                seg_index = len(lst) - 1

        if not is_final:
            gap_sec = float(tin.get("segment_gap_silence_sec") or 0.0)
            if gap_sec > 0.0:
                for sub in tps.split_pcm_into_frames(tps.gap_silence_pcm_l16le(gap_sec)):
                    self._tts_router.route_pcm_l16le(sub, inject_maai=inject)
            self._tts_router.enqueue_break(
                {
                    "type": "tts_play_break",
                    "generation": br_gen,
                    "complete_kind": "segment",
                    "segment_index": seg_index,
                },
            )

    def _flush_held_prefetch_pcm(self, request_id: str) -> None:
        """確定済み先読みの未再生 PCM を順に再生する。"""
        while True:
            item: dict | None = None
            inject = True
            tin: dict = {}
            do_note = False
            with self._llm_pending_lock:
                st = self._llm_pending.get(request_id)
                if not isinstance(st, dict) or not bool(st.get("committed")):
                    return
                if self._vd_interrupt_stop.is_set():
                    return
                held = st.get("held_pcm")
                if not isinstance(held, list):
                    held = []
                    st["held_pcm"] = held
                flushed = int(st.get("held_flushed", 0) or 0)
                if flushed >= len(held):
                    return
                item = held[flushed]
                st["held_flushed"] = flushed + 1
                inject = bool(st.get("voice_dialog_inject_ch2", True))
                tin = dict(st.get("tts_stream_inputs") or {})
                if not bool(st.get("latency_tts_noted")):
                    st["latency_tts_noted"] = True
                    do_note = True
            if not isinstance(item, dict):
                continue
            pcm = item.get("pcm")
            if not isinstance(pcm, (bytes, bytearray)) or not pcm:
                continue
            if do_note:
                # 緑帯は先読み中に閉じ済みのことが多い。再生開始から TTS 帯を測る。
                self.begin_tts_latency_for_playback()
                self.note_assistant_tts_request_for_latency()
            self._route_pcm_l16le_with_audible(bytes(pcm), inject=inject)
            self._append_segment_and_break(
                request_id,
                str(item.get("text", "") or ""),
                is_final=bool(item.get("is_final", False)),
                tin=tin,
                inject=inject,
                br_gen=int(item.get("br_gen", 0) or 0),
            )

    def _arm_prefetch_tail_if_ready(self, request_id: str) -> None:
        """先読み LLM+合成が終わり、確定済みなら tail break をアームする。"""
        with self._vd_lifecycle_lock:
            turn_epoch = self._vd_start_epoch
        with self._llm_pending_lock:
            st = self._llm_pending.get(request_id)
            if not isinstance(st, dict):
                return
            if not (bool(st.get("committed")) and bool(st.get("llm_done")) and bool(st.get("synth_complete"))):
                return
            if bool(st.get("tail_armed")):
                return
            st["tail_armed"] = True
        armed_tg = None
        with self._vd_tts_tail_lock:
            if (not self._vd_interrupt_stop.is_set()) and self._vd_start_epoch_current(turn_epoch):
                armed_tg = int(self._orch_vd_tts_state.get("tail_gen", 0) or 0) + 1
                self._orch_vd_tts_state["tail_gen"] = armed_tg
                self._orch_vd_tts_state["tail_pending"] = True
        if armed_tg is not None:
            self._tts_router.enqueue_break({"type": "tts_play_break", "generation": armed_tg})
        with self._llm_pending_lock:
            self._pop_llm_pending_locked(request_id)

    def _llm_play_tts_segment(self, request_id: str, segment_text: str, *, is_final: bool) -> None:
        txt = str(segment_text or "").strip()
        if not txt:
            return
        with self._llm_pending_lock:
            st = self._llm_pending.get(request_id)
            if st is None:
                return
            engine = str(st.get("tts_engine", "")).strip()
            tin = dict(st.get("tts_stream_inputs") or {})
            inject = bool(st.get("voice_dialog_inject_ch2", True))
            st["tts_seg_break_gen"] = int(st.get("tts_seg_break_gen", 0)) + 1
            br_gen = int(st["tts_seg_break_gen"])
            hold = bool(st.get("prefetch")) and not bool(st.get("committed"))
            do_note = False
            if (not hold) and bool(st.get("prefetch")) and bool(st.get("committed")):
                if not bool(st.get("latency_tts_noted")):
                    st["latency_tts_noted"] = True
                    do_note = True
        if not engine:
            return

        if self._vd_interrupt_stop.is_set():
            return

        if do_note:
            self.note_assistant_tts_request_for_latency()

        # hold=True: ストリームでも再生せず PCM だけ集約。合成中に commit されても
        # 合成後に hold 再判定し、必要ならここで再生する。
        pcm = self._synthesize_segment_pcm_l16le(engine, tin, txt, hold=hold, inject=inject)
        if pcm is None:
            return

        with self._llm_pending_lock:
            st = self._llm_pending.get(request_id)
            if st is None or self._vd_interrupt_stop.is_set():
                return
            if bool(st.get("prefetch")) and not bool(st.get("committed")):
                held = st.get("held_pcm")
                if not isinstance(held, list):
                    held = []
                    st["held_pcm"] = held
                first_held = len(held) == 0
                held.append(
                    {
                        "pcm": bytes(pcm),
                        "text": str(segment_text or ""),
                        "is_final": bool(is_final),
                        "br_gen": br_gen,
                    }
                )
                hold_now = True
            else:
                first_held = False
                hold_now = False

        if hold_now:
            if first_held:
                # 通常は最初の文を TTS に渡した時点で閉じ済み（その場合は何もしない）。
                self.close_llm_latency_band_only()
                # 最初の文の合成が確定前に終わった。ここでマゼンタ帯を閉じる（先読みの時点に描く）。
                self.close_tts_latency_band_at_prefetch_synth()
            return

        # 合成中に再生していない場合（hold 開始だった／非ストリーム）はここで送出。
        played_during_synth = (not hold) and tps.is_binary_stream_engine(engine)
        if not played_during_synth:
            self._route_pcm_l16le_with_audible(pcm, inject=inject)

        self._append_segment_and_break(
            request_id, segment_text, is_final=is_final, tin=tin, inject=inject, br_gen=br_gen
        )

    def on_mmd_send_complete(self, comp: dict[str, object]) -> None:
        suppress = bool(self._orch_vd_tts_state.get("suppress_orch"))
        ck = str(comp.get("complete_kind") or "").strip().lower()
        with self._vd_tts_seg_lock:
            segs = self._orch_vd_tts_state.get("segments")
            seg_list: list[str] = [str(x) for x in segs] if isinstance(segs, list) else []
        if ck == "segment":
            if suppress:
                return
            try:
                si = int(comp.get("segment_index", 0) or 0)
            except (TypeError, ValueError):
                si = 0
            cumulative = "".join(seg_list[: si + 1])
            try:
                if self._orchestrator is not None:
                    self._orchestrator.on_assistant_segment_played(cumulative)
            except Exception:
                pass
            return
        try:
            gen = int(comp.get("generation", 0) or 0)
            with self._vd_tts_tail_lock:
                tg = int(self._orch_vd_tts_state.get("tail_gen") or 0)
                matched = gen == tg and bool(self._orch_vd_tts_state.get("tail_pending"))
        except (TypeError, ValueError):
            return
        if matched:
            tail_done = self.llm_gui_test_fire_tail_done if suppress else self.fire_tts_done_safe
            try:
                burst_ms = int(comp.get("burst_compensation_ms", 0) or 0)
            except (TypeError, ValueError):
                burst_ms = 0
            burst_ms = max(0, min(600_000, burst_ms))
            delay_sec = burst_ms / 1000.0
            if delay_sec > 0:
                t = threading.Timer(delay_sec, tail_done)
                with self._tail_done_timer_lock:
                    if self._tail_done_timer is not None:
                        self._tail_done_timer.cancel()
                    self._tail_done_timer = t
                t.start()
            else:
                with self._tail_done_timer_lock:
                    if self._tail_done_timer is not None:
                        self._tail_done_timer.cancel()
                    self._tail_done_timer = None
                tail_done()

    # -- LLM worker event handler ---------------------------------------------

    @staticmethod
    def _pending_is_gui_test(st: dict | None) -> bool:
        return isinstance(st, dict) and str(st.get("source", "")).strip() == "llm_gui_test"

    @staticmethod
    def _cancel_llm_watchdog_locked(state: dict | None) -> None:
        """Cancel a pending watchdog while the caller holds ``_llm_pending_lock``."""
        if not isinstance(state, dict):
            return
        timer = state.pop("_llm_watchdog_timer", None)
        cancel = getattr(timer, "cancel", None)
        if callable(cancel):
            cancel()

    def _remember_llm_terminal_locked(self, request_id: str) -> None:
        """Remember recent terminal requests so late worker events are harmless."""
        terminal_ids = getattr(self, "_llm_terminal_ids", None)
        if not isinstance(terminal_ids, dict):
            terminal_ids = {}
            self._llm_terminal_ids = terminal_ids
        terminal_ids[request_id] = None
        while len(terminal_ids) > 256:
            terminal_ids.pop(next(iter(terminal_ids)))

    def _was_llm_terminal_locked(self, request_id: str) -> bool:
        terminal_ids = getattr(self, "_llm_terminal_ids", None)
        return isinstance(terminal_ids, dict) and request_id in terminal_ids

    def _set_llm_pending_locked(self, request_id: str, state: dict) -> None:
        previous = self._llm_pending.get(request_id)
        self._cancel_llm_watchdog_locked(previous)
        terminal_ids = getattr(self, "_llm_terminal_ids", None)
        if isinstance(terminal_ids, dict):
            terminal_ids.pop(request_id, None)
        self._llm_pending[request_id] = state

    def _pop_llm_pending_locked(self, request_id: str) -> dict | None:
        state = self._llm_pending.pop(request_id, None)
        self._cancel_llm_watchdog_locked(state)
        return state if isinstance(state, dict) else None

    def _llm_watchdog_seconds(self) -> float:
        raw = self._llm_settings.get("llm_watchdog_sec", 20.0)
        try:
            watchdog_sec = float(raw)
        except (TypeError, ValueError):
            watchdog_sec = 20.0
        return watchdog_sec

    def _new_llm_watchdog_timer(
        self,
        delay_sec: float,
        request_id: str,
        token: object,
        watchdog_sec: float,
    ) -> threading.Timer:
        timer = threading.Timer(
            delay_sec,
            self._check_llm_watchdog,
            args=(request_id, token, watchdog_sec),
        )
        timer.daemon = True
        return timer

    def _arm_llm_watchdog(self, request_id: str, state: dict) -> None:
        watchdog_sec = self._llm_watchdog_seconds()
        if not watchdog_sec > 0.0 or self._pending_is_gui_test(state):
            return
        token = object()
        timer = self._new_llm_watchdog_timer(watchdog_sec, request_id, token, watchdog_sec)
        with self._llm_pending_lock:
            current = self._llm_pending.get(request_id)
            if current is not state or current.get("_llm_terminal"):
                return
            current.setdefault("last_activity", time.perf_counter())
            current["_llm_watchdog_token"] = token
            current["_llm_watchdog_timer"] = timer
        timer.start()

    def _check_llm_watchdog(
        self,
        request_id: str,
        token: object,
        watchdog_sec: float,
    ) -> None:
        next_timer: threading.Timer | None = None
        timed_out = False
        with self._llm_pending_lock:
            state = self._llm_pending.get(request_id)
            if (
                not isinstance(state, dict)
                or state.get("_llm_watchdog_token") is not token
                or state.get("_llm_terminal")
                or self._pending_is_gui_test(state)
            ):
                return
            now = time.perf_counter()
            try:
                last_activity = float(state.get("last_activity", now))
            except (TypeError, ValueError):
                last_activity = now
            remaining = watchdog_sec - max(0.0, now - last_activity)
            if remaining > 0.0:
                next_timer = self._new_llm_watchdog_timer(
                    max(0.001, remaining),
                    request_id,
                    token,
                    watchdog_sec,
                )
                state["_llm_watchdog_timer"] = next_timer
            else:
                # Claim terminal handling before leaving the lock. A simultaneous
                # done/error observes this marker and becomes a no-op.
                state["_llm_terminal"] = "watchdog"
                state["_llm_watchdog_timer"] = None
                self._remember_llm_terminal_locked(request_id)
                timed_out = True
        if next_timer is not None:
            next_timer.start()
            return
        if not timed_out:
            return
        try:
            self._supervisor.send_llm({"type": "cancel", "request_id": request_id})
        except Exception:
            pass
        try:
            self._emit(
                {
                    "type": "log",
                    "message": (
                        f"[LLM][WATCHDOG] no response for {watchdog_sec:g} s; "
                        f"giving up request_id={request_id}"
                    ),
                }
            )
        finally:
            self.handle_llm_worker_event(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": "llm_timeout",
                    "_watchdog_token": token,
                }
            )

    def handle_llm_worker_event(self, payload: dict[str, object]) -> None:
        """LLM ワーカーからのイベントを処理する（新モードのみ）。"""

        etype = str(payload.get("type", "")).strip().lower()
        if etype == "ready":
            return
        if etype == "log":
            msg = str(payload.get("message", "")).strip()
            if msg:
                self._emit({"type": "log", "message": f"[LLMWorker] {msg}"})
            return
        request_id = str(payload.get("request_id", "")).strip()
        if request_id:
            with self._llm_pending_lock:
                wu_st = self._llm_pending.get(request_id)
            if isinstance(wu_st, dict) and str(wu_st.get("source", "")).strip() == "llm_warmup":
                if etype in {"warmup_done", "error"}:
                    if etype == "error" or not bool(payload.get("ok", True)):
                        wu_st["error"] = (
                            str(payload.get("message", "") or "").strip() or "LLM warmup failed"
                        )
                    done_ev = wu_st.get("done")
                    if isinstance(done_ev, threading.Event):
                        done_ev.set()
                return
        if not request_id:
            return
        if etype == "error":
            watchdog_token = payload.get("_watchdog_token")
            with self._llm_pending_lock:
                current = self._llm_pending.get(request_id)
                was_terminal = self._was_llm_terminal_locked(request_id)
                if isinstance(current, dict):
                    terminal = str(current.get("_llm_terminal", ""))
                    if watchdog_token is not None:
                        if terminal != "watchdog" or current.get("_llm_watchdog_token") is not watchdog_token:
                            popped = None
                        else:
                            popped = self._pop_llm_pending_locked(request_id)
                    elif terminal:
                        popped = None
                    else:
                        current["_llm_terminal"] = "error"
                        self._remember_llm_terminal_locked(request_id)
                        popped = self._pop_llm_pending_locked(request_id)
                else:
                    popped = None
            error_msg = str(payload.get("message", "")).strip() or "不明なエラー"
            if popped is None:
                if watchdog_token is not None or was_terminal or error_msg.strip().lower() == "cancelled":
                    return
                self._emit({**payload, "_src": "llm_raw"})
                return
            if self._pending_is_gui_test(popped):
                self.reset_turn_latency()
                self._orch_vd_tts_state["suppress_orch"] = False
                self._emit(
                    {
                        "type": "llm_test_finished",
                        "ok": False,
                        "request_id": request_id,
                        "backend": str(popped.get("backend", "")),
                        "model": str(popped.get("model", "")),
                        "context_len": int(popped.get("context_len", 10) or 10),
                        "assistant_text": str(popped.get("assistant_text", "") or ""),
                        "error": error_msg,
                        "ttft_sec": 0.0,
                        "total_sec": 0.0,
                        "total_overall_sec": 0.0,
                    }
                )
                return
            if bool(popped.get("prefetch")) and not bool(popped.get("committed")):
                if self._orchestrator is not None:
                    try:
                        self._orchestrator.mark_prefetch_failed()
                    except Exception:
                        pass
                self._emit(
                    {
                        "type": "log",
                        "message": f"[LLM][PREFETCH] failed: {error_msg}",
                    }
                )
                return
            self.reset_turn_latency()
            if self._orchestrator is not None:
                self._orchestrator.on_llm_finished(
                    ok=False,
                    assistant_text=str(popped.get("assistant_text", "") or ""),
                    error=error_msg,
                )
            self._emit({"type": "llm_done", "ok": False, "error": error_msg, "request_id": request_id})
            return

        with self._llm_pending_lock:
            state = self._llm_pending.get(request_id)
            was_terminal = self._was_llm_terminal_locked(request_id)
            if (
                isinstance(state, dict)
                and etype in {"chunk", "tts_segment"}
                and not self._pending_is_gui_test(state)
                and not state.get("_llm_terminal")
            ):
                state["last_activity"] = time.perf_counter()
        if state is None:
            if was_terminal:
                return
            self._emit({**payload, "_src": "llm_raw"})
            return
        if state.get("_llm_terminal"):
            return
        is_gui = self._pending_is_gui_test(state)
        if etype == "chunk":
            if self._vd_interrupt_stop.is_set():
                return
            text = str(payload.get("text", ""))
            if text:
                with self._llm_pending_lock:
                    st_mut = self._llm_pending.get(request_id)
                    if isinstance(st_mut, dict):
                        st_mut["assistant_text"] = str(st_mut.get("assistant_text", "")) + text
                        at = str(st_mut.get("assistant_text", ""))
                    else:
                        at = ""
                if at:
                    if is_gui:
                        self._emit(
                            {
                                "type": "llm_test_chat_chunk",
                                "text": text,
                                "assistant_text": at,
                                "request_id": request_id,
                            }
                        )
                    else:
                        self._emit({"type": "llm_chunk", "assistant_text": at, "request_id": request_id})
            return
        if etype == "tts_segment":
            if self._vd_interrupt_stop.is_set():
                return
            segment_text = str(payload.get("segment_text", ""))
            is_final = bool(payload.get("is_final", False))
            if segment_text.strip():
                prefetch_hold = False
                with self._llm_pending_lock:
                    st_seg0 = self._llm_pending.get(request_id)
                    if isinstance(st_seg0, dict):
                        prefetch_hold = bool(st_seg0.get("prefetch")) and not bool(
                            st_seg0.get("committed")
                        )
                if prefetch_hold:
                    # 先読みも通常ターンと同じく、最初の文を TTS に渡した時点で緑帯を閉じ、
                    # 同じ時点からマゼンタ帯を測り始める（終わりは合成完了か最初の音）。
                    self.close_llm_latency_band_only()
                    self.start_tts_latency_at_prefetch_handoff()
                else:
                    self.note_assistant_tts_request_for_latency()
                self._emit({"type": "llm_tts_segment", "segment_text": segment_text,
                      "is_final": is_final, "request_id": request_id})
                if is_gui:
                    cum = ""
                    with self._llm_pending_lock:
                        st_seg = self._llm_pending.get(request_id)
                        if isinstance(st_seg, dict):
                            cum = str(st_seg.get("assistant_text", "") or "")
                    self._emit(
                        {
                            "type": "llm_test_tts_segment",
                            "cumulative": cum,
                            "pushed_segment": segment_text.strip(),
                            "request_id": request_id,
                            "is_final": is_final,
                        }
                    )

                with self._vd_tts_jobs_lock:
                    self._vd_tts_jobs["inflight"] = int(self._vd_tts_jobs.get("inflight", 0) or 0) + 1
                    self._vd_tts_inflight_by_rid[request_id] = self._vd_tts_inflight_by_rid.get(request_id, 0) + 1
                # FIFO キュー＋単一ワーカーで発話順どおり逐次処理（Lock 取得順依存を排除）。
                self._ensure_vd_tts_seg_worker()
                self._vd_tts_seg_queue.put((request_id, segment_text, is_final))
            return
        if etype == "done":
            with self._llm_pending_lock:
                st_done = self._llm_pending.get(request_id)
                if not isinstance(st_done, dict) or st_done.get("_llm_terminal"):
                    return
                st_done["_llm_terminal"] = "done"
                self._cancel_llm_watchdog_locked(st_done)
                self._remember_llm_terminal_locked(request_id)
            is_done_gui = self._pending_is_gui_test(st_done)
            tts_via_segments = bool(str(st_done.get("tts_engine", "")).strip())
            assistant_text = str(payload.get("assistant_text", "")) or str(
                st_done.get("assistant_text", ""),
            )
            prefetch_uncommitted = bool(st_done.get("prefetch")) and not bool(st_done.get("committed"))
            prefetch_committed = bool(st_done.get("prefetch")) and bool(st_done.get("committed"))
            if prefetch_uncommitted:
                with self._llm_pending_lock:
                    st_pf = self._llm_pending.get(request_id)
                    if isinstance(st_pf, dict):
                        st_pf["llm_done"] = True
                        st_pf["llm_done_ok"] = True
                        st_pf["assistant_text"] = assistant_text
                self._emit(
                    {
                        "type": "log",
                        "message": (
                            "[LLM][PREFETCH] llm_done "
                            f"request_id={request_id} "
                            f"text_len={len(assistant_text.strip())}"
                        ),
                    }
                )

                def _after_prefetch_llm_done() -> None:
                    deadline = time.monotonic() + 120.0
                    with self._vd_tts_jobs_lock:
                        while int(self._vd_tts_jobs.get("inflight", 0) or 0) > 0:
                            rem = deadline - time.monotonic()
                            if rem <= 0:
                                break
                            self._vd_tts_segments_done.wait(timeout=min(1.0, rem))
                    with self._llm_pending_lock:
                        st_fin = self._llm_pending.get(request_id)
                        if not isinstance(st_fin, dict):
                            return
                        st_fin["synth_complete"] = True
                        committed_now = bool(st_fin.get("committed"))
                    if committed_now:
                        self._flush_held_prefetch_pcm(request_id)
                        self._arm_prefetch_tail_if_ready(request_id)

                threading.Thread(
                    target=_after_prefetch_llm_done, daemon=True, name="vd-after-prefetch-llm-done"
                ).start()
                return
            if not is_done_gui:
                if self._orchestrator is not None:
                    self._orchestrator.on_llm_finished(
                        ok=True,
                        assistant_text=assistant_text,
                        error="",
                    )
                self._emit(
                    {
                        "type": "llm_done",
                        "ok": True,
                        "assistant_text": assistant_text,
                        "request_id": request_id,
                        "ttft_sec": float(payload.get("ttft_sec", 0.0) or 0.0),
                        "total_sec": float(payload.get("total_sec", 0.0) or 0.0),
                    }
                )
            else:
                self._emit(
                    {
                        "type": "llm_test_finished",
                        "ok": True,
                        "request_id": request_id,
                        "backend": str(st_done.get("backend", "")),
                        "model": str(st_done.get("model", "")),
                        "context_len": int(st_done.get("context_len", 10) or 10),
                        "assistant_text": assistant_text,
                        "ttft_sec": float(payload.get("ttft_sec", 0.0) or 0.0),
                        "total_sec": float(payload.get("total_sec", 0.0) or 0.0),
                        "total_overall_sec": float(payload.get("total_overall_sec", 0.0) or 0.0),
                    }
                )

            def _after_llm_done() -> None:
                # ターン開始時の dialog epoch を捕捉（stop→start を跨いだ straggler 検出用）。
                with self._vd_lifecycle_lock:
                    turn_epoch = self._vd_start_epoch
                try:
                    deadline = time.monotonic() + 120.0
                    with self._vd_tts_jobs_lock:
                        while int(self._vd_tts_jobs.get("inflight", 0) or 0) > 0:
                            rem = deadline - time.monotonic()
                            if rem <= 0:
                                break
                            self._vd_tts_segments_done.wait(timeout=min(1.0, rem))
                    if prefetch_committed:
                        with self._llm_pending_lock:
                            st_fin = self._llm_pending.get(request_id)
                            if isinstance(st_fin, dict):
                                st_fin["llm_done"] = True
                                st_fin["llm_done_ok"] = True
                                st_fin["synth_complete"] = True
                        self._flush_held_prefetch_pcm(request_id)
                        self._arm_prefetch_tail_if_ready(request_id)
                        return
                    tail_cb = (
                        self.llm_gui_test_fire_tail_done
                        if is_done_gui and bool(self._orch_vd_tts_state.get("suppress_orch"))
                        else self.fire_tts_done_safe
                    )
                    if tts_via_segments:
                        # 割り込み（execute_user_interrupt）が tail を確定済み、または stop→start で
                        # ターンが交代済みなら再アームしない。再アームすると完了/中断済みターンに対し
                        # on_tts_done が二重発火し状態機械が壊れる(#1)。フラグ check と arm を tail ロック下で
                        # 原子的に行い、フラグ clear と arm の TOCTOU を塞ぐ(#1 round-3)。
                        armed_tg = None
                        with self._vd_tts_tail_lock:
                            if (not self._vd_interrupt_stop.is_set()) and self._vd_start_epoch_current(turn_epoch):
                                armed_tg = int(self._orch_vd_tts_state.get("tail_gen", 0) or 0) + 1
                                self._orch_vd_tts_state["tail_gen"] = armed_tg
                                self._orch_vd_tts_state["tail_pending"] = True
                        if armed_tg is not None:
                            self._tts_router.enqueue_break({"type": "tts_play_break", "generation": armed_tg})
                    elif is_done_gui:
                        self._orch_vd_tts_state["suppress_orch"] = False
                finally:
                    # prefetch 確定経路は _arm_prefetch_tail_if_ready が先に pop しうる。
                    with self._llm_pending_lock:
                        self._pop_llm_pending_locked(request_id)

            threading.Thread(target=_after_llm_done, daemon=True, name="vd-after-llm-done").start()
            return
        self._emit({**payload, "_src": "llm_raw"})

    # -- GUI tests ------------------------------------------------------------

    def dispatch_llm_gui_test(self, cmd: dict[str, object]) -> None:
        """GUI の LLM テストを音声対話と同じ LLM→分段TTS合成→ペーサー経路で実行する。"""
        request_id = f"llm-{int(time.time() * 1000)}-{threading.get_ident()}"
        self._vd_interrupt_stop.clear()
        messages_raw = cmd.get("messages") or []
        msgs = messages_raw if isinstance(messages_raw, list) else []
        backend = str(cmd.get("backend", "LM Studio")).strip()
        model = str(cmd.get("model", "")).strip()
        host = str(cmd.get("host", "") or "").strip()
        api_key = str(cmd.get("api_key", "") or "").strip()
        api_key_env = str(cmd.get("api_key_env", "") or "").strip()
        context_len = 10
        try:
            context_len = int(cmd.get("context_len", 10) or 10)
        except (TypeError, ValueError):
            pass
        max_tokens = 128
        try:
            max_tokens = int(cmd.get("max_tokens", 128) or 128)
        except (TypeError, ValueError):
            pass
        request_messages = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in msgs
            if isinstance(m, dict)
        ]
        tts_stream_inputs = dict(cmd.get("tts_stream_inputs") or {}) if isinstance(cmd.get("tts_stream_inputs"), dict) else {}
        tts_engine = str(cmd.get("tts_stream_engine", "")).strip()

        def _fail_gui_test(msg: str, *, asst_text: str = "") -> None:
            self._orch_vd_tts_state["suppress_orch"] = False
            self._emit(
                {
                    "type": "llm_test_finished",
                    "ok": False,
                    "request_id": request_id,
                    "backend": backend,
                    "model": model,
                    "context_len": context_len,
                    "assistant_text": str(asst_text),
                    "error": msg,
                    "ttft_sec": 0.0,
                    "total_sec": 0.0,
                    "total_overall_sec": 0.0,
                }
            )

        state: dict[str, object] = {
            "request_id": request_id,
            "source": "llm_gui_test",
            "assistant_text": "",
            "tts_engine": tts_engine,
            "tts_stream_inputs": dict(tts_stream_inputs),
            "voice_dialog_inject_ch2": bool(cmd.get("voice_dialog_inject_ch2", True)),
            "tts_seg_ord": 0,
            "tts_seg_break_gen": 0,
            "backend": backend,
            "model": model,
            "context_len": context_len,
        }
        self._orch_vd_tts_state["suppress_orch"] = True
        with self._vd_tts_seg_lock:
            self._orch_vd_tts_state["segments"] = []
        self._orch_vd_tts_state["first_audible"] = False
        self._orch_vd_tts_state["playback_pcm_bytes"] = 0
        self._orch_vd_tts_state["tail_pending"] = False
        with self._llm_pending_lock:
            self._set_llm_pending_locked(request_id, state)
        self.begin_llm_turn_for_latency()

        # 状態パネルとエンジン管理の判定をこのイベントで同期する（TTS テストと同じ）。
        # 出さないと「停止」のまま見え、動いている LLM ワーカーを見落として
        # エンジン管理の適用が通ってしまう。
        self._emit({"type": "module_state", "module": "llm",
                    "state": "starting", "source": "llm_gui_test"})
        try:
            self._supervisor.ensure_llm(timeout_sec=40.0)
        except Exception as ex:
            self._emit({"type": "module_state", "module": "llm",
                        "state": "error", "error": str(ex), "source": "llm_gui_test"})
            with self._llm_pending_lock:
                self._pop_llm_pending_locked(request_id)
            _fail_gui_test(str(ex))
            return
        self._emit({"type": "module_state", "module": "llm",
                    "state": "running", "source": "llm_gui_test"})

        start_payload: dict[str, object] = {
            "type": "start",
            "request_id": request_id,
            "backend": backend,
            "host": host,
            "model": model,
            "api_key": api_key,
            "api_key_env": api_key_env,
            "messages": request_messages,
            "max_tokens": max_tokens,
            "stream": bool(cmd.get("stream", True)),
            "tts_stream_inputs": dict(tts_stream_inputs),
        }
        lc = cmd.get("llm_custom")
        if isinstance(lc, dict):
            start_payload.update(lc)

        if not self._supervisor.send_llm(start_payload):
            with self._llm_pending_lock:
                self._pop_llm_pending_locked(request_id)
            _fail_gui_test("LLM送信失敗")

    def dispatch_tts_gui_test(self, cmd: dict[str, object]) -> None:
        """GUI の TTS 単体テストを音声対話 nonstream と同じ合成・再生経路で実行する。"""
        engine = str(cmd.get("engine", "")).strip()
        utterance = str(cmd.get("utterance", "")).strip()
        inputs_obj = cmd.get("inputs") if isinstance(cmd.get("inputs"), dict) else {}
        inputs: dict[str, object] = dict(inputs_obj)
        tmpl = cmd.get("template")
        if engine == "TEMPLATE" and isinstance(tmpl, dict):
            inputs["template"] = tmpl
        inject = bool(cmd.get("voice_dialog_inject_ch2", True))
        started_at = float(cmd.get("started_at", time.perf_counter()) or time.perf_counter())
        mmd_running = bool(cmd.get("mmd_running", False))

        def _fin(**kwargs: object) -> None:
            pl: dict[str, object] = {"type": "tts_test_finished"}
            for k, v in kwargs.items():
                pl[str(k)] = v
            self._emit(pl)

        if not engine or not utterance:
            _fin(
                ok=False,
                engine=engine,
                error="engine または utterance が空です。",
                started_at=started_at,
                mmd_running=mmd_running,
            )
            return

        try:
            if tps.is_worker_engine(engine):
                wc = cmd.get("tts_worker_config")
                if not isinstance(wc, dict):
                    _fin(
                        ok=False,
                        engine=engine,
                        error="TTS ワーカー用 tts_worker_config がありません。",
                        started_at=started_at,
                        mmd_running=mmd_running,
                    )
                    return
                # 状態パネルとエンジン管理の判定はこのイベントで同期する。
                # 出さないと「停止」のまま見え、動いている TTS を見落として
                # エンジン管理の適用が通ってしまう（uv sync がワーカーの掴んでいる
                # DLL を置き換えられず失敗する）。voice_dialog 経路と同じ扱い。
                self._emit({"type": "module_state", "module": "tts",
                            "state": "starting", "source": "tts_gui_test", "engine": engine})
                try:
                    self._supervisor.ensure_tts(engine=engine, config=wc, timeout_sec=180.0)
                except Exception as ex:
                    self._emit({"type": "module_state", "module": "tts",
                                "state": "error", "error": str(ex),
                                "source": "tts_gui_test", "engine": engine})
                    _fin(
                        ok=False,
                        engine=engine,
                        error=str(ex),
                        started_at=started_at,
                        mmd_running=mmd_running,
                    )
                    return
                self._emit({"type": "module_state", "module": "tts",
                            "state": "running", "source": "tts_gui_test", "engine": engine,
                            "config_digest": _tts_config_digest(wc)})

            tracker: dict[str, int] = {"bytes": 0}
            first_audible = False
            playback_aborted = False
            voice = ""

            def _route_test_frame(sub: bytes) -> bool:
                """1 フレームを routing する。stop_flag が立っていれば True（中断）を返す。"""
                nonlocal first_audible
                if self._vd_nonstream_stop_flag.is_set():
                    return True
                try:
                    sub_max = int(audioop.max(sub, 2))
                except Exception:
                    sub_max = 0
                if (not first_audible) and sub_max > 120:
                    first_audible = True
                self._tts_router.route_pcm_l16le(sub, inject_maai=inject, playback_tracker=tracker)
                return False

            self.nonstream_begin()
            try:
                self._vd_nonstream_stop_flag.clear()
                if tps.is_binary_stream_engine(engine):
                    chunk_aborted = [False]

                    def _on_test_chunk(chunk_l16be: bytes, _meta: dict) -> None:
                        for sub in tps.split_pcm_into_frames(tps.to_l16le(chunk_l16be)):
                            if _route_test_frame(sub):
                                chunk_aborted[0] = True
                                break

                    try:
                        _, voice = self._tts_router.synthesize_l16be(engine, inputs, utterance, on_chunk=_on_test_chunk)
                    except Exception as ex:
                        _fin(
                            ok=False,
                            engine=engine,
                            error=str(ex),
                            started_at=started_at,
                            mmd_running=mmd_running,
                        )
                        return
                    playback_aborted = chunk_aborted[0]
                else:
                    try:
                        audio_l16be, voice = self._tts_router.synthesize_l16be(engine, inputs, utterance)
                    except Exception as ex:
                        _fin(
                            ok=False,
                            engine=engine,
                            error=str(ex),
                            started_at=started_at,
                            mmd_running=mmd_running,
                        )
                        return

                    chunk_le = tps.to_l16le(audio_l16be, source_endian="be")
                    if not chunk_le:
                        _fin(
                            ok=False,
                            engine=engine,
                            error="空の PCM です。",
                            voice=str(voice),
                            started_at=started_at,
                            mmd_running=mmd_running,
                        )
                        return

                    for sub in tps.split_pcm_into_frames(chunk_le):
                        if _route_test_frame(sub):
                            playback_aborted = True
                            break

                if not playback_aborted:
                    try:
                        self._tts_router.enqueue_break({"type": "tts_play_break"})
                    except Exception:
                        pass
            finally:
                self.nonstream_end()

            if playback_aborted:
                _fin(
                    ok=False,
                    engine=engine,
                    error="interrupt",
                    voice=str(voice),
                    started_at=started_at,
                    mmd_running=mmd_running,
                    playback_via_orchestrator=True,
                    stream_bytes=int(tracker.get("bytes", 0)),
                    first_audible=first_audible,
                )
                return

            _fin(
                ok=True,
                engine=engine,
                voice=str(voice),
                started_at=started_at,
                mmd_running=mmd_running,
                playback_via_orchestrator=True,
                stream_bytes=int(tracker.get("bytes", 0)),
                first_audible=first_audible,
            )
        except Exception as ex:
            _fin(
                ok=False,
                engine=engine,
                error=str(ex),
                started_at=started_at,
                mmd_running=mmd_running,
            )

    # -- LLM dispatch (real flow) ---------------------------------------------

    def dispatch_llm_request(self, *, prefetch: bool = False) -> None:
        """Orchestrator の trigger_llm に応答して LLM ワーカーへリクエストを送る。

        ユーザー発話は引数で受けない。送るメッセージ列は
        `orchestrator.history_snapshot()` から組む（そこに既に user 行が入っている）ため。

        prefetch=True のときはターン未取得の先読み。PCM は保持し、p_now 確定まで再生しない。
        """
        request_id = f"llm-{int(time.time() * 1000)}-{threading.get_ident()}"
        self._vd_interrupt_stop.clear()
        settings = dict(self._llm_settings)
        context_len = 10
        try:
            context_len = int(settings.get("llm_context_len", 10) or 10)
        except (TypeError, ValueError):
            pass
        max_tokens = 128
        try:
            max_tokens = int(settings.get("llm_max_tokens", 128) or 128)
        except (TypeError, ValueError):
            pass
        backend = str(settings.get("llm_backend", "LM Studio"))
        model = str(settings.get("llm_model", "")).strip()
        conn = settings.get("llm_connection", {})
        if not isinstance(conn, dict):
            conn = {}
        snapshot = self._orchestrator.history_snapshot() if self._orchestrator is not None else {}
        msgs = snapshot.get("messages", []) if isinstance(snapshot, dict) else []
        if not isinstance(msgs, list):
            msgs = []
        system_msg = msgs[0] if msgs else {"role": "system", "content": ""}
        non_system = [m for m in msgs[1:] if isinstance(m, dict) and m.get("role") in {"user", "assistant"}]
        if len(non_system) > context_len:
            non_system = non_system[-context_len:]
        # TTS 分割用に積んだ content="" の assistant 行を LLM に渡すと、
        # Gemma/LM Studio が即 finish_reason=stop の空応答を返すことがある。
        request_messages: list[dict[str, str]] = []
        for m in ([system_msg] + non_system):
            role = str(m.get("role", ""))
            content = str(m.get("content", ""))
            if role != "system" and not content.strip():
                continue
            request_messages.append({"role": role, "content": content})
        if not request_messages:
            request_messages = [{"role": "system", "content": str(system_msg.get("content", ""))}]
        tts_stream_inputs = dict(settings.get("tts_stream_inputs") or {}) if isinstance(settings.get("tts_stream_inputs"), dict) else {}
        tts_engine = str(settings.get("tts_engine", "")).strip()
        state = {
            "request_id": request_id,
            "assistant_text": "",
            "tts_engine": tts_engine,
            "tts_stream_inputs": dict(tts_stream_inputs),
            "voice_dialog_inject_ch2": bool(settings.get("voice_dialog_inject_ch2", True)),
            "tts_seg_ord": 0,
            "tts_seg_break_gen": 0,
            "prefetch": bool(prefetch),
            "committed": False,
            "held_pcm": [],
            "held_flushed": 0,
            "llm_done": False,
            "llm_done_ok": True,
            "synth_complete": False,
            "latency_tts_noted": False,
            "tail_armed": False,
            "last_activity": time.perf_counter(),
        }
        with self._vd_tts_seg_lock:
            self._orch_vd_tts_state["segments"] = []
        self._orch_vd_tts_state["first_audible"] = False
        self._orch_vd_tts_state["playback_pcm_bytes"] = 0
        self._orch_vd_tts_state["tail_pending"] = False
        with self._llm_pending_lock:
            self._set_llm_pending_locked(request_id, state)
        start_payload: dict[str, object] = {
            "type": "start",
            "request_id": request_id,
            "backend": backend,
            "host": conn.get("host", ""),
            "model": model,
            "api_key": conn.get("api_key", ""),
            "api_key_env": conn.get("api_key_env", ""),
            "messages": request_messages,
            "max_tokens": max_tokens,
            "stream": True,
            "tts_stream_inputs": tts_stream_inputs,
        }
        self._arm_llm_watchdog(request_id, state)
        send_ok = bool(self._supervisor.send_llm(start_payload))
        if not send_ok:
            with self._llm_pending_lock:
                self._remember_llm_terminal_locked(request_id)
                self._pop_llm_pending_locked(request_id)
            if prefetch:
                if self._orchestrator is not None:
                    try:
                        self._orchestrator.mark_prefetch_failed()
                    except Exception:
                        pass
                self._emit({"type": "log", "message": "[LLM][PREFETCH] send failed"})
                return
            self.reset_turn_latency()
            self._event_queue.put({"type": "llm_finished", "ok": False, "assistant_text": "", "error": "LLM送信失敗"})

    # -- Worker lifecycle -----------------------------------------------------

    def _send_audio_output_remote_plugin_from_config(self, audio_output_cfg: dict) -> None:
        """output_mode=mmd のとき config 内の remote_plugin_settings を Audio Output へ適用する。"""
        mode = str(audio_output_cfg.get("output_mode", "mmd") or "mmd").strip().lower()
        if mode != "mmd":
            return
        rp = audio_output_cfg.get("remote_plugin_settings")
        if not isinstance(rp, dict) or not rp:
            return
        try:
            self._supervisor.send_audio_output(
                {"type": "remote_plugin_apply_settings", "settings": dict(rp)}
            )
        except Exception:
            pass

    def bootstrap_audio_io_workers_early(
        self, app_cfg: dict | None, *, assume_mmd_running: bool = False,
    ) -> None:
        """Orchestrator ワーカー起動直後に Audio Input / Output だけ先に起動する。

        `controller_config.json` を `ConfigStore.load()` した dict を渡す。`mdf_path` の MDF から
        Remote プラグイン用キーをマージするのは `dialog/settings` 側（JSON 単体では足りない）。
        MMD モードなら `remote_plugin_apply_settings` まで送り、MMDAgent が接続可能な状態にする。
        後から `start_voice_dialog` で渡る設定 JSON が同一なら `ensure_*` は再起動しない。
        `assume_mmd_running=True` のときだけ保存値 `mmd` を落とさず、MDF で選ばれた
        Remote 接続を立てる。
        """
        cfg = app_cfg if isinstance(app_cfg, dict) else {}
        try:
            from . import settings as vds
        except ImportError:
            self._emit({"type": "log", "message": "[Bootstrap] voice_dialog_settings を読み込めませんでした"})
            return

        # AudioInput（マイク取り込み）はここでは先行起動しない。
        # MMDAgent 接続に必要なのは AudioOutput（SND 送出）だけで、mic は不要。
        # audio_input は ASR / MaAI / 音声対話 / WAV 開始時の _ensure_audio_input_worker で
        # on-demand 起動する。テスト停止では落とさず、GUI 終了時に teardown する。
        # こうしないと「何も開始していないのにマイクが常時 ON」になる。

        try:
            aoc = dict(vds.build_audio_output_worker_config_dict(
                cfg, assume_mmd_running=assume_mmd_running,
            ))
        except Exception as ex:
            self._emit({"type": "log", "message": f"[Bootstrap] audio_output 設定の組み立てに失敗: {ex}"})
            return

        self._emit({"type": "module_state", "module": "audio_output",
                    "state": "starting", "source": "bootstrap"})
        try:
            self._supervisor.ensure_audio_output(config=aoc)
        except Exception as ex:
            self._emit({"type": "module_state", "module": "audio_output",
                        "state": "error", "error": str(ex), "source": "bootstrap"})
            self._emit({"type": "log", "message": f"[Bootstrap] AudioOutput 起動に失敗: {ex}"})
            return
        self._emit({"type": "module_state", "module": "audio_output",
                    "state": "running", "source": "bootstrap"})

        self._send_audio_output_remote_plugin_from_config(aoc)
        om = str(aoc.get("output_mode", "") or "").strip()
        self._emit(
            {
                "type": "log",
                "message": f"[Bootstrap] Audio Input/Output を先行起動しました (output_mode={om!r})。",
            }
        )

    def _voice_dialog_llm_warmup_messages(self, settings: dict) -> list[dict[str, str]]:
        """本番の初回ターンに近い messages でウォームアップする。"""
        system_prompt = str(settings.get("system_prompt", "")).strip() or "あなたは簡潔で丁寧なアシスタントです。"
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        opening = str(settings.get("voice_dialog_opening_utterance", "")).strip()
        if opening:
            messages.append({"role": "assistant", "content": opening})
        messages.append({"role": "user", "content": "hi"})
        return messages

    def _begin_llm_worker_warmup(self, settings: dict) -> dict[str, object] | None:
        """LLM ワーカーへ warmup を送り、完了待ち用の state を返す。"""
        model = str(settings.get("llm_model", "")).strip()
        if not model:
            return None
        backend = str(settings.get("llm_backend", "LM Studio")).strip() or "LM Studio"
        conn = settings.get("llm_connection")
        if not isinstance(conn, dict):
            conn = {}
        request_id = f"llm-wu-{int(time.time() * 1000)}-{threading.get_ident()}"
        wait_state: dict[str, object] = {
            "source": "llm_warmup",
            "done": threading.Event(),
            "error": "",
            "backend": backend,
            "model": model,
        }
        with self._llm_pending_lock:
            self._set_llm_pending_locked(request_id, wait_state)
        payload: dict[str, object] = {
            "type": "warmup",
            "request_id": request_id,
            "backend": backend,
            "host": str(conn.get("host", "")),
            "model": model,
            "api_key": str(conn.get("api_key", "")),
            "api_key_env": str(conn.get("api_key_env", "")),
            "force": bool(settings.get("force", True)),
            "stream": True,
            "messages": self._voice_dialog_llm_warmup_messages(settings),
        }
        if "send_temperature" in settings:
            payload["send_temperature"] = settings.get("send_temperature")
        if "send_max_tokens" in settings:
            payload["send_max_tokens"] = settings.get("send_max_tokens")
        if "temperature" in settings:
            payload["temperature"] = settings.get("temperature")
        if "custom_extra_body_thinking" in settings:
            payload["custom_extra_body_thinking"] = settings.get("custom_extra_body_thinking")
        self._llm_warmup_error = ""
        self._emit(
            {
                "type": "log",
                "message": f"[VoiceDialog] LLM ウォームアップ開始: backend={backend}, model={model}",
            }
        )
        if not self._supervisor.send_llm(payload):
            with self._llm_pending_lock:
                self._pop_llm_pending_locked(request_id)
            self._llm_warmup_error = "LLM worker が未接続です"
            self._emit({"type": "log", "message": "[VoiceDialog] LLM ウォームアップ送信失敗（未接続）"})
            return None
        wait_state["request_id"] = request_id
        return wait_state

    def _await_llm_worker_warmup(
        self,
        wait_state: dict[str, object],
        *,
        epoch: "int | None",
        timeout_sec: float = 120.0,
    ) -> None:
        """warmup 完了まで待つ。失敗しても音声対話開始自体は止めない。"""
        request_id = str(wait_state.get("request_id", "") or "").strip()
        done_ev = wait_state.get("done")
        try:
            if not isinstance(done_ev, threading.Event):
                self._llm_warmup_error = "internal: bad LLM warmup wait state"
                return
            deadline = time.time() + max(1.0, float(timeout_sec))
            while time.time() < deadline:
                if not self._vd_start_epoch_current(epoch):
                    return
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                if done_ev.wait(timeout=min(0.5, remaining)):
                    break
            if not self._vd_start_epoch_current(epoch):
                return
            if not done_ev.is_set():
                self._llm_warmup_error = "LLM ウォームアップがタイムアウトしました"
                self._emit(
                    {
                        "type": "log",
                        "message": "[VoiceDialog] LLM ウォームアップがタイムアウトしました（初回応答が遅くなることがあります）",
                    }
                )
                return
            err = str(wait_state.get("error", "") or "").strip()
            if err:
                self._llm_warmup_error = err
                self._emit({"type": "log", "message": f"[VoiceDialog] LLM ウォームアップ失敗: {err}"})
                return
            self._llm_warmup_error = ""
            model = str(wait_state.get("model", "") or "")
            self._emit(
                {
                    "type": "log",
                    "message": f"[VoiceDialog] LLM ウォームアップ完了: model={model}",
                }
            )
        finally:
            if request_id:
                with self._llm_pending_lock:
                    self._pop_llm_pending_locked(request_id)

    def ensure_voice_dialog_workers(self, settings: dict, *, epoch: "int | None" = None) -> None:
        """音声対話開始時にワーカー群を起動する。

        epoch を渡すと、各重い ensure_*（特に最大180sの TTS）の前に start epoch を再確認し、
        stop / 新 start が来ていれば残りの ensure をスキップする。これにより停止が単一
        lifecycle FIFO 上で進行中 ensure の後ろに長時間ブロックされるのを防ぐ(#16)。"""
        worker_configs = settings.get("worker_configs")
        if not isinstance(worker_configs, dict):
            worker_configs = {}
        asr_cfg = worker_configs.get("asr", {})
        tts_cfg = worker_configs.get("tts", {})
        llm_cfg = worker_configs.get("llm", {})
        maai_cfg = worker_configs.get("maai", {})
        audio_input_cfg = worker_configs.get("audio_input", {})
        audio_output_cfg = worker_configs.get("audio_output", {})
        tts_engine = str(settings.get("tts_engine", "")).strip()

        def _emit_module_state(module: str, state: str, error: str = "") -> None:
            # 手動 start_module を経ない起動のため、Controller の状態パネル/ready フラグは
            # このイベントで同期する（module_started は start_module_async 専用のまま維持）。
            payload: dict[str, object] = {
                "type": "module_state",
                "module": module,
                "state": state,
                "source": "voice_dialog",
            }
            # どのエンジンが動いているかも伝える。Controller 側は
            # 「ready かつ engine が一致」で稼働中と判定するので、engine が無いと
            # メイン画面から起動しても TTS テスト画面が「停止」のままになる。
            if module == "tts" and tts_engine:
                payload["engine"] = tts_engine
                # 実際に適用した設定の指紋。GUI はこれと自分が組み立てた設定を
                # 比べて「起動し直す必要があるか」を判断する。GUI 側だけで
                # 覚えていると、別経路（TTS テスト送信）が ensure_tts を
                # 別設定で呼んだときに食い違う。
                payload["config_digest"] = _tts_config_digest(
                    worker_configs.get("tts") if isinstance(worker_configs, dict) else None
                )
            if error:
                payload["error"] = error
            self._emit(payload)

        if not self._vd_start_epoch_current(epoch):
            return
        if isinstance(asr_cfg, dict) and asr_cfg:
            asr_route = {
                "host": self._asr_audio_server.host,
                "port": self._asr_audio_server.port,
            }
            asr_cfg_copy = dict(asr_cfg)
            asr_cfg_copy.pop("audio_ingress_host", None)
            asr_cfg_copy.pop("audio_ingress_port", None)
            _emit_module_state("asr", "starting")
            try:
                self._supervisor.ensure_asr(
                    config=asr_cfg_copy,
                    ipc_host=asr_route["host"],
                    ipc_port=asr_route["port"],
                )
                _emit_module_state("asr", "running")
            except Exception as ex:
                _emit_module_state("asr", "error", str(ex))
                self._emit({"type": "log", "message": f"[Supervisor] ASR start failed: {ex}"})
        if not self._vd_start_epoch_current(epoch):
            return
        if isinstance(tts_cfg, dict) and tts_cfg and tts_engine:
            _emit_module_state("tts", "starting")
            try:
                self._supervisor.ensure_tts(engine=tts_engine, config=tts_cfg)
                # LLM と同じ理由: ワーカーが起きただけでは合成できるとは限らない。
                tts_warmup_error = str(getattr(self, "_tts_warmup_error", "") or "")
                if tts_warmup_error:
                    _emit_module_state("tts", "error", tts_warmup_error)
                else:
                    _emit_module_state("tts", "running")
            except Exception as ex:
                _emit_module_state("tts", "error", str(ex))
                self._emit({"type": "log", "message": f"[Supervisor] TTS start failed: {ex}"})
        if not self._vd_start_epoch_current(epoch):
            return
        llm_warmup_wait: dict[str, object] | None = None
        llm_ensure_ok = False
        if isinstance(llm_cfg, dict):
            _emit_module_state("llm", "starting")
            try:
                self._supervisor.ensure_llm()
                llm_ensure_ok = True
                # ワーカー起動だけではエンドポイントのモデルロードは終わらない。
                # 初回ユーザー発話で載ると「最初だけ極端に遅い」ので、起動中に
                # 本番と同じ LLM ワーカーへストリーミング warmup を投げておく。
                llm_warmup_wait = self._begin_llm_worker_warmup(settings)
            except Exception as ex:
                _emit_module_state("llm", "error", str(ex))
                self._emit({"type": "log", "message": f"[Supervisor] LLM start failed: {ex}"})
        if not self._vd_start_epoch_current(epoch):
            if llm_warmup_wait is not None:
                self._await_llm_worker_warmup(llm_warmup_wait, epoch=epoch)
            return
        if isinstance(maai_cfg, dict) and maai_cfg:
            maai_route = {
                "host": self._maai_audio_server.host,
                "port": self._maai_audio_server.port,
            }
            maai_cfg_copy = dict(maai_cfg)
            maai_cfg_copy.pop("audio_ingress_host", None)
            maai_cfg_copy.pop("audio_ingress_port", None)
            _emit_module_state("maai", "starting")
            try:
                self._supervisor.ensure_maai(
                    config=maai_cfg_copy,
                    ipc_host=maai_route["host"],
                    ipc_port=maai_route["port"],
                )
                _emit_module_state("maai", "running")
            except Exception as ex:
                _emit_module_state("maai", "error", str(ex))
                self._emit({"type": "log", "message": f"[Supervisor] MaAI start failed: {ex}"})
        # audio_input / audio_output は Bootstrap と設定保存時の同期のみで起動・取り直しする。
        # 音声対話開始のたびに ensure するとプロセス再起動が走り得るため行わない。
        if isinstance(audio_input_cfg, dict) and audio_input_cfg:
            try:
                ai_running = bool(self._supervisor.audio_input.is_running())
            except Exception:
                ai_running = False
            if not ai_running:
                # 未起動ならここで起動する。GUI は `start_module` で先に起動しているので
                # 通常この分岐に入らない（＝GUI の挙動は変わらない）。headless では
                # 誰も起動しないため、以前はログを出すだけで**音声が 1 フレームも
                # 流れなかった**（F4）。起動済みのときは触らないので、既存の
                # 「開始のたびにプロセスを取り直さない」方針も保たれる。
                self._emit(
                    {
                        "type": "log",
                        "message": "[VoiceDialog] AudioInput が未起動なので起動します。",
                    }
                )
                aic = dict(audio_input_cfg)
                aic.pop("audio_forward_host", None)
                aic.pop("audio_forward_port", None)
                try:
                    self._supervisor.ensure_audio_input(
                        config=aic,
                        ipc_host=self._audio_input_server.host,
                        ipc_port=self._audio_input_server.port,
                    )
                    self._emit({"type": "module_started", "module": "audio_input",
                                "source": "voice_dialog"})
                except Exception as ex:
                    self._emit({"type": "module_start_failed", "module": "audio_input",
                                "error": str(ex), "source": "voice_dialog"})
        # audio_output は Bootstrap で先行起動済みを前提とする。ここで ensure すると
        # 開始のたびにプロセス取り直しが走り得るため行わない。設定反映は Controller が
        # start_module(ensure) / remote_plugin_apply_settings で行う。
        if isinstance(audio_output_cfg, dict) and audio_output_cfg:
            try:
                ao_running = bool(self._supervisor.audio_output.is_running())
            except Exception:
                ao_running = False
            if not ao_running:
                # F4 と同じ理由。GUI は Bootstrap 済みなのでこの分岐に入らない。
                self._emit(
                    {
                        "type": "log",
                        "message": "[VoiceDialog] AudioOutput が未起動なので起動します。",
                    }
                )
                try:
                    self._supervisor.ensure_audio_output(config=dict(audio_output_cfg))
                    self._send_audio_output_remote_plugin_from_config(dict(audio_output_cfg))
                    self._relay_blink_dialog_active()
                    self._emit({"type": "module_started", "module": "audio_output",
                                "source": "voice_dialog"})
                except Exception as ex:
                    self._emit({"type": "module_start_failed", "module": "audio_output",
                                "error": str(ex), "source": "voice_dialog"})
        if llm_warmup_wait is not None:
            self._await_llm_worker_warmup(llm_warmup_wait, epoch=epoch)
        if llm_ensure_ok and self._vd_start_epoch_current(epoch):
            # `ensure_llm()` が通ってもエンドポイントに繋がるとは限らない。
            # warmup 失敗（接続拒否は数 ms で終わる）を running で上書きしない。
            warmup_error = str(getattr(self, "_llm_warmup_error", "") or "")
            if warmup_error:
                _emit_module_state("llm", "error", warmup_error)
            else:
                _emit_module_state("llm", "running")
        self._llm_settings.update(settings)

    def start_module_async(self, module: str, config: dict, payload: dict) -> None:
        """バックグラウンドスレッドで supervisor.ensure_* を実行する。"""
        source = str(payload.get("source", "")).strip()
        changed = True
        try:
            if module == "asr":
                route = {
                    "host": self._asr_audio_server.host,
                    "port": self._asr_audio_server.port,
                }
                cfg = dict(config)
                cfg.pop("audio_ingress_host", None)
                cfg.pop("audio_ingress_port", None)
                self._supervisor.ensure_asr(config=cfg, ipc_host=route["host"], ipc_port=route["port"])
            elif module == "asr_compare":
                route = {
                    "host": self._asr_compare_audio_server.host,
                    "port": self._asr_compare_audio_server.port,
                }
                cfg = dict(config)
                cfg.pop("audio_ingress_host", None)
                cfg.pop("audio_ingress_port", None)
                self._supervisor.ensure_asr_compare(config=cfg, ipc_host=route["host"], ipc_port=route["port"])
            elif module == "llm":
                self._supervisor.ensure_llm()
            elif module == "tts":
                engine = str(payload.get("engine", "")).strip()
                if engine == "VOICEVOX":
                    host = str(payload.get("host") or config.get("host") or "127.0.0.1")
                    try:
                        port = int(payload.get("port") or config.get("port") or 50021)
                    except (TypeError, ValueError):
                        port = 50021
                    exe = str(
                        payload.get("voicevox_exe_path")
                        or config.get("voicevox_exe_path")
                        or ""
                    )
                    self._supervisor.ensure_voicevox(host=host, port=port, exe_path=exe)
                elif engine == "TEMPLATE":
                    pass
                else:
                    self._supervisor.ensure_tts(engine=engine, config=config)
            elif module == "maai":
                route = {
                    "host": self._maai_audio_server.host,
                    "port": self._maai_audio_server.port,
                }
                cfg = dict(config)
                cfg.pop("audio_ingress_host", None)
                cfg.pop("audio_ingress_port", None)
                self._supervisor.ensure_maai(config=cfg, ipc_host=route["host"], ipc_port=route["port"])
            elif module == "audio_input":
                aic = dict(config)
                aic.pop("audio_forward_host", None)
                aic.pop("audio_forward_port", None)
                changed = self._supervisor.ensure_audio_input(
                    config=aic,
                    ipc_host=self._audio_input_server.host,
                    ipc_port=self._audio_input_server.port,
                )
            elif module == "audio_output":
                changed = self._supervisor.ensure_audio_output(config=config)
                self._send_audio_output_remote_plugin_from_config(config)
                if changed:
                    self._relay_blink_dialog_active()
            else:
                self._emit({"type": "log", "message": f"[Supervisor] unknown module: {module}"})
                return
            self._emit(
                {
                    "type": "module_started",
                    "module": module,
                    "source": source,
                    "unchanged": not changed,
                }
            )
        except Exception as ex:
            self._emit({"type": "module_start_failed", "module": module, "error": str(ex), "source": source})

    # -- Ingress forwarders ---------------------------------------------------

    def forward_asr_ingress_event(
        self,
        payload: dict[str, object],
        *,
        mark_ready: bool,
        mark_compare_ready: bool = False,
    ) -> None:
        """ASR ワーカー → JsonLineEventServer。ready で Supervisor を解放し、認識結果等は orchestrator へ。"""
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("type", "")).strip().lower()
        if not event_type:
            return
        if event_type == "ready":
            if mark_ready:
                self._supervisor.mark_asr_ready()
            if mark_compare_ready:
                self._supervisor.mark_asr_compare_ready()
            return
        if event_type == "log":
            self._emit({"type": "log", "message": str(payload.get("message", ""))})
            return
        if event_type == "error":
            self._emit({"type": "log", "message": f"[ASR][ERROR] {payload.get('message', '')}"})
            return
        if event_type == "fatal":
            if mark_ready:
                self._supervisor.mark_asr_fatal(str(payload.get("message", "")))
            if mark_compare_ready:
                self._supervisor.mark_asr_compare_fatal(str(payload.get("message", "")))
            self._emit({"type": "log", "message": f"[ASR][FATAL] {payload.get('message', '')}"})
            return
        if event_type == "stopped":
            self._emit({"type": "log", "message": "[ASR] worker stopped"})
            return
        if event_type == "result":
            received_ts = time.time()
            engine_raw = str(payload.get("engine", "asr")).strip() or "asr"
            text = str(payload.get("text", ""))
            latency = float(payload.get("latency_sec", 0.0) or 0.0)
            compare_slot = str(payload.get("compare_slot", "")).strip().lower()
            segment_start_ts = payload.get("segment_start_ts")
            segment_end_ts = payload.get("segment_end_ts")
            utterance_index = int(payload.get("utterance_index", 0) or 0)
            is_final_raw = payload.get("is_final")
            is_streaming_raw = payload.get("is_streaming")
            is_final = bool(is_final_raw) if isinstance(is_final_raw, bool) else True
            is_streaming = bool(is_streaming_raw) if isinstance(is_streaming_raw, bool) else False
            result_payload = {
                "engine": engine_raw,
                "text": text,
                "latency_sec": latency,
                "is_final": is_final,
                "is_streaming": is_streaming,
                "compare_slot": compare_slot,
                "segment_start_ts": segment_start_ts,
                "segment_end_ts": segment_end_ts,
                "final_arrival_ts": received_ts,
                "utterance_index": utterance_index,
            }
            self._event_queue.put({"type": "asr_result", **result_payload})
            return
        if event_type == "turn":
            probability = float(payload.get("probability", 0.0) or 0.0)
            turn_ev: dict[str, object] = {"type": "turn", "probability": probability}
            if "p_future" in payload:
                try:
                    turn_ev["p_future"] = float(payload.get("p_future", 0.0) or 0.0)
                except (TypeError, ValueError):
                    turn_ev["p_future"] = 0.0
            ts_raw = payload.get("ts")
            if ts_raw is not None:
                try:
                    turn_ev["ts"] = float(ts_raw)
                except (TypeError, ValueError):
                    pass
            st_raw = payload.get("source_ts_end")
            if st_raw is not None:
                try:
                    turn_ev["source_ts_end"] = float(st_raw)
                except (TypeError, ValueError):
                    pass
            self._event_queue.put(turn_ev)
            return
        if event_type == "inference_state":
            self._event_queue.put({"type": "inference_state", "busy": bool(payload.get("busy", False))})
            return
        if event_type == "force_finalize_status":
            try:
                request_id = int(payload.get("request_id", 0) or 0)
            except (TypeError, ValueError):
                request_id = 0
            self._event_queue.put(
                {
                    "type": "force_finalize_status",
                    "is_speaking": bool(payload.get("is_speaking", False)),
                    "request_id": request_id,
                    "ts": float(payload.get("ts", time.time()) or time.time()),
                }
            )
            return
        if event_type == "force_finalize_outcome":
            try:
                request_id = int(payload.get("request_id", 0) or 0)
            except (TypeError, ValueError):
                request_id = 0
            self._event_queue.put(
                {
                    "type": "force_finalize_outcome",
                    "status": str(payload.get("status", "")).strip().lower(),
                    "request_id": request_id,
                    "ts": float(payload.get("ts", time.time()) or time.time()),
                    "error": str(payload.get("error", "")),
                }
            )
            return
        if event_type == "maai_result":
            _mr = payload.get("result", {})
            _result = dict(_mr) if isinstance(_mr, dict) else {}
            self._event_queue.put(
                {
                    "type": "maai_result",
                    "compare_slot": str(payload.get("compare_slot", "")).strip().lower(),
                    "result": _result,
                }
            )
            return

    def on_maai_ingress_event(self, payload: dict[str, object]) -> None:
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "ready":
            self._supervisor.mark_maai_ready()
            # ワーカーが ready に付けた頷きのしきい値の既定値を、MaAI テスト画面の「既定: …」用に送る
            # （以前はここで捨てていたので、Qt も Web も「既定: —」のままだった）
            defaults = payload.get("nod_threshold_defaults")
            self._emit({"type": "maai_nod_threshold_defaults",
                        "defaults": defaults if isinstance(defaults, dict) else None})
            return
        if event_type in {"log", "error", "fatal"}:
            msg = str(payload.get("message", ""))
            if event_type == "fatal":
                self._supervisor.mark_maai_fatal(msg)
                self._emit(
                    {
                        "type": "module_state",
                        "module": "maai",
                        "state": "error",
                        "source": "worker_fatal",
                        "error": msg or "MaAI worker reported a fatal error",
                    }
                )
            self._emit({"type": "log", "message": f"[MaAI][{event_type.upper()}] {msg}"})
            return
        if event_type == "stopped":
            self._emit({"type": "log", "message": "[MaAI] worker stopped"})
            return
        self._event_queue.put(dict(payload))

    # -- Nonstream TTS playback (opening / nonstream / scheduled_exit) --------

    def speak_nonstream_async(
        self,
        text: str,
        *,
        allow_inactive: bool = False,
        silent: bool = False,
        inject_only: bool = False,
        no_inject: bool = False,
        source: str = "",
        on_done: Callable[[bool], None] | None = None,
    ) -> None:
        """Controller 経由でなく worker 内で nonstream TTS を再生する。

        - `silent=True` のとき合成・再生をせず、即座に done を返す（MaAI 相槌の silent モード）。
        - `inject_only=True` のとき合成はするが、スピーカー/MMD へは出さず system_audio として
          MaAI 入力へだけ流す（相槌「再生しない（MaAI注入のみ）」）。
        - `no_inject=True` のとき合成・再生はするが MaAI 入力へは流さない
          （相槌「再生する（MaAI注入しない）」）。`inject_only` と同時指定時は `inject_only` 優先。
        - 完了後は `on_done(playback_aborted: bool)` を呼ぶ。
        """
        utterance = (text or "").strip()
        if not utterance:
            if on_done is not None:
                try:
                    on_done(True)
                except Exception:
                    pass
            return
        # inject_only と no_inject が同時なら注入のみを優先（矛盾回避）
        if inject_only:
            no_inject = False

        def _runner() -> None:
            playback_aborted = True
            try:
                if silent:
                    playback_aborted = False
                    return

                is_backchannel = source == "maai_backchannel"
                self.nonstream_begin(backchannel=is_backchannel)
                try:
                    engine = str(self._llm_settings.get("tts_engine", "")).strip()
                    tin_obj = self._llm_settings.get("tts_stream_inputs") or {}
                    tin = dict(tin_obj) if isinstance(tin_obj, dict) else {}
                    inject = bool(self._llm_settings.get("voice_dialog_inject_ch2", True))
                    if no_inject:
                        do_inject = False
                    elif inject_only:
                        do_inject = True
                    else:
                        do_inject = inject

                    if not engine:
                        self._emit({"type": "log", "message": "[VD-Nonstream] tts_engine 未設定のため再生スキップ"})
                        return

                    if tps.is_binary_stream_engine(engine):
                        # streaming path: VAD precheck は合成前に行う
                        if (not allow_inactive) and self.silero_vad_at_or_above_interrupt_threshold():
                            self._emit({"type": "log", "message": "[VD-Nonstream] 合成前にユーザー VAD が閾値以上のため再生をスキップします。"})
                            return
                        if not allow_inactive:
                            self.note_assistant_tts_request_for_latency()

                        ns_aborted = [False]
                        ns_first_audible_sent = [False]

                        def _on_ns_chunk(chunk_l16be: bytes, _meta: dict) -> None:
                            if self._vd_nonstream_stop_flag.is_set():
                                ns_aborted[0] = True
                                return
                            chunk_l16le = tps.to_l16le(chunk_l16be)
                            if not ns_first_audible_sent[0]:
                                try:
                                    chunk_max_ns = int(audioop.max(chunk_l16le, 2))
                                except Exception:
                                    chunk_max_ns = 0
                                if chunk_max_ns > 120:
                                    ns_first_audible_sent[0] = True
                                    if not allow_inactive:
                                        try:
                                            self._event_queue.put({"type": "tts_audible"})
                                            self._emit({"type": "tts_audible"})
                                            self.emit_tts_latency_on_first_audible()
                                        except Exception:
                                            pass
                            for sub in tps.split_pcm_into_frames(chunk_l16le):
                                self._tts_router.route_pcm_l16le(
                                    sub, inject_maai=do_inject, inject_only=inject_only
                                )

                        try:
                            self._tts_router.synthesize_l16be(engine, tin, utterance, on_chunk=_on_ns_chunk)
                        except Exception as ex:
                            self._emit({"type": "log", "message": f"[VD-Nonstream] 合成失敗 source={source} err={ex}"})
                            return

                        if ns_aborted[0]:
                            return
                        if not inject_only:
                            try:
                                self._tts_router.enqueue_break({"type": "tts_play_break"})
                            except Exception:
                                pass
                        playback_aborted = False
                    else:
                        try:
                            audio_l16be, _voice = self._tts_router.synthesize_l16be(engine, tin, utterance)
                        except Exception as ex:
                            self._emit({"type": "log", "message": f"[VD-Nonstream] 合成失敗 source={source} err={ex}"})
                            return

                        if self._vd_nonstream_stop_flag.is_set():
                            return

                        # VAD precheck: メイン発話のみ。相槌(allow_inactive=True)はスキップ。
                        if (not allow_inactive) and self.silero_vad_at_or_above_interrupt_threshold():
                            self._emit(
                                {
                                    "type": "log",
                                    "message": (
                                        "[VD-Nonstream] 合成後・再生前にユーザー VAD が閾値以上のため再生をスキップします。"
                                    ),
                                }
                            )
                            return

                        if not allow_inactive:
                            self.note_assistant_tts_request_for_latency()

                        chunk_le = tps.to_l16le(audio_l16be, source_endian="be")
                        if not chunk_le:
                            self._emit({"type": "log", "message": f"[VD-Nonstream] 空 PCM source={source}"})
                            return
                        ns_first_audible_sent = False
                        for sub in tps.split_pcm_into_frames(chunk_le):
                            if self._vd_nonstream_stop_flag.is_set():
                                return
                            if not ns_first_audible_sent:
                                try:
                                    sub_max_ns = int(audioop.max(sub, 2))
                                except Exception:
                                    sub_max_ns = 0
                                if sub_max_ns > 120:
                                    ns_first_audible_sent = True
                                    if not allow_inactive:
                                        try:
                                            self._event_queue.put({"type": "tts_audible"})
                                            self._emit({"type": "tts_audible"})
                                            self.emit_tts_latency_on_first_audible()
                                        except Exception:
                                            pass
                            self._tts_router.route_pcm_l16le(
                                sub, inject_maai=do_inject, inject_only=inject_only
                            )
                        if not inject_only:
                            try:
                                self._tts_router.enqueue_break({"type": "tts_play_break"})
                            except Exception:
                                pass
                        playback_aborted = False
                finally:
                    self.nonstream_end(backchannel=is_backchannel)
            finally:
                if on_done is not None:
                    try:
                        on_done(bool(playback_aborted))
                    except Exception:
                        pass

        self._vd_nonstream_stop_flag.clear()
        threading.Thread(
            target=_runner, daemon=True, name=f"orch-vd-nonstream-{source or 'tts'}"
        ).start()

    # -- worker dispatch helpers ---------------------------------------------

    def _relay_mmd_burst_setting(self, settings_obj: dict) -> None:
        """MMD のライブ設定を audio/output_worker に反映（ペーサ等は別プロセス側）。"""
        try:
            bf = int(settings_obj.get("mmd_burst_snd_frames", 20) or 20)
        except (TypeError, ValueError):
            bf = 20
        bf = max(0, min(500, bf))
        live = {
            "mmd_burst_snd_frames": bf,
            "mmd_blink_enable": normalize_blink_enable(
                settings_obj.get("mmd_blink_enable"), True
            ),
            "mmd_blink_interval_mean_sec": normalize_blink_interval_mean_sec(
                settings_obj.get("mmd_blink_interval_mean_sec", DEFAULT_BLINK_INTERVAL_MEAN_SEC)
            ),
            "mmd_blink_morph": sanitize_blink_morph(
                settings_obj.get("mmd_blink_morph", "まばたき")
            ),
            "mmd_blink_mode": normalize_blink_mode(settings_obj.get("mmd_blink_mode")),
            "mmd_blink_dialog_active": self._vd_voice_dialog_running_for_maai_tick.is_set(),
        }
        try:
            self._supervisor.send_audio_output({"type": "update_settings", "settings": live})
        except Exception:
            pass

    def _relay_blink_dialog_active(self) -> None:
        """「対話中のみ」まばたき用に、対話中かどうかを audio/output_worker へ知らせる。

        起動し直した worker は対話中でないと思って立ち上がるため、ensure の後にも送る。
        """
        active = self._vd_voice_dialog_running_for_maai_tick.is_set()
        try:
            self._supervisor.send_audio_output(
                {"type": "update_settings", "settings": {"mmd_blink_dialog_active": active}}
            )
        except Exception:
            pass

    def _ensure_vd_lifecycle_worker(self) -> None:
        if self._vd_lifecycle_thread is not None and self._vd_lifecycle_thread.is_alive():
            return
        t = threading.Thread(target=self._vd_lifecycle_worker_loop, daemon=True, name="vd-lifecycle")
        self._vd_lifecycle_thread = t
        t.start()

    def _vd_lifecycle_worker_loop(self) -> None:
        while True:
            job = self._vd_lifecycle_queue.get()
            if job is None:
                break
            try:
                job()
            except Exception as ex:
                try:
                    self._emit({"type": "log", "message": f"[VD-lifecycle] {ex}"})
                except Exception:
                    pass

    def finalize_start_voice_dialog(self, payload: dict) -> None:
        """ensure_* 完了後にイベントループ上で呼ばれ、対話 state を起動する（race回避）。
        進行中に stop / 新しい start が来ていれば epoch 不一致で起動をスキップする。"""
        try:
            epoch = int(payload.get("_epoch", -1))
        except (TypeError, ValueError):
            epoch = -1
        with self._vd_lifecycle_lock:
            if epoch != self._vd_start_epoch:
                return  # この start は stop / 新 start に上書きされた
        settings_obj = payload.get("_settings") if isinstance(payload.get("_settings"), dict) else {}
        self.update_settings(settings_obj)
        self._relay_mmd_burst_setting(settings_obj)
        self.reset_turn_latency()
        self.clear_interrupt_stop()
        self.clear_nonstream_stop_flag()
        self.notify_maai_system_inject_activity()
        self._vd_voice_dialog_running_for_maai_tick.set()
        self._relay_blink_dialog_active()
        if self._orchestrator is not None:
            self._orchestrator.start(settings_obj)

    def handle_start_voice_dialog(self, settings_obj: dict) -> None:
        self._emit({"type": "status", "state": "starting", "message": "音声対話：起動中…"})
        with self._vd_lifecycle_lock:
            self._vd_start_epoch += 1
            epoch = self._vd_start_epoch
        self._ensure_vd_lifecycle_worker()

        def _job() -> None:
            # 重い ensure_*（最大180s）を専用スレッドで実行し、イベントループを塞がない。
            # 完了後、state 起動は finalize でイベントループ上へ戻す（orchestrator.state 競合回避）。
            self.ensure_voice_dialog_workers(settings_obj, epoch=epoch)
            self._event_queue.put(
                {"type": "_vd_start_finalize", "_epoch": epoch, "_settings": settings_obj}
            )

        self._vd_lifecycle_queue.put(_job)

    def handle_update_settings(self, settings_obj: dict) -> None:
        if self._orchestrator is not None:
            self._orchestrator.update_settings(settings_obj)
        self.update_settings(settings_obj)
        self._relay_mmd_burst_setting(settings_obj)

    def handle_stop_voice_dialog(self, payload: dict) -> None:
        keep_maai = bool(payload.get("keep_maai", False))
        keep_asr = bool(payload.get("keep_asr", False))
        # 進行中 start の finalize を無効化（停止後に再起動させないため）。
        with self._vd_lifecycle_lock:
            self._vd_start_epoch += 1
        self._vd_voice_dialog_running_for_maai_tick.clear()
        self._relay_blink_dialog_active()
        self._mmd.cancel_nod_motion()
        self.reset_turn_latency()
        self._emit({"type": "status", "state": "stopping", "message": "音声対話：停止中…"})
        # 停止は発話を確定的に止めてからワーカーを落とす(#13)。割り込みフラグを立て、保留 LLM /
        # セグメント合成ジョブを破棄し、audio_output を即フラッシュする。次 start の finalize が
        # clear_interrupt_stop / clear_nonstream_stop_flag するため、ここで立てても安全。
        self._vd_interrupt_stop.set()
        self._vd_nonstream_stop_flag.set()
        with self._tail_done_timer_lock:
            if self._tail_done_timer is not None:
                self._tail_done_timer.cancel()
                self._tail_done_timer = None
        self._discard_pending_tts_segments()
        with self._llm_pending_lock:
            for request_id in list(self._llm_pending):
                self._remember_llm_terminal_locked(str(request_id))
                self._pop_llm_pending_locked(request_id)
        try:
            self._supervisor.send_audio_output({"type": "interrupt"})
        except Exception:
            pass
        # state 停止はイベントループ上で即時に（このハンドラは loop 上で実行される）。
        if self._orchestrator is not None:
            self._orchestrator.stop()
        # 重いワーカー停止は専用スレッドへ（start の ensure と直列化）。
        self._ensure_vd_lifecycle_worker()

        def _stop_workers_job() -> None:
            self._supervisor.stop_voice_dialog_workers(keep_maai=keep_maai, keep_asr=keep_asr)
            # 停止した実体に合わせ Controller の ready フラグを下げる(#2)。orchestrator 主導の停止
            # （scheduled_exit 等、Controller の _stop_voice_dialog を経由しない経路）でも
            # _asr_ready / _maai_ready が stale True で残らないよう module_stopped を明示発火する。
            if not keep_asr:
                self._emit({"type": "module_stopped", "module": "asr"})
                self._emit({"type": "module_stopped", "module": "asr_compare"})
            if not keep_maai:
                self._emit({"type": "module_stopped", "module": "maai"})

        self._vd_lifecycle_queue.put(_stop_workers_job)

    def _run_audio_send_loop(self, name: str, srv: Any, q: "queue.Queue") -> None:
        """per-target 音声送信スレッド。キューからフレームを取り出して送る。
        srv.send() がブロックしても他のターゲットには影響しない。"""
        _err_count = 0
        while True:
            payload = q.get()
            if payload is None:
                return
            # ダウンストリーム（ASR/MaAI ワーカー）はほぼ実時間速度でしか音声を
            # 消費できない。起動時のモデルロード中にこのキュー（および TCP 送信
            # バッファ）が満杯まで溜まると、その後も満杯のまま流れ続け、VAD/プロットが
            # 数秒遅れる固定遅延になる。バックログが大きいときは古いフレームを捨てて
            # 最新フレームへ追いつく。asr/worker 側はギャップ検出で絶対 ts/sample に
            # 再アンカーするため、ドロップしてもクロックはズレない。
            if q.qsize() > 50:  # > ~1.0s 分溜まったら
                _keep_to = 5  # ~0.1s 残して破棄
                while q.qsize() > _keep_to:
                    try:
                        payload = q.get_nowait()
                    except queue.Empty:
                        break
            try:
                if srv.is_connected():
                    srv.send(payload)
                _err_count = 0
            except Exception:
                _err_count += 1
                if _err_count % 200 == 1:
                    try:
                        self._emit({"type": "log", "message": f"[audio-send-{name}] 送信失敗 (連続{_err_count}フレーム)"})
                    except Exception:
                        pass

    def handle_audio_frame(self, cmd: str, payload: dict) -> None:
        frame_payload = {
            "type": cmd,
            "audio": payload.get("audio", []),
            "sample_start": payload.get("sample_start"),
            "sample_end": payload.get("sample_end"),
            "ts_start": payload.get("ts_start"),
            "ts_end": payload.get("ts_end"),
            # 音声入力ワーカーが送り出した壁時計時刻。ASR が端から端までの遅延（latency_sec）の起点に使う。
            # 落とすと ASR ワーカーの受信時刻が起点になり、対話サーバーを経由する分の遅れが測れない。
            "emitted_at": payload.get("emitted_at"),
            "sample_rate": payload.get("sample_rate", 16000),
            "channel": payload.get("channel", "user"),
            "compare_slot": payload.get("compare_slot", "top"),
        }
        for q in (self._asr_audio_q, self._asr_compare_audio_q, self._maai_audio_q):
            try:
                q.put_nowait(frame_payload)
            except queue.Full:
                pass  # 遅延ワーカー向けフレームはドロップ
        # Controller が GuiPlot チェックで描画可否を判定する。ワーカー側で
        # maai_plot_audio_enabled を見て抑止すると、MaAI 単独起動など settings 未同期時に
        # Input waveform が常に空になる。
        self._emit(frame_payload)

    def handle_run_llm_test(self, payload: dict) -> None:
        rid = str(payload.get("request_id", "")).strip() or f"run-llm-{int(time.time() * 1000)}"
        input_text = str(payload.get("input_text", "")).strip()
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
        llm_conn = (
            payload.get("llm_connection") if isinstance(payload.get("llm_connection"), dict) else {}
        )
        llm_payload = {
            "type": "start",
            "request_id": rid,
            "messages": [
                {"role": "system", "content": str(settings.get("system_prompt", ""))},
                {"role": "user", "content": input_text},
            ],
            "backend": str(settings.get("llm_backend", "LM Studio")),
            "model": str(settings.get("llm_model", "")),
            "max_tokens": int(settings.get("llm_max_tokens", 128) or 128),
            "stream": bool(payload.get("stream", False)),
            # LLM ワーカーは host/api_key/api_key_env をトップレベルで読む（nested "connection" は無視される）。
            "host": str(llm_conn.get("host", "")),
            "api_key": str(llm_conn.get("api_key", "")),
            "api_key_env": str(llm_conn.get("api_key_env", "")),
        }
        if not self._supervisor.send_llm(llm_payload):
            self._emit(
                {
                    "type": "llm_error",
                    "request_id": rid,
                    "message": "LLM worker が未接続です（先に LLM ワーカーを起動してください）。",
                }
            )
        else:
            self._emit({"type": "log", "message": f"[run_llm_test] dispatched rid={rid}"})

    def handle_run_tts_test(self, payload: dict) -> None:
        rid = str(payload.get("request_id", "")).strip() or f"run-tts-{int(time.time() * 1000)}"
        engine = str(payload.get("engine", "")).strip()
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
        if not engine:
            self._emit({"type": "tts_error", "request_id": rid, "message": "engine が未指定です。"})
            return
        if tps.is_worker_engine(engine):
            ok = self._supervisor.send_tts(
                {
                    "type": "synthesize",
                    "request_id": rid,
                    "engine": engine,
                    "inputs": dict(inputs or {}),
                }
            )
            if not ok:
                self._emit(
                    {
                        "type": "tts_error",
                        "request_id": rid,
                        "message": "TTS worker が未接続です。",
                    }
                )
        elif tps.is_bridge_engine(engine):
            req = tps.SynthRequest(rid, engine, dict(inputs or {}))
            try:
                if engine == "VOICEVOX":
                    cmd_dict = tps.build_voicevox_bridge_command(req)
                else:
                    cmd_dict = tps.build_template_bridge_command(req)
                self._supervisor.handle_tts_bridge_command(cmd_dict)
            except Exception as ex:
                self._emit({"type": "tts_error", "request_id": rid, "message": str(ex)})
        else:
            self._emit({"type": "tts_error", "request_id": rid, "message": f"未対応エンジン: {engine}"})

    def handle_run_asr_test(self, payload: dict) -> None:
        rid = str(payload.get("request_id", "")).strip() or f"run-asr-{int(time.time() * 1000)}"
        command_str = str(payload.get("command", "")).strip()
        if command_str:
            self._supervisor.asr.send_command(command_str)
            self._emit({"type": "log", "message": f"[run_asr_test] dispatched rid={rid} cmd={command_str}"})
        else:
            self._emit(
                {
                    "type": "log",
                    "message": "[run_asr_test] command が未指定です（force_finalize 等を指定してください）。",
                }
            )

    def handle_run_maai_test(self, payload: dict) -> None:
        rid = str(payload.get("request_id", "")).strip() or f"run-maai-{int(time.time() * 1000)}"
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else None
        if isinstance(settings, dict):
            self._emit(
                {
                    "type": "log",
                    "message": (
                        f"[run_maai_test] use start_module/update_settings RPC; rid={rid}"
                    ),
                }
            )
        else:
            self._emit({"type": "log", "message": f"[run_maai_test] noop rid={rid}"})

    def handle_start_module(self, payload: dict) -> None:
        module = str(payload.get("module", "")).strip().lower()
        config = payload.get("config")
        config_obj = config if isinstance(config, dict) else {}
        threading.Thread(
            target=self.start_module_async,
            args=(module, config_obj, dict(payload)),
            daemon=True,
            name=f"start-module-{module}",
        ).start()

    def handle_stop_module(self, payload: dict) -> None:
        module = str(payload.get("module", "")).strip().lower()
        if not module:
            return

        def _run() -> None:
            try:
                self._supervisor.stop_module(module)
                self._emit({"type": "module_stopped", "module": module})
            except Exception as ex:
                self._emit(
                    {
                        "type": "log",
                        "message": f"[Supervisor] stop_module({module}) failed: {ex}",
                    }
                )

        threading.Thread(
            target=_run,
            daemon=True,
            name=f"stop-module-{module}",
        ).start()

    def handle_start_llm_warmup(self, payload: dict) -> None:
        wcmd = dict(payload)

        def _emit_ms(state: str, error: str = "") -> None:
            p: dict[str, object] = {
                "type": "module_state",
                "module": "llm",
                "state": state,
                "source": "voice_dialog",
            }
            if error:
                p["error"] = error
            self._emit(p)

        def _run() -> None:
            _emit_ms("starting")
            try:
                self._supervisor.ensure_llm(timeout_sec=40.0)
                settings: dict[str, object] = {
                    "llm_backend": str(wcmd.get("backend", "LM Studio") or "LM Studio"),
                    "llm_model": str(wcmd.get("model", "") or ""),
                    "llm_connection": {
                        "host": str(wcmd.get("host", "") or ""),
                        "api_key": str(wcmd.get("api_key", "") or ""),
                        "api_key_env": str(wcmd.get("api_key_env", "") or ""),
                    },
                    "system_prompt": str(self._llm_settings.get("system_prompt", "") or ""),
                    "voice_dialog_opening_utterance": str(
                        self._llm_settings.get("voice_dialog_opening_utterance", "") or ""
                    ),
                    "force": bool(wcmd.get("force", True)),
                }
                for key in (
                    "send_temperature",
                    "send_max_tokens",
                    "temperature",
                    "custom_extra_body_thinking",
                ):
                    if key in wcmd:
                        settings[key] = wcmd.get(key)
                wait = self._begin_llm_worker_warmup(settings)
                if wait is not None:
                    self._await_llm_worker_warmup(wait, epoch=None)
                err = str(getattr(self, "_llm_warmup_error", "") or "")
                if not str(settings.get("llm_model", "")).strip():
                    err = err or "model が空です"
                if err:
                    _emit_ms("error", err)
                else:
                    _emit_ms("running")
            except Exception as ex:
                self._llm_warmup_error = str(ex)
                _emit_ms("error", str(ex))
                self._emit({"type": "log", "message": f"[LLM-Warmup] {ex}"})

        threading.Thread(target=_run, daemon=True, name="orch-llm-warmup").start()

    def handle_start_tts_worker_warmup(self, payload: dict) -> None:
        tcmd = dict(payload)

        def _run() -> None:
            try:
                eng = str(tcmd.get("engine", "")).strip()
                cfg = tcmd.get("tts_worker_config")
                ins = tcmd.get("inputs") if isinstance(tcmd.get("inputs"), dict) else {}
                to = float(tcmd.get("timeout_sec", 60.0) or 60.0)
                if not eng or not isinstance(cfg, dict):
                    raise RuntimeError("engine / tts_worker_config が不正です。")
                if self._tts_router is None:
                    raise RuntimeError("tts_router が未設定です。")
                self._tts_router.warmup_tts_worker_synthesis(eng, cfg, dict(ins), to)
                self._emit({"type": "log", "message": f"[TTS-Warmup] {eng} 完了"})
                self._tts_warmup_error = ""
            except Exception as ex:
                # LLM warmup と揃える: warmup が通らないなら「稼働」にしない。
                # 成功時に running は出さない（TTS のパネルはエンジン名を detail に
                # 持っており、detail が空の module_state で上書きするとラベルが消える）。
                message = f"[TTS-Warmup] {ex}"
                self._tts_warmup_error = str(ex)
                self._emit({"type": "log", "message": message})
                self._emit({
                    "type": "module_state",
                    "module": "tts",
                    "state": "error",
                    "source": "voice_dialog",
                    "error": str(ex),
                })

        threading.Thread(target=_run, daemon=True, name="orch-tts-warmup").start()

    def handle_start_llm_gui_test(self, payload: dict) -> None:
        gui_cmd = dict(payload)

        def _run() -> None:
            try:
                self.dispatch_llm_gui_test(gui_cmd)
            except Exception as ex:
                self._orch_vd_tts_state["suppress_orch"] = False
                self._emit(
                    {
                        "type": "llm_test_finished",
                        "ok": False,
                        "backend": str(gui_cmd.get("backend", "")),
                        "model": str(gui_cmd.get("model", "")),
                        "context_len": int(gui_cmd.get("context_len", 10) or 10),
                        "assistant_text": "",
                        "error": str(ex),
                        "ttft_sec": 0.0,
                        "total_sec": 0.0,
                        "total_overall_sec": 0.0,
                    }
                )
                self._emit({"type": "log", "message": f"[Orchestrator][GUI-LLM] {ex}"})

        threading.Thread(target=_run, daemon=True, name="orch-llm-gui-test").start()

    def handle_start_tts_gui_test(self, payload: dict) -> None:
        tcmd = dict(payload)

        def _run() -> None:
            try:
                self.dispatch_tts_gui_test(tcmd)
            except Exception as ex:
                self._emit(
                    {
                        "type": "tts_test_finished",
                        "ok": False,
                        "engine": str(tcmd.get("engine", "")),
                        "error": str(ex),
                        "started_at": float(tcmd.get("started_at", time.perf_counter()) or time.perf_counter()),
                        "mmd_running": bool(tcmd.get("mmd_running", False)),
                    }
                )

        threading.Thread(target=_run, daemon=True, name="orch-tts-gui-test").start()

    def llm_dispatch_thread(self, *, prefetch: bool = False) -> None:
        """`begin_llm_turn_for_latency` + orchestrator.on_llm_state(True) + dispatch_llm_request をスレッドで実行する。

        ユーザー発話は引数で受けない（`dispatch_llm_request` が履歴から組むため）。
        prefetch 時も緑帯（LLM 遅延）は開始する。llm_in_progress は立てない（p_now を捨てないため）。
        """
        self.begin_llm_turn_for_latency()
        if not prefetch and self._orchestrator is not None:
            self._orchestrator.on_llm_state(True)
        threading.Thread(
            target=self.dispatch_llm_request,
            kwargs={"prefetch": bool(prefetch)},
            daemon=True,
            name="orch-llm-dispatch-prefetch" if prefetch else "orch-llm-dispatch",
        ).start()

    # -- worker dispatch helpers (orchestrator passthrough) ------------------

    def handle_turn(self, payload: dict) -> None:
        prob = float(payload.get("probability", 0.0) or 0.0)
        try:
            p_future = float(payload.get("p_future", 0.0) or 0.0)
        except (TypeError, ValueError):
            p_future = 0.0
        events = (
            self._orchestrator.on_turn_probability(prob, p_future=p_future)
            if self._orchestrator is not None
            else []
        )
        # ターン取得フロー以外のフレームでは orchestrator が [] のみ。確率・時刻を UI に渡す。
        # from_orchestrator=True は「ターン取得イベントではない」印。Controller は縦線を引かない。
        if not any(str(ev.get("type", "")).strip().lower() == "turn" for ev in events):
            ts_raw = payload.get("ts", time.time())
            try:
                ts_f = float(ts_raw) if ts_raw is not None else time.time()
            except (TypeError, ValueError):
                ts_f = time.time()
            turn_out: dict[str, object] = {
                "type": "turn",
                "probability": prob,
                "p_future": p_future,
                "ts": ts_f,
                "from_orchestrator": True,
            }
            st = payload.get("source_ts_end")
            if st is not None:
                try:
                    turn_out["source_ts_end"] = float(st)
                except (TypeError, ValueError):
                    pass
            self._emit(turn_out)

    def handle_maai_result(self, payload: dict) -> None:
        result = payload.get("result")
        result_obj = result if isinstance(result, dict) else {}
        # Silero VAD / p_now を worker 内に保存（割り込み判定で使用）
        self.update_vad_from_maai_result(result_obj)
        if self._orchestrator is not None:
            self._orchestrator.on_maai_result(result_obj)
        # 音声対話非 active 時は on_maai_result が [] のみのため、プロット用に必ず Controller へ渡す
        self._emit(
            {
                "type": "maai_result",
                "compare_slot": str(payload.get("compare_slot", "")).strip().lower(),
                "result": dict(result_obj),
                "from_orchestrator": True,
            }
        )

    def _handle_maai_aux_viz(self, payload: dict, orchestrator_method: str) -> None:
        result = payload.get("result")
        orch = self._orchestrator
        if orch is not None:
            getattr(orch, orchestrator_method)(result if isinstance(result, dict) else {})
        viz = dict(payload)
        viz["from_orchestrator"] = True
        self._emit(viz)

    def handle_maai_nod_result(self, payload: dict) -> None:
        self._handle_maai_aux_viz(payload, "on_maai_nod_result")

    def handle_maai_bc_result(self, payload: dict) -> None:
        self._handle_maai_aux_viz(payload, "on_maai_bc_result")

    def handle_raw_audio(self, payload: dict) -> None:
        self._emit(
            {
                "type": "raw_audio",
                "compare_slot": str(payload.get("compare_slot", "")).strip().lower(),
                "audio": payload.get("audio", []),
                "ts_start": payload.get("ts_start"),
                "ts_end": payload.get("ts_end"),
                "sample_start": payload.get("sample_start"),
                "sample_end": payload.get("sample_end"),
                "channel": payload.get("channel"),
                # 受け取った時点の壁時計。ts_* はサンプルクロックで、ブラウザマイクでは壁時計から
                # 遅れていく。Web のプロットはこの差で壁時計の時刻（遅延帯など）を音声時刻へ換算する。
                "wall_ts": time.time(),
            }
        )

    def handle_force_finalize_status(self, payload: dict) -> None:
        if self._orchestrator is None:
            return
        try:
            request_id = int(payload.get("request_id", 0) or 0)
        except Exception:
            request_id = 0
        self._orchestrator.on_force_finalize_status(
            is_speaking=bool(payload.get("is_speaking", False)),
            request_id=request_id,
            ts=float(payload.get("ts", time.time()) or time.time()),
        )

    def handle_force_finalize_outcome(self, payload: dict) -> None:
        if self._orchestrator is None:
            return
        try:
            request_id = int(payload.get("request_id", 0) or 0)
        except Exception:
            request_id = 0
        self._orchestrator.on_force_finalize_outcome(
            status=str(payload.get("status", "")).strip().lower(),
            request_id=request_id,
            ts=float(payload.get("ts", time.time()) or time.time()),
            error=str(payload.get("error", "")),
        )

    def handle_llm_finished(self, payload: dict) -> None:
        if self._orchestrator is None:
            return
        self._orchestrator.on_llm_finished(
            ok=bool(payload.get("ok", False)),
            assistant_text=str(payload.get("assistant_text", "")),
            error=str(payload.get("error", "")),
        )
