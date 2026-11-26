"""`_on_dialog_orchestrator_event` の分岐本体（controller.py から移設）。

Orchestrator から届くイベントを controller の状態・シグナルへ反映する。
`dialog/worker.py` の DISPATCH テーブルと同じ形にしてある。

**順序に意味がある**: 元は 40 個の独立した `if` が並んでおり（すべて `return` で終わる
ので実質 elif）、その途中に `_src`（送信元ワーカー）で振り分ける 3 分岐が挟まっていた。
`_src` より後ろの event_type 分岐は「_src が llm_admin/tts/audio_output のどれでもない」
ときだけ到達する。この順序を保つため、テーブルを EARLY / LATE の 2 枚に分けてある。
本文は controller.py から 1 文字も変えずに移してある。
"""

from __future__ import annotations

import json
import time

from PySide6.QtCore import QTimer


class OrchestratorEventMixin:
    """Orchestrator イベントの分岐本体。`MMDAgentController` に mixin する。"""

    def _orch_ev_ready(self, payload: dict, event_type: str) -> None:
        if str(payload.get("worker", "")).strip() == "audio_output":
            self._remote_plugin_settings_digest_sent = None
            self._audio_output_worker_ready_event.set()
            om = str(payload.get("output_mode", "local") or "local").strip().lower()
            self._audio_output_worker_started_output_mode = om or "local"
            self.process_output_signal.emit("[AudioOutput] ready")
            # メイン ready の時点では Audio Output 未起動で remote_plugin がスキップされ得る。
            # ここで再送して MMDAgent ブリッジを確実に構成する。
            if self._avatar_output_available():
                self._send_remote_plugin_settings()
            self._try_send_maai_mmd_av_preamble_once()
            return
        self._dialog_orchestrator_ready_event.set()
        if getattr(self, "_remote_backend_enabled", False):
            self._remote_orchestrator_retry_scheduled = False
        self._push_dialog_orchestrator_settings(force=True)
        QTimer.singleShot(0, self._sync_orchestrator_audio_io_after_ready)
        # MMD 起動が先で Orchestrator が後から立つ場合、launch_mmdagent 時点では
        # まだ IPC が無く Audio Output Worker へ remote_plugin が届いていない。
        if self._avatar_output_available():
            self._send_remote_plugin_settings()
        self._try_send_maai_mmd_av_preamble_once()
        self.process_output_signal.emit("[DialogOrchestrator] ready")
        return

    def _orch_ev_status(self, payload: dict, event_type: str) -> None:
        msg = str(payload.get("message", "")).strip()
        state = str(payload.get("state", "")).strip().lower()
        if msg:
            self.voice_dialog_status_label.setText(msg)
            self.process_output_signal.emit(f"[DialogOrchestrator] {msg}")
        if state in {"running", "starting"}:
            self._voice_dialog_active = state == "running"
            self._voice_dialog_starting = state == "starting"
        elif state in {"stopping", "stopped"}:
            self._voice_dialog_active = False
            self._voice_dialog_starting = state == "stopping"
        self._set_voice_dialog_controls_state()
        return

    def _orch_ev_history_snapshot(self, payload: dict, event_type: str) -> None:
        history = payload.get("history")
        if isinstance(history, dict):
            msgs = history.get("messages")
            pending = history.get("pending_texts")
            partial = history.get("current_partial")
            llm_in_progress = history.get("llm_in_progress")
            if isinstance(msgs, list):
                self._voice_dialog_messages = [
                    {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                    for m in msgs
                    if isinstance(m, dict)
                ]
            if isinstance(pending, list):
                self._voice_dialog_pending_texts = [str(x) for x in pending]
            if partial is not None:
                self._voice_dialog_current_partial = str(partial)
            tb = history.get("turn_blocked_until_user_speech")
            if tb is None:
                tb = history.get("turn_locked_until_tts_done")
            if tb is not None:
                if (
                    hasattr(self, "maai_plot_window")
                    and self.maai_plot_window is not None
                    and hasattr(self, "asr_maai_gui_plot_check")
                    and self.asr_maai_gui_plot_check.isChecked()
                    and hasattr(self.maai_plot_window, "update_turn_lock_state")
                ):
                    snap_ts = history.get("ts")
                    self.maai_plot_window.update_turn_lock_state(
                        bool(tb),
                        event_ts=float(snap_ts) if snap_ts is not None else time.time(),
                    )
            self._request_voice_dialog_history_refresh()
        return

    def _orch_ev_turn(self, payload: dict, event_type: str) -> None:
        prob = float(payload.get("probability", 0.0) or 0.0)
        # dialog/worker が「オーケストレータが turn を返さないフレーム」用に
        # from_orchestrator=True の合成 turn を毎推論送る。これを縦線にするとフレーム毎に
        # InfiniteLine が増えて描画が破綻するため、実ターン取得（オーケストレータ発行）のみプロットする。
        is_synthetic_turn_frame = bool(payload.get("from_orchestrator"))
        if not is_synthetic_turn_frame:
            self.process_output_signal.emit(f"[ASR][TURN] p={prob:.3f}")
        if (
            not is_synthetic_turn_frame
            and hasattr(self, "maai_plot_window")
            and self.maai_plot_window is not None
            and self.asr_maai_gui_plot_check.isChecked()
        ):
            self.maai_plot_window.update_voice_dialog_turn(
                event_ts=float(payload.get("ts", time.time()) or time.time())
            )
        return

    def _orch_ev_trigger_llm(self, payload: dict, event_type: str) -> None:
        user_text = str(payload.get("user_text", "")).strip()
        is_prefetch = bool(payload.get("prefetch", False))
        if is_prefetch:
            self.process_output_signal.emit("[VoiceDialog/Orchestrator] prefetch (p_future) started")
            if (
                hasattr(self, "maai_plot_window")
                and self.maai_plot_window is not None
                and hasattr(self, "asr_maai_gui_plot_check")
                and self.asr_maai_gui_plot_check.isChecked()
                and hasattr(self.maai_plot_window, "update_voice_dialog_prefetch")
            ):
                ts_raw = payload.get("ts", time.time())
                try:
                    ts_f = float(ts_raw) if ts_raw is not None else time.time()
                except (TypeError, ValueError):
                    ts_f = time.time()
                self.maai_plot_window.update_voice_dialog_prefetch(event_ts=ts_f)
        elif user_text:
            self.process_output_signal.emit(f"[VoiceDialog/Orchestrator] trigger_llm: {user_text[:80]}")
        return

    def _orch_ev_module_started(self, payload: dict, event_type: str) -> None:
        module = str(payload.get("module", "")).strip().lower()
        ev = self._module_ready_events.get(module)
        if ev is not None:
            ev.set()
        if not payload.get("unchanged"):
            self.process_output_signal.emit(f"[Orchestrator] {module} started")
        self._set_module_panel_state(module, "running")
        if module in ("asr", "asr_compare"):
            self._asr_ready = True
            self.asr_start_finished_signal.emit(True, "")
        elif module == "maai":
            self._maai_ready = True
            self._maai_starting = False
            self._update_maai_test_button_states()
            self.process_output_signal.emit("[MaAI] worker ready")
        elif module == "llm":
            self._llm_worker_ready_event.set()
        elif module == "tts":
            self._tts_worker_ready_event.set()
        elif module == "audio_input":
            self._audio_input_worker_ready_event.set()
            self._sync_remote_mic_capture()
        elif module == "audio_output":
            # module_started が ready より先に届く場合の保険（output_mode は worker ready で確定）
            self._audio_output_worker_ready_event.set()
        return

    def _orch_ev_module_start_failed(self, payload: dict, event_type: str) -> None:
        module = str(payload.get("module", "")).strip().lower()
        error = str(payload.get("error", "")).strip()
        self._module_start_errors[module] = error
        self._set_module_panel_state(module, "error", error)
        ev = self._module_ready_events.get(module)
        if ev is not None:
            ev.set()
        self.process_output_signal.emit(f"[Orchestrator] {module} start failed: {error}")
        if module in ("asr", "asr_compare"):
            self._asr_starting = False
            self.asr_start_finished_signal.emit(False, error)
        elif module == "maai":
            self._maai_starting = False
            self._update_maai_test_button_states()
        return

    def _orch_ev_module_stopped(self, payload: dict, event_type: str) -> None:
        module = str(payload.get("module", "")).strip().lower()
        self.process_output_signal.emit(f"[Orchestrator] {module} stopped")
        self._set_module_panel_state(module, "stopped")
        if module == "audio_output":
            self._audio_output_worker_ready_event.clear()
            self._audio_output_worker_started_output_mode = None
        elif module == "audio_input":
            self._audio_input_worker_ready_event.clear()
            self._audio_input_processing_event.clear()
            self._stop_remote_mic_capture()
        elif module == "maai":
            self._maai_ready = False
            self._update_maai_test_button_states()
        elif module in ("asr", "asr_compare"):
            # orchestrator 主導の停止（scheduled_exit 等）でも _asr_ready が stale True で
            # 残らないようにする(#2)。Controller 主導の停止経路でも冪等に下がる。
            # _asr_starting はここで触らない（maai 分岐と同様 ready のみ）。module_stopped は
            # 非同期かつ世代非タグなので、旧 stop の遅延イベントが新 start 進行中に届くと
            # 起動状態を誤キャンセルし launch ボタンが復活して二重起動を招く(#2 round-3)。
            # _asr_starting は _begin_asr_starting / _on_asr_start_finished / module_start_failed が管理する。
            self._asr_ready = False
        return

    def _orch_ev_module_state(self, payload: dict, event_type: str) -> None:
        module = str(payload.get("module", "")).strip().lower()
        state = str(payload.get("state", "")).strip().lower()
        error = str(payload.get("error", "")).strip()
        if state == "running":
            if module == "asr":
                self._asr_ready = True
                self._set_asr_controls_state()
            elif module == "maai":
                self._maai_ready = True
                self._maai_starting = False
                self._update_maai_test_button_states()
            elif module == "llm":
                self._llm_worker_ready_event.set()
            elif module == "tts":
                self._tts_worker_ready_event.set()
                # 「どのエンジンか」も控える。_is_tts_engine_running は
                # ready と engine の両方を見るので、これが None のままだと
                # メイン画面から起動した TTS がテスト画面で「停止」に見える。
                # 自前で起動した場合は _ensure_tts_worker が既に入れている。
                engine = str(payload.get("engine", "")).strip()
                if engine:
                    self._tts_worker_engine = engine
                # 実際に適用された設定の指紋。TTS テスト送信のように GUI を
                # 経由しない経路で ensure_tts が別設定に差し替えても、これで
                # 追随できる（自前で覚えるだけだと古いまま残り、起動し直しが
                # 必要なのに省いてしまう）。
                digest = str(payload.get("config_digest", "")).strip()
                if digest:
                    self._tts_worker_config_digest = digest
            elif module == "audio_input":
                self._audio_input_worker_ready_event.set()
                self._sync_remote_mic_capture()
            elif module == "audio_output":
                # ワーカー自身の ready が本命だが、それが来ない場合がある:
                # ensure_audio_output は同じ設定で既に動いていれば起動し直さず、
                # 新しい ready を出さない。先行 bootstrap もこのイベントしか出さない。
                # ready を待ってから running を出す実装なので、ここで立てて早すぎない。
                # output_mode はこのイベントに無いので
                # _audio_output_worker_started_output_mode は触らない（ready 側の担当）。
                self._audio_output_worker_ready_event.set()
        self._set_module_panel_state(module, state, error)
        return

    def _orch_ev_worker_disconnected(self, payload: dict, event_type: str) -> None:
        worker = str(payload.get("worker", "")).strip().lower()
        module = {"llm": "llm", "tts": "tts"}.get(worker)
        if module == "llm":
            self._llm_worker_ready_event.clear()
        elif module == "tts":
            self._tts_worker_ready_event.clear()
        if module:
            self._set_module_panel_state(module, "error", "ワーカー IPC が切断されました")
            self.process_output_signal.emit(f"[Orchestrator] {worker} worker IPC 切断")
        return

    def _orch_ev_request_nod_motion(self, payload: dict, event_type: str) -> None:
        self._maai_test_log_surface_nod(dict(payload))
        return

    # 非ストリーム TTS（オープニング / 相槌 / スケジュール終了）は
    # dialog/worker 内で合成・再生まで完結。Controller から重複して
    # _voice_dialog_speak_nonstream_async すると二重合成・on_tts_done 二重になり得るため、
    # ここではログ・UI ミラーのみ。
    def _orch_ev_request_nonstream_tts(self, payload: dict, event_type: str) -> None:
        assistant_text = str(payload.get("assistant_text", "")).strip()
        if assistant_text:
            source = str(payload.get("source", "")).strip().lower()
            if source == "maai_backchannel":
                # 相槌は MaAI ログ側に本文を出すので、ここでは重ねてログしない。
                self._maai_test_log_surface_backchannel(dict(payload), assistant_text)
            else:
                self.process_output_signal.emit(f"[VD] 発話: {self._log_snip(assistant_text)}")
        return

    def _orch_ev_request_opening_tts(self, payload: dict, event_type: str) -> None:
        assistant_text = str(payload.get("assistant_text", "")).strip()
        if assistant_text:
            self.process_output_signal.emit(f"[VD] opening: {self._log_snip(assistant_text)}")
        return

    def _orch_ev_request_scheduled_exit_tts(self, payload: dict, event_type: str) -> None:
        assistant_text = str(payload.get("assistant_text", "")).strip()
        if assistant_text:
            self.process_output_signal.emit(f"[VD] 終了発話: {self._log_snip(assistant_text)}")
        return

    def _orch_ev_tts_cumulative_strip_prefix(self, payload: dict, event_type: str) -> None:
        self._voice_dialog_tts_snap_prefix = str(payload.get("prefix", "") or "")
        return

    def _orch_ev_tts_mmd_send_complete(self, payload: dict, event_type: str) -> None:
        if self._is_dialog_orchestrator_runtime_active():
            return
        if str(payload.get("complete_kind") or "").strip().lower() == "segment":
            try:
                si = int(payload.get("segment_index", 0))
            except (TypeError, ValueError):
                si = 0
            snap = list(getattr(self, "_voice_dialog_tts_segments_snapshot", []) or [])
            if not snap:
                return
            if 0 <= si < len(snap):
                cumulative = "".join(snap[: si + 1])
            else:
                cumulative = "".join(snap)
            pfx = str(getattr(self, "_voice_dialog_tts_snap_prefix", "") or "")
            if pfx and cumulative.startswith(pfx):
                cumulative = cumulative[len(pfx) :].lstrip()
            self._send_dialog_orchestrator_command_sync(
                {"type": "assistant_segment_played", "cumulative_text": cumulative}
            )
            return
        return

    def _orch_ev_latency(self, payload: dict, event_type: str) -> None:
        self.voice_dialog_latency_signal.emit(payload)
        return

    def _orch_ev_asr_result(self, payload: dict, event_type: str) -> None:
        p = dict(payload)
        p["from_orchestrator"] = True
        self._append_asr_result_event(p)
        return

    def _orch_ev_audio_frame(self, payload: dict, event_type: str) -> None:
        if self.asr_maai_gui_plot_check.isChecked():
            raw_payload = {
                "channel": "system" if event_type == "system_audio_frame" else "user",
                "compare_slot": str(payload.get("compare_slot", "")).strip().lower(),
                "audio": payload.get("audio", []),
                "ts_start": payload.get("ts_start"),
                "ts_end": payload.get("ts_end"),
                "sample_start": payload.get("sample_start"),
                "sample_end": payload.get("sample_end"),
            }
            self.raw_audio_event_signal.emit(raw_payload)
        return

    def _orch_ev_log(self, payload: dict, event_type: str) -> None:
        msg = str(payload.get("message", "")).strip() or json.dumps(payload, ensure_ascii=False)
        self.process_output_signal.emit(f"[DialogOrchestrator] {msg}")
        return

    def _orch_ev_mmd_av_preamble_done(self, payload: dict, event_type: str) -> None:
        ok = bool(payload.get("ok", False))
        if ok:
            self._maai_mmd_av_preamble_sent = True
            self._maai_mmd_av_preamble_fail_count = 0
        else:
            self._maai_mmd_av_preamble_fail_count += 1
            if (
                self._maai_mmd_av_preamble_fail_count < 150
                and self._maai_ready
                and not self._maai_mmd_av_preamble_sent
                and (
                    self._is_maai_backchannel_enabled()
                    or self._is_nod_para_enabled()
                )
                and self._effective_audio_output_mode() == "mmd"
            ):
                QTimer.singleShot(400, self._try_send_maai_mmd_av_preamble_once)
        return

    def _orch_ev_remote_status(self, payload: dict, event_type: str) -> None:
        origin = str(payload.get("_remote_bridge_origin", "") or "").strip().lower()
        inc = bool(payload.get("connected", False))
        mode = str(payload.get("mode", ""))
        if origin == "dialog_orchestrator":
            self._orch_mmd_bridge_connected = inc
            self._orch_mmd_bridge_mode = mode
            conn_str = "connected" if self._orch_mmd_bridge_connected else "disconnected"
            self.process_output_signal.emit(
                f"[DialogOrchestrator] remote bridge {conn_str} (mode={self._orch_mmd_bridge_mode})"
            )
            if inc:
                self._try_send_maai_mmd_av_preamble_once()
        elif origin == "audio_output":
            self._audio_mmd_bridge_connected = inc
            self._audio_mmd_bridge_mode = mode
            conn_str = "connected" if self._audio_mmd_bridge_connected else "disconnected"
            self.process_output_signal.emit(
                f"[AudioOutput] MMD remote bridge {conn_str} (mode={self._audio_mmd_bridge_mode})"
            )
            if inc:
                self._try_send_maai_mmd_av_preamble_once()
        else:
            self.process_output_signal.emit(
                f"[remote_status] origin不明 (connected={inc}, mode={mode})"
            )
        return

    def _orch_ev_nod_motion_chart(self, payload: dict, event_type: str) -> None:
        self._update_nod_motion_chart(payload)
        return

    def _orch_ev_nod_trigger_visual(self, payload: dict, event_type: str) -> None:
        self._update_maai_plot_nod_trigger(payload)
        return

    def _orch_ev_bc_trigger_visual(self, payload: dict, event_type: str) -> None:
        if self._maai_test_surface_jsonl_fp is not None:
            self._maai_test_bc_trigger_log_buffer = dict(payload)
        self._update_maai_plot_bc_trigger(payload)
        return

    def _orch_ev_request_interrupt(self, payload: dict, event_type: str) -> None:
        if not self._is_dialog_orchestrator_runtime_active():
            if self._voice_dialog_interrupt_system_speech():
                self.voice_dialog_latency_signal.emit(
                    {
                        "kind": "interrupt",
                        "end_ts": time.time(),
                    }
                )
        return

    # MaAI-specific results from orchestrator
    def _orch_ev_maai_result(self, payload: dict, event_type: str) -> None:
        result = payload.get("result")
        result_obj = result if isinstance(result, dict) else {}
        self.maai_result_event_signal.emit({
            "compare_slot": str(payload.get("compare_slot", "")).strip().lower(),
            "result": result_obj,
            "from_orchestrator": bool(payload.get("from_orchestrator", False)),
        })
        return

    def _orch_ev_maai_nod_result(self, payload: dict, event_type: str) -> None:
        self.maai_result_event_signal.emit(dict(payload))
        return

    # LLM events from orchestrator voice dialog (llm_chunk, llm_done, llm_tts_segment)
    def _orch_ev_llm_chunk(self, payload: dict, event_type: str) -> None:
        assistant_text = str(payload.get("assistant_text", ""))
        if self._voice_dialog_active or self._voice_dialog_starting:
            self.voice_dialog_llm_chunk_signal.emit({"assistant_text": assistant_text})
        return

    def _orch_ev_llm_done(self, payload: dict, event_type: str) -> None:
        ok = bool(payload.get("ok", False))
        assistant_text = str(payload.get("assistant_text", ""))
        if self._voice_dialog_active or self._voice_dialog_starting:
            error = "" if ok else str(payload.get("error", "") or "").strip()
            self.process_output_signal.emit(
                f"[VoiceDialog] LLM完了: ok={ok} text={assistant_text[:80]}"
                + (f" error={error[:200]}" if error else "")
            )
        return

    def _orch_ev_llm_tts_segment(self, payload: dict, event_type: str) -> None:
        segment_text = str(payload.get("segment_text", ""))
        is_final = bool(payload.get("is_final", False))
        if segment_text and (self._voice_dialog_active or self._voice_dialog_starting):
            self.process_output_signal.emit(
                f"[VoiceDialog] TTS segment: {segment_text[:60]} final={is_final}"
            )
        return

    def _orch_ev_llm_test_chat_chunk(self, payload: dict, event_type: str) -> None:
        text = str(payload.get("text", ""))
        if text:
            self.llm_test_chat_chunk_signal.emit({"text": text})
        return

    def _orch_ev_llm_test_tts_segment(self, payload: dict, event_type: str) -> None:
        cumulative = str(payload.get("cumulative", ""))
        pushed = str(payload.get("pushed_segment", ""))
        if cumulative:
            self.llm_tts_stream_push_signal.emit(
                {
                    "source": "llm_test",
                    "cumulative": cumulative,
                    "pushed_segment": pushed,
                }
            )
        return

    def _orch_ev_llm_test_finished(self, payload: dict, event_type: str) -> None:
        self.llm_test_chat_finished_signal.emit(dict(payload))
        return

    def _orch_ev_tts_test_finished(self, payload: dict, event_type: str) -> None:
        self.tts_test_synthesis_finished_signal.emit(dict(payload))
        return

    # MaAI nod threshold defaults
    def _orch_ev_maai_nod_threshold_defaults(self, payload: dict, event_type: str) -> None:
        nd = payload.get("defaults")
        self._maai_nod_threshold_defaults = nd if isinstance(nd, dict) else None
        self._update_maai_nod_threshold_default_labels()
        return

    # Dialog Orchestrator 経由の Audio Input と直接 IPC のとき同じ処理（WAV 終了で MaAI 停止など）
    def _orch_ev_wav_playback_finished(self, payload: dict, event_type: str) -> None:
        self._on_audio_input_ipc_event(payload if isinstance(payload, dict) else {})
        return


# 値はメソッド名。関数を直接持つと getattr を通らず、通常のメソッド解決を
# 迂回してしまう（差し替えが効かない）。
# _src 判定より前に評価される event_type（元の 0〜25 番目）
ORCH_EVENT_HANDLERS_EARLY: dict[str, str] = {
    "ready": "_orch_ev_ready",
    "status": "_orch_ev_status",
    "history_snapshot": "_orch_ev_history_snapshot",
    "turn": "_orch_ev_turn",
    "trigger_llm": "_orch_ev_trigger_llm",
    "module_started": "_orch_ev_module_started",
    "module_start_failed": "_orch_ev_module_start_failed",
    "module_stopped": "_orch_ev_module_stopped",
    "module_state": "_orch_ev_module_state",
    "worker_disconnected": "_orch_ev_worker_disconnected",
    "request_nod_motion": "_orch_ev_request_nod_motion",
    "request_nonstream_tts": "_orch_ev_request_nonstream_tts",
    "request_opening_tts": "_orch_ev_request_opening_tts",
    "request_scheduled_exit_tts": "_orch_ev_request_scheduled_exit_tts",
    "tts_cumulative_strip_prefix": "_orch_ev_tts_cumulative_strip_prefix",
    "tts_mmd_send_complete": "_orch_ev_tts_mmd_send_complete",
    "latency": "_orch_ev_latency",
    "asr_result": "_orch_ev_asr_result",
    "audio_frame": "_orch_ev_audio_frame",
    "system_audio_frame": "_orch_ev_audio_frame",
    "log": "_orch_ev_log",
    "error": "_orch_ev_log",
    "interrupt_condition": "_orch_ev_log",
    "mmd_av_preamble_done": "_orch_ev_mmd_av_preamble_done",
    "remote_status": "_orch_ev_remote_status",
    "nod_motion_chart": "_orch_ev_nod_motion_chart",
    "nod_trigger_visual": "_orch_ev_nod_trigger_visual",
    "bc_trigger_visual": "_orch_ev_bc_trigger_visual",
    "request_interrupt": "_orch_ev_request_interrupt",
}

# _src 判定より後に評価される event_type（元の 29〜39 番目）
ORCH_EVENT_HANDLERS_LATE: dict[str, str] = {
    "maai_result": "_orch_ev_maai_result",
    "maai_nod_result": "_orch_ev_maai_nod_result",
    "maai_bc_result": "_orch_ev_maai_nod_result",
    "llm_chunk": "_orch_ev_llm_chunk",
    "llm_done": "_orch_ev_llm_done",
    "llm_tts_segment": "_orch_ev_llm_tts_segment",
    "llm_test_chat_chunk": "_orch_ev_llm_test_chat_chunk",
    "llm_test_tts_segment": "_orch_ev_llm_test_tts_segment",
    "llm_test_finished": "_orch_ev_llm_test_finished",
    "tts_test_finished": "_orch_ev_tts_test_finished",
    "maai_nod_threshold_defaults": "_orch_ev_maai_nod_threshold_defaults",
    "wav_playback_finished": "_orch_ev_wav_playback_finished",
    "wav_command_done": "_orch_ev_wav_playback_finished",
    "processing_started": "_orch_ev_wav_playback_finished",
    "processing_stopped": "_orch_ev_wav_playback_finished",
}
