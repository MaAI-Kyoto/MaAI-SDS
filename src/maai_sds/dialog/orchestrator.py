from __future__ import annotations

import math
import time
import random
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..utils import nod_para_frame_decision

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

def _maai_audio_timeline_extra(payload: dict[str, Any]) -> dict[str, object]:
    """MaAI ワーカー付与の source_sample_end / audio_timeline_sec を表出イベントへ引き継ぐ。"""
    out: dict[str, object] = {}
    if "audio_timeline_sec" in payload:
        try:
            out["audio_timeline_sec"] = float(payload["audio_timeline_sec"])
        except (TypeError, ValueError):
            pass
    if "source_sample_end" in payload:
        try:
            out["source_sample_end"] = int(payload["source_sample_end"])
        except (TypeError, ValueError):
            pass
    return out

@dataclass
class DialogOrchestratorState:
    active: bool = False
    llm_in_progress: bool = False
    final_seq: int = 0
    wait_nonstream_maai_turn: dict[str, object] | None = None
    wait_smart_turn_final: bool = False
    waiting_asr_latency: bool = False
    pending_texts: list[str] = field(default_factory=list)
    pending_utterance_indices: list[int | None] = field(default_factory=list)
    current_partial: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    vad_active: bool = False
    asr_inference_busy: bool = False
    last_is_streaming_asr: bool = False
    turn_blocked_until_user_speech: bool = False
    waiting_tts_audible: bool = False
    tts_audible_detected: bool = False
    pending_assistant_text: str | None = None
    # 同一 LLM 応答内でユーザー割り込みが入ったとき、controller 側のセグメント累積から除く先頭プレフィックス
    tts_cumulative_strip_prefix: str = ""
    current_partial_utterance_index: int | None = None
    active_user_msg_index: int | None = None
    active_user_msg_utterance_order: list[int] = field(default_factory=list)
    active_user_msg_utterance_text: dict[int, str] = field(default_factory=dict)
    # MaAI ストリーミングで partial 時点のターンで user を積んだあと、対応 final で履歴を差し替える。
    active_user_msg_await_final_replace: bool = False
    # utterance_index -> messages 内の user 行インデックス（同一発話の partial 再ターンで行を増やさない）
    streaming_user_row_by_utterance: dict[int, int] = field(default_factory=dict)
    last_turn_ts: float | None = None
    last_turn_wall_ts: float | None = None
    prev_turn_probability: float | None = None
    prev_p_future: float | None = None
    # p_future しきい値上抜けで開始した LLM/TTS 先読み（ターン未取得）。
    prefetch_active: bool = False
    prefetch_failed: bool = False
    # streaming ASR で utterance_index 欠落時に使う負の仮ID（履歴行と _stream_utt の対応用）
    streaming_anonymous_seq: int = 0
    last_backchannel_ts: float = 0.0
    backchannel_consecutive_over_threshold: int = 0
    backchannel_buf_ts_start: list[float] = field(default_factory=list)
    backchannel_buf_ts_end: list[float] = field(default_factory=list)
    # nod_para state
    nod_para_consecutive_count: int = 0
    nod_para_buf_p_nod: list[float] = field(default_factory=list)
    nod_para_buf_range: list[float] = field(default_factory=list)
    nod_para_buf_speed: list[float] = field(default_factory=list)
    nod_para_buf_swing_up: list[float] = field(default_factory=list)
    nod_para_buf_swing_up_decided: list[int] = field(default_factory=list)
    nod_para_buf_raw_range: list[float] = field(default_factory=list)
    nod_para_buf_raw_speed: list[float] = field(default_factory=list)
    nod_para_buf_ts_start: list[float] = field(default_factory=list)
    nod_para_buf_ts_end: list[float] = field(default_factory=list)
    nod_para_buf_count_idx: list[int] = field(default_factory=list)
    nod_para_prob_dropped: bool = True
    last_nod_para_ts: float = 0.0
    # 音声対話開始時刻（wall）と、経過秒経過後に「次のターン検知」で終了発話する待ち状態
    voice_dialog_start_wall_ts: float | None = None
    scheduled_exit_waiting_for_turn: bool = False
    # 開始時システム発話の TTS が未完了（再生中）のとき True。終了後はターン取得ブロックを解除する。
    opening_tts_pending: bool = False
    # on_maai_result で user_side となった最終時刻（wall）。アイドル経過後ターン取得用。
    last_user_speech_detect_wall_ts: float | None = None

class DialogOrchestrator:
    """UI非依存の音声対話状態機械。

    実行時機能をプロセス分離しやすくするため、
    controller.py から切り離した最小コア状態を保持する。
    """

    def __init__(
        self,
        *,
        on_emit: Callable[[dict], None] | None = None,
        on_nod_motion: Callable[[dict], None] | None = None,
        on_trigger_llm: Callable[..., None] | None = None,
        on_request_force_finalize: Callable[[], None] | None = None,
        on_request_tts_done: Callable[[], None] | None = None,
        on_request_opening_tts: Callable[[str], None] | None = None,
        on_request_nonstream_tts: Callable[[str, str], None] | None = None,
        on_request_interrupt: Callable[[str], None] | None = None,
        on_request_scheduled_exit_tts: Callable[[str], None] | None = None,
        on_commit_prefetch: Callable[[], None] | None = None,
        on_cancel_prefetch: Callable[[], None] | None = None,
    ) -> None:
        self.state = DialogOrchestratorState()
        self.settings: dict[str, Any] = {}
        self._on_emit: Callable[[dict], None] = on_emit or (lambda ev: None)
        self._on_nod_motion: Callable[[dict], None] = on_nod_motion or (lambda ev: None)
        self._on_trigger_llm: Callable[..., None] = on_trigger_llm or (lambda text, **_kw: None)
        self._on_request_force_finalize: Callable[[], None] = on_request_force_finalize or (lambda: None)
        self._on_request_tts_done: Callable[[], None] = on_request_tts_done or (lambda: None)
        self._on_request_opening_tts: Callable[[str], None] = on_request_opening_tts or (lambda text: None)
        self._on_request_nonstream_tts: Callable[[str, str], None] = on_request_nonstream_tts or (lambda text, src: None)
        self._on_request_interrupt: Callable[[str], None] = on_request_interrupt or (lambda reason: None)
        self._on_request_scheduled_exit_tts: Callable[[str], None] = on_request_scheduled_exit_tts or (lambda text: None)
        self._on_commit_prefetch: Callable[[], None] = on_commit_prefetch or (lambda: None)
        self._on_cancel_prefetch: Callable[[], None] = on_cancel_prefetch or (lambda: None)

    def _dispatch_then_return(self, events: list[dict]) -> list[dict]:
        for ev in events:
            self._dispatch_event(ev)
        return events

    def _dispatch_event(self, ev: dict) -> None:
        etype = str(ev.get("type", ""))
        if etype == "request_nod_motion":
            self._on_nod_motion(ev)
        elif etype == "trigger_llm":
            self._on_trigger_llm(
                str(ev.get("user_text", "")),
                prefetch=bool(ev.get("prefetch", False)),
            )
        elif etype == "commit_prefetch":
            self._on_commit_prefetch()
        elif etype == "cancel_prefetch":
            self._on_cancel_prefetch()
        elif etype == "request_force_finalize":
            self._on_request_force_finalize()
        elif etype == "request_tts_done":
            self._on_request_tts_done()
        elif etype == "request_opening_tts":
            self._on_request_opening_tts(str(ev.get("assistant_text", "")))
        elif etype == "request_nonstream_tts":
            self._on_request_nonstream_tts(str(ev.get("assistant_text", "")), str(ev.get("source", "")))
        elif etype == "request_interrupt":
            self._on_request_interrupt(str(ev.get("reason", "")))
        elif etype == "request_scheduled_exit_tts":
            self._on_request_scheduled_exit_tts(str(ev.get("assistant_text", "")))
        else:
            self._on_emit(ev)

    def start(self, settings: dict[str, Any]) -> list[dict[str, object]]:
        self.settings = dict(settings or {})
        system_prompt = str(self.settings.get("system_prompt", "")).strip() or "あなたは簡潔で丁寧なアシスタントです。"
        opening = str(self.settings.get("voice_dialog_opening_utterance", "")).strip()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if opening:
            messages.append({"role": "assistant", "content": opening})
        self.state = DialogOrchestratorState(active=True, messages=messages)
        self.state.voice_dialog_start_wall_ts = time.time()
        self.state.scheduled_exit_waiting_for_turn = False
        if opening:
            self.state.pending_assistant_text = opening
            self.state.waiting_tts_audible = True
            self.state.tts_audible_detected = False
            self.state.turn_blocked_until_user_speech = True
            self.state.opening_tts_pending = True
        events: list[dict[str, object]] = [
            {"type": "status", "state": "running", "message": "音声対話：実行中"},
            {"type": "history_snapshot", "history": self.history_snapshot()},
        ]
        if opening:
            events.append({"type": "request_opening_tts", "assistant_text": opening})
        events.extend(
            [
                {"type": "latency", "kind": "asr", "value": None},
                {"type": "latency", "kind": "llm", "value": None},
                {"type": "latency", "kind": "tts", "value": None},
            ]
        )
        self.reset_nod_para_timing_state()
        return self._dispatch_then_return(events)

    def stop(self) -> list[dict[str, object]]:
        self.state = DialogOrchestratorState()
        self.reset_nod_para_timing_state()
        return self._dispatch_then_return([
            {"type": "status", "state": "stopped", "message": "音声対話：停止"},
            {"type": "history_snapshot", "history": self.history_snapshot()},
        ])

    def update_settings(self, settings: dict[str, Any]) -> list[dict[str, object]]:
        self.settings.update(dict(settings or {}))
        sec = _safe_float(self.settings.get("voice_dialog_scheduled_exit_sec", 0), 0.0)
        utterance = str(self.settings.get("voice_dialog_scheduled_exit_utterance", "")).strip()
        if sec <= 0.0 or not utterance:
            self.state.scheduled_exit_waiting_for_turn = False
        return self._dispatch_then_return([])

    def interrupt(self) -> list[dict[str, object]]:
        return self._dispatch_then_return([{"type": "interrupt", "message": "interrupt requested"}])

    def _maai_turn_threshold(self) -> float:
        value = _safe_float(self.settings.get("maai_turn_threshold", 0.5), 0.5)
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _interrupt_vad_threshold(self) -> float:
        value = _safe_float(
            self.settings.get(
                "interrupt_vad_threshold",
                self.settings.get("vad_on_threshold", 0.1),
            ),
            0.1,
        )
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _interrupt_p_now_threshold(self) -> float:
        value = _safe_float(
            self.settings.get(
                "interrupt_p_now_threshold",
                self.settings.get("maai_turn_threshold", 0.5),
            ),
            0.5,
        )
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _maybe_arm_scheduled_exit(self) -> None:
        if not self.state.active or self.state.scheduled_exit_waiting_for_turn:
            return
        sec = _safe_float(self.settings.get("voice_dialog_scheduled_exit_sec", 0), 0.0)
        if sec <= 0.0:
            return
        utterance = str(self.settings.get("voice_dialog_scheduled_exit_utterance", "")).strip()
        if not utterance:
            return
        t0 = self.state.voice_dialog_start_wall_ts
        if t0 is None:
            return
        if time.time() >= float(t0) + float(sec):
            self.state.scheduled_exit_waiting_for_turn = True

    def _interrupt_maai_vad_threshold(self) -> float:
        """MaAI の vad[0] 用。Silero スコア用の interrupt_vad_threshold とスケールが異なるため分離する。"""
        value = _safe_float(
            self.settings.get(
                "interrupt_maai_vad_threshold",
                self.settings.get("vad_on_threshold", 0.1),
            ),
            0.1,
        )
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _context_len(self) -> int:
        try:
            v = int(str(self.settings.get("llm_context_len", "10")).strip() or "10")
        except Exception:
            v = 10
        return max(1, v)

    def _turn_backend(self) -> str:
        return str(self.settings.get("turn_backend", "maai")).strip().lower() or "maai"

    def _asr_nonstream_pseudo_enabled(self) -> bool:
        """非ストリーミング ASR でも疑似ストリーミングを有効にしているか（controller_config の Pseudo streaming）。"""
        v = self.settings.get("asr_nonstream_pseudo_enabled", True)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return bool(v)

    def _asr_effective_streaming_for_maai_turn(self) -> bool:
        """ターン検出で MaAI ストリーミング分岐に入るか。最終 is_final が is_streaming=False でも疑似なら真。"""
        return bool(self.state.last_is_streaming_asr) or self._asr_nonstream_pseudo_enabled()

    def _backchannel_enabled(self) -> bool:
        return bool(self.settings.get("maai_backchannel_enabled", False))

    def _backchannel_threshold(self) -> float:
        value = _safe_float(self.settings.get("maai_backchannel_prob_threshold", 0.5), 0.5)
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _backchannel_min_interval_sec(self) -> float:
        value = _safe_float(self.settings.get("maai_backchannel_min_interval_sec", 1.0), 1.0)
        return max(0.0, value)

    def _backchannel_required_consecutive_frames(self) -> int:
        try:
            value = int(str(self.settings.get("maai_backchannel_consecutive_frames", 1)).strip() or "1")
        except Exception:
            value = 1
        return max(1, value)

    def _main_assistant_tts_busy(self) -> bool:
        """メイン応答の TTS が可聴送出を始めたら真（tts_done で tts_audible_detected が落ちるまで）。

        以前は waiting_tts_audible 必須としていたが、ストリーミング TTS では llm_finished より先に
        可聴チャンクが届き得る。その間 waiting が False のままだと相槌 request_nonstream_tts が
        メインと同一音声出力キューへ混ざり、PCM がフレーム単位で交互になりぶつぶつ聞こえる。
        llm_finished 後〜最初の可聴前の短い隙間では tts_audible_detected が False のため相槌可。
        """
        return bool(self.state.tts_audible_detected)

    def _select_backchannel_text(self) -> str:
        dictionary_path = str(self.settings.get("maai_backchannel_dictionary_path", "")).strip()
        if dictionary_path:
            try:
                data = json.loads(Path(dictionary_path).read_text(encoding="utf-8"))
                candidates: list[tuple[str, float]] = []
                entries = data.get("entries") if isinstance(data, dict) else None
                if not isinstance(entries, list):
                    entries = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    token = str(entry.get("text", "")).strip()
                    weight = _safe_float(entry.get("weight", 1.0), 1.0)
                    if token:
                        candidates.append((token, max(0.0, weight)))
                if candidates:
                    texts = [item[0] for item in candidates]
                    weights = [item[1] for item in candidates]
                    if any(w > 0.0 for w in weights):
                        return random.choices(texts, weights=weights, k=1)[0]
                    return random.choice(texts)
            except Exception:
                pass
        return "はい。"

    def _collect_user_utterances(self, *, include_partial: bool) -> list[tuple[int | None, str]]:
        items: list[tuple[int | None, str]] = []
        for i, t in enumerate(self.state.pending_texts):
            text = str(t).strip()
            if text:
                idx = self.state.pending_utterance_indices[i] if i < len(self.state.pending_utterance_indices) else None
                items.append((idx, text))
        if include_partial and self.state.current_partial:
            items.append((self.state.current_partial_utterance_index, self.state.current_partial))
        return items

    def _pop_user_utterances(self, *, include_partial: bool) -> list[tuple[int | None, str]]:
        items = self._collect_user_utterances(include_partial=include_partial)
        if not items:
            return []
        self.state.pending_texts.clear()
        self.state.pending_utterance_indices.clear()
        if include_partial:
            self.state.current_partial = ""
            self.state.current_partial_utterance_index = None
        return items

    def _trim_messages_keep_system(self) -> bool:
        """system 以外を llm_context_len で切り詰め。トリム時は active/streaming 対応をリセットする。"""
        system_msg = self.state.messages[0] if self.state.messages else {"role": "system", "content": ""}
        before_len = len(self.state.messages)
        non_system = [m for m in self.state.messages[1:] if m.get("role") in {"user", "assistant"}]
        context_len = self._context_len()
        if len(non_system) > context_len:
            non_system = non_system[-context_len:]
        self.state.messages = [system_msg] + non_system
        trimmed = len(self.state.messages) != before_len
        if trimmed:
            self.state.active_user_msg_index = None
            self.state.active_user_msg_utterance_order.clear()
            self.state.active_user_msg_utterance_text.clear()
            self.state.active_user_msg_await_final_replace = False
            self.state.streaming_user_row_by_utterance.clear()
        return trimmed

    def _rebuild_streaming_user_row_map(self) -> None:
        self.state.streaming_user_row_by_utterance.clear()
        for i, m in enumerate(self.state.messages):
            if m.get("role") != "user":
                continue
            u = m.get("_stream_utt")
            if u is None:
                continue
            try:
                self.state.streaming_user_row_by_utterance[int(u)] = int(i)
            except (TypeError, ValueError):
                pass

    def _prepare_tts_split_before_streaming_user_append(
        self,
    ) -> tuple[dict[str, object] | None, bool]:
        """TTS 再生中に新しい user 行が割り込む直前に、直前 assistant を確定して pending を残りだけにする。"""
        st = self.state
        pending = (st.pending_assistant_text or "").strip()
        if not pending or not st.waiting_tts_audible:
            return (None, False)
        msgs = st.messages
        if not msgs or msgs[-1].get("role") != "assistant":
            return (None, False)

        tail_played = str(msgs[-1].get("content", ""))
        if not tail_played.strip():
            if len(msgs) >= 2 and msgs[-2].get("role") == "user":
                msgs.pop()
                return (None, bool(pending.strip()))
            return (None, False)

        if pending.startswith(tail_played):
            suffix = pending[len(tail_played) :]
        else:
            tps = tail_played.strip()
            suffix = pending[len(tps) :] if tps and pending.startswith(tps) else pending

        prev_strip = st.tts_cumulative_strip_prefix or ""
        new_full_strip = prev_strip + tail_played
        st.tts_cumulative_strip_prefix = new_full_strip
        msgs[-1] = {**msgs[-1], "content": tail_played}
        st.pending_assistant_text = suffix if suffix.strip() else None

        rebase: dict[str, object] = {"type": "tts_cumulative_strip_prefix", "prefix": new_full_strip}
        append_assistant = bool(suffix.strip())
        return (rebase, append_assistant)

    def _apply_streaming_asr_to_history(self, text: str, utterance_index: int | None) -> dict[str, object] | None:
        """ストリーミング ASR: utterance_index ごとに 1 行の user（partial/final を同一行で更新）。

        TTS 待ち中に新規 user 行を追加するときは直前 assistant を分割し、controller 用の strip イベントを返す。
        """
        if utterance_index is None:
            self.state.streaming_anonymous_seq -= 1
            u = int(self.state.streaming_anonymous_seq)
        else:
            u = int(utterance_index)
        self._rebuild_streaming_user_row_map()
        row = self.state.streaming_user_row_by_utterance.get(u)
        new_msg: dict[str, Any] = {"role": "user", "content": str(text), "_stream_utt": u}
        rebase_evt: dict[str, object] | None = None
        if row is not None and 0 <= int(row) < len(self.state.messages) and self.state.messages[int(row)].get("role") == "user":
            self.state.messages[int(row)] = new_msg
        else:
            rebase_evt, append_assistant = self._prepare_tts_split_before_streaming_user_append()
            self.state.messages.append(new_msg)
            if append_assistant:
                self.state.messages.append({"role": "assistant", "content": ""})
        self._trim_messages_keep_system()
        self._rebuild_streaming_user_row_map()
        return rebase_evt

    def _emit_asr_latency_if_waiting(
        self,
        events: list[dict],
        *,
        end_wall_ts: float | None = None,
    ) -> None:
        if not (self.state.waiting_asr_latency and self.state.last_turn_ts is not None):
            return
        now_perf = time.perf_counter()
        now_wall = end_wall_ts if end_wall_ts is not None else time.time()
        value: float | None = max(0.0, now_perf - self.state.last_turn_ts)
        # 保険: 万一フラグがリークして古い last_turn_ts と差を取った場合、非現実的に
        # 大きい値になる。誤った数値を出すより None（"—" 表示）にする。
        if value is not None and value > 30.0:
            value = None
        events.append({
            "type": "latency",
            "kind": "asr",
            "value": value,
            "start_ts": self.state.last_turn_wall_ts,
            "end_ts": now_wall,
        })
        self.state.waiting_asr_latency = False

    def _purge_empty_assistant_messages(self) -> int:
        """TTS 分割用の content="" assistant 行を履歴から除去する。

        末尾や途中に空 assistant が残ると LM Studio/Gemma が即 stop（空応答）しやすい。
        """
        msgs = self.state.messages
        if not msgs:
            return 0
        kept: list[dict[str, Any]] = []
        removed = 0
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            content = str(m.get("content", ""))
            # system は空でも残す。空 assistant のみ除去（空 user は通常発生しないが同様に落とす）。
            if i > 0 and role in {"assistant", "user"} and not content.strip():
                removed += 1
                continue
            kept.append(m)
        if removed:
            self.state.messages = kept
            self._rebuild_streaming_user_row_map()
        return removed

    def _trigger_llm_from_history_only(self, *, trigger_path: str) -> list[dict[str, object]]:
        """messages は既に確定している前提で LLM のみ起動（ストリーミング ASR 用・バッファは消さない）。"""
        if self.state.opening_tts_pending:
            return []
        self._purge_empty_assistant_messages()
        self.state.llm_in_progress = True
        self.state.active_user_msg_await_final_replace = False
        path_text = str(trigger_path or "unknown").strip() or "unknown"
        return [
            {"type": "history_snapshot", "history": self.history_snapshot()},
            {
                "type": "log",
                "message": (
                    "[LLM][START] "
                    f"path={path_text} "
                    "user_text_len=0 "
                    f"final_seq={int(self.state.final_seq)} "
                    "mode=history_snapshot"
                ),
            },
            {"type": "trigger_llm", "user_text": ""},
        ]

    def _last_user_text(self) -> str:
        for m in reversed(self.state.messages):
            if not isinstance(m, dict):
                continue
            if str(m.get("role", "")) == "user":
                return str(m.get("content", "")).strip()
        return ""

    def clear_prefetch(self) -> None:
        """runtime の割り込み等から先読みフラグだけ落とす。"""
        self.state.prefetch_active = False
        self.state.prefetch_failed = False

    def mark_prefetch_failed(self) -> None:
        """先読み LLM/TTS が失敗した。p_now では通常のターン取得に戻す。"""
        if self.state.prefetch_active:
            self.state.prefetch_failed = True

    def _cancel_prefetch_events(self, *, reason: str) -> list[dict[str, object]]:
        if not self.state.prefetch_active:
            return []
        self.state.prefetch_active = False
        self.state.prefetch_failed = False
        reason_text = str(reason or "unknown").strip() or "unknown"
        return [
            {
                "type": "log",
                "message": f"[LLM][PREFETCH] cancel reason={reason_text}",
            },
            {"type": "cancel_prefetch"},
        ]

    def _start_prefetch_from_history(self, *, trigger_path: str) -> list[dict[str, object]]:
        """p_future 立ち上がりで LLM を先読みする。ターンは取らない（llm_in_progress を立てない）。"""
        if self.state.opening_tts_pending or self.state.prefetch_active:
            return []
        if self.state.llm_in_progress or self.state.turn_blocked_until_user_speech:
            return []
        if not self._last_user_text():
            return []
        self._purge_empty_assistant_messages()
        self.state.prefetch_active = True
        self.state.prefetch_failed = False
        path_text = str(trigger_path or "unknown").strip() or "unknown"
        return [
            {"type": "history_snapshot", "history": self.history_snapshot()},
            {
                "type": "log",
                "message": (
                    "[LLM][PREFETCH] "
                    f"path={path_text} "
                    f"final_seq={int(self.state.final_seq)} "
                    "mode=history_snapshot"
                ),
            },
            {"type": "trigger_llm", "user_text": "", "prefetch": True},
        ]

    def _commit_prefetch_turn(
        self, p_now: float, log_prefix: str, *, branch_tag: str
    ) -> list[dict[str, object]]:
        """先読みを確定し、保持 PCM の再生へ進む（新規 LLM は起動しない）。"""
        self.state.prefetch_active = False
        self.state.prefetch_failed = False
        self.state.turn_blocked_until_user_speech = True
        self.state.llm_in_progress = True
        self.state.last_turn_ts = time.perf_counter()
        self.state.last_turn_wall_ts = time.time()
        self.state.waiting_asr_latency = True
        events: list[dict[str, object]] = [
            {"type": "log", "message": f"{log_prefix} branch={branch_tag}"},
            {"type": "turn", "probability": p_now, "ts": self.state.last_turn_wall_ts},
            {"type": "history_snapshot", "history": self.history_snapshot()},
        ]
        self._emit_asr_latency_if_waiting(events)
        events.append({"type": "commit_prefetch"})
        return self._dispatch_then_return(events)

    def _record_user_message(
        self,
        utterances: list[tuple[int | None, str]],
        *,
        trigger_path: str = "",
        await_final_replace: bool = False,
    ) -> list[dict[str, object]]:
        user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
        if not user_text:
            if self.state.opening_tts_pending:
                return []
            # 空テキストでも既存履歴のみでLLMを実行する。
            self.state.active_user_msg_await_final_replace = False
            self.state.streaming_user_row_by_utterance.clear()
            self.state.llm_in_progress = True
            path_text_empty = str(trigger_path or "unknown").strip() or "unknown"
            return [
                {"type": "history_snapshot", "history": self.history_snapshot()},
                {
                    "type": "log",
                    "message": (
                        "[LLM][START] "
                        f"path={path_text_empty} "
                        "user_text_len=0 "
                        f"final_seq={int(self.state.final_seq)} "
                        "mode=history_only"
                    ),
                },
                {"type": "trigger_llm", "user_text": ""},
            ]
        uniq_idx = sorted({int(idx) for idx, _t in utterances if idx is not None})
        coalesce_key: int | None = uniq_idx[0] if len(uniq_idx) == 1 else None
        reuse_msg_idx: int | None = None
        if coalesce_key is not None:
            mapped = self.state.streaming_user_row_by_utterance.get(int(coalesce_key))
            if mapped is not None:
                mi = int(mapped)
                if 0 <= mi < len(self.state.messages) and self.state.messages[mi].get("role") == "user":
                    reuse_msg_idx = mi
        # 中間 final 後に utterance_index が変わっても、直前が user なら同一認識行に上書きする。
        if reuse_msg_idx is None and await_final_replace and self.state.messages:
            li = len(self.state.messages) - 1
            if self.state.messages[li].get("role") == "user":
                reuse_msg_idx = li
        if reuse_msg_idx is not None:
            self.state.messages[int(reuse_msg_idx)] = {"role": "user", "content": user_text}
        else:
            self.state.messages.append({"role": "user", "content": user_text})
        trimmed = self._trim_messages_keep_system()
        self._rebuild_streaming_user_row_map()
        if self.state.messages:
            order: list[int] = []
            text_map: dict[int, str] = {}
            for idx, utext in utterances:
                if idx is None:
                    continue
                i = int(idx)
                if i not in text_map:
                    order.append(i)
                text_map[i] = str(utext).strip()
            if (not trimmed) and reuse_msg_idx is not None:
                msg_idx = int(reuse_msg_idx)
            else:
                msg_idx = len(self.state.messages) - 1
            self.state.active_user_msg_index = msg_idx
            self.state.active_user_msg_utterance_order = order
            self.state.active_user_msg_utterance_text = text_map
            self.state.active_user_msg_await_final_replace = bool(await_final_replace)
            if coalesce_key is not None and (not trimmed):
                self.state.streaming_user_row_by_utterance[int(coalesce_key)] = int(msg_idx)
        else:
            self.state.active_user_msg_await_final_replace = False
        path_text = str(trigger_path or "unknown").strip() or "unknown"
        return [
            {"type": "history_snapshot", "history": self.history_snapshot()},
            {
                "type": "log",
                "message": (
                    "[LLM][START] "
                    f"path={path_text} "
                    f"user_text_len={len(user_text)} "
                    f"final_seq={int(self.state.final_seq)}"
                ),
            },
            {"type": "trigger_llm", "user_text": user_text},
        ]

    def on_maai_result(self, result: dict[str, Any]) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        if not self.state.active:
            return self._dispatch_then_return(events)
        silero_vad_obj = result.get("silero_vad_score")
        vad_obj = result.get("vad")
        pnow_obj = result.get("p_now")
        try:
            p_now = float(pnow_obj[1] if isinstance(pnow_obj, (list, tuple)) else pnow_obj)
        except Exception:
            return self._dispatch_then_return(events)
        # 割り込み・ターンロック解除: silero_vad_score を優先（controller が ASR Silero をマージ）
        try:
            if silero_vad_obj is not None:
                vad_for_interrupt = float(
                    silero_vad_obj[0] if isinstance(silero_vad_obj, (list, tuple)) else silero_vad_obj
                )
                vad_thr = float(self._interrupt_vad_threshold())
            else:
                vad_for_interrupt = float(vad_obj[0] if isinstance(vad_obj, (list, tuple)) else vad_obj)
                vad_thr = float(self._interrupt_maai_vad_threshold())
        except Exception:
            return self._dispatch_then_return(events)
        self.state.vad_active = vad_for_interrupt >= vad_thr
        user_side = self.state.vad_active and p_now < self._interrupt_p_now_threshold()
        if user_side and (not self.state.opening_tts_pending):
            self.state.last_user_speech_detect_wall_ts = time.time()
        # 開始直後は p_now が未立ち上がりで 0 に近いことが多く、VAD だけで user_side が真になり
        # ターンロックが誤解除される。開始時システム発話の再生中は user_side では解除しない
        # （解除は on_tts_done / 割り込み中止のみ）。request_interrupt は下でそのまま出す。
        if (
            self.state.turn_blocked_until_user_speech
            and user_side
            and (not self.state.opening_tts_pending)
        ):
            self.state.turn_blocked_until_user_speech = False
        interrupt_wanted = (
            bool(self.settings.get("interrupt_on_user_speech", True))
            and self.state.vad_active
            and p_now < self._interrupt_p_now_threshold()
            and (not self.state.opening_tts_pending)
        )
        was_interrupt = bool(getattr(self.state, "_interrupt_condition_active", False))
        self.state._interrupt_condition_active = interrupt_wanted
        if interrupt_wanted:
            # 実際の interrupt 実行は controller / dialog/runtime 側。
            # オープニング発話は最後まで再生する方針のため、再生中は要求しない。
            # interrupt_condition はログ用で毎フレーム不要。立ち上がりだけ出す。
            if not was_interrupt:
                events.append(
                    {"type": "interrupt_condition", "vad": vad_for_interrupt, "p_now": p_now}
                )
            # LLM 完了後の TTS のみの区間では llm_in_progress も turn_blocked も false になり得る。
            # 再生有無の判定は _voice_dialog_interrupt_system_speech 側に任せる。
            events.append(
                {
                    "type": "request_interrupt",
                    "reason": "user_speech_detected",
                    "vad": vad_for_interrupt,
                    "p_now": p_now,
                    "llm_in_progress": bool(self.state.llm_in_progress),
                    "turn_blocked_until_user_speech": bool(self.state.turn_blocked_until_user_speech),
                }
            )
        return self._dispatch_then_return(events)

    # ---- nod_para settings helpers ----

    def _nod_para_enabled(self) -> bool:
        return bool(self.settings.get("nod_para_enabled", False))

    def _nod_para_prob_threshold(self) -> float:
        v = _safe_float(self.settings.get("nod_para_prob_threshold", 0.6), 0.6)
        return max(0.0, min(1.0, v))

    def _nod_para_min_consecutive_frames(self) -> int:
        try:
            v = int(str(self.settings.get("nod_para_min_consecutive_frames", 3)).strip() or "3")
        except Exception:
            v = 3
        return max(1, v)

    def _nod_para_min_interval_ms(self) -> float:
        return max(0.0, _safe_float(self.settings.get("nod_para_min_interval_ms", 1500), 1500))

    def _nod_para_swing_up_threshold(self) -> float | None:
        """振り上げのしきい値。None（設定が空欄）ならモデルの判定（nod_swing_up_pred）を使う。"""
        v = self.settings.get("nod_para_swing_up_threshold")
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return max(0.0, min(1.0, _safe_float(v, 0.5)))

    def _nod_para_count_thresholds(self) -> dict[str, float] | None:
        """頷き回数のしきい値 t0〜t2。None（どれかが空欄）ならモデルの判定（nod_repetitions_pred）を使う。"""
        th = self.settings.get("nod_para_count_thresholds")
        if not isinstance(th, dict):
            return None
        try:
            return {k: float(th[k]) for k in ("t0", "t1", "t2")}
        except (KeyError, TypeError, ValueError):
            return None

    def _clear_nod_para_buffer(self) -> None:
        s = self.state
        s.nod_para_consecutive_count = 0
        s.nod_para_buf_p_nod.clear()
        s.nod_para_buf_range.clear()
        s.nod_para_buf_speed.clear()
        s.nod_para_buf_swing_up.clear()
        s.nod_para_buf_swing_up_decided.clear()
        s.nod_para_buf_raw_range.clear()
        s.nod_para_buf_raw_speed.clear()
        s.nod_para_buf_ts_start.clear()
        s.nod_para_buf_ts_end.clear()
        s.nod_para_buf_count_idx.clear()

    def reset_nod_para_timing_state(self) -> None:
        self._clear_nod_para_buffer()
        s = self.state
        s.nod_para_prob_dropped = True
        s.last_nod_para_ts = 0.0

    _NOD_FRAME_RANGE_BOUNDS: tuple[float, float] = (0.0, 1.0)
    _NOD_FRAME_SPEED_BOUNDS: tuple[float, float] = (0.0, 1.0)

    def _nod_frame_range_bounds(self) -> tuple[float, float]:
        """range_rad クリップ境界（設定で上書き可、既定は物理上限）。"""
        d_lo, d_hi = self._NOD_FRAME_RANGE_BOUNDS
        lo = max(0.0, _safe_float(self.settings.get("nod_para_range_clip_min", d_lo), d_lo))
        hi = max(1e-6, _safe_float(self.settings.get("nod_para_range_clip_max", d_hi), d_hi))
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)

    def _nod_frame_speed_bounds(self) -> tuple[float, float]:
        """speed クリップ境界（設定で上書き可、既定は物理上限）。"""
        d_lo, d_hi = self._NOD_FRAME_SPEED_BOUNDS
        lo = max(0.0, _safe_float(self.settings.get("nod_para_speed_clip_min", d_lo), d_lo))
        hi = max(1e-6, _safe_float(self.settings.get("nod_para_speed_clip_max", d_hi), d_hi))
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)

    def _clamp_nod_frame_range(self, raw: float) -> float:
        lo, hi = self._nod_frame_range_bounds()
        return max(lo, min(hi, abs(raw)))

    def _clamp_nod_frame_speed(self, raw: float) -> float:
        lo, hi = self._nod_frame_speed_bounds()
        return max(lo, min(hi, abs(raw)))

    def _parse_nod_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        # 回数（0=1回, 1=2回, 2=3回以上）と振り上げは、モデルと同じ規則で決める（utils.nod_para_frame_decision）。
        # 以前は maai 0.2.27 の nod_repetitions を読まず常に 1 回で、しきい値も掛け算（モデルは割り算）だった。
        frame_count_idx, frame_swing_up, frame_swing = nod_para_frame_decision(
            payload, self._nod_para_count_thresholds(), self._nod_para_swing_up_threshold(),
        )

        raw_range = _safe_float(payload.get("nod_range", 0.25), 0.25)
        raw_speed = _safe_float(payload.get("nod_speed", 0.7), 0.7)
        frame_range = self._clamp_nod_frame_range(raw_range)
        frame_speed = self._clamp_nod_frame_speed(raw_speed)
        frame_ts_start = _safe_float(payload.get("source_ts_start"), time.time())
        frame_ts_end = _safe_float(payload.get("source_ts_end"), frame_ts_start)
        if frame_ts_end < frame_ts_start:
            frame_ts_end = frame_ts_start
        p_nod = _safe_float(payload.get("p_nod"), -1.0)
        return {
            "frame_count_idx": frame_count_idx,
            "frame_range": frame_range,
            "frame_speed": frame_speed,
            "raw_range": raw_range,
            "raw_speed": raw_speed,
            "frame_swing": frame_swing,
            "frame_swing_up": frame_swing_up,
            "frame_ts_start": frame_ts_start,
            "frame_ts_end": frame_ts_end,
            "p_nod": p_nod,
        }

    def _append_nod_motion_events(
        self,
        *,
        nod_count: int,
        avg_range: float,
        avg_speed: float,
        use_pre_rise: bool,
        buffer_frames: int,
        chart_p_nod: float,
        chart_raw_range: float,
        chart_raw_speed: float,
        chart_swing_up_prob: float,
        used_start_ts: float,
        used_end_ts: float,
        trigger_ts: float,
        timeline_extra: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        extra = dict(timeline_extra or {})
        vis: dict[str, object] = {
            "type": "nod_trigger_visual",
            "used_start_ts": float(used_start_ts),
            "used_end_ts": float(used_end_ts),
            "trigger_ts": float(trigger_ts),
            "buffer_frames": int(buffer_frames),
            "count": int(nod_count),
        }
        vis.update(extra)
        motion: dict[str, object] = {
            "type": "request_nod_motion",
            "range_rad": float(avg_range),
            "count": int(nod_count),
            "use_pre_rise": bool(use_pre_rise),
            "velocity": float(avg_speed),
            "fps": int(_safe_float(self.settings.get("nod_para_fps", 30), 30)),
            "decay_rate": float(_safe_float(self.settings.get("nod_para_decay_rate", 0.8), 0.8)),
            "interpolation": str(self.settings.get("nod_para_interpolation", "cosine")),
            "pre_rise_ratio": float(_safe_float(self.settings.get("nod_para_pre_rise_ratio", 0.8), 0.8)),
            "scale": float(_safe_float(self.settings.get("nod_para_scale", 3.0), 3.0)),
            "speed_scale": float(_safe_float(self.settings.get("nod_para_speed_scale", 1.2), 1.2)),
            "eye_scale": float(_safe_float(self.settings.get("nod_para_eye_scale", 1.0), 1.0)),
            "bone": str(self.settings.get("nod_para_bone", "頭")),
            "buffer_frames": int(buffer_frames),
            "chart_p_nod": float(chart_p_nod),
            "chart_raw_range": float(chart_raw_range),
            "chart_raw_speed": float(chart_raw_speed),
            "chart_swing_up_prob": float(chart_swing_up_prob),
        }
        motion.update(extra)
        return [vis, motion]

    def _resolve_series_params_for_fire(
        self,
        *,
        use_range: list[float],
        use_speed: list[float],
        use_swing: list[float],
        use_raw_range: list[float],
        use_raw_speed: list[float],
        use_count_idx: list[int],
        use_p_nod: list[float],
        use_swing_up: list[int] | None = None,
    ) -> tuple[int, float, float, bool, float, float, float, float]:
        avg_range = sum(use_range) / len(use_range)
        avg_speed = sum(use_speed) / len(use_speed)
        count_hist = [0, 0, 0]
        for ci in use_count_idx:
            if 0 <= ci < 3:
                count_hist[ci] += 1
        mode_idx = 0
        for i in range(1, 3):
            if count_hist[i] > count_hist[mode_idx]:
                mode_idx = i
        nod_count = mode_idx + 1
        # 振り上げはフレームごとに決めた 0/1（nod_para_frame_decision）の多数決。
        swing_binary = list(use_swing_up) if use_swing_up else [0] * len(use_swing)
        swing_ones = sum(swing_binary)
        swing_zeros = len(swing_binary) - swing_ones
        use_pre_rise = swing_ones > swing_zeros
        chart_p_nod = sum(use_p_nod) / len(use_p_nod)
        chart_raw_range = sum(use_raw_range) / len(use_raw_range)
        chart_raw_speed = sum(use_raw_speed) / len(use_raw_speed)
        chart_swing_up_prob = sum(use_swing) / len(use_swing)
        return (nod_count, avg_range, avg_speed, use_pre_rise, chart_p_nod, chart_raw_range, chart_raw_speed, chart_swing_up_prob)

    def on_maai_nod_result(self, result: dict[str, Any]) -> list[dict[str, object]]:
        payload = dict(result)
        payload.setdefault("maai_mode", "nod_para")
        events: list[dict[str, object]] = []
        events.extend(self.on_maai_result(payload))

        if not self._nod_para_enabled():
            return self._dispatch_then_return(events)

        p_nod = _safe_float(payload.get("p_nod"), -1.0)
        if p_nod < 0:
            return self._dispatch_then_return(events)

        # --- timing model ---
        if p_nod < 0.5:
            self.state.nod_para_prob_dropped = True

        threshold = self._nod_para_prob_threshold()
        if p_nod < threshold:
            self._clear_nod_para_buffer()
            return self._dispatch_then_return(events)

        fr = self._parse_nod_frame(payload)
        frame_count_idx = int(fr["frame_count_idx"])
        frame_range = float(fr["frame_range"])
        frame_speed = float(fr["frame_speed"])
        raw_range = float(fr["raw_range"])
        raw_speed = float(fr["raw_speed"])
        frame_ts_start = float(fr["frame_ts_start"])
        frame_ts_end = float(fr["frame_ts_end"])
        frame_swing = float(fr["frame_swing"])
        frame_swing_up = int(fr["frame_swing_up"])

        s = self.state
        s.nod_para_consecutive_count += 1
        s.nod_para_buf_p_nod.append(p_nod)
        s.nod_para_buf_range.append(frame_range)
        s.nod_para_buf_speed.append(frame_speed)
        s.nod_para_buf_swing_up.append(frame_swing)
        s.nod_para_buf_swing_up_decided.append(frame_swing_up)
        s.nod_para_buf_raw_range.append(raw_range)
        s.nod_para_buf_raw_speed.append(raw_speed)
        s.nod_para_buf_ts_start.append(frame_ts_start)
        s.nod_para_buf_ts_end.append(frame_ts_end)
        s.nod_para_buf_count_idx.append(frame_count_idx)

        now_ms = time.time() * 1000.0

        if (now_ms - s.last_nod_para_ts * 1000.0) < self._nod_para_min_interval_ms():
            return self._dispatch_then_return(events)

        if not s.nod_para_prob_dropped:
            self._clear_nod_para_buffer()
            return self._dispatch_then_return(events)

        req_frames = self._nod_para_min_consecutive_frames()
        if s.nod_para_consecutive_count < req_frames:
            return self._dispatch_then_return(events)

        use_range = s.nod_para_buf_range[-req_frames:]
        use_speed = s.nod_para_buf_speed[-req_frames:]
        use_swing = s.nod_para_buf_swing_up[-req_frames:]
        use_raw_range = s.nod_para_buf_raw_range[-req_frames:]
        use_raw_speed = s.nod_para_buf_raw_speed[-req_frames:]
        use_ts_start = s.nod_para_buf_ts_start[-req_frames:]
        use_ts_end = s.nod_para_buf_ts_end[-req_frames:]
        use_count_idx = s.nod_para_buf_count_idx[-req_frames:]
        use_p_nod = s.nod_para_buf_p_nod[-req_frames:]
        use_swing_up = s.nod_para_buf_swing_up_decided[-req_frames:]

        (
            nod_count,
            avg_range,
            avg_speed,
            use_pre_rise,
            chart_p_nod,
            chart_raw_range,
            chart_raw_speed,
            chart_swing_up_prob,
        ) = self._resolve_series_params_for_fire(
            use_range=use_range,
            use_speed=use_speed,
            use_swing=use_swing,
            use_raw_range=use_raw_range,
            use_raw_speed=use_raw_speed,
            use_count_idx=use_count_idx,
            use_p_nod=use_p_nod,
            use_swing_up=use_swing_up,
        )

        used_start_ts = float(use_ts_end[0]) if use_ts_end else time.time()
        used_end_ts = float(use_ts_end[-1]) if use_ts_end else used_start_ts
        trigger_ts = used_end_ts
        buffer_frames = req_frames

        self._clear_nod_para_buffer()
        s.last_nod_para_ts = time.time()
        s.nod_para_prob_dropped = False

        events.extend(
            self._append_nod_motion_events(
                nod_count=nod_count,
                avg_range=avg_range,
                avg_speed=avg_speed,
                use_pre_rise=use_pre_rise,
                buffer_frames=buffer_frames,
                chart_p_nod=chart_p_nod,
                chart_raw_range=chart_raw_range,
                chart_raw_speed=chart_raw_speed,
                chart_swing_up_prob=chart_swing_up_prob,
                used_start_ts=used_start_ts,
                used_end_ts=used_end_ts,
                trigger_ts=trigger_ts,
                timeline_extra=_maai_audio_timeline_extra(payload),
            )
        )

        return self._dispatch_then_return(events)

    def on_maai_bc_result(self, result: dict[str, Any]) -> list[dict[str, object]]:
        payload = dict(result)
        payload.setdefault("maai_mode", "bc")
        events: list[dict[str, object]] = []
        if self._backchannel_enabled():
            # maai.model の bc 出力契約に合わせて p_bc を利用する。
            prob_obj = payload.get("p_bc")
            prob = _safe_float(prob_obj, -1.0)
            threshold = self._backchannel_threshold()
            now = time.time()
            elapsed = now - float(self.state.last_backchannel_ts or 0.0)
            frame_ts_start = _safe_float(payload.get("source_ts_start"), now)
            frame_ts_end = _safe_float(payload.get("source_ts_end"), frame_ts_start)
            if frame_ts_end < frame_ts_start:
                frame_ts_end = frame_ts_start
            if prob >= threshold:
                self.state.backchannel_consecutive_over_threshold += 1
                self.state.backchannel_buf_ts_start.append(frame_ts_start)
                self.state.backchannel_buf_ts_end.append(frame_ts_end)
            else:
                self.state.backchannel_consecutive_over_threshold = 0
                self.state.backchannel_buf_ts_start.clear()
                self.state.backchannel_buf_ts_end.clear()
            required_frames = self._backchannel_required_consecutive_frames()
            if (
                self.state.backchannel_consecutive_over_threshold >= required_frames
                and elapsed >= self._backchannel_min_interval_sec()
                and (not self._main_assistant_tts_busy())
            ):
                assistant_text = self._select_backchannel_text().strip()
                if assistant_text:
                    used_ts_start = self.state.backchannel_buf_ts_start[-required_frames:]
                    used_ts_end = self.state.backchannel_buf_ts_end[-required_frames:]
                    used_start_ts = float(used_ts_start[0]) if used_ts_start else now
                    used_end_ts = float(used_ts_end[-1]) if used_ts_end else used_start_ts
                    self.state.last_backchannel_ts = now
                    self.state.backchannel_consecutive_over_threshold = 0
                    self.state.backchannel_buf_ts_start.clear()
                    self.state.backchannel_buf_ts_end.clear()
                    _tl = _maai_audio_timeline_extra(payload)
                    bc_vis: dict[str, object] = {
                        "type": "bc_trigger_visual",
                        "used_start_ts": float(used_start_ts),
                        "used_end_ts": float(used_end_ts),
                        "trigger_ts": float(used_end_ts),
                        "buffer_frames": int(required_frames),
                    }
                    bc_vis.update(_tl)
                    events.append(bc_vis)
                    bc_tts: dict[str, object] = {
                        "type": "request_nonstream_tts",
                        "assistant_text": assistant_text,
                        "source": "maai_backchannel",
                    }
                    bc_tts.update(_tl)
                    events.append(bc_tts)
        events.extend(self.on_maai_result(payload))
        return self._dispatch_then_return(events)

    def _run_turn_acquire_flow(
        self, p_now: float, log_prefix: str, *, branch_tag: str,
        arm_asr_latency: bool = True
    ) -> list[dict[str, object]]:
        """MaAI の crossed_up またはアイドル沈黙経過後のターン取得の共通処理。

        arm_asr_latency: ユーザー発話駆動のターン（cross_up）でのみ ASR レイテンシ計測を
        武装する。アイドル沈黙のプロアクティブ発話では対応する ASR final が来ないため、
        ここで True にすると waiting_asr_latency が残り、次の実発話の final が古い
        last_turn_ts との差で巨大なレイテンシを表示してしまう（=表示バグ）。
        """
        self.state.turn_blocked_until_user_speech = True
        self.state.last_turn_ts = time.perf_counter()
        self.state.last_turn_wall_ts = time.time()
        events: list[dict[str, object]] = [
            {"type": "log", "message": f"{log_prefix} branch={branch_tag}"},
            {"type": "turn", "probability": p_now, "ts": self.state.last_turn_wall_ts},
            {"type": "history_snapshot", "history": self.history_snapshot()},
        ]
        # アイドルターンは武装しない。さらに、過去に武装したまま実 ASR が来ていない
        # リークもここでクリアする（古い last_turn_ts での誤計測を防ぐ）。
        self.state.waiting_asr_latency = arm_asr_latency
        turn_backend = self._turn_backend()
        is_streaming_asr = bool(self.state.last_is_streaming_asr)
        maai_streaming_turn = self._asr_effective_streaming_for_maai_turn()
        if (not maai_streaming_turn) and turn_backend == "maai":
            self.state.wait_nonstream_maai_turn = {
                "baseline_final_seq": int(self.state.final_seq),
                "awaiting_force_finalize_status": True,
                "force_finalize_is_speaking": None,
                "force_finalize_request_id": 0,
                "awaiting_inference_state": False,
                "inference_state_deadline_ts": None,
                "saw_busy_true": False,
                "post_busy_result_deadline_ts": None,
            }
            events.append({"type": "log", "message": f"{log_prefix} branch=maai_nonstream_force_finalize_wait"})
            events.append({"type": "request_force_finalize"})
            return self._dispatch_then_return(events)
        if maai_streaming_turn and turn_backend == "maai":
            events.append({"type": "log", "message": f"{log_prefix} branch=maai_streaming_history_llm"})
            self._emit_asr_latency_if_waiting(events)
            events.extend(
                self._trigger_llm_from_history_only(trigger_path="maai_streaming_turn")
            )
            return self._dispatch_then_return(events)
        if (not is_streaming_asr) and turn_backend == "smart-turn":
            if self.state.pending_texts:
                events.append(
                    {"type": "log", "message": f"{log_prefix} branch=smart_turn_nonstream_immediate_pending_final"}
                )
                utterances = self._pop_user_utterances(include_partial=False)
                user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
                self.state.llm_in_progress = bool(user_text)
                events.extend(
                    self._record_user_message(
                        utterances,
                        trigger_path="smart_turn_nonstream_immediate_pending_final",
                    )
                )
                return self._dispatch_then_return(events)
            self.state.wait_smart_turn_final = True
            events.append({"type": "log", "message": f"{log_prefix} branch=smart_turn_nonstream_wait_final"})
            return self._dispatch_then_return(events)
        if self.state.last_is_streaming_asr:
            events.append({"type": "log", "message": f"{log_prefix} branch=fallback_streaming_history_llm"})
            self._emit_asr_latency_if_waiting(events)
            events.extend(self._trigger_llm_from_history_only(trigger_path="fallback_streaming_turn"))
            return self._dispatch_then_return(events)
        events.append({"type": "log", "message": f"{log_prefix} branch=fallback_turn_handler"})
        fallback_utterances = self._pop_user_utterances(include_partial=True)
        fallback_text = " ".join(str(t).strip() for (_idx, t) in fallback_utterances if str(t).strip()).strip()
        self.state.llm_in_progress = bool(fallback_text)
        events.extend(
            self._record_user_message(
                fallback_utterances,
                trigger_path="fallback_turn_handler",
            )
        )
        return self._dispatch_then_return(events)

    def _maybe_idle_take_turn(self) -> list[dict[str, object]]:
        """ターンロック解除中かつ最後の user_side から一定秒経過後に、crossed_up と同様にターン取得する。"""
        sec = max(0.0, _safe_float(self.settings.get("voice_dialog_idle_take_turn_sec", 0), 0.0))
        if sec <= 0.0:
            return []
        if (
            self.state.turn_blocked_until_user_speech
            or self.state.opening_tts_pending
            or self.state.llm_in_progress
            or self.state.prefetch_active
        ):
            return []
        if self.state.wait_nonstream_maai_turn is not None or self.state.wait_smart_turn_final:
            return []
        ts = self.state.last_user_speech_detect_wall_ts
        if ts is None:
            return []
        if time.time() - float(ts) < sec:
            return []
        threshold = self._maai_turn_threshold()
        prev = self.state.prev_turn_probability
        p_syn = float(prev if prev is not None else threshold)
        log_prefix = (
            "[TURN][PATH] "
            f"p_now={p_syn:.3f} "
            f"prev_p={'none' if prev is None else f'{float(prev):.3f}'} "
            f"threshold={float(threshold):.3f}"
        )
        # アイドル沈黙のプロアクティブ発話には対応する ASR final が無いので、
        # ASR レイテンシ計測を武装しない（武装すると次の実発話で誤計測になる）。
        return self._run_turn_acquire_flow(
            p_syn, log_prefix, branch_tag="idle_silence_take_turn", arm_asr_latency=False
        )

    def on_turn_probability(
        self, probability: float, *, p_future: float | None = None
    ) -> list[dict[str, object]]:
        if not self.state.active:
            return []
        threshold = self._maai_turn_threshold()
        p_now = float(probability)
        prev_p = self.state.prev_turn_probability
        self.state.prev_turn_probability = p_now
        p_fut = float(p_future) if p_future is not None else 0.0
        prev_fut = self.state.prev_p_future
        self.state.prev_p_future = p_fut
        log_prefix = (
            "[TURN][PATH] "
            f"p_now={p_now:.3f} "
            f"prev_p={'none' if prev_p is None else f'{float(prev_p):.3f}'} "
            f"p_future={p_fut:.3f} "
            f"prev_p_future={'none' if prev_fut is None else f'{float(prev_fut):.3f}'} "
            f"threshold={float(threshold):.3f}"
        )
        crossed_up = (
            (prev_p is not None)
            and (float(prev_p) <= float(threshold))
            and (p_now >= float(threshold))
        )
        p_future_crossed_up = (
            (prev_fut is not None)
            and (float(prev_fut) <= float(threshold))
            and (p_fut >= float(threshold))
        )
        p_future_crossed_down = (
            (prev_fut is not None)
            and (float(prev_fut) >= float(threshold))
            and (p_fut < float(threshold))
        )
        self._maybe_arm_scheduled_exit()
        if (
            self.state.scheduled_exit_waiting_for_turn
            and crossed_up
            and (not self.state.llm_in_progress)
            and (not self.state.opening_tts_pending)
        ):
            exit_text = str(self.settings.get("voice_dialog_scheduled_exit_utterance", "")).strip()
            sec = _safe_float(self.settings.get("voice_dialog_scheduled_exit_sec", 0), 0.0)
            if exit_text and sec > 0.0:
                cancel_ev = self._cancel_prefetch_events(reason="scheduled_exit")
                self.state.scheduled_exit_waiting_for_turn = False
                self.state.messages.append({"role": "assistant", "content": exit_text})
                self.state.pending_assistant_text = exit_text
                self.state.waiting_tts_audible = True
                self.state.tts_audible_detected = False
                self.state.last_turn_ts = time.perf_counter()
                self.state.last_turn_wall_ts = time.time()
                self.state.turn_blocked_until_user_speech = True
                return self._dispatch_then_return(
                    cancel_ev
                    + [
                        {"type": "log", "message": f"{log_prefix} branch=scheduled_exit_tts"},
                        {"type": "turn", "probability": p_now},
                        {"type": "history_snapshot", "history": self.history_snapshot()},
                        {"type": "request_scheduled_exit_tts", "assistant_text": exit_text},
                    ]
                )

        # 未確定の先読み: p_future がしきい値を下にまたいだら取り消す。
        if self.state.prefetch_active and p_future_crossed_down:
            return self._dispatch_then_return(
                self._cancel_prefetch_events(reason="p_future_cross_down")
            )

        # p_now でターンを取る前に、p_future 立ち上がりで LLM+TTS を先読みする。
        # llm_in_progress / turn_blocked は立てない（後続の p_now クロスを捨てないため）。
        if (
            p_future_crossed_up
            and (not crossed_up)
            and (not self.state.prefetch_active)
            and (not self.state.llm_in_progress)
            and (not self.state.turn_blocked_until_user_speech)
            and (not self.state.opening_tts_pending)
        ):
            pref_events = self._start_prefetch_from_history(trigger_path="p_future_cross_up")
            if pref_events:
                return self._dispatch_then_return(pref_events)

        if self.state.llm_in_progress or self.state.turn_blocked_until_user_speech:
            if self.state.llm_in_progress:
                # LLM 処理中の通常フレームは黙る。
                # crossed_up なのに取れなかったときだけ出す（次フレームの「当然ブロック」は出さない）。
                if crossed_up:
                    return self._dispatch_then_return(
                        [{"type": "log", "message": f"{log_prefix} branch=blocked reason=llm_in_progress"}]
                    )
                return []
            return []
        if not crossed_up:
            return []

        # p_now 上抜け: 先読みがあれば確定して再生。失敗していれば通常のターン取得へ。
        if self.state.prefetch_active:
            if self.state.prefetch_failed:
                cancel_ev = self._cancel_prefetch_events(reason="prefetch_failed_fallback")
                if cancel_ev:
                    self._dispatch_then_return(cancel_ev)
                return self._run_turn_acquire_flow(
                    p_now, log_prefix, branch_tag="cross_up_prefetch_failed_fallback"
                )
            return self._commit_prefetch_turn(
                p_now, log_prefix, branch_tag="cross_up_commit_prefetch"
            )

        # _run_turn_acquire_flow 内で既に _dispatch_then_return 済み。ここで再ラップすると trigger_llm が二重に走る。
        return self._run_turn_acquire_flow(p_now, log_prefix, branch_tag="cross_up_detected")

    def on_asr_result(self, payload: dict[str, Any]) -> list[dict[str, object]]:
        events: list[dict[str, object]] = [{"type": "asr_result", **dict(payload)}]
        if not self.state.active:
            return self._dispatch_then_return(events)
        # オープニング発話は最後まで再生する方針。再生中(opening_tts_pending)に来た
        # ユーザー発話は履歴に記録せず破棄し、オープニング完了後に来た発話から記録を始める。
        # （ASR結果イベント自体は表示用に emit するが、pending/履歴/ターンには載せない）
        if self.state.opening_tts_pending:
            return self._dispatch_then_return(events)
        text = str(payload.get("text", "")).strip()
        self.state.last_is_streaming_asr = bool(payload.get("is_streaming", False))
        is_final = bool(payload.get("is_final", False))
        utterance_index_obj = payload.get("utterance_index")
        try:
            utterance_index = int(utterance_index_obj) if utterance_index_obj is not None else None
        except Exception:
            utterance_index = None
        # 空の partial だけ弾く。空の is_final は force_finalize 等であり得、ここで return すると
        # waiting_asr_latency が解除されず次のターンまで残る。
        if not text and not is_final:
            return self._dispatch_then_return(events)
        payload_streaming = bool(payload.get("is_streaming", False))
        if payload_streaming:
            if text or is_final:
                rebase = self._apply_streaming_asr_to_history(text, utterance_index)
                if isinstance(rebase, dict) and str(rebase.get("type", "")).strip():
                    events.append(dict(rebase))
            if is_final:
                now_wall = float(payload.get("final_arrival_ts", time.time()) or time.time())
                self._emit_asr_latency_if_waiting(events, end_wall_ts=now_wall)
                self.state.final_seq += 1
            self.state.current_partial = ""
            self.state.current_partial_utterance_index = None
            events.append({"type": "history_snapshot", "history": self.history_snapshot()})
            return self._dispatch_then_return(events)
        if is_final:
            now_wall = float(payload.get("final_arrival_ts", time.time()) or time.time())
            self._emit_asr_latency_if_waiting(events, end_wall_ts=now_wall)
            self.state.final_seq += 1
            replaced_streaming_msg = False
            msg_idx = self.state.active_user_msg_index
            if (
                utterance_index is not None
                and msg_idx is not None
                and 0 <= int(msg_idx) < len(self.state.messages)
                and self.state.messages[int(msg_idx)].get("role") == "user"
                and int(utterance_index) in self.state.active_user_msg_utterance_text
            ):
                self.state.active_user_msg_utterance_text[int(utterance_index)] = text
                parts: list[str] = []
                for idx in self.state.active_user_msg_utterance_order:
                    t = str(self.state.active_user_msg_utterance_text.get(idx, "")).strip()
                    if t:
                        parts.append(t)
                merged = " ".join(parts).strip()
                if merged:
                    self.state.messages[int(msg_idx)] = {"role": "user", "content": merged}
                    replaced_streaming_msg = True
                    self.state.active_user_msg_await_final_replace = False
            if (
                (not replaced_streaming_msg)
                and text
                and self.state.active_user_msg_await_final_replace
            ):
                msg_idx_fb = self.state.active_user_msg_index
                if (
                    msg_idx_fb is not None
                    and 0 <= int(msg_idx_fb) < len(self.state.messages)
                    and self.state.messages[int(msg_idx_fb)].get("role") == "user"
                ):
                    self.state.messages[int(msg_idx_fb)] = {"role": "user", "content": text}
                    if utterance_index is not None:
                        ui = int(utterance_index)
                        self.state.active_user_msg_utterance_text[ui] = text
                        if ui not in self.state.active_user_msg_utterance_order:
                            self.state.active_user_msg_utterance_order.append(ui)
                    replaced_streaming_msg = True
                    self.state.active_user_msg_await_final_replace = False
            if (
                (not replaced_streaming_msg)
                and text
                and utterance_index is not None
            ):
                row_idx = self.state.streaming_user_row_by_utterance.get(int(utterance_index))
                if (
                    row_idx is not None
                    and 0 <= int(row_idx) < len(self.state.messages)
                    and self.state.messages[int(row_idx)].get("role") == "user"
                ):
                    self.state.messages[int(row_idx)] = {"role": "user", "content": text}
                    replaced_streaming_msg = True
            if not replaced_streaming_msg and text:
                self.state.pending_texts.append(text)
                self.state.pending_utterance_indices.append(utterance_index)
            self.state.current_partial = ""
            self.state.current_partial_utterance_index = None
            if self.state.wait_nonstream_maai_turn is not None:
                baseline = int(self.state.wait_nonstream_maai_turn.get("baseline_final_seq", 0) or 0)
                if self.state.final_seq > baseline:
                    self.state.wait_nonstream_maai_turn = None
                    utterances = self._pop_user_utterances(include_partial=False)
                    user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
                    self.state.llm_in_progress = bool(user_text)
                    events.extend(
                        self._record_user_message(
                            utterances,
                            trigger_path="nonstream_wait_maai_final_arrived",
                        )
                    )
            if self.state.wait_smart_turn_final:
                self.state.wait_smart_turn_final = False
                utterances = self._pop_user_utterances(include_partial=False)
                user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
                self.state.llm_in_progress = bool(user_text)
                events.extend(
                    self._record_user_message(
                        utterances,
                        trigger_path="smart_turn_waited_final_arrived",
                    )
                )
        else:
            self.state.current_partial = text
            self.state.current_partial_utterance_index = utterance_index
        events.append({"type": "history_snapshot", "history": self.history_snapshot()})
        return self._dispatch_then_return(events)

    def on_inference_state(self, busy: bool) -> list[dict[str, object]]:
        self.state.asr_inference_busy = bool(busy)
        waiting = self.state.wait_nonstream_maai_turn
        if waiting is not None:
            if bool(waiting.get("awaiting_inference_state", False)):
                waiting["awaiting_inference_state"] = False
                waiting["inference_state_deadline_ts"] = None
            now = float(time.time())
            if bool(busy):
                waiting["saw_busy_true"] = True
                waiting["post_busy_result_deadline_ts"] = None
            elif bool(waiting.get("saw_busy_true", False)):
                # busy解除直後に final が遅れて届くケースを吸収する猶予。
                if waiting.get("post_busy_result_deadline_ts") is None:
                    waiting["post_busy_result_deadline_ts"] = now + 0.3
        return []

    def on_force_finalize_status(
        self,
        *,
        is_speaking: bool,
        request_id: int = 0,
        ts: float,
    ) -> list[dict[str, object]]:
        waiting = self.state.wait_nonstream_maai_turn
        if waiting is not None:
            waiting["awaiting_force_finalize_status"] = False
            waiting["force_finalize_is_speaking"] = bool(is_speaking)
            waiting["force_finalize_request_id"] = int(request_id)
            waiting["force_finalize_status_ts"] = float(ts)
            if not bool(is_speaking):
                events: list[dict[str, object]] = [
                    {
                        "type": "log",
                        "message": (
                            "[TURN][PATH] "
                            "branch=force_finalize_status_received "
                            f"is_speaking={bool(is_speaking)} "
                            f"request_id={int(request_id)}"
                        ),
                    },
                    {
                        "type": "log",
                        "message": "[TURN][PATH] branch=nonstream_not_speaking_status",
                    },
                ]
                # is_speaking=False でも ASR が busy の間は結果到着を優先して待つ。
                if self.state.asr_inference_busy:
                    waiting["saw_busy_true"] = True
                    events.append(
                        {
                            "type": "log",
                            "message": "[TURN][PATH] branch=nonstream_not_speaking_wait_busy_result",
                        }
                    )
                else:
                    self.state.wait_nonstream_maai_turn = None
                    utterances = self._pop_user_utterances(include_partial=False)
                    user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
                    self.state.llm_in_progress = bool(user_text)
                    events.append(
                        {
                            "type": "log",
                            "message": "[TURN][PATH] branch=nonstream_not_speaking_immediate_llm",
                        }
                    )
                    events.extend(
                        self._record_user_message(
                            utterances,
                            trigger_path="nonstream_not_speaking_immediate_llm",
                        )
                    )
                return self._dispatch_then_return(events)
            waiting["awaiting_inference_state"] = True
            waiting["inference_state_deadline_ts"] = float(time.time() + 0.35)
            return self._dispatch_then_return([
                {
                    "type": "log",
                    "message": (
                        "[TURN][PATH] "
                        f"branch=force_finalize_status_received "
                        f"is_speaking={bool(is_speaking)} "
                        f"request_id={int(request_id)}"
                    ),
                }
            ])
        return []

    def on_force_finalize_outcome(
        self,
        *,
        status: str,
        request_id: int = 0,
        ts: float,
        error: str = "",
    ) -> list[dict[str, object]]:
        waiting = self.state.wait_nonstream_maai_turn
        if waiting is None:
            return []
        current_request_id = int(waiting.get("force_finalize_request_id", 0) or 0)
        if current_request_id and int(request_id) and current_request_id != int(request_id):
            return []
        waiting["force_finalize_outcome_status"] = str(status).strip().lower()
        waiting["force_finalize_outcome_ts"] = float(ts)
        if error:
            waiting["force_finalize_outcome_error"] = str(error)
        if str(status).strip().lower() in {
            "skipped_short",
            "empty_text",
            "error",
            "no_audio",
            "not_speaking",
        }:
            waiting["force_finalize_terminal_no_result"] = True
        return self._dispatch_then_return([
            {
                "type": "log",
                "message": (
                    "[TURN][PATH] "
                    f"branch=force_finalize_outcome "
                    f"status={str(status).strip().lower()} "
                    f"request_id={int(request_id)}"
                ),
            }
        ])

    def on_llm_state(self, in_progress: bool) -> list[dict[str, object]]:
        self.state.llm_in_progress = bool(in_progress)
        return self._dispatch_then_return([{"type": "history_snapshot", "history": self.history_snapshot()}])

    def on_llm_finished(
        self,
        *,
        ok: bool,
        assistant_text: str,
        error: str = "",
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        self.state.llm_in_progress = False
        text = str(assistant_text or "").strip()
        if ok and text:
            self.state.tts_cumulative_strip_prefix = ""
            self.state.pending_assistant_text = text
            self.state.waiting_tts_audible = True
            # 応答の TTS は llm/worker → tts_segment 経路で既に開始済み。可聴フラグは維持する。
        elif not ok:
            # LLM 失敗・キャンセルでもユーザー発話は履歴に残す（IPC 切断時と同様）。
            self.state.active_user_msg_await_final_replace = False
        if (not ok) or (ok and (not text)):
            self.state.tts_cumulative_strip_prefix = ""
            self.state.pending_assistant_text = None
            self.state.turn_blocked_until_user_speech = False
            self.state.last_user_speech_detect_wall_ts = None
            self._purge_empty_assistant_messages()
            # 空応答/失敗時は再生が発生しないため、controller へ tts_done 要求を返す。
            events.append(
                {
                    "type": "request_tts_done",
                    "reason": "llm_error_or_empty",
                    "error": str(error or ""),
                }
            )
        events.insert(0, {"type": "history_snapshot", "history": self.history_snapshot()})
        return self._dispatch_then_return(events)

    def on_assistant_reply(self, text: str, ok: bool) -> list[dict[str, object]]:
        self.state.llm_in_progress = False
        if ok and text.strip():
            self.state.tts_cumulative_strip_prefix = ""
            self.state.pending_assistant_text = text.strip()
            self.state.waiting_tts_audible = True
            self.state.tts_audible_detected = False
        elif not ok:
            self.state.active_user_msg_await_final_replace = False
        if not ok:
            # 失敗時はTTS再生が行われないためブロックを解除する。
            self.state.turn_blocked_until_user_speech = False
            self.state.last_user_speech_detect_wall_ts = None
            self.state.waiting_tts_audible = False
            self.state.tts_audible_detected = False
            self.state.pending_assistant_text = None
            self.state.tts_cumulative_strip_prefix = ""
        elif ok and not text.strip():
            self.state.turn_blocked_until_user_speech = False
            self.state.last_user_speech_detect_wall_ts = None
            self.state.waiting_tts_audible = False
            self.state.tts_audible_detected = False
            self.state.pending_assistant_text = None
            self.state.tts_cumulative_strip_prefix = ""
        return self._dispatch_then_return([{"type": "history_snapshot", "history": self.history_snapshot()}])

    def on_tts_audible(self) -> list[dict[str, object]]:
        # waiting_tts_audible は llm_finished 設定後にしか立たないため、
        # 先に届く可聴通知でも常に記録する（on_tts_done の無音判定用）。
        self.state.tts_audible_detected = True
        return []

    def on_assistant_segment_played(self, cumulative_text: str) -> list[dict[str, object]]:
        """分割文ごとに対応する PCM がペーサで送り切られたあと、単一 assistant レコードの content を累積更新する。"""
        text = str(cumulative_text or "").strip()
        if not text:
            return []
        msgs = self.state.messages
        if msgs and msgs[-1].get("role") == "assistant":
            msgs[-1] = {**msgs[-1], "content": text}
        else:
            self.state.messages.append({"role": "assistant", "content": text})
        return self._dispatch_then_return([{"type": "history_snapshot", "history": self.history_snapshot()}])

    def on_tts_done(self, *, playback_aborted: bool = False) -> list[dict[str, object]]:
        if playback_aborted:
            self.state.pending_assistant_text = None
            self.state.waiting_tts_audible = False
            self.state.tts_audible_detected = False
            self.state.turn_blocked_until_user_speech = False
            self.state.last_user_speech_detect_wall_ts = None
            self.state.tts_cumulative_strip_prefix = ""
            self.state.opening_tts_pending = False
        else:
            pending = (self.state.pending_assistant_text or "").strip()
            if pending:
                if self.state.waiting_tts_audible and (not self.state.tts_audible_detected):
                    # 可聴に到達せず終了（無音など）は assistant を履歴に残さない。
                    # 直前が user のみのときは pop しない（user 行は消えない）。
                    if self.state.messages and self.state.messages[-1].get("role") == "assistant":
                        self.state.messages.pop()
                    pass
                else:
                    msgs = self.state.messages
                    if msgs and msgs[-1].get("role") == "assistant":
                        msgs[-1] = {**msgs[-1], "content": pending}
                    else:
                        self.state.messages.append({"role": "assistant", "content": pending})
            self.state.pending_assistant_text = None
            self.state.waiting_tts_audible = False
            self.state.tts_audible_detected = False
            self.state.tts_cumulative_strip_prefix = ""
            if self.state.opening_tts_pending:
                self.state.opening_tts_pending = False
                # 開始発話終了後も、ユーザー発話で user_side が立つまでターン取得を抑止する（直後の crossed_up で空履歴 LLM が走るのを防ぐ）
                self.state.turn_blocked_until_user_speech = True
        return self._dispatch_then_return([
            {"type": "history_snapshot", "history": self.history_snapshot()},
            {"type": "log", "message": "tts_done"},
        ])

    def on_tick(self) -> list[dict[str, object]]:
        if not self.state.active:
            return []
        self._maybe_arm_scheduled_exit()
        if self.state.llm_in_progress or self.state.prefetch_active:
            return []
        idle_events = self._maybe_idle_take_turn()
        if idle_events:
            # _maybe_idle_take_turn → _run_turn_acquire_flow 内で既に dispatch 済み
            return idle_events
        waiting = self.state.wait_nonstream_maai_turn
        if waiting is None:
            return []
        now = float(time.time())
        if bool(waiting.get("awaiting_force_finalize_status", False)):
            # force_finalize_status を受信するまで分岐を確定しない。
            return []
        baseline = int(waiting.get("baseline_final_seq", 0) or 0)
        if self.state.final_seq > baseline:
            self.state.wait_nonstream_maai_turn = None
            utterances = self._pop_user_utterances(include_partial=False)
            user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
            self.state.llm_in_progress = bool(user_text)
            events = [{"type": "log", "message": "[TURN][PATH] branch=nonstream_wait_maai_final_arrived"}]
            events.extend(self._record_user_message(
                utterances,
                trigger_path="nonstream_wait_maai_final_arrived",
            ))
            return self._dispatch_then_return(events)
        if bool(waiting.get("awaiting_inference_state", False)):
            deadline_obj = waiting.get("inference_state_deadline_ts")
            deadline_ts = float(deadline_obj) if deadline_obj is not None else 0.0
            if deadline_ts > now:
                return []
            waiting["awaiting_inference_state"] = False
            waiting["inference_state_deadline_ts"] = None
        if bool(waiting.get("force_finalize_terminal_no_result", False)) and (not self.state.asr_inference_busy):
            self.state.wait_nonstream_maai_turn = None
            utterances = self._pop_user_utterances(include_partial=False)
            user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
            self.state.llm_in_progress = bool(user_text)
            events = [{"type": "log", "message": "[TURN][PATH] branch=nonstream_wait_force_finalize_terminal_no_result"}]
            events.extend(self._record_user_message(
                utterances,
                trigger_path="nonstream_wait_force_finalize_terminal_no_result",
            ))
            return self._dispatch_then_return(events)
        if not bool(waiting.get("force_finalize_is_speaking", False)):
            # is_speaking=False でも、busy観測後は結果到着判定を
            # final_seq増加で行い、busy解除のみでは先へ進まない。
            saw_busy_true = bool(waiting.get("saw_busy_true", False))
            if saw_busy_true:
                deadline_obj = waiting.get("post_busy_result_deadline_ts")
                deadline_ts = float(deadline_obj) if deadline_obj is not None else 0.0
                if self.state.asr_inference_busy:
                    return []
                if deadline_ts > now:
                    return []
            elif self.state.asr_inference_busy:
                return []
            # busyでなければ結果待ちは打ち切り、現時点の保持結果で先へ進む。
            self.state.wait_nonstream_maai_turn = None
            utterances = self._pop_user_utterances(include_partial=False)
            user_text = " ".join(str(t).strip() for (_idx, t) in utterances if str(t).strip()).strip()
            self.state.llm_in_progress = bool(user_text)
            events = [{"type": "log", "message": "[TURN][PATH] branch=nonstream_wait_not_speaking_or_busy_done_no_new_final"}]
            events.extend(
                self._record_user_message(
                    utterances,
                    trigger_path="nonstream_wait_not_speaking_or_busy_done_no_new_final",
                )
            )
            return self._dispatch_then_return(events)
        # is_speaking=True の場合は「強制推論の結果(final)」到着を待ってから進む。
        if bool(waiting.get("force_finalize_is_speaking", False)):
            if not bool(waiting.get("_logged_waiting_speaking", False)):
                waiting["_logged_waiting_speaking"] = True
                return self._dispatch_then_return([{"type": "log", "message": "[TURN][PATH] branch=nonstream_wait_speaking_waiting_final_seq"}])
        return []

    def history_snapshot(self) -> dict[str, object]:
        clean_msgs = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in self.state.messages
        ]
        return {
            "ts": time.time(),
            "messages": clean_msgs,
            "pending_texts": list(self.state.pending_texts),
            "pending_utterance_indices": list(self.state.pending_utterance_indices),
            "current_partial": str(self.state.current_partial),
            "active": bool(self.state.active),
            "llm_in_progress": bool(self.state.llm_in_progress),
            "vad_active": bool(self.state.vad_active),
            "turn_blocked_until_user_speech": bool(self.state.turn_blocked_until_user_speech),
            "waiting_tts_audible": bool(self.state.waiting_tts_audible),
            "tts_audible_detected": bool(self.state.tts_audible_detected),
            "pending_assistant_text": str(self.state.pending_assistant_text or ""),
            "current_partial_utterance_index": self.state.current_partial_utterance_index,
        }
