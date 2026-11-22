from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from typing import Any

# custom トークナイザ用: 主要な絵文字ブロック（広い単一レンジは使わない — 漢字等を誤除去しないため）
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F1E0-\U0001F1FF"  # regional indicators (flags)
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FAFF"  # Extended-A
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002300-\U000023FF"  # Misc technical
    "\U000024C2-\U000024FF"  # enclosed symbols (e.g. circled letters)
    "\U0001F170-\U0001F18F"  # enclosed alphanumerics supplement (flags etc.)
    "\U0001F191-\U0001F1AC"  # enclosed ideographic / terms
    "\U000020E3"  # combining enclosing keycap
    "\U0000FE0F"  # variation selector-16
    "]+",
    flags=re.UNICODE,
)

def _strip_emoji_for_custom_tokenizer(text: str) -> str:
    if not text:
        return text
    t = _EMOJI_RE.sub("", text)
    # 連結絵文字の ZWJ が単独で残ることがある（日本語 TTS 用途では除去してよい）
    return t.replace("\u200d", "")

def _strip_parenthetical_for_custom_tokenizer(text: str) -> str:
    """半角 () / 全角 （）で囲まれた部分を除去（ネストは内側から繰り返し）。"""
    if not text:
        return text
    t = text
    while True:
        nxt = re.sub(r"\([^()]*\)", "", t)
        if nxt == t:
            break
        t = nxt
    while True:
        nxt = re.sub(r"（[^（）]*）", "", t)
        if nxt == t:
            break
        t = nxt
    return t

def _strip_standalone_warai_for_custom_tokenizer(text: str) -> str:
    """ネットスラングの単独「笑」を落とす。笑い/笑う/笑わ/笑え/笑お/笑っ…や笑顔は残す。"""
    if not text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c != "笑":
            out.append(c)
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        # 空文字は in の対象にすると常に真になるため、先頭1文字だけ見る
        if nxt and nxt[0] in "いうわえおっ":
            out.append("笑")
            i += 1
            continue
        if nxt and "\u4e00" <= nxt <= "\u9fff" and nxt != "笑":
            out.append("笑")
            i += 1
            continue
        # 単独の笑・句読点直前の笑・「笑笑」など
        i += 1
    return "".join(out)

from ..ipc.jsonline import JsonLineDuplexClient
from .service import LLMService
from ..ipc.worker_runtime import connect_worker_client, finish_worker

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM worker process")
    parser.add_argument("--host", required=True, type=str)
    parser.add_argument("--port", required=True, type=int)
    return parser

def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "0", "false", "no", "off"}:
            return False
        if s in {"1", "true", "yes", "on"}:
            return True
        return bool(s)
    return default if value is None else bool(value)

def _safe_int(value: object, default: int, *, min_value: int = 1) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = int(default)
    return max(int(min_value), parsed)

def _apply_custom_tokenizer_input_strips(text: str, inputs: dict[str, object]) -> str:
    """tokenizer_mode=custom 用: 各セグメントを TTS に送る直前に適用（絵文字・括弧・単独笑）。"""
    src = str(text or "")
    if not src:
        return src
    if _to_bool(inputs.get("custom_strip_emoji", True), True):
        src = _strip_emoji_for_custom_tokenizer(src)
    if _to_bool(inputs.get("custom_strip_parenthetical", False), False):
        src = _strip_parenthetical_for_custom_tokenizer(src)
    if _to_bool(inputs.get("custom_strip_standalone_warai", False), False):
        src = _strip_standalone_warai_for_custom_tokenizer(src)
    return src

def _split_sentences_custom_delimiters(text: str, inputs: dict[str, object]) -> list[str]:
    """tokenizer_mode=custom 用: custom_delimiters に含まれる文字の位置で順次セグメントを確定する（外部トーカナイザ不使用）。"""
    src = str(text or "")
    if not src.strip():
        return []
    raw = str(inputs.get("custom_delimiters", "") or "").strip()
    if not raw:
        raw = "。．！？!?\\n"
    delims = raw.replace("\\n", "\n").replace("\\t", "\t")
    delim_set = frozenset(delims)

    min_first = _safe_int(inputs.get("minimum_first_fragment_length", 4), 4, min_value=1)
    min_sent = _safe_int(inputs.get("minimum_sentence_length", 4), 4, min_value=1)
    ctx = _safe_int(inputs.get("context_size", 12), 12, min_value=1)
    fast = _to_bool(inputs.get("fast_sentence_fragment", False), False)
    if fast:
        min_first = max(1, min_first // 2)
        min_sent = max(1, min_sent // 2)
        ctx = max(1, ctx // 2)

    # 先頭以外は文長しきい値と context_size の大きい方まで溜めてから区切る（順次・貪欲）
    next_needed = max(min_sent, ctx)

    out: list[str] = []
    buf: list[str] = []
    first = True

    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in delim_set:
            d = ch
            while i < n and src[i] == d:
                buf.append(src[i])
                i += 1
            piece = "".join(buf).strip()
            if not piece:
                continue
            need = min_first if first else next_needed
            if len(piece) >= need:
                out.append(piece)
                buf = []
                first = False
        else:
            buf.append(ch)
            i += 1

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)

    normalized = [s for s in (x.strip() for x in out) if s]
    return normalized if normalized else ([src.strip()] if src.strip() else [])

def main() -> int:
    args = _build_parser().parse_args()
    client = JsonLineDuplexClient(args.host, args.port)
    if not connect_worker_client(client, worker_name="LLMWorker"):
        return 2

    stop_event = threading.Event()
    command_queue: list[dict[str, Any]] = []
    command_lock = threading.Lock()
    command_event = threading.Event()
    # cancel フラグ。_on_cmd(IPC リーダースレッド)と消費ループ/ストリーミング(メインスレッド)が
    # 同時アクセスするため command_lock で保護する。上限超過時は全消去(clear)だと in-flight 中の
    # cancel まで消えて取りこぼす(#17)。挿入順 dict にして最古から退避し、新しい cancel は残す。
    cancelled_ids: "dict[str, None]" = {}
    _CANCELLED_CAP = 256
    active_ids: set[str] = set()

    def _mark_cancelled(rid: str) -> None:
        rid = str(rid or "").strip()
        if not rid:
            return
        with command_lock:
            cancelled_ids.pop(rid, None)
            cancelled_ids[rid] = None  # 末尾(最新)へ
            while len(cancelled_ids) > _CANCELLED_CAP:
                cancelled_ids.pop(next(iter(cancelled_ids)), None)

    def _is_cancelled(rid: str) -> bool:
        with command_lock:
            return rid in cancelled_ids

    def _clear_cancelled(rid: str) -> None:
        with command_lock:
            cancelled_ids.pop(rid, None)

    def emit(event_type: str, **data: object) -> None:
        payload: dict[str, object] = {"type": event_type}
        payload.update(data)
        try:
            client.send(payload)
        except Exception:
            pass

    service = LLMService(on_log=lambda message: emit("log", message=message))

    def _on_cmd(payload: dict[str, object]) -> None:
        # cancel は run_chat 実行中（単一のコマンド消費ループがブロック中）でも効く必要がある。
        # 消費ループは run_chat の間デキューできないため、ここ（IPC リーダースレッド）で
        # 即座に cancelled_ids へ反映する（is_cancelled ラムダがストリーミング中に参照する）。
        if str(payload.get("type", "")).strip().lower() == "cancel":
            _mark_cancelled(str(payload.get("request_id", "")).strip())
            return
        with command_lock:
            command_queue.append(dict(payload))
        command_event.set()

    def _on_disconnect() -> None:
        stop_event.set()
        command_event.set()

    client.start_reader(on_event=_on_cmd, on_disconnected=_on_disconnect)

    def _stdin_watcher() -> None:
        try:
            for raw in sys.stdin:
                if raw.strip().lower() == "stop":
                    stop_event.set()
                    command_event.set()
                    return
            stop_event.set()
            command_event.set()
        except Exception:
            stop_event.set()
            command_event.set()

    threading.Thread(target=_stdin_watcher, daemon=True, name="llm-worker-stdin").start()

    def _run_start(cmd: dict[str, Any], request_id: str) -> None:
        """1 件の LLM 要求を処理する。要求ごとのスレッドで動くので、取り消した要求の
        最初のトークン待ちで次の要求が止まらない（emit は IPC 送信ロックで直列化される）。"""
        with command_lock:
            active_ids.add(request_id)

        backend = str(cmd.get("backend", "LM Studio"))
        model = str(cmd.get("model", "")).strip()
        host = str(cmd.get("host", "")).strip()
        api_key = str(cmd.get("api_key", "")).strip()
        api_key_env = str(cmd.get("api_key_env", "")).strip()
        stream = bool(cmd.get("stream", True))
        max_tokens = int(cmd.get("max_tokens", 128) or 128)
        send_temperature_raw = cmd.get("send_temperature", None)
        send_max_tokens_raw = cmd.get("send_max_tokens", None)
        send_temperature: bool | None
        send_max_tokens: bool | None
        if send_temperature_raw is None:
            send_temperature = None
        else:
            send_temperature = _to_bool(send_temperature_raw, True)
        if send_max_tokens_raw is None:
            send_max_tokens = None
        else:
            send_max_tokens = _to_bool(send_max_tokens_raw, True)
        try:
            temperature_val = float(cmd.get("temperature", 0.7))
        except Exception:
            temperature_val = 0.7
        custom_extra_body_thinking = _to_bool(cmd.get("custom_extra_body_thinking", False), False)
        tts_stream_inputs_raw = cmd.get("tts_stream_inputs")
        tts_stream_inputs = dict(tts_stream_inputs_raw) if isinstance(tts_stream_inputs_raw, dict) else {}
        tts_to_speech = bool(str(tts_stream_inputs.get("engine", "")).strip())
        tts_input_streaming_enabled = _to_bool(tts_stream_inputs.get("input_streaming_enabled", True), True)
        messages_raw = cmd.get("messages")
        messages: list[dict[str, str]] = []
        if isinstance(messages_raw, list):
            for item in messages_raw:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", ""))
                if role in {"system", "user", "assistant"}:
                    messages.append({"role": role, "content": content})
        started_at = time.perf_counter()
        split_state = {"assistant_text": "", "pushed_complete_count": 0, "sent_text": ""}

        def emit_tts_segments_from_accumulated(*, final: bool) -> None:
            if (not tts_to_speech) or _is_cancelled(request_id):
                return
            full = str(split_state.get("assistant_text", ""))
            if not full.strip():
                return
            if not tts_input_streaming_enabled:
                if final:
                    segs_once = _split_sentences_custom_delimiters(full, tts_stream_inputs)
                    if segs_once:
                        seg_out = "".join(
                            _apply_custom_tokenizer_input_strips(str(s), tts_stream_inputs)
                            for s in segs_once
                        )
                    else:
                        seg_out = _apply_custom_tokenizer_input_strips(full, tts_stream_inputs)
                    emit(
                        "tts_segment",
                        request_id=request_id,
                        segment_text=seg_out,
                        segment_index=0,
                        is_final=True,
                    )
                return
            segs = _split_sentences_custom_delimiters(full, tts_stream_inputs)
            if not segs:
                return
            complete_count = max(0, len(segs) - 1)
            pushed = int(split_state.get("pushed_complete_count", 0) or 0)
            sent_text = str(split_state.get("sent_text", ""))
            if complete_count > pushed:
                for idx in range(pushed, complete_count):
                    raw_seg = str(segs[idx])
                    if not raw_seg:
                        continue
                    out = _apply_custom_tokenizer_input_strips(raw_seg, tts_stream_inputs)
                    sent_text += raw_seg
                    if not str(out).strip():
                        continue
                    emit(
                        "tts_segment",
                        request_id=request_id,
                        segment_text=out,
                        segment_index=idx,
                        is_final=False,
                    )
                split_state["pushed_complete_count"] = complete_count
                split_state["sent_text"] = sent_text
            if final:
                # ストリーミング中は文単位で既に送っている。確定時は全文を再送せず、
                # 未確定末尾のみ送る（sent_text 追跡と再分割のズレで startswith が偽になり
                # final_raw=full になると TTS・履歴が二重になる）。
                sent_text = str(split_state.get("sent_text", ""))
                if full.startswith(sent_text):
                    final_raw = full[len(sent_text) :]
                elif segs:
                    final_raw = str(segs[-1])
                else:
                    final_raw = full
                final_out = _apply_custom_tokenizer_input_strips(final_raw, tts_stream_inputs)
                if not str(final_out).strip():
                    return
                emit(
                    "tts_segment",
                    request_id=request_id,
                    segment_text=str(final_out),
                    segment_index=max(0, len(segs) - 1),
                    is_final=True,
                )
                split_state["sent_text"] = full

        try:
            def _run_once() -> Any:
                split_state["assistant_text"] = ""
                split_state["pushed_complete_count"] = 0
                split_state["sent_text"] = ""
                return service.run_chat(
                    backend=backend,
                    host=host,
                    model=model,
                    api_key=api_key,
                    api_key_env=api_key_env,
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=stream,
                    send_temperature=send_temperature,
                    send_max_tokens=send_max_tokens,
                    temperature=temperature_val,
                    custom_extra_body_thinking=custom_extra_body_thinking,
                    on_chunk=lambda text: (
                        emit("chunk", request_id=request_id, text=text),
                        split_state.__setitem__(
                            "assistant_text",
                            str(split_state.get("assistant_text", "")) + str(text),
                        ),
                        emit_tts_segments_from_accumulated(final=False),
                    ),
                    is_cancelled=lambda: _is_cancelled(request_id),
                )

            result = _run_once()
            if _is_cancelled(request_id):
                emit("error", request_id=request_id, message="cancelled")
            elif not str(getattr(result, "assistant_text", "") or "").strip():
                emit("error", request_id=request_id, message="empty_response")
            else:
                split_state["assistant_text"] = str(
                    result.assistant_text or split_state.get("assistant_text", "")
                )
                emit_tts_segments_from_accumulated(final=True)
                emit(
                    "done",
                    request_id=request_id,
                    assistant_text=result.assistant_text,
                    ttft_sec=float(result.ttft_sec),
                    total_sec=float(result.total_sec),
                    total_overall_sec=float(time.perf_counter() - started_at),
                )
        except Exception as ex:
            emit("error", request_id=request_id, message=str(ex))
        finally:
            with command_lock:
                active_ids.discard(request_id)
            _clear_cancelled(request_id)

    emit("ready")

    while not stop_event.is_set():
        command_event.wait(timeout=0.2)
        command_event.clear()
        while True:
            with command_lock:
                if not command_queue:
                    break
                cmd = command_queue.pop(0)
            cmd_type = str(cmd.get("type", "")).strip().lower()
            request_id = str(cmd.get("request_id", "")).strip()
            if cmd_type == "cancel":
                _mark_cancelled(request_id)
                continue
            if cmd_type == "warmup":
                backend = str(cmd.get("backend", "LM Studio"))
                model = str(cmd.get("model", "")).strip()
                host = str(cmd.get("host", "")).strip()
                api_key = str(cmd.get("api_key", "")).strip()
                api_key_env = str(cmd.get("api_key_env", "")).strip()
                send_temperature_raw = cmd.get("send_temperature", None)
                send_max_tokens_raw = cmd.get("send_max_tokens", None)
                if send_temperature_raw is None:
                    send_temperature = None
                else:
                    send_temperature = _to_bool(send_temperature_raw, True)
                if send_max_tokens_raw is None:
                    send_max_tokens = None
                else:
                    send_max_tokens = _to_bool(send_max_tokens_raw, True)
                try:
                    temperature_val = float(cmd.get("temperature", 0.7))
                except Exception:
                    temperature_val = 0.7
                custom_extra_body_thinking = _to_bool(cmd.get("custom_extra_body_thinking", False), False)
                messages_raw = cmd.get("messages")
                messages: list[dict[str, str]] = []
                if isinstance(messages_raw, list):
                    for item in messages_raw:
                        if not isinstance(item, dict):
                            continue
                        role = str(item.get("role", "")).strip()
                        content = str(item.get("content", ""))
                        if role in {"system", "user", "assistant"}:
                            messages.append({"role": role, "content": content})
                ok = service.warmup_client_once(
                    backend=backend,
                    host=host,
                    model=model,
                    api_key=api_key,
                    api_key_env=api_key_env,
                    force=_to_bool(cmd.get("force", False), False),
                    send_temperature=send_temperature,
                    send_max_tokens=send_max_tokens,
                    temperature=temperature_val,
                    custom_extra_body_thinking=custom_extra_body_thinking,
                    messages=messages or None,
                    stream=_to_bool(cmd.get("stream", True), True),
                )
                emit(
                    "warmup_done",
                    request_id=request_id,
                    ok=bool(ok),
                    message="" if ok else "LLM warmup failed",
                )
                continue
            if cmd_type != "start":
                continue
            if not request_id:
                continue
            if _is_cancelled(request_id):
                emit("error", request_id=request_id, message="cancelled")
                _clear_cancelled(request_id)
                continue
            threading.Thread(
                target=_run_start, args=(cmd, request_id), daemon=True,
                name=f"llm-request-{request_id}",
            ).start()

    with command_lock:
        still_running = list(active_ids)
    for rid in still_running:
        _mark_cancelled(rid)
    finish_worker(client, lambda: emit("stopped"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
