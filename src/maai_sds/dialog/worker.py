from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

from .orchestrator import DialogOrchestrator
from ..ipc.jsonline import JsonLineDuplexServer, JsonLineEventServer
from ..ipc.worker_runtime import start_parent_death_watchdog
from ..ipc.orchestrator_supervisor import OrchestratorSupervisor
from ..mmd.runtime import MmdRuntime
from ..tts.routing import TtsRouter
from .runtime import VoiceDialogRuntime

from .. import paths as _paths


# DISPATCH に無いイベントタイプを受け取った際、各タイプ初回だけ警告する記録。
# プロトコル不一致（ワーカーとのバージョン差・フィールド改名）で機能が無言で
# 欠落するのを可視化するため（type を捨てるだけだと故障が一切表面化しない）。
_WARNED_UNKNOWN_EVENT_TYPES: set[str] = set()


def _resolve_config_path():
    """Bootstrap が読む設定ファイル。`--config PATH` があればそれを使う。

    無指定のときは従来どおり自分の隣の `controller_config.json`（挙動不変）。
    """
    from pathlib import Path

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser().resolve()
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    return _paths.config_path()


def _assume_mmd_running() -> bool:
    """`--assume-mmd-running` があれば Bootstrap でも audio_output_mode=mmd を保つ。

    既定では MMDAgent 未起動とみなし mmd を local に落とす。Web サーバは別プロセスで
    起動するネイティブ MMDAgent-EX の状態を判定できないため、このフラグを指定する。
    """
    return "--assume-mmd-running" in sys.argv[1:]


def parse_listen_endpoint(value: str) -> tuple[str, int]:
    """Parse the IPv4/hostname ``HOST:PORT`` form accepted by ``--listen``."""
    text = str(value or "").strip()
    host, sep, port_text = text.rpartition(":")
    if not sep or not host.strip() or not port_text.strip():
        raise ValueError(f"listen endpoint must be HOST:PORT, got {value!r}")
    try:
        port = int(port_text)
    except ValueError as ex:
        raise ValueError(f"listen port must be an integer, got {port_text!r}") from ex
    if not 0 <= port <= 65535:
        raise ValueError(f"listen port is outside 0..65535: {port}")
    return host.strip(), port


def _listen_endpoint() -> tuple[str, int] | None:
    argv = sys.argv[1:]
    raw = ""
    for index, arg in enumerate(argv):
        if arg == "--listen":
            if index + 1 >= len(argv):
                raise ValueError("--listen requires HOST:PORT")
            raw = argv[index + 1]
            break
        if arg.startswith("--listen="):
            raw = arg.split("=", 1)[1]
            break
    if not raw:
        raw = os.environ.get("MAAI_SDS_UI_LISTEN", "")
    return parse_listen_endpoint(raw) if str(raw).strip() else None


def main() -> int:
    _worker_start_ts = time.time()  # stdin 未接続の判定に使う（_stdin_watcher 参照）
    listen_endpoint = _listen_endpoint()
    listen_mode = listen_endpoint is not None
    listen_host, listen_port = listen_endpoint or ("127.0.0.1", 0)
    ui_server = JsonLineDuplexServer(listen_host, listen_port)
    if listen_mode:
        print(
            json.dumps(
                {"ui_host": ui_server.host, "ui_port": ui_server.port, "listen_mode": True}
            ),
            flush=True,
        )
    else:
        # The local Qt launcher consumes this exact one-field bootstrap record.
        print(json.dumps({"ui_port": ui_server.port}), flush=True)

    # 頷き補間 cubic_spline 等で nod_motion が scipy を遅延 import すると、
    # 初回だけ DLL ロード等で数十秒ブロックし「MaAI テストが固まった」ように見える。
    # ワーカー起動時にまとめてロードして、初回頷き生成と重ねない。
    try:
        import scipy.interpolate  # noqa: F401
    except ImportError:
        pass

    stop_event = threading.Event()
    event_queue: queue.Queue[dict[str, object]] = queue.Queue()
    audio_input_server = JsonLineEventServer()
    asr_audio_server = JsonLineEventServer()
    asr_compare_audio_server = JsonLineEventServer()
    maai_audio_server = JsonLineEventServer()

    def emit(payload: dict[str, object]) -> None:
        try:
            if ui_server.is_connected():
                ui_server.send(dict(payload))
        except Exception:
            pass

    # -- Supervisor / mmd / runtime / tts_router 構成 ----------------------------
    # supervisor は最初に作る必要がある（mmd/runtime が依存するため）。
    # orchestrator は VoiceDialogRuntime のコールバックを受けるので、runtime 構築後に作る。

    def log_fn(msg: str) -> None:
        emit({"type": "log", "message": msg})

    def _audio_output_worker_fan_in(p: dict) -> None:
        wrapped = {"_src": "audio_output", **dict(p)}
        event_queue.put(wrapped)
        et = str(p.get("type", "")).strip().lower()
        # tts_maai_pcm_frame はキュー側で system_audio 注入のみ。UI へ送ると Controller が
        # 重複注入していたため除外（無音ティックは VoiceDialogRuntime 内スレッド）。
        if et == "tts_maai_pcm_frame":
            return
        # ready は _handle_event が {type, worker, output_mode} に正規化して 1 回だけ emit する。
        # ここでも emit すると Controller に [AudioOutput] ready が二重表示される。
        if et == "ready":
            return
        emit(wrapped)

    def _on_supervisor_tts_event(p: dict) -> None:
        tts_router.complete_from_event(p)
        wrapped = {"_src": "tts", **p}
        emit(wrapped)
        if str(p.get("type", "")).strip().lower() == "fatal":
            message = str(p.get("message", "") or "TTS worker reported a fatal error")
            emit(
                {
                    "type": "module_state",
                    "module": "tts",
                    "state": "error",
                    "source": "worker_fatal",
                    "error": message,
                }
            )

    supervisor = OrchestratorSupervisor(
        log_fn=log_fn,
        on_llm_event=lambda p: event_queue.put({"_src": "llm", **p}),
        on_tts_event=_on_supervisor_tts_event,
        on_audio_output_event=_audio_output_worker_fan_in,
        on_llm_admin_event=lambda p: emit({"_src": "llm_admin", **p}),
    )

    mmd = MmdRuntime(
        emit=emit,
        send_audio_output=supervisor.send_audio_output,
        log_fn=log_fn,
    )

    runtime = VoiceDialogRuntime(
        emit=emit,
        event_queue=event_queue,
        supervisor=supervisor,
        mmd=mmd,
        audio_input_server=audio_input_server,
        asr_audio_server=asr_audio_server,
        asr_compare_audio_server=asr_compare_audio_server,
        maai_audio_server=maai_audio_server,
    )

    tts_router = TtsRouter(
        emit=emit,
        supervisor=supervisor,
        vd_state=runtime.vd_state,
        on_maai_system_inject_activity=runtime.notify_maai_system_inject_activity,
    )
    runtime.set_tts_router(tts_router)

    orchestrator = DialogOrchestrator(**runtime.as_orchestrator_callbacks())
    runtime.set_orchestrator(orchestrator)

    # -- stdin watcher --

    def _stdin_watcher() -> None:
        """stdin の "stop" で終了する。親が stdin を閉じたら（EOF）親の死とみなして終了する。

        ただし **stdin がそもそも繋がっていない**（`< /dev/null`、CI/cron、
        テストハーネスなど）場合は起動直後に即 EOF になる。以前はそれも
        「親が閉じた」と解釈して `finally` で停止していたため、
        **非対話起動では 2 秒で自死していた**（F6）。
        1 行も読めずに即 EOF なら「stdin 未接続」と判断し、停止させない。
        """
        got_line = False
        try:
            for raw in sys.stdin:
                got_line = True
                if raw.strip().lower() == "stop":
                    stop_event.set()
                    return
        except Exception:
            pass
        if listen_mode:
            # A persistent service is independently managed by Compose (or a
            # service manager).  EOF only means there is no interactive
            # console; an explicit "stop" line above still works.
            return
        if not got_line and (time.time() - _worker_start_ts) < 5.0:
            emit({
                "type": "log",
                "message": "[Orchestrator] stdin 未接続のため stdin 監視を無効化します"
                           "（stop コマンドは IPC 経由で受け付けます）。",
            })
            return
        stop_event.set()

    # -- Dispatch table for _handle_event ----------------------------------------
    # 各エントリは payload(dict) を受け取り、None を返す。
    # 重い処理は runtime / mmd / supervisor のメソッドに集約済み。

    def _settings_from_payload(p: dict) -> dict:
        s = p.get("settings")
        return s if isinstance(s, dict) else {}

    def _relay_inner_payload(p: dict, send_fn, fail_log: str) -> None:
        inner = p.get("payload")
        if isinstance(inner, dict) and not send_fn(inner):
            emit({"type": "log", "message": fail_log, "source": str(p.get("source", "")).strip()})

    def _passthrough_emit(p: dict) -> None:
        emit(dict(p))

    def _h_internal_mmd_complete(p: dict) -> None:
        completion = p.get("completion")
        if isinstance(completion, dict):
            runtime.on_mmd_send_complete(completion)

    def _h_interrupt(_p: dict) -> None:
        mmd.cancel_nod_motion()
        orchestrator.interrupt()

    def _h_send_to_llm(p: dict) -> None:
        _relay_inner_payload(p, supervisor.send_llm, "[Supervisor] send_to_llm failed: not connected")

    def _h_send_to_tts(p: dict) -> None:
        _relay_inner_payload(p, supervisor.send_tts, "[Supervisor] send_to_tts failed: not connected")

    def _h_send_to_tts_bridge(p: dict) -> None:
        inner = p.get("payload")
        if isinstance(inner, dict):
            try:
                supervisor.handle_tts_bridge_command(inner)
            except Exception as ex:
                emit({"type": "log", "message": f"[Supervisor] send_to_tts_bridge failed: {ex}"})

    def _h_send_to_llm_admin(p: dict) -> None:
        inner = p.get("payload")
        if isinstance(inner, dict):
            try:
                supervisor.handle_llm_admin_http_command(inner)
            except Exception as ex:
                emit({"type": "log", "message": f"[Supervisor] send_to_llm_admin failed: {ex}"})

    def _h_send_to_audio_output(p: dict) -> None:
        inner = p.get("payload")
        if isinstance(inner, dict) and not supervisor.send_audio_output(inner):
            emit({"type": "log", "message": "[Supervisor] send_to_audio_output failed: not connected"})

    def _h_inject_system_audio(p: dict) -> None:
        data_b64 = str(p.get("audio_b64", "")).strip()
        if data_b64:
            supervisor.send_audio_input_command(f"system_audio:{data_b64}")

    def _h_inject_user_audio(p: dict) -> None:
        data_b64 = str(p.get("audio_b64", "")).strip()
        if data_b64:
            supervisor.send_audio_input_command(f"user_audio:{data_b64}")

    def _h_asr_send_command(p: dict) -> None:
        cmd_str = str(p.get("command", "")).strip()
        if cmd_str:
            supervisor.asr.send_command(cmd_str)

    def _h_audio_input_send_command(p: dict) -> None:
        cmd_str = str(p.get("command", "")).strip()
        if not cmd_str:
            return
        if supervisor.audio_input.is_running():
            supervisor.send_audio_input_command(cmd_str)
            return
        if cmd_str != "wav_play":
            return

        def _wait_send() -> None:
            deadline = time.time() + 15.0
            while time.time() < deadline:
                if supervisor.audio_input.is_running():
                    supervisor.send_audio_input_command(cmd_str)
                    return
                time.sleep(0.05)
            emit({"type": "log", "message": "[AudioInput] 未起動のため wav_play を送れませんでした"})

        threading.Thread(target=_wait_send, daemon=True, name="audio-input-wav-play").start()

    def _h_reset_nod_para_timing(_p: dict) -> None:
        orchestrator.reset_nod_para_timing_state()
        emit({"type": "log", "message": "[Orchestrator] nod_para timing state reset"})

    def _relay_audio_output_payload(p: dict) -> None:
        if not supervisor.send_audio_output(dict(p)):
            emit({"type": "log", "message": "[Supervisor] audio_output relay failed"})

    def _h_wav_other_ch_audio(p: dict) -> None:
        supervisor.send_audio_output(
            {"type": "tts_play_chunk", "audio_b64": p.get("audio_b64", ""), "inject_maai": False}
        )

    def _h_wav_other_ch_audio_end(_p: dict) -> None:
        supervisor.send_audio_output({"type": "tts_play_break"})

    DISPATCH: dict[str, "object"] = {
        "_orch_internal_mmd_complete": _h_internal_mmd_complete,
        # voice dialog lifecycle
        "start_voice_dialog": lambda p: runtime.handle_start_voice_dialog(_settings_from_payload(p)),
        "stop_voice_dialog": runtime.handle_stop_voice_dialog,
        # ensure_* 完了後に runtime から event_queue 経由で戻され、対話 state 起動をループ上で実行する。
        "_vd_start_finalize": runtime.finalize_start_voice_dialog,
        "update_settings": lambda p: runtime.handle_update_settings(_settings_from_payload(p)),
        "interrupt": _h_interrupt,
        "reset_nod_para_timing": _h_reset_nod_para_timing,
        # warmup / GUI tests
        "start_llm_gui_test": runtime.handle_start_llm_gui_test,
        "start_tts_gui_test": runtime.handle_start_tts_gui_test,
        "start_llm_warmup": runtime.handle_start_llm_warmup,
        "start_tts_worker_warmup": runtime.handle_start_tts_worker_warmup,
        # MMD / audio/output_worker へ IPC リレー（ペーサ・bridge は別プロセス）
        "mmd_audio_flush": lambda _p: supervisor.send_audio_output({"type": "interrupt"}),
        "flush": lambda _p: supervisor.send_audio_output({"type": "interrupt"}),
        "mmd_av_preamble_once": lambda _p: supervisor.send_audio_output({"type": "mmd_av_preamble_once"}),
        "tts_play_chunk": _relay_audio_output_payload,
        "tts_play_break": _relay_audio_output_payload,
        "wav_other_ch_audio": _h_wav_other_ch_audio,
        "wav_other_ch_audio_end": _h_wav_other_ch_audio_end,
        # ASR / MaAI / turn
        "asr_result": orchestrator.on_asr_result,
        "turn": runtime.handle_turn,
        "maai_result": runtime.handle_maai_result,
        "maai_nod_result": runtime.handle_maai_nod_result,
        "maai_bc_result": runtime.handle_maai_bc_result,
        # audio frames
        "audio_frame": lambda p: runtime.handle_audio_frame("audio_frame", p),
        "system_audio_frame": lambda p: runtime.handle_audio_frame("system_audio_frame", p),
        "raw_audio": runtime.handle_raw_audio,
        # orchestrator passthroughs
        "inference_state": lambda p: orchestrator.on_inference_state(bool(p.get("busy", False))),
        "force_finalize_status": runtime.handle_force_finalize_status,
        "force_finalize_outcome": runtime.handle_force_finalize_outcome,
        "llm_state": lambda p: orchestrator.on_llm_state(bool(p.get("in_progress", False))),
        "llm_finished": runtime.handle_llm_finished,
        "assistant_reply": lambda p: orchestrator.on_assistant_reply(
            str(p.get("text", "")), bool(p.get("ok", False)),
        ),
        "tts_audible": lambda _p: orchestrator.on_tts_audible(),
        "assistant_segment_played": lambda p: orchestrator.on_assistant_segment_played(
            str(p.get("cumulative_text", "")),
        ),
        "tts_done": lambda p: orchestrator.on_tts_done(
            playback_aborted=bool(p.get("playback_aborted", False)),
        ),
        "ping": lambda _p: emit({"type": "pong", "ts": time.time()}),
        # module lifecycle
        "start_module": runtime.handle_start_module,
        "stop_module": runtime.handle_stop_module,
        # worker relays
        "force_finalize": lambda _p: supervisor.asr.send_command("force_finalize"),
        "inject_system_audio": _h_inject_system_audio,
        "inject_user_audio": _h_inject_user_audio,
        "send_to_llm": _h_send_to_llm,
        "send_to_tts": _h_send_to_tts,
        "send_to_tts_bridge": _h_send_to_tts_bridge,
        "send_to_llm_admin": _h_send_to_llm_admin,
        "send_to_audio_output": _h_send_to_audio_output,
        "asr_send_command": _h_asr_send_command,
        "audio_input_send_command": _h_audio_input_send_command,
        # test RPCs
        "run_llm_test": runtime.handle_run_llm_test,
        "run_tts_test": runtime.handle_run_tts_test,
        "run_asr_test": runtime.handle_run_asr_test,
        "run_maai_test": runtime.handle_run_maai_test,
        # passthrough (Audio Input ワーカー → UI 転送。
        # WAV 範囲終了時の MaAI 自動停止 _on_wav_playback_finished_auto に必須)
        "wav_playback_finished": _passthrough_emit,
        "wav_command_done": _passthrough_emit,
        "wav_monitor_audio": _passthrough_emit,
        "processing_started": _passthrough_emit,
        "processing_stopped": _passthrough_emit,
    }

    def _handle_event(payload: dict[str, object]) -> None:
        # 新モード: LLM ワーカーからのイベントは別ルーティング
        src = str(payload.get("_src", "")).strip().lower()
        if src == "llm":
            runtime.handle_llm_worker_event(payload)
            return
        if src == "audio_output":
            et_audio = str(payload.get("type", "")).strip().lower()
            if et_audio == "ready":
                emit(
                    {
                        "type": "ready",
                        "worker": "audio_output",
                        "output_mode": payload.get("output_mode", "local"),
                    }
                )
                return
            if et_audio == "tts_mmd_send_complete":
                comp = {k: v for k, v in payload.items() if k != "_src"}
                runtime.on_mmd_send_complete(comp)
                return
            if et_audio == "tts_maai_pcm_frame":
                b64 = payload.get("audio_b64")
                if isinstance(b64, str) and b64:
                    try:
                        supervisor.send_audio_input_command(f"system_audio:{b64}")
                    except Exception:
                        pass
                    else:
                        runtime.notify_maai_system_inject_activity()
                return
            if et_audio == "remote_status":
                origin = str(payload.get("_remote_bridge_origin", "") or "").strip().lower()
                if origin == "audio_output":
                    mmd.on_remote_status_event(payload)
                    # MmdRuntime と同じ接続イベントを UI にも渡す。Web/GUI の表示専用で、
                    # SND・頷きの制御経路は従来どおり MmdRuntime / AudioOutput のまま。
                    emit({k: v for k, v in payload.items() if k != "_src"})
                return
            return

        cmd = str(payload.get("type", "")).strip().lower()
        # ワーカー(audio_input 等)のライフサイクル/ログ系イベントは DISPATCH コマンドではない。
        # - log/error: Controller へ転送して可視化する（audio_input のログが従来は捨てられていた）。
        # - ready/stopped: 各サーバの on_event 側で処理済み or 無害なので握りつぶす。
        #   特に audio_input の bare 'ready' を素通しで Controller へ送ると orchestrator-ready と
        #   誤認され audio I/O 再同期が走るため、ここで必ず止める。
        if cmd in ("log", "error"):
            emit(dict(payload))
            return
        if cmd == "fatal":
            # MaAI などイベントサーバ経由のワーカーの致命的エラー。以前は DISPATCH に無く
            # 理由ごと捨てていたため、UI では「起動中」のまま止まって見えた。
            message = str(payload.get("message", "") or "worker reported a fatal error")
            origin = src or str(payload.get("worker", "") or "worker")
            emit({"type": "log", "message": f"[{origin}][FATAL] {message}"})
            module = origin.replace("_worker", "").replace("worker", "").strip(" _-")
            if module in {"asr", "llm", "tts", "maai", "audio_input", "audio_output"}:
                emit(
                    {
                        "type": "module_state",
                        "module": module,
                        "state": "error",
                        "source": "worker_fatal",
                        "error": message,
                    }
                )
            emit(dict(payload))
            return
        if cmd in ("ready", "stopped"):
            return
        handler = DISPATCH.get(cmd)
        if handler is not None:
            handler(payload)  # type: ignore[operator]
        elif cmd and cmd not in _WARNED_UNKNOWN_EVENT_TYPES:
            # 未知タイプは黙って捨てず、各タイプ初回だけ警告（プロトコル不一致の可視化）。
            _WARNED_UNKNOWN_EVENT_TYPES.add(cmd)
            print(
                f"[Orchestrator][WARN] unknown event type (no DISPATCH handler): {cmd!r}",
                file=sys.stderr,
                flush=True,
            )

    # -- Server ingress / event ingestion ----------------------------------------

    def _on_event(payload: dict[str, object]) -> None:
        event_queue.put(dict(payload or {}))

    def _on_audio_input_event(payload: dict[str, object]) -> None:
        p = dict(payload or {})
        event_type = str(p.get("type", "")).strip().lower()
        if event_type == "ready":
            supervisor.mark_audio_input_ready()
        elif event_type == "fatal":
            supervisor.mark_audio_input_fatal(str(p.get("message", "")))
            p["_src"] = "audio_input"
        event_queue.put(p)

    audio_input_server.start(on_event=_on_audio_input_event)
    asr_audio_server.start(on_event=lambda p: runtime.forward_asr_ingress_event(p, mark_ready=True, mark_compare_ready=False))
    asr_compare_audio_server.start(on_event=lambda p: runtime.forward_asr_ingress_event(p, mark_ready=False, mark_compare_ready=True))
    maai_audio_server.start(on_event=runtime.on_maai_ingress_event)

    # Audio Input / Output のみ先に起動し、MMDAgent が remote 接続できる状態にする（音声対話開始を待たない）。
    try:
        from pathlib import Path

        from ..config_store import ConfigStore

        # Bootstrap は `start_voice_dialog` より前に走るので設定ファイルを直読みする。
        # 以前は常に自分の隣の `controller_config.json` 固定で、`headless_main.py
        # --config other.json` を指定しても audio I/O だけ別設定で立ち上がっていた（F5）。
        # `--config` を受け取れるようにし、無指定なら従来どおり隣のファイルを使う。
        _orch_cfg_path = _resolve_config_path()
        runtime.bootstrap_audio_io_workers_early(
            ConfigStore(_orch_cfg_path).load(),
            assume_mmd_running=_assume_mmd_running(),
        )
    except Exception as ex:
        emit({"type": "log", "message": f"[Bootstrap] 先行起動をスキップ: {ex}"})

    threading.Thread(target=_stdin_watcher, daemon=True, name="dialog-orch-stdin").start()
    # stdin の EOF だけでは足りない。上の watcher は「起動 5 秒以内に EOF」を
    # stdin 未接続と見なして自分を無効化するため、GUI がクラッシュした場合に
    # 親の死を拾えない。実際にこれで orchestrator と audio_output が 18 組
    # 生き残り、CPU とメモリを食って MaAI が実時間に追いつかなくなった。
    # PID の生存を直接見る監視を併走させる。stop_event 経由なので stop_all が
    # 走り、子ワーカーも連れて落ちる。
    if not listen_mode:
        start_parent_death_watchdog(
            stop_event.set, worker_name="Orchestrator",
        )

    def _send_ui_connection_snapshot() -> None:
        """listen モード専用: UI がつながるたびに ready と現在の状態を送り直す。"""
        emit({"type": "ready"})
        active = bool(orchestrator.state.active)
        emit(
            {
                "type": "status",
                "state": "running" if active else "stopped",
                "message": "音声対話：実行中" if active else "音声対話：停止",
            }
        )
        emit({"type": "history_snapshot", "history": orchestrator.history_snapshot()})
        for event in supervisor.ui_state_snapshot():
            emit(event)

    ui_server.start(
        on_event=_on_event,
        on_disconnected=lambda: None,
        # listen モード（Docker の Qt 接続）は Qt の再起動で何度もつながり直すので、
        # 接続のたびに状態を送り直す。子プロセスとして起動された通常モードは従来どおり。
        on_connected=_send_ui_connection_snapshot if listen_mode else None,
    )

    # 通常モード: ready は「現在生きている接続」に対して一度だけ送る。Controller は ready 受信ごとに
    # audio I/O の再同期（ensure_*）を走らせるため、起動中の再接続 churn のたびに送ると
    # ワーカーが多重起動する。接続確立の最初の accept ではなく、安定した接続を都度確認して送る。
    def _send_ready_on_connect() -> None:
        deadline = time.time() + 60.0
        while time.time() < deadline and not stop_event.is_set():
            if ui_server.is_connected():
                emit({"type": "ready"})
                return
            time.sleep(0.1)

    if not listen_mode:
        threading.Thread(target=_send_ready_on_connect, daemon=True, name="ready-wait").start()

    # -- Main loop ---------------------------------------------------------------

    try:
        next_tick = time.monotonic()
        while not stop_event.is_set():
            now_mono = time.monotonic()
            if now_mono >= next_tick:
                orchestrator.on_tick()
                next_tick = now_mono + 0.05
            wait_sec = max(0.0, next_tick - time.monotonic())
            try:
                payload = event_queue.get(timeout=min(0.05, wait_sec if wait_sec > 0.0 else 0.01))
            except queue.Empty:
                continue
            _handle_event(payload)
            while True:
                try:
                    payload = event_queue.get_nowait()
                except queue.Empty:
                    break
                _handle_event(payload)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        mmd.cancel_nod_motion()
        supervisor.stop_all()
        for srv in (audio_input_server, asr_audio_server, asr_compare_audio_server, maai_audio_server):
            try:
                srv.stop()
            except Exception:
                pass
        try:
            emit({"type": "stopped"})
        except BaseException:
            pass
        try:
            ui_server.stop()
        except BaseException:
            pass
        import os as _os
        _os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
