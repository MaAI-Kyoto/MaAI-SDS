"""Qt を使わずに音声対話コアを起動するヘッドレスエントリーポイント。

Usage:
    python headless_main.py [--config controller_config.json]
                            [--engine VOICEVOX-Core]
                            [--override key=value ...]
                            [--no-start]
                            [voice_dialog | stop | status | run_llm_test ...]

`controller_config.json` は Controller GUI が保存するのと同じ JSON（dict）形式。
このスクリプトは下記の処理を順に行う:

1. `dialog/worker.py` を `python -u -m maai_sds.dialog.worker` で別プロセス起動
2. stdout の `{"ui_port": <port>}` を読んで `JsonLineDuplexClient` で接続
3. `dialog/settings.build_start_voice_dialog_payload(cfg)` で設定変換
4. `start_voice_dialog` を送信し、`status: running` を待つ
5. SIGINT/SIGTERM を受けたら `stop_voice_dialog` を送って `status: stopped` を待ち終了
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config_store import ConfigStore  # type: ignore
from .ipc.jsonline import JsonLineDuplexClient  # type: ignore
from .dialog import settings as vds

from . import paths as _paths


# ワーカースクリプトは `.py` が並ぶ場所にある（パッケージ化してもここを指す）。
_DEFAULT_WORKER_SCRIPT = _paths.worker_script("dialog", "worker.py")


def _parse_override(raw: str) -> tuple[str, Any]:
    """`--override key=value` を `(key, value)` に変換。値は JSON で解釈し、失敗したら文字列。"""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--override の書式は key=value: {raw!r}")
    key, _, value = raw.partition("=")
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"--override の key が空: {raw!r}")
    text = value.strip()
    try:
        return key, json.loads(text)
    except json.JSONDecodeError:
        return key, text


def _apply_overrides(cfg: dict, overrides: list[tuple[str, Any]]) -> dict:
    out = dict(cfg)
    for key, value in overrides:
        out[key] = value
    return out


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MaAI-SDS headless voice dialog runner",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="voice_dialog",
        choices=["voice_dialog", "stop", "status", "run_llm_test", "run_tts_test", "run_asr_test", "run_maai_test"],
        help="サブコマンド (default: voice_dialog)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="controller_config.json",
        help="GUI 互換の設定 JSON パス (default: controller_config.json)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="TTS エンジン上書き (例: VOICEVOX, VOICEVOX-Core, Kokoro-RealtimeTTS...)",
    )
    parser.add_argument(
        "--override",
        type=_parse_override,
        action="append",
        default=[],
        help="設定の上書き. `--override key=value` (value は JSON)。複数指定可。",
    )
    parser.add_argument(
        "--worker-script",
        type=str,
        default=str(_DEFAULT_WORKER_SCRIPT),
        help="orchestrator worker のスクリプトパス",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="既存 worker のポートに接続する。0 のとき新たに worker を起動。",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="`start_voice_dialog` を送らずに接続だけ確立する (デバッグ用)。",
    )
    parser.add_argument(
        "--assume-mmd-running",
        action="store_true",
        help="MMD TCP server is already running; preserve audio_output_mode=mmd.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=20.0,
        help="`status: running` を待つ秒数 (default: 20)",
    )
    parser.add_argument(
        "--wav-play-delay",
        type=float,
        default=0.0,
        help="WAV 入力のとき、対話が running になってから再生を始めるまでの秒数（開始の挨拶を待つときなど）",
    )
    parser.add_argument(
        "--no-wav-play",
        action="store_true",
        help="WAV 入力でも再生を始めない（GUI の再生ボタンに当たる wav_play を送らない）",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default="こんにちは",
        help="run_llm_test / run_tts_test 用の入力テキスト",
    )
    return parser.parse_args()


def _spawn_worker(
    worker_script: Path,
    config_path: str | None = None,
    *,
    assume_mmd_running: bool = False,
) -> tuple[subprocess.Popen, int]:
    """Worker を起動して `ui_port` を取得する。

    `config_path` を渡すと worker の Bootstrap（audio I/O 先行起動）も同じ設定を読む。
    渡さないと worker は自分の隣の `controller_config.json` を使うため、
    `--config` を指定しても audio I/O だけ別設定になっていた（F5）。
    `assume_mmd_running` は Bootstrap の audio_output にも渡す。渡さないと
    保存が mmd でも local で先行起動し、後から start_module で取り直すことになる。
    """
    # `-m maai_sds.<name>` で起動する（パス起動だとパッケージ相対 import が使えない）。
    cmd = [sys.executable, "-u", "-m", _paths.package_module_name(worker_script)]
    if config_path:
        cmd += ["--config", str(config_path)]
    if assume_mmd_running:
        cmd.append("--assume-mmd-running")
    creationflags = 0
    try:
        # Windows: 親プロセスの Ctrl+C は送らないようにする（自前で stop を送る）。
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    except Exception:
        creationflags = 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(_paths.REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    if proc.stdout is None:
        raise RuntimeError("worker stdout が取得できませんでした。")

    ui_port = 0
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = (proc.stdout.read() or "").strip()[-1024:]
            except Exception:
                pass
            raise RuntimeError(f"worker が起動直後に終了しました: rc={proc.returncode} tail={tail!r}")
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        line = line.strip()
        try:
            info = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            print(f"[worker] {line}", flush=True)
            continue
        if isinstance(info, dict) and "ui_port" in info:
            ui_port = int(info.get("ui_port", 0) or 0)
            break
        print(f"[worker] {json.dumps(info, ensure_ascii=False)}", flush=True)

    if ui_port <= 0:
        raise RuntimeError("Dialog Orchestrator の ui_port を取得できませんでした。")
    return proc, ui_port


def _start_console_pump(proc: subprocess.Popen, prefix: str = "[worker]") -> threading.Thread:
    """worker の stdout を別スレッドでミラーする。"""
    def _pump() -> None:
        if proc.stdout is None:
            return
        for raw in proc.stdout:
            try:
                print(f"{prefix} {raw.rstrip()}", flush=True)
            except Exception:
                pass

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return t


class HeadlessClient:
    def __init__(self, host: str, port: int) -> None:
        self.client = JsonLineDuplexClient(host, port)
        self._running = threading.Event()
        self._stopped = threading.Event()
        self._dialog_state = "idle"
        self._lock = threading.Lock()
        self._llm_done = threading.Event()
        self._llm_text = ""
        self._llm_error = ""
        self._llm_request_id = ""
        self._module_ready_events: dict[str, threading.Event] = {}
        self._module_start_errors: dict[str, str] = {}

    def connect(self, attempts: int = 200, sleep_sec: float = 0.05) -> None:
        self.client.connect(attempts=attempts, sleep_sec=sleep_sec)

    def start_reader(self) -> None:
        self.client.start_reader(
            on_event=self._on_event,
            on_disconnected=lambda: self._stopped.set(),
        )

    def _on_event(self, payload_obj: object) -> None:
        if not isinstance(payload_obj, dict):
            return
        et = str(payload_obj.get("type", "")).strip().lower()
        if et == "status":
            state = str(payload_obj.get("state", "")).strip().lower()
            with self._lock:
                self._dialog_state = state or self._dialog_state
            if state == "running":
                self._running.set()
            elif state == "stopped":
                self._stopped.set()
        elif et == "log":
            msg = str(payload_obj.get("message", ""))
            if msg:
                print(f"[orch] {msg}", flush=True)
        elif et == "error":
            msg = str(payload_obj.get("message", ""))
            print(f"[orch:error] {msg}", file=sys.stderr, flush=True)
        elif et == "asr_result":
            # 確定した認識結果だけを表示する（途中経過は量が多い）
            if bool(payload_obj.get("is_final")):
                text = str(payload_obj.get("text", "") or "").strip()
                if text:
                    print(f"[asr] {text}", flush=True)
        elif et == "llm_done":
            reply = str(payload_obj.get("text", "") or "").strip()
            if reply:
                print(f"[llm] {reply}", flush=True)
            rid = str(payload_obj.get("request_id", "")).strip()
            if not self._llm_request_id or rid == self._llm_request_id:
                self._llm_text = str(payload_obj.get("text", "") or "")
                self._llm_done.set()
        elif et == "llm_error":
            self._llm_error = str(payload_obj.get("message", "") or "")
            self._llm_done.set()
        elif et == "module_started":
            module = str(payload_obj.get("module", "")).strip()
            if module:
                self._module_start_errors.pop(module, None)
                self._module_ready_events.setdefault(module, threading.Event()).set()
        elif et == "module_start_failed":
            module = str(payload_obj.get("module", "")).strip()
            if module:
                self._module_start_errors[module] = str(payload_obj.get("error", "") or "")
                self._module_ready_events.setdefault(module, threading.Event()).set()

    def send(self, payload: dict) -> None:
        self.client.send(dict(payload))

    def wait_running(self, timeout: float) -> bool:
        return self._running.wait(timeout=timeout)

    def wait_stopped(self, timeout: float) -> bool:
        return self._stopped.wait(timeout=timeout)

    def wait_llm_done(self, timeout: float) -> bool:
        return self._llm_done.wait(timeout=timeout)

    def wait_module_started(self, module: str, timeout: float) -> bool:
        ev = self._module_ready_events.setdefault(module, threading.Event())
        ok = ev.wait(timeout=timeout)
        ev.clear()
        return ok and module not in self._module_start_errors

    def module_start_error(self, module: str) -> str:
        return self._module_start_errors.get(module, "")

    @property
    def state(self) -> str:
        with self._lock:
            return self._dialog_state

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def _command_voice_dialog(
    cfg: dict,
    *,
    engine: str | None,
    headless: HeadlessClient,
    ready_timeout: float,
    no_start: bool,
    assume_mmd_running: bool,
    wav_play: bool = True,
    wav_play_delay: float = 0.0,
) -> int:
    if no_start:
        print("[headless] connect-only mode (no start_voice_dialog).", flush=True)
        try:
            while not headless.wait_stopped(timeout=1.0):
                pass
        except KeyboardInterrupt:
            pass
        return 0

    # Bootstrap にも `--assume-mmd-running` を渡すようにしたので先行起動は mmd のまま。
    # それでも `start_module` で同じ設定を送り直す。digest が一致していれば再起動せず、
    # 既存 worker へ `--port` 接続したとき（Bootstrap を経由しない）の穴を塞ぐ。
    try:
        audio_output_cfg = vds.build_audio_output_worker_config_dict(
            cfg, assume_mmd_running=assume_mmd_running,
        )
        headless.send({
            "type": "start_module",
            "module": "audio_output",
            "config": audio_output_cfg,
            "source": "headless",
        })
        if not headless.wait_module_started("audio_output", timeout=30.0):
            err = headless.module_start_error("audio_output")
            print(
                f"[headless] audio_output 設定同期に失敗またはタイムアウト: {err or '(timeout)'}",
                file=sys.stderr,
                flush=True,
            )
    except Exception as ex:
        print(f"[headless] audio_output 設定同期をスキップ: {ex}", file=sys.stderr, flush=True)

    payload = {
        "type": "start_voice_dialog",
        "settings": vds.build_start_voice_dialog_payload(
            cfg, tts_engine=engine, assume_mmd_running=assume_mmd_running,
        ),
    }
    print(
        f"[headless] start_voice_dialog tts_engine={payload['settings'].get('tts_engine')!r}",
        flush=True,
    )
    headless.send(payload)
    if not headless.wait_running(timeout=ready_timeout):
        print(
            f"[headless] status: running を {ready_timeout:.1f} 秒以内に確認できませんでした。",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print("[headless] dialog state = running. Ctrl+C で停止。", flush=True)
    try:
        # WAV 入力は一時停止の状態で始まり、GUI では再生ボタンが wav_play を送って初めて流れる。
        # ヘッドレスには再生ボタンが無いので、ここで送る（送らないと ASR に無音しか届かない）。
        if wav_play and str(cfg.get("audio_input_mode", "") or "").strip().lower() == "wav":
            if wav_play_delay <= 0 or not headless.wait_stopped(timeout=wav_play_delay):
                print("[headless] WAV 入力の再生を始めます（wav_play）", flush=True)
                headless.send({"type": "audio_input_send_command", "command": "wav_play"})
        while True:
            if headless.wait_stopped(timeout=0.5):
                break
    except KeyboardInterrupt:
        print("\n[headless] SIGINT received. stopping...", flush=True)
        headless.send({"type": "stop_voice_dialog"})
        headless.wait_stopped(timeout=15.0)
    print(f"[headless] dialog state = {headless.state}", flush=True)
    return 0


def _command_stop(headless: HeadlessClient) -> int:
    headless.send({"type": "stop_voice_dialog"})
    headless.wait_stopped(timeout=15.0)
    return 0


def _command_status(headless: HeadlessClient) -> int:
    print(f"[headless] dialog state = {headless.state}", flush=True)
    return 0


def _command_run_llm_test(
    cfg: dict, headless: HeadlessClient, *, input_text: str, timeout_sec: float = 60.0
) -> int:
    rid = f"headless-llm-{int(time.time() * 1000)}"
    headless._llm_request_id = rid  # type: ignore[attr-defined]
    headless._llm_done.clear()
    payload = {
        "type": "run_llm_test",
        "request_id": rid,
        "input_text": input_text,
        "settings": vds.build_dialog_orchestrator_settings(cfg),
        "llm_connection": vds.active_llm_connection_settings(cfg),
    }
    headless.send(payload)
    print(f"[headless] run_llm_test rid={rid}", flush=True)
    if not headless.wait_llm_done(timeout=timeout_sec):
        print("[headless] run_llm_test timeout", file=sys.stderr, flush=True)
        return 3
    if headless._llm_error:  # type: ignore[attr-defined]
        print(f"[headless] llm_error: {headless._llm_error}", file=sys.stderr, flush=True)  # type: ignore[attr-defined]
        return 4
    print(f"[llm] {headless._llm_text}")  # type: ignore[attr-defined]
    return 0


def _command_run_tts_test(
    cfg: dict,
    headless: HeadlessClient,
    *,
    engine: str | None,
    input_text: str,
) -> int:
    eng = (engine or vds.cfg_str(cfg, "tts_engine", "VOICEVOX") or "VOICEVOX").strip()
    rid = f"headless-tts-{int(time.time() * 1000)}"
    inputs = vds.build_tts_inputs(cfg, engine=eng, utterance=input_text)
    headless.send(
        {"type": "run_tts_test", "request_id": rid, "engine": eng, "inputs": inputs}
    )
    print(f"[headless] run_tts_test rid={rid} engine={eng!r}", flush=True)
    # 合成完了の専用イベントが headless へ配線されていないため、終了処理（main の finally で
    # ワーカーを terminate）が合成前に走らないよう猶予を置く。
    # 注意: このサブコマンドは `voice_dialog` で TTS ワーカーが既に起動・接続済みであることを
    # 前提にしている（`runtime.handle_run_tts_test` は未接続なら `tts_error` を emit するだけで
    # 何も合成しない。headless の CLI 側では emit イベントを表示しないため、単独実行すると
    # 無音のまま何もせず終了する）。単独プロセスとして cold start から叩く用途には向かない。
    # 10秒は TTS ワーカーが接続済みの場合の合成時間の目安として置かれていたが、コンテナでの
    # 検証時にモデル読み込みに余裕を持たせる目的で 30 秒に伸ばした。
    time.sleep(30.0)
    return 0


def _command_run_asr_test(headless: HeadlessClient, *, command: str = "force_finalize") -> int:
    rid = f"headless-asr-{int(time.time() * 1000)}"
    headless.send({"type": "run_asr_test", "request_id": rid, "command": command})
    print(f"[headless] run_asr_test rid={rid} command={command!r}", flush=True)
    # force_finalize の結果が出る前にワーカーが terminate されないよう猶予を置く。
    time.sleep(3.0)
    return 0


def _command_run_maai_test(cfg: dict, headless: HeadlessClient) -> int:
    rid = f"headless-maai-{int(time.time() * 1000)}"
    settings = vds.build_dialog_orchestrator_settings(cfg)
    headless.send({"type": "run_maai_test", "request_id": rid, "settings": settings})
    print(f"[headless] run_maai_test rid={rid}", flush=True)
    # noop に近い処理だが、ログ送出がワーカー terminate より先に届くよう短い猶予を置く。
    time.sleep(0.5)
    return 0


def _install_signal_handlers(headless: HeadlessClient) -> None:
    """SIGINT/SIGTERM で stop_voice_dialog を送る。"""
    def _handler(_sig, _frame):
        try:
            print(f"\n[headless] signal {_sig} received; sending stop_voice_dialog...", flush=True)
            headless.send({"type": "stop_voice_dialog"})
        except Exception:
            pass
        # 2 度目以降の Ctrl+C は KeyboardInterrupt にする
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        pass
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):
            pass


def main() -> int:
    args = _build_args()

    config_path = Path(args.config)
    store = ConfigStore(config_path)
    cfg = store.load() or {}
    if not cfg:
        print(f"[headless] config not found or empty: {config_path}", file=sys.stderr)

    if args.override:
        cfg = _apply_overrides(cfg, args.override)
    if args.engine:
        cfg["tts_engine"] = args.engine

    proc: subprocess.Popen | None = None
    if args.port and int(args.port) > 0:
        port = int(args.port)
        print(f"[headless] connecting to existing worker at 127.0.0.1:{port}", flush=True)
    else:
        worker_script = Path(args.worker_script)
        if not worker_script.exists():
            print(f"[headless] worker script not found: {worker_script}", file=sys.stderr)
            return 1
        print(f"[headless] launching worker: {worker_script}", flush=True)
        try:
            proc, port = _spawn_worker(
                worker_script,
                config_path=args.config,
                assume_mmd_running=bool(args.assume_mmd_running),
            )
        except Exception as ex:
            print(f"[headless] worker spawn failed: {ex}", file=sys.stderr, flush=True)
            return 1
        _start_console_pump(proc)
        print(f"[headless] worker ui_port = {port}", flush=True)

    headless = HeadlessClient("127.0.0.1", port)
    try:
        headless.connect(attempts=400, sleep_sec=0.05)
    except Exception as ex:
        print(f"[headless] connect failed: {ex}", file=sys.stderr, flush=True)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        return 1
    headless.start_reader()
    _install_signal_handlers(headless)

    rc = 0
    try:
        cmd = (args.command or "voice_dialog").strip().lower()
        if cmd == "voice_dialog":
            rc = _command_voice_dialog(
                cfg,
                engine=args.engine or cfg.get("tts_engine"),
                headless=headless,
                ready_timeout=float(args.ready_timeout or 20.0),
                no_start=bool(args.no_start),
                assume_mmd_running=bool(args.assume_mmd_running),
                wav_play=not bool(args.no_wav_play),
                wav_play_delay=float(args.wav_play_delay or 0.0),
            )
        elif cmd == "stop":
            rc = _command_stop(headless)
        elif cmd == "status":
            rc = _command_status(headless)
        elif cmd == "run_llm_test":
            rc = _command_run_llm_test(cfg, headless, input_text=args.input_text)
        elif cmd == "run_tts_test":
            rc = _command_run_tts_test(
                cfg, headless, engine=args.engine, input_text=args.input_text
            )
        elif cmd == "run_asr_test":
            rc = _command_run_asr_test(headless)
        elif cmd == "run_maai_test":
            rc = _command_run_maai_test(cfg, headless)
        else:
            print(f"[headless] command not implemented yet: {cmd}", file=sys.stderr, flush=True)
            rc = 5
    finally:
        try:
            headless.close()
        except Exception:
            pass
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
