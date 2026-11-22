import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .. import paths as _paths
from .worker_runtime import PARENT_PID_ENV as _PARENT_PID_ENV

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")


def _worker_module_args(worker_script: Path) -> list[str]:
    """ワーカーを `-m maai_sds.<...>` 形式で起動するための引数。"""
    return ["-m", _paths.package_module_name(worker_script)]


def _infer_worker_console_label(svc: "WorkerProcessService") -> str:
    name = type(svc).__name__
    for suffix in ("WorkerProcessService", "ProcessService"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _attach_worker_process_console_output(proc: subprocess.Popen, label: str) -> None:
    """子プロセスの PIPE 出力を親の stdout/stderr に行単位で流す（Orchestrator 配下のワーカー用）。"""

    def _pump(stream, stream_tag: str) -> None:
        if stream is None:
            return
        prefix = f"[{label}][{stream_tag}] "
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\r\n")
                # ONNX Runtime writes UTF-16 LE via Win32 API; read as UTF-8
                # this produces null bytes between every character. Strip them
                # so null-only lines (the stray \x00 after each warning) are
                # suppressed and remaining text is readable.
                line = line.replace("\x00", "")
                if line and _ANSI_RE.sub("", line).strip():
                    if stream_tag == "STDERR":
                        print(prefix + line, file=sys.stderr, flush=True)
                    else:
                        print(prefix + line, file=sys.stdout, flush=True)
        except Exception:
            pass

    if proc.stdout is not None:
        threading.Thread(
            target=_pump,
            args=(proc.stdout, "STDOUT"),
            daemon=True,
            name=f"{label}-stdout",
        ).start()
    if proc.stderr is not None:
        threading.Thread(
            target=_pump,
            args=(proc.stderr, "STDERR"),
            daemon=True,
            name=f"{label}-stderr",
        ).start()


def _drain_worker_process_output_silently(proc: subprocess.Popen) -> None:
    """PIPE を必ず読み捨ててパイプバッファ満杯による子プロセスのデッドロックを防ぐ。

    stdout/stderr=PIPE で起動したが出力を親で利用しないワーカー用。誰も readline
    しないと OS のパイプバッファ（Windows で数十 KB）が埋まった時点で子の write が
    ブロックしハングするため、コンソール転送しない場合でも必ず排出する。"""

    def _drain(stream) -> None:
        if stream is None:
            return
        try:
            for _ in iter(stream.readline, ""):
                pass
        except Exception:
            pass

    for stream, tag in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        if stream is not None:
            threading.Thread(
                target=_drain,
                args=(stream,),
                daemon=True,
                name=f"worker-drain-{tag}",
            ).start()



class _WorkerSubprocessStaleGuard:
    """start 直前に、生存中の旧 Popen を stop し、終了済みなら参照のみクリアする。"""

    _lock: threading.Lock
    process: subprocess.Popen | None

    def _reap_stale_before_start(
        self,
        *,
        on_wait: Callable[[], None] | None = None,
        timeout_sec: int = 3,
    ) -> None:
        with self._lock:
            stale = self.process
        if stale is None:
            return
        if stale.poll() is None:
            self.stop(timeout_sec=timeout_sec, on_wait=on_wait)
        else:
            with self._lock:
                if self.process is stale:
                    self.process = None


class MMDProcessService:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, exec_path: Path, mdf_path: Path, env: dict) -> subprocess.Popen:
        if self.process is not None:
            if self.process.poll() is None:
                self.stop(timeout_sec=3)
            else:
                self.process = None
        self.process = subprocess.Popen(
            [str(exec_path), str(mdf_path)],
            cwd=str(mdf_path.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return self.process

    def stop(self, timeout_sec: int = 3) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = None
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout_sec)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = None


class WorkerProcessService(_WorkerSubprocessStaleGuard):
    """共通ワーカープロセス管理。個別 ProcessService クラスの基底として使う。"""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self.process is not None and self.process.poll() is None

    def _start_process(
        self,
        cmd: list[str],
        cwd: str | Path,
        *,
        on_wait: Callable[[], None] | None = None,
        forward_console: bool = False,
        console_label: str | None = None,
        drain_silently: bool = True,
    ) -> subprocess.Popen:
        self._reap_stale_before_start(on_wait=on_wait)
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        # 起動した側の PID を渡す。ワーカーはこれが消えたら自分で終了する
        # （stdin の EOF だけでは、GUI がクラッシュしたときに orchestrator が
        # 生き残って溜まった）。setdefault では祖父の値を継いでしまうので代入する。
        env[_PARENT_PID_ENV] = str(os.getpid())
        with self._lock:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            proc = self.process
        if proc is not None:
            if forward_console:
                label = console_label if console_label is not None else _infer_worker_console_label(self)
                _attach_worker_process_console_output(proc, label)
            elif drain_silently:
                # PIPE を開いた以上、コンソール転送しない場合でも必ず排出する
                # （親が stdout を自前で読むワーカーは drain_silently=False を渡す）。
                _drain_worker_process_output_silently(proc)
        return proc

    def send_command(self, command: str) -> None:
        with self._lock:
            proc = self.process
            if proc is None or proc.stdin is None or proc.poll() is not None:
                return
            try:
                proc.stdin.write(command + "\n")
                proc.stdin.flush()
            except Exception:
                pass

    def stop(self, timeout_sec: int = 3, on_wait: Callable[[], None] | None = None) -> None:
        with self._lock:
            proc = self.process
            if proc is None:
                return
            if proc.poll() is not None:
                self.process = None
                return
        self.send_command("stop")
        deadline = time.time() + max(0.1, float(timeout_sec))
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if on_wait is not None:
                try:
                    on_wait()
                except Exception:
                    pass
            time.sleep(0.05)
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        with self._lock:
            if self.process is proc:
                self.process = None

    def start(
        self,
        python_exe: Path,
        worker_script: Path,
        host: str,
        port: int,
        config_json: str,
        *,
        forward_console: bool = False,
        on_wait: Callable[[], None] | None = None,
    ) -> subprocess.Popen:
        """`python -u SCRIPT --host --port --config-json` 型ワーカーの共通起動。

        固有の引数を持つワーカー（TTS=--engine, LLM=extra_args など）は start() を上書きする。
        """
        cmd = [
            str(python_exe), "-u", *_worker_module_args(worker_script),
            "--host", str(host), "--port", str(int(port)),
            "--config-json", config_json,
        ]
        return self._start_process(
            cmd, _paths.REPO_ROOT, on_wait=on_wait, forward_console=forward_console
        )


class ASRWorkerProcessService(WorkerProcessService):
    pass


class AudioInputWorkerProcessService(WorkerProcessService):
    pass


class AudioOutputWorkerProcessService(WorkerProcessService):
    """Audio output worker (MMD / local / TCP burst pacing)."""


class TTSWorkerProcessService(WorkerProcessService):
    def start(
        self,
        python_exe: Path,
        worker_script: Path,
        host: str,
        port: int,
        engine: str,
        config_json: str,
        *,
        forward_console: bool = False,
        on_wait: Callable[[], None] | None = None,
    ) -> subprocess.Popen:
        cmd = [
            str(python_exe), "-u", *_worker_module_args(worker_script),
            "--host", str(host), "--port", str(int(port)),
            "--engine", str(engine), "--config-json", config_json,
        ]
        return self._start_process(
            cmd, _paths.REPO_ROOT, on_wait=on_wait, forward_console=forward_console
        )


class LLMWorkerProcessService(WorkerProcessService):
    def start(
        self,
        python_exe: Path,
        worker_script: Path,
        host: str,
        port: int,
        *,
        forward_console: bool = False,
        on_wait: Callable[[], None] | None = None,
        extra_args: list[str] | None = None,
    ) -> subprocess.Popen:
        cmd = [
            str(python_exe), "-u", *_worker_module_args(worker_script),
            "--host", str(host), "--port", str(int(port)),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return self._start_process(
            cmd, _paths.REPO_ROOT, on_wait=on_wait, forward_console=forward_console
        )


class DialogOrchestratorProcessService(WorkerProcessService):
    def start(
        self,
        python_exe: Path,
        worker_script: Path,
        *,
        on_wait: Callable[[], None] | None = None,
    ) -> subprocess.Popen:
        # -u: パイプ先でも ui_port 行を即座に flush し、親の readline と競合しにくくする。
        # stdout は親（Controller）が ui_port 取得→_attach_process_console_output で
        # 専有読取するため、ここでの自動排出は無効化する（二重読取で ui_port を取り逃す）。
        cmd = [str(python_exe), "-u", *_worker_module_args(worker_script)]
        return self._start_process(
            cmd, _paths.REPO_ROOT, on_wait=on_wait, drain_silently=False
        )


class MaAIWorkerProcessService(WorkerProcessService):
    pass
