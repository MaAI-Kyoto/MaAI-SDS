"""controller.py から切り出した mixin（Phase 3-2）。

**mixin にしているのが要点**。`self` は `MMDAgentController` のままなので、
メソッド本文を 1 文字も書き換えずに移せる（協調オブジェクトにすると self が
別物になり、ウィジェット参照を全部書き換える必要が出る）。
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from .. import paths as _paths
from ..asr import qwen_docker


def _require_docker_available() -> None:
    """Docker Desktop 未インストール/未起動を、PowerShell の exit=1 に埋もれさせず
    GUI のエラーポップアップにそのまま出せる形で先に検知する。"""
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker が見つかりません。Docker Desktop をインストールし、"
            "'docker' コマンドが PATH に通っていることを確認してください: "
            "https://www.docker.com/products/docker-desktop/"
        )
    try:
        cp = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as ex:
        raise RuntimeError(f"Docker デーモンの確認に失敗しました: {ex}") from ex
    if cp.returncode != 0:
        raise RuntimeError("Docker デーモンに接続できません。Docker Desktop を起動してから再試行してください。")


class QwenAsrServerMixin:
    """QwenAsrServerMixin: controller.py から移設したメソッド群。"""

    def _start_qwen_asr_server(
        self,
        model: str,
        gpu_memory_utilization: str,
        max_model_len: str,
        port: str,
        session_timeout_sec: str,
        chunk_sec: str,
        enforce_eager: bool,
    ) -> None:
        if getattr(self, "_remote_backend_enabled", False):
            # Docker の対話サーバー: 同じ compose の qwen3-asr サービスを使う（PC 側のスクリプトは使わない）
            self._start_qwen_asr_server_docker(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                session_timeout_sec=session_timeout_sec,
                chunk_sec=chunk_sec,
                enforce_eager=enforce_eager,
            )
            return
        _require_docker_available()
        served_model_name = self._derive_qwen_served_model_name(model)
        script = _paths.scripts_dir() / "start_qwen3_asr.ps1"
        if not script.exists():
            raise FileNotFoundError(f"Qwen ASR 起動スクリプトが見つかりません: {script}")
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Model",
            model,
            "-ServedModelName",
            served_model_name,
            "-GpuMemoryUtilization",
            gpu_memory_utilization,
            "-MaxModelLen",
            max_model_len,
            "-HostPort",
            port,
            "-SessionTimeoutSec",
            session_timeout_sec,
            "-ChunkSec",
            chunk_sec,
            "-EnforceEager",
            "1" if enforce_eager else "0",
        ]
        cp = subprocess.Popen(
            cmd,
            cwd=str(script.parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._attach_process_console_output("Qwen3-ASR-Start", cp)
        # 開始コマンドを送信した直後からコンテナログ追尾を開始する。
        self._start_qwen_asr_container_logs()
        rc = cp.wait()
        if rc != 0:
            raise RuntimeError(f"Qwen ASR 起動に失敗しました (exit={rc})")

    def _qwen_asr_docker_health_url(self) -> str:
        host = (getattr(self, "_remote_orchestrator_endpoint", None) or ("127.0.0.1", 0))[0] or "127.0.0.1"
        return f"http://{host}:{qwen_docker.HOST_PORT}/health"

    def _start_qwen_asr_server_docker(
        self,
        *,
        model: str,
        gpu_memory_utilization: str,
        max_model_len: str,
        session_timeout_sec: str,
        chunk_sec: str,
        enforce_eager: bool,
    ) -> None:
        """Docker の対話サーバーと同じ compose で qwen3-asr を起動し、/health が返るまで待つ。

        compose の場所を知らない（手で compose を起動した）ときは起動できないので、動いているかだけ確かめる。
        """
        url = self._qwen_asr_docker_health_url()
        target = self._dialog_server_docker_target()
        if target is None:
            if qwen_docker.health_ok(url):
                self._start_qwen_asr_container_logs()
                return
            raise RuntimeError(qwen_docker.START_HINT)
        if any(Path(f).name == "docker-compose.cpu.yml" for f in target.files):
            raise RuntimeError(
                "Qwen3-ASR（vLLM）は GPU が要ります。CPU 版の Docker（docker-compose.cpu.yml）では使えません。"
            )
        _require_docker_available()
        env = qwen_docker.compose_env(
            model=model,
            served_model_name=self._derive_qwen_served_model_name(model),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            session_timeout_sec=session_timeout_sec,
            chunk_sec=chunk_sec,
            enforce_eager=enforce_eager,
        )
        self._start_qwen_asr_container_logs()
        self.process_output_signal.emit(
            "[Qwen3-ASR-Container][INFO] docker compose で qwen3-asr を起動します（初回はイメージの取得に時間がかかります）"
        )
        # 初回はイメージ（数 GB）の取得があるので長めに待つ
        ok, output = target.run(
            "start", service=qwen_docker.SERVICE, profile=qwen_docker.PROFILE, env=env, timeout_sec=3600.0
        )
        if output:
            self.process_output_signal.emit(output)
        if not ok:
            raise RuntimeError(f"Qwen3-ASR のコンテナを起動できませんでした。\n{output}")
        qwen_docker.wait_health(url, on_progress=self.process_output_signal.emit)

    def _stop_qwen_asr_server(self) -> None:
        if getattr(self, "_remote_backend_enabled", False):
            target = self._dialog_server_docker_target()
            if target is None:
                self.process_output_signal.emit(
                    "[Qwen3-ASR-Container][INFO] compose の場所を知らないので、qwen3-asr のコンテナは止めません"
                )
                return
            ok, output = target.run("stop", service=qwen_docker.SERVICE, profile=qwen_docker.PROFILE)
            if output:
                self.process_output_signal.emit(output)
            if not ok:
                raise RuntimeError(f"Qwen3-ASR のコンテナを停止できませんでした。\n{output}")
            return
        script = _paths.scripts_dir() / "stop_qwen3_asr.ps1"
        if not script.exists():
            raise FileNotFoundError(f"Qwen ASR 停止スクリプトが見つかりません: {script}")
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ]
        cp = subprocess.run(
            cmd,
            cwd=str(script.parent.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        output = (cp.stdout or "").strip()
        err = (cp.stderr or "").strip()
        if output:
            self.process_output_signal.emit(output)
        if err:
            self.process_output_signal.emit(err)
        if cp.returncode != 0:
            raise RuntimeError(f"Qwen ASR 停止に失敗しました (exit={cp.returncode})")

    def _start_qwen_asr_container_logs(self) -> None:
        # 開始/停止をロックで直列化し、logs -f の多重起動を防ぐ。
        with self._qwen_asr_logs_lock:
            self._qwen_asr_logs_follow_enabled = True
            running = self._qwen_asr_logs_proc
            if running is not None:
                if running.poll() is None:
                    return
                self._qwen_asr_logs_proc = None
        project_root = _paths.REPO_ROOT
        # Docker の対話サーバーでは compose の qwen3-asr（container_name は maai-sds-qwen3-asr）を追う
        name = qwen_docker.CONTAINER_NAME if getattr(self, "_remote_backend_enabled", False) else "qwen3-asr"
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"$name='{name}'; "
            "while ($true) { docker inspect $name *> $null; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep -Milliseconds 500 }; docker logs --timestamps --tail 0 --follow $name",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as ex:
            self._asr_log(f"Qwen3-ASR コンテナログ追尾の開始に失敗: {ex}")
            return
        with self._qwen_asr_logs_lock:
            running = self._qwen_asr_logs_proc
            if running is not None and running.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                return
            self._qwen_asr_logs_proc = proc
        self._attach_process_console_output("Qwen3-ASR-Container", proc)

        def _watch_logs_exit() -> None:
            rc = proc.wait()
            should_retry = False
            with self._qwen_asr_logs_lock:
                if self._qwen_asr_logs_proc is proc:
                    self._qwen_asr_logs_proc = None
                # コンテナ再作成時は exit=0 で follow が終わることがあるため、
                # follow有効中は終了コードに関係なく再接続を試みる。
                should_retry = self._qwen_asr_logs_follow_enabled
            if rc != 0:
                self.process_output_signal.emit(
                    f"[Qwen3-ASR-Container][WARN] docker logs --follow が終了しました (exit={rc})"
                )
            if should_retry:
                # コンテナ未作成・再作成などで follow が終了した場合に再接続する。
                time.sleep(1.0)
                # sleep 中に _stop_qwen_asr_container_logs() が呼ばれていたら復活させない（TOCTOU回避）。
                with self._qwen_asr_logs_lock:
                    still_enabled = self._qwen_asr_logs_follow_enabled
                if still_enabled:
                    self._start_qwen_asr_container_logs()

        threading.Thread(target=_watch_logs_exit, daemon=True, name="qwen-asr-logs-watch").start()

    def _stop_qwen_asr_container_logs(self) -> None:
        with self._qwen_asr_logs_lock:
            self._qwen_asr_logs_follow_enabled = False
            proc = self._qwen_asr_logs_proc
            self._qwen_asr_logs_proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _start_qwen_asr_server_async(
        self,
        model: str,
        gpu_memory_utilization: str,
        max_model_len: str,
        port: str,
        session_timeout_sec: str,
        chunk_sec: str,
        enforce_eager: bool,
    ) -> None:
        self.process_output_signal.emit(
            "[Qwen3-ASR-Container][INFO] Qwen3-ASR コンテナ起動コマンドを実行中..."
        )

        def _worker() -> None:
            try:
                self._start_qwen_asr_server(
                    model=model,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    port=port,
                    session_timeout_sec=session_timeout_sec,
                    chunk_sec=chunk_sec,
                    enforce_eager=enforce_eager,
                )
                self.process_output_signal.emit("[Qwen3-ASR-Container][INFO] Qwen3-ASR コンテナ起動完了")
                self.qwen_asr_start_finished_signal.emit(True, "")
            except Exception as ex:
                self.qwen_asr_start_finished_signal.emit(False, str(ex))

        threading.Thread(target=_worker, daemon=True, name="qwen-asr-bootstrap").start()

    def _on_qwen_asr_start_finished(self, ok: bool, error_message: str) -> None:
        if not self._asr_starting:
            return
        if self._asr_stop_requested_during_start:
            self._asr_stop_requested_during_start = False
            self._asr_starting = False
            self._asr_ready = False
            self._stop_qwen_asr_container_logs()
            if self.asr_qwen_stop_container_check.isChecked():
                try:
                    self._stop_qwen_asr_server()
                    self.asr_result_signal.emit("[INFO] Qwen3-ASR コンテナを停止しました。")
                except Exception as ex:
                    QMessageBox.warning(self, "Qwen ASRの停止エラー", str(ex))
            self.status_label.setText("状態：ASR停止")
            self._set_asr_controls_state()
            return
        if not ok:
            self._asr_starting = False
            self._asr_ready = False
            self._stop_qwen_asr_container_logs()
            QMessageBox.warning(self, "Qwen ASRの起動エラー", error_message)
            self.status_label.setText("状態：ASR起動失敗")
            self._set_asr_controls_state()
            return
        if self._pending_asr_config is None:
            self._asr_starting = False
            self._asr_ready = False
            QMessageBox.warning(self, "ASR起動エラー", "ASR設定の受け渡しに失敗しました。")
            self.status_label.setText("状態：ASR起動失敗")
            self._set_asr_controls_state()
            return
        # 比較モードで external-asr-tcp を使う場合は、Qwen起動完了後に外部ASRも起動する。
        if self.asr_compare_mode_check.isChecked():
            top_engine = self.asr_compare_top_engine_combo.currentText().strip()
            bottom_engine = self.asr_compare_bottom_engine_combo.currentText().strip()
            if ("external-asr-tcp" in {top_engine, bottom_engine}) and self.asr_external_start_script_edit.text().strip():
                try:
                    self._start_external_asr_script(self.asr_external_start_script_edit.text().strip())
                except Exception as ex:
                    self._asr_starting = False
                    self._asr_ready = False
                    QMessageBox.warning(self, "外部ASRの起動エラー", str(ex))
                    self.status_label.setText("状態：ASR起動失敗")
                    self._set_asr_controls_state()
                    return
        self.status_label.setText("状態：ASRワーカー起動中…")
        pending = dict(self._pending_asr_config)
        compare_mode = bool(pending.get("compare_mode_enabled", False))
        worker = self._asr_compare_worker_script if compare_mode else self._asr_worker_script
        self._launch_asr_worker(pending, worker_script=worker)
