"""controller.py から切り出した mixin（Phase 3-2）。

**mixin にしているのが要点**。`self` は `MMDAgentController` のままなので、
メソッド本文を 1 文字も書き換えずに移せる（協調オブジェクトにすると self が
別物になり、ウィジェット参照を全部書き換える必要が出る）。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from PySide6.QtWidgets import QMessageBox

from ..dialog_server_docker import DockerComposeTarget
from .theme import STATE_COLORS, lamp_style

DIALOG_SERVER_STOPPED_MESSAGE = (
    "対話サーバーが停止しています。「モジュール状態」の対話サーバーの ▶ で起動してください。"
)

# 対話サーバーの行の表示（状態 → (文字, ランプの色)）。waiting は自動起動・再接続を待っているとき。
DIALOG_SERVER_STATE_STYLES: dict[str, tuple[str, str]] = {
    "running": ("稼働", STATE_COLORS["running"]),
    "starting": ("起動中", STATE_COLORS["starting"]),
    "waiting": ("接続待ち", STATE_COLORS["starting"]),
    "stopping": ("停止中", STATE_COLORS["starting"]),
    "stopped": ("停止", STATE_COLORS["stopped"]),
    "error": ("異常", STATE_COLORS["error"]),
}


class ModuleStatusPanelMixin:
    """ModuleStatusPanelMixin: controller.py から移設したメソッド群。"""

    def _header_status_values(self) -> dict[str, tuple[str, str]]:
        """{chip: (状態, 表示文字)}。状態は theme.STATE_COLORS のキー。"""
        from .theme import STATE_COLORS  # noqa: F401  (キーの一覧)

        where = self._dialog_server_where()
        server_state = self._dialog_server_state()
        backend = {
            "running": ("running", f"対話サーバー：接続済み（{where}）"),
            "starting": ("starting", f"対話サーバー：起動中（{where}）"),
            "waiting": ("starting", f"対話サーバー：接続待ち（{where}）"),
            "stopping": ("starting", f"対話サーバー：停止中（{where}）"),
            "stopped": ("stopped", f"対話サーバー：停止（{where}）"),
            "error": ("error", f"対話サーバー：異常（{where}）"),
        }.get(server_state, ("stopped", f"対話サーバー（{where}）"))

        process = getattr(self, "process_service", None)
        try:
            proc_running = bool(process is not None and process.is_running())
        except Exception:
            proc_running = False
        connected = bool(
            getattr(self, "_orch_mmd_bridge_connected", False)
            or getattr(self, "_audio_mmd_bridge_connected", False)
        )
        mode = str(getattr(self, "_orch_mmd_bridge_mode", "") or getattr(self, "_audio_mmd_bridge_mode", "") or "")
        if connected:
            mmd = ("running", f"MMDAgent-EX：接続済み{f'（{mode}）' if mode else ''}")
        elif proc_running:
            mmd = ("starting", "MMDAgent-EX：起動中（未接続）")
        else:
            mmd = ("stopped", "MMDAgent-EX：未接続")

        if getattr(self, "_voice_dialog_active", False):
            voice = ("running", "音声対話：実行中")
        elif getattr(self, "_voice_dialog_starting", False):
            voice = ("starting", "音声対話：起動中")
        else:
            voice = ("stopped", "音声対話：停止")
        return {"backend": backend, "mmd": mmd, "voice": voice}

    def _refresh_header_status(self) -> None:
        from .theme import STATE_COLORS

        self._sync_stop_all_button()
        self._refresh_dialog_server_row()
        chips = getattr(self, "header_chips", None) or {}
        values = self._header_status_values()
        for key, (lamp, text) in chips.items():
            state, label = values.get(key, ("stopped", ""))
            color = STATE_COLORS.get(state, STATE_COLORS["stopped"])
            style = f"background-color: {color}; border-radius: 5px;"
            if lamp.styleSheet() != style:
                lamp.setStyleSheet(style)
            if text.text() != label:
                text.setText(label)
        mmd_label = getattr(self, "mmd_status_label", None)
        if mmd_label is not None:
            process = getattr(self, "process_service", None)
            try:
                proc_running = bool(process is not None and process.is_running())
            except Exception:
                proc_running = False
            connected = values["mmd"][0] == "running"
            mode = str(getattr(self, "_orch_mmd_bridge_mode", "") or "")
            line = (
                f"プロセス：{'起動中' if proc_running else '停止'}"
                f"　接続：{'接続済み' + (f'（{mode}）' if mode else '') if connected else '未接続'}"
            )
            if mmd_label.text() != line:
                mmd_label.setText(line)

    # -- 対話サーバー（dialog orchestrator）の起動・停止 -------------------------------------
    # ローカルではこの GUI の子プロセス、Docker では compose の orchestrator サービス。
    # ■ で止めたら、▶（またはモジュールの起動・音声対話の開始）まで自動では起動し直さない。

    def _dialog_server_where(self) -> str:
        return "Docker" if getattr(self, "_remote_backend_enabled", False) else "ローカル"

    def _dialog_server_state(self) -> str:
        busy = str(getattr(self, "_dialog_server_busy", "") or "")
        if busy:
            return busy
        ready = getattr(self, "_dialog_orchestrator_ready_event", None)
        if ready is not None and ready.is_set():
            return "running"
        if getattr(self, "_dialog_server_user_stopped", False):
            return "stopped"
        if getattr(self, "_dialog_server_error", ""):
            return "error"
        return "waiting"

    def _dialog_server_docker_target(self) -> DockerComposeTarget | None:
        if not getattr(self, "_remote_backend_enabled", False):
            return None
        return DockerComposeTarget.from_env()

    def _dialog_server_summary(self) -> tuple[str, str]:
        """(行に出す短い文, ツールチップ)。"""
        if not getattr(self, "_remote_backend_enabled", False):
            return "このPC", "この GUI の子プロセスとして動いています。"
        host, port = getattr(self, "_remote_orchestrator_endpoint", None) or ("", 0)
        text = f"Docker {host}:{port}"
        if self._dialog_server_docker_target() is None:
            return text, "compose の場所を知らないため、▶ / ■ はつなぎ直す・切るだけです（コンテナは起動・停止しません）。"
        return text, "▶ / ■ で docker compose の orchestrator サービスを起動・停止します。"

    def _refresh_dialog_server_row(self) -> None:
        lamp = getattr(self, "dialog_server_lamp", None)
        if lamp is None:
            return
        state = self._dialog_server_state()
        text, color = DIALOG_SERVER_STATE_STYLES.get(state, DIALOG_SERVER_STATE_STYLES["stopped"])
        style = lamp_style(color)
        if lamp.styleSheet() != style:
            lamp.setStyleSheet(style)
        label = self.dialog_server_state_label
        if label.text() != text:
            label.setText(text)
        error = str(getattr(self, "_dialog_server_error", "") or "")
        label.setToolTip(error)
        summary, tip = self._dialog_server_summary()
        combo = getattr(self, "dialog_server_mode_combo", None)
        if combo is not None:
            combo.setToolTip(error or f"{summary}\n{tip}\n切り替えると GUI を開き直します。")
        docker_without_compose = (
            getattr(self, "_remote_backend_enabled", False) and self._dialog_server_docker_target() is None
        )
        # Docker で compose の場所を知らないときも、■（切断して再接続しない）と ▶（再接続）は使える。
        can_start = state in ("stopped", "error", "waiting") and not (docker_without_compose and state == "waiting")
        can_stop = state in ("running", "waiting", "error")
        if self.dialog_server_start_btn.isEnabled() != can_start:
            self.dialog_server_start_btn.setEnabled(can_start)
        if self.dialog_server_stop_btn.isEnabled() != can_stop:
            self.dialog_server_stop_btn.setEnabled(can_stop)

    def _run_in_thread_pumping(self, fn: Callable[[], Any]) -> Any:
        """fn を別スレッドで実行し、終わるまで UI のイベントを回しながら待つ。"""
        result: dict[str, Any] = {}
        done = threading.Event()

        def runner() -> None:
            try:
                result["value"] = fn()
            except BaseException as ex:  # noqa: BLE001
                result["error"] = ex
            finally:
                done.set()

        threading.Thread(target=runner, daemon=True, name="dialog-server-control").start()
        while not done.wait(0.05):
            self._pump_ui_events()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _dialog_server_start(self) -> bool:
        """対話サーバーを起動してつなぐ。成功したら True。"""
        if getattr(self, "_dialog_server_busy", ""):
            return False
        self._dialog_server_busy = "starting"
        self._dialog_server_error = ""
        self._dialog_server_user_stopped = False
        self._refresh_header_status()
        try:
            target = self._dialog_server_docker_target()
            if target is not None:
                self._log_stream("[対話サーバー] docker compose up -d orchestrator")
                ok, output = self._run_in_thread_pumping(lambda: target.run("start"))
                for line in output.splitlines()[-5:]:
                    self._log_stream(f"[対話サーバー] {line}")
                if not ok:
                    raise RuntimeError(f"Docker の対話サーバーを起動できませんでした: {output[-300:]}")
            if getattr(self, "_remote_backend_enabled", False):
                # コンテナが立ち上がってポートを開くまで少しかかる。
                deadline = time.monotonic() + 120.0
                while True:
                    try:
                        self._ensure_dialog_orchestrator_worker()
                        break
                    except Exception as ex:
                        if self._controller_closing or time.monotonic() > deadline:
                            raise RuntimeError(f"対話サーバーにつながりません: {ex}") from ex
                        end = time.monotonic() + 0.3
                        while time.monotonic() < end:
                            self._pump_ui_events()
                            time.sleep(0.05)
            else:
                self._ensure_dialog_orchestrator_worker()
            self._log_stream("[対話サーバー] 起動しました")
            return True
        except Exception as ex:
            self._dialog_server_error = str(ex)
            self._log_stream(f"[対話サーバー][ERROR] {ex}")
            QMessageBox.warning(self, "対話サーバー", str(ex))
            return False
        finally:
            self._dialog_server_busy = ""
            self._refresh_header_status()

    def _dialog_server_stop(self) -> None:
        """音声対話とモジュールを止めてから、対話サーバーを止める。"""
        if getattr(self, "_dialog_server_busy", "") or self._dialog_server_state() == "stopped":
            return
        self._dialog_server_busy = "stopping"
        self._refresh_header_status()
        try:
            try:
                if self._dialog_orchestrator_ready_event.is_set():
                    self._module_panel_stop_all()
            except Exception as ex:
                self._log_stream(f"[対話サーバー] モジュールの停止に失敗: {ex}")
            self._dialog_server_user_stopped = True
            self._dialog_server_error = ""
            self._teardown_dialog_orchestrator_ipc()
            target = self._dialog_server_docker_target()
            if target is not None:
                self._log_stream("[対話サーバー] docker compose stop orchestrator")
                ok, output = self._run_in_thread_pumping(lambda: target.run("stop"))
                for line in output.splitlines()[-5:]:
                    self._log_stream(f"[対話サーバー] {line}")
                if not ok:
                    self._dialog_server_error = f"Docker の対話サーバーを停止できませんでした: {output[-300:]}"
                    self._log_stream(f"[対話サーバー][ERROR] {self._dialog_server_error}")
            for module in self._MODULE_PANEL_KEYS:
                if self._module_panel_states.get(module, {}).get("state") != "stopped":
                    self._set_module_panel_state(module, "stopped")
            self._log_stream("[対話サーバー] 停止しました")
        finally:
            self._dialog_server_busy = ""
            self._refresh_header_status()
            self._refresh_module_status_panel()

    def _dialog_server_current_mode(self) -> str:
        return "docker" if getattr(self, "_remote_backend_enabled", False) else "local"

    def _on_dialog_server_mode_activated(self, index: int) -> None:
        """「対話サーバー」の ローカル / Docker を選んだとき。選んだモードで GUI を開き直す。"""
        from .. import dialog_server_mode

        combo = self.dialog_server_mode_combo
        mode = str(combo.itemData(index) or "")
        current = self._dialog_server_current_mode()
        if mode not in dialog_server_mode.MODES or mode == current:
            return

        def revert() -> None:
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(current))
            combo.blockSignals(False)

        label = dialog_server_mode.MODE_LABELS[mode]
        answer = QMessageBox.question(
            self,
            "対話サーバーの切り替え",
            f"対話サーバーを「{label}」に切り替えます。\n"
            "GUI を開き直します（会話の履歴は消え、動いているモジュールと音声対話は止まります）。\n"
            "よろしいですか？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            revert()
            return
        try:
            dialog_server_mode.save_mode(mode)
            dialog_server_mode.spawn_relaunch(mode)
        except Exception as ex:
            revert()
            self._log_stream(f"[対話サーバー][ERROR] 切り替えに失敗: {ex}")
            QMessageBox.warning(self, "対話サーバーの切り替え", f"開き直せませんでした: {ex}")
            return
        self._log_stream(f"[対話サーバー] 「{label}」で開き直します")
        self.close()

    def _dialog_server_resume_for_action(self) -> bool:
        """■ で止めていたら、モジュールの起動や音声対話の開始の前に対話サーバーを起動する。"""
        if not getattr(self, "_dialog_server_user_stopped", False):
            return True
        return self._dialog_server_start()

    def _set_module_panel_state(self, module: str, state: str, detail: str = "") -> None:
        """モジュール状態を更新する（任意スレッドから可）。asr_compare は ASR 行に集約。"""
        m = str(module).strip().lower()
        if m == "asr_compare":
            m = "asr"
        if m not in self._module_panel_states:
            return
        self.module_panel_state_signal.emit(m, str(state).strip().lower(), str(detail))

    def _on_module_panel_state(self, module: str, state: str, detail: str) -> None:
        cur = self._module_panel_states.get(module)
        if cur is None:
            return
        if state not in self._MODULE_PANEL_STATE_STYLES:
            state = "stopped"
        # 失敗直後に届く旧 stop の module_stopped で赤表示が消えないようにする。
        # エラー解除は次の起動（starting）で行う。
        if state == "stopped" and cur.get("state") == "error":
            return
        cur["state"] = state
        cur["detail"] = detail
        if module == "maai" and state != "running":
            self._maai_process_metrics = {}
        self._refresh_module_status_panel()

    def _refresh_module_status_panel(self) -> None:
        batch_btn = getattr(self, "module_status_start_all_btn", None)
        if batch_btn is not None:
            states = [
                self._module_panel_states.get(module, {}).get("state", "stopped")
                for module in self._MODULE_PANEL_KEYS
            ]
            busy = getattr(self, "_module_start_all_busy", False) or "starting" in states
            batch_btn.setEnabled(
                not busy and any(state in ("stopped", "error") for state in states)
            )
            batch_btn.setText("起動中…" if busy else "一括起動")
        self._sync_stop_all_button()
        lamps = getattr(self, "module_status_lamps", None)
        state_labels = getattr(self, "module_status_state_labels", None)
        if not lamps or not state_labels:
            return
        for module in self._MODULE_PANEL_KEYS:
            st = self._module_panel_states.get(module, {})
            state = str(st.get("state", "stopped"))
            text, color = self._MODULE_PANEL_STATE_STYLES.get(
                state, self._MODULE_PANEL_STATE_STYLES["stopped"]
            )
            lamp = lamps.get(module)
            if lamp is not None:
                lamp.setStyleSheet(lamp_style(color))
                lamp.setToolTip(text)
            label = state_labels.get(module)
            if label is not None:
                label.setText(text)
                label.setToolTip(str(st.get("detail", "")).strip())
            start_btns = getattr(self, "module_status_start_btns", None)
            stop_btns = getattr(self, "module_status_stop_btns", None)
            if start_btns and stop_btns:
                sb = start_btns.get(module)
                tb = stop_btns.get(module)
                if sb is not None:
                    sb.setEnabled(state in ("stopped", "error"))
                if tb is not None:
                    tb.setEnabled(state in ("running", "starting"))
        self._refresh_module_status_summaries()
        self._refresh_maai_status_metrics()
        sync_wav = getattr(self, "_sync_wav_module_panel", None)
        if callable(sync_wav):
            sync_wav()

    def _refresh_module_status_summaries(self) -> None:
        labels = getattr(self, "module_status_summary_labels", None)
        if not labels:
            return
        for module in self._MODULE_PANEL_KEYS:
            label = labels.get(module)
            if label is None:
                continue
            full = self._module_settings_summary(module)
            short = full if len(full) <= 46 else full[:45] + "…"
            label.setText(short)
            label.setToolTip(full)

    def _module_panel_start_all(self) -> None:
        # Individual launchers may pump Qt events while loading a model.
        # Guard reentry during those nested event loops.
        if getattr(self, "_module_start_all_busy", False):
            return
        self._module_start_all_busy = True
        self._refresh_module_status_panel()
        try:
            for module in self._MODULE_PANEL_KEYS:
                state = self._module_panel_states.get(module, {}).get("state", "stopped")
                if state not in ("stopped", "error"):
                    continue
                try:
                    self._module_panel_start(module)
                except Exception as ex:
                    self._set_module_panel_state(module, "error", str(ex))
                    self._log_stream(f"[一括起動] {module} 起動失敗: {ex}")
        finally:
            self._module_start_all_busy = False
            self._refresh_module_status_panel()

    def _sync_stop_all_button(self) -> None:
        """一括停止は、モジュールか音声対話のどれかが動いているときだけ押せる。"""
        stop_btn = getattr(self, "module_status_stop_all_btn", None)
        if stop_btn is None:
            return
        states = [
            self._module_panel_states.get(module, {}).get("state", "stopped")
            for module in self._MODULE_PANEL_KEYS
        ]
        voice = bool(getattr(self, "_voice_dialog_active", False) or getattr(self, "_voice_dialog_starting", False))
        stopping = bool(getattr(self, "_module_stop_all_busy", False))
        enabled = not stopping and (voice or any(state in ("running", "starting") for state in states))
        if stop_btn.isEnabled() != enabled:
            stop_btn.setEnabled(enabled)
        text = "停止中…" if stopping else "一括停止"
        if stop_btn.text() != text:
            stop_btn.setText(text)

    def _module_panel_stop_all(self) -> None:
        """動いているモジュールをすべて止める。音声対話中なら先に音声対話を終える。"""
        if getattr(self, "_module_stop_all_busy", False):
            return
        self._module_stop_all_busy = True
        self._refresh_module_status_panel()
        try:
            if getattr(self, "_voice_dialog_active", False) or getattr(self, "_voice_dialog_starting", False):
                try:
                    self._stop_voice_dialog()
                except Exception as ex:
                    self._log_stream(f"[一括停止] 音声対話の終了に失敗: {ex}")
            for module in self._MODULE_PANEL_KEYS:
                state = self._module_panel_states.get(module, {}).get("state", "stopped")
                if state not in ("running", "starting"):
                    continue
                try:
                    self._module_panel_stop(module)
                except Exception as ex:
                    self._set_module_panel_state(module, "error", str(ex))
                    self._log_stream(f"[一括停止] {module} 停止失敗: {ex}")
        finally:
            self._module_stop_all_busy = False
            self._refresh_module_status_panel()

    def _module_panel_start(self, module: str) -> None:
        if not self._dialog_server_resume_for_action():
            return
        if module == "asr":
            self.launch_asr()
        elif module == "llm":
            self.run_llm_warmup_now()
        elif module == "tts":
            engine = self.tts_engine_combo.currentText().strip() or "VOICEVOX"
            if engine == "Kokoro-RealtimeTTS":
                self.launch_kokoro_realtimetts()
            elif engine == "VOICEVOX-Core":
                self.launch_voicevox_core()
            elif engine == "VOICEVOX":
                self.launch_voicevox()
            elif engine == "AivisSpeech":
                self.launch_aivisspeech()
            elif engine in {"Supertonic-3", "Qwen3-TTS-CPP", "LFM2.5-Audio", "Irodori-v4", "Tsukasa"}:
                self.launch_worker_tts(engine)
            else:
                self._log_stream(f"TTS エンジン {engine} の起動方法が不明です")
        elif module == "maai":
            self.launch_maai_test()

    def _module_panel_stop(self, module: str) -> None:
        if module == "asr":
            self.stop_asr()
        elif module == "llm":
            self._teardown_llm_ipc()
            backend = self.llm_backend_combo.currentText().strip() or "LM Studio"
            if backend == "LM Studio":
                self.unload_lmstudio_non_current_models(
                    silent=True, reason="module_stop", unload_all=True
                )
            self._set_module_panel_state("llm", "stopped")
        elif module == "tts":
            engine = self._tts_worker_engine
            if engine == "Kokoro-RealtimeTTS":
                self.stop_kokoro_realtimetts()
            elif engine == "VOICEVOX-Core":
                self.stop_voicevox_core()
            elif engine == "AivisSpeech":
                self.stop_aivisspeech()
            elif engine in {"Supertonic-3", "Qwen3-TTS-CPP", "LFM2.5-Audio", "Irodori-v4", "Tsukasa"}:
                self.stop_worker_tts(engine)
            else:
                self.stop_voicevox()
        elif module == "maai":
            self.stop_maai_test()
