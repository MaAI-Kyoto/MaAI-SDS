"""controller.py から切り出した mixin（Phase 3-2）。

**mixin にしているのが要点**。`self` は `MMDAgentController` のままなので、
メソッド本文を 1 文字も書き換えずに移せる（協調オブジェクトにすると self が
別物になり、ウィジェット参照を全部書き換える必要が出る）。
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from ..obs import remote as obs_remote


class ObsRemoteMixin:
    """ObsRemoteMixin: controller.py から移設したメソッド群。"""

    def _get_obs_remote_connection(self) -> tuple[str, int, str]:
        host_e = getattr(self, "obs_remote_host_edit", None)
        port_e = getattr(self, "obs_remote_port_edit", None)
        pw_e = getattr(self, "obs_remote_password_edit", None)
        host = str(host_e.text()).strip() if host_e is not None else "127.0.0.1"
        if not host:
            host = "127.0.0.1"
        port_s = str(port_e.text()).strip() if port_e is not None else "4455"
        try:
            port = int(port_s)
        except ValueError:
            port = 4455
        port = max(1, min(65535, port))
        password = str(pw_e.text()) if pw_e is not None else ""
        return host, port, password

    def _is_obs_remote_enabled(self) -> bool:
        cb = getattr(self, "obs_remote_enabled_check", None)
        return bool(cb is not None and cb.isChecked())

    def _obs_remote_try_start(self) -> bool:
        self._obs_remote_refcount += 1
        if self._obs_remote_refcount > 1:
            self.process_output_signal.emit(f"[OBS] 録画セッション継続（参照 {self._obs_remote_refcount}）")
            return True
        host, port, password = self._get_obs_remote_connection()
        ok, err = obs_remote.start_record(host, port, password)
        if ok:
            self.process_output_signal.emit(f"[OBS] 録画開始 ({host}:{port})")
            return True
        self._obs_remote_refcount -= 1
        self.process_output_signal.emit(f"[OBS] 録画開始に失敗: {err}")
        return False

    def _obs_remote_try_stop(self) -> None:
        if self._obs_remote_refcount <= 0:
            return
        self._obs_remote_refcount -= 1
        if self._obs_remote_refcount > 0:
            self.process_output_signal.emit(f"[OBS] 参照解放（残り {self._obs_remote_refcount}）")
            return
        host, port, password = self._get_obs_remote_connection()
        ok, err = obs_remote.stop_record(host, port, password)
        if ok:
            self.process_output_signal.emit(f"[OBS] 録画停止 ({host}:{port})")
        else:
            self.process_output_signal.emit(f"[OBS] 録画停止に失敗: {err}")

    def _obs_remote_release_maai(self) -> None:
        if not self._obs_acquired_maai:
            return
        self._obs_acquired_maai = False
        self._obs_remote_try_stop()

    def _obs_remote_release_voice(self) -> None:
        if not self._obs_acquired_voice:
            return
        self._obs_acquired_voice = False
        self._obs_remote_try_stop()

    def _obs_remote_release_manual(self) -> None:
        if not self._obs_acquired_manual:
            return
        self._obs_acquired_manual = False
        self._obs_remote_try_stop()

    def _on_obs_remote_manual_start_clicked(self) -> None:
        if not self._is_obs_remote_enabled():
            QMessageBox.information(
                self,
                "OBS",
                "先に「遠隔録画を有効にする（WebSocket）」をオンにしてください。",
            )
            return
        if self._obs_acquired_manual:
            self.process_output_signal.emit("[OBS] 手動セッションは既に取得済みです。")
            return
        if self._obs_remote_try_start():
            self._obs_acquired_manual = True

    def _on_obs_remote_manual_stop_clicked(self) -> None:
        if not self._obs_acquired_manual:
            self.process_output_signal.emit("[OBS] 手動による参照がありません。")
            return
        self._obs_remote_release_manual()

    def _on_obs_remote_test_clicked(self) -> None:
        host, port, password = self._get_obs_remote_connection()
        ok, err = obs_remote.test_connection(host, port, password)
        if ok:
            QMessageBox.information(self, "OBS", "WebSocket接続に成功しました。")
        else:
            QMessageBox.warning(self, "OBS", f"WebSocket接続に失敗しました：\n{err}")
