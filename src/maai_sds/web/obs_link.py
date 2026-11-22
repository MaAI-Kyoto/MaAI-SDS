"""OBS 録画の参照カウント（PySide の gui/obs_mixin.py と同じ規則）。

録画は「手動」「音声対話」「MaAI テスト」の 3 つの理由で取得でき、どれかが
残っている間は録画を続け、最後の 1 つが外れたときに止める。

- 音声対話: obs_remote_enabled と obs_remote_link_voice がオンのとき、開始で取得・終了で解放
- MaAI テスト: obs_remote_enabled と obs_remote_link_maai がオンのとき、起動で取得・停止で解放
- 手動: 録画開始/停止ボタン（obs_remote_enabled が必要）
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from ..dialog.settings import cfg_bool
from . import parity as web_parity

REASONS = ("manual", "voice", "maai")


class ObsRecordingSessions:
    def __init__(
        self,
        load_config: Callable[[], dict[str, Any]],
        log: Callable[[str], None],
        *,
        start: Callable[[dict[str, Any]], tuple[bool, str]] = web_parity.obs_start,
        stop: Callable[[dict[str, Any]], tuple[bool, str]] = web_parity.obs_stop,
    ) -> None:
        self._load_config = load_config
        self._log = log
        self._start = start
        self._stop = stop
        self._lock = threading.Lock()
        self._acquired: set[str] = set()

    @staticmethod
    def enabled(cfg: dict[str, Any]) -> bool:
        return cfg_bool(cfg, "obs_remote_enabled", False)

    @classmethod
    def linked(cls, cfg: dict[str, Any], reason: str) -> bool:
        if not cls.enabled(cfg):
            return False
        return cfg_bool(cfg, f"obs_remote_link_{reason}", False)

    def is_acquired(self, reason: str) -> bool:
        with self._lock:
            return reason in self._acquired

    def acquire(self, reason: str, cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
        """reason の参照を取る。最初の参照なら録画を始める。"""
        if reason not in REASONS:
            raise ValueError(reason)
        cfg = self._load_config() if cfg is None else cfg
        with self._lock:
            if reason in self._acquired:
                return True, ""
            if self._acquired:
                self._acquired.add(reason)
                self._log(f"[OBS] 録画セッション継続（参照 {len(self._acquired)}）")
                return True, ""
            ok, err = self._start(cfg)
            if not ok:
                self._log(f"[OBS] 録画開始に失敗: {err}")
                return False, err
            self._acquired.add(reason)
        host, port, _ = web_parity.obs_connection(cfg)
        self._log(f"[OBS] 録画開始 ({host}:{port})")
        return True, ""

    def release(self, reason: str, cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
        """reason の参照を外す。最後の参照なら録画を止める。持っていなければ何もしない。"""
        with self._lock:
            if reason not in self._acquired:
                return True, ""
            self._acquired.discard(reason)
            if self._acquired:
                self._log(f"[OBS] 参照解放（残り {len(self._acquired)}）")
                return True, ""
            cfg = self._load_config() if cfg is None else cfg
            ok, err = self._stop(cfg)
        host, port, _ = web_parity.obs_connection(cfg)
        if ok:
            self._log(f"[OBS] 録画停止 ({host}:{port})")
        else:
            self._log(f"[OBS] 録画停止に失敗: {err}")
        return ok, err

    def acquire_if_linked(self, reason: str) -> None:
        cfg = self._load_config()
        if self.linked(cfg, reason):
            self.acquire(reason, cfg)

    def release_in_background(self, reason: str) -> None:
        """イベント受信スレッドから呼ぶ用（OBS への接続待ちで受信を止めない）。"""
        if not self.is_acquired(reason):
            return
        threading.Thread(target=self.release, args=(reason,), daemon=True, name=f"obs-release-{reason}").start()
