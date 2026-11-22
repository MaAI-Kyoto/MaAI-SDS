"""MMD モデルのまばたきモーフを会話とは独立して駆動する。"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable

from .avatar_keys import blink_command


# 成人の自発性瞬目は約 20 回/分（杉山・田多 2007, 617 名）。連続瞬目を含めてそれに近づける。
DEFAULT_BLINK_INTERVAL_MEAN_SEC = 3.0
DEFAULT_BLINK_MORPH = "まばたき"
MIN_BLINK_INTERVAL_MEAN_SEC = 1.0
MAX_BLINK_INTERVAL_MEAN_SEC = 30.0
BLINK_MODE_ALWAYS = "always"
BLINK_MODE_DIALOG = "dialog"
DEFAULT_BLINK_MODE = BLINK_MODE_ALWAYS


def normalize_blink_enable(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_blink_interval_mean_sec(value: object) -> float:
    try:
        interval = float(value)
    except (TypeError, ValueError):
        interval = DEFAULT_BLINK_INTERVAL_MEAN_SEC
    if not math.isfinite(interval):
        interval = DEFAULT_BLINK_INTERVAL_MEAN_SEC
    return max(MIN_BLINK_INTERVAL_MEAN_SEC, min(MAX_BLINK_INTERVAL_MEAN_SEC, interval))


def normalize_blink_mode(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {BLINK_MODE_DIALOG, "対話中のみ"}:
        return BLINK_MODE_DIALOG
    return DEFAULT_BLINK_MODE


def blink_should_run(enable: bool, mode: str, dialog_active: bool) -> bool:
    return bool(enable) and (mode != BLINK_MODE_DIALOG or bool(dialog_active))


def sanitize_blink_morph(value: object) -> str:
    morph = str(value or "").strip()
    if not morph or any(char in morph for char in ",=\r\n"):
        return DEFAULT_BLINK_MORPH
    return morph


class AutoBlinker:
    """自然な間隔でまばたきを送る（``KEYVALUE_SET|MaAI_Blink|<重み>``）。

    まばたきのモーフは、あらかじめ ``MODEL_BINDFACE`` でこの KeyValue に結び付けておく
    （``MmdRemoteBridge.set_blink_binding``、mmd/avatar_keys.py）。
    """

    # 瞬目間隔は右に長く裾を引く対数正規に近く、CV は 0.8〜1.1 程度（Johnston 2013 ほか）。
    _INTERVAL_CV = 0.9
    # 単発の瞬目どうしはこれより詰めない（短い間隔は下の連続瞬目で作る）。
    _MIN_SINGLE_INTERVAL_SEC = 1.0
    # 0.5 s 未満で続く瞬目は全体の 15%（2 連 11%, 3 連以上 4% 以内）。発生単位に直した値。
    _BURST_COUNT_PROBABILITIES = ((3, 0.015), (2, 0.06))
    # 連続瞬目の前の瞬目が開き終わってから次が始まるまで。開始間隔が 0.5 s 未満に収まる。
    _BURST_GAP_SEC = (0.08, 0.18)
    # 自発性瞬目は閉じが速く、開きが 2〜3 倍遅い（VanderWerf 2003: 92 / 242 ms）。
    # MMDAgent-EX 実機で 150 ms は閉じ切って見えず 330 ms は遅く感じたため、
    # 閉じと保持は閉じ切って見える長さを保ち、開きだけ延ばして緩やかに開き切る形にする。
    _CLOSE_SEC = 0.070
    _HOLD_SEC = 0.040
    _OPEN_SEC = 0.170
    # 1 回ごとの長さのばらつき（自発性瞬目の持続時間の SD は平均の約 2 割）。
    _DURATION_JITTER = 0.15
    # 送信の目安間隔。Windows の待ちは約 15.6 ms 単位に切り上がるので、
    # 段数ではなく経過時間からモーフ値を決め、全体の長さを保つ。
    _STEP_SEC = 0.016

    def __init__(
        self,
        send_fn: Callable[[str], bool | None],
        *,
        enable: bool = True,
        interval_mean_sec: float = DEFAULT_BLINK_INTERVAL_MEAN_SEC,
        morph: str = DEFAULT_BLINK_MORPH,
        clock: Callable[[], float] = time.monotonic,
        wait_fn: Callable[[float], bool | None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._send_fn = send_fn
        self._clock = clock
        self._wait_fn = wait_fn
        self._rng = rng if rng is not None else random.Random()
        self._settings_lock = threading.Lock()
        self._enable = normalize_blink_enable(enable)
        self._interval_mean_sec = normalize_blink_interval_mean_sec(interval_mean_sec)
        self._morph = sanitize_blink_morph(morph)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._settings_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="mmd-auto-blink",
            )
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        with self._settings_lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread is None or thread.is_alive():
            self._send_zero(self._snapshot()[2])

    def update(
        self,
        *,
        enable: bool | None = None,
        interval_mean_sec: float | None = None,
        morph: str | None = None,
    ) -> None:
        with self._settings_lock:
            before = (self._enable, self._interval_mean_sec, self._morph)
            if enable is not None:
                self._enable = normalize_blink_enable(enable)
            if interval_mean_sec is not None:
                self._interval_mean_sec = normalize_blink_interval_mean_sec(interval_mean_sec)
            if morph is not None:
                self._morph = sanitize_blink_morph(morph)
            changed = before != (self._enable, self._interval_mean_sec, self._morph)
        # 値が同じ更新で待ちを起こすと、次のまばたきまでの間隔が引き直しになり、
        # 設定が頻繁に届く間は一度もまばたきしなくなる。
        if changed:
            self._wake_event.set()

    def _snapshot(self) -> tuple[bool, float, str]:
        with self._settings_lock:
            return self._enable, self._interval_mean_sec, self._morph

    def _sample_interval_sec(self, mean_sec: float | None = None) -> float:
        mean = normalize_blink_interval_mean_sec(
            self._snapshot()[1] if mean_sec is None else mean_sec
        )
        floor = min(self._MIN_SINGLE_INTERVAL_SEC, 0.6 * mean)
        sigma = math.sqrt(math.log(1.0 + self._INTERVAL_CV ** 2))
        mu = self._lognormal_mu_for_truncated_mean(mean, sigma, floor)
        for _ in range(50):
            interval = self._rng.lognormvariate(mu, sigma)
            if interval >= floor:
                return interval
        return mean

    @staticmethod
    def _lognormal_mu_for_truncated_mean(mean: float, sigma: float, floor: float) -> float:
        """下限 floor で切った対数正規の平均が mean になる mu を二分法で求める。"""

        def phi(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        log_floor = math.log(floor)

        def truncated_mean(mu: float) -> float:
            tail = phi((mu - log_floor) / sigma)
            if tail <= 1e-12:
                return floor
            return math.exp(mu + sigma * sigma / 2.0) * phi((mu + sigma * sigma - log_floor) / sigma) / tail

        lo, hi = log_floor - 10.0 * sigma, math.log(mean)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if truncated_mean(mid) < mean:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _sample_burst_count(self) -> int:
        draw = self._rng.random()
        for count, probability in self._BURST_COUNT_PROBABILITIES:
            if draw < probability:
                return count
            draw -= probability
        return 1

    def _sample_burst_gap_sec(self) -> float:
        return self._rng.uniform(*self._BURST_GAP_SEC)

    def _sample_duration_scale(self) -> float:
        scale = self._rng.gauss(1.0, self._DURATION_JITTER)
        return max(0.75, min(1.3, scale))

    def _wait(self, duration_sec: float) -> bool:
        """指定時間待つ。停止・設定更新で中断したときは True。"""
        deadline = self._clock() + max(0.0, duration_sec)
        while not self._stop_event.is_set():
            if self._wake_event.is_set():
                self._wake_event.clear()
                return True
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                return False
            if self._wait_fn is None:
                interrupted = self._wake_event.wait(remaining)
            else:
                interrupted = bool(self._wait_fn(remaining))
            if interrupted or self._wake_event.is_set():
                self._wake_event.clear()
                return True
        return True

    @staticmethod
    def _ease(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _command(morph: str, weight: float) -> str:
        # モーフは結び付け（MODEL_BINDFACE）で決まるので、送るのは重みだけ
        return blink_command(weight)

    def _send(self, morph: str, weight: float) -> None:
        try:
            self._send_fn(self._command(morph, weight))
        except Exception:
            pass

    def _send_zero(self, morph: str) -> None:
        self._send(morph, 0.0)

    def _blink_once(self, morph: str) -> bool:
        """約 280 ms のまばたきを 1 回送る。完了時は True。"""
        morph = sanitize_blink_morph(morph)
        scale = self._sample_duration_scale()
        completed = False
        try:
            phases = ((self._CLOSE_SEC * scale, True), (self._OPEN_SEC * scale, False))
            for duration, closing in phases:
                if closing:
                    # 閉じ始めの 0 は送らない。最初の送信が一歩目に当たるよう始点をずらす。
                    start = self._clock() - self._STEP_SEC
                else:
                    # 閉じ切りの 1 は送り済み。閉じたまま保ってから開き始める。
                    if self._wait(self._HOLD_SEC * scale):
                        return False
                    start = self._clock()
                    if self._wait(self._STEP_SEC):
                        return False
                while True:
                    remaining = start + duration - self._clock()
                    # 残りが 1 ms を切ったら終点とみなす（浮動小数の端数で終わらなくなるのを防ぐ）。
                    fraction = 1.0 if remaining <= 0.001 else 1.0 - remaining / duration
                    # 開きは速く動き出して最後はゆっくり開き切る。
                    weight = self._ease(fraction) if closing else (1.0 - fraction) ** 2
                    self._send(morph, weight)
                    if fraction >= 1.0:
                        break
                    if self._wait(min(self._STEP_SEC, remaining)):
                        return False
            completed = True
            return True
        finally:
            if not completed:
                self._send_zero(morph)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                enable, mean_sec, morph = self._snapshot()
                if not enable:
                    self._wake_event.wait()
                    self._wake_event.clear()
                    continue
                if self._wait(self._sample_interval_sec(mean_sec)):
                    continue
                for index in range(self._sample_burst_count()):
                    if index and self._wait(self._sample_burst_gap_sec()):
                        break
                    enable, _mean_sec, morph = self._snapshot()
                    if not enable or not self._blink_once(morph):
                        break
        finally:
            self._send_zero(self._snapshot()[2])
