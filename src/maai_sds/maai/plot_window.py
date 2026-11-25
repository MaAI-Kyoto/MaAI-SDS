from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from typing import Any

import numpy as np
from PySide6.QtGui import QColor, QResizeEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from ..utils import to_float_or_none

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - optional runtime dependency
    pg = None  # type: ignore[assignment]


DEFAULT_VAP_BIN_TIMES_SEC = [0.2, 0.4, 0.6, 0.8]

# (panel_id, UI ラベル) — _fixed_layout_keys() + asr / voice_latency と一致させる
MAAI_PLOT_VISIBILITY_SPECS: tuple[tuple[str, str], ...] = (
    ("x1", "入力波形1"),
    ("x2", "入力波形2"),
    ("p_now", "ターン取得／割り込み（p_now）"),
    ("p_future", "長期ターン予測（p_future）"),
    ("p_bins", "区間別確率（p_bins）"),
    ("vad", "音声活動検出（MaAI VAD）"),
    ("silero_vad_score", "Silero VAD"),
    ("p_bc", "相槌確率（p_bc）"),
    ("p_nod", "頷き確率（p_nod）"),
    ("nod_range", "頷き幅（nod_range）"),
    ("nod_speed", "頷き速度（nod_speed）"),
    ("nod_count", "頷き回数（nod_count）"),
    ("nod_swing_up", "頷き上振り確率（nod_swing_up）"),
    ("asr", "ASRセグメント"),
    ("voice_latency", "音声対話レイテンシ"),
)

# 既定（設定ファイルに値が無いとき）で非表示にするパネル。区間別確率と、Silero VAD・頷きのパラメータ。
# web/app/plot.js の hiddenByDefault と揃える。
MAAI_PLOT_HIDDEN_BY_DEFAULT: frozenset[str] = frozenset(
    {"p_bins", "silero_vad_score", "nod_range", "nod_speed", "nod_count", "nod_swing_up"}
)


def maai_plot_default_visible(panel_id: str) -> bool:
    return panel_id not in MAAI_PLOT_HIDDEN_BY_DEFAULT


class MaaiPlotWindow(QWidget):
    def __init__(
        self,
        shown_context_sec: int = 10,
        frame_rate: int = 20,
        sample_rate: int = 16000,
        use_fixed_draw_rate: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("MaAIプロット")
        self.resize(1200, 780)

        self.shown_context_sec = int(shown_context_sec)
        self.frame_rate = int(frame_rate)
        self.sample_rate = int(sample_rate)
        self.MAX_CONTEXT_LEN = max(1, self.frame_rate * self.shown_context_sec)
        self.MAX_CONTEXT_WAV_LEN = max(1, self.sample_rate * self.shown_context_sec)
        self.use_fixed_draw_rate = bool(use_fixed_draw_rate)
        self.fixed_draw_interval_raw_sec = 1.0 / 15.0
        self._last_draw_time_by_channel: dict[str, float] = {"raw": 0.0}
        self.waveform_from_raw_audio = True
        self._plot_enabled = pg is not None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if self._plot_enabled:
            _pg_opts: dict[str, Any] = {"antialias": False}
            if os.environ.get("MaAI_PLOT_USE_OPENGL", "").strip().lower() in ("1", "true", "yes"):
                _pg_opts["useOpenGL"] = True
            pg.setConfigOptions(**_pg_opts)
            self.graph = pg.GraphicsLayoutWidget()
            layout.addWidget(self.graph)
        else:
            self.graph = None
            info = QLabel("MaAIプロットを表示するにはpyqtgraphをインストールしてください。")
            info.setWordWrap(True)
            layout.addWidget(info)
        self.setLayout(layout)

        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[str, Any] = {}
        self.keys: list[str] = []
        self.data_buffer: dict[str, Any] = {}
        self.initialized = False
        self.asr_plot: pg.PlotItem | None = None
        self._asr_segments: list[dict[str, Any]] = []
        self._open_stream_segments: dict[tuple[str, str], dict[str, Any]] = {}
        self._final_arrivals: list[dict[str, Any]] = []
        self._last_source_ts_end: float | None = None
        self._last_source_ts_wall: float | None = None
        self._p_bins_bin_times: list[float] = list(DEFAULT_VAP_BIN_TIMES_SEC)
        self._event_series: dict[str, list[tuple[float, float]]] = {}
        self._wave_chunks: dict[str, deque[tuple[float, np.ndarray]]] = {
            "x1": deque(),
            "x2": deque(),
        }
        self._voice_latency_plot: pg.PlotItem | None = None
        self._voice_latency_spans: list[dict[str, Any]] = []
        self._voice_interrupt_action_ts: list[float] = []
        self._voice_turn_events: list[float] = []
        self._voice_prefetch_events: list[float] = []
        # LLM 遅延帯の進行中開始時刻（value=None + start_ts で開き、確定 span または clear で閉じる）
        self._voice_llm_open_start: float | None = None
        self._turn_lock_edges: list[tuple[float, bool]] = []
        self._nod_trigger_regions: list[dict[str, Any]] = []
        self._bc_trigger_regions: list[dict[str, Any]] = []
        self.align_segment_right_edge_to_arrival = False
        self._interrupt_vad_threshold = 0.1
        self._interrupt_p_now_threshold = 0.5
        self._turn_p_now_threshold = 0.5
        self._vad_on_threshold = 0.1
        self._vad_off_threshold = 0.2
        self._threshold_lines: dict[str, Any] = {}
        self._series_xy_scratch_x: dict[str, np.ndarray] = {}
        self._series_xy_scratch_y: dict[str, np.ndarray] = {}
        self._p_curve_split_fp: dict[str, tuple[Any, ...]] = {}
        self._p_curve_split_cached: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._p_bins_draw_cache_arr: np.ndarray | None = None
        self._p_bins_draw_cache_times: tuple[float, ...] | None = None
        self._unknown_result_keys_warned: set[str] = set()
        self._p_bins_last_result_bin_times: Any | None = None
        self._accepting_updates: bool = True
        self._panel_visible: dict[str, bool] = {
            k: True for k in (list(self._fixed_layout_keys()) + ["asr", "voice_latency"])
        }

        self._x_wave = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_WAV_LEN)
        self._x_ctx = np.linspace(-self.shown_context_sec, 0, self.MAX_CONTEXT_LEN)

    def reset_plot(self) -> None:
        if self.graph is not None:
            self.graph.clear()
        self.plots = {}
        self.curves = {}
        self.keys = []
        self.data_buffer = {}
        self.initialized = False
        self.asr_plot = None
        self._asr_segments = []
        self._open_stream_segments = {}
        self._final_arrivals = []
        self._last_source_ts_end = None
        self._last_source_ts_wall = None
        self._event_series = {}
        self._wave_chunks = {
            "x1": deque(),
            "x2": deque(),
        }
        self._voice_latency_plot = None
        self._voice_latency_spans = []
        self._voice_interrupt_action_ts = []
        self._voice_turn_events = []
        self._voice_prefetch_events = []
        self._voice_llm_open_start = None
        self._turn_lock_edges = []
        self._nod_trigger_regions = []
        self._bc_trigger_regions = []
        self._threshold_lines = {}
        self._series_xy_scratch_x = {}
        self._series_xy_scratch_y = {}
        self._p_curve_split_fp = {}
        self._p_curve_split_cached = {}
        self._p_bins_draw_cache_arr = None
        self._p_bins_draw_cache_times = None
        self._unknown_result_keys_warned = set()
        self._p_bins_last_result_bin_times = None
        # _panel_visible はユーザー設定のまま維持する

    def configure_maai_plot_panel_visibility(
        self,
        updates: dict[str, bool] | None = None,
        **kwargs: bool,
    ) -> None:
        """各サブプロットの表示/非表示。キーは MAAI_PLOT_VISIBILITY_SPECS の panel_id。"""
        merged: dict[str, bool] = {}
        if updates:
            merged.update({str(k): bool(v) for k, v in updates.items()})
        merged.update({str(k): bool(v) for k, v in kwargs.items()})
        for k, v in merged.items():
            self._panel_visible[k] = v
        self._sync_plot_panel_visibility()
        if not self.initialized:
            return
        now_ts = self._audio_now_ts()
        if self._panel_visible.get("asr", True) and self.asr_plot is not None:
            self._redraw_asr_segments_axis(draw=True)
        if self._panel_visible.get("voice_latency", True) and self._voice_latency_plot is not None:
            self._redraw_voice_latency()
        self._redraw_waveforms(now_ts)
        self.flush_maai_draw(now_ts=now_ts)

    def _sync_plot_panel_visibility(self) -> None:
        self._relayout_panels()

    def _relayout_panels(self) -> None:
        """非表示パネルをレイアウトから除去してスペースを詰め、
        可視パネル数が 10 以上なら 2 列、9 以下なら 1 列にする。"""
        if not self._plot_enabled or self.graph is None or not self.initialized:
            return
        ci = self.graph.ci
        lay = ci.layout

        all_managed: dict[str, Any] = {}
        for key, p in self.plots.items():
            all_managed[key] = p
        if self.asr_plot is not None:
            all_managed["__asr__"] = self.asr_plot
        if self._voice_latency_plot is not None:
            all_managed["__voice_latency__"] = self._voice_latency_plot

        # pyqtgraph 公式 API でレイアウト＆シーンから全アイテムを除去する。
        # 旧実装は lay.removeAt + ci.items/rows の手動リセットで、各パネルの境界
        # QGraphicsRectItem がシーンに取り残され、表示切替の度に ~12.5 items/回 蓄積していた。
        ci.clear()

        for p in all_managed.values():
            try:
                p.setVisible(False)
            except Exception:
                pass

        ordered_keys = list(self._fixed_layout_keys())
        visible: list[tuple[str, Any]] = []
        asr_vl_inserted = False
        for key in ordered_keys:
            if not self._panel_visible.get(key, True):
                continue
            p = self.plots.get(key)
            if p is None:
                continue
            visible.append((key, p))
            if key == "x1":
                if self._panel_visible.get("asr", True) and self.asr_plot is not None:
                    visible.append(("__asr__", self.asr_plot))
                if self._panel_visible.get("voice_latency", True) and self._voice_latency_plot is not None:
                    visible.append(("__voice_latency__", self._voice_latency_plot))
                asr_vl_inserted = True
        if not asr_vl_inserted:
            if self._panel_visible.get("asr", True) and self.asr_plot is not None:
                visible.append(("__asr__", self.asr_plot))
            if self._panel_visible.get("voice_latency", True) and self._voice_latency_plot is not None:
                visible.append(("__voice_latency__", self._voice_latency_plot))

        n_visible = len(visible)
        # 9 個までは 1 列（縦に並べたほうが時間軸をそろえて読みやすい）。10 個以上で 2 列にする（Web の plot.js と同じ）。
        use_two_cols = n_visible > 9

        if use_two_cols:
            left = [(k, p) for k, p in visible if k not in self._RIGHT_COLUMN_KEYS]
            right = [(k, p) for k, p in visible if k in self._RIGHT_COLUMN_KEYS]
            for row, (_k, p) in enumerate(left):
                ci.addItem(p, row, 0)
                p.setVisible(True)
            for row, (_k, p) in enumerate(right):
                ci.addItem(p, row, 1)
                p.setVisible(True)
        else:
            for row, (_k, p) in enumerate(visible):
                ci.addItem(p, row, 0)
                p.setVisible(True)

        self._configure_maai_grid_columns()
        self._align_plot_left_axes()

    def set_thresholds(
        self,
        *,
        interrupt_vad_threshold: float,
        interrupt_p_now_threshold: float,
        turn_p_now_threshold: float,
        vad_on_threshold: float = 0.1,
        vad_off_threshold: float = 0.2,
    ) -> None:
        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        self._interrupt_vad_threshold = _clamp01(interrupt_vad_threshold)
        self._interrupt_p_now_threshold = _clamp01(interrupt_p_now_threshold)
        self._turn_p_now_threshold = _clamp01(turn_p_now_threshold)
        self._vad_on_threshold = _clamp01(vad_on_threshold)
        self._vad_off_threshold = _clamp01(vad_off_threshold)
        line = self._threshold_lines.get("interrupt_vad")
        if line is not None:
            line.setValue(self._interrupt_vad_threshold)
        line = self._threshold_lines.get("vad_on")
        if line is not None:
            line.setValue(self._vad_on_threshold)
        line = self._threshold_lines.get("vad_off")
        if line is not None:
            line.setValue(self._vad_off_threshold)
        line = self._threshold_lines.get("interrupt_p_now")
        if line is not None:
            line.setValue(1.0 - self._interrupt_p_now_threshold)
        line = self._threshold_lines.get("turn_p_now")
        if line is not None:
            line.setValue(1.0 - self._turn_p_now_threshold)

    def _first_float(self, v: Any, default: float = 0.0) -> float:
        arr = np.array(v, dtype=float).reshape(-1)
        if arr.size == 0:
            return default
        return float(arr[-1])

    def _scalar_nod_plot_value(self, key: str, val: Any) -> float:
        """nod_para 系列用: 表示軸に合わせてクランプしたスカラー（時系列プロット用）。"""
        v = self._first_float(val)
        if key == "nod_count":
            # NaN/Inf だと int(round(v)) が ValueError/OverflowError で落ちるためガードする。
            if not (v == v) or v in (float("inf"), float("-inf")):
                return 0.0
            return float(max(0, min(2, int(round(v)))))
        if key == "nod_range":
            return max(0.035, min(0.15, float(v)))
        if key == "nod_speed":
            return max(0.08, min(0.25, float(v)))
        if key == "nod_swing_up":
            return max(0.0, min(1.0, float(v)))
        return float(v)

    _NOD_PARA_PLOT_KEYS: frozenset[str] = frozenset({"nod_range", "nod_speed", "nod_count", "nod_swing_up"})

    @staticmethod
    def _set_inverted_probability_axis_ticks(plot: pg.PlotItem) -> None:
        plot.getAxis("left").setTicks(
            [[(0.0, "1.0"), (0.2, "0.8"), (0.4, "0.6"), (0.6, "0.4"), (0.8, "0.2"), (1.0, "0.0")]]
        )

    @staticmethod
    def _invert_probability_value(v: float) -> float:
        return 1.0 - max(0.0, min(1.0, float(v)))

    @staticmethod
    def _special_plot_keys() -> tuple[str, ...]:
        return ("x1", "x2", "p_now", "p_future", "p_bins", "vad", "silero_vad_score")

    def _infer_keys(self, result: dict[str, Any]) -> list[str]:
        special_keys = list(self._special_plot_keys())
        meta_keys = {
            "t",
            "source_ts_start",
            "source_ts_end",
            "source_sample_start",
            "source_sample_end",
            "compare_slot",
            "maai_mode",
            "p_bc_threshold",
            "nod_swing_up_threshold",
        }
        inferred: list[str] = [k for k in special_keys if k in result]
        for k, v in result.items():
            if (k in special_keys) or (k in meta_keys) or str(k).startswith("source_"):
                continue
            # 非数値メタ情報（例: maai_mode）が系列化されるのを防ぐ。
            if isinstance(v, (list, tuple, np.ndarray)):
                try:
                    arr = np.array(v, dtype=float).reshape(-1)
                except Exception:
                    continue
                if arr.size <= 0:
                    continue
                inferred.append(k)
                continue
            if to_float_or_none(v) is None:
                continue
            inferred.append(k)
        return inferred

    def _should_draw_now(self, channel: str) -> bool:
        """生波形のみ間引き。ASR セグメント／レイテンシはイベントで即再描画する。"""
        if not self.use_fixed_draw_rate:
            return True
        if channel != "raw":
            return True
        now = time.time()
        last = float(self._last_draw_time_by_channel.get("raw", 0.0))
        interval = float(self.fixed_draw_interval_raw_sec)
        if now - last < interval:
            return False
        self._last_draw_time_by_channel["raw"] = now
        return True

    @staticmethod
    def _tune_wave_plot_item(item: Any) -> None:
        """点数が多い波形を高頻度更新するため、表示解像度に合わせて間引き・クリップする。"""
        if item is None:
            return
        try:
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
        except Exception:
            pass

    @staticmethod
    def _tune_maai_time_series_item(item: Any) -> None:
        if item is None:
            return
        try:
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
        except Exception:
            pass

    # 2段組み: 右列に backchannel / nod 系のみ配置
    _RIGHT_COLUMN_KEYS: frozenset[str] = frozenset(
        {"p_bc", "p_nod", "nod_range", "nod_speed", "nod_count", "nod_swing_up"}
    )
    _NOD_TRIGGER_PLOT_KEYS: frozenset[str] = frozenset(
        {"p_nod", "nod_range", "nod_speed", "nod_count", "nod_swing_up"}
    )

    @classmethod
    def _fixed_layout_keys(cls) -> list[str]:
        """_infer_keys の和集合を固定順で列挙（実行中の graph.clear 再構築を避ける）。"""
        return list(cls._special_plot_keys()) + [
            "p_bc",
            "p_nod",
            "nod_range",
            "nod_speed",
            "nod_count",
            "nod_swing_up",
        ]

    @classmethod
    def _left_row_items_for_keys(cls, keys: list[str]) -> list[str]:
        left_keys = [k for k in keys if k not in cls._RIGHT_COLUMN_KEYS]
        row_items: list[str] = []
        if "x1" in left_keys:
            x1_idx = left_keys.index("x1")
            for i, key in enumerate(left_keys):
                row_items.append(key)
                if i == x1_idx:
                    row_items.append("__asr__")
                    row_items.append("__voice_latency__")
        else:
            row_items.extend(left_keys)
            row_items.append("__asr__")
            row_items.append("__voice_latency__")
        return row_items

    @classmethod
    def _right_row_items_for_keys(cls, keys: list[str]) -> list[str]:
        return [k for k in keys if k in cls._RIGHT_COLUMN_KEYS]

    def _dummy_seed_full_layout(self) -> dict[str, Any]:
        n_bins = len(DEFAULT_VAP_BIN_TIMES_SEC)
        return {
            "x1": [0.0],
            "x2": [0.0],
            "p_now": 0.5,
            "p_future": 0.5,
            "p_bins": np.zeros((2, n_bins), dtype=float),
            "bin_times": list(DEFAULT_VAP_BIN_TIMES_SEC),
            "vad": [0.0, 0.0],
            "silero_vad_score": 0.0,
            "p_bc": 0.0,
            "p_nod": 0.0,
            "nod_range": 0.0,
            "nod_speed": 0.0,
            "nod_count": 0.0,
            "nod_swing_up": 0.0,
        }

    def init_full_layout(self) -> None:
        """固定キー列でグラフを1回だけ構築する。実行中にキー追加で再初期化しない前提。"""
        if self.initialized:
            return
        self._init_graph(self._dummy_seed_full_layout(), layout_keys=self._fixed_layout_keys())

    def closeEvent(self, event) -> None:  # noqa: N802
        self._accepting_updates = False
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        # ウィンドウは親に保持され show() で再表示される。閉じた際に止めた
        # 更新受け付けを復帰させないと、再表示後にプロットが恒久的に無反応になる。
        self._accepting_updates = True
        super().showEvent(event)

    def _split_by_threshold_cached(
        self, curve_key: str, x: np.ndarray, y: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.asarray(x, dtype=float).reshape(-1)
        ys = np.asarray(y, dtype=float).reshape(-1)
        n = int(min(xs.size, ys.size))
        if n <= 0:
            out = self._split_by_threshold_interpolated(xs, ys, threshold)
            self._p_curve_split_fp.pop(curve_key, None)
            self._p_curve_split_cached.pop(curve_key, None)
            return out
        th = float(threshold)
        fp = (n, float(xs[0]), float(xs[n - 1]), float(ys[0]), float(ys[n - 1]), th)
        if self._p_curve_split_fp.get(curve_key) == fp:
            return self._p_curve_split_cached[curve_key]
        out = self._split_by_threshold_interpolated(xs, ys, th)
        self._p_curve_split_fp[curve_key] = fp
        self._p_curve_split_cached[curve_key] = out
        return out

    def _maybe_draw_p_bins(self, arr: np.ndarray, bin_times_value: Any) -> None:
        arr = np.asarray(arr, dtype=float)
        try:
            times_t = tuple(float(x) for x in (bin_times_value or self._p_bins_bin_times or DEFAULT_VAP_BIN_TIMES_SEC))
        except Exception:
            times_t = tuple(float(x) for x in DEFAULT_VAP_BIN_TIMES_SEC)
        prev = self._p_bins_draw_cache_arr
        if (
            prev is not None
            and prev.shape == arr.shape
            and self._p_bins_draw_cache_times == times_t
            and np.array_equal(prev, arr)
        ):
            return
        self._draw_p_bins(arr, bin_times_value)
        self._p_bins_draw_cache_arr = np.array(arr, dtype=float, copy=True)
        self._p_bins_draw_cache_times = times_t

    def _configure_plot(self, plot: pg.PlotItem, *, title: str, x_range: tuple[float, float]) -> None:
        plot.setTitle(title)
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.setXRange(float(x_range[0]), float(x_range[1]), padding=0.0)
        plot.hideButtons()

    def _configure_maai_grid_columns(self) -> None:
        """列幅を揃える。1列時はフル幅、2列時は均等分割。"""
        if not self._plot_enabled or self.graph is None or not self.initialized:
            return
        try:
            lay = self.graph.ci.layout
            ncols = lay.columnCount()
            if ncols < 2:
                lay.setColumnStretchFactor(0, 1)
                lay.invalidate()
                return
            spacing = 6
            lay.setHorizontalSpacing(spacing)
            lay.setColumnStretchFactor(0, 1)
            lay.setColumnStretchFactor(1, 1)
            inner = max(0, int(self.graph.width()) - 8)
            half = max(160, (inner - spacing) // 2)
            lay.setColumnPreferredWidth(0, half)
            lay.setColumnPreferredWidth(1, half)
            lay.invalidate()
        except Exception:
            pass

    def resizeEvent(self, event: QResizeEvent | None) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._configure_maai_grid_columns()

    def _align_plot_left_axes(self) -> None:
        # サブプロットごとに左軸幅が変わると、下段ASR軸のx開始位置がずれて見える。
        # 全プロットの左軸幅を統一して、x軸の表示位置を揃える。
        plots: list[pg.PlotItem] = list(self.plots.values())
        if self.asr_plot is not None:
            plots.append(self.asr_plot)
        if self._voice_latency_plot is not None:
            plots.append(self._voice_latency_plot)
        if not plots:
            return
        max_w = 52
        for p in plots:
            axis = p.getAxis("left")
            if axis is not None:
                try:
                    max_w = max(max_w, int(axis.width()))
                except Exception:
                    pass
        target = max(52, min(120, max_w))
        for p in plots:
            axis = p.getAxis("left")
            if axis is not None:
                axis.setWidth(target)

    @staticmethod
    def _split_by_threshold_interpolated(
        x: np.ndarray, y: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.array(x, dtype=float).reshape(-1)
        ys = np.array(y, dtype=float).reshape(-1)
        n = int(min(xs.size, ys.size))
        if n == 0:
            return np.array([0.0], dtype=float), np.array([np.nan], dtype=float), np.array([np.nan], dtype=float)
        if n == 1:
            v = float(ys[0])
            xv = np.array([float(xs[0])], dtype=float)
            hi = np.array([v if v >= threshold else np.nan], dtype=float)
            lo = np.array([v if v <= threshold else np.nan], dtype=float)
            return xv, hi, lo

        x_out: list[float] = [float(xs[0])]
        y_out: list[float] = [float(ys[0])]
        th = float(threshold)
        for i in range(n - 1):
            x0 = float(xs[i])
            x1 = float(xs[i + 1])
            y0 = float(ys[i])
            y1 = float(ys[i + 1])
            d0 = y0 - th
            d1 = y1 - th
            if (d0 * d1) < 0.0 and (y1 != y0):
                t = (th - y0) / (y1 - y0)
                xc = x0 + (x1 - x0) * t
                x_out.append(float(xc))
                y_out.append(th)
            x_out.append(x1)
            y_out.append(y1)

        x_arr = np.array(x_out, dtype=float)
        y_arr = np.array(y_out, dtype=float)
        hi = np.where(y_arr >= th, y_arr, np.nan)
        lo = np.where(y_arr <= th, y_arr, np.nan)
        return x_arr, hi, lo

    @staticmethod
    def _clip_positive(y: np.ndarray) -> np.ndarray:
        return np.maximum(np.array(y, dtype=float).reshape(-1), 0.0)

    def _append_event_value(self, key: str, value: float, event_ts: float) -> None:
        series = self._event_series.setdefault(key, [])
        series.append((float(event_ts), float(value)))
        min_ts = float(event_ts) - float(self.shown_context_sec)
        if len(series) > 1:
            self._event_series[key] = [(t, v) for (t, v) in series if t >= min_ts]

    def _event_series_xy(self, key: str, now_ts: float) -> tuple[np.ndarray, np.ndarray]:
        series = self._event_series.get(key, [])
        if not series:
            return np.array([-self.shown_context_sec, 0.0], dtype=float), np.array([0.0, 0.0], dtype=float)
        n = len(series)
        sx = self._series_xy_scratch_x.get(key)
        sy = self._series_xy_scratch_y.get(key)
        if sx is None or sy is None or sx.size < n or sy.size < n:
            cap = max(n, 32)
            sx = np.empty(cap, dtype=float)
            sy = np.empty(cap, dtype=float)
            self._series_xy_scratch_x[key] = sx
            self._series_xy_scratch_y[key] = sy
        ts0 = float(now_ts)
        xv = sx[:n]
        yv = sy[:n]
        for i, (t, v) in enumerate(series):
            xv[i] = float(t) - ts0
            yv[i] = float(v)
        return np.array(xv, dtype=float, copy=True), np.array(yv, dtype=float, copy=True)

    def add_nod_trigger_region(
        self,
        *,
        used_start_ts: float,
        used_end_ts: float,
        trigger_ts: float,
        buffer_frames: int | None = None,
        count: int | None = None,
    ) -> None:
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        self._nod_trigger_regions.append(
            {
                "used_start_ts": float(used_start_ts),
                "used_end_ts": float(used_end_ts),
                "trigger_ts": float(trigger_ts),
                "buffer_frames": int(buffer_frames or 0),
                "count": int(count or 0),
            }
        )
        window_start = time.time() - float(self.shown_context_sec)
        self._nod_trigger_regions = [
            r
            for r in self._nod_trigger_regions
            if float(r.get("trigger_ts", 0.0)) >= window_start
        ][-200:]

    def add_bc_trigger_region(
        self,
        *,
        used_start_ts: float,
        used_end_ts: float,
        trigger_ts: float,
        buffer_frames: int | None = None,
    ) -> None:
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        self._bc_trigger_regions.append(
            {
                "used_start_ts": float(used_start_ts),
                "used_end_ts": float(used_end_ts),
                "trigger_ts": float(trigger_ts),
                "buffer_frames": int(buffer_frames or 0),
            }
        )
        window_start = time.time() - float(self.shown_context_sec)
        self._bc_trigger_regions = [
            r
            for r in self._bc_trigger_regions
            if float(r.get("trigger_ts", 0.0)) >= window_start
        ][-200:]

    def _redraw_nod_trigger_regions(self, now_ts: float) -> None:
        if not self._plot_enabled or not self.initialized:
            return
        window_start = float(now_ts) - float(self.shown_context_sec)
        active = [
            region
            for region in self._nod_trigger_regions
            if to_float_or_none(region.get("trigger_ts")) is not None
            and float(region.get("trigger_ts", 0.0)) >= window_start
        ][-200:]
        for key in self._NOD_TRIGGER_PLOT_KEYS:
            if not self._panel_visible.get(key, True):
                continue
            plot = self.plots.get(key)
            if plot is None:
                continue
            to_remove = getattr(plot, "_nod_trigger_overlay_items", [])
            for item in to_remove:
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
            new_items: list[Any] = []
            for region in active:
                start_ts = to_float_or_none(region.get("used_start_ts"))
                end_ts = to_float_or_none(region.get("used_end_ts"))
                trigger_ts = to_float_or_none(region.get("trigger_ts"))
                if start_ts is None or end_ts is None or trigger_ts is None:
                    continue
                x0 = max(float(start_ts), window_start) - float(now_ts)
                x1 = min(float(end_ts), float(now_ts)) - float(now_ts)
                if x1 < x0:
                    x1 = x0
                region_item = pg.LinearRegionItem(
                    values=(x0, x1),
                    orientation="vertical",
                    movable=False,
                    brush=pg.mkBrush(255, 220, 80, 35),
                    pen=pg.mkPen((180, 140, 20, 140), width=1),
                )
                region_item.setZValue(-10)
                plot.addItem(region_item)
                new_items.append(region_item)

                x_line = float(trigger_ts) - float(now_ts)
                line = pg.QtWidgets.QGraphicsLineItem(x_line, 0.0, x_line, 1.0)
                line.setPen(pg.mkPen((220, 60, 60), width=2))
                try:
                    vb = plot.getViewBox()
                    yr = vb.viewRange()[1]
                    line.setLine(x_line, float(yr[0]), x_line, float(yr[1]))
                except Exception:
                    pass
                plot.addItem(line)
                new_items.append(line)

                if key == "p_nod":
                    label = "発火"
                    bf = int(region.get("buffer_frames", 0) or 0)
                    cnt = int(region.get("count", 0) or 0)
                    if bf > 0 or cnt > 0:
                        label = f"発火（回数={cnt}、フレーム={bf}）"
                    txt = pg.TextItem(text=label, color=(220, 60, 60), anchor=(0.5, 1.0))
                    try:
                        vb = plot.getViewBox()
                        y_top = float(vb.viewRange()[1][1])
                    except Exception:
                        y_top = 1.0
                    txt.setPos(x_line, y_top)
                    plot.addItem(txt)
                    new_items.append(txt)
            setattr(plot, "_nod_trigger_overlay_items", new_items)
        self._nod_trigger_regions = active[-200:]

    def _redraw_bc_trigger_regions(self, now_ts: float) -> None:
        if not self._plot_enabled or not self.initialized:
            return
        plot = self.plots.get("p_bc")
        if plot is None or not self._panel_visible.get("p_bc", True):
            return
        old_items = getattr(plot, "_bc_trigger_overlay_items", [])
        for item in old_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass
        window_start = float(now_ts) - float(self.shown_context_sec)
        active = [
            region
            for region in self._bc_trigger_regions
            if to_float_or_none(region.get("trigger_ts")) is not None
            and float(region.get("trigger_ts", 0.0)) >= window_start
        ][-200:]
        new_items: list[Any] = []
        try:
            vb = plot.getViewBox()
            y0, y1 = [float(v) for v in vb.viewRange()[1]]
        except Exception:
            y0, y1 = 0.0, 1.0
        for region in active:
            start_ts = to_float_or_none(region.get("used_start_ts"))
            end_ts = to_float_or_none(region.get("used_end_ts"))
            trigger_ts = to_float_or_none(region.get("trigger_ts"))
            if start_ts is None or end_ts is None or trigger_ts is None:
                continue
            x0 = max(float(start_ts), window_start) - float(now_ts)
            x1 = min(float(end_ts), float(now_ts)) - float(now_ts)
            if x1 < x0:
                x1 = x0
            region_item = pg.LinearRegionItem(
                values=(x0, x1),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(255, 220, 80, 35),
                pen=pg.mkPen((180, 140, 20, 140), width=1),
            )
            region_item.setZValue(-10)
            plot.addItem(region_item)
            new_items.append(region_item)

            x_line = float(trigger_ts) - float(now_ts)
            line = pg.QtWidgets.QGraphicsLineItem(x_line, y0, x_line, y1)
            line.setPen(pg.mkPen((220, 60, 60), width=2))
            plot.addItem(line)
            new_items.append(line)

            bf = int(region.get("buffer_frames", 0) or 0)
            label = f"相槌（フレーム={bf}）" if bf > 0 else "相槌"
            txt = pg.TextItem(text=label, color=(220, 60, 60), anchor=(0.5, 1.0))
            txt.setPos(x_line, y1)
            plot.addItem(txt)
            new_items.append(txt)
        setattr(plot, "_bc_trigger_overlay_items", new_items)
        self._bc_trigger_regions = active

    def _append_wave_chunk(self, key: str, samples: np.ndarray, ts_start: float) -> None:
        chunks = self._wave_chunks.setdefault(key, deque())
        arr = np.asarray(samples, dtype=float).reshape(-1)
        if arr.size == 0:
            return
        chunks.append((float(ts_start), arr))
        # 少し余裕を持って保持し、表示窓外は破棄する。
        keep_after = float(ts_start) - float(self.shown_context_sec) - 2.0
        while chunks:
            head_ts, head_arr = chunks[0]
            head_end = float(head_ts) + (float(head_arr.size) / float(max(1, self.sample_rate)))
            if head_end >= keep_after:
                break
            chunks.popleft()

    def _wave_chunks_xy(self, key: str, now_ts: float) -> tuple[np.ndarray, np.ndarray]:
        chunks = self._wave_chunks.get(key)
        if not chunks:
            return self._x_wave, np.zeros(self.MAX_CONTEXT_WAV_LEN, dtype=float)
        x_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []
        window_start = float(now_ts) - float(self.shown_context_sec)
        sr = float(max(1, self.sample_rate))
        # 入力は通常 ts 順で到着するため、毎描画の全件 sort を避けて負荷を下げる。
        # （遅延イベントは controller 側で可能な限り除外済み）
        for ts0, arr in chunks:
            n = int(arr.size)
            if n <= 0:
                continue
            t = float(ts0) + (np.arange(n, dtype=float) / sr)
            mask = t >= window_start
            if not np.any(mask):
                continue
            t = t[mask]
            a = arr[mask]
            x_all.append(t - float(now_ts))
            y_all.append(a)
        if not x_all:
            return self._x_wave, np.zeros(self.MAX_CONTEXT_WAV_LEN, dtype=float)
        x = np.concatenate(x_all)
        y = np.concatenate(y_all)
        if x.size > self.MAX_CONTEXT_WAV_LEN:
            x = x[-self.MAX_CONTEXT_WAV_LEN :]
            y = y[-self.MAX_CONTEXT_WAV_LEN :]
        return x, y

    def _redraw_waveforms(self, now_ts: float) -> None:
        for key in ("x1", "x2"):
            if not self._panel_visible.get(key, True):
                continue
            handle = self.curves.get(key)
            if handle is None:
                continue
            wx, wy = self._wave_chunks_xy(key, now_ts)
            if isinstance(handle, dict):
                wave = handle.get("wave")
                if wave is not None:
                    wave.setData(wx, wy)
            else:
                handle.setData(wx, wy)

    def _init_voice_latency_plot(self, row: int) -> None:
        if (not self._plot_enabled) or self.graph is None:
            return
        plot = self.graph.addPlot(row=row, col=0)
        self._voice_latency_plot = plot
        self._configure_plot(
            plot,
            title="音声対話レイテンシ（ASR／LLM／TTS）＋ターン／先読み／割り込み",
            x_range=(-self.shown_context_sec, 0.0),
        )
        plot.setYRange(0.0, 1.0, padding=0.02)
        plot.getAxis("left").setTicks([[]])
        self._redraw_voice_latency()

    def _redraw_voice_latency(self, now_ts: float | None = None) -> None:
        plot = self._voice_latency_plot
        if plot is None:
            return
        if not self._panel_visible.get("voice_latency", True):
            return
        # Keep action timestamps intact; use the same viewport origin as the
        # prediction curves and waveforms, including immediate event redraws.
        now_ts = self._audio_now_ts() if now_ts is None else float(now_ts)
        window_start = float(now_ts) - float(self.shown_context_sec)
        plot.clear()
        self._configure_plot(
            plot,
            title="音声対話レイテンシ（ASR／LLM／TTS）＋ターン／先読み／割り込み",
            x_range=(-self.shown_context_sec, 0.0),
        )
        plot.setYRange(0.0, 1.0, padding=0.02)
        plot.getAxis("left").setTicks([[]])
        # 下段にターン取得ロック（背景）→ その上に ASR/LLM/TTS を同一帯へ重畳（色のみ区別）
        self._draw_turn_lock_band(plot, now_ts, window_start)
        lane_rgb = {
            "asr": (245, 189, 0),
            "llm": (120, 220, 120),
            "tts": (230, 110, 220),
        }
        span_y0, span_h = 0.09, 0.88
        active: list[dict[str, Any]] = []
        drawn_span_keys: set[tuple[str, int, int]] = set()
        for span in self._voice_latency_spans:
            kind = str(span.get("kind", "")).strip().lower()
            if kind not in lane_rgb:
                continue
            start_ts = to_float_or_none(span.get("start_ts"))
            end_ts = to_float_or_none(span.get("end_ts"))
            if start_ts is None or end_ts is None:
                continue
            if end_ts < start_ts:
                end_ts = start_ts
            if end_ts < window_start:
                continue
            span_key = (
                kind,
                int(round(float(start_ts) * 1000.0)),
                int(round(float(end_ts) * 1000.0)),
            )
            if span_key in drawn_span_keys:
                continue
            drawn_span_keys.add(span_key)
            active.append({"kind": kind, "start_ts": start_ts, "end_ts": end_ts})
            clip_s = max(start_ts, window_start)
            clip_e = min(end_ts, float(now_ts))
            if clip_e < clip_s:
                continue
            x0 = clip_s - float(now_ts)
            width = max(0.003, clip_e - clip_s)
            rgb = lane_rgb[kind]
            rect = pg.QtWidgets.QGraphicsRectItem(x0, span_y0, width, span_h)
            rect.setBrush(pg.mkBrush(QColor(rgb[0], rgb[1], rgb[2], 130)))
            rect.setPen(pg.mkPen((40, 40, 40), width=1))
            plot.addItem(rect)
            dur_ms = max(0.0, (end_ts - start_ts) * 1000.0)
            txt = pg.TextItem(text=f"{dur_ms:.0f}ms", color=(20, 20, 20), anchor=(0.0, 0.5))
            txt.setPos(x0 + 0.005, span_y0 + (span_h / 2.0))
            plot.addItem(txt)
        self._voice_latency_spans = active[-300:]
        # p_future 先読み開始など、まだ閉じられていない LLM 帯を now まで伸ばして描く。
        open_llm = self._voice_llm_open_start
        if open_llm is not None:
            open_s = float(open_llm)
            if open_s <= float(now_ts):
                clip_s = max(open_s, window_start)
                clip_e = float(now_ts)
                if clip_e >= clip_s:
                    x0 = clip_s - float(now_ts)
                    width = max(0.003, clip_e - clip_s)
                    rgb = lane_rgb["llm"]
                    rect = pg.QtWidgets.QGraphicsRectItem(x0, span_y0, width, span_h)
                    rect.setBrush(pg.mkBrush(QColor(rgb[0], rgb[1], rgb[2], 90)))
                    rect.setPen(pg.mkPen((40, 40, 40), width=1))
                    plot.addItem(rect)
                    dur_ms = max(0.0, (clip_e - open_s) * 1000.0)
                    txt = pg.TextItem(
                        text=f"{dur_ms:.0f}ms…", color=(20, 20, 20), anchor=(0.0, 0.5)
                    )
                    txt.setPos(x0 + 0.005, span_y0 + (span_h / 2.0))
                    plot.addItem(txt)
        # ターン取得タイミングを同一時系列へ重畳表示する。
        active_turns: list[float] = []
        for ts in self._voice_turn_events:
            t = float(ts)
            if t < window_start:
                continue
            active_turns.append(t)
            x = t - float(now_ts)
            line = pg.InfiniteLine(
                pos=x,
                angle=90,
                pen=pg.mkPen((255, 255, 255), width=1.2, style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            plot.addItem(line)
            label = pg.TextItem(text="ターン", color=(235, 235, 235), anchor=(0.5, 1.0))
            label.setPos(x, 0.98)
            plot.addItem(label)
        self._voice_turn_events = active_turns[-300:]
        active_pref: list[float] = []
        for ts in self._voice_prefetch_events:
            t = float(ts)
            if t < window_start:
                continue
            active_pref.append(t)
            x = t - float(now_ts)
            pref_line = pg.InfiniteLine(
                pos=x,
                angle=90,
                pen=pg.mkPen((80, 210, 255), width=1.3, style=pg.QtCore.Qt.PenStyle.DashDotLine),
            )
            plot.addItem(pref_line)
            pref_label = pg.TextItem(text="先読み", color=(140, 230, 255), anchor=(0.5, 1.0))
            pref_label.setPos(x, 0.88)
            plot.addItem(pref_label)
        self._voice_prefetch_events = active_pref[-300:]
        active_intr: list[float] = []
        for ts in self._voice_interrupt_action_ts:
            t = float(ts)
            if t < window_start:
                continue
            active_intr.append(t)
            x = t - float(now_ts)
            intr_line = pg.InfiniteLine(
                pos=x,
                angle=90,
                pen=pg.mkPen((255, 65, 65), width=1.4, style=pg.QtCore.Qt.PenStyle.DotLine),
            )
            plot.addItem(intr_line)
            intr_label = pg.TextItem(text="割込", color=(255, 150, 140), anchor=(0.5, 1.0))
            intr_label.setPos(x, 0.93)
            plot.addItem(intr_label)
        self._voice_interrupt_action_ts = active_intr[-300:]

    def _turn_locked_at_wall_time(self, t: float) -> bool:
        state = False
        for et, es in self._turn_lock_edges:
            if float(et) <= float(t):
                state = bool(es)
            else:
                break
        return state

    def _draw_turn_lock_band(self, plot: Any, now_ts: float, window_start: float) -> None:
        if not self._turn_lock_edges:
            return
        y0, h = 0.0, 0.085
        rgb = (220, 90, 70)
        boundary_ts: list[float] = [window_start]
        for et, _es in self._turn_lock_edges:
            ft = float(et)
            if window_start < ft <= float(now_ts):
                boundary_ts.append(ft)
        boundary_ts.append(float(now_ts))
        boundary_ts = sorted(set(boundary_ts))
        for i in range(len(boundary_ts) - 1):
            t_mid = (boundary_ts[i] + boundary_ts[i + 1]) * 0.5
            if not self._turn_locked_at_wall_time(t_mid):
                continue
            clip_s = max(boundary_ts[i], window_start)
            clip_e = min(boundary_ts[i + 1], float(now_ts))
            if clip_e <= clip_s:
                continue
            x0 = clip_s - float(now_ts)
            width = max(0.003, clip_e - clip_s)
            rect = pg.QtWidgets.QGraphicsRectItem(x0, y0, width, h)
            rect.setBrush(pg.mkBrush(QColor(rgb[0], rgb[1], rgb[2], 100)))
            rect.setPen(pg.mkPen((60, 40, 40), width=1))
            plot.addItem(rect)
        if self._turn_locked_at_wall_time(float(now_ts)):
            label = pg.TextItem(
                text="ターン取得ブロック（ユーザー発話待ち）",
                color=(255, 220, 200),
                anchor=(0.0, 0.0),
            )
            label.setPos(-float(self.shown_context_sec) + 0.02, y0 + 0.012)
            plot.addItem(label)

    def update_turn_lock_state(self, locked: bool, event_ts: float | None = None) -> None:
        """DialogOrchestrator の turn_blocked_until_user_speech を可視化するためのエッジを記録する。"""
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        ts = float(event_ts) if event_ts is not None else time.time()
        if self._turn_lock_edges and self._turn_lock_edges[-1][1] == bool(locked):
            return
        self._turn_lock_edges.append((ts, bool(locked)))
        if len(self._turn_lock_edges) > 400:
            self._turn_lock_edges = self._turn_lock_edges[-300:]
        if not self.initialized:
            self.init_full_layout()
        if self._panel_visible.get("voice_latency", True):
            self._redraw_voice_latency()

    def _init_graph(self, result: dict[str, Any], *, layout_keys: list[str] | None = None) -> None:
        if (not self._plot_enabled) or self.graph is None:
            self.initialized = False
            return
        self.keys = list(layout_keys) if layout_keys is not None else self._infer_keys(result)
        self._p_curve_split_fp.clear()
        self._p_curve_split_cached.clear()
        self._p_bins_draw_cache_arr = None
        self._p_bins_draw_cache_times = None
        self.graph.clear()
        self.plots = {}
        self.curves = {}
        self.data_buffer = {}
        self.asr_plot = None

        left_items = self._left_row_items_for_keys(self.keys)
        right_items = self._right_row_items_for_keys(self.keys)
        layout_entries: list[tuple[int, int, str]] = [
            *( (0, row, key) for row, key in enumerate(left_items) ),
            *( (1, row, key) for row, key in enumerate(right_items) ),
        ]

        for col, row, key in layout_entries:
            plot = self.graph.addPlot(row=row, col=col)
            if key == "__asr__":
                self.asr_plot = plot
                self._configure_plot(plot, title="ASRセグメント", x_range=(-self.shown_context_sec, 0.0))
                plot.setYRange(0.0, 1.0, padding=0.0)
                plot.getAxis("left").setTicks([[]])
                continue
            if key == "__voice_latency__":
                self._voice_latency_plot = plot
                self._configure_plot(
                    plot,
                    title="音声対話レイテンシ（ASR／LLM／TTS）＋ターン／先読み／割り込み",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(0.0, 1.0, padding=0.02)
                plot.getAxis("left").setTicks([[]])
                continue

            self.plots[key] = plot
            val = result.get(key, [0.0])
            if key == "x1":
                self._configure_plot(plot, title="入力波形1", x_range=(-self.shown_context_sec, 0.0))
                plot.setYRange(-1.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_WAV_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                wave_curve = plot.plot(self._x_wave, buf, pen=pg.mkPen((220, 200, 40), width=1.0))
                self._tune_wave_plot_item(wave_curve)
                self.curves[key] = wave_curve
            elif key == "x2":
                self._configure_plot(plot, title="入力波形2", x_range=(-self.shown_context_sec, 0.0))
                plot.setYRange(-1.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_WAV_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                wave_curve = plot.plot(self._x_wave, buf, pen=pg.mkPen((80, 160, 240), width=1.0))
                self._tune_wave_plot_item(wave_curve)
                self.curves[key] = wave_curve
            elif key == "p_now":
                self._configure_plot(
                    plot,
                    title="p_now（緑破線：ターンしきい値／赤破線：割り込みしきい値）",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(0.0, 1.0, padding=0.0)
                self._set_inverted_probability_axis_ticks(plot)
                plot.addLine(y=0.5, pen=pg.mkPen((120, 120, 120), style=pg.QtCore.Qt.PenStyle.DashLine))
                buf = np.ones(self.MAX_CONTEXT_LEN, dtype=float) * 0.5
                self.data_buffer[key] = list(buf)
                x_fill, hi, lo = self._split_by_threshold_interpolated(self._x_ctx, buf, 0.5)
                high_curve = plot.plot(
                    x_fill,
                    hi,
                    pen=pg.mkPen((245, 189, 0), width=1.5),
                    fillLevel=0.5,
                    brush=pg.mkBrush(245, 189, 0, 90),
                )
                low_curve = plot.plot(
                    x_fill,
                    lo,
                    pen=pg.mkPen((80, 160, 240), width=1.5),
                    fillLevel=0.5,
                    brush=pg.mkBrush(80, 160, 240, 90),
                )
                self._tune_maai_time_series_item(high_curve)
                self._tune_maai_time_series_item(low_curve)
                self.curves[key] = {"high": high_curve, "low": low_curve}
                self._threshold_lines["turn_p_now"] = plot.addLine(
                    y=float(1.0 - self._turn_p_now_threshold),
                    pen=pg.mkPen((80, 200, 120), width=1.2, style=pg.QtCore.Qt.PenStyle.DashLine),
                )
                self._threshold_lines["interrupt_p_now"] = plot.addLine(
                    y=float(1.0 - self._interrupt_p_now_threshold),
                    pen=pg.mkPen((230, 90, 90), width=1.2, style=pg.QtCore.Qt.PenStyle.DashLine),
                )
            elif key == "p_future":
                self._configure_plot(
                    plot,
                    title="出力p_future（長期ターン予測）",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(0.0, 1.0, padding=0.0)
                self._set_inverted_probability_axis_ticks(plot)
                plot.addLine(y=0.5, pen=pg.mkPen((120, 120, 120), style=pg.QtCore.Qt.PenStyle.DashLine))
                buf = np.ones(self.MAX_CONTEXT_LEN, dtype=float) * 0.5
                self.data_buffer[key] = list(buf)
                x_fill, hi, lo = self._split_by_threshold_interpolated(self._x_ctx, buf, 0.5)
                high_curve = plot.plot(
                    x_fill,
                    hi,
                    pen=pg.mkPen((245, 120, 0), width=1.5),
                    fillLevel=0.5,
                    brush=pg.mkBrush(245, 189, 0, 80),
                )
                low_curve = plot.plot(
                    x_fill,
                    lo,
                    pen=pg.mkPen((80, 160, 240), width=1.5),
                    fillLevel=0.5,
                    brush=pg.mkBrush(80, 160, 240, 90),
                )
                self._tune_maai_time_series_item(high_curve)
                self._tune_maai_time_series_item(low_curve)
                self.curves[key] = {"high": high_curve, "low": low_curve}
            elif key == "p_bins":
                self._configure_plot(plot, title="p_bins（現在時刻の区間別確率）", x_range=(0.0, 1.0))
                self.data_buffer[key] = np.array(val, dtype=float)
                self._maybe_draw_p_bins(self.data_buffer[key], result.get("bin_times"))
            elif key == "vad":
                self._configure_plot(plot, title="音声活動検出（VAD）", x_range=(-self.shown_context_sec, 0.0))
                plot.setYRange(-1.0, 1.0, padding=0.0)
                plot.addLine(y=0.0, pen=pg.mkPen((120, 120, 120), style=pg.QtCore.Qt.PenStyle.DashLine))
                b1 = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                b2 = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                self.data_buffer[key] = [list(b1), list(b2)]
                c1 = plot.plot(
                    self._x_ctx,
                    b1,
                    pen=pg.mkPen((245, 189, 0), width=1.5),
                    fillLevel=0.0,
                    brush=pg.mkBrush(245, 189, 0, 90),
                )
                c2 = plot.plot(
                    self._x_ctx,
                    -b2,
                    pen=pg.mkPen((80, 160, 240), width=1.5),
                    fillLevel=0.0,
                    brush=pg.mkBrush(80, 160, 240, 90),
                )
                self._tune_maai_time_series_item(c1)
                self._tune_maai_time_series_item(c2)
                self.curves[key] = (c1, c2)
            elif key == "silero_vad_score":
                self._configure_plot(
                    plot,
                    title="Silero VAD（緑：開始／青：終了／赤：割り込み）",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(0.0, 1.0, padding=0.0)
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                _silero_curve = plot.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((236, 112, 99), width=1.8),
                    fillLevel=0.0,
                    brush=pg.mkBrush(236, 112, 99, 70),
                )
                self._tune_maai_time_series_item(_silero_curve)
                self.curves[key] = _silero_curve
                self._threshold_lines["vad_on"] = plot.addLine(
                    y=float(self._vad_on_threshold),
                    pen=pg.mkPen((60, 180, 90), width=1.1, style=pg.QtCore.Qt.PenStyle.DotLine),
                )
                self._threshold_lines["vad_off"] = plot.addLine(
                    y=float(self._vad_off_threshold),
                    pen=pg.mkPen((90, 150, 240), width=1.1, style=pg.QtCore.Qt.PenStyle.DotLine),
                )
                self._threshold_lines["interrupt_vad"] = plot.addLine(
                    y=float(self._interrupt_vad_threshold),
                    pen=pg.mkPen((230, 90, 90), width=1.2, style=pg.QtCore.Qt.PenStyle.DashLine),
                )
            elif key == "nod_range":
                self._configure_plot(
                    plot,
                    title="頷き幅（nod_range、0.035～0.15）",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(0.035, 0.15, padding=0.0)
                plot.getAxis("left").setTicks(
                    [[(0.035, "0.035"), (0.093, "0.093"), (0.15, "0.15")]]
                )
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                _nr = plot.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((180, 140, 220), width=1.6),
                )
                self._tune_maai_time_series_item(_nr)
                self.curves[key] = _nr
            elif key == "nod_speed":
                self._configure_plot(
                    plot,
                    title="頷き速度（nod_speed、0.08～0.25）",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(0.08, 0.25, padding=0.0)
                plot.getAxis("left").setTicks(
                    [[(0.08, "0.08"), (0.165, "0.165"), (0.25, "0.25")]]
                )
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                _nsp = plot.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((100, 200, 220), width=1.6),
                )
                self._tune_maai_time_series_item(_nsp)
                self.curves[key] = _nsp
            elif key == "nod_count":
                # stepMode では len(x)==len(y)+1 が必要なため、等長バッファ初期化と相性が悪い。折れ線で更新する。
                self._configure_plot(
                    plot,
                    title="頷き回数（nod_count、クラス1／2／3以上）",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(-0.15, 2.15, padding=0.0)
                plot.getAxis("left").setTicks([[(0.0, "1"), (1.0, "2"), (2.0, "3+")]])
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                _nc = plot.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((255, 170, 50), width=2.0),
                )
                self._tune_maai_time_series_item(_nc)
                self.curves[key] = _nc
            elif key == "nod_swing_up":
                # p_bc と同様のレイアウト（生確率・塗りつぶし・閾値線）。系列色はピンク。
                self._configure_plot(
                    plot,
                    title="頷き上振り確率（nod_swing_up）",
                    x_range=(-self.shown_context_sec, 0.0),
                )
                plot.setYRange(0.0, 1.0, padding=0.0)
                plot.addLine(y=0.5, pen=pg.mkPen((120, 120, 120), style=pg.QtCore.Qt.PenStyle.DashLine))
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                _nsu = plot.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((240, 100, 160), width=1.5),
                    fillLevel=0.0,
                    brush=pg.mkBrush(240, 100, 160, 80),
                )
                self._tune_maai_time_series_item(_nsu)
                self.curves[key] = _nsu
                self._threshold_lines["nod_swing_up_threshold"] = plot.addLine(
                    y=0.5,
                    pen=pg.mkPen((230, 90, 90), width=1.2, style=pg.QtCore.Qt.PenStyle.DashLine),
                )
            elif key.startswith("p_"):
                probability_title = {
                    "p_bc": "相槌確率（p_bc）",
                    "p_nod": "頷き確率（p_nod）",
                }.get(key, key)
                self._configure_plot(plot, title=probability_title, x_range=(-self.shown_context_sec, 0.0))
                plot.setYRange(0.0, 1.0, padding=0.0)
                plot.addLine(y=0.5, pen=pg.mkPen((120, 120, 120), style=pg.QtCore.Qt.PenStyle.DashLine))
                buf = np.zeros(self.MAX_CONTEXT_LEN, dtype=float)
                self.data_buffer[key] = list(buf)
                _pk = plot.plot(
                    self._x_ctx,
                    buf,
                    pen=pg.mkPen((80, 200, 120), width=1.5),
                    fillLevel=0.0,
                    brush=pg.mkBrush(80, 200, 120, 80),
                )
                self._tune_maai_time_series_item(_pk)
                self.curves[key] = _pk
                if key == "p_bc":
                    self._threshold_lines["p_bc_threshold"] = plot.addLine(
                        y=0.5,
                        pen=pg.mkPen((230, 90, 90), width=1.2, style=pg.QtCore.Qt.PenStyle.DashLine),
                    )
            else:
                arr = np.array(val, dtype=float).reshape(-1) if isinstance(val, (np.ndarray, list, tuple)) else np.zeros(1)
                if arr.size <= 1:
                    arr = np.zeros(8, dtype=float)
                self._configure_plot(plot, title=key, x_range=(0.0, float(arr.size)))
                self.data_buffer[key] = list(arr)
                _gen = plot.plot(np.arange(arr.size), arr, pen=pg.mkPen((120, 220, 120), width=1.0))
                self._tune_maai_time_series_item(_gen)
                self.curves[key] = _gen

        if self._voice_latency_plot is None:
            self._init_voice_latency_plot(len(left_items))

        self.initialized = True
        self._relayout_panels()
        self._redraw_voice_latency()
        self._redraw_asr_segments_axis(draw=True)

    def _engine_color(self, engine: str) -> tuple[int, int, int]:
        fixed = {
            "qwen3-asr": (31, 119, 180),
            "external-asr-tcp": (214, 39, 40),
            "chirp3-asr": (44, 160, 44),
            "faster-whisper": (255, 127, 14),
            "sherpa-onnx": (148, 103, 189),
        }
        key = str(engine or "asr").strip().lower()
        if key in fixed:
            return fixed[key]
        palette = [
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
            (188, 189, 34),
            (23, 190, 207),
        ]
        digest = hashlib.md5(key.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % len(palette)
        return palette[idx]

    def _draw_p_bins(self, arr: np.ndarray, bin_times_value: Any) -> None:
        if not self._plot_enabled:
            return
        if not self._panel_visible.get("p_bins", True):
            return
        plot = self.plots.get("p_bins")
        if plot is None:
            return
        arr = np.array(arr, dtype=float)
        plot.clear()
        if arr.ndim != 2 or arr.shape[0] != 2:
            plot.setTitle(f"p_bins（不正な形状：{arr.shape}）")
            return
        n_bins = int(arr.shape[1])
        try:
            bin_times = [float(x) for x in (bin_times_value or DEFAULT_VAP_BIN_TIMES_SEC)]
        except Exception:
            bin_times = list(DEFAULT_VAP_BIN_TIMES_SEC)
        if len(bin_times) != n_bins:
            bin_times = [1.0 for _ in range(n_bins)]
        self._p_bins_bin_times = list(bin_times)
        edges = np.concatenate([[0.0], np.cumsum(bin_times)])

        plot.setTitle("p_bins（現在時刻の区間別確率）")
        plot.setXRange(0.0, float(edges[-1]), padding=0.0)
        plot.setYRange(0.0, 2.0, padding=0.0)
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.addLine(y=1.0, pen=pg.mkPen((130, 130, 130), width=1))
        for x in edges[1:-1]:
            plot.addLine(x=float(x), pen=pg.mkPen((130, 130, 130), style=pg.QtCore.Qt.PenStyle.DashLine))

        for s in range(2):
            y0 = 1.0 if s == 0 else 0.0
            for b in range(n_bins):
                v = float(np.clip(arr[s, b], 0.0, 1.0))
                if not np.isfinite(v):  # np.clip は NaN を保持し int(NaN) が落ちるため
                    v = 0.0
                x0 = float(edges[b])
                w = float(bin_times[b])
                hue_base = (70, 140, 230) if s == 0 else (250, 150, 40)
                color = (
                    int(hue_base[0] * (0.35 + 0.65 * v)),
                    int(hue_base[1] * (0.35 + 0.65 * v)),
                    int(hue_base[2] * (0.35 + 0.65 * v)),
                )
                bar = pg.BarGraphItem(
                    x=[x0 + (w / 2.0)],
                    width=[w],
                    y0=[y0],
                    height=[1.0],
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen((30, 30, 30), width=0.8),
                )
                plot.addItem(bar)
                txt_color = (255, 255, 255) if v >= 0.55 else (20, 20, 20)
                txt = pg.TextItem(text=f"{v:.2f}", color=txt_color, anchor=(0.5, 0.5))
                txt.setPos(x0 + (w / 2.0), y0 + 0.5)
                plot.addItem(txt)

    def _note_audio_now(self, ts: float) -> None:
        """最新の音声イベント時刻（audio/input_worker のサンプルクロック）を記録。

        波形軸はイベント配送遅延を避けるため壁時計ではなくこの音声時刻を「now（右端）」
        に使う。ASR セグメント帯など、波形と同じタイムラインに重ねるパネルも同じ now を
        使うことで、配送遅延ぶんの定常ズレを無くす。"""
        prev = self._last_source_ts_end
        if prev is None or ts > prev:
            self._last_source_ts_end = float(ts)
            self._last_source_ts_wall = time.time()

    def _audio_now_ts(self) -> float:
        """波形軸と同じ「now」。

        オーディオイベントが届いている間はそのサンプルクロックを返す。
        オーディオが止まった後も壁時計で進め続けることで、停止中も全パネルが
        同一の「now」でスクロールし続ける。
        """
        ts = self._last_source_ts_end
        if ts is None:
            return time.time()
        wall = getattr(self, "_last_source_ts_wall", None)
        if wall is None:
            return float(ts)
        return float(ts) + max(0.0, time.time() - float(wall))

    def _redraw_asr_segments_axis(self, draw: bool) -> None:
        del draw
        if not self._plot_enabled:
            return
        if self.asr_plot is None:
            return
        if not self._panel_visible.get("asr", True):
            return
        now_ts = self._audio_now_ts()
        window_start = now_ts - float(self.shown_context_sec)
        plot = self.asr_plot
        plot.clear()
        self._configure_plot(plot, title="ASRセグメント", x_range=(-self.shown_context_sec, 0.0))
        plot.setYRange(0.0, 1.0, padding=0.0)
        plot.getAxis("left").setTicks([[]])

        active: list[dict[str, Any]] = []
        for seg in self._asr_segments:
            s = to_float_or_none(seg.get("start_ts"))
            e = to_float_or_none(seg.get("end_ts"))
            if s is None:
                continue
            if e is None:
                e = now_ts
            if self.align_segment_right_edge_to_arrival:
                arrival_ts = to_float_or_none(seg.get("arrival_ts"))
                if arrival_ts is None:
                    arrival_ts = to_float_or_none(seg.get("final_arrival_ts"))
                if arrival_ts is not None:
                    shift = float(arrival_ts) - float(e)
                    s = float(s) + shift
                    e = float(e) + shift
            if e < s:
                e = s
            if e < window_start:
                continue
            active.append(seg)
            clip_s = max(s, window_start)
            clip_e = min(e, now_ts)
            width = max(0.001, clip_e - clip_s)
            x0 = clip_s - now_ts
            color_rgb = self._engine_color(str(seg.get("engine", "asr")))
            is_final = bool(seg.get("is_final", True))
            alpha = 170 if is_final else 110

            rect = pg.QtWidgets.QGraphicsRectItem(x0, 0.15, width, 0.7)
            color = QColor(color_rgb[0], color_rgb[1], color_rgb[2], alpha)
            rect.setBrush(pg.mkBrush(color))
            rect.setPen(pg.mkPen((30, 30, 30), width=1))
            plot.addItem(rect)

            label = str(seg.get("text", "")).strip()
            if label:
                tx = max(x0 + 0.02, -self.shown_context_sec + 0.02)
                txt = pg.TextItem(text=label, color=(20, 20, 20), anchor=(0.0, 0.5))
                txt.setPos(tx, 0.5)
                plot.addItem(txt)
        self._asr_segments = active[-200:]
        # final受信時刻マーカー（受信した瞬間の時刻）を重ねる。
        marker_active: list[dict[str, Any]] = []
        for m in self._final_arrivals:
            ts = to_float_or_none(m.get("ts"))
            if ts is None or ts < window_start:
                continue
            marker_active.append(m)
            x = float(ts) - now_ts
            engine = str(m.get("engine", "asr"))
            color_rgb = self._engine_color(engine)
            line = pg.QtWidgets.QGraphicsLineItem(x, 0.05, x, 0.95)
            line.setPen(pg.mkPen((color_rgb[0], color_rgb[1], color_rgb[2]), width=2))
            plot.addItem(line)
            txt = pg.TextItem(text="確定", color=(20, 20, 20), anchor=(0.5, 1.0))
            txt.setPos(x, 0.98)
            plot.addItem(txt)
        self._final_arrivals = marker_active[-200:]
        # 古いセグメント剪定後に open-stream 参照がズレないよう、生存オブジェクトだけ残す。
        alive_ids = {id(seg) for seg in self._asr_segments}
        self._open_stream_segments = {
            key: seg
            for key, seg in self._open_stream_segments.items()
            if id(seg) in alive_ids
        }
        self._align_plot_left_axes()

    def update_raw_audio_frame(
        self,
        *,
        audio: Any,
        ts_start: Any = None,
        ts_end: Any = None,
        compare_slot: str = "",
        system_audio: bool = False,
    ) -> None:
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        if system_audio:
            key = "x2"
        elif str(compare_slot).strip().lower() == "bottom":
            key = "x2"
        else:
            key = "x1"
        if not self.initialized:
            self.init_full_layout()
        if key not in self.curves:
            return

        x = np.array(audio, dtype=float).reshape(-1)
        if x.size == 0:
            return
        start_ts = to_float_or_none(ts_start)
        end_ts = to_float_or_none(ts_end)
        if start_ts is None and end_ts is not None:
            start_ts = float(end_ts) - (float(x.size) / float(max(1, self.sample_rate)))
        if start_ts is None:
            start_ts = time.time() - (float(x.size) / float(max(1, self.sample_rate)))
        if end_ts is not None:
            self._note_audio_now(float(end_ts))
        self._append_wave_chunk(key, x, float(start_ts))
        # x1/x2 の波形描画は `_wave_chunks`（直上の `_append_wave_chunk`）が担当する。
        # `data_buffer[key]` を読むのは `flush_maai_draw` の
        # `waveform_from_raw_audio == False` の分岐だけなので、読む側と同じ条件で守る。
        #
        # 以前は無条件に走っており、**毎フレーム 16 万要素（sample_rate 16000 ×
        # shown_context_sec 10）の Python リストを作り直していた**。誰も読まない値のために
        # 1 フレームあたり数 MB を確保し続けるので、生音声が速く届く構成
        # （qwen3-asr など）では Qt のキューが詰まって MemoryError で落ちた。
        if not self.waveform_from_raw_audio:
            buf = self.data_buffer.get(key, [])
            if not isinstance(buf, list):
                buf = list(np.array(buf, dtype=float).reshape(-1))
            buf = (buf + list(x))[-self.MAX_CONTEXT_WAV_LEN:]
            self.data_buffer[key] = buf

        draw = self._should_draw_now("raw")
        if draw and (
            self._panel_visible.get("x1", True) or self._panel_visible.get("x2", True)
        ):
            # イベント配送遅延があると、現在時刻基準では右端(0秒)より手前で
            # 波形が途切れて見えることがある。波形軸はイベント時刻(ts_end)を優先する。
            wave_now_ts = float(end_ts) if end_ts is not None else time.time()
            # x1/x2 のどちらかだけ更新イベントが来ても、両方を同時に再描画して
            # 無音区間を含めて時間軸が流れ続けるようにする。
            self._redraw_waveforms(wave_now_ts)

    def update_asr_segment(
        self,
        *,
        engine: str,
        text: str,
        segment_start_ts: Any,
        segment_end_ts: Any,
        is_final: bool,
        is_streaming: bool,
        compare_slot: str = "",
        final_arrival_ts: Any = None,
    ) -> None:
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        now_ts = self._audio_now_ts()
        start_ts = to_float_or_none(segment_start_ts)
        end_ts = to_float_or_none(segment_end_ts)
        if start_ts is None and end_ts is None:
            start_ts = now_ts
            end_ts = now_ts
        elif start_ts is None:
            start_ts = end_ts
        elif end_ts is None:
            end_ts = now_ts
        if start_ts is None:
            start_ts = now_ts
        if end_ts is None:
            end_ts = start_ts
        if end_ts < start_ts:
            end_ts = start_ts

        slot = (compare_slot or "").strip().lower()
        stream_key = (slot, (engine or "asr").strip())
        arrival_ts = to_float_or_none(final_arrival_ts)
        if arrival_ts is None:
            arrival_ts = now_ts
        if bool(is_final):
            self._final_arrivals.append({"ts": float(arrival_ts), "engine": engine})
        if is_streaming and (not is_final):
            seg = self._open_stream_segments.get(stream_key)
            if seg is not None and any(seg is s for s in self._asr_segments):
                seg["text"] = text
                seg["end_ts"] = end_ts
                seg["is_final"] = False
                seg["arrival_ts"] = float(arrival_ts)
            else:
                seg = {
                    "engine": engine,
                    "text": text,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "is_final": False,
                    "arrival_ts": float(arrival_ts),
                }
                self._asr_segments.append(seg)
                self._open_stream_segments[stream_key] = seg
        else:
            seg = self._open_stream_segments.pop(stream_key, None)
            if seg is not None and any(seg is s for s in self._asr_segments):
                seg["text"] = text
                seg["end_ts"] = end_ts
                seg["is_final"] = bool(is_final)
                seg["arrival_ts"] = float(arrival_ts)
                if bool(is_final):
                    seg["final_arrival_ts"] = arrival_ts
            else:
                self._asr_segments.append(
                    {
                        "engine": engine,
                        "text": text,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "is_final": bool(is_final),
                        "arrival_ts": float(arrival_ts),
                        "final_arrival_ts": arrival_ts if bool(is_final) else None,
                    }
                )

        # 非表示時もリストが無制限に増えないよう件数上限を無条件で適用する。
        if len(self._asr_segments) > 6000:
            del self._asr_segments[:-3000]
        if len(self._final_arrivals) > 6000:
            del self._final_arrivals[:-3000]
        if not self.initialized:
            self.init_full_layout()
        if self._panel_visible.get("asr", True) or self._panel_visible.get("voice_latency", True):
            self._redraw_voice_latency()
            self._redraw_asr_segments_axis(draw=True)

    def _voice_latency_span_is_duplicate(self, kind: str, start_ts: float, end_ts: float) -> bool:
        """同一 kind・ほぼ同一の時間幅の帯が既にあれば真（二重 emit / 再計測の重複描画を抑止）。"""
        ks = str(kind).strip().lower()
        for sp in self._voice_latency_spans[-150:]:
            if str(sp.get("kind", "")).strip().lower() != ks:
                continue
            try:
                s = float(sp.get("start_ts", 0.0))
                e = float(sp.get("end_ts", 0.0))
            except (TypeError, ValueError):
                continue
            if abs(s - start_ts) < 2e-3 and abs(e - end_ts) < 2e-3:
                return True
        return False

    def _voice_latency_merge_same_start(self, kind: str, start_ts: float, end_ts: float) -> bool:
        """同一 kind で開始時刻がほぼ同じ帯があれば end_ts を更新する（開始同一・終了のみ異なる重複を1本に）。"""
        ks = str(kind).strip().lower()
        # ウォールクロック由来のわずかな差で同一ターンテイキングの二重帯が残るのを避ける
        eps = 0.05
        for sp in reversed(self._voice_latency_spans[-200:]):
            if str(sp.get("kind", "")).strip().lower() != ks:
                continue
            try:
                s = float(sp.get("start_ts", 0.0))
            except (TypeError, ValueError):
                continue
            if abs(s - start_ts) < eps:
                sp["end_ts"] = float(end_ts)
                return True
        return False

    def update_voice_dialog_latency(
        self,
        *,
        kind: str,
        latency_sec: float | None,
        event_ts: float | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> None:
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        kind_key = str(kind).strip().lower()
        if kind_key == "interrupt":
            ev_ts = float(event_ts) if event_ts is not None else time.time()
            e_ts = to_float_or_none(end_ts)
            ts = float(e_ts if e_ts is not None else ev_ts)
            if self._voice_interrupt_action_ts:
                if abs(ts - float(self._voice_interrupt_action_ts[-1])) < 2e-3:
                    return
            if not self.initialized:
                self.init_full_layout()
            self._voice_interrupt_action_ts.append(ts)
            if len(self._voice_interrupt_action_ts) > 6000:  # 非表示時の無制限増加を防ぐ
                del self._voice_interrupt_action_ts[:-3000]
            if self._panel_visible.get("voice_latency", True):
                self._redraw_voice_latency()
            return
        if kind_key not in {"asr", "llm", "tts"}:
            return
        now_ts = time.time()
        ev_ts = float(event_ts) if event_ts is not None else now_ts
        s_ts = to_float_or_none(start_ts)
        e_ts = to_float_or_none(end_ts)
        # value=None: start_ts あり→進行中帯を開く / なし→進行中帯を閉じる（cancel 等）
        if latency_sec is None and e_ts is None:
            if kind_key == "llm":
                if not self.initialized:
                    self.init_full_layout()
                if s_ts is not None:
                    self._voice_llm_open_start = float(s_ts)
                else:
                    self._voice_llm_open_start = None
                if self._panel_visible.get("voice_latency", True):
                    self._redraw_voice_latency()
            return
        if s_ts is None and e_ts is not None and latency_sec is not None:
            s_ts = float(e_ts) - max(0.0, float(latency_sec))
        if e_ts is None and s_ts is not None and latency_sec is not None:
            e_ts = float(s_ts) + max(0.0, float(latency_sec))
        if s_ts is None and e_ts is None and latency_sec is not None:
            e_ts = ev_ts
            s_ts = float(e_ts) - max(0.0, float(latency_sec))
        if s_ts is None or e_ts is None:
            return
        fs = float(s_ts)
        fe = float(e_ts)
        if kind_key == "llm":
            # 確定帯が入ったら進行中表示は不要。
            self._voice_llm_open_start = None
        if self._voice_latency_span_is_duplicate(kind_key, fs, fe):
            return
        if self._voice_latency_merge_same_start(kind_key, fs, fe):
            if not self.initialized:
                self.init_full_layout()
            if self._panel_visible.get("voice_latency", True):
                self._redraw_voice_latency()
            return
        span = {"kind": kind_key, "start_ts": fs, "end_ts": fe}
        if not self.initialized:
            self.init_full_layout()
        self._voice_latency_spans.append(span)
        if len(self._voice_latency_spans) > 6000:  # 非表示時の無制限増加を防ぐ
            del self._voice_latency_spans[:-3000]
        if self._panel_visible.get("voice_latency", True):
            self._redraw_voice_latency()

    def update_voice_dialog_turn(self, *, event_ts: float | None = None) -> None:
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        if not self.initialized:
            self.init_full_layout()
        ts = float(event_ts) if event_ts is not None else time.time()
        if self._voice_turn_events:
            if abs(ts - float(self._voice_turn_events[-1])) < 2e-3:
                return
        self._voice_turn_events.append(ts)
        if len(self._voice_turn_events) > 6000:  # 非表示時の無制限増加を防ぐ
            del self._voice_turn_events[:-3000]
        if self._panel_visible.get("voice_latency", True):
            self._redraw_voice_latency()

    def update_voice_dialog_prefetch(self, *, event_ts: float | None = None) -> None:
        """p_future 先読み開始時刻を音声対話レイテンシ帯に縦線で残す。"""
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        if not self.initialized:
            self.init_full_layout()
        ts = float(event_ts) if event_ts is not None else time.time()
        if self._voice_prefetch_events:
            if abs(ts - float(self._voice_prefetch_events[-1])) < 2e-3:
                return
        self._voice_prefetch_events.append(ts)
        if len(self._voice_prefetch_events) > 6000:
            del self._voice_prefetch_events[:-3000]
        if self._panel_visible.get("voice_latency", True):
            self._redraw_voice_latency()

    def ingest_result(self, result: dict[str, Any], *, event_ts: float | None = None) -> None:
        """MaAI 結果を内部バッファへ反映するのみ（setData / 波形再描画はしない）。"""
        if not self._accepting_updates:
            return
        if not self._plot_enabled:
            return
        if not self.initialized:
            self.init_full_layout()
        if not self.initialized:
            return
        source_ts = to_float_or_none(result.get("source_ts_end"))
        if source_ts is not None:
            self._note_audio_now(source_ts)
        # MaAI 曲線は波形と同じ音声サンプルクロック（source_ts_end）に載せる。右端の「now」
        # （_audio_now_ts）も同じ時計なので、Windows の時刻同期で壁時計が跳んでも
        # 波形と曲線がずれない（2026-11-10 に +1.79 秒跳び、曲線が 1.8 秒先へ外れた）。
        # 右端には処理遅延ぶん（中央値 ~0.01 秒）の空白が出るだけ。source_ts_end が無い
        # 結果だけ、推論完了時の壁時計（result["t"]）に戻す。
        infer_t = to_float_or_none(result.get("t"))
        if event_ts is not None:
            ev = float(event_ts)
        elif source_ts is not None:
            ev = float(source_ts)
        else:
            ev = infer_t if infer_t is not None else time.time()
        meta_keys = {
            "t",
            "source_ts_start",
            "source_ts_end",
            "source_sample_start",
            "source_sample_end",
            "compare_slot",
            "maai_mode",
            "p_bc_threshold",
            "nod_swing_up_threshold",
            "bin_times",
            "silero_vad_plot_only",
        }
        for rk in result:
            if rk in meta_keys or str(rk).startswith("source_"):
                continue
            if rk not in self.keys and rk not in meta_keys:
                self._unknown_result_keys_warned.add(str(rk))

        for key in self.keys:
            if key not in result:
                continue
            val = result[key]
            if key in ("x1", "x2"):
                if self.waveform_from_raw_audio:
                    continue
                buf = self.data_buffer.get(key, [])
                x = np.array(val, dtype=float).reshape(-1)
                tail = x[-max(1, self.sample_rate // max(1, self.frame_rate)) :]
                buf = (list(buf) + list(tail))[-self.MAX_CONTEXT_WAV_LEN :]
                self.data_buffer[key] = buf
            elif key in ("p_now", "p_future") or (key.startswith("p_") and key != "p_bins"):
                v = self._first_float(val)
                if key in ("p_now", "p_future"):
                    v = self._invert_probability_value(v)
                self._append_event_value(key, v, ev)
                if key == "p_bc":
                    line = self._threshold_lines.get("p_bc_threshold")
                    if line is not None:
                        th = to_float_or_none(result.get("p_bc_threshold"))
                        if th is not None:
                            line.setValue(max(0.0, min(1.0, th)))
            elif key == "vad":
                arr = np.array(val, dtype=float).reshape(-1)
                vad1 = float(arr[0]) if arr.size > 0 else 0.0
                vad2 = float(arr[1]) if arr.size > 1 else 0.0
                self._append_event_value("vad_1", vad1, ev)
                self._append_event_value("vad_2", vad2, ev)
            elif key == "silero_vad_score":
                score = self._first_float(val, default=0.0)
                self._append_event_value(key, score, ev)
            elif key in self._NOD_PARA_PLOT_KEYS:
                v = self._scalar_nod_plot_value(key, val)
                self._append_event_value(key, v, ev)
                if key == "nod_swing_up":
                    line = self._threshold_lines.get("nod_swing_up_threshold")
                    if line is not None:
                        th = to_float_or_none(result.get("nod_swing_up_threshold"))
                        if th is not None:
                            line.setValue(max(0.0, min(1.0, th)))
            elif key == "p_bins":
                arr = np.array(val, dtype=float)
                self.data_buffer[key] = arr
                if "bin_times" in result:
                    self._p_bins_last_result_bin_times = result.get("bin_times")
            else:
                arr = np.array(val, dtype=float).reshape(-1) if isinstance(val, (np.ndarray, list, tuple)) else None
                if arr is None or arr.size == 0:
                    continue
                self.data_buffer[key] = list(arr)

    def flush_maai_draw(self, now_ts: float | None = None) -> None:
        """ingest で溜めた状態をプロットへ反映する。Controller からは最大 5Hz で呼ぶ想定。"""
        if not self._accepting_updates:
            return
        if not self._plot_enabled or not self.initialized:
            return
        # 全パネルで波形と同じ「now」を使う。割り込みなどの実行時刻は
        # 壁時計のまま保持し、表示の右端だけを音声側と揃える。
        audio_now = self._audio_now_ts()

        for key in self.keys:
            if not self._panel_visible.get(key, True):
                continue
            if key in ("x1", "x2"):
                if self.waveform_from_raw_audio:
                    continue
                buf = self.data_buffer.get(key, [])
                if key not in self.curves:
                    continue
                curve = self.curves[key]
                if isinstance(curve, dict):
                    wave = curve.get("wave")
                    if wave is not None:
                        wave.setData(self._x_wave, np.array(buf, dtype=float))
                else:
                    curve.setData(self._x_wave, np.array(buf, dtype=float))
            elif key in ("p_now", "p_future") or (key.startswith("p_") and key != "p_bins"):
                if key not in self.curves:
                    continue
                x, y = self._event_series_xy(key, audio_now)
                if key in ("p_now", "p_future") and isinstance(self.curves[key], dict):
                    x_fill, hi, lo = self._split_by_threshold_cached(key, x, y, 0.5)
                    self.curves[key]["high"].setData(x_fill, hi)
                    self.curves[key]["low"].setData(x_fill, lo)
                else:
                    self.curves[key].setData(x, y)
            elif key == "vad":
                if key not in self.curves:
                    continue
                c1, c2 = self.curves[key]
                x1, y1 = self._event_series_xy("vad_1", audio_now)
                x2, y2 = self._event_series_xy("vad_2", audio_now)
                c1.setData(x1, self._clip_positive(y1))
                c2.setData(x2, -self._clip_positive(y2))
            elif key == "silero_vad_score":
                if key not in self.curves:
                    continue
                x, y = self._event_series_xy(key, audio_now)
                self.curves[key].setData(x, self._clip_positive(y))
            elif key in self._NOD_PARA_PLOT_KEYS:
                if key not in self.curves:
                    continue
                x, y = self._event_series_xy(key, audio_now)
                self.curves[key].setData(x, y)
            elif key == "p_bins":
                arr = self.data_buffer.get(key)
                if isinstance(arr, np.ndarray):
                    self._maybe_draw_p_bins(
                        arr,
                        self._p_bins_last_result_bin_times
                        if self._p_bins_last_result_bin_times is not None
                        else self._p_bins_bin_times,
                    )
            else:
                buf = self.data_buffer.get(key)
                if not isinstance(buf, list) or key not in self.curves:
                    continue
                arr = np.array(buf, dtype=float).reshape(-1)
                if arr.size == 0:
                    continue
                self.curves[key].setData(np.arange(arr.size), arr)

        if self._panel_visible.get("x1", True) or self._panel_visible.get("x2", True):
            self._redraw_waveforms(audio_now)
        if any(self._panel_visible.get(k, True) for k in self._NOD_TRIGGER_PLOT_KEYS):
            self._redraw_nod_trigger_regions(audio_now)
        if self._panel_visible.get("p_bc", True):
            self._redraw_bc_trigger_regions(audio_now)
        if self._panel_visible.get("asr", True) or self._panel_visible.get("voice_latency", True):
            self._redraw_voice_latency(audio_now)
            self._redraw_asr_segments_axis(draw=True)

    def tick_wall_clock_plot_scroll(self) -> None:
        """MaAI 推論が無い間も、壁時計で全カーブを左に流す。"""
        if not self._accepting_updates:
            return
        if not self._plot_enabled or not self.initialized:
            return
        # flush_maai_draw が _redraw_voice_latency / _redraw_asr_segments_axis を含むため
        # ここで個別に呼び出さない（二重実行で plot.clear() が倍増するのを防ぐ）
        self.flush_maai_draw(now_ts=time.time())

    def update_result(self, result: dict[str, Any]) -> None:
        """ingest + 即時 flush（単発呼び出し用）。高頻度経路は ingest_result / flush_maai_draw に分ける。"""
        if not self._plot_enabled:
            return
        self.ingest_result(result)
        ev = to_float_or_none(result.get("source_ts_end"))
        if ev is None:
            ev = time.time()
        self.flush_maai_draw(now_ts=ev)
