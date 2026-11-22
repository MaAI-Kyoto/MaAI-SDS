"""頷きモーション系列の1回描画用チャート（Kyodai NodMotionChartWindow の Qt 移植）。"""

from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


class NodMotionChartWidget(QWidget):
    """生成直後の pitch 時系列を1回だけ更新して表示する。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._motion: list[float] | None = None
        self._time_axis: list[float] | None = None
        self._interpolation: str = ""
        self._velocity: float = 0.0
        self._nod_count: int = 0
        self._p_nod: float = 0.0
        self._range_rad: float = 0.0
        self._swing_up_prob: float = 0.0
        self._use_pre_rise: bool = False
        self._buffer_frames: int = 0
        self._raw_range: float = 0.0
        self._raw_speed: float = 0.0

    def set_chart_data(
        self,
        motion: Sequence[float],
        time_axis: Sequence[float],
        interpolation: str,
        velocity: float,
        nod_count: int,
        p_nod: float,
        range_rad: float,
        swing_up_prob: float,
        use_pre_rise: bool,
        buffer_frames: int,
        raw_range: float,
        raw_speed: float,
    ) -> None:
        self._motion = [float(x) for x in motion]
        self._time_axis = [float(x) for x in time_axis]
        self._interpolation = str(interpolation)
        self._velocity = float(velocity)
        self._nod_count = int(nod_count)
        self._p_nod = float(p_nod)
        self._range_rad = float(range_rad)
        self._swing_up_prob = float(swing_up_prob)
        self._use_pre_rise = bool(use_pre_rise)
        self._buffer_frames = int(buffer_frames)
        self._raw_range = float(raw_range)
        self._raw_speed = float(raw_speed)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        w = float(self.width())
        h = float(self.height())
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin_left, margin_right = 60.0, 20.0
        margin_top, margin_bottom = 40.0, 40.0
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        p.fillRect(self.rect(), QColor(255, 255, 255))

        motion = self._motion
        time_axis = self._time_axis
        if not motion or not time_axis or len(motion) != len(time_axis):
            p.setPen(QColor(128, 128, 128))
            p.setFont(QFont("sans-serif", 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "モーションデータがありません")
            p.end()
            return

        t_min = 0.0
        t_max = 3.0
        y_min = -0.1
        y_max = 0.1
        t_range = t_max - t_min
        y_range = y_max - y_min

        pen_light = QPen(QColor(211, 211, 211))
        pen_light.setWidthF(0.5)
        p.setPen(pen_light)
        p.setFont(QFont("sans-serif", 10))

        y_ticks = 5
        for i in range(y_ticks + 1):
            y_val = y_min + y_range * i / y_ticks
            py = margin_top + plot_h - (y_val - y_min) / y_range * plot_h
            p.drawLine(int(margin_left), int(py), int(margin_left + plot_w), int(py))
            p.setPen(QColor(105, 105, 105))
            p.drawText(2, int(py + 4), f"{y_val:.3f}")
            p.setPen(pen_light)

        t_ticks = 5
        for i in range(t_ticks + 1):
            t_val = t_min + t_range * i / t_ticks
            px = margin_left + (t_val - t_min) / t_range * plot_w
            p.drawLine(int(px), int(margin_top), int(px), int(margin_top + plot_h))
            p.setPen(QColor(105, 105, 105))
            p.drawText(int(px - 12), int(margin_top + plot_h + 14), f"{t_val:.2f}")
            p.setPen(pen_light)

        if y_min < 0 < y_max:
            zero_y = margin_top + plot_h - (0.0 - y_min) / y_range * plot_h
            dash_pen = QPen(QColor(128, 128, 128))
            dash_pen.setWidthF(1.0)
            dash_pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(dash_pen)
            p.drawLine(int(margin_left), int(zero_y), int(margin_left + plot_w), int(zero_y))

        border_pen = QPen(QColor(105, 105, 105))
        border_pen.setWidthF(1.0)
        p.setPen(border_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(int(margin_left), int(margin_top), int(plot_w), int(plot_h))

        curve_pen = QPen(QColor(0, 128, 0))
        curve_pen.setWidthF(2.0)
        curve_pen.setStyle(Qt.PenStyle.SolidLine)
        p.setPen(curve_pen)
        poly = QPolygonF()
        for i in range(len(motion)):
            px = margin_left + (time_axis[i] - t_min) / t_range * plot_w
            py = margin_top + plot_h - (motion[i] - y_min) / y_range * plot_h
            poly.append(QPointF(px, py))
        if poly.size() >= 2:
            p.drawPolyline(poly)
        elif poly.size() == 1:
            p.drawPoint(int(poly.at(0).x()), int(poly.at(0).y()))

        p.setPen(QColor(0, 0, 0))
        p.setFont(QFont("sans-serif", 13))
        title = (
            f"頷きモーション（回数={self._nod_count}、{self._interpolation}、"
            f"頷き確率={self._p_nod:.3f}）"
        )
        p.drawText(int(margin_left), int(margin_top - 12), title)

        info_x = margin_left + plot_w - 200
        info_y = margin_top + 16
        line_h = 15.0
        p.setFont(QFont("sans-serif", 11))
        p.setPen(QColor(0, 0, 180))
        p.drawText(int(info_x), int(info_y), "VAP予測パラメーター：")
        p.setFont(QFont("sans-serif", 10))

        rr = abs(self._raw_range)
        range_clamped = rr < 0.025 or rr > 0.15
        p.setPen(QColor(200, 0, 0) if range_clamped else QColor(60, 60, 60))
        deg = math.degrees(self._range_rad)
        p.drawText(
            int(info_x),
            int(info_y + line_h),
            f"振れ幅：{self._raw_range:.4f}→{self._range_rad:.4f} rad（{deg:.1f}°）"
            f"{' クリップ済み' if range_clamped else ''}",
        )

        rs = abs(self._raw_speed)
        speed_clamped = rs < 0.05 or rs > 0.25
        p.setPen(QColor(200, 0, 0) if speed_clamped else QColor(60, 60, 60))
        p.drawText(
            int(info_x),
            int(info_y + line_h * 2),
            f"速度：{self._raw_speed:.4f}→{self._velocity:.3f} rad/s"
            f"{' クリップ済み' if speed_clamped else ''}",
        )

        p.setPen(QColor(60, 60, 60))
        p.drawText(int(info_x), int(info_y + line_h * 3), f"回数：{self._nod_count}")
        swing_txt = "（有効）" if self._use_pre_rise else "（無効）"
        p.drawText(
            int(info_x),
            int(info_y + line_h * 4),
            f"振り上げ：{self._swing_up_prob:.3f}{swing_txt}",
        )
        p.drawText(int(info_x), int(info_y + line_h * 5), f"フレーム数：{self._buffer_frames}")

        p.setFont(QFont("sans-serif", 11))
        p.setPen(QColor(105, 105, 105))
        p.drawText(int(margin_left + plot_w / 2 - 20), int(h - 2), "時間（秒）")

        p.save()
        p.translate(12, int(margin_top + plot_h / 2 + 30))
        p.rotate(-90)
        p.drawText(0, 0, "角度（rad）")
        p.restore()

        p.end()
