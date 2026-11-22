"""Nod motion generator — ported from Nodding_motion_generator/generate_motion.py.

Returns pitch time-series (radians) and a matching time axis (seconds).
scipy-based interpolation modes are optional; if scipy is not installed they
fall back to cosine interpolation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def generate_natural_nod(
    range_rad: float,
    count: int,
    use_pre_rise: bool,
    velocity: float,
    fps: int = 30,
    decay_rate: float = 0.6,
    interpolation: str = "cosine",
    pre_rise_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a natural nodding motion sequence.

    Parameters
    ----------
    range_rad : float
        Nod depth in radians (absolute value).
    count : int
        Number of nods (>= 1).
    use_pre_rise : bool
        Whether to include a pre-rise before the first nod.
    velocity : float
        Target average angular velocity (rad/s).
    fps : int
        Output frame rate.
    decay_rate : float
        Amplitude decay per nod (0–1).
    interpolation : str
        ``"cosine"`` (always available), ``"cubic_spline"``,
        ``"cubic_hermite"``, or ``"hermite_extrema"`` (require scipy).
    pre_rise_ratio : float
        Pre-rise amplitude as a ratio of *range_rad*.

    Returns
    -------
    motion : np.ndarray
        Pitch values in radians, one per frame.
    time_axis : np.ndarray
        Time stamps in seconds, same length as *motion*.
    """
    count = max(1, int(count))
    fps = max(1, int(fps))

    keyframe_vals: list[float] = [0.0]
    current_amp = abs(float(range_rad))

    if use_pre_rise:
        keyframe_vals.extend([
            current_amp * float(pre_rise_ratio),
            -current_amp,
            0.0,
        ])
    else:
        keyframe_vals.extend([-current_amp, 0.0])

    for _ in range(count - 1):
        current_amp *= float(decay_rate)
        keyframe_vals.extend([-current_amp, 0.0])

    n_seg = len(keyframe_vals) - 1
    distances: list[float] = []
    vel_scales: list[float] = []
    for i in range(n_seg):
        d = abs(keyframe_vals[i + 1] - keyframe_vals[i])
        distances.append(d)
        if d < 1e-6:
            vel_scales.append(1.0)
        else:
            if use_pre_rise:
                nod_idx = 0 if i < 3 else 1 + (i - 3) // 2
            else:
                nod_idx = i // 2
            vel_scales.append(float(decay_rate) ** (nod_idx * 0.5))

    total_distance = sum(distances)
    raw_total = sum(d / vs for d, vs in zip(distances, vel_scales))
    total_time = total_distance / float(velocity) if float(velocity) > 1e-9 else 0.0

    keyframe_frames: list[int] = [0]
    for d, vs in zip(distances, vel_scales):
        if raw_total > 1e-9 and d >= 1e-6:
            duration = (d / vs) * total_time / raw_total
            n_frames = max(1, int(round(duration * fps)))
        else:
            n_frames = 0
        keyframe_frames.append(keyframe_frames[-1] + n_frames)

    keyframe_times = [f / fps for f in keyframe_frames]
    total_frames = keyframe_frames[-1]

    if total_frames <= 0:
        return np.array([0.0]), np.array([0.0])

    time_axis = np.arange(total_frames) / fps

    motion = _interpolate(
        interpolation, keyframe_times, keyframe_vals, keyframe_frames, time_axis,
    )
    return motion, time_axis


def _interpolate(
    method: str,
    keyframe_times: list[float],
    keyframe_vals: list[float],
    keyframe_frames: list[int],
    time_axis: np.ndarray,
) -> np.ndarray:
    if method == "cubic_spline":
        try:
            from scipy.interpolate import CubicSpline

            kt = np.asarray(keyframe_times, dtype=float)
            if kt.size >= 2 and np.all(np.diff(kt) > 1e-15):
                kv = np.asarray(keyframe_vals, dtype=float)
                cs = CubicSpline(kt, kv, bc_type=((2, 0), (1, 0)))
                return np.asarray(cs(time_axis), dtype=float)
        except ImportError:
            pass
        except Exception:
            # scipy 未導入・重複キーフレーム時刻・数値例外などは cosine にフォールバック
            pass
    elif method == "cubic_hermite":
        try:
            from scipy.interpolate import CubicHermiteSpline

            kt = np.asarray(keyframe_times, dtype=float)
            if kt.size >= 2 and np.all(np.diff(kt) > 1e-15):
                dydx = np.zeros(len(keyframe_vals))
                cs = CubicHermiteSpline(kt, keyframe_vals, dydx)
                return np.asarray(cs(time_axis), dtype=float)
        except ImportError:
            pass
        except Exception:
            pass
    elif method == "hermite_extrema":
        try:
            from scipy.interpolate import BPoly

            kt = np.asarray(keyframe_times, dtype=float)
            if kt.size < 2 or not np.all(np.diff(kt) > 1e-15):
                raise ValueError("keyframe times not strictly increasing")
            n = len(keyframe_vals)
            dydx = np.zeros(n)
            d2ydx2 = np.zeros(n)
            if n >= 2 and keyframe_times[1] - keyframe_times[0] > 1e-9:
                dydx[0] = (keyframe_vals[1] - keyframe_vals[0]) / (
                    keyframe_times[1] - keyframe_times[0]
                )
            if n >= 2 and keyframe_times[-1] - keyframe_times[-2] > 1e-9:
                dydx[-1] = 0.0
            for i in range(1, n - 1):
                dt_prev = keyframe_times[i] - keyframe_times[i - 1]
                dt_next = keyframe_times[i + 1] - keyframe_times[i]
                if dt_prev > 1e-9 and dt_next > 1e-9:
                    d2ydx2[i] = 2 * (
                        (keyframe_vals[i + 1] - keyframe_vals[i]) / dt_next
                        - (keyframe_vals[i] - keyframe_vals[i - 1]) / dt_prev
                    ) / (dt_prev + dt_next)
            yi = [[keyframe_vals[i], dydx[i], d2ydx2[i]] for i in range(n)]
            bpoly = BPoly.from_derivatives(keyframe_times, yi, orders=5)
            return np.asarray(bpoly(time_axis), dtype=float)
        except ImportError:
            pass
        except Exception:
            pass

    # Cosine fallback
    segments: list[float] = []
    for i in range(len(keyframe_vals) - 1):
        n_frames = keyframe_frames[i + 1] - keyframe_frames[i]
        if n_frames <= 0 or abs(keyframe_vals[i + 1] - keyframe_vals[i]) < 1e-6:
            continue
        t = np.linspace(0, math.pi, n_frames)
        seg = keyframe_vals[i] + (keyframe_vals[i + 1] - keyframe_vals[i]) * (1.0 - np.cos(t)) / 2.0
        segments.extend(seg.tolist())
    if not segments:
        return np.zeros(max(1, len(time_axis)), dtype=float)
    return np.array(segments, dtype=float)

