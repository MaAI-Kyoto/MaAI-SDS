from __future__ import annotations

from typing import Any

# vap_nod_para の nod 閾値一致判定（float 比較）
NOD_PARA_THRESH_MATCH_EPS = 1e-6


def apply_nod_count_thresholds_python(prob: list[float], thresholds: dict[str, float]) -> int:
    """VapGPT_nod_para._apply_count_thresholds と同じロジック（CPU・torch なし）。"""
    t = [float(thresholds.get(f"t{i}", 1.0)) for i in range(3)]
    p = [float(x) for x in prob[:3]]
    while len(p) < 3:
        p.append(0.0)
    eps = 1e-8
    ratios = [p[i] / max(t[i], eps) for i in range(3)]
    return int(max(range(3), key=lambda i: ratios[i]))


def _nod_para_count_thresholds_equal(a: dict[str, float], b: dict[str, float]) -> bool:
    for k in ("t0", "t1", "t2"):
        if abs(float(a.get(k, 1.0)) - float(b.get(k, 1.0))) > NOD_PARA_THRESH_MATCH_EPS:
            return False
    return True


def apply_maai_nod_para_prediction_aliases(
    result: dict[str, Any],
    *,
    count_thresholds_ui: dict[str, float] | None = None,
    swing_threshold_ui: float | None = None,
    count_thresholds_default: dict[str, float] | None = None,
    swing_threshold_default: float | None = None,
) -> None:
    """nod_para の離散予測を nod_count / nod_swing_up に集約し、*_pred キーは削除する。

    UI 閾値がワーカーから受け取った nod インスタンスの既定と一致する場合は、
    パッケージ forward が返した *_pred をそのまま使う。
    異なる場合は nod_count（確率リスト）と nod_swing_up（確率）に UI 閾値を再適用する。

    生の確率は `nod_count_probs` / `nod_swing_up_prob` に保持し、
    `nod_count` は離散クラス（0/1/2）へ正規化する。
    """
    if not isinstance(result, dict):
        return
    if "nod_count_pred" not in result and "nod_swing_up_pred" not in result:
        return

    nc_existing = result.get("nod_count")
    if isinstance(nc_existing, (list, tuple)) and "nod_count_probs" not in result:
        try:
            result["nod_count_probs"] = [float(x) for x in nc_existing]
        except (TypeError, ValueError):
            pass

    use_packaged_pred = True
    if (
        count_thresholds_ui is not None
        and swing_threshold_ui is not None
        and count_thresholds_default is not None
        and swing_threshold_default is not None
    ):
        use_packaged_pred = _nod_para_count_thresholds_equal(
            count_thresholds_ui, count_thresholds_default
        ) and abs(float(swing_threshold_ui) - float(swing_threshold_default)) <= NOD_PARA_THRESH_MATCH_EPS

    if not use_packaged_pred:
        nc_raw = result.get("nod_count")
        su_raw = result.get("nod_swing_up")
        if isinstance(nc_raw, (list, tuple)) and count_thresholds_ui is not None and swing_threshold_ui is not None:
            try:
                swing_prob = float(su_raw)
            except (TypeError, ValueError):
                use_packaged_pred = True
            else:
                probs = [float(x) for x in nc_raw]
                pred_c = apply_nod_count_thresholds_python(probs, count_thresholds_ui)
                pred_s = 1 if swing_prob >= float(swing_threshold_ui) else 0
                result["nod_swing_up_prob"] = float(swing_prob)
                result["nod_count"] = int(pred_c)
                result["nod_swing_up"] = float(pred_s)
                result.pop("nod_count_pred", None)
                result.pop("nod_swing_up_pred", None)
                return
        use_packaged_pred = True

    if "nod_count_pred" in result and result.get("nod_count_pred") is not None:
        try:
            result["nod_count"] = int(result["nod_count_pred"])
        except (TypeError, ValueError):
            result["nod_count"] = result["nod_count_pred"]
        del result["nod_count_pred"]
    if "nod_swing_up_pred" in result and result.get("nod_swing_up_pred") is not None:
        try:
            result["nod_swing_up_prob"] = float(result["nod_swing_up"])
        except (TypeError, ValueError):
            pass
        v = result["nod_swing_up_pred"]
        try:
            result["nod_swing_up"] = float(int(v))
        except (TypeError, ValueError):
            try:
                result["nod_swing_up"] = float(v)
            except (TypeError, ValueError):
                pass
        del result["nod_swing_up_pred"]


def to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def json_config_bool(cfg: dict, key: str, default: bool) -> bool:
    """json.loads 済みの dict から真偽を取り出す。bool / int / float / str / null に対応する。"""
    if key not in cfg:
        return default
    v = cfg[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, float):
        return v != 0.0
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("", "0", "false", "no", "off", "none", "null"):
        return False
    return s in ("1", "true", "yes", "on")


def to_float_or_none(value: object) -> float | None:
    """float に変換できれば float、できなければ None。

    `asr/service` と `maai/plot_window` に同じ実装が 2 つあったのを集約したもの。
    `float(x)` が投げる例外を全部飲むので、None・空文字・非数値のどれでも None になる。
    """
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
