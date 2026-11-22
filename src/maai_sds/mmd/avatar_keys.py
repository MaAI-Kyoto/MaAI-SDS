"""頷き・目・まばたきを公式 MMDAgent-EX の仕組みで動かすためのコマンド。

MMDAgent-EX の `MODEL_BINDBONE` / `MODEL_BINDFACE` で、ボーンの回転や表情（モーフ）を KeyValue の数値に
結び付けておき、`KEYVALUE_SET` で数値を変えて動かす。どちらも MMDAgent-EX 本体のメッセージで、
Plugin_Remote は `__AV` で始まらない行をそのまま本体へ渡す。

- 頭：KeyValue `MaAI_HeadPitch` は頷きの角度（ラジアン、正が上向き）。-1〜1 を頭ボーンの X 軸回転
  +57.3°〜-57.3° に結び付ける（MMD の X 軸回転は正が下向き）。結び付けたボーンはモーションの動きを
  上書きするので、待機モーションの頭の揺れは止まる。
- 目：同じ数値を左目・右目にも結び付け、頭と逆向きに `eye_scale` 倍だけ回して視線を保つ。
- まばたき：KeyValue `MaAI_Blink`（0〜1）を表情モーフの重みに結び付ける。

結び付けは MMDAgent-EX がモデルを読み込んだあとでないと効かない（読み込み前は無視される）ので、
接続時とモデルの読み込み完了（`MODEL_EVENT_ADD|<別名>`）のたびに送り直す。
"""

from __future__ import annotations

HEAD_PITCH_KEY = "MaAI_HeadPitch"
BLINK_KEY = "MaAI_Blink"

# KeyValue の -1〜1（ラジアン）を回転角（度）へ。
_PITCH_LIMIT_RAD = 1.0
_RAD_TO_DEG = 57.29578

DEFAULT_HEAD_BONE = "頭"
EYE_BONES = ("左目", "右目")


def _deg(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _bind_bone(alias: str, bone: str, rx_at_min: float, rx_at_max: float) -> str:
    lo = _PITCH_LIMIT_RAD
    return (
        f"MODEL_BINDBONE|{HEAD_PITCH_KEY}|{-lo:g}|{lo:g}|{alias}|{bone}"
        f"|0,0,0|{_deg(rx_at_min)},0,0|0,0,0|{_deg(rx_at_max)},0,0\n"
    )


def head_bind_commands(alias: str, bone: str = DEFAULT_HEAD_BONE, eye_scale: float = 0.0) -> list[str]:
    """頭（と目）のボーンを `MaAI_HeadPitch` に結び付けるコマンド。"""
    bone = str(bone or "").strip() or DEFAULT_HEAD_BONE
    full = _PITCH_LIMIT_RAD * _RAD_TO_DEG
    # 頭：数値が正（上向き）のとき X 軸回転は負
    commands = [_bind_bone(alias, bone, full, -full)]
    if abs(float(eye_scale)) > 1e-6:
        eye = full * float(eye_scale)
        for eye_bone in EYE_BONES:
            commands.append(_bind_bone(alias, eye_bone, -eye, eye))
    return commands


def blink_bind_command(alias: str, morph: str) -> str:
    """表情モーフ（まばたき）を `MaAI_Blink`（0〜1）に結び付けるコマンド。"""
    return f"MODEL_BINDFACE|{BLINK_KEY}|0|1|{alias}|{morph}|0|1\n"


def head_pitch_command(pitch_rad: float) -> str:
    value = max(-_PITCH_LIMIT_RAD, min(_PITCH_LIMIT_RAD, float(pitch_rad)))
    return f"KEYVALUE_SET|{HEAD_PITCH_KEY}|{value:.4f}\n"


def blink_command(weight: float) -> str:
    if weight <= 0.0:
        value = "0"
    elif weight >= 1.0:
        value = "1"
    else:
        value = f"{weight:.4f}".rstrip("0").rstrip(".")
    return f"KEYVALUE_SET|{BLINK_KEY}|{value}\n"
