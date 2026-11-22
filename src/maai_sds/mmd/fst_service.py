"""MMDAgent FST（有限状態トランスデューサ）の起動メッセージと KEY 束縛。

本家は `MMDAgent-EX.exe foo.mdf` と同名の `foo.fst` を走らせる。書式は空白区切りの

    src dst input [output...]

で、`#` 行はコメント。初期状態 `0` から `<eps>`（および `MODEL_ADD` 後に発火する
`MODEL_EVENT_ADD|alias`）を本家どおり辿り、`STAGE` / `MODEL_ADD` / `MOTION_ADD` /
`CAMERA` を取り出す。`KEY|n` はキー束縛として残す。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

EPS_SYMBOLS = frozenset({"<eps>", "ε", "eps", ""})


@dataclass(frozen=True)
class FstArc:
    src: str
    dst: str
    input: str
    output: str


def read_fst_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace")


def parse_fst_text(text: str) -> list[FstArc]:
    arcs: list[FstArc] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        src, dst, inp = parts[0], parts[1], parts[2]
        output = " ".join(parts[3:]) if len(parts) > 3 else ""
        arcs.append(FstArc(src=src, dst=dst, input=inp, output=output))
    return arcs


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for tok in raw.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            out.append(0.0)
    return out


def _parse_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw.strip())
    except (TypeError, ValueError, AttributeError):
        return default


def parse_message(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text or text in EPS_SYMBOLS:
        return None
    bits = text.split("|")
    head = bits[0].strip().upper()
    if head == "STAGE":
        arg = bits[1] if len(bits) > 1 else ""
        parts = [p.strip() for p in arg.split(",")]
        floor = parts[0] if parts else ""
        background = parts[1] if len(parts) > 1 else ""
        return {"type": "STAGE", "floor": floor, "background": background, "raw": text}
    if head == "MODEL_ADD":
        alias = bits[1].strip() if len(bits) > 1 else "0"
        path = bits[2].strip() if len(bits) > 2 else ""
        return {"type": "MODEL_ADD", "alias": alias, "path": path, "raw": text}
    if head == "MOTION_ADD":
        return {
            "type": "MOTION_ADD",
            "alias": bits[1].strip() if len(bits) > 1 else "0",
            "name": bits[2].strip() if len(bits) > 2 else "",
            "file": bits[3].strip() if len(bits) > 3 else "",
            "part": (bits[4].strip().upper() if len(bits) > 4 else "FULL") or "FULL",
            "loop": (bits[5].strip().upper() if len(bits) > 5 else "ONCE") or "ONCE",
            "smooth": (bits[6].strip().upper() if len(bits) > 6 else "OFF") or "OFF",
            "reposition": (bits[7].strip().upper() if len(bits) > 7 else "OFF") or "OFF",
            "raw": text,
        }
    if head == "CAMERA":
        lookat = _parse_floats(bits[1]) if len(bits) > 1 else []
        rotation = _parse_floats(bits[2]) if len(bits) > 2 else []
        while len(lookat) < 3:
            lookat.append(0.0)
        while len(rotation) < 3:
            rotation.append(0.0)
        return {
            "type": "CAMERA",
            "lookat": lookat[:3],
            "rotation": rotation[:3],
            "distance": _parse_float(bits[3], 100.0) if len(bits) > 3 else 100.0,
            "fovy": _parse_float(bits[4], 16.0) if len(bits) > 4 else 16.0,
            "raw": text,
        }
    return {"type": head, "args": bits[1:], "raw": text}


def _is_eps(symbol: str) -> bool:
    return (symbol or "").strip() in EPS_SYMBOLS


def _model_event_alias(symbol: str) -> str | None:
    text = (symbol or "").strip()
    prefix = "MODEL_EVENT_ADD|"
    if text.upper().startswith(prefix):
        return text.split("|", 1)[1].strip()
    return None


def walk_startup(arcs: list[FstArc], initial: str = "0") -> list[dict[str, Any]]:
    """初期状態から ε 遷移（と MODEL_EVENT_ADD）を辿って起動メッセージ列を得る。"""
    by_src: dict[str, list[FstArc]] = {}
    for arc in arcs:
        by_src.setdefault(arc.src, []).append(arc)

    startup: list[dict[str, Any]] = []
    added_models: set[str] = set()
    state = initial
    seen: set[tuple[str, str, str, str]] = set()
    for _ in range(500):
        fired = False
        for arc in by_src.get(state, []):
            key = (arc.src, arc.dst, arc.input, arc.output)
            if key in seen:
                continue
            event_alias = _model_event_alias(arc.input)
            can_fire = _is_eps(arc.input) if event_alias is None else event_alias in added_models
            if not can_fire:
                continue
            seen.add(key)
            message = parse_message(arc.output)
            if message:
                startup.append(message)
                if message.get("type") == "MODEL_ADD":
                    added_models.add(str(message.get("alias", "0")))
            state = arc.dst
            fired = True
            break
        if not fired:
            break
    return startup


def collect_keys(arcs: list[FstArc]) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    for arc in arcs:
        inp = (arc.input or "").strip()
        if not inp.upper().startswith("KEY|"):
            continue
        code = inp.split("|", 1)[1].strip()
        keys.append({"key": code, "message": parse_message(arc.output), "raw": arc.output})
    return keys


def compile_fst(text: str) -> dict[str, Any]:
    arcs = parse_fst_text(text)
    return {"startup": walk_startup(arcs), "keys": collect_keys(arcs)}


def compile_fst_for_mdf(mdf_path: Path, fallback_text: str = "") -> dict[str, Any]:
    fst_path = mdf_path.with_suffix(".fst")
    if fst_path.is_file():
        compiled = compile_fst(read_fst_text(fst_path))
        compiled["path"] = str(fst_path)
        compiled["missing"] = False
        return compiled
    if fallback_text.strip():
        compiled = compile_fst(fallback_text)
        compiled["path"] = str(fst_path)
        compiled["missing"] = True
        compiled["fallback"] = True
        return compiled
    return {"startup": [], "keys": [], "path": str(fst_path), "missing": True}
