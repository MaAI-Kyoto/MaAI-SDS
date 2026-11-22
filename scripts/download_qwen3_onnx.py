"""Qwen3-ASR の sherpa-onnx 用 ONNX を ModelScope から取得する。

0.6B int8 の公式 tar は GitHub（setup_assets の Qwen3AsrOnnxModel）を使う。
こちらは GitHub に無い版:

  0.6b-fp32  → sherpa-onnx-qwen3-asr-0.6B-fp32/   （約 3.7GB）
  1.7b-int8  → sherpa-onnx-qwen3-asr-1.7B-int8/   （約 2.4GB）

元データ: https://modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODEL = "zengshuishui/Qwen3-ASR-onnx"
REVISION = "master"

_TOKENIZER = (
    ("tokenizer/chat_template.json", "tokenizer/chat_template.json", 1_161),
    ("tokenizer/config.json", "tokenizer/config.json", 6_193),
    ("tokenizer/merges.txt", "tokenizer/merges.txt", 1_671_853),
    ("tokenizer/preprocessor_config.json", "tokenizer/preprocessor_config.json", 330),
    ("tokenizer/tokenizer_config.json", "tokenizer/tokenizer_config.json", 12_487),
    ("tokenizer/vocab.json", "tokenizer/vocab.json", 2_776_833),
)

VARIANTS: dict[str, dict] = {
    "0.6b-fp32": {
        "dest": "sherpa-onnx-qwen3-asr-0.6B-fp32",
        "files": (
            *_TOKENIZER,
            ("model_0.6B/conv_frontend.onnx", "conv_frontend.onnx", 44_148_281),
            ("model_0.6B/encoder.onnx", "encoder.onnx", 5_594_598),
            ("model_0.6B/encoder.onnx.data", "encoder.onnx.data", 701_367_808),
            ("model_0.6B/decoder.onnx", "decoder.onnx", 843_211),
            ("model_0.6B/decoder.onnx.data", "decoder.onnx.data", 3_006_500_864),
        ),
    },
    "1.7b-int8": {
        "dest": "sherpa-onnx-qwen3-asr-1.7B-int8",
        "files": (
            *_TOKENIZER,
            ("model_1.7B/conv_frontend.onnx", "conv_frontend.onnx", 48_080_441),
            ("model_1.7B/encoder.int8.onnx", "encoder.int8.onnx", 314_222_162),
            ("model_1.7B/decoder.int8.onnx", "decoder.int8.onnx", 2_037_458_645),
        ),
    },
}


def url_for(path: str) -> str:
    return (
        f"https://www.modelscope.cn/api/v1/models/{MODEL}/repo"
        f"?Revision={REVISION}&FilePath={path}"
    )


def download(remote: str, local: Path, expected: int) -> None:
    if local.is_file() and local.stat().st_size == expected:
        print(f"skip {local.as_posix()} ({expected} bytes)", flush=True)
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    existing = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"User-Agent": "MaAI-SDS/qwen3-onnx"}
    if existing and existing < expected:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"resume {local.name} from {existing}", flush=True)
    else:
        existing = 0
        mode = "wb"
        print(f"get {local.name} ({expected} bytes)", flush=True)
    req = urllib.request.Request(url_for(remote), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open(mode) as out:
        total = expected
        got = existing
        last = -1
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            pct = int(got * 100 / total) if total else 0
            if pct != last and pct % 5 == 0:
                print(f"  {local.name} {got}/{total} ({pct}%)", flush=True)
                last = pct
    size = tmp.stat().st_size
    if size != expected:
        raise RuntimeError(f"{local.name}: got {size} bytes, expected {expected}")
    tmp.replace(local)
    print(f"ok {local.name}", flush=True)


def fetch_variant(root: Path, variant: str) -> Path:
    spec = VARIANTS[variant]
    dest = root / spec["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    for remote, name, expected in spec["files"]:
        download(remote, dest / name, expected)
    print("done", dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="0.6b-fp32",
        help="取得する版（既定: 0.6b-fp32）",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="リポジトリ（またはデータ）ルート",
    )
    args = parser.parse_args()
    fetch_variant(args.root.resolve(), args.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
