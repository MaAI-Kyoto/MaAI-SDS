"""Export Nemotron 3.5 NeMo preprocessor to ONNX for sherpa window-mel path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

INPUT_NAMES = ["audio_signal", "length"]
OUTPUT_NAMES = ["mel", "mel_len"]


def resolve_python(explicit: str) -> str:
    if explicit.strip():
        path = Path(explicit.strip())
        if not path.is_file():
            raise FileNotFoundError(f"Python not found: {explicit}")
        return str(path)

    candidates = [
        ROOT / ".venv_nemotron" / "Scripts" / "python.exe",
        Path(r"C:/NeMo/.venv/Scripts/python.exe"),
        Path(sys.executable),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "NeMo Python not found. Pass --python or create .venv_nemotron "
        "(scripts/setup_nemotron_venv.ps1)."
    )


def export_preprocessor(out_path: Path, model_name: str) -> None:
    import torch

    import nemo.collections.asr as nemo_asr

    class PreprocessorWrapper(torch.nn.Module):
        def __init__(self, preprocessor: torch.nn.Module) -> None:
            super().__init__()
            self.preprocessor = preprocessor

        def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
            mel, mel_len = self.preprocessor(
                input_signal=audio_signal, length=length
            )
            return mel, mel_len

    device = torch.device("cpu")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model = asr_model.to(device)
    asr_model.preprocessor = asr_model.preprocessor.to(device)
    asr_model.eval()

    wrapper = PreprocessorWrapper(asr_model.preprocessor).eval().to(device)

    # 15s window @ 16 kHz is enough for dynamic-axis smoke test.
    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    dummy_sec = 15.0
    dummy_samples = int(sample_rate * dummy_sec)
    audio = torch.randn(1, dummy_samples, dtype=torch.float32, device=device)
    length = torch.tensor([dummy_samples], dtype=torch.int64, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".export.onnx")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (audio, length),
            str(tmp_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes={
                "audio_signal": {0: "batch", 1: "time"},
                "length": {0: "batch"},
                "mel": {0: "batch", 2: "mel_time"},
                "mel_len": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    if out_path.is_file():
        out_path.unlink()
    tmp_path.rename(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace / NeMo model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output preprocessor.onnx path",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="MaAI-SDS packaged model dir (writes preprocessor.onnx inside)",
    )
    parser.add_argument(
        "--python",
        default="",
        help="NeMo venv python",
    )
    args = parser.parse_args()

    if args.model_dir is not None:
        out_path = args.model_dir / "preprocessor.onnx"
    elif args.out is not None:
        out_path = args.out
    else:
        print("Specify --out or --model-dir", file=sys.stderr)
        return 2

    python = resolve_python(args.python)
    print(f"Python: {python}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Output: {out_path}", flush=True)

    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        [
            python,
            str(Path(__file__).resolve()),
            "--_inner",
            "--model",
            args.model,
            "--out",
            str(out_path),
        ],
        cwd=str(ROOT),
        env=env,
    )
    return int(proc.returncode)


def _inner_main(model_name: str, out_path: Path) -> None:
    export_preprocessor(out_path, model_name)


if __name__ == "__main__":
    if "--_inner" in sys.argv:
        inner = argparse.ArgumentParser()
        inner.add_argument("--_inner", action="store_true")
        inner.add_argument("--model", required=True)
        inner.add_argument("--out", type=Path, required=True)
        a = inner.parse_args()
        _inner_main(a.model, a.out)
        raise SystemExit(0)
    raise SystemExit(main())
