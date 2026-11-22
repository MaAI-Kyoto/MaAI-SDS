"""CUDA が実際に使えるかを判定し、要求されたデバイス指定を実情に落とす。

GPU が無い（あるいは CUDA ランタイムが揃っていない）マシンでは、CUDA を要求した
エンジンが **例外で落ちる**。実測:

    RuntimeError: OrtSessionOptionsAppendExecutionProvider_Cuda:
                  Failed to load shared library

`onnxruntime.get_available_providers()` はビルドに含まれる provider を並べるだけで、
実際にロードできるかは見ていない（GPU を隠した状態でも CUDAExecutionProvider が
列挙される）。そのため「本当に使えるか」は 1 ノードのモデルでセッションを作って
確かめるしかない。判定は 1 プロセス 1 回だけで足りるのでキャッシュする。

torch 側は `torch.cuda.is_available()` が信用できるのでそれを使う。
"""
from __future__ import annotations

import functools
from typing import Callable


def _log(on_output: Callable[[str], None] | None, message: str) -> None:
    if on_output is None:
        print(message, flush=True)
        return
    try:
        on_output(message)
    except Exception:  # noqa: BLE001
        pass


@functools.lru_cache(maxsize=1)
def onnx_cuda_available() -> bool:
    """ONNX Runtime の CUDA EP が実際に登録できるか。

    プロジェクトの DLL 事前解決（asr/ort_windows_preload）を通してから判定する。
    これを飛ばすと GPU があるマシンでも CPU しか登録されず、誤って CPU へ
    落としてしまう（実機で確認済み）。
    """
    try:
        from .asr import ort_windows_preload

        ort_windows_preload.ensure()
    except Exception:  # noqa: BLE001
        pass
    try:
        import onnxruntime as ort
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        options = ort.SessionOptions()
        options.log_severity_level = 4  # 判定中のエラーログは出さない
        # 判定するだけなので 1 本で足りる。未指定だとコア数-1 本作られ、
        # torch のロードと重なる起動直後に無駄な負荷がかかる。
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(), sess_options=options,
            providers=["CUDAExecutionProvider"],
        )
        return "CUDAExecutionProvider" in session.get_providers()
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _wants_cuda(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("cuda", "gpu"))


def resolve_onnx_provider(
    requested: str, *, label: str = "", on_output: Callable[[str], None] | None = None
) -> str:
    """ONNX Runtime の provider 名を実情に合わせる。CUDA が無ければ cpu。"""
    if not _wants_cuda(requested):
        return str(requested or "cpu").strip() or "cpu"
    if onnx_cuda_available():
        return str(requested).strip()
    _log(on_output, f"[device] {label or 'onnx'}: CUDA が使えないため CPU で実行します（低速）。")
    return "cpu"


def resolve_torch_device(
    requested: str, *, label: str = "", on_output: Callable[[str], None] | None = None
) -> str:
    """torch のデバイス指定（cuda / cuda:0 など）を実情に合わせる。"""
    if not _wants_cuda(requested):
        return str(requested or "cpu").strip() or "cpu"
    if torch_cuda_available():
        return str(requested).strip()
    _log(on_output, f"[device] {label or 'torch'}: CUDA が使えないため CPU で実行します（低速）。")
    return "cpu"

# --- UI の選択肢のうち GPU を要する値 --------------------------------------
# CPU 構成（環境構築時に -Device cpu を選んだ環境）では、これらを選んでも動かない。
# 黙って cpu に落とす処理は dialog/settings.py 側に入っているが、選べてしまうと
# 「選んだのに効かない」ことになるので、UI では出したまま押せなくする
# （エンジン管理のチェックボックスと同じ扱い）。
#
# 鍵は **ウィジェットの属性名**。これらのデバイス系設定は APP_CONFIG_FIELDS に
# 載っていない（MDF/global 側の別管理）ため config キーからは attr を引けない。
# attr なら PyQt は getattr、Web はマニフェストの `attr` フィールドで同じ語彙が使える。
GPU_ONLY_CHOICES: dict[str, tuple[str, ...]] = {
    "asr_device_combo": ("cuda",),
    # CTranslate2 の float16 系は GPU 専用。float32 は CPU でも動くので残す。
    "asr_compute_type_combo": ("int8_float16", "float16"),
    "asr_sherpa_provider_combo": ("GPU (cuda)",),
    "asr_qwen3_onnx_provider_combo": ("GPU (cuda)",),
    "asr_parakeet_onnx_provider_combo": ("GPU (cuda)",),
    "asr_sensevoice_onnx_provider_combo": ("GPU (cuda)",),
    "asr_funasr_nano_onnx_provider_combo": ("GPU (cuda)",),
    "asr_cohere_onnx_provider_combo": ("GPU (cuda)",),
    "asr_nemotron_onnx_compute_combo": ("GPU (fp32)",),
    "maai_device_combo": ("cuda",),
    # AUTO も GPU を試すので CPU 構成では塞ぐ。旧値 "GPU" も残す（互換）。
    "voicevox_core_device_combo": ("GPU (CUDA)", "GPU (DirectML)", "GPU", "AUTO"),
}

# CPU 構成で押せなくする理由。ツールチップに出す。
GPU_ONLY_REASON = "この環境は CPU 構成で作られているため選べません（環境構築時に -Device cpu を指定）。"


def is_cpu_build() -> bool:
    """環境構築時に CPU 構成を選んだ環境か。

    実行時に CUDA の有無を探るのではなく、`.device_extra` を見る。探る方式は
    ONNX Runtime の CUDA セッションを作ると後続の torch.cuda.is_available() が
    False を返すようになるため、どのエンジンが先に聞いたかで答えが変わる。
    """
    try:
        from . import engine_manager

        return engine_manager.device_extra() == "cpu"
    except Exception:  # noqa: BLE001
        return False


def disabled_choices() -> dict[str, tuple[str, ...]]:
    """この環境で選ばせない選択肢。GPU 構成なら空。"""
    return dict(GPU_ONLY_CHOICES) if is_cpu_build() else {}
