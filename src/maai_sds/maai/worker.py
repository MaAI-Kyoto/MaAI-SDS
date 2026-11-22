from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths as _paths
from .frame_capture import install_chunk_frame_capture, project_captured_frames

# SDS 同梱アセット（onnx / ローカル .pt）。MaAI のリポジトリの置き場所には依存しない。
_DEFAULT_MAAI_ROOT = _paths.assets_root()


def _resolve_maai_root(config: dict[str, Any]) -> Path:
    raw = str(config.get("maai_root") or config.get("maai_onnx_root") or "").strip()
    if raw:
        return Path(raw)
    # 互換: 旧設定の maai_src_path (.../src) があれば親を root とみなす
    src = str(config.get("maai_src_path") or "").strip()
    if src:
        p = Path(src)
        return p.parent if p.name.lower() == "src" else p
    return _DEFAULT_MAAI_ROOT


def _pair_if_exists(onnx_path: Path, meta_path: Path | None = None) -> tuple[str, str] | None:
    meta = meta_path if meta_path is not None else Path(str(onnx_path) + ".json")
    if onnx_path.is_file() and meta.is_file():
        return str(onnx_path), str(meta)
    return None


def _resolve_inference_chunk_frames(config: dict[str, Any]) -> int:
    """``inference_chunk_frames``: Mimi encoder が 1 回の ONNX 呼び出しで処理する

    フレーム数。maai 側は Mimi encoder 限定でこれを検証する（Torch 経路や
    encoder_type!="mimi" では 1 以外を渡すと ValueError）。既定 1 は従来どおり
    毎フレーム推論。
    """
    _raw = config.get("inference_chunk_frames", 1)
    n = int(_raw) if _raw is not None else 1
    if n < 1:
        raise ValueError(f"inference_chunk_frames must be at least 1: got {n}")
    return n


def _chunk_frame_plot_ts(result: Any, n_frames: int, frame_rate: float) -> list[float]:
    """1 回の推論で出た N フレームに、プロット用の推論時刻 ``t`` を 1 ホップずつずらして振る。

    取り込んだフレームには ``t`` が無く、GUI は受信時刻で描くので、同じ回の
    フレームが同じ横位置に重なって階段状に見える。最後のフレームを推論完了時刻
    （無ければ今）に置き、前のフレームを 1/frame_rate ずつ前へ並べる。頷きの判定は
    ``audio_timeline_sec`` を使うので影響しない。
    """
    t_last = None
    if isinstance(result, dict):
        try:
            t_last = float(result.get("t"))
        except (TypeError, ValueError):
            t_last = None
    if t_last is None:
        t_last = time.time()
    hop = 1.0 / max(1e-6, float(frame_rate))
    n = max(1, int(n_frames))
    return [t_last - hop * (n - 1 - i) for i in range(n)]


def _resolve_onnx_bundle(config: dict[str, Any]) -> dict[str, Any]:
    """Mimi encoder ONNX パスを解決する。

    ``use_mimi_onnx`` が True のとき ORT。``use_local_mimi_onnx`` が True のときだけ
    maai_root / 明示パスのローカルファイルを渡す。False なら maai 側が HF から取得する。
    BC / nod / VAP transformer は常に Torch。

    ``inference_chunk_frames`` > 1 のときは 1 フレーム版と別ファイルの ONNX
    （``continuous_mimi_{precision}_{N}f.onnx``）が要る。ローカル指定時に
    それが無いと（HF への暗黙フォールバックはしないので）ここで明示的に失敗する。
    """
    use_mimi = bool(config.get("use_mimi_onnx", True))
    use_local = bool(config.get("use_local_mimi_onnx", True))
    chunk_frames = _resolve_inference_chunk_frames(config)
    if chunk_frames > 1 and not use_mimi:
        raise ValueError(
            "inference_chunk_frames > 1 currently supports the Mimi encoder only "
            f"(got inference_chunk_frames={chunk_frames}, use_mimi_onnx=False)."
        )
    out: dict[str, Any] = {
        "use_mimi_onnx": bool(use_mimi),
        "use_local_mimi_onnx": bool(use_local),
        "inference_chunk_frames": chunk_frames,
    }
    if not use_mimi or not use_local:
        return out

    root = _resolve_maai_root(config)
    out["maai_root"] = str(root)

    # maai の download_continuous_mimi_onnx() と同じ命名規則:
    # 1 フレームは無印、N>1 は "_{N}f" サフィックス。
    _suffix = "" if chunk_frames == 1 else f"_{chunk_frames}f"
    mimi_onnx = Path(
        str(config.get("mimi_onnx_path") or "").strip()
        or (root / "onnx" / "onnx_v5" / f"continuous_mimi_fp32{_suffix}.onnx")
    )
    mimi_meta = Path(
        str(config.get("mimi_onnx_meta_path") or "").strip()
        or (str(mimi_onnx) + ".json" if str(mimi_onnx).endswith(".onnx") else str(mimi_onnx.with_suffix(".json")))
    )
    # onnx_v5 は ``continuous_mimi_fp32.json``（.onnx.json ではない）
    if not mimi_meta.is_file():
        alt = mimi_onnx.with_suffix(".json")
        if alt.is_file():
            mimi_meta = alt
    mimi_pair = _pair_if_exists(mimi_onnx, mimi_meta)
    if mimi_pair is not None:
        out["mimi_local_onnx_fp32_path"] = mimi_pair[0]
        out["mimi_local_onnx_fp32_meta_path"] = mimi_pair[1]
    else:
        # ローカル指定なのにファイルが無いと HF へ暗黙フォールバックしないよう明示する。
        hint = (
            f"（inference_chunk_frames={chunk_frames} 用の ONNX が別途必要です。"
            " use_local_mimi_onnx を無効にすると Hugging Face から自動取得されます）"
            if chunk_frames > 1 else ""
        )
        raise FileNotFoundError(
            "use_local_mimi_onnx=True ですが Mimi ONNX/meta が見つかりません: "
            f"onnx={mimi_onnx} meta={mimi_meta}{hint}"
        )

    return out


def _mimi_onnx_kwargs(bundle: dict[str, Any]) -> dict[str, Any]:
    if not bundle.get("use_mimi_onnx", True):
        return {"use_mimi_onnx": False}
    kw: dict[str, Any] = {
        "use_mimi_onnx": True,
        "inference_chunk_frames": int(bundle.get("inference_chunk_frames") or 1),
    }
    if bundle.get("mimi_local_onnx_fp32_path") and bundle.get("mimi_local_onnx_fp32_meta_path"):
        kw["mimi_local_onnx_fp32_path"] = bundle["mimi_local_onnx_fp32_path"]
        kw["mimi_local_onnx_fp32_meta_path"] = bundle["mimi_local_onnx_fp32_meta_path"]
    return kw

from ..ipc.jsonline import JsonLineEventClient
from ..ipc.worker_runtime import connect_worker_client, parse_json_object, safe_emit, to_jsonable as _to_jsonable
from ..audio.convert import f32_from_pcm16_bytes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaAI worker process")
    parser.add_argument("--host", required=True, type=str)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--config-json", required=True, type=str)
    return parser


def _redirect_stderr_to_devnull(os_mod, sys_mod) -> None:
    """シャットダウン時に stderr を /dev/null に向け直す。
    CRT fd 2（dup2）と Win32 STDERR ハンドル（Windows のみ）の両方を置き換える。
    """
    try:
        _fd = os_mod.open(os_mod.devnull, os_mod.O_WRONLY)
        os_mod.dup2(_fd, 2)
        os_mod.close(_fd)
    except Exception:
        pass
    try:
        sys_mod.stderr = open(os_mod.devnull, "w")
    except Exception:
        pass


def _incoming_base_ts(ts_end: float, sample_end: int, sample_rate: int) -> float:
    return float(ts_end) - (float(sample_end) / float(max(1, sample_rate)))


def _is_audio_timeline_discontinuous(
    *,
    prev_sample_end: int,
    prev_base_ts: float | None,
    sample_start: int,
    incoming_base_ts: float,
    rebase_tol_sec: float = 0.25,
) -> bool:
    """WAV stop→play やファンアウトドロップによるタイムライン断絶を検出する。"""
    if prev_base_ts is None:
        return False
    if int(sample_start) < int(prev_sample_end):
        return True
    if abs(int(sample_start) - int(prev_sample_end)) > 1:
        return True
    if abs(float(incoming_base_ts) - float(prev_base_ts)) > float(rebase_tol_sec):
        return True
    return False


# --- main() から切り出したヘルパ ---------------------------------------------
# 元は main の中のクロージャだった。掴んでいた外側の値（config / sample_rate）を
# 引数の先頭に明示しただけで、本文は変えていない。
# 状態を多く掴むループ系（_drain_user_chunks / *_result_loop など）は
# main のクロージャのまま残してある（状態が main の途中で段階的に作られるため）。

def _cfg_float(config, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(config.get(key, default))
        if v != v:  # NaN
            return default
        return max(lo, min(hi, v))
    except Exception:
        return default


def _cfg_str(config, key: str, default: str) -> str:
    s = str(config.get(key, "") or "").strip()
    return s if s else default


def _clear_queue_input(qi: Any) -> None:
    clear_fn = getattr(qi, "clear", None)
    if callable(clear_fn):
        try:
            clear_fn()
        except Exception:
            pass


def _clear_result_queue(model: Any) -> None:
    rq = getattr(model, "result_dict_queue", None)
    if rq is None:
        return
    while True:
        try:
            rq.get_nowait()
        except Exception:
            break


def _chunk_size_from_frame_rate(sample_rate, frame_rate: float) -> int:
    hz = float(frame_rate)
    if hz <= 0.0:
        hz = 10.0
    return max(1, int(round(float(sample_rate) / hz)))


def _inference_hop_sec(sample_rate, frame_rate: float) -> float:
    """1 推論ステップに対応する音声長（秒）。プロットの source_ts_start 幅に使う。"""
    n = _chunk_size_from_frame_rate(sample_rate, frame_rate)
    return float(n) / float(max(1, sample_rate))


def _last_float(v: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size > 0:
            return float(arr[-1])
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return float(default)


def _silence_maai_bc_frame_prints() -> None:
    """maai の BC フレームごとの debug print を捨てる。

    maai 4af9dfdd2 の `maai/models/vap_bc.py:176` が BC 推論のたびに
        print(f"p_bc: {p_bc:.4f}, p_bc_detect: {p_bc_detect:.4f}")
    を出す（上流のデバッグ出力。固定していた 108eb0fcf には無かった）。
    20Hz でコンソールが埋まり、stdout 経由の転送コストも乗る。

    `p_bc` は `maai_bc_result` イベントで既に届いており
    （プロットの系列にもなっている）ので、この行を捨てても情報は失わない。
    **この 1 行だけ**を落とし、`freeze encoder` など他の print はそのまま通す。
    ライブラリ本体は書き換えない（パッチを当てると上流更新のたびに壊れる）。
    """
    try:
        from maai.models import vap_bc  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001  -- 版によって場所が変わっても起動は続ける
        return
    real_print = print

    def _filtered(*args: object, **kwargs: object) -> None:
        first = args[0] if args else ""
        if isinstance(first, str) and first.startswith("p_bc: "):
            return
        real_print(*args, **kwargs)  # type: ignore[arg-type]

    vap_bc.print = _filtered  # type: ignore[attr-defined]


def _pin_pcores_for_realtime() -> None:
    """MaAI を高速コアへ寄せ、音声処理との競合による推論待ちを防ぐ。

    Windows は OS の EfficiencyClass でコアを判定する（先頭 8 個とは限らない）。
    以下の Linux 用の既存方針はそのまま維持する。

    Intel hybrid（例: Ultra 9 275HX は 8P+16E）を Linux は 24 均等コアに見せる。
    torch スレッドが E-core に散ると、同じ cpu 設定でも Windows 前景の
    PyQt より推論が実時間を割る。先頭 8 論理 CPU が P-core になりやすいので
    そこにピンする。ただし仮想マシン上（WSL2 / Docker Desktop）では番号が
    P-core と対応しないので固定しない（``linux_cpus_are_virtual``）。

    OpenMP 環境変数はセットしない。MaaiMultiple は BC/Nod/VAP をスレッド
    プールで同時実行するので、``OMP_MAX_ACTIVE_LEVELS=1`` や
    ``KMP_BLOCKTIME=0`` は入れ子並列を潰して RTF が約 2 倍になる
    （Windows 実測: 同じ 8/1・ORT 2 で 0.31 → 0.63。``multi_mic.py`` は 0.35）。
    """
    if sys.platform == "win32":
        from .cpu_affinity import pin_windows_performance_cores

        cores = pin_windows_performance_cores()
        if cores:
            print(f"[MaAIWorker][CPU] performance-core affinity={list(cores)}",
                  file=sys.stderr, flush=True)
        return
    if not hasattr(os, "sched_setaffinity"):
        return
    from .cpu_affinity import linux_cpus_are_virtual

    if linux_cpus_are_virtual():
        print("[MaAIWorker][CPU] 仮想 CPU のため固定しません（番号が P コアと対応しない）",
              file=sys.stderr, flush=True)
        return
    try:
        ncpu = os.cpu_count() or 0
        if ncpu >= 16:
            p_cores = set(range(min(8, ncpu)))
            os.sched_setaffinity(0, p_cores)
            print(
                f"[MaAIWorker][CPU] affinity={sorted(p_cores)} of 0-{ncpu - 1}",
                file=sys.stderr,
                flush=True,
            )
    except OSError:
        pass


class _InlineExecutor:
    """submit() の場でその関数を実行する（``concurrent.futures`` 互換の最小限）。"""

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as ex:  # noqa: BLE001 - Future に渡して呼び出し側で再送出
            fut.set_exception(ex)
        return fut

    def shutdown(self, wait: bool = True, **_: Any) -> None:
        return None


def _serialize_multi_subs_in_vm(maai_multi: Any) -> bool:
    """仮想マシン上では MaaiMultiple の BC/Nod/VAP を 1 スレッドで順に回す。

    MaaiMultiple は 3 モデルをスレッドプールで並列に走らせる。小さな torch 演算の
    たびに GIL を渡し合うので、1 フレームで約 1,700 回スレッドが切り替わる（順に
    回すと約 40 回）。WSL2 / Docker Desktop ではスレッドを起こすコストがネイティブの
    2〜3 倍（queue 往復 28〜32 µs 対 10〜17 µs）なので、これが推論時間の差になる。
    実測（Docker on Windows・bc + nod_para + vap_mc・1 フレーム）:
      並列 torch 8 スレッド 70.8 ms、順に回す 36.1 ms（ネイティブ並列は 57〜65 ms）。
    ネイティブでは並列の方が速いことがあるので、仮想 CPU のときだけ切り替える。
    ``start()`` の前に呼ぶ。プールをその場で実行する executor に差し替える
    （``start()`` は既存のプールを作り直さない）。
    """
    if sys.platform == "win32":
        return False
    from .cpu_affinity import linux_cpus_are_virtual

    if not linux_cpus_are_virtual():
        return False
    executor = getattr(maai_multi, "_sub_executor", None)
    if executor is None:
        return False
    maai_multi._sub_executor = _InlineExecutor()
    executor.shutdown(wait=False)
    print("[MaAIWorker][CPU] 仮想 CPU のため BC/Nod/VAP を順に実行します（並列だとスレッド切替が重い）",
          file=sys.stderr, flush=True)
    return True


def main() -> int:
    args = _build_parser().parse_args()
    try:
        config = parse_json_object(args.config_json, worker_name="MaAIWorker")
    except ValueError as ex:
        print(str(ex), file=sys.stderr, flush=True)
        return 2

    config.pop("audio_ingress_host", None)
    config.pop("audio_ingress_port", None)

    client = JsonLineEventClient(args.host, args.port)
    if not connect_worker_client(client, worker_name="MaAIWorker"):
        return 2

    def emit(event_type: str, **data: object) -> None:
        safe_emit(client.send, event_type, **data)

    # Keep Torch's internal pools bounded before MaAI creates any models.
    # MaaiMultiple concurrently runs BC/Nod/VAP, so the default 24/24 pools
    # oversubscribe this CPU alongside ONNX Runtime.
    torch_intra_threads = int(
        _cfg_float(config, "torch_cpu_intra_threads", 1.0, 1.0, 24.0)
    )
    _pin_pcores_for_realtime()
    import torch

    # interop も設定可能にする。f592640 で 1 にハードコードされたが、それ以前は
    # torch の既定（このマシンでは 24）だった。MaaiMultiple は BC/Nod/VAP を
    # 同時に走らせるので、op 間の並列を 1 に絞ると 3 モデル分が直列化しうる。
    # 一度でも並列実行が始まると変更できないので、モデル生成の前に決める。
    torch_interop_threads = int(
        _cfg_float(config, "torch_cpu_interop_threads", 1.0, 1.0, 24.0)
    )
    torch.set_num_threads(torch_intra_threads)
    try:
        torch.set_num_interop_threads(torch_interop_threads)
    except RuntimeError as ex:
        print(f"[MaAIWorker][Torch] interop の設定に失敗（既に並列実行済み）: {ex}",
              file=sys.stderr, flush=True)
        torch_interop_threads = torch.get_num_interop_threads()
    print(
        f"[MaAIWorker][Torch] intra={torch_intra_threads} inter={torch_interop_threads}",
        file=sys.stderr,
        flush=True,
    )

    # Import maai lazily so startup can fail with clear worker error.
    from maai import Maai  # type: ignore[import-not-found]
    from maai.input import Base as MaaiInputBase  # type: ignore[import-not-found]

    _silence_maai_bc_frame_prints()

    class QueueInput(MaaiInputBase):
        def __init__(self) -> None:
            super().__init__()

        def start(self) -> None:
            return

        def push(self, frame: np.ndarray) -> None:
            arr = np.asarray(frame, dtype=np.float32).reshape(-1)
            if arr.size <= 0:
                return
            self._put_to_all_queues([float(v) for v in arr])

        def clear(self) -> None:
            """購読キューを空にする（WAV 再開時の古い音声を捨てる）。"""
            with self._lock:
                for q in self._subscriber_queues:
                    try:
                        q.queue.clear()
                    except Exception:
                        while True:
                            try:
                                q.get_nowait()
                            except Exception:
                                break

    mode = str(config.get("mode", "both")).strip().lower()
    if mode not in {"bc", "nod_para", "both", "none"}:
        mode = "both"
    device = str(config.get("device", "cpu")).strip() or "cpu"
    bc_frame_rate = _cfg_float(config, "bc_frame_rate", 10.0, 0.1, 500.0)
    bc_lang = _cfg_str(config, "bc_lang", "jp")
    bc_model_type = _cfg_str(config, "bc_model_type", "normal")
    bc_local_model = (_cfg_str(config, "bc_local_model", "") or "").strip() or None
    nod_frame_rate = _cfg_float(config, "nod_frame_rate", 12.5, 0.1, 500.0)
    nod_lang = _cfg_str(config, "nod_lang", "jp")
    nod_model_type = _cfg_str(config, "nod_model_type", "normal-ver2")

    turn_vap_enabled = bool(config.get("turn_vap_enabled", False))
    turn_vap_mode = _cfg_str(config, "turn_vap_mode", "vap")
    turn_vap_lang = _cfg_str(config, "turn_vap_lang", "jp")
    turn_vap_frame_rate = _cfg_float(config, "turn_vap_frame_rate", 12.5, 0.1, 500.0)
    turn_vap_model_type = _cfg_str(config, "turn_vap_model_type", "normal")
    turn_vap_local_model = (_cfg_str(config, "turn_vap_local_model", "") or "").strip() or None
    turn_vap_device = str(config.get("turn_vap_device", "") or "").strip()
    turn_vap_device_resolved = turn_vap_device if turn_vap_device else device

    onnx_bundle = _resolve_onnx_bundle(config)
    mimi_kw = _mimi_onnx_kwargs(onnx_bundle)
    use_multi = bool(config.get("use_multi", False))
    multi_frame_rate = _cfg_float(config, "multi_frame_rate", 10.0, 0.1, 500.0)
    multi_model_type = _cfg_str(config, "multi_model_type", "normal-ver2")

    # Mimi ONNX のみ。SDS は ASR 等と同居するため ORT スレッドを絞る。
    onnx_thread_kw: dict[str, Any] = {}
    if mimi_kw.get("use_mimi_onnx"):
        onnx_thread_kw = {
            "mimi_onnx_cpu_intra_threads": int(
                _cfg_float(config, "mimi_onnx_cpu_intra_threads", 2.0, 1.0, 64.0)
            ),
            "mimi_onnx_cpu_inter_threads": int(
                _cfg_float(config, "mimi_onnx_cpu_inter_threads", 1.0, 1.0, 64.0)
            ),
        }
    print(
        "[MaAIWorker][ONNX] "
        f"mimi={bool(mimi_kw.get('use_mimi_onnx'))} "
        f"local={bool(onnx_bundle.get('use_local_mimi_onnx'))} "
        f"mimi_path={mimi_kw.get('mimi_local_onnx_fp32_path')} "
        f"device={device} "
        f"threads={onnx_thread_kw or 'ort-default'} "
        f"root={onnx_bundle.get('maai_root')}",
        file=sys.stderr,
        flush=True,
    )

    stop_event = threading.Event()
    sample_rate = 16000
    # ユーザー ch1 音声の累積サンプル終端（WAV 等のタイムライン推定用）。system_audio は更新しない。
    # base_ts は audio/input_worker（正準ソース）の音声サンプルクロックのアンカー
    # （sample 0 の壁時刻）。source ts を ASR/プロットと同一タイムラインに乗せるため、
    # 受信フレームの (ts_end, sample_end) から確定する。
    # pending_sample_ends: モデルへ push したチャンクの sample_end FIFO。
    # get_result のスタンプは「最新受信 sample_end」ではなくこの FIFO を使い、
    # バッファ滞留時に曲線・ターンが未来へずれるのを防ぐ。
    user_audio_timeline = {"sample_end": 0, "base_ts": None}
    audio_timeline_lock = threading.Lock()
    # モデル参照（断絶時の runtime reset 用）。load 後に埋める。
    live_models: list[Any] = []
    frame_captures: dict[str, Any] = {}

    def _note_user_sample_end_from_obj(obj: dict[str, object]) -> bool:
        """タイムラインを更新する。断絶（巻き戻し/rebase/ギャップ）なら True。"""
        se = obj.get("sample_end")
        ss = obj.get("sample_start")
        if se is None:
            return False
        try:
            v = int(se)
            if v < 0:
                return False
            sample_start = int(ss) if ss is not None else v
            ts_end = obj.get("ts_end")
            if ts_end is None:
                return False
            incoming_base = _incoming_base_ts(float(ts_end), v, sample_rate)
            with audio_timeline_lock:
                prev_se = int(user_audio_timeline.get("sample_end", 0) or 0)
                prev_base = user_audio_timeline.get("base_ts")
                discontinuous = _is_audio_timeline_discontinuous(
                    prev_sample_end=prev_se,
                    prev_base_ts=float(prev_base) if prev_base is not None else None,
                    sample_start=sample_start,
                    incoming_base_ts=incoming_base,
                )
                user_audio_timeline["sample_end"] = v
                if discontinuous:
                    user_audio_timeline["base_ts"] = float(incoming_base)
                elif user_audio_timeline.get("base_ts") is None:
                    user_audio_timeline["base_ts"] = float(incoming_base)
            return discontinuous
        except Exception:
            return False

    def _canonical_source_ts(se: int) -> float:
        """共有サンプル位置 se を、音声入力層と同一の音声サンプル時刻へ変換。

        base_ts 未確定（音声フレーム未着）の間のみ壁時計にフォールバックする。
        """
        with audio_timeline_lock:
            base = user_audio_timeline.get("base_ts")
        if base is not None:
            return float(base) + float(se) / float(sample_rate)
        return time.time()

    models: list[tuple[str, Any]] = []
    pending_models: list[tuple[str, Any]] = []
    model_slots: dict[str, dict[str, Any]] = {}

    def _reset_model_runtime(model: Any) -> None:
        reset_fn = getattr(model, "reset_runtime_state", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass
        _clear_result_queue(model)
        subs = getattr(model, "sub_maais", None)
        if isinstance(subs, (list, tuple)):
            for sub in subs:
                _reset_model_runtime(sub)

    def _reset_audio_pipeline_on_discontinuity() -> None:
        """WAV 再開・ドロップギャップ時: バッファと推論キューを捨ててライブ端へ復帰する。"""
        for slot in list(model_slots.values()):
            slot["user_buffer"] = np.empty(0, dtype=np.float32)
            slot["system_buffer"] = np.empty(0, dtype=np.float32)
            slot["buffer_sample_start"] = None
            pending = slot.get("pending_sample_ends")
            if isinstance(pending, deque):
                pending.clear()
            slot["legacy_startup_discard_frames"] = _legacy_startup_discard_frames()
            _clear_queue_input(slot.get("ch1_input"))
            _clear_queue_input(slot.get("ch2_input"))
        for model in list(live_models):
            _reset_model_runtime(model)

        for capture in frame_captures.values():
            capture.clear()

    def _inference_chunk_frames_now() -> int:
        return max(1, int(mimi_kw.get("inference_chunk_frames", 1) or 1))

    def _legacy_startup_discard_frames() -> int:
        """ONNX Mimi が最初のエンコーダ出力を1回捨てる分を、1フレーム時と同じ遅れに揃える。

        1フレーム時は捨てる音声が1フレームなので、結果は常に1フレーム前の sample_end に付く。
        Nフレーム時にその1回が N フレーム分だと遅れが N になる。起動時に N-1 個だけ捨て、
        残りの1個を1フレーム時と同じラグとして残す。Torch Mimi は捨てないので 0。
        """
        n = _inference_chunk_frames_now()
        if n <= 1 or not bool(mimi_kw.get("use_mimi_onnx", True)):
            return 0
        return n - 1

    def _pop_legacy_sample_ends(slot_name: str, count: int) -> list[int]:
        """公開結果1フレームにつき sample_end を1個、古い順に取り出す。

        ``count==1`` かつ起動時破棄が 0 のときは、従来の ``_pop_pushed_sample_end(1)`` と同じ。
        """
        need = max(1, int(count))
        slot = model_slots.get(slot_name)
        out: list[int] = []
        if slot is not None:
            pending = slot.get("pending_sample_ends")
            if isinstance(pending, deque):
                discard = max(0, int(slot.get("legacy_startup_discard_frames", 0) or 0))
                while discard > 0 and pending:
                    pending.popleft()
                    discard -= 1
                slot["legacy_startup_discard_frames"] = discard
                for _ in range(need):
                    if not pending:
                        break
                    out.append(int(pending.popleft()))
        if len(out) < need:
            with audio_timeline_lock:
                newest = int(user_audio_timeline.get("sample_end", 0) or 0)
            while len(out) < need:
                out.append(newest)
        return out

    def _stamp_legacy_frame(
        payload: dict[str, Any], *, frame_rate: float, sample_end: int
    ) -> None:
        """1フレーム時の ``_stamp_inference_ts`` と同じ幅で、渡された sample_end を載せる。"""
        hop = _inference_hop_sec(sample_rate, frame_rate)
        se = int(sample_end)
        ts_end = _canonical_source_ts(se)
        payload["source_ts_end"] = float(ts_end)
        payload["source_ts_start"] = float(ts_end - hop)
        payload["source_sample_end"] = se
        payload["audio_timeline_sec"] = float(se) / float(max(1, int(sample_rate)))

    def _take_chunk_frames(capture_key: str, public: dict) -> list[dict]:
        """チャンク推論で保持した全フレーム。無ければ公開結果（最後の1フレーム）だけ。"""
        capture = frame_captures.get(capture_key)
        captured = capture.pop() if capture is not None else None
        return project_captured_frames(captured, public)

    def _create_slot(model_name: str, frame_rate: float) -> dict[str, Any]:
        return {
            "name": model_name,
            "frame_rate": float(frame_rate),
            "chunk_size": _chunk_size_from_frame_rate(sample_rate, frame_rate),
            "ch1_input": QueueInput(),
            "ch2_input": QueueInput(),
            "user_buffer": np.empty(0, dtype=np.float32),
            "system_buffer": np.empty(0, dtype=np.float32),
            "buffer_sample_start": None,
            "pending_sample_ends": deque(),
            "legacy_startup_discard_frames": _legacy_startup_discard_frames(),
        }

    def _append_audio_to_slots(
        audio_f32: np.ndarray,
        *,
        is_system: bool,
        sample_start: int | None = None,
    ) -> None:
        x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if x.size <= 0:
            return
        for slot in list(model_slots.values()):
            key = "system_buffer" if is_system else "user_buffer"
            buf = np.asarray(slot.get(key), dtype=np.float32).reshape(-1)
            if (not is_system) and buf.size <= 0 and sample_start is not None:
                slot["buffer_sample_start"] = int(sample_start)
            new = x.copy() if buf.size <= 0 else np.concatenate([buf, x])
            # 上限超過分は古い側を捨てる。音声対話+WAV では長く溜めるとターンが
            # 実時間から外れるため、約 1 秒分だけ残してライブ端へ追いつく。
            chunk_size = int(slot.get("chunk_size", 0) or 0)
            if chunk_size > 0:
                cap = chunk_size * 12
                if new.size > cap:
                    dropped = int(new.size - cap)
                    new = new[new.size - cap:].copy()
                    if not is_system:
                        bss = slot.get("buffer_sample_start")
                        if bss is not None:
                            slot["buffer_sample_start"] = int(bss) + dropped
            slot[key] = new

    def _drain_user_chunks() -> None:
        # モデル未 ready のあいだは push しない（購読前の push は捨てられ、
        # pending FIFO だけが増えてスタンプがずれる）。バッファは cap で直近のみ保持。
        if not live_models:
            return
        for slot in list(model_slots.values()):
            chunk_size = int(slot.get("chunk_size", 0) or 0)
            if chunk_size <= 0:
                continue
            ch1_input = slot.get("ch1_input")
            ch2_input = slot.get("ch2_input")
            if ch1_input is None or ch2_input is None:
                continue
            user_buf = np.asarray(slot.get("user_buffer"), dtype=np.float32).reshape(-1)
            system_buf = np.asarray(slot.get("system_buffer"), dtype=np.float32).reshape(-1)
            pending = slot.get("pending_sample_ends")
            if not isinstance(pending, deque):
                pending = deque()
                slot["pending_sample_ends"] = pending
            bss = slot.get("buffer_sample_start")
            if bss is None:
                with audio_timeline_lock:
                    se_now = int(user_audio_timeline.get("sample_end", 0) or 0)
                bss = max(0, se_now - int(user_buf.size))
                slot["buffer_sample_start"] = int(bss)
            # system 待ち上限を短くし（~0.5s）、非対称ドロップ時もターンを実時間に保つ。
            max_user_backlog = chunk_size * 6
            while user_buf.size >= chunk_size:
                if system_buf.size >= chunk_size:
                    x2 = system_buf[:chunk_size].copy()
                    system_buf = system_buf[chunk_size:]
                elif system_buf.size == 0:
                    x2 = np.zeros(chunk_size, dtype=np.float32)
                elif user_buf.size > max_user_backlog:
                    x2 = np.zeros(chunk_size, dtype=np.float32)
                    x2[: system_buf.size] = system_buf
                    system_buf = np.empty(0, dtype=np.float32)
                else:
                    break
                x1 = user_buf[:chunk_size].copy()
                user_buf = user_buf[chunk_size:]
                chunk_sample_end = int(bss) + int(chunk_size)
                bss = int(chunk_sample_end)
                pending.append(int(chunk_sample_end))
                while len(pending) > 64:
                    pending.popleft()
                ch1_input.push(x1)
                ch2_input.push(x2)
            slot["user_buffer"] = user_buf
            slot["system_buffer"] = system_buf
            slot["buffer_sample_start"] = int(bss) if user_buf.size > 0 else None

    def _stdin_watcher() -> None:
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if line.lower() == "stop":
                    stop_event.set()
                    return
                if line.startswith("system_audio:"):
                    try:
                        audio_bytes = base64.b64decode(line[13:])
                        audio_f32 = f32_from_pcm16_bytes(audio_bytes)
                        _append_audio_to_slots(audio_f32, is_system=True)
                    except Exception:
                        pass
                    continue
                if line.startswith("system_audio_frame:"):
                    try:
                        obj = json.loads(line[len("system_audio_frame:") :])
                        audio_f32 = np.asarray(obj.get("audio", []), dtype=np.float32).reshape(-1)
                        if audio_f32.size <= 0:
                            continue
                        _append_audio_to_slots(audio_f32, is_system=True)
                    except Exception:
                        pass
                    continue
                if line.startswith("audio_frame:"):
                    try:
                        obj = json.loads(line[len("audio_frame:") :])
                        if not isinstance(obj, dict):
                            continue
                        discontinuous = _note_user_sample_end_from_obj(obj)
                        if discontinuous:
                            _reset_audio_pipeline_on_discontinuity()
                        x1 = np.asarray(obj.get("audio", []), dtype=np.float32).reshape(-1)
                        if x1.size <= 0:
                            continue
                        ss = obj.get("sample_start")
                        sample_start = int(ss) if ss is not None else None
                        _append_audio_to_slots(x1, is_system=False, sample_start=sample_start)
                        _drain_user_chunks()
                    except Exception:
                        pass
                    continue
            stop_event.set()
        except Exception:
            stop_event.set()

    def _on_audio_event(payload: dict[str, object]) -> None:
        event_type = str(payload.get("type", "")).strip().lower()
        if event_type == "system_audio_frame":
            audio_f32 = np.asarray(payload.get("audio", []), dtype=np.float32).reshape(-1)
            if audio_f32.size > 0:
                _append_audio_to_slots(audio_f32, is_system=True)
            return
        if event_type == "audio_frame":
            discontinuous = _note_user_sample_end_from_obj(payload)
            if discontinuous:
                _reset_audio_pipeline_on_discontinuity()
            x1 = np.asarray(payload.get("audio", []), dtype=np.float32).reshape(-1)
            if x1.size > 0:
                ss = payload.get("sample_start")
                sample_start = int(ss) if ss is not None else None
                _append_audio_to_slots(x1, is_system=False, sample_start=sample_start)
                _drain_user_chunks()
            return

    _loop_err_last_log: dict[str, float] = {}

    def _log_loop_error(where: str, ex: BaseException) -> None:
        """結果ループの emit/シリアライズ失敗を握りつぶさず、最大 ~5 秒に 1 回ログする。

        握りつぶすと IPC 切断やシリアライズ不能が「MaAI が無出力」という見えない停止になり、
        原因切り分けができなくなるため。
        """
        now = time.monotonic()
        if now - _loop_err_last_log.get(where, 0.0) < 5.0:
            return
        _loop_err_last_log[where] = now
        try:
            emit("log", message=f"[MaAIWorker] {where} result loop error: {ex!r}")
        except Exception:
            print(f"[MaAIWorker] {where} result loop error: {ex!r}", file=sys.stderr, flush=True)

    def _result_loop(model_name: str, maai_model: Any, frame_rate: float) -> None:
        event_type = "maai_bc_result" if model_name == "bc" else "maai_nod_result"
        while not stop_event.is_set():
            try:
                result = maai_model.get_result()
            except Exception:
                if stop_event.is_set():
                    return
                continue
            if not isinstance(result, dict):
                continue
            try:
                frames = _take_chunk_frames(model_name, result)
                sample_ends = _pop_legacy_sample_ends(model_name, len(frames))
                if len(frames) < len(sample_ends):
                    sample_ends = sample_ends[-len(frames) :]
                plot_ts = _chunk_frame_plot_ts(result, len(frames), float(frame_rate))
                for frame, sample_end, t_plot in zip(frames, sample_ends, plot_ts):
                    payload = _to_jsonable(frame)
                    if not isinstance(payload, dict):
                        continue
                    payload["maai_mode"] = model_name
                    _stamp_legacy_frame(
                        payload, frame_rate=float(frame_rate), sample_end=int(sample_end)
                    )
                    payload["t"] = t_plot
                    emit(event_type, result=payload)
            except Exception as ex:
                _log_loop_error(model_name, ex)

    def _turn_vap_result_loop(maai_model: Any, frame_rate: float) -> None:
        while not stop_event.is_set():
            try:
                result = maai_model.get_result()
            except Exception:
                if stop_event.is_set():
                    return
                continue
            if not isinstance(result, dict):
                continue
            try:
                frames = _take_chunk_frames("turn_vap", result)
                sample_ends = _pop_legacy_sample_ends("turn_vap", len(frames))
                if len(frames) < len(sample_ends):
                    sample_ends = sample_ends[-len(frames) :]
                plot_ts = _chunk_frame_plot_ts(result, len(frames), float(frame_rate))
                for frame, sample_end, t_plot in zip(frames, sample_ends, plot_ts):
                    payload = _to_jsonable(frame)
                    if not isinstance(payload, dict):
                        continue
                    payload["maai_mode"] = "turn_vap"
                    _stamp_legacy_frame(
                        payload, frame_rate=float(frame_rate), sample_end=int(sample_end)
                    )
                    payload["t"] = t_plot
                    ts_end = float(payload.get("source_ts_end", time.time()))
                    emit("maai_result", result=payload)
                    emit(
                        "turn",
                        probability=_last_float(frame.get("p_now"), 0.0),
                        p_future=_last_float(frame.get("p_future"), 0.0),
                        ts=ts_end,
                        source_ts_end=ts_end,
                    )
            except Exception as ex:
                _log_loop_error("turn_vap", ex)

    def _multi_result_loop(maai_multi: Any, frame_rate: float) -> None:
        # inference_chunk_frames > 1 でも、キャプチャした全フレームを1フレーム時と同じ順で出す。
        while not stop_event.is_set():
            try:
                result = maai_multi.get_result()
            except Exception:
                if stop_event.is_set():
                    return
                continue
            if not isinstance(result, dict):
                continue
            try:
                bc_frames: list[dict] = []
                nod_legacy_frames: list[dict] = []
                turn_frames: list[dict] = []
                if isinstance(result.get("bc"), dict):
                    bc_frames = _take_chunk_frames("multi:bc", result["bc"])
                if isinstance(result.get("nod_para"), dict):
                    nod_legacy_frames = _take_chunk_frames("multi:nod_para", result["nod_para"])
                if isinstance(result.get("turn_vap"), dict):
                    turn_frames = _take_chunk_frames("multi:turn_vap", result["turn_vap"])
                n_legacy = max(
                    (len(frames) for frames in (bc_frames, nod_legacy_frames, turn_frames) if frames),
                    default=0,
                )
                # 公開結果が欠けた場合も推論チャンク分だけ FIFO を進める。
                if n_legacy <= 0:
                    n_legacy = _inference_chunk_frames_now()
                sample_ends = _pop_legacy_sample_ends("multi", n_legacy)

                def _ends_for(frames: list[dict]) -> list[int]:
                    if not frames:
                        return []
                    if len(frames) <= len(sample_ends):
                        return sample_ends[-len(frames) :]
                    return sample_ends

                if bc_frames:
                    plot_ts = _chunk_frame_plot_ts(result, len(bc_frames), float(frame_rate))
                    for frame, sample_end, t_plot in zip(bc_frames, _ends_for(bc_frames), plot_ts):
                        payload = _to_jsonable(frame)
                        if not isinstance(payload, dict):
                            continue
                        payload["maai_mode"] = "bc"
                        _stamp_legacy_frame(
                            payload, frame_rate=float(frame_rate), sample_end=int(sample_end)
                        )
                        payload["t"] = t_plot
                        emit("maai_bc_result", result=payload)
                if nod_legacy_frames:
                    plot_ts = _chunk_frame_plot_ts(result, len(nod_legacy_frames), float(frame_rate))
                    for frame, sample_end, t_plot in zip(
                        nod_legacy_frames, _ends_for(nod_legacy_frames), plot_ts
                    ):
                        payload = _to_jsonable(frame)
                        if not isinstance(payload, dict):
                            continue
                        payload["maai_mode"] = "nod_para"
                        _stamp_legacy_frame(
                            payload, frame_rate=float(frame_rate), sample_end=int(sample_end)
                        )
                        payload["t"] = t_plot
                        emit("maai_nod_result", result=payload)
                if turn_frames:
                    plot_ts = _chunk_frame_plot_ts(result, len(turn_frames), float(frame_rate))
                    for frame, sample_end, t_plot in zip(turn_frames, _ends_for(turn_frames), plot_ts):
                        payload = _to_jsonable(frame)
                        if not isinstance(payload, dict):
                            continue
                        payload["maai_mode"] = "turn_vap"
                        _stamp_legacy_frame(
                            payload, frame_rate=float(frame_rate), sample_end=int(sample_end)
                        )
                        payload["t"] = t_plot
                        ts_end = float(payload.get("source_ts_end", time.time()))
                        emit("maai_result", result=payload)
                        emit(
                            "turn",
                            probability=_last_float(frame.get("p_now"), 0.0),
                            p_future=_last_float(frame.get("p_future"), 0.0),
                            ts=ts_end,
                            source_ts_end=ts_end,
                        )
            except Exception as ex:
                _log_loop_error("multi", ex)

    multi_slot: dict[str, Any] | None = None
    multi_configs: list[dict] | None = None
    if use_multi:
        from maai import MaaiMultiple  # type: ignore[import-not-found]

        multi_configs = []
        if mode in {"bc", "both"}:
            _bc_cfg: dict[str, Any] = {"mode": "bc", "lang": bc_lang, "label": "bc"}
            if bc_local_model is not None:
                _bc_cfg["local_model"] = bc_local_model
            multi_configs.append(_bc_cfg)
        if mode in {"nod_para", "both"}:
            _nod_cfg: dict[str, Any] = {"mode": "nod_para", "lang": nod_lang, "label": "nod_para"}
            multi_configs.append(_nod_cfg)
        if turn_vap_enabled:
            _tv_cfg: dict[str, Any] = {
                "mode": turn_vap_mode,
                "lang": turn_vap_lang,
                "label": "turn_vap",
            }
            if turn_vap_local_model is not None:
                _tv_cfg["local_model"] = turn_vap_local_model
            multi_configs.append(_tv_cfg)
        if not multi_configs:
            emit("fatal", message="マルチモード: 有効なモデルがありません。")
            client.close()
            return 1
        if any(c.get("mode") == "nod_para" for c in multi_configs):
            from transformers import MimiConfig as _MimiConfig  # type: ignore
            from transformers import MimiModel as _MimiModel  # type: ignore
            _ = (_MimiConfig, _MimiModel)
        multi_slot = _create_slot("multi", multi_frame_rate)
        model_slots["multi"] = multi_slot
    elif mode in {"nod_para", "both"}:
        # nod_para 初期化中の transformers import ハングを避けるため、
        # IPC 受信スレッド起動より先に import を済ませる。
        from transformers import MimiConfig as _MimiConfig  # type: ignore
        from transformers import MimiModel as _MimiModel  # type: ignore
        _ = (_MimiConfig, _MimiModel)

    # コンストラクタ（モデルロード）より先に IPC 受信スレッドを起動し、
    # 送信側 OS ソケットバッファが詰まってオーケストレータをブロックするのを防ぐ。
    threading.Thread(target=_stdin_watcher, daemon=True, name="maai-worker-stdin").start()
    client.start_reader(_on_audio_event, on_disconnected=lambda: stop_event.set())

    if use_multi:
        assert multi_slot is not None and multi_configs is not None
        try:
            maai_multi = MaaiMultiple(
                configs=multi_configs,
                audio_ch1=multi_slot["ch1_input"],
                audio_ch2=multi_slot["ch2_input"],
                frame_rate=multi_frame_rate,
                device=device,
                model_type=multi_model_type,
                **mimi_kw,
                **onnx_thread_kw,
            )
            if _inference_chunk_frames_now() > 1:
                for cfg in multi_configs:
                    label = str(cfg.get("label") or "")
                    if not label:
                        continue
                    capture = install_chunk_frame_capture(
                        maai_multi.get_sub_maai(label),
                        inference_chunk_frames=_inference_chunk_frames_now(),
                    )
                    if capture is not None:
                        frame_captures[f"multi:{label}"] = capture
            _serialize_multi_subs_in_vm(maai_multi)
            maai_multi.start()
        except Exception as ex:
            emit("fatal", message=str(ex))
            client.close()
            return 1
        live_models.append(maai_multi)
        # ロード中に溜まったバッファは捨て、ready 以降のライブ端から推論する。
        _reset_audio_pipeline_on_discontinuity()
        multi_nod_threshold_defaults: dict[str, object] | None = None
        try:
            nod_sub = maai_multi.get_sub_maai("nod_para")
            vap_obj = getattr(nod_sub, "vap", None)
            # maai の VapNodPara は閾値を `nod_repetitions_thresholds` で保持する
            # （`nod_count_thresholds` という属性は存在せず、従来は常に None だった）。
            if vap_obj is not None and hasattr(vap_obj, "nod_repetitions_thresholds"):
                multi_nod_threshold_defaults = {
                    "nod_count_thresholds": dict(vap_obj.nod_repetitions_thresholds),
                    "nod_swing_up_threshold": float(vap_obj.nod_swing_up_threshold),
                }
        except Exception:
            multi_nod_threshold_defaults = None
        emit("ready", nod_threshold_defaults=multi_nod_threshold_defaults)
        threading.Thread(
            target=_multi_result_loop,
            args=(maai_multi, float(multi_frame_rate)),
            daemon=True,
            name="maai-worker-result-multi",
        ).start()
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            import os as _os
            import sys as _sys
            _redirect_stderr_to_devnull(_os, _sys)
            try:
                emit("stopped")
            except BaseException:
                pass
            try:
                client.close()
            except BaseException:
                pass
            _os._exit(0)
        return 0

    try:
        if mode in {"bc", "both"}:
            bc_slot = _create_slot("bc", bc_frame_rate)
            model_slots["bc"] = bc_slot
            bc_model = Maai(
                mode="bc",
                lang=bc_lang,
                frame_rate=bc_frame_rate,
                audio_ch1=bc_slot["ch1_input"],
                audio_ch2=bc_slot["ch2_input"],
                device=device,
                model_type=bc_model_type,
                local_model=bc_local_model,
                **mimi_kw,
                **onnx_thread_kw,
            )
            capture = install_chunk_frame_capture(
                bc_model, inference_chunk_frames=_inference_chunk_frames_now()
            )
            if capture is not None:
                frame_captures["bc"] = capture
            pending_models.append(("bc", bc_model))
        if mode in {"nod_para", "both"}:
            nod_slot = _create_slot("nod_para", nod_frame_rate)
            model_slots["nod_para"] = nod_slot
            nod_model = Maai(
                mode="nod_para",
                lang=nod_lang,
                frame_rate=nod_frame_rate,
                audio_ch1=nod_slot["ch1_input"],
                audio_ch2=nod_slot["ch2_input"],
                device=device,
                model_type=nod_model_type,
                **mimi_kw,
                **onnx_thread_kw,
            )
            capture = install_chunk_frame_capture(
                nod_model, inference_chunk_frames=_inference_chunk_frames_now()
            )
            if capture is not None:
                frame_captures["nod_para"] = capture
            pending_models.append(("nod_para", nod_model))
        if turn_vap_enabled:
            turn_slot = _create_slot("turn_vap", turn_vap_frame_rate)
            model_slots["turn_vap"] = turn_slot
            turn_model = Maai(
                mode=turn_vap_mode,
                lang=turn_vap_lang,
                frame_rate=turn_vap_frame_rate,
                audio_ch1=turn_slot["ch1_input"],
                audio_ch2=turn_slot["ch2_input"],
                device=turn_vap_device_resolved,
                model_type=turn_vap_model_type,
                local_model=turn_vap_local_model,
                **mimi_kw,
                **onnx_thread_kw,
            )
            capture = install_chunk_frame_capture(
                turn_model, inference_chunk_frames=_inference_chunk_frames_now()
            )
            if capture is not None:
                frame_captures["turn_vap"] = capture
            pending_models.append(("turn_vap", turn_model))
        for model_name, maai_model in pending_models:
            maai_model.start()
            models.append((model_name, maai_model))
            live_models.append(maai_model)
        if not models:
            emit("fatal", message="有効な MaAI モードがありません。")
            return 1
        _reset_audio_pipeline_on_discontinuity()
    except Exception as ex:
        emit("fatal", message=str(ex))
        client.close()
        return 1

    for model_name, maai_model in models:
        if model_name == "turn_vap":
            threading.Thread(
                target=_turn_vap_result_loop,
                args=(maai_model, float(turn_vap_frame_rate)),
                daemon=True,
                name="maai-worker-result-turn_vap",
            ).start()
        elif model_name == "bc":
            threading.Thread(
                target=_result_loop,
                args=(model_name, maai_model, float(bc_frame_rate)),
                daemon=True,
                name=f"maai-worker-result-{model_name}",
            ).start()
        elif model_name == "nod_para":
            threading.Thread(
                target=_result_loop,
                args=(model_name, maai_model, float(nod_frame_rate)),
                daemon=True,
                name=f"maai-worker-result-{model_name}",
            ).start()
        else:
            threading.Thread(
                target=_result_loop,
                args=(model_name, maai_model, 10.0),
                daemon=True,
                name=f"maai-worker-result-{model_name}",
            ).start()

    nod_threshold_defaults: dict[str, object] | None = None
    for _name, _maai_model in models:
        if _name != "nod_para":
            continue
        try:
            vap = getattr(_maai_model, "vap", None)
            # 属性名は `nod_repetitions_thresholds`（`nod_count_thresholds` は存在しない）。
            if vap is not None and hasattr(vap, "nod_repetitions_thresholds"):
                nod_threshold_defaults = {
                    "nod_count_thresholds": dict(vap.nod_repetitions_thresholds),
                    "nod_swing_up_threshold": float(vap.nod_swing_up_threshold),
                }
        except Exception:
            nod_threshold_defaults = None
        break
    emit("ready", nod_threshold_defaults=nod_threshold_defaults)
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        import os as _os
        import sys as _sys
        _redirect_stderr_to_devnull(_os, _sys)
        try:
            emit("stopped")
        except BaseException:
            pass
        try:
            client.close()
        except BaseException:
            pass
        _os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
