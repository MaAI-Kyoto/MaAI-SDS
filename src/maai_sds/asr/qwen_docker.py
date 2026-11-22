"""Qwen3-ASR（vLLM・公式の qwenllm/qwen3-asr イメージ）のコンテナを、GUI・Web から起動・停止するための部品。

- この PC の対話サーバー: docker-compose.qwen-asr.yml（container_name qwen3-asr）で PC 側にコンテナを立てる。
  Qt は `scripts/start_qwen3_asr.ps1` で、Web は local_compose_target() で同じファイルを使う
- Docker の対話サーバー: docker-compose.yml の `qwen3-asr` サービス（プロファイル `qwen3`）を使い、対話サーバーからは
  サービス名（qwen3-asr:80）でつなぐ。Qt と PC 側の Web は、対話サーバーと同じ compose で
  `--profile qwen3 up -d qwen3-asr` を実行し、ホストに出した 18000 番の /health が返るまで待つ
- compose の中の Web: 自分もコンテナの中で compose を操作できないので、/health に届くかだけ確かめ、
  届かなければ起動のしかたを伝える
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

CONTAINER_REPO = "/data/shared/Qwen3-ASR"  # compose がリポジトリを置く場所（docker-compose*.yml の ./:/data/shared/Qwen3-ASR）
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"

SERVICE = "qwen3-asr"
PROFILE = "qwen3"
CONTAINER_NAME = "maai-sds-qwen3-asr"  # docker-compose.yml の container_name（PC 側の qwen3-asr と分ける）
HOST_PORT = 18000  # docker-compose.yml がホストに出すポート（web の 8000 とぶつからないよう）
START_HINT = (
    "Qwen3-ASR（vLLM）のサービスが動いていません。Docker の対話サーバーを "
    "`docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile qwen3 up -d` のように "
    "`--profile qwen3` を付けて起動してください（初回はイメージとモデルの取得に数分かかります）。"
)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def uses_qwen(cfg: dict[str, Any]) -> bool:
    """設定の ASR が Qwen3-ASR（vLLM）を使うか（比較モードの上段・下段も見る）。"""
    if _truthy(cfg.get("asr_compare_mode_enabled", False)):
        return "qwen3-asr" in {str(cfg.get("asr_compare_top_engine", "")).strip(),
                               str(cfg.get("asr_compare_bottom_engine", "")).strip()}
    return str(cfg.get("asr_engine", "")).strip() == "qwen3-asr"


def local_compose_target(repo_root: Path) -> Any:
    """この PC の対話サーバー用の compose（docker-compose.qwen-asr.yml。Qt の scripts/start_qwen3_asr.ps1 と同じ）。

    このファイルの qwen3-asr にはプロファイルが無いので、--profile qwen3 を付けても常に対象になる。
    公開するポートは環境変数 QWEN_ASR_HOST_PORT（設定の asr_qwen_port）で決まる。
    """
    from ..dialog_server_docker import DockerComposeTarget

    return DockerComposeTarget(files=(str(repo_root / "docker-compose.qwen-asr.yml"),), workdir=str(repo_root))


def container_model_path(path_str: str, repo_root: Path) -> str:
    """PC 側のモデルのフォルダーを、Qwen3-ASR のコンテナから見たパスにする（リポジトリの外ならそのまま）。"""
    text = path_str.strip()
    norm = text.replace("\\", "/")
    if norm.startswith("/data/"):
        return norm
    host = Path(text).expanduser()
    if not host.is_absolute():
        host = (repo_root / host).resolve()
    else:
        try:
            host = host.resolve()
        except OSError:
            return norm
    try:
        rel = host.relative_to(repo_root)
    except ValueError:
        return host.as_posix()
    return f"{CONTAINER_REPO}/{rel.as_posix()}"


def served_model_name(model_ref: str) -> str:
    """vLLM に名乗らせるモデル名（モデル参照の末尾。dialog.settings.derive_qwen_served_model_name と同じ式）。"""
    normalized = (model_ref or "").strip().replace("\\", "/").rstrip("/")
    return normalized.split("/")[-1] if normalized else "Qwen3-ASR-0.6B"


def compose_env_from_config(cfg: dict[str, Any], *, repo_root: Path,
                            host_path_of: Callable[[str], Path | None] | None = None) -> dict[str, str]:
    """保存済みの設定から compose の環境変数を作る（Web 用。Qt は画面の入力欄から作る）。

    host_path_of は、設定に入っている対話サーバー側のパスを PC 側のパスに戻す関数（Docker のマウントの対応）。
    """
    custom = str(cfg.get("asr_qwen_custom_model_path") or "").strip()
    if custom:
        host = host_path_of(custom) if host_path_of is not None else None
        model = container_model_path(str(host) if host is not None else custom, repo_root)
    else:
        model = str(cfg.get("asr_qwen_model") or "").strip() or DEFAULT_MODEL
    return compose_env(
        model=model,
        served_model_name=served_model_name(model),
        gpu_memory_utilization=str(cfg.get("asr_qwen_gpu_memory_utilization") or "").strip() or "0.8",
        max_model_len=str(cfg.get("asr_qwen_max_model_len") or "").strip() or "4096",
        session_timeout_sec=str(cfg.get("asr_qwen_session_timeout_sec") or "").strip() or "60",
        chunk_sec=str(cfg.get("asr_qwen_chunk_sec") or "").strip() or "2.0",
        enforce_eager=_truthy(cfg.get("asr_qwen_vllm_enforce_eager", False)),
    )


def compose_env(*, model: str, served_model_name: str, gpu_memory_utilization: str, max_model_len: str,
                session_timeout_sec: str, chunk_sec: str, enforce_eager: bool) -> dict[str, str]:
    """docker-compose.yml の qwen3-asr サービスに渡す環境変数（start_qwen3_asr.ps1 と同じ名前）。"""
    return {
        "QWEN_ASR_MODEL": model,
        "QWEN_ASR_SERVED_MODEL_NAME": served_model_name,
        "QWEN_ASR_GPU_MEMORY_UTILIZATION": gpu_memory_utilization,
        "QWEN_ASR_MAX_MODEL_LEN": max_model_len,
        "QWEN_ASR_SESSION_TIMEOUT_SEC": session_timeout_sec,
        "QWEN_ASR_CHUNK_SEC": chunk_sec,
        "QWEN_ASR_ENFORCE_EAGER": "1" if enforce_eager else "",
    }


def health_ok(url: str, timeout_sec: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_health(url: str, *, timeout_sec: float = 600.0, interval_sec: float = 2.0,
                on_progress: Callable[[str], None] | None = None,
                probe: Callable[[str], bool] = health_ok) -> None:
    """/health が 200 を返すまで待つ（初回はイメージとモデルの取得で数分かかる）。"""
    deadline = time.monotonic() + timeout_sec
    next_note = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if probe(url):
            return
        if on_progress and time.monotonic() >= next_note:
            on_progress(f"[Qwen3-ASR] {url} の準備を待っています…（初回は数分かかります）")
            next_note = time.monotonic() + 30.0
        time.sleep(interval_sec)
    raise RuntimeError(f"Qwen3-ASR が {int(timeout_sec)} 秒で準備できませんでした（{url}）")
