"""LLM 管理向け HTTP 操作（LM Studio / OpenAI のモデル一覧・アンロード）。

`OrchestratorSupervisor` から委譲される。挙動・送信イベント (`type` /
`kind` / フィールド) は移管前と完全互換。
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .common import normalize_lmstudio_base_url as _shared_normalize_lmstudio_base_url
from .common import resolve_openai_api_key as _shared_resolve_openai_api_key


def _ensure_http_url(url: str) -> str:
    """http/https 以外のスキーム（file:// 等）を拒否して SSRF / ローカルファイル
    読み出しを防ぐ。設定由来の host をそのまま urlopen に渡す前に必ず通すこと。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r} (http/https only)")
    if not parsed.hostname:
        raise ValueError("URL host is empty")
    return url


def _normalize_lmstudio_base_url(host: str) -> str:
    """`llm/common` に集約（Phase 4）。3 モジュールに同一コピーが 3 個あった。"""
    return _shared_normalize_lmstudio_base_url(host)


def _list_lmstudio_loaded_instances(host: str, timeout_sec: float = 10.0) -> tuple[list[str], str]:
    base = _normalize_lmstudio_base_url(host)
    url = f"{base}/api/v1/models"
    try:
        _ensure_http_url(url)
        with urlopen(url, timeout=max(0.1, float(timeout_sec))) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        models = data.get("models") or []
        instances: list[str] = []
        seen: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            loaded = model.get("loaded_instances") or []
            if not isinstance(loaded, list):
                continue
            for inst in loaded:
                if not isinstance(inst, dict):
                    continue
                inst_id = inst.get("id")
                if isinstance(inst_id, str):
                    text = inst_id.strip()
                    if text and text not in seen:
                        seen.add(text)
                        instances.append(text)
        return instances, ""
    except Exception as ex:
        return [], f"{url}: {ex}"


def _unload_lmstudio_model(host: str, model: str, timeout_sec: float = 10.0) -> tuple[bool, str]:
    base = _normalize_lmstudio_base_url(host)
    url = f"{base}/api/v1/models/unload"
    payloads = [
        {"instance_id": model},
        {"model": model},
    ]
    last_error = ""
    for body in payloads:
        try:
            _ensure_http_url(url)
            payload = json.dumps(body).encode("utf-8")
            req = Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=max(0.1, float(timeout_sec))) as resp:
                _ = resp.read()
            return True, ""
        except Exception as ex:
            last_error = f"{url} body={body}: {ex}"
    return False, last_error


def _probe_tcp(host: str, *, timeout_sec: float) -> tuple[bool, str]:
    """`host` の TCP ポートが開いているか。開いていなければ理由を返す。"""
    base = _normalize_lmstudio_base_url(host)
    parsed = urlparse(base)
    if not parsed.hostname:
        return False, f"{host}: URL からホストを取れない"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_sec):
            return True, ""
    except Exception as ex:  # noqa: BLE001
        return False, f"{parsed.hostname}:{port} に接続できない: {ex}"


def unload_all_loaded_models(
    host: str, *, budget_sec: float = 4.0
) -> tuple[list[str], list[str], str]:
    """LM Studio 上のロード済みインスタンスを全部アンロードする。

    **worker / IPC を一切経由しない同期版。**アプリ終了時に呼ぶためで、
    そのタイミングでは Orchestrator ワーカーがもう止まっている
    （`LlmAdminHttp.handle` 経由の `lmstudio_bulk_unload` はそこを通るので使えない）。

    `budget_sec` は全体の上限。LM Studio が起動していない・`host` が遠い場合に
    GUI の終了が止まらないよう、超えたら残りを諦める。例外は投げない。

    戻り値は `(アンロードしたもの, 未処理のもの, エラー文)`。
    """
    deadline = time.monotonic() + max(0.1, float(budget_sec))

    def _left() -> float:
        return deadline - time.monotonic()

    # HTTP の前に TCP で当たりを取る。Windows は**閉じたポートへの接続が
    # 「拒否」ではなくタイムアウトする**ので、LM Studio 未起動のときに
    # `urlopen` の timeout を丸々待ってしまう（実測 2.00s）。
    # プローブなら不在は約 0.5 秒で分かり、アプリの終了が待たされない。
    reachable, probe_err = _probe_tcp(host, timeout_sec=min(0.5, max(0.05, _left())))
    if not reachable:
        return [], [], probe_err

    loaded, err = _list_lmstudio_loaded_instances(host, timeout_sec=min(2.0, max(0.1, _left())))
    if err:
        return [], [], err
    done: list[str] = []
    skipped: list[str] = []
    for model in loaded:
        if _left() <= 0.0:
            skipped.append(model)
            continue
        ok, unload_err = _unload_lmstudio_model(
            host, model, timeout_sec=min(2.0, max(0.1, _left()))
        )
        if ok:
            done.append(model)
        else:
            skipped.append(model)
            err = err or unload_err
    return done, skipped, err


class LlmAdminHttp:
    """LM Studio / OpenAI 関連の admin HTTP コマンドを処理する。

    `on_event` には `OrchestratorSupervisor` の `_emit_llm_admin_event` 経由で
    Controller に転送するためのコールバックを渡す。
    """

    def __init__(self, on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._on_event = on_event

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._on_event:
            try:
                self._on_event(payload)
            except Exception:
                pass

    def handle(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        op = str(payload.get("op", "")).strip().lower()
        rid = str(payload.get("request_id", "") or "").strip()
        if not rid:
            return
        if op == "lmstudio_fetch_models":
            threading.Thread(
                target=self._lmstudio_fetch_models_thread,
                args=(rid, dict(payload)),
                daemon=True,
                name="llm-admin-lmstudio-fetch",
            ).start()
            return
        if op == "lmstudio_bulk_unload":
            threading.Thread(
                target=self._lmstudio_bulk_unload_thread,
                args=(rid, dict(payload)),
                daemon=True,
                name="llm-admin-lmstudio-unload",
            ).start()
            return
        if op == "openai_fetch_models":
            threading.Thread(
                target=self._openai_fetch_models_thread,
                args=(rid, dict(payload)),
                daemon=True,
                name="llm-admin-openai-fetch",
            ).start()
            return

    def _lmstudio_fetch_models_thread(self, request_id: str, payload: dict[str, Any]) -> None:
        host = str(payload.get("host", "") or "").strip() or "http://127.0.0.1:1234"
        silent = bool(payload.get("silent", False))
        base = host.rstrip("/")
        url = f"{base}/v1/models"
        names: list[str] = []
        error = ""
        try:
            _ensure_http_url(url)
            with urlopen(url, timeout=10.0) as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8", errors="replace"))
            items = data.get("data") or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                mid = item.get("id")
                if isinstance(mid, str):
                    m = mid.strip()
                    if m:
                        names.append(m)
        except Exception as ex:
            error = str(ex)
        self._emit(
            {
                "type": "llm_admin_result",
                "request_id": request_id,
                "kind": "lmstudio_fetch",
                "host": host,
                "models": names,
                "error": error,
                "silent": silent,
            }
        )

    def _lmstudio_bulk_unload_thread(self, request_id: str, payload: dict[str, Any]) -> None:
        host = str(payload.get("host", "") or "").strip() or "http://127.0.0.1:1234"
        current_model = str(payload.get("current_model", "") or "").strip()
        unload_all = bool(payload.get("unload_all", False))
        silent = bool(payload.get("silent", False))
        reason = str(payload.get("reason", "") or "")
        loaded, list_error = _list_lmstudio_loaded_instances(host)
        if list_error:
            self._emit(
                {
                    "type": "llm_admin_result",
                    "request_id": request_id,
                    "kind": "lmstudio_unload",
                    "ok": False,
                    "host": host,
                    "reason": reason,
                    "silent": silent,
                    "error": f"モデル一覧取得失敗: {list_error}",
                    "attempted": 0,
                    "success_count": 0,
                    "failed": [],
                }
            )
            return
        targets = loaded if unload_all else [m for m in loaded if m != current_model]
        success_count = 0
        failed: list[dict[str, str]] = []
        for model in targets:
            ok, err = _unload_lmstudio_model(host, model)
            if ok:
                success_count += 1
            else:
                failed.append({"model": model, "error": err})
        self._emit(
            {
                "type": "llm_admin_result",
                "request_id": request_id,
                "kind": "lmstudio_unload",
                "ok": len(failed) == 0,
                "host": host,
                "reason": reason,
                "silent": silent,
                "error": "",
                "attempted": len(targets),
                "success_count": success_count,
                "failed": failed,
            }
        )

    def _openai_fetch_models_thread(self, request_id: str, payload: dict[str, Any]) -> None:
        api_key = str(payload.get("api_key", "") or "").strip()
        silent = bool(payload.get("silent", False))
        names: list[str] = []
        error = ""
        try:
            from openai import OpenAI  # noqa: PLC0415

            if not api_key:
                raise RuntimeError("OpenAI API Key が空です。")
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            data = getattr(models, "data", None) or []
            for item in data:
                mid = getattr(item, "id", None)
                if isinstance(mid, str):
                    text = mid.strip()
                    if text:
                        names.append(text)
        except Exception as ex:
            error = str(ex)
        self._emit(
            {
                "type": "llm_admin_result",
                "request_id": request_id,
                "kind": "openai_fetch",
                "models": names,
                "error": error,
                "silent": silent,
            }
        )
