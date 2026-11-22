"""LLM 接続まわりの共通ヘルパ（Phase 4: 重複の集約）。

`normalize_lmstudio_base_url` は `llm/admin_http.py` / `llm/service.py` に
**本文が完全に同一のコピーが 3 個**あった。`resolve_openai_api_key` も 2 個あった。
3 モジュールは互いに import 関係が無く、片方だけ直すと食い違うため、ここへ集約する。
"""

from __future__ import annotations

import os


def running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def normalize_docker_loopback_host(host: str) -> str:
    """プレーンなホスト名の 127.0.0.1 / localhost を Docker では host.docker.internal にする。"""
    text = (host or "").strip()
    if not running_in_docker():
        return text
    if text.lower() in {"127.0.0.1", "localhost", "::1"}:
        return "host.docker.internal"
    return text


def normalize_lmstudio_base_url(host: str) -> str:
    """LM Studio / OpenAI 互換ゲートウェイの base URL を正規化する。

    - 空なら `http://127.0.0.1:1234`
    - スキームが無ければ `http://` を付ける
    - 末尾の `/`、`/v1`、`/api/v1` を落とす（呼び出し側が `/v1` を足すため）
    - Docker 内の 127.0.0.1 / localhost はホスト側へ届かないので
      `host.docker.internal` に付け替える
    """
    base = host.strip() or "http://127.0.0.1:1234"
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if base.endswith("/api/v1"):
        base = base[:-7]
    return _rewrite_loopback_for_docker(base)


def _rewrite_loopback_for_docker(base: str) -> str:
    if not running_in_docker():
        return base
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(base)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        return base
    host = "host.docker.internal"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    return urlunparse(parsed._replace(netloc=f"{userinfo}{netloc}"))


def resolve_openai_api_key(api_key: str, api_key_env: str) -> str:
    """API キーを解決する。直接指定が空なら環境変数名から読む。"""
    resolved_key = (api_key or "").strip()
    if not resolved_key and (api_key_env or "").strip():
        resolved_key = os.environ.get(api_key_env.strip(), "").strip()
    return resolved_key
