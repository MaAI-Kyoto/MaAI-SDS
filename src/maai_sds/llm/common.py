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


# LM Studio が「モデルが無い・読み込まれていない」ときに返すエラーの言い回し
_LMSTUDIO_MODEL_ERRORS = ("No models loaded", "model_not_found", "Model not found", "not found", "does not exist")


def lmstudio_model_keys(base_url: str, timeout_sec: float = 3.0) -> list[str] | None:
    """LM Studio に入っている（ダウンロード済みの）LLM のキーの一覧。取れなければ None。"""
    import json
    from urllib.request import urlopen

    base = normalize_lmstudio_base_url(base_url)
    try:
        with urlopen(f"{base}/api/v1/models", timeout=max(0.1, float(timeout_sec))) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - 取れなければ知らせを足さないだけ
        return None
    return [
        str(m.get("key"))
        for m in (data.get("models") or [])
        if isinstance(m, dict) and m.get("key") and m.get("type", "llm") == "llm"
    ]


def _same_lmstudio_model(requested: str, key: str) -> bool:
    """`gemma@q4_k_m` と `gemma` のように、量子化の指定（@ 以降）の有無が違っても同じモデルとみなす。"""
    a, b = requested.strip().lower(), key.strip().lower()
    return a == b or a.split("@", 1)[0] == b or b.split("@", 1)[0] == a


def lmstudio_model_error_hint(base_url: str, model: str, error: object, keys: list[str] | None = None) -> str | None:
    """LM Studio の「No models loaded」などのエラーを、何をすればよいかが分かる文にする（当てはまらなければ None）。

    LM Studio は、頼まれたモデルが入っていれば自動で読み込む（Just-in-Time）。入っていないモデルを頼むと
    「No models loaded」と返すので、それだけでは「取っていない」ことが分からない。
    """
    message = str(error)
    if not any(s in message for s in _LMSTUDIO_MODEL_ERRORS):
        return None
    if keys is None:
        keys = lmstudio_model_keys(base_url)
    if keys is None:
        return None
    model = (model or "").strip()
    if not model:
        return "LLM のモデルが指定されていません。「LLM（応答生成）」の画面で、LM Studio のモデルを選んでください。"
    if not any(_same_lmstudio_model(model, k) for k in keys):
        return (
            f"LM Studio に {model} が入っていません。`lms get {model.split('@', 1)[0]}` で取得するか、"
            "「LLM（応答生成）」の画面で、入っているモデルを選び直してください"
            "（LM Studio は入っているモデルだけを自動で読み込みます）。"
        )
    if "No models loaded" in message:
        return (
            f"LM Studio に {model} は入っていますが、読み込まれていません。LM Studio の Developer の設定で"
            "「Just-in-Time Model Loading」を有効にするか、LM Studio でモデルを読み込んでください。"
        )
    return None
