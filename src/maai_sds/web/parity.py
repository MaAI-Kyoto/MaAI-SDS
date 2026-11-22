"""Web UI の PyQt 同等操作向けヘルパ（パス砂場・MDF・OBS・LLM 管理・マイク）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .. import paths
from ..dialog.settings import LEGACY_TTS_ENGINE_ALIASES
from ..gui.settings import BOOL_KEYS, DEFAULT_FST_CONTENT, FIELD_SPECS, REMOVED_LEGACY_KEYS
from ..mmd.default_paths import fill_default_mmdagent_paths
from ..mmd.mdf_service import MdfService
from ..obs import remote as obs_remote


_UI_MANIFEST_PATH = Path(__file__).resolve().parent / "ui_manifest.json"
_APP_UI_DEFAULTS: dict[str, Any] | None = None

_AUDIO_OUTPUT_MODE_ALIASES = {
    "mmd": "mmd",
    "mmd (remote)": "mmd",
    "local": "local",
    "ローカルスピーカー": "local",
    "tcp": "tcp",
    "tcp 送出": "tcp",
}


def canonical_audio_output_mode(value: Any) -> str:
    raw = str(value or "mmd").strip()
    return _AUDIO_OUTPUT_MODE_ALIASES.get(raw.lower(), _AUDIO_OUTPUT_MODE_ALIASES.get(raw, raw.lower()))


def app_ui_defaults() -> dict[str, Any]:
    """Web 画面に出るウィジェット既定値（JSON にキーが無いときの表示と同じ）。"""
    global _APP_UI_DEFAULTS
    if _APP_UI_DEFAULTS is not None:
        return _APP_UI_DEFAULTS
    defaults: dict[str, Any] = {}
    try:
        data = json.loads(_UI_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        _APP_UI_DEFAULTS = {}
        return _APP_UI_DEFAULTS

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            key = obj.get("config_key")
            kind = obj.get("kind")
            if (
                key
                and obj.get("scope") != "mdf"
                and kind in {"line", "combo", "check", "textarea"}
                and "value" in obj
            ):
                defaults.setdefault(str(key), obj.get("value"))
            for val in obj.values():
                walk(val)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    _APP_UI_DEFAULTS = defaults
    return defaults


_COMBO_LABEL_VALUES: dict[str, dict[str, str]] | None = None


def combo_label_values() -> dict[str, dict[str, str]]:
    """{config_key: {表示名や大小違いの値: 保存値}}。表示名と保存値が別の combo だけ。"""
    global _COMBO_LABEL_VALUES
    if _COMBO_LABEL_VALUES is not None:
        return _COMBO_LABEL_VALUES
    table: dict[str, dict[str, str]] = {}
    try:
        data = json.loads(_UI_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            key = obj.get("config_key")
            items = obj.get("items")
            if key and obj.get("kind") == "combo" and isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    value = str(item.get("value", ""))
                    label = str(item.get("label", ""))
                    if value and label and label != value:
                        mapping = table.setdefault(str(key), {})
                        mapping.setdefault(label, value)
                        mapping.setdefault(value.lower(), value)
            for val in obj.values():
                walk(val)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    # Wording-only UI changes must not strand values that older Web builds
    # accidentally saved as their visible labels instead of itemData.
    legacy_labels = {
        "audio_output_mode": {
            "MMD (Remote)": "mmd",
            "TCP 送出": "tcp",
        },
        # 削除した TTS エンジンの旧名（dialog/settings.py の LEGACY_TTS_ENGINE_ALIASES）。
        "tts_engine": dict(LEGACY_TTS_ENGINE_ALIASES),
    }
    for key, aliases in legacy_labels.items():
        table.setdefault(key, {}).update(aliases)
    _COMBO_LABEL_VALUES = table
    return table


def canonical_combo_values(cfg: dict[str, Any]) -> dict[str, Any]:
    """表示名のまま保存された combo 値（"GPU (cuda)" 等）を Qt と同じ保存値へ直す。

    以前の Web UI は表示名を保存していた。その設定ファイルも読めるように、
    読み込み・保存のたびにここを通す（保存すればファイル自体も直る）。
    """
    out = dict(cfg)
    for key, mapping in combo_label_values().items():
        raw = out.get(key)
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        fixed = mapping.get(text) or mapping.get(text.lower())
        if fixed is not None and fixed != raw:
            out[key] = fixed
    return out


def merge_app_ui_defaults(cfg: dict[str, Any], *, fill_mmdagent_paths: bool = True) -> dict[str, Any]:
    """ファイルに無いキーだけ画面既定で埋める。保存値は上書きしない。

    fill_mmdagent_paths: Docker ではパスはコンテナ内のものなので False にする。
    """
    out = dict(app_ui_defaults())
    out.setdefault("audio_input_mic_device_index", "-1")
    out.setdefault("web_mic_source", "browser")
    out.setdefault("web_browser_mic_device_id", "")
    out.update(canonical_combo_values(cfg))
    out["audio_output_mode"] = canonical_audio_output_mode(out.get("audio_output_mode"))
    # Qt と同じく、EXE / MDF が未設定（または存在しない）なら既定の場所から探す。
    return fill_default_mmdagent_paths(out) if fill_mmdagent_paths else out


def customize_ui_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Add Web-only controls and canonical combo values to a fresh manifest.

    ``ui_manifest.json`` remains a generated Qt snapshot, so these Web-only
    adjustments deliberately happen at response time and do not make
    ``extract_web_ui_manifest.py --check`` platform-dependent.
    """

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            attr = obj.get("attr")
            if attr == "audio_output_mode_combo":
                obj["items"] = [
                    {"label": "MMD（Remote）", "value": "mmd"},
                    {"label": "ローカルスピーカー", "value": "local"},
                    {"label": "TCP送出", "value": "tcp"},
                ]
                obj["value"] = "mmd"
            elif attr == "audio_input_mic_device_combo":
                # Qt saves this combo outside APP_CONFIG_FIELDS, so the generated
                # manifest has config_key=null.  Web needs the numeric PortAudio
                # device id when local mic is selected.
                obj["config_key"] = "audio_input_mic_device_index"
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(manifest)
    windows = manifest.get("windows")
    if not isinstance(windows, list):
        return manifest
    audio_window = next(
        (window for window in windows if isinstance(window, dict) and window.get("id") == "audio_input_settings_window"),
        None,
    )
    if not isinstance(audio_window, dict):
        return manifest
    groups = audio_window.get("groups")
    if not isinstance(groups, list):
        return manifest
    if any(
        isinstance(group, dict) and group.get("title") == "Web UIマイク入力元"
        for group in groups
    ):
        return manifest
    groups.insert(
        1,
        {
            "title": "Web UIマイク入力元",
            "rows": [
                {
                    "label": "micモードの入力元",
                    "widgets": [
                        {
                            "attr": "web_mic_source_combo",
                            "config_key": "web_mic_source",
                            "scope": "app",
                            "tooltip": (
                                "ブラウザのマイク：このブラウザのマイクをWebSocketで送信します。"
                                "ローカルマイク：Webサーバーを実行しているWindowsのPortAudioデバイスを使います。"
                                "Dockerではブラウザのマイク固定です。"
                            ),
                            "kind": "combo",
                            "items": [
                                {"label": "ブラウザのマイク", "value": "browser"},
                                {"label": "WebサーバーPCのローカルマイク", "value": "local"},
                            ],
                            "value": "browser",
                        }
                    ],
                }
            ],
        },
    )
    return manifest


def mdf_defaults() -> dict[str, str]:
    defaults: dict[str, str] = {}
    for rows in FIELD_SPECS.values():
        for key, _label, default in rows:
            defaults[str(key)] = str(default)
    return defaults


def mdf_field_keys() -> list[str]:
    keys: list[str] = []
    for rows in FIELD_SPECS.values():
        for key, _label, _default in rows:
            keys.append(str(key))
    return keys


def mdf_service() -> MdfService:
    return MdfService(
        BOOL_KEYS,
        mdf_defaults(),
        REMOVED_LEGACY_KEYS,
        DEFAULT_FST_CONTENT,
    )


def browse_roots(extra: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    for raw in (
        paths.REPO_ROOT,
        paths.data_root(),
        paths.input_wav_dir(),
        paths.host_uploads_dir(),
        Path("/app"),
        Path("/app/mmdagent_assets"),
        Path("/app/input_wav"),
        Path("/host_repo"),
        Path("/app/voicevox_runtime"),
        Path("/app/maai_assets"),
        Path("/app/logs"),
        Path(r"C:\MMDAgent-EX\Release"),
        paths.assets_root(),
        paths.logs_dir(),
        extra,
    ):
        if raw is None:
            continue
        try:
            resolved = Path(raw).resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    for extra_root in (
        os.environ.get("MAAI_SDS_DATA_ROOT", "").strip(),
        os.environ.get("MAAI_SDS_MMD_ASSETS", "").strip(),
    ):
        if not extra_root:
            continue
        try:
            env_path = Path(extra_root).resolve()
        except OSError:
            continue
        if env_path.is_dir() and env_path not in roots:
            roots.append(env_path)
    return roots


def write_roots(extra: Path | None = None) -> list[Path]:
    """書き込みを許可するディレクトリ。`browse_roots`（読み取り用）より意図的に狭い。

    読み取りはリポジトリ全体を見せてよいが、書き込みまで同じ広さにすると
    `PUT /api/file` で `src/maai_sds/**.py` を上書きでき、次回起動で任意コードが動く。
    Docker では `.:/host_repo` が rw マウント（モデル取得に書き込みが要るので ro に
    できない）なので、ホスト側のソースまで書き換えられる状態だった。

    API が実際に書く必要があるのは MDF/FST（MMD アセット配下）とアップロード先だけ。
    `tts_template.json` だけは REPO_ROOT 直下にあるので、ディレクトリではなく
    ファイル単位で `writable_files()` 側に置く。
    """
    roots: list[Path] = []
    for raw in (
        paths.input_wav_dir(),
        paths.host_uploads_dir(),
        extra,
        os.environ.get("MAAI_SDS_MMD_ASSETS", "").strip() or None,
    ):
        if raw is None:
            continue
        try:
            resolved = Path(raw).resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def writable_files() -> set[Path]:
    """ディレクトリ単位では許可できない、書き込み対象の個別ファイル。

    `tts_template.json` は REPO_ROOT 直下なので、ディレクトリごと許可すると
    リポジトリ全体が書き込み可能になってしまう。UI から `PUT /api/file` で
    書かれるのはこの 1 ファイルだけ（web/app/app.js の TTS テンプレート保存）。
    """
    try:
        return {paths.tts_template_path().resolve()}
    except OSError:
        return set()


def resolve_sandboxed(path: str, extra: Path | None = None, *, for_write: bool = False) -> Path:
    raw = (path or "").strip() or "."
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = paths.REPO_ROOT / candidate
    try:
        resolved = candidate.resolve()
    except OSError as ex:
        raise ValueError(f"invalid path: {ex}") from ex
    if for_write and resolved in writable_files():
        return resolved
    for root in (write_roots(extra) if for_write else browse_roots(extra)):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        "path is outside writable directories" if for_write else "path is outside allowed directories"
    )


def list_dir(path: str, extra: Path | None = None, *, for_write: bool = False) -> dict[str, Any]:
    """`for_write` のときは書き込み可能なルートだけを返す。

    読み取り用の広いルートをそのまま保存ダイアログに出すと、選べるのに保存だけ
    400 で弾かれる場所（リポジトリ直下や logs）が見えてしまう。
    """
    target = resolve_sandboxed(path, extra, for_write=for_write)
    if not target.is_dir():
        raise FileNotFoundError(str(target))
    entries = []
    try:
        children = list(target.iterdir())
    except OSError as ex:
        raise OSError(str(ex)) from ex
    for entry in sorted(children, key=lambda item: (not item.is_dir(), item.name.lower())):
        entries.append({"name": entry.name, "is_dir": entry.is_dir()})
    return {
        "path": str(target),
        "entries": entries,
        "roots": [str(r) for r in (write_roots(extra) if for_write else browse_roots(extra))],
    }


def read_text_file(path: str, extra: Path | None = None) -> str:
    target = resolve_sandboxed(path, extra)
    if not target.is_file():
        raise FileNotFoundError(str(target))
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return target.read_text(encoding="utf-8", errors="replace")


def write_text_file(path: str, content: str, extra: Path | None = None) -> None:
    target = resolve_sandboxed(path, extra, for_write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def load_mdf(mdf_path: Path) -> dict[str, str]:
    defaults = mdf_defaults()
    if not mdf_path.is_file():
        return dict(defaults)
    lines, _enc = mdf_service().read_mdf(mdf_path)
    parsed = MdfService.parse_mdf_key_values(lines)
    return {**defaults, **parsed}


def save_mdf(
    mdf_path: Path,
    values: dict[str, Any],
    *,
    create: bool = False,
    touched: Iterable[str] | None = None,
) -> dict[str, str]:
    """MDF を書き戻す。

    `touched` は **利用者が実際に編集したキー** だけを渡すこと（Web UI の
    `state.mdfDirty`）。ここを `values` の全キーにすると、UI が既定値込みの全 99 キーを
    送ってくる作りなので `FIELD_SPECS` の既定値が丸ごと明示行として書き出され、
    3 行の .mdf が 165 行に膨張する（実測）。本家 MMDAgent-EX の内部既定ではなく
    このリポジトリの既定で固定されてしまうため、既定のままのキーは書かないのが正しい。
    PyQt 版（gui/controller.py の MDF 保存）も loaded_keys / touched_keys を
    同じ意味で使い分けている。

    `touched=None` は「編集なし」= もともとファイルにあったキーだけを更新する。
    """
    if create and not mdf_path.exists():
        mdf_path.parent.mkdir(parents=True, exist_ok=True)
        mdf_path.write_text("", encoding="utf-8")
    if not mdf_path.is_file():
        raise FileNotFoundError(str(mdf_path))
    # MDF は 1 行 1 `KEY=value` なので、値に改行が入ると任意の行を差し込めてしまう
    # （MdfService.set_mdf_value は f"{key}={value}" をそのまま 1 行として書く）。
    # 改行を含む値は MDF として表現できないので、ここで弾く。
    for key, value in values.items():
        if value is not None and ("\n" in str(value) or "\r" in str(value)):
            raise ValueError(f"MDF の値に改行は使えません: {key}")
    normalized = {str(k): "" if v is None else str(v) for k, v in values.items()}
    keys = mdf_field_keys()
    touched_keys = {str(k) for k in (touched or ())} & set(normalized)
    loaded = set(MdfService.parse_mdf_key_values(mdf_service().read_mdf(mdf_path)[0]).keys())

    def widget_to_value(key: str) -> str:
        if key in normalized:
            return normalized[key]
        return mdf_defaults().get(key, "")

    ok, _loaded = mdf_service().write_values(
        mdf_path=mdf_path,
        field_keys=keys,
        widget_to_value=widget_to_value,
        loaded_keys=loaded | touched_keys,
        touched_keys=touched_keys,
    )
    if not ok:
        raise OSError(f"failed to write {mdf_path}")
    return load_mdf(mdf_path)


def obs_connection(cfg: dict[str, Any]) -> tuple[str, int, str]:
    host = str(cfg.get("obs_remote_host") or "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(str(cfg.get("obs_remote_port") or "4455").strip() or "4455")
    except ValueError:
        port = 4455
    password = str(cfg.get("obs_remote_password") or "")
    return host, port, password


def obs_test(cfg: dict[str, Any]) -> tuple[bool, str]:
    host, port, password = obs_connection(cfg)
    ok, err = obs_remote.test_connection(host, port, password)
    return ok, err or ""


def obs_start(cfg: dict[str, Any]) -> tuple[bool, str]:
    host, port, password = obs_connection(cfg)
    ok, err = obs_remote.start_record(host, port, password)
    return ok, err or ""


def obs_stop(cfg: dict[str, Any]) -> tuple[bool, str]:
    host, port, password = obs_connection(cfg)
    ok, err = obs_remote.stop_record(host, port, password)
    return ok, err or ""


def _ensure_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL host is empty")
    return url


def fetch_lmstudio_models(host: str) -> tuple[list[str], str]:
    from ..llm.common import normalize_lmstudio_base_url

    base = normalize_lmstudio_base_url(host or "http://127.0.0.1:1234")
    url = f"{base}/v1/models"
    names: list[str] = []
    try:
        _ensure_http_url(url)
        with urlopen(url, timeout=10.0) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        for item in data.get("data") or []:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if isinstance(mid, str) and mid.strip():
                names.append(mid.strip())
        return names, ""
    except Exception as ex:
        return [], str(ex)


def unload_lmstudio_others(host: str, current_model: str) -> dict[str, Any]:
    from ..llm.admin_http import _list_lmstudio_loaded_instances, _unload_lmstudio_model
    from ..llm.common import normalize_lmstudio_base_url

    host = normalize_lmstudio_base_url(host or "http://127.0.0.1:1234")
    loaded, list_error = _list_lmstudio_loaded_instances(host)
    if list_error:
        return {"ok": False, "error": f"モデル一覧取得失敗: {list_error}", "attempted": 0, "success_count": 0}
    current = (current_model or "").strip()
    targets = [m for m in loaded if m != current]
    success_count = 0
    failed: list[dict[str, str]] = []
    for model in targets:
        ok, err = _unload_lmstudio_model(host, model)
        if ok:
            success_count += 1
        else:
            failed.append({"model": model, "error": err})
    return {
        "ok": len(failed) == 0,
        "error": "",
        "attempted": len(targets),
        "success_count": success_count,
        "failed": failed,
    }


def fetch_openai_models(api_key: str) -> tuple[list[str], str]:
    names: list[str] = []
    try:
        from openai import OpenAI

        if not (api_key or "").strip():
            raise RuntimeError("OpenAI APIキーが空です。")
        client = OpenAI(api_key=api_key.strip())
        models = client.models.list()
        data = getattr(models, "data", None) or []
        for item in data:
            mid = getattr(item, "id", None)
            if isinstance(mid, str) and mid.strip():
                names.append(mid.strip())
        return names, ""
    except Exception as ex:
        return [], str(ex)


def list_audio_input_devices() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"id": -1, "label": "default"}]
    try:
        import sounddevice as sd
    except Exception:
        return out
    try:
        devs = sd.query_devices()
    except Exception:
        return out
    for idx, dev in enumerate(devs):
        if int(dev.get("max_input_channels", 0) or 0) <= 0:
            continue
        name = str(dev.get("name", f"device-{idx}"))
        out.append({"id": int(idx), "label": f"{idx}: {name}"})
    return out
