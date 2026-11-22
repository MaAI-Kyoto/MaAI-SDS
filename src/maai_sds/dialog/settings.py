"""controller_config.json から start_voice_dialog ペイロードを組み立てる純関数モジュール。

GUI/Qt や `controller.py` には依存しない。
- ヘッドレス CLI（`headless_main.py`）と GUI（`controller.py`）の両方が共有する。
- 入力は `ConfigStore.load()` の戻り値（`controller_config.json` の dict）。`mdf_path` が有効なら
  `build_audio_output_worker_config_dict` 内で同 MDF の `Plugin_Remote_*` をマージする（GUI と整合）。
- 出力は `dialog/worker` が `start_voice_dialog` で受ける settings dict。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .. import paths as _paths
from ..llm.common import normalize_docker_loopback_host

_DEFAULT_VOICEVOX_CORE_DEVICE = "CPU" if os.name != "nt" else "GPU"




def _is_cpu_build() -> bool:
    """この環境が CPU 構成か（環境構築時に選んだ .device_extra を見る）。"""
    try:
        from .. import engine_manager
    except Exception:  # noqa: BLE001
        return False
    return engine_manager.device_extra() == "cpu"


def _compute_type(value: str, *, label: str) -> str:
    """CPU 構成では GPU 前提の compute type を int8 に落とす。

    faster-whisper（CTranslate2）の float16 / int8_float16 は GPU 用で、CPU では
    使えない。デバイスだけ cpu に落としても compute type が float16 のままだと
    そこで失敗するので、対にして扱う。
    """
    text = str(value or "").strip()
    if "float16" not in text.lower() and "float32" not in text.lower():
        return text
    if not _is_cpu_build():
        return text
    print(f"[device] {label}: CPU 構成のため compute type を int8 にします（設定値 {text!r}）。",
          flush=True)
    return "int8"

def _device(value: str, *, label: str) -> str:
    """CPU 構成では CUDA 要求を cpu へ落とす。

    デバイス指定は controller_config に入っており、GPU 機で書かれた設定を CPU 機で
    そのまま読むと、CUDA を要求したエンジンが例外で落ちる（フォールバックしない）:

        RuntimeError: OrtSessionOptionsAppendExecutionProvider_Cuda:
                      Failed to load shared library

    設定ファイルを GPU 用 / CPU 用に分けると二重管理になるので、環境構築時に決めた
    `.device_extra`（engine_manager.device_extra）が cpu のときだけここで落とす。
    実行時に CUDA の可否を探る方式は採らない — ONNX Runtime の CUDA セッションを
    作ると後続の torch.cuda.is_available() が False を返すようになり、どのエンジンが
    先に聞いたかで答えが変わる（実機で確認済み）。
    """
    text = str(value or "").strip()
    if not text.lower().startswith(("cuda", "gpu")):
        return text
    if not _is_cpu_build():
        return text
    print(f"[device] {label}: CPU 構成のため cpu で実行します（設定値 {text!r}）。", flush=True)
    return "cpu"

def _venv_python_setting(cfg: dict, key: str, venv_dir: str) -> str:
    return str(_paths.resolve_venv_python(cfg_str(cfg, key, ""), venv_dir))


def _loopback_host(cfg: dict, key: str, default: str) -> str:
    return normalize_docker_loopback_host(cfg_str(cfg, key, default) or default)


_DEFAULT_VOICEVOX_ORT = (
    "voicevox_runtime/onnxruntime/lib/voicevox_onnxruntime.dll"
    if os.name == "nt"
    else "voicevox_runtime/onnxruntime/lib/libvoicevox_onnxruntime.so"
)


# ----- 基本ヘルパ ----------------------------------------------------------------

def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def cfg_str(cfg: dict, key: str, default: str = "") -> str:
    v = cfg.get(key, None)
    if v is None:
        return default
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip()


# 削除した TTS エンジンの旧名 → 置き換え先。保存済みの設定がそのまま動くようにする。
LEGACY_TTS_ENGINE_ALIASES: dict[str, str] = {
    # PyTorch 版 Qwen3-TTS（専用 venv・1 文 8〜9 秒）は削除した。audio.cpp 版へ移す。
    "Qwen3-TTS": "Qwen3-TTS-CPP",
}


def normalize_tts_engine(name: Any) -> str:
    """TTS エンジン名を前後空白を落として返す。削除済みエンジンの旧名は置き換え先にする。"""
    text = str(name or "").strip()
    return LEGACY_TTS_ENGINE_ALIASES.get(text, text)


def cfg_tts_engine(cfg: dict, default: str = "VOICEVOX") -> str:
    """cfg の `tts_engine`（空なら default）を `normalize_tts_engine` に通して返す。"""
    return normalize_tts_engine(cfg_str(cfg, "tts_engine", default) or default)


def cfg_int(cfg: dict, key: str, default: int) -> int:
    v = cfg.get(key, None)
    if _is_missing(v):
        return default
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def cfg_float(cfg: dict, key: str, default: float) -> float:
    v = cfg.get(key, None)
    if _is_missing(v):
        return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def cfg_bool(cfg: dict, key: str, default: bool) -> bool:
    v = cfg.get(key, None)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _cfg_blink_morph(cfg: dict) -> str:
    from ..mmd.blink import sanitize_blink_morph

    return sanitize_blink_morph(cfg_str(cfg, "mmd_blink_morph", "まばたき"))


def _cfg_blink_mode(cfg: dict) -> str:
    from ..mmd.blink import normalize_blink_mode

    return normalize_blink_mode(cfg.get("mmd_blink_mode"))


def _cfg_blink_interval_mean_sec(cfg: dict) -> float:
    from ..mmd.blink import DEFAULT_BLINK_INTERVAL_MEAN_SEC, normalize_blink_interval_mean_sec

    return normalize_blink_interval_mean_sec(
        cfg_float(cfg, "mmd_blink_interval_mean_sec", DEFAULT_BLINK_INTERVAL_MEAN_SEC)
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ----- LLM 関連 -------------------------------------------------------------------

def resolve_llm_split_settings(cfg: dict) -> dict[str, object]:
    custom_delimiters = cfg_str(cfg, "llm_split_custom_delimiters", "")
    if not custom_delimiters:
        custom_delimiters = "。．！？!?\\n"
    custom_delimiters = custom_delimiters.replace("\\n", "\n").replace("\\t", "\t")
    return {
        "custom_split_mode": "punctuation",
        "custom_delimiters": custom_delimiters,
        "custom_strip_emoji": cfg_bool(cfg, "llm_split_custom_strip_emoji", False),
        "custom_strip_parenthetical": cfg_bool(cfg, "llm_split_custom_strip_parenthetical", False),
        "custom_strip_standalone_warai": cfg_bool(cfg, "llm_split_custom_strip_standalone_warai", False),
    }


def _normalize_compat_llm_host(host: str) -> str:
    """LM Studio / Custom の OpenAI 互換 URL を正規化する。

    Docker 内の 127.0.0.1 / localhost はコンテナ自身なので
    host.docker.internal へ付け替える。
    """
    from ..llm.common import normalize_lmstudio_base_url

    return normalize_lmstudio_base_url(host)


def active_llm_connection_settings(cfg: dict) -> dict[str, str]:
    """`controller.py:_active_llm_connection_settings` 相当。"""
    backend = cfg_str(cfg, "llm_backend", "LM Studio") or "LM Studio"
    if backend == "Custom":
        custom_host = cfg_str(cfg, "llm_custom_base_url", "")
        return {
            "host": _normalize_compat_llm_host(custom_host) if custom_host else "",
            "api_key": cfg_str(cfg, "llm_custom_api_key", ""),
            "api_key_env": cfg_str(cfg, "llm_custom_api_key_env", "CUSTOM_LLM_API_KEY"),
        }
    if backend == "OpenAI":
        return {
            "host": "",
            "api_key": cfg_str(cfg, "llm_api_key", ""),
            "api_key_env": cfg_str(cfg, "llm_api_key_env", "OPENAI_API_KEY"),
        }
    return {
        "host": _normalize_compat_llm_host(cfg_str(cfg, "llm_host", "http://127.0.0.1:1234")),
        "api_key": "",
        "api_key_env": "",
    }


def llm_model_for_backend(cfg: dict, backend: str) -> str:
    """`controller.py` の `_llm_model_by_backend` に近い動き。"""
    bk = (backend or "").strip() or "LM Studio"
    if bk == "OpenAI":
        return cfg_str(cfg, "llm_model_openai", "")
    if bk == "Custom":
        return cfg_str(cfg, "llm_model_custom", "")
    return cfg_str(cfg, "llm_model_lmstudio", cfg_str(cfg, "llm_model", ""))


# ----- TTS inputs ---------------------------------------------------------------

_TTS_VOICE_CONFIG_KEYS: dict[str, str] = {
    "VOICEVOX": "tts_voice_voicevox",
    "VOICEVOX-Core": "tts_voice_voicevox_core",
    "Kokoro-RealtimeTTS": "tts_voice_kokoro_realtimetts",
    "Cartesia-Realtime": "tts_voice_cartesia",
    "TEMPLATE": "tts_voice_template",
    "AivisSpeech": "tts_voice_aivisspeech",
    "Supertonic-3": "tts_voice_supertonic",
    "Irodori-v4": "tts_voice_irodori",
    "Qwen3-TTS-CPP": "qwen3_tts_cpp_speaker",
}

# 共通の話者欄（tts_voice）の値を、設定画面が保存時に書き写すエンジン別キー。
# Qt の `_save_app_config` と同じ 8 エンジン（Qwen3-TTS-CPP は専用欄なので含めない）。
TTS_VOICE_CONFIG_KEYS_SAVED_BY_UI: dict[str, str] = {
    engine: key for engine, key in _TTS_VOICE_CONFIG_KEYS.items() if engine != "Qwen3-TTS-CPP"
}


def _looks_like_voice_id(value: str) -> bool:
    """話者 ID（VOICEVOX などの数字）や Supertonic のプリセット（F1〜M5）か。"""
    text = value.strip().upper()
    return text.isdigit() or (len(text) == 2 and text[0] in "FM" and text[1] in "12345")


def _tts_voice_for_engine(cfg: dict, engine: str, default: str = "") -> str:
    """Return the persisted per-engine voice instead of another engine's active value."""
    key = _TTS_VOICE_CONFIG_KEYS.get(engine, "")
    if key:
        voice = cfg_str(cfg, key, "")
        if voice:
            return voice
    fallback = cfg_str(cfg, "tts_voice", "")
    # Irodori の声は文章（キャプション）。共通の欄に別のエンジンの話者 ID（例の設定では VOICEVOX の "8"）が
    # 入っていると、それがそのままキャプションとして送られてしまうので使わない。
    if engine == "Irodori-v4" and _looks_like_voice_id(fallback):
        fallback = ""
    return fallback or default


def build_tts_inputs(
    cfg: dict,
    *,
    engine: str,
    utterance: str = "",
    input_streaming_enabled: bool = False,
    split_settings: dict[str, object] | None = None,
) -> dict[str, object]:
    """`controller.py:_build_tts_inputs` 相当。"""
    engine = normalize_tts_engine(engine)
    host = _loopback_host(cfg, "tts_host", "127.0.0.1") or "127.0.0.1"
    if engine in ("Kokoro-RealtimeTTS", "Cartesia-Realtime", "VOICEVOX-Core", "AivisSpeech", "Supertonic-3", "Qwen3-TTS-CPP", "LFM2.5-Audio", "Irodori-v4", "Tsukasa"):
        port = 0
        base_path = ""
    else:
        port_str = cfg_str(cfg, "tts_port", "")
        if engine == "VOICEVOX" and not port_str:
            port_str = "50021"
        try:
            port = int(port_str) if port_str else 0
        except (TypeError, ValueError):
            port = 0
        base_path = cfg_str(cfg, "tts_base_path", "")
    voice = _tts_voice_for_engine(cfg, engine)
    if not voice:
        voice = {
            "VOICEVOX": "1",
            "VOICEVOX-Core": "8",
            "Kokoro-RealtimeTTS": "jf_alpha",
            "Qwen3-TTS-CPP": "Vivian",
            "Cartesia-Realtime": "f786b574-daa5-4673-aa0c-cbe3e8534c02",
            "AivisSpeech": "888753761",
            "Supertonic-3": "F1",
            "Irodori-v4": "明るく自然な日本語の声",
        }.get(engine, "kyoto_shiori")
    model_value = cfg_str(cfg, "tts_model", "")
    if engine == "TEMPLATE":
        model_value = ""
    tok = dict(split_settings) if isinstance(split_settings, dict) else resolve_llm_split_settings(cfg)
    inputs: dict[str, object] = {
        "engine": engine,
        "host": host,
        "port": port,
        "base_path": base_path,
        "voice": voice,
        "model": model_value,
        "language": (
            cfg_str(cfg, "qwen3_tts_cpp_language", "japanese") or "japanese"
            if engine == "Qwen3-TTS-CPP"
            else "ja"
        ),
        "text": utterance,
        "auth_mode": cfg_str(cfg, "tts_auth_mode", "none") or "none",
        "auth_env": cfg_str(cfg, "tts_auth_env", ""),
        "auth_header": cfg_str(cfg, "tts_auth_header", "X-API-Key") or "X-API-Key",
        "auth_prefix": cfg_str(cfg, "tts_auth_prefix", "Bearer ") or "Bearer ",
        "custom_delimiters": tok["custom_delimiters"],
        "custom_strip_emoji": tok["custom_strip_emoji"],
        "custom_strip_parenthetical": tok["custom_strip_parenthetical"],
        "custom_strip_standalone_warai": tok["custom_strip_standalone_warai"],
        "input_streaming_enabled": bool(input_streaming_enabled),
        "segment_gap_silence_sec": _clamp(cfg_float(cfg, "tts_segment_gap_silence_sec", 0.0), 0.0, 10.0),
    }
    if engine == "Cartesia-Realtime":
        inputs.update(
            {
                "model_id": cfg_str(cfg, "cartesia_tts_model_id", "sonic-3") or "sonic-3",
                "voice_id": voice,
                "sample_rate": cfg_str(cfg, "cartesia_tts_sample_rate", "16000") or "16000",
            }
        )
    elif engine == "Kokoro-RealtimeTTS":
        inputs.update({"language": "ja", "speed": "1.0"})
    elif engine == "VOICEVOX-Core":
        inputs.update(
            {
                "speed_scale": _clamp(cfg_float(cfg, "voicevox_core_speed_scale", 1.0), 0.5, 2.0),
                "onnxruntime_file": cfg_str(
                    cfg,
                    "voicevox_core_onnxruntime_file",
                    _DEFAULT_VOICEVOX_ORT,
                ),
                "openjtalk_dict": cfg_str(
                    cfg,
                    "voicevox_core_openjtalk_dict",
                    r"voicevox_runtime/dict/open_jtalk_dic_utf_8-1.11",
                ),
                "vvm_path": cfg_str(cfg, "voicevox_core_vvm_path", r"voicevox_runtime/models/vvms/0.vvm"),
                "acceleration_mode": cfg_str(cfg, "voicevox_core_device", _DEFAULT_VOICEVOX_CORE_DEVICE) or _DEFAULT_VOICEVOX_CORE_DEVICE,
                "cpu_num_threads": int(_clamp(cfg_int(cfg, "voicevox_core_cpu_num_threads", 16), 1, 32)),
            }
        )
    elif engine in ("Irodori-v4", "Tsukasa"):
        inputs["reference_audio"] = cfg_str(
            cfg, "irodori_reference_audio" if engine == "Irodori-v4" else "tsukasa_reference_audio", ""
        )
    return inputs


# ----- 各ワーカー設定 ----------------------------------------------------------

def build_tts_worker_config(cfg: dict, engine: str) -> dict[str, object]:
    """`controller.py:_build_tts_worker_config` 相当。"""
    engine = normalize_tts_engine(engine)
    if engine == "Qwen3-TTS-CPP":
        seed_key = "qwen3_tts_cpp_seed"
        seed_text = cfg_str(cfg, seed_key, "")
        return {
            "host": _loopback_host(cfg, "qwen3_tts_cpp_host", "127.0.0.1") or "127.0.0.1",
            "port": int(_clamp(cfg_int(cfg, "qwen3_tts_cpp_port", 18081), 1, 65535)),
            "model_id": cfg_str(cfg, "qwen3_tts_cpp_model_id", "qwen3-tts-customvoice")
            or "qwen3-tts-customvoice",
            "voice": cfg_str(cfg, "qwen3_tts_cpp_speaker", "Vivian") or "Vivian",
            "instruct": cfg_str(cfg, "qwen3_tts_cpp_instruct", ""),
            "language": cfg_str(cfg, "qwen3_tts_cpp_language", "japanese") or "japanese",
            "do_sample": cfg_bool(cfg, "qwen3_tts_cpp_do_sample", True),
            "temperature": cfg_float(cfg, "qwen3_tts_cpp_temperature", 0.9),
            "top_k": cfg_int(cfg, "qwen3_tts_cpp_top_k", 50),
            "top_p": cfg_float(cfg, "qwen3_tts_cpp_top_p", 1.0),
            "repetition_penalty": cfg_float(cfg, "qwen3_tts_cpp_repetition_penalty", 1.05),
            "seed": cfg_int(cfg, seed_key, 0) if seed_text else None,
            "server_exe": cfg_str(
                cfg,
                "qwen3_tts_cpp_server_exe",
                _paths.default_audiocpp_server(),
            )
            or _paths.default_audiocpp_server(),
            "model_path": cfg_str(
                cfg,
                "qwen3_tts_cpp_model_path",
                "audio.cpp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/"
                "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf",
            )
            or (
                "audio.cpp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF/"
                "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf"
            ),
        }
    if engine == "LFM2.5-Audio":
        # 声は選べない（モデルの声 1 つ）ので voice は渡さない。model_path はフォルダー。
        return {
            "host": _loopback_host(cfg, "lfm2_audio_host", "127.0.0.1") or "127.0.0.1",
            "port": int(_clamp(cfg_int(cfg, "lfm2_audio_port", 18082), 1, 65535)),
            "model_id": cfg_str(cfg, "lfm2_audio_model_id", "lfm2.5-audio-jp") or "lfm2.5-audio-jp",
            "model_path": cfg_str(cfg, "lfm2_audio_model_path", "audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF")
            or "audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF",
            "server_exe": cfg_str(cfg, "lfm2_audio_server_exe", _paths.default_audiocpp_server())
            or _paths.default_audiocpp_server(),
        }
    if engine == "Kokoro-RealtimeTTS":
        return {
            "voice": _tts_voice_for_engine(cfg, engine, "jf_alpha"),
            "language": "ja",
            "speed": cfg_str(cfg, "kokoro_realtime_speed", "1.0") or "1.0",
            "input_streaming_enabled": False,
        }
    if engine == "Cartesia-Realtime":
        return {
            "model_id": cfg_str(cfg, "cartesia_tts_model_id", "sonic-3") or "sonic-3",
            "voice_id": _tts_voice_for_engine(cfg, engine, "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
            "sample_rate": cfg_str(cfg, "cartesia_tts_sample_rate", "16000") or "16000",
            "api_key": cfg_str(cfg, "cartesia_tts_api_key", ""),
            "api_key_env": cfg_str(cfg, "cartesia_tts_api_key_env", "") or "CARTESIA_API_KEY",
        }
    if engine == "VOICEVOX-Core":
        return {
            "voice": _tts_voice_for_engine(cfg, engine, "8"),
            "onnxruntime_file": cfg_str(
                cfg,
                "voicevox_core_onnxruntime_file",
                _DEFAULT_VOICEVOX_ORT,
            ),
            "openjtalk_dict": cfg_str(
                cfg,
                "voicevox_core_openjtalk_dict",
                r"voicevox_runtime/dict/open_jtalk_dic_utf_8-1.11",
            ),
            "vvm_path": cfg_str(cfg, "voicevox_core_vvm_path", r"voicevox_runtime/models/vvms/0.vvm"),
            "acceleration_mode": cfg_str(cfg, "voicevox_core_device", _DEFAULT_VOICEVOX_CORE_DEVICE) or _DEFAULT_VOICEVOX_CORE_DEVICE,
            "cpu_num_threads": int(_clamp(cfg_int(cfg, "voicevox_core_cpu_num_threads", 16), 1, 32)),
        }
    # VOICEVOX / TEMPLATE は worker を使わない（bridge 経由）。
    # 契約は Controller に合わせて **例外**にする（黙って {} を返すと
    # 呼び出し側のガード漏れに気づけない）。worker_configs 側で {} に落とす。
    if engine == "AivisSpeech":
        return {
            "host": _loopback_host(cfg, "aivisspeech_host", "127.0.0.1") or "127.0.0.1",
            "port": int(_clamp(cfg_int(cfg, "aivisspeech_port", 10104), 1, 65535)),
            "voice": _tts_voice_for_engine(cfg, engine, "888753761"),
        }
    if engine == "Supertonic-3":
        return {
            "host": _loopback_host(cfg, "supertonic_host", "127.0.0.1") or "127.0.0.1",
            "port": int(_clamp(cfg_int(cfg, "supertonic_port", 18080), 1, 65535)),
            "model_id": cfg_str(cfg, "supertonic_model_id", "supertonic-3") or "supertonic-3",
            "server_exe": cfg_str(cfg, "supertonic_server_exe", _paths.default_audiocpp_server())
            or _paths.default_audiocpp_server(),
            "model_path": cfg_str(
                cfg,
                "supertonic_model_path",
                "audio.cpp/models/Supertonic-3-GGUF/supertonic-3-orig.gguf",
            ),
            "voice": _tts_voice_for_engine(cfg, engine, "F1"),
            "steps": int(_clamp(cfg_int(cfg, "supertonic_steps", 8), 1, 8)),
        }
    if engine == "Irodori-v4":
        return {
            # speechcpp（既定。speech.cpp・専用の環境が要らず速い）/ audiocpp（audio.cpp の 500M v3 /
            # 600M v3 VoiceDesign）/ pytorch（.venv_irodori）
            "backend": cfg_str(cfg, "irodori_backend", "speechcpp") or "speechcpp",
            "audiocpp_model": cfg_str(cfg, "irodori_audiocpp_model", "500m-v3") or "500m-v3",
            "audiocpp_port": int(_clamp(cfg_int(cfg, "irodori_audiocpp_port", 18083), 1, 65535)),
            "host": _loopback_host(cfg, "irodori_audiocpp_host", "127.0.0.1") or "127.0.0.1",
            "server_exe": cfg_str(cfg, "irodori_audiocpp_server_exe", _paths.default_audiocpp_server())
            or _paths.default_audiocpp_server(),
            "speechcpp_model": cfg_str(cfg, "irodori_speechcpp_model", "irodori-tts-mf:q8_0") or "irodori-tts-mf:q8_0",
            "speechcpp_device": cfg_str(cfg, "irodori_speechcpp_device", "gpu") or "gpu",
            "speechcpp_steps": int(_clamp(cfg_int(cfg, "irodori_speechcpp_steps", 4), 1, 40)),
            "seed": int(_clamp(cfg_int(cfg, "irodori_seed", 1), 0, 2**31 - 1)),
            "speechcpp_exe": cfg_str(cfg, "speechcpp_exe", ""),
            "irodori_python": _venv_python_setting(cfg, "irodori_python", ".venv_irodori"),
            "checkpoint": cfg_str(cfg, "irodori_checkpoint", "Aratako/Irodori-TTS-v4.1-Small")
            or "Aratako/Irodori-TTS-v4.1-Small",
            "voice": _tts_voice_for_engine(cfg, engine, "明るく自然な日本語の声"),
            "reference_audio": cfg_str(cfg, "irodori_reference_audio", ""),
            "steps": int(_clamp(cfg_int(cfg, "irodori_steps", 16), 1, 40)),
        }
    if engine == "Tsukasa":
        return {"root": cfg_str(cfg, "tsukasa_root", "Tsukasa-HF") or "Tsukasa-HF", "reference_audio": cfg_str(cfg, "tsukasa_reference_audio", ""), "steps": int(_clamp(cfg_int(cfg, "tsukasa_steps", 5), 3, 15)), "alpha": _clamp(cfg_float(cfg, "tsukasa_alpha", 0.3), 0.0, 1.0), "beta": _clamp(cfg_float(cfg, "tsukasa_beta", 0.7), 0.0, 1.0)}
    raise RuntimeError(f"TTS worker未対応エンジンです: {engine}")


def derive_qwen_served_model_name(model_ref: str) -> str:
    """Qwen ASR の served model 名を導く（`controller._derive_qwen_served_model_name` と同一）。

    設定に入っているのはモデル参照（HF ID かローカルパス）で、vLLM に渡すのは
    その末尾要素。GUI 側と同じ式にしてある。
    """
    text = (model_ref or "").strip()
    if not text:
        return "Qwen3-ASR-0.6B"
    normalized = text.replace("\\", "/").rstrip("/")
    leaf = normalized.split("/")[-1]
    return leaf if leaf else "Qwen3-ASR-0.6B"


def _looks_like_hf_repo(text: str) -> bool:
    s = (text or "").strip().replace("\\", "/")
    if not s or s.startswith("/") or (len(s) >= 2 and s[1] == ":"):
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(p.strip() for p in parts)


def _whisper_local_path_exists(model_path: str) -> bool:
    """faster-whisper の Path がローカルの実在パスか。"""
    text = (model_path or "").strip()
    if not text:
        return False
    try:
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate.exists()
        return _paths.resolve_under_repo(candidate).exists()
    except OSError:
        return False


def _resolve_whisper_refs(cfg: dict) -> tuple[str, str, str]:
    """faster-whisper のサイズ / HF / Path。旧設定で HF ID が混ざっていても分ける。"""
    name = cfg_str(cfg, "asr_whisper_model", "large-v3") or "large-v3"
    path = cfg_str(cfg, "asr_whisper_model_path", "")
    if "asr_whisper_hf" in cfg:
        hf = cfg_str(cfg, "asr_whisper_hf", "")
    else:
        hf = ""
        if path and not _whisper_local_path_exists(path) and _looks_like_hf_repo(path):
            hf = path
            path = ""
        elif _looks_like_hf_repo(name):
            hf = name
            name = "large-v3"
        else:
            hf = "deepdml/faster-whisper-large-v3-turbo-ct2"
    return name, hf, path

def build_asr_config_dict(cfg: dict) -> dict[str, object]:
    """`controller.py:_build_asr_config_dict` 相当。"""
    engine = cfg_str(cfg, "asr_engine", "sherpa-onnx") or "sherpa-onnx"
    whisper_name, whisper_hf, whisper_path = _resolve_whisper_refs(cfg)
    return {
        "asr_engine": engine,
        "vad_backend": cfg_str(cfg, "asr_vad_backend", "maai") or "maai",
        "turn_backend": cfg_str(cfg, "asr_turn_backend", "maai") or "maai",
        "maai_output_gui_plot": cfg_bool(cfg, "asr_maai_gui_plot", False),
        "sample_rate": 16000,
        "frame_rate": _clamp(cfg_float(cfg, "maai_test_turn_vap_frame_rate", 20.0), 0.1, 500.0),
        "maai_vap_mode": cfg_str(cfg, "maai_test_turn_vap_mode", "vap_mc") or "vap_mc",
        "maai_vap_lang": cfg_str(cfg, "maai_test_turn_vap_lang", "jp_kyoto") or "jp_kyoto",
        "maai_vap_device": "cuda" if _device(cfg_str(cfg, "maai_device", "cuda") or "cuda", label="MaAI").lower() == "cuda" else "cpu",
        "maai_vap_model_type": cfg_str(cfg, "maai_test_turn_vap_model_type", "normal") or "normal",
        "inference_queue_size": 4,
        "vad_start_threshold": cfg_float(cfg, "asr_vad_on_threshold", 0.1),
        "vad_start_hold_sec": cfg_float(cfg, "asr_vad_on_sec", 0.1),
        "vad_end_threshold": cfg_float(cfg, "asr_vad_off_threshold", 0.2),
        "vad_end_hold_sec": cfg_float(cfg, "asr_vad_off_sec", 0.1),
        "min_segment_sec": cfg_float(cfg, "asr_min_segment_sec", 0.5),
        "pre_roll_buffer_sec": cfg_float(cfg, "asr_pre_roll_buffer_sec", 1.0),
        "nonstream_pseudo_enabled": cfg_bool(cfg, "asr_nonstream_pseudo_enabled", True),
        "nonstream_pseudo_chunk_sec": cfg_float(cfg, "asr_nonstream_pseudo_chunk_sec", 0.25),
        "whisper_model_name": whisper_name,
        "whisper_hf_id": whisper_hf,
        "whisper_model_path": whisper_path,
        "whisper_device": _device(cfg_str(cfg, "asr_device", "cuda") or "cuda", label="faster-whisper"),
        "whisper_compute_type": _compute_type(
            cfg_str(cfg, "asr_compute_type", "int8_float16") or "int8_float16",
            label="faster-whisper"),
        "sherpa_tokens_path": cfg_str(
            cfg, "asr_sherpa_tokens", "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt"
        )
        or "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt",
        "sherpa_encoder_path": cfg_str(
            cfg,
            "asr_sherpa_encoder",
            "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.int8.onnx",
        )
        or "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.int8.onnx",
        "sherpa_decoder_path": cfg_str(
            cfg,
            "asr_sherpa_decoder",
            "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.onnx",
        )
        or "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.onnx",
        "sherpa_joiner_path": cfg_str(
            cfg,
            "asr_sherpa_joiner",
            "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.int8.onnx",
        )
        or "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.int8.onnx",
        "sherpa_num_threads": int(_clamp(cfg_int(cfg, "asr_sherpa_num_threads", 2), 1, 32)),
        "sherpa_provider": _device(
            cfg_str(cfg, "asr_sherpa_provider", "cpu") or "cpu", label="sherpa-onnx"),
        "qwen3_onnx_conv_frontend_path": cfg_str(
            cfg, "asr_qwen3_onnx_conv_frontend",
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx",
        ) or "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx",
        "qwen3_onnx_encoder_path": cfg_str(
            cfg, "asr_qwen3_onnx_encoder",
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx",
        ) or "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx",
        "qwen3_onnx_decoder_path": cfg_str(
            cfg, "asr_qwen3_onnx_decoder",
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx",
        ) or "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx",
        "qwen3_onnx_tokenizer_path": cfg_str(
            cfg, "asr_qwen3_onnx_tokenizer",
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer",
        ) or "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer",
        "qwen3_onnx_language": cfg_str(cfg, "asr_qwen3_onnx_language", "Japanese"),
        "qwen3_onnx_provider": _device(cfg_str(cfg, "asr_qwen3_onnx_provider", "cpu") or "cpu", label="qwen3-asr-onnx"),
        "qwen3_onnx_num_threads": cfg_int(cfg, "asr_qwen3_onnx_num_threads", 8),
        # sherpa-onnx OfflineRecognizer 系。既定値は ASRService のコンストラクタと同じ
        # （= k2-fsa release tarball を展開したディレクトリ名）。provider が cpu 固定寄りな
        # 理由と ITN を既定 True にしている根拠は asr/engines_sherpa_offline.py の冒頭。
        "parakeet_onnx_model_dir": cfg_str(
            cfg, "asr_parakeet_onnx_dir", "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"
        )
        or "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
        "parakeet_onnx_provider": _device(
            cfg_str(cfg, "asr_parakeet_onnx_provider", "cpu") or "cpu", label="parakeet-onnx"),
        "parakeet_onnx_num_threads": cfg_int(cfg, "asr_parakeet_onnx_num_threads", 8),
        "sensevoice_onnx_model_dir": cfg_str(
            cfg, "asr_sensevoice_onnx_dir", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
        )
        or "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        "sensevoice_onnx_provider": _device(
            cfg_str(cfg, "asr_sensevoice_onnx_provider", "cpu") or "cpu",
            label="sensevoice-onnx"),
        "sensevoice_onnx_num_threads": cfg_int(cfg, "asr_sensevoice_onnx_num_threads", 8),
        "sensevoice_onnx_language": cfg_str(cfg, "asr_sensevoice_onnx_language", "ja"),
        "sensevoice_onnx_use_itn": cfg_bool(cfg, "asr_sensevoice_onnx_itn", True),
        "funasr_nano_onnx_model_dir": cfg_str(
            cfg, "asr_funasr_nano_onnx_dir", "sherpa-onnx-funasr-nano-int8-2025-12-30"
        )
        or "sherpa-onnx-funasr-nano-int8-2025-12-30",
        "funasr_nano_onnx_provider": _device(
            cfg_str(cfg, "asr_funasr_nano_onnx_provider", "cpu") or "cpu",
            label="funasr-nano-onnx"),
        "funasr_nano_onnx_num_threads": cfg_int(cfg, "asr_funasr_nano_onnx_num_threads", 8),
        "funasr_nano_onnx_language": cfg_str(cfg, "asr_funasr_nano_onnx_language", "ja"),
        "funasr_nano_onnx_use_itn": cfg_bool(cfg, "asr_funasr_nano_onnx_itn", True),
        "funasr_nano_onnx_max_new_tokens": cfg_int(cfg, "asr_funasr_nano_onnx_max_new_tokens", 128),
        "cohere_onnx_model_dir": cfg_str(
            cfg, "asr_cohere_onnx_dir", "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01"
        )
        or "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
        "cohere_onnx_provider": _device(
            cfg_str(cfg, "asr_cohere_onnx_provider", "cpu") or "cpu", label="cohere-onnx"),
        "cohere_onnx_num_threads": cfg_int(cfg, "asr_cohere_onnx_num_threads", 8),
        "cohere_onnx_language": cfg_str(cfg, "asr_cohere_onnx_language", "ja"),
        "cohere_onnx_use_punct": cfg_bool(cfg, "asr_cohere_onnx_punct", True),
        "cohere_onnx_use_itn": cfg_bool(cfg, "asr_cohere_onnx_itn", True),
        # speech.cpp（asr/engines_speechcpp.py）。device は gpu（外付けの GPU、無ければ CPU）/ cpu
        "speechcpp_asr_model": cfg_str(cfg, "asr_speechcpp_model", "haas-ja-v1.0") or "haas-ja-v1.0",
        "speechcpp_asr_device": cfg_str(cfg, "asr_speechcpp_device", "gpu") or "gpu",
        "speechcpp_asr_threads": int(_clamp(cfg_int(cfg, "asr_speechcpp_threads", 0), 0, 64)),
        "speechcpp_asr_language": cfg_str(cfg, "asr_speechcpp_language", ""),
        "speechcpp_exe": cfg_str(cfg, "speechcpp_exe", ""),
        "nemotron_onnx_tokens_path": cfg_str(cfg, "asr_nemotron_onnx_tokens", ""),
        "nemotron_onnx_encoder_path": cfg_str(cfg, "asr_nemotron_onnx_encoder", ""),
        "nemotron_onnx_decoder_path": cfg_str(cfg, "asr_nemotron_onnx_decoder", ""),
        "nemotron_onnx_joiner_path": cfg_str(cfg, "asr_nemotron_onnx_joiner", ""),
        "nemotron_onnx_language": cfg_str(cfg, "asr_nemotron_onnx_language", ""),
        # Execution mode preset: "gpu-fp32" (CUDA, best latency+accuracy) or
        # "cpu-int8" (no GPU; fp32 on CPU is slow). Derives provider + quant.
        "nemotron_onnx_provider": _device(
            "cpu" if cfg_str(cfg, "asr_nemotron_onnx_compute", "gpu-fp32") == "cpu-int8" else "cuda",
            label="nemotron-onnx"),
        "nemotron_onnx_quant": (
            "int8"
            if _is_cpu_build()
            or cfg_str(cfg, "asr_nemotron_onnx_compute", "gpu-fp32") == "cpu-int8"
            else "fp32"
        ),
        "smart_turn_model_path": cfg_str(cfg, "asr_smart_turn_model", "smart-turn-v3.2-gpu.onnx")
        or "smart-turn-v3.2-gpu.onnx",
        "smart_turn_threshold": cfg_float(cfg, "asr_smart_turn_threshold", 0.5),
        # 既定 2。上げると他のワーカー（MaAI・TTS）からコアを奪う。
        "smart_turn_num_threads": int(_clamp(cfg_int(cfg, "asr_smart_turn_num_threads", 2), 1, 32)),
        # ASR プロセスの torch 上限（既定 2）。turn_backend="maai" のときの
        # ターン判定 MaAI がここを使う。上げると MaAI ワーカーからコアを奪う。
        "torch_cpu_intra_threads": int(_clamp(cfg_int(cfg, "asr_torch_cpu_intra_threads", 2), 1, 32)),
        "nemotron_model": cfg_str(cfg, "asr_nemotron_model", "nvidia/nemotron-3.5-asr-streaming-0.6b")
        or "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "nemotron_chunk_size_ms": cfg_int(cfg, "asr_nemotron_chunk_size_ms", 320),
        "nemotron_window_sec": cfg_float(cfg, "asr_nemotron_window_sec", 15.0),
        "nemotron_max_symbols": cfg_int(cfg, "asr_nemotron_max_symbols", 30),
        "nemotron_lang": cfg_str(cfg, "asr_nemotron_lang", "ja-JP") or "ja-JP",
        "nemotron_device": _device(cfg_str(cfg, "asr_nemotron_device", "cuda") or "cuda", label="nemotron-asr"),
        "nemotron_finalize_mode": cfg_str(cfg, "asr_nemotron_finalize_mode", "vad") or "vad",
        # 既定は `scripts/setup_nemotron_venv.ps1` が作る .venv_nemotron。
        # 以前は外部の絶対パス C:/NeMo/.venv（NeMo ソースツリーの venv）を指していて、
        # そのパスが無い環境では nemotron-asr が起動できなかった。
        # 相対パスは asr/service 側でリポジトリルート基準に解決される。
        "nemotron_python": _venv_python_setting(cfg, "asr_nemotron_python", ".venv_nemotron"),
        "nemotron_remote_host": _loopback_host(cfg, "asr_nemotron_remote_host", ""),
        "nemotron_remote_port": cfg_int(cfg, "asr_nemotron_remote_port", 0),
        # --- ここから: 以前は `launch_asr`（ASR 単体起動）にしか無く、音声対話の
        # ASR ワーカー設定には渡っていなかった項目。nemotron と同じ取りこぼしで、
        # qwen3-asr / chirp3-asr / external-asr-tcp は設定欄の値が捨てられていた。
        # 既定は ASRService のコンストラクタ既定（= ui.py のウィジェット初期値）。
        # chirp3-asr (Google Cloud Speech-to-Text V2)
        "chirp_project_id": cfg_str(cfg, "asr_chirp_project_id", ""),
        "chirp_location": cfg_str(cfg, "asr_chirp_location", "global") or "global",
        "chirp_recognizer": cfg_str(cfg, "asr_chirp_recognizer", "_") or "_",
        "chirp_model": cfg_str(cfg, "asr_chirp_model", "chirp_3") or "chirp_3",
        # external-asr-tcp
        "external_asr_host": _loopback_host(cfg, "asr_external_host", "127.0.0.1") or "127.0.0.1",
        "external_asr_port": cfg_int(cfg, "asr_external_port", 19090),
        "external_asr_audio_host": _loopback_host(cfg, "asr_external_audio_host", "127.0.0.1")
        or "127.0.0.1",
        "external_asr_audio_port": cfg_int(cfg, "asr_external_audio_port", 14600),
        # qwen3-asr (vLLM / WebSocket)
        "qwen_host": _loopback_host(cfg, "asr_qwen_host", "127.0.0.1") or "127.0.0.1",
        "qwen_port": cfg_int(cfg, "asr_qwen_port", 8000),
        "qwen_ws_path": cfg_str(cfg, "asr_qwen_ws_path", "/ws") or "/ws",
        "qwen_model": derive_qwen_served_model_name(cfg_str(cfg, "asr_qwen_model", "")),
        "qwen_chunk_sec": cfg_float(cfg, "asr_qwen_chunk_sec", 2.0),
        "qwen_timeout_sec": cfg_float(cfg, "asr_qwen_timeout_sec", 30.0),
        "qwen_session_timeout_sec": cfg_float(cfg, "asr_qwen_session_timeout_sec", 60.0),
        "qwen_no_text_flush_sec": cfg_float(cfg, "asr_qwen_no_text_flush_sec", 0.0),
        "qwen_min_return_audio_sec": cfg_float(cfg, "asr_qwen_min_return_audio_sec", 0.0),
        "qwen_finish_on_vad_end": cfg_bool(cfg, "asr_qwen_finish_on_vad_end", True),
        # faster-whisper: ローカル Path が実在するときだけローカル参照にする
        "whisper_use_local_path": _whisper_local_path_exists(
            cfg_str(cfg, "asr_whisper_model_path", "")
        ),
    }


def asr_worker_ready_timeout_sec(config: dict[str, object]) -> float:
    """ASR worker の ready 待ち秒数。Nemotron はモデルロード＋サブプロセス起動に時間がかかる。"""
    engine = str(config.get("asr_engine", "") or "").strip().lower()
    if engine == "nemotron-asr":
        return 300.0
    return 90.0


def build_audio_input_worker_config(
    cfg: dict, *, force_browser_mic: bool = False,
) -> dict[str, object]:
    """`controller.py:_build_audio_input_worker_config` 相当。

    Web UI は ``force_browser_mic=True`` で mic を browser モードにする。
    コンテナの PortAudio はホストマイクを見えないため、ブラウザから PCM を受ける。
    """
    mode = cfg_str(cfg, "audio_input_mode", "mic").lower() or "mic"
    if force_browser_mic and mode == "mic":
        mode = "browser"
    gain = _clamp(cfg_float(cfg, "audio_input_gain", 1.0), 0.0, 20.0)
    # AEC は Controller の保存キー（audio_input_enable_aec / audio_input_wav_enable_aec）に合わせる。
    # 旧来の "audio_input_wav_aec" は存在しないキーで常に False になり、Controller 側と食い違っていた。
    aec_on = cfg_bool(cfg, "audio_input_enable_aec", cfg_bool(cfg, "audio_input_wav_enable_aec", False))
    # frame_rate / vad_backend は audio_input ワーカー未使用のため含めない（Controller 側と合わせる）。
    out: dict[str, object] = {
        "sample_rate": 16000,
        "input_mode": mode,
        "start_paused": mode == "wav",
        "input_gain": float(gain),
        "enable_aec": aec_on,
        "wav_enable_aec": aec_on,
    }
    ref_val = cfg_str(cfg, "audio_input_wav_aec_ref", "other_ch") or "other_ch"
    if mode != "wav":
        ref_val = "system"
    out["aec_ref"] = ref_val
    out["wav_aec_ref"] = ref_val
    if mode == "mic":
        # 保存キーは audio_input_mic_device_index（旧 "audio_input_mic_device" はタイポで常に -1 → 食い違い）。
        dev = cfg_int(cfg, "audio_input_mic_device_index", -1)
        if dev >= 0:
            out["mic_device_index"] = int(dev)
    elif mode == "browser":
        pass
    elif mode == "wav":
        out["wav_path"] = _paths.resolve_runtime_wav_path(cfg_str(cfg, "audio_input_wav_path", ""))
        playback = cfg_bool(cfg, "audio_input_wav_local_playback", bool(force_browser_mic))
        # PyQt は PortAudio でスピーカーへ出す。Docker/Web はデバイスが無く
        # OutputStream が fatal になるので、ワーカー再生は切ってブラウザへ PCM を送る。
        out["wav_local_playback"] = playback and not force_browser_mic
        out["wav_browser_monitor"] = bool(force_browser_mic) and playback
        # ローカル再生の出力デバイス（名前の部分一致、空なら既定）。GUI にまだ欄が無く、
        # GUI の保存は知っているキーだけで書き直すので、一時的に変えたいときは環境変数で渡す。
        out["wav_local_playback_device"] = (
            cfg_str(cfg, "audio_input_wav_local_playback_device", "")
            or os.environ.get("MAAI_SDS_WAV_PLAYBACK_DEVICE", "")
        ).strip()
        out["wav_send_other_ch_to_mmd"] = cfg_bool(cfg, "audio_input_wav_other_ch_to_mmd", False)
        out["wav_other_ch_to_maai_ch2"] = cfg_bool(cfg, "audio_input_wav_other_ch_to_maai_ch2", False)
        out["wav_channel_index"] = max(0, min(1, cfg_int(cfg, "audio_input_wav_channel_index", 0)))
        out["wav_start_sec"] = cfg_str(cfg, "audio_input_wav_start_sec", "0") or "0"
        out["wav_end_sec"] = cfg_str(cfg, "audio_input_wav_end_sec", "")
    else:
        host = cfg_str(cfg, "audio_input_tcp_host", "127.0.0.1") or "127.0.0.1"
        port = cfg_int(cfg, "audio_input_tcp_port", 15600)
        out["tcp_host"] = host
        out["tcp_port"] = int(max(1, min(65535, port)))
    return out


# `build_remote_plugin_tcp_settings_dict` が参照する MDF 由来キー（通常は controller_config.json に無い）。
_PLUGIN_REMOTE_MDF_KEYS: tuple[str, ...] = (
    "Plugin_Remote_EnableServer",
    "Plugin_Remote_EnableClient",
    "Plugin_Remote_Hostname",
    "Plugin_Remote_Websocket_Host",
    "Plugin_Remote_Websocket_Port",
    "Plugin_Remote_Websocket_Directory",
    "Plugin_Remote_Port",
    "Plugin_Remote_ListenPort",
)


def merge_plugin_remote_from_mdf_into_cfg(cfg: dict) -> dict:
    """`mdf_path` の MDF から Plugin_Remote_* を cfg に重ねる（cfg は破壊しない）。

    GUI は `load_mdf_settings` で MDF をウィジェットへ載せ `_build_remote_plugin_tcp_settings_dict` に使うが、
    先行起動は JSON のみ見るため、マージしないと `remote_plugin_settings` のダイジェストが GUI とずれ、
    音声対話開始のたびに audio_output が再起動する。
    """
    out = dict(cfg)
    path_s = str(out.get("mdf_path", "") or "").strip()
    if not path_s:
        return out
    try:
        mdf_path = Path(path_s)
    except Exception:
        return out
    if not mdf_path.is_file():
        return out
    try:
        try:
            text = mdf_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = mdf_path.read_text(encoding="cp932", errors="replace")
    except OSError:
        return out
    lines = text.splitlines()
    from ..mmd.mdf_service import MdfService
    parsed = MdfService.parse_mdf_key_values(lines)
    for k in _PLUGIN_REMOTE_MDF_KEYS:
        if k not in parsed:
            continue
        val = str(parsed[k])
        if k == "Plugin_Remote_Websocket_Directory" and val and not val.startswith("/"):
            val = f"/{val}"
        out[k] = val
    return out


_MODEL_ADD_RE = re.compile(r"MODEL_ADD\|([^|\s]+)\|")


def avatar_model_alias_from_mdf(mdf_path: str) -> str:
    """MDF と同じ名前の FST から、最初に MODEL_ADD するモデルの別名を読む（無ければ "0"）。

    Plugin_Remote は `__AV_SETMODEL,<別名>` で指定したモデルだけに音声を渡し、口パクする
    （モデルが割り当てられていないと受け取った音声を捨てる）。別名は FST ごとに違い、公式のサンプル集の
    example_websocket.fst は `gene`、自分で作る FST では `0` が多い。FST の書式（4 列の旧形式・
    `0 LOOP:` のブロック形式）を問わず、コメント以外の行の最初の `MODEL_ADD|<別名>|` を使う。
    """
    text_path = str(mdf_path or "").strip()
    if not text_path:
        return "0"
    fst = Path(text_path).with_suffix(".fst")
    try:
        raw = fst.read_bytes()
    except OSError:
        return "0"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp932", errors="replace")
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = _MODEL_ADD_RE.search(line)
        if match:
            return match.group(1)
    return "0"


def build_remote_plugin_tcp_settings_dict(
    cfg: dict,
    *,
    running_in_docker: bool | None = None,
    log_docker_substitution: bool = True,
) -> dict[str, object]:
    """`controller.py:_build_remote_plugin_tcp_settings_dict` 相当。

    MDF に WebSocket の Port/Directory が無いとき、GUI は `load_mdf_settings` で
    `gui/settings.FIELD_SPECS` の既定（Port=80, Directory=/）に戻してから MDF を上塗りする。
    ここを 9001 / /chat にすると先行起動と音声対話開始で `remote_plugin_settings` が食い違い、
    audio_output が毎回再起動する。
    """
    # Controller: `_widget_to_value(...) or "9001"` は「空欄」のときだけ 9001。既定ウィジェットは "80"。
    ws_port = cfg_int(cfg, "Plugin_Remote_Websocket_Port", 80)
    # Controller: `or "/chat"` は空欄時。MDF 未設定の既定ウィジェットは "/"。
    ws_dir = cfg_str(cfg, "Plugin_Remote_Websocket_Directory", "/") or "/chat"
    enable_server = cfg_bool(cfg, "Plugin_Remote_EnableServer", False)
    enable_client = cfg_bool(cfg, "Plugin_Remote_EnableClient", False)
    docker = _paths.running_in_docker() if running_in_docker is None else bool(running_in_docker)
    settings: dict[str, object] = {
        "tcp_enable_server": enable_server,
        "tcp_enable_client": enable_client,
        "ws_host": cfg_str(cfg, "Plugin_Remote_Websocket_Host", "localhost") or "localhost",
        "ws_port": ws_port,
        "ws_directory": ws_dir,
        "tcp_port": cfg_int(cfg, "Plugin_Remote_Port", 60001),
        "tcp_listen_port": cfg_int(cfg, "Plugin_Remote_ListenPort", 60001),
        "tcp_server_host": cfg_str(cfg, "Plugin_Remote_Bind_Host", "0.0.0.0"),
        # Native の従来動作は常に loopback 接続。Docker の TCP client モードだけ、
        # MDF の Hostname を接続先指定として使い loopback を Windows ホストへ読み替える。
        "tcp_client_host": "127.0.0.1",
        # 音声と口パクを渡すモデル（`__AV_SETMODEL,<別名>`）。FST の MODEL_ADD から読む。
        # GUI は mdf_path を外して渡すので、先に読んだ avatar_model_alias を優先する。
        "model_alias": cfg_str(cfg, "avatar_model_alias", "")
        or avatar_model_alias_from_mdf(cfg_str(cfg, "mdf_path", "")),
    }

    if not docker:
        return settings

    loopback = {"", "localhost", "127.0.0.1"}
    if enable_server:
        # Plugin_Remote は MMDAgent-EX 側が server、Python 側が TCP client。
        configured = cfg_str(cfg, "Plugin_Remote_Hostname", "localhost") or "localhost"
        if configured.strip().lower() in loopback:
            settings["tcp_client_host"] = "host.docker.internal"
            if log_docker_substitution:
                print(
                    "Remote bridge: Docker TCP client の接続先を "
                    f"{configured!r} から 'host.docker.internal' へ置換します",
                    flush=True,
                )
        else:
            settings["tcp_client_host"] = configured.strip()
    elif enable_client:
        # Plugin_Remote は MMDAgent-EX 側が client、Python 側が TCP server。
        host = str(settings.get("tcp_server_host") or "").strip()
        if host.lower() in loopback:
            settings["tcp_server_host"] = "0.0.0.0"
    else:
        # WebSocket は常に Python 側が server。
        host = str(settings.get("ws_host") or "").strip()
        if host.lower() in loopback:
            settings["ws_host"] = "0.0.0.0"
    return settings


def build_audio_output_worker_config_dict(
    cfg: dict,
    *,
    output_mode: str | None = None,
    assume_mmd_running: bool = False,
) -> dict[str, object]:
    """`controller.py:_build_audio_output_worker_config_dict` 相当。

    `assume_mmd_running=False`（既定）のとき、保存が ``mmd`` でも MMDAgent 未起動とみなし
    ``output_mode`` を ``local`` に落とす。`Controller._effective_audio_output_mode` と揃え、
    Bootstrap 直後の `_sync_orchestrator_audio_output_from_ui` で無駄にプロセスを取り直さない。
    MMD 起動済みでワーカーを ``mmd`` で立てたい場合のみ ``assume_mmd_running=True``。
    """
    cfg = merge_plugin_remote_from_mdf_into_cfg(cfg)
    rp = build_remote_plugin_tcp_settings_dict(cfg)
    tcp_host = cfg_str(cfg, "audio_output_tcp_host", "127.0.0.1") or "127.0.0.1"
    tcp_port = cfg_int(cfg, "audio_output_tcp_port", 15601)
    tcp_port = max(1, min(65535, tcp_port))
    saved = (output_mode or cfg_str(cfg, "audio_output_mode", "mmd") or "mmd").strip().lower()
    # Web の旧マニフェストは Qt combo の itemData ではなく表示名を保存し得た。
    # 既存 controller_config.json も bootstrap 時点から正しい経路へ復旧できるようにする。
    saved = {
        "mmd (remote)": "mmd",
        "ローカルスピーカー": "local",
        "tcp 送出": "tcp",
    }.get(saved, saved)
    if saved == "mmd" and not assume_mmd_running:
        eff_mode = "local"
    else:
        eff_mode = saved
    return {
        "output_mode": eff_mode,
        "sample_rate": 16000,
        "mmd_burst_snd_frames": _clamp(cfg_int(cfg, "mmd_burst_snd_frames", 20), 0, 500),
        "mmd_blink_enable": cfg_bool(cfg, "mmd_blink_enable", True),
        "mmd_blink_interval_mean_sec": _cfg_blink_interval_mean_sec(cfg),
        "mmd_blink_morph": _cfg_blink_morph(cfg),
        "mmd_blink_mode": _cfg_blink_mode(cfg),
        "remote_plugin_settings": rp,
        "tcp_output_host": tcp_host,
        "tcp_output_port": tcp_port,
    }


def _resolve_backchannel_dictionary(cfg: dict) -> str:
    """相槌の言い方の一覧のパス。相対パス・空欄はリポジトリ（Docker ではホストのリポジトリ）基準で読む。

    対話サーバーの今のフォルダーから探すと、起動のしかたによって見つからず、相槌がいつも
    既定の「はい。」になる。空欄も GUI の既定と同じ backchannel_forms.json にする。
    """
    raw = cfg_str(cfg, "maai_backchannel_dictionary_path", "") or "backchannel_forms.json"
    resolved = _paths.resolve_docker_local_file(raw)
    if resolved and not Path(resolved).is_absolute():
        resolved = str(_paths.resolve_under_repo(resolved))
    return resolved


def build_maai_worker_config(cfg: dict) -> dict[str, object]:
    """`controller.py:_build_maai_worker_config` 相当。"""
    bc_enabled = cfg_bool(cfg, "maai_enable_bc", True)
    nod_enabled = cfg_bool(cfg, "maai_enable_nod", True)
    if bc_enabled and nod_enabled:
        mode = "both"
    elif bc_enabled:
        mode = "bc"
    elif nod_enabled:
        mode = "nod_para"
    else:
        mode = "none"
    out: dict[str, object] = {
        "mode": mode,
        "device": "cuda" if _device(cfg_str(cfg, "maai_device", "cuda") or "cuda", label="MaAI").lower() == "cuda" else "cpu",
        "bc_frame_rate": _clamp(cfg_float(cfg, "maai_test_bc_frame_rate", 10.0), 0.1, 500.0),
        "bc_lang": cfg_str(cfg, "maai_test_bc_lang", "jp") or "jp",
        "bc_model_type": cfg_str(cfg, "maai_test_bc_model_type", "normal") or "normal",
    }
    _bc_local = _paths.resolve_docker_local_file(cfg_str(cfg, "maai_test_bc_local_model", "") or "")
    if _bc_local:
        out["bc_local_model"] = _bc_local
    out.update({
        "nod_frame_rate": _clamp(cfg_float(cfg, "maai_test_nod_frame_rate", 12.5), 0.1, 500.0),
        "nod_lang": cfg_str(cfg, "maai_test_nod_lang", "jp") or "jp",
        "nod_model_type": cfg_str(cfg, "maai_test_nod_model_type", "normal-ver2") or "normal-ver2",
        # MaAI マルチモデル（GUI は送っていたが headless は送っていなかった）。
        # 既定値は Controller の `_maai_test_line_float(..., 10.0, 0.1, 500.0)` /
        # `_maai_test_line_str(..., "normal-ver2")` に合わせる。
        "use_multi": cfg_bool(cfg, "maai_multi_enabled", False),
        "multi_frame_rate": _clamp(cfg_float(cfg, "maai_multi_frame_rate", 10.0), 0.1, 500.0),
        "multi_model_type": cfg_str(cfg, "maai_multi_model_type", "normal-ver2") or "normal-ver2",
        "maai_root": (
            cfg_str(cfg, "maai_root", "")
            or str(_paths.assets_root())
        ),
        "use_mimi_onnx": cfg_bool(cfg, "maai_use_mimi_onnx", True),
        # 既定は HF（maai-kyoto/continuous-mimi-onnx）から取る。ローカルの ONNX はセットアップが置かない
        "use_local_mimi_onnx": cfg_bool(cfg, "maai_use_local_mimi_onnx", False),
        # maai 側は Mimi encoder 限定で N フレームまとめて 1 回の ONNX 呼び出しに
        # 詰める（inference_chunk_frames）。既定 1（従来どおり毎フレーム推論）。
        "inference_chunk_frames": int(
            _clamp(cfg_int(cfg, "maai_inference_chunk_frames", 1), 1, 8)
        ),
        "mimi_onnx_cpu_intra_threads": int(
            _clamp(cfg_int(cfg, "maai_mimi_onnx_cpu_intra_threads", 8), 1, 64)
        ),
        "mimi_onnx_cpu_inter_threads": int(
            _clamp(cfg_int(cfg, "maai_mimi_onnx_cpu_inter_threads", 1), 1, 64)
        ),
        # f592640 より前は torch の既定（24）だった。3 モデル同時なので上げると
        # 効く場合がある。既定は現状維持の 1。
        "torch_cpu_interop_threads": int(
            _clamp(cfg_int(cfg, "maai_torch_cpu_interop_threads", 1), 1, 24)
        ),
        "torch_cpu_intra_threads": int(
            _clamp(cfg_int(cfg, "maai_torch_cpu_intra_threads", 1), 1, 24)
        ),
    })
    _maai_src = (cfg_str(cfg, "maai_src_path", "") or "").strip()
    if _maai_src:
        out["maai_src_path"] = _maai_src
    if bool(out.get("use_mimi_onnx")) and bool(out.get("use_local_mimi_onnx")):
        for _k_src, _k_dst in (
            ("maai_mimi_onnx_path", "mimi_onnx_path"),
            ("maai_mimi_onnx_meta_path", "mimi_onnx_meta_path"),
        ):
            _v = _paths.resolve_docker_local_file(cfg_str(cfg, _k_src, "") or "")
            if _v:
                out[_k_dst] = _v
    else:
        # HF 取得に任せるためローカルパスを渡さない
        out.pop("maai_root", None)
    if cfg_bool(cfg, "maai_enable_turn_vap", True):
        out["turn_vap_enabled"] = True
        out["turn_vap_mode"] = cfg_str(cfg, "maai_test_turn_vap_mode", "vap_mc") or "vap_mc"
        out["turn_vap_lang"] = cfg_str(cfg, "maai_test_turn_vap_lang", "jp_kyoto") or "jp_kyoto"
        out["turn_vap_frame_rate"] = _clamp(cfg_float(cfg, "maai_test_turn_vap_frame_rate", 20.0), 0.1, 500.0)
        out["turn_vap_model_type"] = cfg_str(cfg, "maai_test_turn_vap_model_type", "normal") or "normal"
        _tv_local = _paths.resolve_docker_local_file(cfg_str(cfg, "maai_test_turn_vap_local_model", "") or "")
        if _tv_local:
            out["turn_vap_local_model"] = _tv_local
    return out


def build_worker_configs(
    cfg: dict,
    tts_engine: str,
    *,
    audio_output_mode: str | None = None,
    assume_mmd_running: bool = False,
    force_browser_mic: bool = False,
) -> dict[str, object]:
    """`controller.py:_build_worker_configs` 相当。

    Controller と同じく **各ビルダーを個別に try/except して失敗時は `{}`** にする。
    1 つのエンジン設定が不正でも音声対話全体の起動を止めないため。
    `audio_output_mode` / `assume_mmd_running` は Controller が MMDAgent の起動状態を
    渡すための口（既定は headless 相当＝MMD 未起動とみなす）。
    """
    def _safe(fn):
        try:
            return fn()
        except Exception:
            return {}

    return {
        "asr": _safe(lambda: build_asr_config_dict(cfg)),
        "tts": _safe(lambda: build_tts_worker_config(cfg, tts_engine)),
        "llm": {
            "connection": _safe(lambda: active_llm_connection_settings(cfg)),
        },
        "maai": _safe(lambda: build_maai_worker_config(cfg)),
        "audio_input": _safe(lambda: build_audio_input_worker_config(
            cfg, force_browser_mic=force_browser_mic,
        )),
        "audio_output": _safe(lambda: build_audio_output_worker_config_dict(
            cfg, output_mode=audio_output_mode, assume_mmd_running=assume_mmd_running,
        )),
    }


# ----- start_voice_dialog ペイロード -----------------------------------------

def build_orchestrator_tts_stream_inputs(cfg: dict) -> dict[str, object]:
    """`update_settings` 用の `tts_stream_inputs`（TEMPLATE は template 同梱）。

    もとは `controller.py:_build_orchestrator_tts_stream_inputs`。
    start_voice_dialog だけで送ると、実行中に TTS エンジンを切り替えたとき
    orchestrator 側の `tts_stream_inputs` が更新されず template が欠落し、
    相槌（maai_backchannel）の TEMPLATE 合成が失敗する。ライブ更新でも必ず同梱する。

    Controller と同じく **全体を try/except → `{}`** にする（テンプレート JSON が
    不正なら inputs 全体を空にする、という既存の挙動を保つ）。
    """
    engine = cfg_tts_engine(cfg, "")
    try:
        inputs = build_tts_inputs(
            cfg,
            engine=engine,
            utterance="",
            input_streaming_enabled=cfg_bool(cfg, "tts_input_streaming_enabled", False),
            split_settings=resolve_llm_split_settings(cfg),
        )
        if engine == "TEMPLATE":
            inputs["template"] = parse_tts_template_text(
                cfg_str(cfg, "tts_template", ""), strict=True
            )
        return inputs
    except Exception:
        return {}


def build_dialog_orchestrator_settings(cfg: dict) -> dict[str, object]:
    """`controller.py:_build_dialog_orchestrator_settings` 相当（GUI 非依存）。"""
    return {
        "system_prompt": cfg_str(cfg, "llm_system_prompt", ""),
        "maai_turn_threshold": _clamp(cfg_float(cfg, "asr_maai_turn_threshold", 0.5), 0.0, 1.0),
        "vad_on_threshold": cfg_float(cfg, "asr_vad_on_threshold", 0.1),
        "interrupt_vad_threshold": cfg_float(cfg, "voice_dialog_interrupt_vad_threshold", 0.0),
        "interrupt_p_now_threshold": cfg_float(cfg, "voice_dialog_interrupt_p_now_threshold", 0.0),
        "interrupt_on_user_speech": cfg_bool(cfg, "voice_dialog_interrupt_on_user_speech", False),
        "maai_backchannel_enabled": cfg_bool(cfg, "maai_backchannel_enabled", False),
        "maai_backchannel_prob_threshold": _clamp(
            cfg_float(cfg, "maai_backchannel_prob_threshold", 0.5), 0.0, 1.0
        ),
        "maai_backchannel_min_interval_sec": max(
            0.0, cfg_float(cfg, "maai_backchannel_min_interval_sec", 1.0)
        ),
        "maai_backchannel_consecutive_frames": max(1, cfg_int(cfg, "maai_backchannel_consecutive_frames", 1)),
        "maai_backchannel_dictionary_path": _resolve_backchannel_dictionary(cfg),
        # 相槌再生コンボ:
        # - silent: 合成はするがスピーカー/MMDへ出さず system_audio として MaAI 入力へだけ流す
        # - play_no_inject: スピーカー/MMD へは出すが MaAI 入力へは流さない
        # （controller 側と同じキー名・意味に揃える）。
        "maai_backchannel_inject_only": cfg_str(cfg, "maai_backchannel_playback", "play").lower() == "silent",
        "maai_backchannel_no_inject": (
            cfg_str(cfg, "maai_backchannel_playback", "play").lower() == "play_no_inject"
        ),
        "turn_backend": cfg_str(cfg, "asr_turn_backend", "maai").lower() or "maai",
        "asr_nonstream_pseudo_enabled": cfg_bool(cfg, "asr_nonstream_pseudo_enabled", True),
        "asr_engine": cfg_str(cfg, "asr_engine", "sherpa-onnx").lower() or "sherpa-onnx",
        "tts_engine": cfg_tts_engine(cfg),
        "llm_backend": cfg_str(cfg, "llm_backend", "LM Studio") or "LM Studio",
        "llm_model": llm_model_for_backend(cfg, cfg_str(cfg, "llm_backend", "LM Studio") or "LM Studio"),
        # Controller は欄が空なら空文字をそのまま送る（下流の
        # `str(x).strip() or "10"` / `int(x or 128)` が既定に落とすので等価）。
        # `or "10"` を付けると Controller と出力が食い違うため素通しにする。
        "llm_context_len": cfg_str(cfg, "llm_context_len", "10"),
        "llm_max_tokens": cfg_str(cfg, "llm_max_tokens", "128"),
        # 以下は Controller だけが送っていた（headless に欠けていた）。
        "nod_para_range_clip_min": cfg_float(cfg, "nod_para_range_clip_min", 0.0),
        "nod_para_range_clip_max": cfg_float(cfg, "nod_para_range_clip_max", 1.0),
        "nod_para_speed_clip_min": cfg_float(cfg, "nod_para_speed_clip_min", 0.0),
        "nod_para_speed_clip_max": cfg_float(cfg, "nod_para_speed_clip_max", 1.0),
        "tts_stream_inputs": build_orchestrator_tts_stream_inputs(cfg),
        "maai_plot_audio_enabled": cfg_bool(cfg, "asr_maai_gui_plot", False),
        # nod_para
        "nod_para_enabled": cfg_bool(cfg, "nod_para_enabled", False),
        "nod_para_prob_threshold": cfg_float(cfg, "nod_para_prob_threshold", 0.6),
        "nod_para_min_consecutive_frames": cfg_int(cfg, "nod_para_min_consecutive_frames", 3),
        "nod_para_min_interval_ms": cfg_float(cfg, "nod_para_min_interval_ms", 1500.0),
        "nod_para_swing_up_threshold": cfg_float(cfg, "maai_test_nod_swing_threshold", 0.5),
        "nod_para_count_weight_1": cfg_float(cfg, "maai_test_nod_count_t0", 1.0),
        "nod_para_count_weight_2": cfg_float(cfg, "maai_test_nod_count_t1", 1.0),
        "nod_para_count_weight_3": cfg_float(cfg, "maai_test_nod_count_t2", 1.0),
        "nod_para_decay_rate": cfg_float(cfg, "nod_para_decay_rate", 0.6),
        "nod_para_fps": cfg_int(cfg, "nod_para_fps", 30),
        "nod_para_interpolation": cfg_str(cfg, "nod_para_interpolation", "cosine") or "cosine",
        "nod_para_pre_rise_ratio": cfg_float(cfg, "nod_para_pre_rise_ratio", 0.8),
        "nod_para_scale": cfg_float(cfg, "nod_para_scale", 1.0),
        "nod_para_speed_scale": cfg_float(cfg, "nod_para_speed_scale", 1.0),
        "nod_para_eye_scale": cfg_float(cfg, "nod_para_eye_scale", 1.0),
        "nod_para_bone": cfg_str(cfg, "nod_para_bone", "\u982d") or "\u982d",
        "mmd_burst_snd_frames": int(_clamp(cfg_int(cfg, "mmd_burst_snd_frames", 20), 0, 500)),
        "mmd_blink_enable": cfg_bool(cfg, "mmd_blink_enable", True),
        "mmd_blink_interval_mean_sec": _cfg_blink_interval_mean_sec(cfg),
        "mmd_blink_morph": _cfg_blink_morph(cfg),
        "mmd_blink_mode": _cfg_blink_mode(cfg),
        "voice_dialog_opening_utterance": cfg_str(cfg, "voice_dialog_opening_utterance", ""),
        "voice_dialog_scheduled_exit_sec": max(0.0, cfg_float(cfg, "voice_dialog_scheduled_exit_sec", 0.0)),
        "voice_dialog_scheduled_exit_utterance": cfg_str(cfg, "voice_dialog_scheduled_exit_utterance", ""),
        "voice_dialog_idle_take_turn_sec": max(0.0, cfg_float(cfg, "voice_dialog_idle_take_turn_sec", 0.0)),
        "audio_input_mode": cfg_str(cfg, "audio_input_mode", "mic").lower() or "mic",
        "wav_start_sec": cfg_float(cfg, "audio_input_wav_start_sec", 0.0),
        "wav_send_other_ch_to_mmd": (
            cfg_bool(cfg, "audio_input_wav_other_ch_to_mmd", False)
            and cfg_str(cfg, "audio_input_mode", "mic").lower() == "wav"
        ),
    }


def parse_tts_template_text(raw: str, *, strict: bool = False) -> dict[str, object]:
    """TTS テンプレート JSON 文字列を dict にする。GUI / headless 共通の解析。

    エラー方針だけが両者で違うのでパラメータにしている:
    - `strict=True` （GUI）: 不正な JSON / 非 object は `RuntimeError`。
      ユーザーがテンプレートを直せるようにメッセージで知らせる。
    - `strict=False`（headless）: `{}` を返して起動を止めない。
    """
    import json as _json

    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        data = _json.loads(text)
    except Exception as ex:
        if strict:
            raise RuntimeError(f"TTSテンプレートJSONが不正です: {ex}") from ex
        return {}
    if not isinstance(data, dict):
        if strict:
            raise RuntimeError("TTSテンプレートJSONの最上位は object である必要があります")
        return {}
    return data


def load_tts_template_dict_from_cfg(cfg: dict) -> dict[str, object]:
    """controller の `_load_tts_template_dict` 相当（GUI非依存）。

    cfg["tts_template"]（GUIが保存した JSON 文字列）を優先し、空なら tts_template.json を読む。
    解析できない場合は空 dict を返す（headless 起動を止めない）。
    """
    raw = str(cfg.get("tts_template", "") or "").strip()
    if not raw:
        try:
            raw = _paths.tts_template_path().read_text(encoding="utf-8").strip()
        except Exception:
            raw = ""
    return parse_tts_template_text(raw, strict=False)


def build_start_voice_dialog_payload(
    cfg: dict,
    *,
    tts_engine: str | None = None,
    assume_mmd_running: bool = False,
    force_browser_mic: bool = False,
) -> dict[str, object]:
    """controller_config.json の dict から `start_voice_dialog` の `settings` を組み立てる。

    GUI/CLI から共通で使えるエントリ。`tts_engine` を None にすると cfg から取得。
    """
    engine = normalize_tts_engine(tts_engine or cfg_tts_engine(cfg))
    settings = build_dialog_orchestrator_settings(dict(cfg))
    settings["worker_configs"] = build_worker_configs(
        dict(cfg), engine, assume_mmd_running=assume_mmd_running,
        force_browser_mic=force_browser_mic,
    )
    settings["llm_connection"] = active_llm_connection_settings(dict(cfg))
    try:
        settings["tts_stream_inputs"] = build_tts_inputs(
            cfg,
            engine=engine,
            utterance="",
            input_streaming_enabled=cfg_bool(cfg, "tts_input_streaming_enabled", False),
            split_settings=resolve_llm_split_settings(cfg),
        )
    except Exception:
        settings["tts_stream_inputs"] = {}
    # TEMPLATE TTS は template 設定を同梱しないと相槌等の合成が失敗する（controller と同じ）。
    if engine == "TEMPLATE":
        tpl = load_tts_template_dict_from_cfg(cfg)
        if isinstance(settings.get("tts_stream_inputs"), dict) and tpl:
            settings["tts_stream_inputs"]["template"] = tpl
    settings["voice_dialog_inject_ch2"] = True
    return settings


# ----- Web UI のモジュール開始/停止 ------------------------------------------

_WEB_WORKER_TTS = {
    "Qwen3-TTS-CPP",
    "LFM2.5-Audio",
    "Kokoro-RealtimeTTS",
    "Cartesia-Realtime",
    "VOICEVOX-Core",
    "AivisSpeech",
    "Supertonic-3",
    "Irodori-v4",
    "Tsukasa",
}


def _web_llm_warmup_command(cfg: dict) -> dict[str, object]:
    backend = cfg_str(cfg, "llm_backend", "LM Studio") or "LM Studio"
    conn = active_llm_connection_settings(cfg)
    cmd: dict[str, object] = {
        "type": "start_llm_warmup",
        "backend": backend,
        "host": conn.get("host", ""),
        "model": llm_model_for_backend(cfg, backend),
        "api_key": conn.get("api_key", ""),
        "api_key_env": conn.get("api_key_env", ""),
        "force": True,
    }
    if backend == "Custom":
        cmd["send_temperature"] = cfg_bool(cfg, "llm_custom_send_temperature", True)
        cmd["send_max_tokens"] = cfg_bool(cfg, "llm_custom_send_max_tokens", True)
        cmd["temperature"] = cfg_float(cfg, "llm_custom_temperature", 0.7)
        cmd["custom_extra_body_thinking"] = cfg_bool(cfg, "llm_custom_extra_body_thinking", False)
    return cmd


def _web_llm_custom_fields(cfg: dict) -> dict[str, object]:
    return {
        "send_temperature": cfg_bool(cfg, "llm_custom_send_temperature", True),
        "send_max_tokens": cfg_bool(cfg, "llm_custom_send_max_tokens", True),
        "temperature": cfg_float(cfg, "llm_custom_temperature", 0.7),
        "custom_extra_body_thinking": cfg_bool(cfg, "llm_custom_extra_body_thinking", False),
    }


def _web_start_module_commands(
    cfg: dict,
    module: str,
    *,
    engine: str | None = None,
    assume_mmd_running: bool = True,
    force_browser_mic: bool = True,
) -> list[dict[str, object]]:
    requested_engine = (engine or "").strip()
    tts_engine = normalize_tts_engine(
        requested_engine
        if module == "tts" and requested_engine
        else cfg_tts_engine(cfg)
    )
    worker_cfg = dict(cfg)
    if module == "asr" and requested_engine:
        worker_cfg["asr_engine"] = requested_engine
    workers = build_worker_configs(
        worker_cfg,
        tts_engine,
        assume_mmd_running=assume_mmd_running,
        force_browser_mic=force_browser_mic,
    )
    audio = workers.get("audio_input") if isinstance(workers.get("audio_input"), dict) else {}
    if module == "asr":
        asr_cfg = dict(workers.get("asr")) if isinstance(workers.get("asr"), dict) else {}
        asr_module = "asr"
        if cfg_bool(cfg, "asr_compare_mode_enabled", False):
            # PySide の start_asr と同じく、比較モードでは比較ワーカー（asr_compare）を起動する。
            top = cfg_str(cfg, "asr_compare_top_engine", "") or ""
            bottom = cfg_str(cfg, "asr_compare_bottom_engine", "") or ""
            if top == bottom:
                raise ValueError("比較モードでは上段/下段で異なるエンジンを選択してください。")
            asr_cfg.update(
                compare_mode_enabled=True,
                compare_top_engine=top,
                compare_bottom_engine=bottom,
            )
            asr_module = "asr_compare"
        cmds: list[dict[str, object]] = [
            {
                "type": "start_module",
                "module": asr_module,
                "config": asr_cfg,
                "source": "web",
            }
        ]
        if audio:
            cmds.append({"type": "start_module", "module": "audio_input", "config": audio, "source": "web"})
        return cmds
    if module == "llm":
        return [_web_llm_warmup_command(cfg)]
    if module in {"audio_input", "audio-input"}:
        return [{"type": "start_module", "module": "audio_input", "config": audio, "source": "web"}]
    if module == "tts":
        engine = tts_engine
        try:
            tts_cfg = build_tts_worker_config(cfg, engine)
        except Exception:
            tts_cfg = {}
        cmds: list[dict[str, object]] = [
            {
                "type": "start_module",
                "module": "tts",
                "config": tts_cfg,
                "engine": engine,
                "source": "web",
            }
        ]
        if engine == "VOICEVOX":
            cmds[0]["host"] = _loopback_host(cfg, "tts_host", "127.0.0.1") or "127.0.0.1"
            cmds[0]["port"] = int(_clamp(cfg_int(cfg, "tts_port", 50021), 1, 65535)) or 50021
            cmds[0]["voicevox_exe_path"] = cfg_str(cfg, "voicevox_exe_path", "")
        elif engine == "AivisSpeech":
            # PySide の「起動」ボタン（launch_aivisspeech）と同じく Engine 本体も起動する。
            # ワーカーは HTTP クライアントなので、Engine が無いと合成できない。
            cmds[:0] = _web_tts_bridge_commands(
                cfg,
                "launch_aivisspeech",
                host=_loopback_host(cfg, "aivisspeech_host", "127.0.0.1") or "127.0.0.1",
                port=int(_clamp(cfg_int(cfg, "aivisspeech_port", 10104), 1, 65535)) or 10104,
                use_gpu=True,
            )
        if tts_cfg and engine in _WEB_WORKER_TTS:
            try:
                inputs = build_tts_inputs(
                    cfg,
                    engine=engine,
                    utterance="ウォームアップです。",
                    input_streaming_enabled=False,
                    split_settings=resolve_llm_split_settings(cfg),
                )
            except Exception:
                inputs = {}
            cmds.append(
                {
                    "type": "start_tts_worker_warmup",
                    "engine": engine,
                    "tts_worker_config": tts_cfg,
                    "inputs": inputs,
                    "timeout_sec": 180.0 if engine == "Irodori-v4" else 90.0,
                }
            )
        return cmds
    if module == "maai":
        cmds = [
            {
                "type": "start_module",
                "module": "maai",
                "config": workers.get("maai") if isinstance(workers.get("maai"), dict) else {},
                "source": "web",
            }
        ]
        if audio:
            cmds.append({"type": "start_module", "module": "audio_input", "config": audio, "source": "web"})
        return cmds
    return [{"type": "start_module", "module": module, "config": {}, "source": "web"}]


def _web_tts_bridge_commands(cfg: dict, op: str, **kwargs: object) -> list[dict[str, object]]:
    """PySide の `_tts_bridge_rpc` と同じ: exe パスを渡してから op を送る。"""
    import time

    return [
        {
            "type": "send_to_tts_bridge",
            "payload": {
                "op": "configure",
                "voicevox_exe_path": cfg_str(cfg, "voicevox_exe_path", ""),
                "aivisspeech_exe_path": cfg_str(cfg, "aivisspeech_exe_path", ""),
            },
        },
        {
            "type": "send_to_tts_bridge",
            "payload": {"op": op, "request_id": f"web-{op}-{time.time_ns()}", **kwargs},
        },
    ]


def _web_stop_module_commands(
    cfg: dict, module: str, *, engine: str | None = None
) -> list[dict[str, object]]:
    cmds: list[dict[str, object]] = [{"type": "stop_module", "module": module}]
    if module == "tts":
        # PySide の停止ボタン（stop_voicevox / stop_aivisspeech）と同じく、
        # 外部プロセスのエンジンはワーカーと一緒に止める。
        tts_engine = normalize_tts_engine(engine or cfg_tts_engine(cfg))
        if tts_engine == "VOICEVOX":
            cmds.extend(_web_tts_bridge_commands(cfg, "stop_voicevox"))
        elif tts_engine == "AivisSpeech":
            cmds.extend(_web_tts_bridge_commands(cfg, "stop_aivisspeech"))
    if module == "asr":
        cmds.append({"type": "stop_module", "module": "asr_compare"})
    elif module == "llm":
        backend = cfg_str(cfg, "llm_backend", "LM Studio") or "LM Studio"
        if backend == "LM Studio":
            import time

            conn = active_llm_connection_settings(cfg)
            cmds.append(
                {
                    "type": "send_to_llm_admin",
                    "payload": {
                        "op": "lmstudio_bulk_unload",
                        "request_id": f"web-llm-stop-{time.time_ns()}",
                        "host": conn.get("host", ""),
                        "current_model": llm_model_for_backend(cfg, backend),
                        "unload_all": True,
                        "silent": True,
                        "reason": "module_stop",
                    },
                }
            )
    return cmds


def _web_tts_gui_test(
    cfg: dict,
    payload: dict,
    *,
    assume_mmd_running: bool = True,
    force_browser_mic: bool = True,
) -> list[dict[str, object]]:
    import time

    engine = normalize_tts_engine(payload.get("engine") or cfg_tts_engine(cfg))
    utterance = str(payload.get("utterance") or cfg_str(cfg, "tts_test_text", "") or "")
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else None
    if not inputs:
        try:
            inputs = build_tts_inputs(
                cfg,
                engine=engine,
                utterance=utterance,
                input_streaming_enabled=False,
                split_settings=resolve_llm_split_settings(cfg),
            )
        except Exception:
            inputs = {"engine": engine, "text": utterance}
    orch: dict[str, object] = {
        "type": "start_tts_gui_test",
        "engine": engine,
        "utterance": utterance,
        "inputs": inputs,
        "voice_dialog_inject_ch2": True,
        "mmd_running": bool(assume_mmd_running),
        "started_at": float(payload.get("started_at") or time.perf_counter()),
    }
    if engine == "TEMPLATE":
        tpl = load_tts_template_dict_from_cfg(cfg)
        if tpl:
            orch["template"] = tpl
    try:
        tts_cfg = build_tts_worker_config(cfg, engine)
    except Exception:
        tts_cfg = {}
    if tts_cfg:
        orch["tts_worker_config"] = tts_cfg
    cmds = _web_start_module_commands(
        cfg,
        "tts",
        engine=engine,
        assume_mmd_running=assume_mmd_running,
        force_browser_mic=force_browser_mic,
    )
    cmds.append(orch)
    return cmds


def _web_llm_gui_test(
    cfg: dict,
    payload: dict,
    *,
    assume_mmd_running: bool = True,
    force_browser_mic: bool = True,
) -> list[dict[str, object]]:
    backend = str(payload.get("backend") or cfg_str(cfg, "llm_backend", "LM Studio") or "LM Studio").strip()
    conn = active_llm_connection_settings(cfg)
    host = str(payload.get("host") or conn.get("host") or "")
    if backend == "LM Studio" or (backend == "Custom" and host):
        host = _normalize_compat_llm_host(host)
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    engine = cfg_tts_engine(cfg)
    tts_stream_inputs: dict[str, object] = {}
    if cfg_bool(cfg, "llm_tts_stream_enabled", False):
        try:
            tts_stream_inputs = build_tts_inputs(
                cfg,
                engine=engine,
                utterance="",
                input_streaming_enabled=cfg_bool(cfg, "llm_test_tts_input_streaming_enabled", True),
                split_settings=resolve_llm_split_settings(cfg),
            )
        except Exception:
            tts_stream_inputs = {}
        if engine == "TEMPLATE" and isinstance(tts_stream_inputs, dict):
            tpl = load_tts_template_dict_from_cfg(cfg)
            if tpl:
                tts_stream_inputs["template"] = tpl
    orch: dict[str, object] = {
        "type": "start_llm_gui_test",
        "backend": backend,
        "model": str(payload.get("model") or llm_model_for_backend(cfg, backend) or ""),
        "context_len": int(payload.get("context_len") or cfg_int(cfg, "llm_context_len", 10)),
        "max_tokens": int(payload.get("max_tokens") or cfg_int(cfg, "llm_max_tokens", 128)),
        "stream": True,
        "messages": [dict(m) for m in messages if isinstance(m, dict)],
        "host": host,
        "api_key": conn.get("api_key", ""),
        "api_key_env": conn.get("api_key_env", ""),
        "tts_stream_engine": engine,
        "tts_stream_inputs": tts_stream_inputs,
        "voice_dialog_inject_ch2": True,
        "llm_custom": _web_llm_custom_fields(cfg) if backend == "Custom" else {},
    }
    cmds: list[dict[str, object]] = [_web_llm_warmup_command(cfg)]
    if tts_stream_inputs:
        cmds.extend(
            _web_start_module_commands(
                cfg,
                "tts",
                engine=engine,
                assume_mmd_running=assume_mmd_running,
                force_browser_mic=force_browser_mic,
            )
        )
    cmds.append(orch)
    return cmds


def expand_web_ui_commands(
    cfg: dict,
    payload: dict,
    *,
    assume_mmd_running: bool = True,
    force_browser_mic: bool = True,
) -> list[dict[str, object]]:
    """ブラウザの /api/command を Orchestrator 向けコマンド列に展開する。"""
    kind = str(payload.get("type", "")).strip()
    if kind == "start_voice_dialog":
        out = dict(payload)
        out["settings"] = build_start_voice_dialog_payload(
            cfg,
            tts_engine=cfg.get("tts_engine"),
            assume_mmd_running=assume_mmd_running,
            force_browser_mic=force_browser_mic,
        )
        return [out]
    if kind in {"start_module", "module_start"}:
        engine = str(payload.get("engine") or "").strip() or None
        return _web_start_module_commands(
            cfg,
            str(payload.get("module", "")).strip().lower(),
            engine=engine,
            assume_mmd_running=assume_mmd_running,
            force_browser_mic=force_browser_mic,
        )
    if kind in {"stop_module", "module_stop"}:
        return _web_stop_module_commands(
            cfg,
            str(payload.get("module", "")).strip().lower(),
            engine=str(payload.get("engine") or "").strip() or None,
        )
    if kind == "stop_voice_dialog":
        # PySide の _stop_voice_dialog と同じく、停止の直後に中断扱いの tts_done を送る。
        # keep_asr / keep_maai（開始前から動いていたものは残す）はサーバが付ける。
        return [dict(payload), {"type": "tts_done", "playback_aborted": True}]
    if kind == "start_tts_gui_test":
        return _web_tts_gui_test(
            cfg,
            payload,
            assume_mmd_running=assume_mmd_running,
            force_browser_mic=force_browser_mic,
        )
    if kind == "start_llm_gui_test":
        return _web_llm_gui_test(
            cfg,
            payload,
            assume_mmd_running=assume_mmd_running,
            force_browser_mic=force_browser_mic,
        )
    if kind == "start_llm_warmup":
        cmd = _web_llm_warmup_command(cfg)
        cmd["force"] = bool(payload.get("force", True))
        return [cmd]
    if kind == "audio_input_send_command" and str(payload.get("command") or "").strip() == "wav_play":
        # PyQt の再生ボタンは未起動なら AudioInputWorker を起こしてから wav_play する。
        workers = build_worker_configs(
            dict(cfg), cfg_tts_engine(cfg),
            assume_mmd_running=assume_mmd_running,
            force_browser_mic=force_browser_mic,
        )
        audio = workers.get("audio_input") if isinstance(workers.get("audio_input"), dict) else {}
        cmds: list[dict[str, object]] = []
        if audio:
            cmds.append({"type": "start_module", "module": "audio_input", "config": audio, "source": "web"})
        cmds.append(dict(payload))
        return cmds
    return [dict(payload)]
