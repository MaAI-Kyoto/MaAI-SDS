"""audio.cpp で動かす TTS エンジン（LFM2.5-Audio・Irodori の audio.cpp 版）。

どちらも `synthesize`（1 文をまとめて返す）と `synthesize_stream`（届いた音から少しずつ返す）を持つ。
返す音声は、ほかのエンジンと同じ 16 kHz・16 bit・ビッグエンディアンの PCM。

実測（RTX 5070 Ti Laptop、awesome-japanese-speech。AC 電源・高パフォーマンス）：

| モデル | 1 文 | 最初の声（ストリーミング） | 読みの CER |
|---|---:|---:|---:|
| LFM2.5-Audio-1.5B-JP | 328 ms | 51 ms | 0.007 |
| Irodori-TTS 500M v3 | 351 ms | 263 ms | 0.004 |
| Irodori-TTS 600M v3 VoiceDesign | 366 ms | 280 ms | 0.004 |
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from .. import paths as _paths
from ..audio.convert import StreamingPcm16Resampler, pcm16le_to_pcm16be_16k, swap_pcm16_endian
from .audiocpp_client import AudioCppServer, AudioCppSpeechClient

LFM2_MODEL_DIR = "audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF"
IRODORI_AUDIOCPP_MODELS: dict[str, dict[str, str]] = {
    # 設定値 → audio.cpp のモデル id と GGUF。id は docker/audiocpp-server.json と同じにする（Docker では
    # compose の audiocpp サービスを共有するため）。
    "500m-v3": {
        "id": "irodori-tts-500m-v3",
        "path": "audio.cpp/models/Irodori-TTS-500M-v3-GGUF/irodori-tts-500m-v3-q8_0.gguf",
        "label": "Irodori-TTS 500M v3",
    },
    "600m-v3-voicedesign": {
        "id": "irodori-tts-600m-v3-voicedesign",
        "path": "audio.cpp/models/Irodori-TTS-600M-v3-VoiceDesign-GGUF/irodori-tts-600m-v3-voicedesign-q8_0.gguf",
        "label": "Irodori-TTS 600M v3 VoiceDesign（声を文章で指定）",
        "caption": "1",
    },
}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _AudioCppStreamingEngine:
    """audio.cpp の 1 モデルを合成するエンジンの共通部分。子クラスは `_body` と `_voice_label` を決める。"""

    family = ""
    fallback_rate = 24000
    name = "audiocpp"

    def __init__(self, cfg: dict[str, Any], *, model_id: str, model_path: str, default_port: int) -> None:
        self.model_id = model_id
        self.model_path = _paths.resolve_under_repo(model_path)
        self.server = AudioCppServer(
            host=str(cfg.get("host", "127.0.0.1") or "127.0.0.1"),
            port=int(cfg.get("port", default_port) or default_port),
            models=[{"id": self.model_id, "family": self.family, "path": str(self.model_path)}],
            server_exe=str(cfg.get("server_exe", "") or ""),
            # "backend" は Irodori の動かし方（speechcpp / audiocpp / pytorch）に使っているので、audio.cpp の
            # 実行デバイスは別の名前にする（cuda / cpu / vulkan）。空なら置いてあるサーバーの種類から決め、
            # スレッド数も 0 なら実行デバイスから決める（以前は cuda・8 固定で、4 コアの PC では合成が止まった）。
            backend=str(cfg.get("audiocpp_backend", "") or ""),
            threads=int(cfg.get("audiocpp_threads", 0) or 0),
            name=self.name,
        )
        self.client = AudioCppSpeechClient(self.server)
        self.sample_rate = 0

    # 子クラスが決める
    def _body(self, req: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _voice_label(self, req: dict[str, Any]) -> str:
        return ""

    def load(self) -> None:
        started = self.server.ensure()
        # サンプリング周波数はヘッダーに無いので、短い文で 1 回合成して求める（GPU のウォームアップも兼ねる）
        self.sample_rate = self.client.probe_sample_rate(self._body({"text": "こんにちは。"}), self.fallback_rate)
        _log(
            f"[{self.name}] ready: {self.model_id} at {self.server.base_url}"
            f"{' (started)' if started else ''} rate={self.sample_rate}"
        )

    def synthesize(self, req: dict[str, Any]) -> tuple[bytes, str]:
        if not str(req.get("text", "")).strip():
            raise ValueError("'text' must not be empty")
        pcm, _duration = self.client.speech(self._body(req))
        if not pcm:
            raise RuntimeError(f"{self.name}: 音声が空でした")
        return pcm16le_to_pcm16be_16k(pcm, self.sample_rate or self.fallback_rate), self._voice_label(req)

    def synthesize_stream(self, req: dict[str, Any], on_chunk: Callable[[bytes], None]) -> tuple[int, str]:
        if not str(req.get("text", "")).strip():
            raise ValueError("'text' must not be empty")
        resampler = StreamingPcm16Resampler(self.sample_rate or self.fallback_rate, 16000)
        total = [0]

        def _emit(raw_le: bytes) -> None:
            out = resampler.process(raw_le)
            if out:
                total[0] += len(out)
                on_chunk(swap_pcm16_endian(out))

        self.client.speech(self._body(req), on_chunk=_emit)
        if total[0] <= 0:
            raise RuntimeError(f"{self.name}: 音声が空でした")
        return total[0], self._voice_label(req)

    def close(self) -> None:
        self.server.close()


class Lfm2AudioEngine(_AudioCppStreamingEngine):
    """LFM2.5-Audio-1.5B-JP（Liquid AI）の TTS。声は選べない（モデルの声 1 つ）。

    ライセンスは LFM Open License v1.0（年商 1000 万ドル未満なら商用可）。
    """

    family = "lfm2_audio"
    fallback_rate = 24000
    name = "lfm2-audio"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(
            cfg,
            model_id=str(cfg.get("model_id", "lfm2.5-audio-jp") or "lfm2.5-audio-jp"),
            model_path=str(cfg.get("model_path", "") or LFM2_MODEL_DIR),
            default_port=18082,
        )
        self.language = str(cfg.get("language", "ja") or "ja")

    def _body(self, req: dict[str, Any]) -> dict[str, Any]:
        return {"model": self.model_id, "input": str(req.get("text", "")).strip(), "language": self.language}

    def _voice_label(self, req: dict[str, Any]) -> str:
        return "LFM2.5-Audio"


class IrodoriAudioCppEngine(_AudioCppStreamingEngine):
    """Irodori-TTS の audio.cpp 版（500M v3 / 600M v3 VoiceDesign）。`irodori_backend` = audiocpp。

    - 声：参照 WAV があればそれを真似る（`voice_ref`）。600M VoiceDesign はキャプション（`voice` の文）で声を指定する。
    - seed を固定する（参照 WAV もキャプションも無いときの声は seed で決まる。speech.cpp 版と同じ理由）。
    """

    family = "irodori_tts"
    fallback_rate = 48000
    name = "irodori-audiocpp"

    def __init__(self, cfg: dict[str, Any]) -> None:
        key = str(cfg.get("audiocpp_model", "500m-v3") or "500m-v3").strip()
        spec = IRODORI_AUDIOCPP_MODELS.get(key) or IRODORI_AUDIOCPP_MODELS["500m-v3"]
        super().__init__(
            cfg,
            model_id=spec["id"],
            model_path=str(cfg.get("audiocpp_model_path", "") or spec["path"]),
            default_port=int(cfg.get("audiocpp_port", 18083) or 18083),
        )
        self.uses_caption = bool(spec.get("caption"))
        self.seed = int(cfg.get("seed", 1) or 0)
        self.default_caption = str(cfg.get("voice", "") or "").strip()
        raw_ref = str(cfg.get("reference_audio", "") or "").strip()
        ref = _paths.resolve_docker_local_file(raw_ref) if raw_ref else ""
        if ref and not Path(ref).is_absolute():
            ref = str(_paths.resolve_under_repo(ref))
        self.reference_audio = ref

    def _caption(self, req: dict[str, Any]) -> str:
        return str(req.get("voice", "") or "").strip() or self.default_caption

    def _body(self, req: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "input": str(req.get("text", "")).strip(),
            "language": "ja",
            "seed": self.seed,
        }
        if self.reference_audio:
            body["voice_ref"] = self.reference_audio
        elif self.uses_caption:
            caption = self._caption(req)
            if caption:
                body["caption"] = caption
        return body

    def _voice_label(self, req: dict[str, Any]) -> str:
        if self.reference_audio:
            return "ref"
        return self._caption(req) if self.uses_caption else "default"
