"""chirp3-asr（Google Cloud Speech-to-Text V2）のストリーミング認識。

`asr/service.ASRService` から**そのまま移した**もの（mixin）。
本文は 1 文字も変えていない。エンジンごとにファイルを分けたのは、
1 エンジンを読む/直すときに他のエンジンを読み飛ばさなくて済むようにするため。
"""

from __future__ import annotations

import json
import os
import queue
import time
from pathlib import Path

import numpy as np


class AsrChirp3EngineMixin:
    """chirp3-asr（Google Cloud Speech-to-Text V2）のストリーミング認識。 `ASRService` に mixin する。"""

    def _init_engine_chirp3(self) -> None:
        if not self.chirp_project_id:
            raise RuntimeError(
                "Chirp3 ASR には Google Cloud Project ID が必要です。"
                " ASR設定の Chirp Project ID か環境変数 GOOGLE_CLOUD_PROJECT を設定してください。"
            )
        self._faster_whisper_model = None
        self._sherpa_recognizer = None

    def _chirp3_stream_loop(self) -> None:
        attempt = 0
        self._log("Chirp3 streaming loop started")
        while not self._stop_event.is_set():
            attempt += 1
            try:
                self._log(f"Chirp3 streaming attempt={attempt} begin")
                self._chirp3_streaming_recognize_once()
                return
            except Exception as ex:
                self._error(f"Chirp3 ASR streaming failed (attempt={attempt}): {ex}")
                if self._stop_event.is_set():
                    return
                time.sleep(min(2.0, 0.2 * max(1, attempt)))

    def _chirp3_streaming_recognize_once(self) -> None:
        self._log("Chirp3: importing google.cloud.speech_v2")
        try:
            from google.cloud.speech_v2 import SpeechClient  # type: ignore
            from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types  # type: ignore
            from google.api_core.client_options import ClientOptions  # type: ignore
            from google.oauth2 import service_account as google_service_account  # type: ignore
            from google.oauth2 import credentials as google_user_credentials  # type: ignore
        except Exception as ex:
            raise RuntimeError(
                "google-cloud-speech が見つかりません。`pip install google-cloud-speech` を実行してください。"
            ) from ex

        def _load_explicit_credentials():
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            cred_path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
            candidates: list[Path] = []
            if cred_path:
                candidates.append(Path(cred_path))
            adc_path = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
            if adc_path.exists():
                candidates.append(adc_path)
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    cred_type = str(raw.get("type", "")).strip()
                except Exception:
                    continue
                try:
                    if cred_type == "service_account":
                        self._log(f"Chirp3: loading service account credentials from {path}")
                        return google_service_account.Credentials.from_service_account_file(
                            str(path), scopes=scopes
                        )
                    if cred_type == "authorized_user":
                        self._log(f"Chirp3: loading ADC user credentials from {path}")
                        return google_user_credentials.Credentials.from_authorized_user_file(
                            str(path), scopes=scopes
                        )
                except Exception as ex:
                    self._log(f"Chirp3: credential load failed ({path}): {ex}")
            return None

        self._log("Chirp3: creating SpeechClient")
        explicit_credentials = _load_explicit_credentials()
        location_normalized = self.chirp_location.strip().lower() or "global"
        client_kwargs = {}
        if explicit_credentials is not None:
            client_kwargs["credentials"] = explicit_credentials
        if location_normalized != "global":
            endpoint = f"{location_normalized}-speech.googleapis.com"
            client_kwargs["client_options"] = ClientOptions(api_endpoint=endpoint)
            self._log(f"Chirp3: using regional endpoint={endpoint}")
        else:
            self._log("Chirp3: using global endpoint")
        client = SpeechClient(**client_kwargs)
        recognizer = (
            f"projects/{self.chirp_project_id}/locations/{self.chirp_location}/recognizers/{self.chirp_recognizer}"
        )
        language_codes = [
            code.strip() for code in self.chirp_language_codes.split(",") if code and code.strip()
        ]
        if not language_codes:
            language_codes = ["ja-JP"]
        if self.chirp_alternative_language_code and self.chirp_alternative_language_code not in language_codes:
            language_codes.append(self.chirp_alternative_language_code)

        recognition_config = cloud_speech_types.RecognitionConfig(
            explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
            ),
            language_codes=language_codes,
            model=self.chirp_model,
        )
        streaming_config = cloud_speech_types.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech_types.StreamingRecognitionFeatures(interim_results=True),
        )
        config_request = cloud_speech_types.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=streaming_config,
        )

        chunk_samples = max(1, int(self.sample_rate * 0.1))  # 公式推奨: 100ms
        buffered = np.zeros(0, dtype=np.float32)
        buffered_start_sample: int | None = None
        last_sent_audio_end_ts: float | None = None
        sent_chunks = 0
        recv_results = 0
        got_first_audio_chunk = False
        last_wait_log_ts = time.time()
        self._log(
            "Chirp3 streaming start: "
            f"recognizer={recognizer} model={self.chirp_model} languages={','.join(language_codes)}"
        )

        def _audio_requests():
            nonlocal buffered, buffered_start_sample, last_sent_audio_end_ts, sent_chunks, got_first_audio_chunk, last_wait_log_ts
            while not self._stop_event.is_set():
                try:
                    frame = self.raw_audio_queue.get(timeout=0.1)
                except queue.Empty:
                    now = time.time()
                    if (now - last_wait_log_ts) >= 5.0 and sent_chunks == 0:
                        self._log("Chirp3 streaming: waiting microphone audio...")
                        last_wait_log_ts = now
                    continue
                frame = self._to_audio_frame(frame)
                chunk = frame.audio
                if not got_first_audio_chunk:
                    got_first_audio_chunk = True
                    self._log("Chirp3 streaming: first microphone chunk received")
                if self.turn_backend != "maai":
                    self._queue_put(self.turn_audio_queue, frame, policy=self.QUEUE_POLICY_DROP_OLDEST)
                if buffered_start_sample is None:
                    buffered_start_sample = int(frame.sample_start)
                buffered = np.concatenate([buffered, chunk.astype(np.float32)], axis=0)
                while buffered.shape[0] >= chunk_samples:
                    seg = buffered[:chunk_samples]
                    buffered = buffered[chunk_samples:]
                    seg_end_ts = (
                        self._sample_index_to_ts(buffered_start_sample + chunk_samples)
                        if buffered_start_sample is not None
                        else float(frame.ts_end)
                    )
                    if buffered_start_sample is not None:
                        buffered_start_sample += chunk_samples
                    if buffered.shape[0] == 0:
                        buffered_start_sample = None
                    sent_chunks += 1
                    if sent_chunks % 50 == 0:
                        self._log(f"Chirp3 streaming: sent_chunks={sent_chunks}")
                    last_sent_audio_end_ts = float(seg_end_ts)
                    yield cloud_speech_types.StreamingRecognizeRequest(
                        audio=self._to_pcm16le_bytes(seg)
                    )
            if buffered.shape[0] > 0:
                sent_chunks += 1
                flush_end_ts = (
                    self._sample_index_to_ts(buffered_start_sample + int(buffered.shape[0]))
                    if buffered_start_sample is not None
                    else None
                )
                if flush_end_ts is not None:
                    last_sent_audio_end_ts = float(flush_end_ts)
                yield cloud_speech_types.StreamingRecognizeRequest(
                    audio=self._to_pcm16le_bytes(buffered)
                )

        def _requests():
            yield config_request
            yield from _audio_requests()

        self._log("Chirp3: calling streaming_recognize")
        responses = client.streaming_recognize(requests=_requests())
        self._log("Chirp3: streaming_recognize call returned iterator")
        chirp_segment_start_ts: float | None = None
        for response in responses:
            if self._stop_event.is_set():
                return
            for result in getattr(response, "results", []):
                alts = getattr(result, "alternatives", [])
                if not alts:
                    continue
                text = str(alts[0].transcript).strip()
                if not text:
                    continue
                recv_results += 1
                if recv_results == 1:
                    self._log("Chirp3 streaming: first transcript received")
                is_final = bool(getattr(result, "is_final", False))
                now_ts = float(last_sent_audio_end_ts) if last_sent_audio_end_ts is not None else time.time()
                if chirp_segment_start_ts is None:
                    chirp_segment_start_ts = now_ts
                self._publish_result(
                    text=text,
                    end_ts=now_ts,
                    engine="chirp3-asr",
                    is_final=is_final,
                    is_streaming=True,
                    segment_start_ts=chirp_segment_start_ts,
                    segment_end_ts=now_ts,
                )
                if is_final:
                    chirp_segment_start_ts = None
