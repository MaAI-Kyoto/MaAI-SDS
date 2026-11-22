"""TTS を少しずつ受け取る部品のテスト（つなぎ目で途切れない変換・audio.cpp の受け取り・audio.cpp のエンジン）。

本物の audio.cpp は使わず、`/health` と `/v1/audio/speech` だけを持つ小さな HTTP サーバーで代える。
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from maai_sds.audio.convert import StreamingPcm16Resampler, resample_pcm16le
from maai_sds.tts.audiocpp_client import AudioCppServer, AudioCppSpeechClient
from maai_sds.tts.engines_audiocpp import IrodoriAudioCppEngine, Lfm2AudioEngine

RATE = 24000
AUDIO = (np.sin(2 * np.pi * 220 * np.arange(RATE) / RATE) * 8000).astype("<i2").tobytes()  # 1 秒


class _FakeAudioCpp(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def log_message(self, *_args) -> None:  # テストの出力を汚さない
        pass

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        if body.get("stream"):
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # 奇数バイトのチャンクも混ぜて送る
            pos = 0
            for size in (1, 999, 4001, 7, 20000):
                part = AUDIO[pos:pos + size]
                pos += size
                self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            rest = AUDIO[pos:]
            self.wfile.write(f"{len(rest):x}\r\n".encode() + rest + b"\r\n0\r\n\r\n")
        else:
            self.send_header("Content-Length", str(len(AUDIO)))
            self.send_header("X-AudioCPP-Audio-Duration-Ms", "1000.0")
            self.end_headers()
            self.wfile.write(AUDIO)


class _Server:
    def __enter__(self) -> "_Server":
        _FakeAudioCpp.bodies = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAudioCpp)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class ResamplerTest(unittest.TestCase):
    def test_streaming_matches_whole_buffer(self) -> None:
        rng = np.random.default_rng(0)
        for rate in (48000, 24000, 22050, 44100):
            x = (np.sin(2 * np.pi * 440 * np.arange(rate) / rate) * 12000).astype("<i2").tobytes()
            whole = np.frombuffer(resample_pcm16le(x, rate), "<i2")
            r = StreamingPcm16Resampler(rate)
            out = bytearray()
            pos = 0
            while pos < len(x):
                n = int(rng.integers(1, 3000))  # 奇数バイトも出る
                out += r.process(x[pos:pos + n])
                pos += n
            got = np.frombuffer(bytes(out), "<i2")
            self.assertEqual(got.size, whole.size, rate)
            self.assertLessEqual(int(np.max(np.abs(got.astype(int) - whole.astype(int)))), 2, rate)

    def test_same_rate_passes_through(self) -> None:
        r = StreamingPcm16Resampler(16000)
        self.assertEqual(r.process(b"\x01\x02\x03") + r.process(b"\x04"), b"\x01\x02\x03\x04")


class AudioCppClientTest(unittest.TestCase):
    def test_stream_chunks_are_even_and_complete(self) -> None:
        with _Server() as srv:
            server = AudioCppServer(host="127.0.0.1", port=srv.port, models=[])
            self.assertFalse(server.ensure())  # 既に応答があるので立ち上げない
            client = AudioCppSpeechClient(server)
            chunks: list[bytes] = []
            pcm, _ = client.speech({"model": "m", "input": "テスト"}, on_chunk=chunks.append)
            self.assertTrue(all(len(c) % 2 == 0 for c in chunks))
            self.assertEqual(b"".join(chunks), AUDIO)
            self.assertEqual(pcm, AUDIO)
            sent = _FakeAudioCpp.bodies[-1]
            self.assertEqual((sent["stream"], sent["stream_format"], sent["response_format"]), (True, "audio", "pcm"))

    def test_probe_sample_rate_from_duration(self) -> None:
        with _Server() as srv:
            client = AudioCppSpeechClient(AudioCppServer(host="127.0.0.1", port=srv.port, models=[]))
            self.assertEqual(client.probe_sample_rate({"model": "m", "input": "x"}, fallback=48000), RATE)


class AudioCppEngineTest(unittest.TestCase):
    def test_irodori_backend_key_does_not_leak_into_audiocpp_device(self) -> None:
        eng = IrodoriAudioCppEngine({"backend": "audiocpp", "audiocpp_model_path": "."})
        self.assertEqual(eng.server.backend, "cuda")

    def test_lfm2_stream_and_offline_give_16k_big_endian(self) -> None:
        with _Server() as srv:
            eng = Lfm2AudioEngine({"port": srv.port, "model_path": "."})
            eng.load()
            self.assertEqual(eng.sample_rate, RATE)
            pcm, voice = eng.synthesize({"text": "こんにちは"})
            self.assertEqual(len(pcm), 32000)  # 1 秒・16 kHz・16 bit
            chunks: list[bytes] = []
            total, _ = eng.synthesize_stream({"text": "こんにちは"}, chunks.append)
            self.assertEqual(total, sum(len(c) for c in chunks))
            # つなぎ目で途切れず、まとめた結果と一致する（線形補間の丸めの差 1〜2 だけ許す）
            got = np.frombuffer(b"".join(chunks), ">i2").astype(int)
            ref = np.frombuffer(pcm, ">i2").astype(int)
            self.assertEqual(got.size, ref.size)
            self.assertLessEqual(int(np.max(np.abs(got - ref))), 2)
            self.assertEqual(_FakeAudioCpp.bodies[-1]["language"], "ja")

    def test_irodori_caption_only_for_voicedesign_and_fixed_seed(self) -> None:
        with _Server() as srv:
            vd = IrodoriAudioCppEngine({"audiocpp_port": srv.port, "audiocpp_model": "600m-v3-voicedesign",
                                        "audiocpp_model_path": ".", "voice": "低い声", "seed": 7})
            vd.load()
            vd.synthesize({"text": "あ"})
            self.assertEqual(_FakeAudioCpp.bodies[-1]["caption"], "低い声")
            self.assertEqual(_FakeAudioCpp.bodies[-1]["seed"], 7)
            base = IrodoriAudioCppEngine({"audiocpp_port": srv.port, "audiocpp_model": "500m-v3",
                                          "audiocpp_model_path": ".", "voice": "低い声"})
            base.load()
            base.synthesize({"text": "あ"})
            self.assertNotIn("caption", _FakeAudioCpp.bodies[-1])
            self.assertEqual(_FakeAudioCpp.bodies[-1]["model"], "irodori-tts-500m-v3")


class IrodoriVoiceFallbackTest(unittest.TestCase):
    def test_voicevox_speaker_id_is_not_used_as_a_caption(self) -> None:
        from maai_sds.dialog import settings as vds

        self.assertEqual(vds._tts_voice_for_engine({"tts_voice": "8"}, "Irodori-v4", "既定"), "既定")
        self.assertEqual(vds._tts_voice_for_engine({"tts_voice": "F1"}, "Irodori-v4", "既定"), "既定")
        self.assertEqual(vds._tts_voice_for_engine({"tts_voice": "低い声"}, "Irodori-v4", "既定"), "低い声")
        self.assertEqual(vds._tts_voice_for_engine({"tts_voice": "8"}, "VOICEVOX-Core", "1"), "8")


if __name__ == "__main__":
    unittest.main()
