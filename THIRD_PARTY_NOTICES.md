# サードパーティのライセンス

MaAI-SDS 本体のソースコードは MIT ライセンスです（[LICENSE](LICENSE)）。この文書では、本体が同梱・利用している
第三者のソフトウェアとモデルのライセンスをまとめています。各ライセンスの正文は、それぞれの配布元を参照してください。

ライセンス表記の確認日：2026-11-12（Python パッケージは `uv.lock` で導入される版、モデルは Hugging Face のライセンス表記に基づきます）

## 1. リポジトリに同梱しているもの

このリポジトリを配布する場合、以下も併せて配布されます。

| 対象 | 場所 | ライセンス | 備考 |
|---|---|---|---|
| sherpa-onnx の改変ソース | `scripts/sherpa_patches/*.cc` | Apache-2.0 | © Xiaomi Corporation ほか。MaAI-SDS で改変（各ファイルの冒頭に記載） |
| Nemotron 前処理 ONNX | `assets/nemotron-3.5-asr-streaming-preprocessor/` | MIT（本体と同じ） | NeMo（Apache-2.0）の前処理から生成。学習済みの重みは含まない |

GitHub Releases で配布する sherpa-onnx のビルド済みパッケージには、sherpa-onnx（Apache-2.0。`scripts/sherpa_patches/` の
パッチを適用してビルド）と ONNX Runtime（MIT）が含まれます。それぞれのライセンスは zip 内に同梱しています
（sherpa-onnx は `sherpa_onnx-*.dist-info/`、ONNX Runtime は `licenses/`）。

## 2. Python パッケージ（利用者がインストールするもの）

大半は MIT / BSD / Apache-2.0 などの寛容なライセンスです。ここでは注意が必要なものに限って挙げます。

### GPL（本体のコードからは import しないもの・任意機能のもの）

| パッケージ | ライセンス | 導入元 | 扱い |
|---|---|---|---|
| PyQt5 | GPL-3.0 | `maai` の必須依存 | 本体からは import しない。配布物にも含めない |
| phonemizer-fork / espeak-ng（espeakng-loader 経由） | GPL-3.0 以降 | `tts-kokoro` extra（kokoro-onnx, misaki） | 任意機能。Kokoro を導入した環境全体を配布する場合は GPL に従う |

GUI には PySide6（LGPL-3.0）を使用しており、PyQt6（GPL）は使用していません。OBS の遠隔録画には obsws-python（GPL-3.0-only）を
使用せず、`websocket-client` で OBS WebSocket v5 と直接通信する独自の実装（`src/maai_sds/obs/remote.py`）を用いています。

### LGPL / MPL（動的に利用する限り問題のないもの）

| パッケージ | ライセンス | 導入元 |
|---|---|---|
| PySide6（pyside6-essentials）/ Qt 6 | LGPL-3.0 | `gui` extra |
| pygame | LGPL-2.1 | `maai` |
| soxr | LGPL-2.1 以降 | `librosa` |
| num2words | LGPL-2.1 | `tts-kokoro` extra（misaki） |
| frozendict | LGPL-3.0 | `tts-kokoro` extra（x-transformers → einx） |
| certifi / tqdm | MPL-2.0（tqdm は MPL-2.0 AND MIT） | 多数 |

### プロプライエタリ

| パッケージ | ライセンス | 導入元 |
|---|---|---|
| nvidia-*-cu12（CUDA / cuDNN ランタイム） | NVIDIA ソフトウェアライセンス | `cu128` extra（torch, onnxruntime-gpu） |

## 3. モデル（利用者が各自ダウンロードするもの）

モデルはリポジトリに含めていません。セットアップ時に配布元から取得するため、利用条件は各モデルのライセンスに従います。
**非商用（NC）に限られるモデルが含まれる点に注意してください。**

### 実行系（利用者がダウンロードするもの）

| 対象 | ライセンス | 備考 |
|---|---|---|
| speech.cpp（nyosegawa/speech.cpp） | MIT | `speechcpp-asr` と Irodori の既定の実行系。セットアップ時に GitHub Releases の実行ファイル（版と SHA-256 を固定）を `vendor/speechcpp/` に配置する |
| audio.cpp（0xShug0/audio.cpp） | Apache-2.0 | © 2026 ShugoAI LLC。`Qwen3-TTS-CPP`・`Supertonic-3`・`LFM2.5-Audio`・Irodori（audio.cpp 版）の実行系。セットアップ時に GitHub Releases の実行ファイルを `audio.cpp/build/windows-cuda-release/bin/` に配置する。GitHub の API はライセンスを「Other」と判定するが、LICENSE の本文は Apache-2.0（2026-11-22 確認） |

### MaAI（VAP によるターンテイキング・相槌・頷き）

`maai` パッケージ本体は MIT です。重みのライセンスはモデルごとに異なります。

| モデル | ライセンス |
|---|---|
| `maai-kyoto/vap_jp`, `vap_mc_jp`, `vad_jp`, `bc_det_jp` ほか、`_kyoto` の付かない多くのモデル | **CC BY-NC-ND 4.0（非商用・改変禁止）** |
| `maai-kyoto/vap_tri`, `vap_mc_tri` | **CC BY-NC-SA 4.0（非商用）** |
| `maai-kyoto/vap_*_kyoto`, `vap_bc_*`, `vap_nod_jp`, `vap_nod_para_jp`, `vap_prompt_jp`, `vap_3party_jp` | MIT |
| `maai-kyoto/continuous-mimi-onnx`, `kyutai/mimi` | CC BY 4.0（クレジット表記が必要） |
| `maai-kyoto/vap_bc_2type_jp`, `vap_nod_timing_jp`, `bc_det_en` | 表記なし（要確認） |

### ASR

| エンジン | モデル | ライセンス |
|---|---|---|
| sherpa-onnx | ReazonSpeech k2 v2 | Apache-2.0 |
| sensevoice-onnx | FunAudioLLM/SenseVoiceSmall | 独自（FunASR Model License） |
| funasr-nano-onnx | FunAudioLLM/Fun-ASR-Nano-2512 | Apache-2.0 |
| parakeet-onnx | nvidia/parakeet-tdt_ctc-0.6b-ja | CC BY 4.0 |
| cohere-asr-onnx | CohereLabs/cohere-transcribe-03-2026 | Apache-2.0 |
| nemotron-asr / nemotron-onnx | nvidia/nemotron-3.5-asr-streaming-0.6b | OpenMDW-1.1 |
| qwen3-asr | Qwen/Qwen3-ASR-0.6B | Apache-2.0 |
| speechcpp-asr | third-intelligence/Haas-ja-v1.0（GGUF は sakasegawa/Haas-ja-v1.0-GGUF） | Apache-2.0 |
| speechcpp-asr | reazon-research/reazonspeech-nemo-v2（GGUF は sakasegawa/reazonspeech-nemo-v2-GGUF） | Apache-2.0 |
| speechcpp-asr | nvidia/parakeet-tdt_ctc-0.6b-ja、Qwen/Qwen3-ASR-1.7B の GGUF | CC BY 4.0、Apache-2.0 |
| faster-whisper | Whisper 系（large-v3-turbo, kotoba-whisper v2.2） | MIT |
| ターン検出 | pipecat-ai/smart-turn-v3 | BSD-2-Clause |
| VAD | silero-vad | MIT |

### TTS

| エンジン | モデル・本体 | ライセンス |
|---|---|---|
| Kokoro 系 | hexgrad/Kokoro-82M | Apache-2.0（音素化に GPL の espeak-ng を使用。上記を参照） |
| Qwen3-TTS | Qwen/Qwen3-TTS-12Hz-*-CustomVoice | Apache-2.0 |
| Irodori-TTS | Aratako/Irodori-TTS-v4.1-Small、v4.1-Small-MF（GGUF は sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF） | MIT |
| Irodori-TTS（audio.cpp 版） | Aratako/Irodori-TTS-500M-v3、Aratako/Irodori-TTS-600M-v3-VoiceDesign（GGUF は audio-cpp/audio.cpp-gguf） | MIT（どちらもモデルカードの表記。2026-11-22 確認） |
| LFM2.5-Audio | LiquidAI/LFM2.5-Audio-1.5B-JP（GGUF は LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF） | **LFM Open License v1.0（年商 1000 万米ドル以上の法人は商用利用不可）**。非商用・研究目的の利用と、年商がそれ未満の商用利用は可 |
| Tsukasa | Respair/Tsukasa_Speech | **CC BY-NC 4.0（非商用）** |
| audio.cpp | audio-cpp/audio.cpp-gguf | 独自（要確認） |
| VOICEVOX / VOICEVOX Core | 本体・音声モデル | 本体は MIT ほか。音声ライブラリはキャラクターごとの利用規約に従う（クレジット表記「VOICEVOX:キャラ名」など） |
| Cartesia | 商用クラウド API | Cartesia の利用規約 |

## 4. 確認中のもの

- ライセンス表記の無いモデル（MaAI の 3 モデル、audio.cpp）は、配布元に確認しています。
