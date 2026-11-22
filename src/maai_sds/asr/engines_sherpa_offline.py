"""sherpa-onnx の OfflineRecognizer だけで完結する 4 エンジン。

`parakeet-onnx` / `sensevoice-onnx` / `funasr-nano-onnx` / `cohere-asr-onnx` は、
どれも「発話全体を 1 回投げて全文を受け取る」非ストリーミング認識で、デコードは
既存の `_decode_with_sherpa`（`engines_local.py`）をそのまま共有する。ここが持つのは
モデルの読み込み（ファクトリの選択と引数）だけなので、1 ファイルにまとめてある。

## 設定はファイルではなくディレクトリで持つ

4 つとも k2-fsa の release tarball を展開しただけの**決まったレイアウト**で配布される
（`model.int8.onnx` + `tokens.txt` 等）。既存の sherpa/qwen3/nemotron エンジンのように
ファイルを 1 本ずつ設定させると項目が 12 個に増えるだけで、実際には全部同じ
ディレクトリを指すことになる。ここでは**ディレクトリ 1 つ**を設定させ、中の
ファイル名は `_pick` が解決する。int8 と fp32 のどちらの命名でも拾えるので、
量子化違いを置き換えるときも設定を変えなくてよい。

## 先頭無音を足さない

`sherpa-onnx` エンジン（ReazonSpeech zipformer）は先頭 0.5 秒の無音で立ち上がりが
安定するが、ここの 4 つは疑似ストリーミングのたびに発話全体を再推論するため、
無音を足すとその分だけ毎回余計に重くなる（`qwen3-asr-onnx` が同じ理由で足していない）。
判定は `_sherpa_needs_lead_silence` で行う。

## 実測に基づく既定値（JSUT 20 発話 / CPU 8 スレッド。README の表と同じ条件）

- `provider` は 4 つとも **cpu**。int8 は dynamic 量子化で CUDA カーネルが効かず、
  実測でも CUDA の方が遅く最悪値も跳ねた（sensevoice: CPU 90ms/最悪123ms に対し
  CUDA 107ms/最悪2315ms）。nemotron-onnx int8 で既に分かっていたのと同じ理屈。
- SenseVoice / Cohere の ITN は **有効**。参照文に句読点があるため、付けた方が CER が
  下がる（SenseVoice: 0.1527 → 0.0897 と 41% 改善）。
- Fun-ASR-Nano の `max_new_tokens` は既定 512 ではなく **128**。自己回帰デコーダなので
  上限がそのまま最悪遅延になる（PyTorch 実装の最悪 2473ms に対し 906ms に収まった）。
"""

from __future__ import annotations

from pathlib import Path

from .. import engine_manager
from .. import paths as _paths


def _pick(root: Path, *names: str) -> str:
    """`root` 直下から最初に見つかった候補を返す（無ければ空文字）。"""
    for name in names:
        candidate = root / name
        if candidate.exists():
            return str(candidate)
    return ""


def _pick_quant(root: Path, provider: str, int8_name: str, full_name: str) -> str:
    """量子化違いが同じディレクトリに同居しているときに、provider 側に合う方を選ぶ。

    k2-fsa の fp32 tarball は `model.int8.onnx` と `model.onnx` の**両方**を含む
    （SenseVoice の 2024-07-17 fp32 版で確認）。固定順で選ぶと、fp32 を落としてきて
    GPU に切り替えたのに黙って int8 が使われる、という取り違えが起きる。

    実測（JSUT 20 発話）では、CPU は int8、GPU は fp32 が速い:
      SenseVoice int8/CPU 89ms ・ fp32/CPU 94ms ・ fp32/GPU 68ms
    int8 が GPU で伸びないのは dynamic 量子化に CUDA カーネルが無いため。
    """
    if provider.strip().lower() == "cuda":
        return _pick(root, full_name, int8_name)
    return _pick(root, int8_name, full_name)


class AsrSherpaOfflineEnginesMixin:
    """sherpa-onnx OfflineRecognizer 系のエンジン群。 `ASRService` に mixin する。"""

    def _import_sherpa_onnx(self):
        engine_manager.require_module_installed("sherpa_onnx", "sherpa-onnx")
        try:
            import sherpa_onnx
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError(
                f"sherpa-onnx をインポートできません: {ex}"
            ) from ex
        return sherpa_onnx

    def _sherpa_offline_model_dir(self, engine_label: str, raw_dir: str) -> Path:
        root = Path(self._resolve_model_path(raw_dir) or raw_dir)
        if not root.is_absolute():
            root = _paths.resolve_under_repo(root)
        if not root.is_dir():
            raise RuntimeError(
                f"{engine_label} のモデルディレクトリがありません: {root}。"
                " GUIの「エンジン管理」で該当エンジンにチェックを入れて「適用」するか、"
                " ASR設定でディレクトリを指定してください。"
            )
        return root

    def _require_files(self, engine_label: str, root: Path, found: dict[str, str]) -> dict[str, str]:
        missing = [name for name, path in found.items() if not path]
        if missing:
            raise RuntimeError(
                f"{engine_label} のモデルファイルが {root} に見つかりません: {', '.join(missing)}。"
                " release tarball を展開したディレクトリを指定してください。"
            )
        return found

    def _after_sherpa_offline_init(self) -> None:
        """他エンジンのハンドルを落とし、先頭無音の付与を無効にする。"""
        self._faster_whisper_model = None
        self._nemotron = None
        self._nemotron_onnx_recognizer = None
        self._sherpa_needs_lead_silence = False

    # ------------------------------------------------------------------ Parakeet
    def _init_engine_parakeet_onnx(self) -> None:
        """NVIDIA Parakeet TDT-CTC 0.6B 日本語（sherpa-onnx の nemo_ctc）。

        k2-fsa の配布物は **CTC ブランチだけをエクスポート**したものである
        （`scripts/nemo/parakeet-tdt_ctc-0.6b-ja/export-onnx-ctc.py` の
        `comment: "Only the CTC branch is exported"`）。TDT デコーダを使う CrispASR の
        GGUF とは別物で、実測でも CER 0.1208 対 0.0773 と差が出る。代わりに CTC は
        非自己回帰なので出力長に遅延が依存せず、最悪値が 193ms と安定する。
        """
        sherpa_onnx = self._import_sherpa_onnx()
        label = "Parakeet TDT-CTC (ja)"
        root = self._sherpa_offline_model_dir(label, self.parakeet_onnx_model_dir)
        files = self._require_files(label, root, {
            "model": _pick_quant(root, self.parakeet_onnx_provider, "model.int8.onnx", "model.onnx"),
            "tokens": _pick(root, "tokens.txt"),
        })
        self._log(
            f"parakeet-onnx: loading sherpa-onnx {self._sherpa_onnx_version_str()} "
            f"provider={self.parakeet_onnx_provider} threads={self.parakeet_onnx_num_threads} dir={root.name}"
        )
        self._sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=files["model"],
            tokens=files["tokens"],
            num_threads=self.parakeet_onnx_num_threads,
            provider=self.parakeet_onnx_provider,
        )
        self._after_sherpa_offline_init()

    # ----------------------------------------------------------------- SenseVoice
    def _init_engine_sensevoice_onnx(self) -> None:
        """SenseVoiceSmall（多言語・非自己回帰）。

        **モデルの取り違えに注意**: k2-fsa には同名系列で `2024-07-17` と `2025-09-09` が
        あるが、後者は `ASLP-lab/WSYue-ASR` の広東語ファインチューンで、日本語音声を
        簡体字で書き起こす（同梱の ja.wav すら「便当制」等になり CER 0.79 だった）。
        日本語には本家 FunAudioLLM/SenseVoice 由来の **2024-07-17** を使うこと。
        """
        sherpa_onnx = self._import_sherpa_onnx()
        label = "SenseVoiceSmall"
        root = self._sherpa_offline_model_dir(label, self.sensevoice_onnx_model_dir)
        files = self._require_files(label, root, {
            "model": _pick_quant(root, self.sensevoice_onnx_provider, "model.int8.onnx", "model.onnx"),
            "tokens": _pick(root, "tokens.txt"),
        })
        self._log(
            f"sensevoice-onnx: loading sherpa-onnx {self._sherpa_onnx_version_str()} "
            f"provider={self.sensevoice_onnx_provider} threads={self.sensevoice_onnx_num_threads} "
            f"language={self.sensevoice_onnx_language or 'auto'} itn={self.sensevoice_onnx_use_itn} dir={root.name}"
        )
        self._sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=files["model"],
            tokens=files["tokens"],
            num_threads=self.sensevoice_onnx_num_threads,
            provider=self.sensevoice_onnx_provider,
            language=self.sensevoice_onnx_language,
            use_itn=self.sensevoice_onnx_use_itn,
        )
        self._after_sherpa_offline_init()

    # ---------------------------------------------------------------- Fun-ASR-Nano
    def _init_engine_funasr_nano_onnx(self) -> None:
        """Fun-ASR-Nano-2512（音声エンコーダ + Qwen3-0.6B LLM デコーダ）。

        自己回帰デコーダなので、他の 3 つと違って**出力長に比例して遅延が伸びる**。
        `max_new_tokens` が最悪遅延の上限を決めるので、対話用に既定 512 から下げてある。
        `temperature` はライブラリ既定の 1e-06（実質 greedy）のまま使い、`seed` も既定
        （42）固定なので、同じ音声には同じ結果が返る。
        """
        sherpa_onnx = self._import_sherpa_onnx()
        label = "Fun-ASR-Nano"
        root = self._sherpa_offline_model_dir(label, self.funasr_nano_onnx_model_dir)
        # llm.fp16.onnx は載せない。sherpa 側が CUDA で明示的に弾き（offline-funasr-nano-model.cc
        # の CheckFp16OnCuda「fp16 LLM models are not supported on CUDA yet」）、CPU では
        # エミュレーションで int8 より遅くなる（1152ms 対 648ms、CER も 0.102 対 0.088 と悪化）。
        # fp32 も CUDA で 628ms と int8/CPU の 629ms に並ぶだけで、3.7GB を置く価値がない。
        files = self._require_files(label, root, {
            "encoder_adaptor": _pick(root, "encoder_adaptor.int8.onnx", "encoder_adaptor.onnx"),
            "llm": _pick(root, "llm.int8.onnx", "llm.fp32.onnx", "llm.onnx"),
            "embedding": _pick(root, "embedding.int8.onnx", "embedding.onnx"),
            "tokenizer": _pick(root, "Qwen3-0.6B"),
        })
        self._log(
            f"funasr-nano-onnx: loading sherpa-onnx {self._sherpa_onnx_version_str()} "
            f"provider={self.funasr_nano_onnx_provider} threads={self.funasr_nano_onnx_num_threads} "
            f"language={self.funasr_nano_onnx_language or 'auto'} itn={self.funasr_nano_onnx_use_itn} "
            f"max_new_tokens={self.funasr_nano_onnx_max_new_tokens} dir={root.name}"
        )
        self._sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor=files["encoder_adaptor"],
            llm=files["llm"],
            embedding=files["embedding"],
            tokenizer=files["tokenizer"],
            num_threads=self.funasr_nano_onnx_num_threads,
            provider=self.funasr_nano_onnx_provider,
            language=self.funasr_nano_onnx_language,
            itn=self.funasr_nano_onnx_use_itn,
            max_new_tokens=self.funasr_nano_onnx_max_new_tokens,
        )
        self._after_sherpa_offline_init()

    # --------------------------------------------------------------------- Cohere
    def _init_engine_cohere_asr_onnx(self) -> None:
        """Cohere Transcribe（14 言語。日本語を含む）。

        encoder は ONNX external data 形式（`encoder.int8.onnx` 本体 3MB ＋
        `encoder.int8.onnx.data` 2.6GB）なので、2 つを同じディレクトリに置いたまま使う。
        """
        sherpa_onnx = self._import_sherpa_onnx()
        label = "Cohere Transcribe"
        root = self._sherpa_offline_model_dir(label, self.cohere_onnx_model_dir)
        files = self._require_files(label, root, {
            "encoder": _pick(root, "encoder.int8.onnx", "encoder.onnx"),
            "decoder": _pick(root, "decoder.int8.onnx", "decoder.onnx"),
            "tokens": _pick(root, "tokens.txt"),
        })
        self._log(
            f"cohere-asr-onnx: loading sherpa-onnx {self._sherpa_onnx_version_str()} "
            f"provider={self.cohere_onnx_provider} threads={self.cohere_onnx_num_threads} "
            f"language={self.cohere_onnx_language or 'auto'} "
            f"punct={self.cohere_onnx_use_punct} itn={self.cohere_onnx_use_itn} dir={root.name}"
        )
        self._sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
            encoder=files["encoder"],
            decoder=files["decoder"],
            tokens=files["tokens"],
            num_threads=self.cohere_onnx_num_threads,
            provider=self.cohere_onnx_provider,
            language=self.cohere_onnx_language,
            use_punct=self.cohere_onnx_use_punct,
            use_itn=self.cohere_onnx_use_itn,
        )
        self._after_sherpa_offline_init()
