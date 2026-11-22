# nemotron-onnx 用のカスタム sherpa-onnx

`nemotron-onnx` エンジン（sherpa-onnx で実行する Nemotron 3.5 の多言語ストリーミング ASR）の品質に必要なパッチは、
PyPI の `sherpa-onnx` には含まれていません。MaAI-SDS では、公式の v1.13.5 に不足分のパッチを適用したカスタムビルドを
`.venv` に導入して使用します。

## 導入方法

通常は自分でビルドする必要はありません。`scripts\setup_env.ps1`（および `scripts/setup_assets.sh`）が、
GitHub Releases に公開しているビルド済みパッケージ（Python 3.11 用）を取得して導入します。
自分でビルドする必要があるのは、Python のバージョンが異なる場合と、パッチを変更する場合のみです。

Docker イメージ（GPU 版・CPU 版とも）は、ビルド時に Linux 用のビルド済みパッケージを同じ Releases から取得して導入します。
CUDA 版のビルドですが、CUDA は使用時にのみ読み込むため、GPU の無い環境でも CPU で動作します。
取得できなかった場合（ネットワークに接続できない、フォークに Releases が無いなど）は、ビルドログに `[sherpa][WARN]` を
出力し、PyPI 版のまま続行します（この場合 `nemotron-onnx` の精度は低下します）。

## 公式版との差分

| 項目 | 公式 | MaAI-SDS のビルド |
|---|---|---|
| 多言語 encoder の入力 `prompt_index` | v1.13.3 以降（[PR #3671](https://github.com/k2-fsa/sherpa-onnx/pull/3671)） | 公式どおり |
| チャンク境界での prediction の二重投入 | v1.13.5（[PR #3785](https://github.com/k2-fsa/sherpa-onnx/pull/3785)、`decoder_out` のキャッシュ） | 公式どおり（独自パッチは適用しない） |
| NeMo の mel / `preprocessor.onnx` | 増分の `knf::Fbank`（前処理 ONNX は探索しない） | 同梱の前処理 ONNX を自動で検出し、波形のリングバッファから NeMo と同じ mel を再計算 |
| RNNT の `max_symbols_per_frame` | 10 固定 | 30（日本語の認識漏れを低減） |

- ベース：[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) v1.13.5（`3dc7c56`）
- ビルドの記録：リポジトリ直下の `.sherpa_onnx_git_built`（コミット・ビルド日時・パッチ内容。Git 管理外）

## パッチ（`scripts/`）

| スクリプト | 内容 |
|---|---|
| `patch_sherpa_msvc_parallel.py` | MSVC の並列ビルドを `-m:1` に設定（OpenFST のビルド失敗への対策） |
| `patch_sherpa_nemo_max_symbols.py` | NeMo デコーダの `max_symbols_per_frame=30` |
| `patch_sherpa_nemo_greedy_doublefeed.py` | v1.13.5 以降では何もしない（公式の #3785 を優先）。古いソース向けの予備 |
| `patch_sherpa_nemo_mel_params.py` | preemphasis・フレーム・窓など fbank のパラメーターを NeMo に合わせる |
| `patch_sherpa_nemo_online_normalize.py` | `online-stream.cc` に NeMo の mel の再計算と `GetNeMoNormalizedFeatures` を追加 |
| `patch_sherpa_nemo_preprocessor.py` | encoder と同じフォルダーの `preprocessor.onnx` を自動で検出し、NeMo と同じ mel の経路を使用 |
| `sherpa_patches/online-stream.nemo.cc` | 上記の C++ 実装のひな形（v1.13.5 の `GetNeMoDecoderOut` を含む） |

### mel の再計算の範囲

生の音声のリングバッファは最大 15 秒分を保持します（まとめて投入された音声のうち、未デコードの部分を失わないため）。
ただし mel を再計算するのはリング全体ではなく、現在のチャンク周辺の小さな窓（過去の数フレームと、`IsReady` が保証する
1 フレーム先）のみです。NA 正規化では mel がフレームごとに確定するため、切り出したチャンクは全体を計算した場合と一致し
（差が 1e-6 未満であることを確認済み）、計算量は発話の長さによらず一定です。per_feature 正規化の場合のみ全体を再計算します。

## 自分でビルドする（Windows）

前提：Git、CMake、MSVC Build Tools（C++）、`uv sync` 済みの `.venv`。ビルドには 30 分以上かかります。

```powershell
.\scripts\upgrade_sherpa_onnx.ps1            # clone・パッチ・ビルド・インストール・確認まで
.\.venv\Scripts\python.exe .\scripts\verify_sherpa_import.py
# => import OK 1.13.5
```

既定は GPU ビルドです（`-EnableGpu`。`onnxruntime-win-x64-gpu-1.24.4` の CUDA 実行プロバイダーを同梱）。
CUDA と cuDNN の DLL は、venv の `nvidia-*` と `torch` の wheel から `ort_windows_preload._add_cuda_dll_dirs()` が
自動で検出します（cuDNN を別途導入する必要はありません）。CPU のみで使用する場合は `-EnableGpu:$false` でビルドできます。

Linux 用のビルド済みパッケージは `scripts/build_sherpa_prebuilt_linux.sh` で作成します。

## Nemotron の ONNX モデル

### チャンク長

多言語の Nemotron のチャンク長は 80 / 320 / 560 / 1120 ms です（既定は 320 ms）。k2-fsa の配布物には
80 / 160 / 320 / 560 / 1120 ms がありますが、160 ms は英語用のため使用しません。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_assets.ps1 -Only NemotronOnnxModels
```

Linux と Docker では `bash scripts/setup_assets.sh --only NemotronOnnxModels` を使います。

### 量子化

`setup_assets` が配置するのは k2-fsa の配布物の int8 版（`encoder.int8.onnx` / `decoder.int8.onnx` / `joiner.int8.onnx`）です。
fp32 の `encoder.onnx` などは、手元で書き出した場合にのみ存在し、配布はしていません。同じフォルダーに fp32 版があれば
そちらを優先します（`nemotron_chunk.resolve_nemotron_onnx_file_path`）。

### `preprocessor.onnx`（全チャンク長で必須）

NeMo と同じ mel を得るための前処理の ONNX です。各チャンク長のフォルダーで `encoder.onnx` と同じ場所に置くと、
sherpa-onnx が自動で検出して使用します。

> [!WARNING]
> `preprocessor.onnx` が無いと `knf::Fbank` にフォールバックし、CER が大きく悪化します（実測 0.17 → 0.34）。

k2-fsa の配布物には含まれていないため、本リポジトリの `assets/nemotron-3.5-asr-streaming-preprocessor/` に同梱しています。
`setup_assets` は、ファイルが無い場合や SHA-256 が一致しない場合に、各チャンク長のフォルダーへ配置します。
前処理はチャンク長に依存せず、学習済みの重みも含みません。作成方法とハッシュ値は
[assets の README](../assets/nemotron-3.5-asr-streaming-preprocessor/README.md) を参照してください。

## 実行モード

設定画面の ASR 設定にある nemotron-onnx の「実行モード」、または `controller_config.json` の `asr_nemotron_onnx_compute` で選択します。

| 実行モード | `asr_nemotron_onnx_compute` | provider / 量子化 | 備考 |
|---|---|---|---|
| GPU (fp32) | `gpu-fp32`（既定） | cuda / fp32 | 遅延が最も小さく、精度も最良。NVIDIA GPU と GPU ビルドの sherpa-onnx が必要 |
| CPU (int8) | `cpu-int8` | cpu / int8 | GPU 不要。CPU の fp32 より高速で、精度はほぼ同等 |

int8 は動的量子化のため CUDA カーネルが効かず、GPU ではかえって遅くなります。

JSUT 20 文を 1 発話ずつまとめて投入した場合の、平均遅延と CER の実測値は次のとおりです。

| 構成 | 遅延 | CER |
|---|---|---|
| fp32 / cuda（推奨） | 約 0.4 s | 0.173 |
| int8 / cpu | 約 1.3 s | 0.180 |
| fp32 / cpu | 約 2.2 s | 0.173 |
| int8 / cuda | 約 1.8 s | — |

## PyPI 版に置き換わった場合の復元

> [!WARNING]
> `uv sync` を単独で実行すると、カスタムビルドが PyPI の `sherpa-onnx` で上書きされます。

`import sherpa_onnx` が `ModuleNotFoundError: sherpa_onnx.display` などで失敗する場合や、`verify_sherpa_import.py` が
表示するバージョンが 1.13.5 でない場合は、次のコマンドで復元できます。

```powershell
.\scripts\setup_env.ps1      # 待避しておいたビルド済みパッケージを戻す（無ければ取ってくる）
```

それでも解決しない場合は、`.\scripts\upgrade_sherpa_onnx.ps1` でビルドし直してください。
`scripts/fix_sherpa_onnx_init.py` は、`.pyd` と `__init__.py` の公開名が一致しなくなった場合の修正用です。
