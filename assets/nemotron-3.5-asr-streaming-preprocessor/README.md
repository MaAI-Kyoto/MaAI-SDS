# Nemotron 3.5 ASR 前処理 ONNX

`preprocessor.onnx` と `preprocessor.export.onnx.data` は、`nvidia/nemotron-3.5-asr-streaming-0.6b` の NeMo 前処理を
sherpa-onnx から実行するための ONNX です。

## 内容

- STFT、パワースペクトル、128 bin の mel filterbank、log 変換、512 サンプルの Hann 窓のみを含み、学習済みの重みは含まない
- チャンク長に依存しないため、80 / 320 / 560 / 1120 ms の全モデルで同じファイルを使用

## 作成方法

`scripts/export_nemotron_preprocessor_onnx.py` で上記のモデルから生成しています。

> [!WARNING]
> `preprocessor.onnx` は外部データをファイル名で参照するため、2 つのファイルの名前を変更しないでください。

## SHA-256

| ファイル | SHA-256 |
|---|---|
| `preprocessor.onnx` | `a21d8788d37a71cef45050988bd9bbd4b19112e60bd7958d52ce1cfe314963c1` |
| `preprocessor.export.onnx.data` | `272f9ed95b43e15437df20b7d124daa6a8c831d8cdde3f40e6e71ec3acd06e38` |

## 配置

`scripts/setup_assets.ps1` / `scripts/setup_assets.sh` は、k2-fsa の各チャンク長の tarball を展開した後、
この 2 ファイルを各チャンク長のディレクトリへコピーします。
