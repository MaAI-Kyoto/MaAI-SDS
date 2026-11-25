# 内部仕様（モジュール別）

MaAI-SDS の主要モジュールについて、役割・処理の流れ・入出力・設定をモジュールごとにまとめた技術資料である。
プロセス構成・起動形態・IPC・音声対話全体のデータの流れは [architecture.md](./architecture.md)、
起動と運用の手順は [running.md](./running.md) を参照のこと。実装の詳細は各 `*.py` を参照のこと。

> [!NOTE]
> アプリ本体のモジュールは `src/maai_sds/` パッケージ配下にあり、ドメイン別のサブパッケージ（`asr/`・`gui/`・`dialog/`・`llm/`・
> `tts/`・`audio/`・`maai/`・`mmd/`・`obs/`・`ipc/`・`web/`）に分かれている。本書の `asr/service.py` という表記は
> `src/maai_sds/asr/service.py` を指す（`scripts/` 配下のものは除く）。ワーカーは `python -m maai_sds.<サブパッケージ>.<名前>`
> （例：`maai_sds.asr.worker`）で起動する。設定ファイル・`maai_assets/`・`logs/`・相対パスで指定したモデルはリポジトリの
> ルートを基準とし、パスの解決は `src/maai_sds/paths.py` に集約している。

## 目次

章は音声対話のパイプラインの順に並べている。音声対話の流れを追う場合は、[対話制御](#対話制御dialog-orchestrator) →
[音声入力](#音声入力audio-input-worker) → [ASR](#asr) → [MaAI](#maaimaai-worker) の順に読むとよい。

| 章 | 主なファイル |
|---|---|
| [対話制御（Dialog Orchestrator）](#対話制御dialog-orchestrator) | `dialog/orchestrator.py`・`dialog/worker.py`・`dialog/runtime.py` |
| [音声入力（Audio Input Worker）](#音声入力audio-input-worker) | `audio/input_worker.py` |
| [ASR](#asr) | `asr/worker.py`・`asr/service.py`・`asr/vad_turn.py`・`asr/engines_*.py`・`asr/compare_worker.py` |
| [MaAI（MaAI Worker）](#maaimaai-worker) | `maai/worker.py` |
| [LLM](#llm) | `llm/worker.py`・`llm/service.py`・`llm/admin_http.py` |
| [TTS と音声出力](#tts-と音声出力) | `tts/worker.py`・`tts/service.py`・`tts/routing.py`・`audio/output_worker.py` |
| [MMDAgent-EX 連携（ブリッジ・ペーサー・頷き）](#mmdagent-ex-連携ブリッジペーサー頷き) | `mmd/remote_bridge.py`・`audio/mmd_pacer.py`・`audio/segmented_pacer.py`・`maai/nod_motion.py`・`mmd/runtime.py` |
| [設定画面（Qt・Web）](#設定画面qtweb) | `main.py`・`gui/controller.py`・`gui/ui.py`・`web/server.py` |
| [設定とユーティリティ](#設定とユーティリティ) | `config_store.py`・`dialog/settings.py`・`gui/app_config_spec.py`・`tts/playback_service.py`・`llm/common.py`・`ipc/worker_runtime.py`・`ipc/queue_policy.py`・`utils.py` |
| [補助サービス](#補助サービス) | `mmd/mdf_service.py`・`obs/remote.py`・`maai/plot_window.py`・`maai/nod_motion_chart_widget.py` |

IPC の共通実装（`ipc/jsonline.py`）とワーカーのプロセス管理（`ipc/process_service.py`・`ipc/orchestrator_supervisor.py`）は
[architecture.md](./architecture.md) で扱う。

---

## 対話制御（Dialog Orchestrator）

### 役割

音声対話の状態遷移を管理し、各ワーカーを束ねる対話サーバーの中核である。Controller（Qt アプリ）や
Web UI とは独立したプロセスとして動作する。

| コンポーネント | 役割 |
|----------------|------|
| `DialogOrchestrator`（`dialog/orchestrator.py`） | UI に依存しない状態機械。ASR・MaAI・ターン・履歴・LLM/TTS 連携の入力を受け、結果をイベント（`list[dict]`）として返す。各イベントは構築時に渡されたコールバックにも配送する |
| `dialog/worker.py` | 対話サーバープロセスのエントリーポイント。IPC サーバー群・`OrchestratorSupervisor`・各ランタイムを構築し、受信イベントを `DialogOrchestrator` とランタイムへ振り分ける |
| `VoiceDialogRuntime`（`dialog/runtime.py`） | 実行時の処理。ワーカーの起動、LLM 要求の送出、TTS の分割合成と再生制御、割り込みの実行、レイテンシ計測、ASR / MaAI からの着信イベントの転送 |
| `MmdRuntime`（`mmd/runtime.py`） | 頷きモーションの送出。MMDAgent-EX への送出は Audio Output ワーカーへの IPC に委譲 |
| `TtsRouter`（`tts/routing.py`） | TTS 合成と PCM の経路選択。`OrchestratorSupervisor.send_audio_output` を主経路とする |
| `OrchestratorSupervisor`（`ipc/orchestrator_supervisor.py`） | 子ワーカーの起動・停止と Duplex IPC の管理（[architecture.md](./architecture.md)） |

### 処理の流れ

#### 起動時の構成

1. UI 用の `JsonLineDuplexServer` を起動し、待受ポートを stdout に 1 行の JSON（`{"ui_port": <port>}`）で通知する。
   Controller はここへ `JsonLineDuplexClient` で接続する（サーバーは対話サーバー側であり、Controller ではない）。
2. 着信用の `JsonLineEventServer` を 4 つ起動する（`audio_input_server` / `asr_audio_server` /
   `asr_compare_audio_server` / `maai_audio_server`）。
   - Audio Input ワーカーからの PCM
   - ASR / ASR Compare / MaAI ワーカーとの間の `audio_frame` / `system_audio_frame` などのイベント
3. `OrchestratorSupervisor`・`MmdRuntime`・`VoiceDialogRuntime`・`TtsRouter`・`DialogOrchestrator` の順に構築する。
   `DialogOrchestrator` には `VoiceDialogRuntime.as_orchestrator_callbacks()` のコールバックを渡す。
4. 設定ファイルを直接読み、Audio Input / Audio Output ワーカーを先行して起動する
   （`bootstrap_audio_io_workers_early`）。音声対話の開始を待たずに MMDAgent-EX が接続できる状態にするためである。
5. Controller との接続が安定した時点で `ready` を 1 回送る。Controller はこれを受けて音声入出力の再同期を行う。
6. メインループ:
   - 50 ms ごとに `orchestrator.on_tick()` を呼ぶ（非ストリーミング MaAI ターン待ちの締め切り、アイドル時のターン取得など）
   - キューに届いたイベントを `_handle_event` で処理する

#### 音声対話の開始と停止

1. Controller が `start_voice_dialog`（`settings` 付き）を Duplex で送る。
2. `VoiceDialogRuntime.ensure_voice_dialog_workers` が `OrchestratorSupervisor` を通じて ASR・LLM・TTS・MaAI・音声入力・音声出力を起動する。
3. 起動の完了後、`finalize_start_voice_dialog`（`ensure_*` の完了後にイベントキュー経由でメインループへ戻す `_vd_start_finalize`）から
   `DialogOrchestrator.start()` を実行する。
4. `stop_voice_dialog` を受けると、ワーカーの停止と状態のリセットを行う。

#### `_handle_event` の振り分け

`_handle_event` は `_src` と `type` で処理先を決め、`type` ごとの処理は `DISPATCH` 表に定義する。
`DISPATCH` に無い `type` は、種類ごとに初回だけ stderr に警告を出す（プロトコル不一致の検出）。

| 送信元 | 主な `type` | 処理先 |
|---|---|---|
| Controller（UI 用 Duplex） | `start_voice_dialog` / `stop_voice_dialog` / `update_settings` / `interrupt` / `force_finalize` / `send_to_llm` / `start_llm_gui_test` / `start_tts_gui_test` など | `VoiceDialogRuntime` / `OrchestratorSupervisor` |
| ASR / MaAI（EventServer 経由。`forward_asr_ingress_event` / `on_maai_ingress_event` でキューに投入） | `asr_result` / `turn` / `maai_result` / `maai_bc_result` / `maai_nod_result` | `on_asr_result` / `handle_turn`（→ `on_turn_probability`）/ `handle_maai_*`（→ `on_maai_*`） |
| ASR（強制確定） | `force_finalize_status` / `force_finalize_outcome` | `VoiceDialogRuntime` |
| LLM ワーカー（`_src: "llm"`） | `chunk` / `tts_segment` / `done` / `error` など | `VoiceDialogRuntime.handle_llm_worker_event` |
| Audio Output ワーカー（`_src: "audio_output"`） | `ready` / `tts_mmd_send_complete` / `tts_maai_pcm_frame` / `remote_status` | 正規化して Controller へ転送、送出完了の処理、MaAI 用の system 音声注入、MMD 接続状態の反映 |
| 内部（ランタイム） | `llm_state` / `llm_finished` / `assistant_reply` / `tts_audible` / `tts_done` など | `DialogOrchestrator` の対応メソッド（`on_llm_state` / `on_tts_done` など） |

`tts_play_chunk` / `tts_play_break` / `mmd_audio_flush` は Audio Output ワーカーへ中継する（ペーサーと
`MmdRemoteBridge` は Audio Output ワーカー側にある）。頷き要求 `request_nod_motion` は Controller へ
emit したうえで、`MmdRuntime.start_nod_motion` が別スレッドで頷きの角度（`KEYVALUE_SET`）を送出する。

### `DialogOrchestrator` の状態機械

閾値・ストリーミング・`force_finalize` の分岐は `DialogOrchestrator` が担う。ターン確率は ASR ワーカー（`smart-turn`）または
MaAI ワーカー（`maai`）から EventServer に届き、`VoiceDialogRuntime.handle_turn` を経由して `on_turn_probability` に渡る。

#### 入力

| メソッド | 処理 |
|---|---|
| `on_asr_result` | ストリーミング / 非ストリーミング、partial / final、`utterance_index` に応じて `messages` / `pending_texts` / `final_seq` を更新する。ストリーミング時は履歴行の差し替えと、TTS 分割位置の補正イベント（`tts_cumulative_strip_prefix`）を出す |
| `on_turn_probability` | MaAI / smart-turn の閾値の上抜け（`crossed_up`）でターンを取得する。「ターンを取る条件」（`asr_maai_turn_condition`）が `p_now_and_future` のときは、上抜けのフレームで `p_future` も閾値以上の場合だけ取る（未満なら `branch=skip reason=p_future_below_threshold` を出し、`p_now` が閾値を下回って再び上抜けるまで待つ。`p_future` を送らない経路では `p_now` だけで判定）。既定は `p_now`。「ターンを取るまでの連続フレーム数」（`asr_maai_turn_hold_frames`、既定 1）を増やすと、上抜けてからしきい値以上が N フレーム続いたフレームでターンを取る（`turn_above_frames`。話の途中の短い間で p_now が一瞬だけ超えたときに取らないため）。ストリーミング時は `_trigger_llm_from_history_only`、非ストリーミング時は `wait_nonstream_maai_turn` と `request_force_finalize` などを使う |
| `on_maai_result` | VAD と `p_now` から割り込み条件（`user_side`）とターンロックの解除を判定する。`last_user_speech_detect_wall_ts`（沈黙後のターン取得用）を更新する |
| `on_maai_bc_result` / `on_maai_nod_result` | 相槌・頷き（nod_para）の発火判定 |
| `on_tick` | `wait_nonstream_maai_turn` のタイムアウト、`final_seq` 待ち、`voice_dialog_idle_take_turn_sec` によるアイドル時のターン取得 |

#### 出力イベント

| イベント | 処理 |
|---|---|
| `trigger_llm` | `VoiceDialogRuntime` が LLM 要求を送出する。Controller にも emit する（表示の更新用） |
| `request_nonstream_tts` / `request_opening_tts` / `request_scheduled_exit_tts` | 対話サーバー内で合成・再生する。Controller には観測用に emit する。開始時と終了時の発話は、ユーザーが話していても最後まで流す（終了時の発話を出したら `scheduled_exit_speaking` を立て、割り込み・ターン取得・沈黙後のターン取得を止める。runtime は直前の割り込みの停止フラグを下ろし、再生前の VAD の確認を飛ばす） |
| `request_force_finalize` | ASR ワーカーへ `force_finalize` を送る |
| `request_interrupt` | `VoiceDialogRuntime.execute_user_interrupt` が音声・LLM・キューを停止する |
| `interrupt_condition` | 割り込み条件の立ち上がり（ログ用） |
| `history_snapshot` | 履歴の同期（UI 表示用。高頻度で出す） |
| `latency` | レイテンシ表示（開始時はプレースホルダー） |

上記以外のイベントは `on_emit` を通じて Controller へ転送する。

#### ターンロック `turn_blocked_until_user_speech`

- アシスタントの発話中や開始直後など、ユーザー側の発話（`user_side`）を確認するまでターン検出（`crossed_up`）を抑止する。
- 開始時のシステム発話（`voice_dialog_opening_utterance`）の再生中は、`user_side` による解除を行わない。
- `voice_dialog_idle_take_turn_sec`：ロック解除後、最後の `user_side` からの沈黙がこの秒数を超えると、
  `crossed_up` と同じ処理（`_run_turn_acquire_flow`）でターンを取得する。

### MMDAgent-EX への出力

- `MmdRemoteBridge` と `MmdAudioPacer` は Audio Output ワーカー内にある。必要に応じて同じ PCM を MaAI の system 入力へ注入する。
- 対話サーバー内の `MmdRuntime` は頷きのコマンドを `mmd_bind_head` / `mmd_send_text` で Audio Output ワーカーへ送る。
- 詳細は [MMDAgent-EX 連携](#mmdagent-ex-連携ブリッジペーサー頷き) を参照。

### 障害時の動作

| 対象 | 動作 |
|---|---|
| ワーカー全般 | IPC が予期せず切断されると、Supervisor が ready をクリアして `worker_disconnected` を通知（`_handle_worker_disconnect`）。次の `ensure_*` で再起動できる |
| ASR | 起動中の停止要求を保留（`_asr_stop_requested_during_start`）。起動に失敗した場合はエラーを表示して停止（自動復旧の機構はない） |
| TTS | ワーカーの `ready` を明示的に待機してから使用 |
| LLM | クライアントのキャッシュ、ウォームアップの重複抑止（`LLMService._warmup_inflight`） |
| Orchestrator | Controller が IPC の切断を検知すると、再接続を自動で試行 |

### 外部クライアントからの利用

`dialog/worker.py` は Qt や `gui/controller.py` に依存しない。UI 用ポートに JSON Lines で接続すれば、
Controller 以外のプロセスからも音声対話とテスト機能を操作できる。その実装例がヘッドレス CLI
（`headless_main.py`、[architecture.md](./architecture.md)）である。

#### 起動オプション

| オプション | 内容 |
|---|---|
| `--config PATH` | 先行起動（音声入出力ワーカー）で読む設定ファイル。省略時はリポジトリルートの `controller_config.json` |
| `--listen HOST:PORT`（または環境変数 `MAAI_SDS_UI_LISTEN`） | UI 用ポートを固定して待ち受ける（Docker 用）。stdout には `ui_host` / `ui_port` / `listen_mode` を出力し、UI が接続するたびに `ready` と現在の状態を送り直す |
| `--assume-mmd-running` | 先行起動時も `audio_output_mode=mmd` を維持する。既定では MMDAgent-EX 未起動とみなして `local` に切り替える |

#### 接続の手順

1. `python -u -m maai_sds.dialog.worker` を起動し、stdout の `{"ui_port": <port>}` を読む。
2. `127.0.0.1:<port>` に `JsonLineDuplexClient` で接続する。
3. 接続が確立すると `{"type": "ready"}` を受信する。
4. `controller_config.json` 形式の dict から `dialog/settings.py` の `build_start_voice_dialog_payload(cfg)` で settings を組み立て、
   `{"type": "start_voice_dialog", "settings": {...}}` を送る。
5. `{"type": "status", "state": "running"}` を受信した時点で音声対話が開始している。

#### 主なコマンド（送信）

| `type` | 用途 |
|--------|------|
| `start_voice_dialog` | 音声対話の開始（`settings` が必須） |
| `update_settings` | 設定の差分更新（dict） |
| `stop_voice_dialog` | 音声対話の停止 |
| `interrupt` | LLM / TTS の打ち切り |
| `force_finalize` | ASR の強制確定 |
| `inject_system_audio` | system 側音声を Audio Input ワーカー経由で MaAI / ASR へ注入（`audio_b64`） |
| `tts_play_chunk` / `tts_play_break` / `mmd_audio_flush` | Audio Output ワーカーのペーサー操作 |
| `start_module` / `stop_module` | ワーカーの個別起動・停止 |
| `send_to_llm` / `send_to_tts` / `send_to_tts_bridge` / `send_to_llm_admin` / `send_to_audio_output` | 各ワーカー・ブリッジへのコマンド転送 |
| `start_llm_gui_test` / `start_tts_gui_test` | 設定画面の LLM・TTS テスト（音声対話と同じ経路で実行） |
| `run_llm_test` / `run_tts_test` / `run_asr_test` / `run_maai_test` | テスト用 RPC |

MMDAgent-EX の接続設定（`remote_plugin_apply_settings` / `remote_plugin_stop`）は、`send_to_audio_output` で
Audio Output ワーカーへ送る。

#### 主なイベント（受信）

| `type` | 意味 |
|--------|------|
| `ready` / `module_started` | 対話サーバーの準備完了 / 個別ワーカーの起動完了 |
| `status` | `state` は `starting` / `running` / `stopping` / `stopped` のいずれか |
| `history_snapshot` | 履歴・LLM の進行状態などの同期 |
| `asr_result` / `maai_result` / `raw_audio` / `inference_state` | 認識結果・MaAI の系列・音声・推論状態の表示用ミラー |
| `turn` | ターン取得（`from_orchestrator=True` はプロット表示用で、ターン取得ではない） |
| `trigger_llm` | LLM の実行開始（要求は対話サーバー内の `dispatch_llm_request` で送出済み。観測用） |
| `request_opening_tts` / `request_nonstream_tts` / `request_scheduled_exit_tts` | 対話サーバー内で再生まで完結する（観測用） |
| `llm_chunk` / `llm_done` | LLM のストリーミング応答 |
| `latency` | 各種レイテンシ |
| `log` / `error` / `interrupt_condition` | ログ |
| `tts_mmd_send_complete` | MMD / Audio Output 経路の SND 送出完了 |

`asr_result` / `turn` / `maai_result` の発生源は各ワーカーであり、対話サーバーが直接受信する。Controller がこれらを中継する必要はない。
`tts_maai_pcm_frame`（MaAI 用の system 入力 PCM）は対話サーバー内で Audio Input ワーカーへ注入し、UI には転送しない。

#### 設定の入手元

- GUI が `ConfigStore` で保存した `controller_config.json` をそのまま利用できる。
- 変換は `build_start_voice_dialog_payload(cfg)`（`dialog/settings.py`。GUI・Web UI と共通）で行う。

---

## 音声入力（Audio Input Worker）

### 役割

- マイク・WAV ファイルなどからモノラル PCM を読み取り、共通の音声ハブとして対話サーバー上の `JsonLineEventServer` へ
  JSON イベントで送信（ASR・MaAI などへの転送は対話サーバーが担当）
- 送信単位は常に 320 サンプル（16 kHz で 20 ms 相当）。ASR・MaAI が別のサイズを必要とする場合は、各ワーカー側で再チャンク
- 本ワーカー自身は VAD を行わず、生の PCM をそのまま送出（設定に VAD バックエンド名が含まれていても同様。VAD は ASR サービス側・MaAI 側で実施）

### 設定・接続

- `--host` / `--port`：対話サーバー内の音声入力用 `JsonLineEventServer`。`JsonLineEventClient` で接続
- `--config-json`：`sample_rate`、`input_mode`（`mic`・`wav`・`browser`・`tcp`。`browser` は Web UI から 16 kHz の PCM を受け取るモード）、デバイスのインデックスなど。旧キー `audio_forward_*` は無視（`pop`）

### 処理の流れ（マイク）

1. `sounddevice` の入力ストリームを開く。リアルタイムのコールバック（`_audio_callback`）では、ブロッキング処理も重い処理も行わない。
   再利用バッファをコピーして上限付きのキュー（`mic_frame_queue`）へ積み、ただちに戻る。キューが満杯のときは最も古いフレームを
   破棄して遅延を抑える。
2. 専用スレッド `_mic_processor_loop` がキューからフレームを取り出し、AEC・リサンプル・ゲイン調整・雑音抑制（下記）・int16 変換を行ったうえで
   `_emit_pcm_bytes` から送信する（ブロッキング送信はこのスレッドで行う）。音声は float32 のモノラルに揃え、320 サンプル単位で送出する。
3. 各チャンクに `sample_start`・`ts_start` などのメタデータを付与し、`emit_audio("audio_frame", ...)`（ユーザー音声）と
   `emit_audio("system_audio_frame", ...)`（システム音声）を送る。
4. すべてのフレームは primary IPC（上記の 1 接続）へ送る。ASR・MaAI への転送は対話サーバーが行う。

> WAV・TCP モードは通常のスレッドで動作するため、コールバックと処理スレッドの分離はマイクモードのみに適用している。

### 雑音抑制

- 「音声入力の設定」の「雑音抑制」（`audio_input_noise_suppression`：`off`・`webrtc`・`rnnoise`、既定 `off`）。どの入力モードでも、
  AEC と入力ゲインの後にかける（`audio/denoise.py`）。ASR と MaAI は同じ音を使うので、両方に効く。
- WebRTC：LiveKit の APM の雑音抑制だけを有効にした、AEC とは別のインスタンス。10 ms 単位。
- RNNoise：pyrnnoise に同梱の RNNoise のライブラリを ctypes で直接呼ぶ（`import pyrnnoise` はファイル入出力のために PyAV を読むので使わない）。
  48 kHz・10 ms 単位なので、soxr で 16 kHz と 48 kHz の間を連続して変換する。
- 出力は入力と同じ長さ（処理の遅れの分は最初に無音を詰める。WebRTC は 10 ms、RNNoise は 10 ms と変換の遅れ）。WAV の 2 ch 目と並べて送る音との対応を保つため。
- 使えないとき（ライブラリが無いなど）はログに出して、雑音抑制なしで続ける。設定はワーカーの起動時に読むので、変えたら音声対話を始め直す。

### WAV モード

- ファイルを先頭から順に読み、マイクと同様に 320 サンプル単位で送出
- `wav_local_playback` を指定した場合は、監視用にローカルでも同時に再生

---

## ASR

ASR ワーカーが音声を受信してから VAD・推論を経て認識結果を送出するまでの処理を、ASR エンジン・VAD・ターン検出の組み合わせごとに、
データが通るキューとスレッドに沿って記述する。

### 役割

| 要素 | 主なファイル | 役割 |
|------|----------------|------|
| 音声入力ワーカー | `audio/input_worker.py`（`dialog/worker.py` 経由で起動） | マイクの PCM を `audio_frame` / `system_audio_frame` として送信。対話サーバーが ASR・MaAI などへ振り分け |
| ASR ワーカー | `asr/worker.py` | `ASRService` を 1 インスタンス保持する別プロセス。`ipc/orchestrator_supervisor.py` の `OrchestratorSupervisor`（`dialog/worker.py` が使用）から起動。`--host` / `--port` で対話サーバー内の `JsonLineEventServer` に接続し、IPC で `ASRService` を駆動 |
| ASR 本体 | `asr/service.py` の `ASRService` | 生音声キュー、VAD による区間分割、推論キュー、各エンジンの推論、結果のコールバック。各エンジンのループは `asr/engines_*.py` の mixin に実装し、`ASRService` に組み込む |
| Controller | `gui/controller.py` | Python ワーカーとは直接 JSON Lines で通信しない（Duplex 接続は対話サーバーのみ）。`asr_result` などを対話サーバーから受け取り、UI への表示と一部コマンドの中継を担当 |

音声は対話サーバーを経由して ASR ワーカーへ届く。Controller の `_on_audio_input_ipc_event` は
`audio_frame` / `system_audio_frame` を処理しない。

`ASRService` の処理の中心は「生音声キュー → セグメントループ → VAD・特徴抽出 → 推論 → コールバック」である。

### 設定

ASR ワーカーは `--config-json` で設定を受け取る。GUI では `gui/controller.py` の `launch_asr()` が画面の入力値を検証し、
`_build_asr_config_dict()`（実体は `dialog/settings.py` の `build_asr_config_dict`）で `asr_config` を構築する。

| キー | 内容 |
|---|---|
| `asr_engine` | `sherpa-onnx`・`faster-whisper`・`qwen3-asr`・`nemotron-asr`・`external-asr-tcp`・`chirp3-asr` など（Chirp のモデル名 `chirp_3` は別のフィールドで指定）。GUI のエンジン選択に表示するのは `chirp3-asr` を除く 12 種（`sherpa-onnx`・`qwen3-asr-onnx`・`nemotron-onnx`・`sensevoice-onnx`・`parakeet-onnx`・`cohere-asr-onnx`・`funasr-nano-onnx`・`speechcpp-asr`・`faster-whisper`・`qwen3-asr`・`nemotron-asr`・`external-asr-tcp`） |
| `vad_backend` | `maai` / `silero`。発話区間の切り出しと特徴抽出に影響 |
| `turn_backend` | `maai` / `smart-turn`。ターン確率の算出元 |
| `nonstream_pseudo_enabled` / `nonstream_pseudo_chunk_sec` | 非ストリーミングのエンジンの疑似ストリーミング |
| `qwen_min_return_audio_sec` | `qwen3-asr` のセッションに `min_return_audio_sec` として渡す値 |
| `asr_nemotron_python` | `nemotron-asr` の専用 venv の Python |
| 接続先 | 対話サーバー構成では `--host` / `--port` の 1 接続のみ。設定 JSON に旧キー `audio_ingress_*` が残っていても `ASRService` に渡さず無視（`asr/worker.py` で `config.pop`） |

エンジンごとの導入と注意点は [asr-engine-notes.md](./asr-engine-notes.md)、sherpa-onnx のパッチ版は
[sherpa-onnx-custom-build.md](./sherpa-onnx-custom-build.md) を参照のこと。

### 起動

- `_begin_asr_starting()` で画面の状態を「起動中」にし、`_launch_asr_worker()` がバックグラウンドで
  `_start_module_via_orchestrator("asr", ...)` を呼び出す。
- IPC の待ち受け：`JsonLineEventServer` は対話サーバープロセス（`dialog/worker.py`）内。Controller はサーバーを立てない
- ワーカープロセスの起動：`OrchestratorSupervisor.ensure_asr`（内部で `ASRWorkerProcessService` を使用）
- ready の待機：`module_started` イベントと `threading.Event`（テスト画面・ASR タブの起動フロー）
- ワーカー側：`asr/worker.py` は `--config-json` を読み込み、`ASRService(...)` を生成して `start()` する。起動に成功すると `ready` を送る。

### 音声チャンクの受信

#### 受信経路

| 経路 | 対象 | 内容 |
|---|---|---|
| 対話サーバー経由（TCP） | `asr/worker.py`・`asr/compare_worker.py` | `--host` / `--port` で対話サーバー内の `JsonLineEventServer` に 1 本だけ接続。同じ TCP 接続上でサーバーが `send` で送る `type=audio_frame` / `system_audio_frame` を `JsonLineEventClient.start_reader` で受信 |
| 標準入力 | `asr/compare_worker.py` | 上記に加え、子プロセスの標準入力でも `audio_frame:{json}`・`system_audio:{base64}`・`system_audio_frame:{json}` の形式で受信 |

`asr/worker.py` の標準入力が受け付けるのは次のコマンドだけである。

- `stop`：ワーカーの終了
- `force_finalize`：現在の発話の強制確定を依頼。結果は `force_finalize_status` イベント（`is_speaking`・`request_id`）で通知

#### 起動直後の音声の破棄（`audio_accepting`）

- `ASRService.start()` が完了し `emit("ready")` するまでは `audio_accepting = False`
- この間に届いた音声は `service.raw_audio_queue` に積まず、バッファごと破棄（重いモデルの起動中に溜まった古い音声を処理しないため）

#### ワーカー内での再チャンク化（`target_chunk_size`）

`asr/worker.py` は、音声入力ワーカーから届く可変長の `audio` を常に 320 サンプル（16-bit モノラルで 640 バイト）単位に
切り直してから VAD・推論のパイプラインへ渡す。320 サンプルは `audio/input_worker.py` の送信単位と同じで、
Silero / MaAI が必要とする長さへの再チャンク化は `ASRService` 側で行う。
`vad_backend` に応じて 512 サンプル（`silero`）または `sample_rate / frame_rate`（それ以外）に切り替えるのは
`asr/compare_worker.py` の経路である（[ASR Compare Worker](#asr-compare-workerasrcompare_workerpy)）。

各チャンクは `AudioFrame(audio, sample_start, sample_end, ts_start, ts_end)` とし、
`queue_put(service.raw_audio_queue, frame, policy=DROP_OLDEST)` で `ASRService.raw_audio_queue` へ投入する。

- 時刻：ペイロードの `ts_start` を起点に、連続するチャンクの `ts_end` を積み上げて求める。`sample_start` も同様に連番で付与する。
- 再同期：受信したペイロードの `sample_start` が期待値から 1 サンプルを超えてずれた場合（上流でフレームが破棄された場合など）は、
  ペイロードの `sample_start` / `ts_start` に再同期する。
- `origin_ts`：各フレームに、ペイロードの `emitted_at`（無い場合は受信時刻）を送出時の壁時計時刻として付与する。

#### システム音声（ch2）

- `system_audio_frame`（比較ワーカーでは標準入力の `system_audio:` / `system_audio_frame:` も）は `service.inject_system_audio` へ
- 内部ではサンプル列を `_system_audio_buffer_vad`（deque、最大 2 秒分）に格納
- MaAI の VAD が `MaaiInput` の ch2 に入力する際に `_get_system_audio_chunk` で取り出し（不足分は 0 で補完）

### `ASRService` 内部のキューとサンプル時計

| 名前 | 内容 |
|---|---|
| `raw_audio_queue` | 全エンジン共通の生 PCM フレームの入口。`AudioFrame`（float32 の音声、サンプル範囲、時刻 `ts_start` / `ts_end`、`origin_ts`）を先入れ先出しで保持 |
| `inference_queue` | VAD で発話を区切って一括推論するエンジン（`faster-whisper` と `SHERPA_DECODE_ENGINES`）の区間分割ループ（`_vad_loop`）が積む推論ジョブ。要素はおおむね `{"audio", "start_ts", "end_ts", "result_mode": "partial"\|"final", ...}`。`maxsize` は `inference_queue_size`（既定 4）で、`DROP_OLDEST` により古いジョブが破棄される場合あり |
| `turn_audio_queue` | smart-turn 用に `_run_segment_loop` が積むフレーム（`maxsize=32`、`DROP_OLDEST`）。積むのは `turn_backend` が `maai` 以外で、`emit_to_inference` または `forward_to_turn_queue` が有効なとき。VAD ループを持たないストリーミングエンジンで `turn_backend == "smart-turn"` のとき、`_turn_segment_loop` が別途 VAD をかけて消費 |
| `smart_turn_queue` | smart-turn の推論対象となる発話区間（`maxsize=4`） |
| `transcript_queue` | `_publish_result` が積む `ASRResult`（コールバックと併用。verbose ログ用） |

サンプル時計（`_sample_clock_base_ts` / `_sample_clock_cursor`）：

- 外部から届いた `AudioFrame` は、`_to_audio_frame` で絶対位置の `sample_start` / `sample_end` と整合するように基準時刻を合わせる
- VAD が返す区間の境界が `None` の場合は、`_resolve_segment_bounds` で音声長から補完

### スレッド構成と推論方式

スレッド構成は `ASRService._build_engine_threads()` と、`start()` で追加するスレッドで決まる。

| `asr_engine` | メインループ | 推論 |
|--------------|----------------|------|
| `sherpa-onnx`・`qwen3-asr-onnx`・`sensevoice-onnx`・`parakeet-onnx`・`cohere-asr-onnx`・`funasr-nano-onnx` | `_vad_loop` + `_asr_loop` | sherpa-onnx の `OfflineRecognizer`（発話区間単位の非ストリーミング認識） |
| `speechcpp-asr` | `_vad_loop` + `_asr_loop` | speech.cpp のワーカーへ発話全体を渡す方式。デコード処理は sherpa-onnx 系と共通 |
| `faster-whisper` | `_vad_loop` + `_asr_loop` | バッチ推論 |
| `qwen3-asr` | `_qwen_stream_loop` のみ（内部で `_qwen_stream_ws_loop` を実行し、失敗時は再試行）。VAD はループ内で実行 | HTTP / WebSocket のセッション |
| `nemotron-asr` | `_nemotron_stream_loop` のみ | 専用 venv のサブプロセス（`asr/nemotron_asr_server.py`）へ逐次投入。VAD と発話の区切りはアプリ側 |
| `nemotron-onnx` | `_nemotron_onnx_stream_loop` のみ | 同一プロセス内の sherpa-onnx の `OnlineRecognizer` による逐次ストリーミング認識 |
| `chirp3-asr` | `_chirp3_stream_loop` のみ | Google Cloud Speech-to-Text V2 によるストリーミング認識 |
| `external-asr-tcp` | `_external_asr_audio_send_loop` + `_external_asr_tcp_loop` | 外部サーバーへの音声転送 |

`SHERPA_DECODE_ENGINES` は `sherpa-onnx`・`qwen3-asr-onnx`・`parakeet-onnx`・`sensevoice-onnx`・`funasr-nano-onnx`・
`cohere-asr-onnx`・`speechcpp-asr` の 7 種である。

ターン検出による追加スレッドは次のとおり。

| `turn_backend` | 追加スレッド |
|----------------|--------------|
| `maai` | なし。ターンの推論は MaAI ワーカー側で行うため、ASR 内ではターン用のループを起動しない |
| `smart-turn` | `_smart_turn_loop`。通常の VAD ループを持たない `external-asr-tcp`・`chirp3-asr`・`nemotron-asr`・`nemotron-onnx` では、セグメント化用の `_turn_segment_loop` も追加 |

`qwen3-asr` はストリームループ内で発話区間を `smart_turn_queue` へ積むため、`_turn_segment_loop` を起動しない。

`on_maai_result` が設定されている場合、`_maai_result_emit_loop` が `_maai_result_event_queue`（`maxsize=64`）から取り出して
コールバックを呼ぶ（MaAI 結果の非同期送出）。

### VAD（`_get_vad_score`）

`_run_segment_loop` はフレームごとに `vad_score = _get_vad_score(audio_chunk, frame)` を求める。
`None` が返ったフレームは無視する（発話状態は遷移しない）。

`vad_backend == "silero"`：

- 入力を 512 サンプル単位に区切って Silero VAD（ONNX）で推論
- 直近のスコアを `_latest_silero_vad_score` に保持（MaAI 結果のペイロードに `silero_vad_score` として付加）
- プロット用に `{"silero_vad_plot_only": True}` を `_emit_maai_result` で送出

`vad_backend == "maai"`：

- 音声を `_vap_maai_chunk_size`（`sample_rate / frame_rate`）単位に分割
- VAD 用の MaAI（`_maai_instance`。`Maai` ライブラリ）へ `put_chunk`。ch2 には `_get_system_audio_chunk(...)` の出力
- `get_result()` で同期的に結果を取得し、`result["vad"][0]` を VAD スコアとして返却
- 取得した結果は `_emit_maai_result` でプロット用に送出

### 区間分割ループ `_run_segment_loop`

処理の中核であり、`SegmentLoopHooks` によって動作が変わる。

| 設定 | 呼び出し元 | 動作 |
|---|---|---|
| `emit_to_inference=True` | `_vad_loop` | `turn_audio_queue` への転送、擬似ストリーミングの partial ジョブと final ジョブの `inference_queue` への投入、`force_finalize` の処理、`speech_active` に応じた `force_finalize_speaking` の制御 |
| `emit_to_inference=False` | `_turn_segment_loop` | ターン検出用の VAD 区間分割。`smart_turn_queue` への投入と `on_segment_*` のみ（推論キューには積まない） |

ストリーミングエンジン（`qwen3-asr`・`nemotron-*`・`external-asr-tcp`）は、`emit_to_inference=False`・`forward_to_turn_queue=True` に
`on_frame` などのフックを渡してこのループを利用する。

#### 発話の開始（`speech_active: False → True`）

条件：`vad_score > vad_start_threshold` が `vad_start_hold_sec` に相当するフレーム数だけ続くこと。

1. `self._last_partial_for_flush = None`（前の発話の partial キャッシュを無効化）
2. `hooks.emit_to_inference` が有効なら `_set_force_finalize_speaking(True)`
3. `current_segment_frames = list(pre_buffer)`（プリロールを含む）
4. `segment_start_ts = first_frame.ts_start`
5. 擬似ストリーミング：`nonstream_pseudo_enabled` かつ `nonstream_pseudo_chunk_sec > 0` なら
   - `pseudo_partial_frames = list(pre_buffer)`
   - `bootstrap_payload`（`result_mode="partial"`）を `inference_queue` へ
6. `next_partial_sample_end = frame.sample_end + pseudo_chunk_samples`

#### 発話中

- 各フレームを `current_segment_frames` に追加
- 擬似ストリーミングの partial：`frame.sample_end >= next_partial_sample_end` となるたびに、`pseudo_partial_frames` 全体から
  `partial_payload`（`result_mode="partial"`）を構築して `inference_queue` へ投入し、`next_partial_sample_end` をチャンク幅だけ進める

#### `force_finalize`（明示的な確定）

`hooks.emit_to_inference` が有効で、`_force_finalize_event` がセットされている場合の処理である。

- `current_segment_frames` がある場合
  - `len(segment_audio) >= sample_rate * min_segment_sec` なら final のペイロードを `inference_queue` へ
  - それ未満なら `_flush_with_last_partial(frame.ts_end)`（直近の partial のテキストを final として `_publish_result`）と結果の通知
- フレームが無い場合は、partial による確定か `no_audio` の結果の通知
- その後、区間の状態と擬似ストリーミングの状態をすべてリセット

#### VAD による通常の終了（無音側）

`vad_score < vad_end_threshold` が `vad_end_hold_sec` に相当するフレーム数だけ続いた場合の処理である。

- `len(segment_audio) >= min_samples` の場合
  - `hooks.emit_to_inference` が有効なら `result_mode="final"` で `inference_queue` へ
  - `smart-turn` なら同じペイロードを `smart_turn_queue` へも投入
- 未満で、`hooks.emit_to_inference` が有効な場合
  - `_flush_with_last_partial(...)`（短い発話が final のキューに積まれない場合の代替）

#### `_flush_with_last_partial`（短い発話でキューに積まれない場合）

- `_publish_result` を通った最後のストリーミング partial（`is_final=False` かつ `is_streaming=True`）を `_last_partial_for_flush` に保持
- 必要に応じて、同じテキストを `is_final=True`・`is_streaming=True` の結果として再送出（時刻は引数の `end_ts` とキャッシュした `segment_start_ts` など）

### 一括推論 `_asr_loop`（Whisper・sherpa-onnx 系）

非ストリーミングのエンジンでも、`nonstream_pseudo_enabled` が有効であれば VAD 区間内でチャンクごとに partial の推論を行い、
発話の終端で final を出力する（疑似ストリーミング）。`_asr_loop` は `inference_queue` からジョブを取得して処理する。

1. `result_mode`（`partial` / `final`）から `is_partial` を判定
2. 長さの確認
   - partial：0.05 秒未満は破棄（`force_finalize` の要求に対応するジョブなら結果を通知）
   - final：`min_segment_sec` 未満は破棄
3. `_set_inference_busy(True)` の間に `_decode_segment` を実行
   - `is_final = not is_partial`
   - `is_streaming = is_partial or 擬似ストリーミングが有効`（擬似ストリーミング中は final も `is_streaming=True`）

| エンジン | 関数 | 内容 |
|---|---|---|
| `faster-whisper` | `_decode_with_faster_whisper()`（`asr/engines_local.py`） | `transcribe` のセグメントごとに `_publish_result`。セグメントの相対時刻を `start_ts` / `end_ts` に加算して絶対時刻に変換し、`segment_start_ts` / `segment_end_ts` を付けて出力 |
| `SHERPA_DECODE_ENGINES`（sherpa-onnx 系・speech.cpp） | `_decode_with_sherpa()`（`asr/engines_local.py`） | `_sherpa_needs_lead_silence` が立つエンジン（`sherpa-onnx` の ReazonSpeech zipformer）では先頭の安定化のため 0.5 秒の無音を付加してデコード。発話全体を再推論する他のエンジンには付加しない。`decode_stream` の結果を 1 本のテキストとして `_publish_result` |

#### `_publish_result`

- `latency_sec`：マイクのフレームが送り出されてから結果が出るまでの壁時計の時間（`_end_to_end_latency_sec`）。
  フレームの `origin_ts`（`emitted_at`）を起点とし、得られない経路では音声時刻を基準とする `_report_latency_sec` で代替
- `utterance_index`：partial の系列で共通。final で区切られると次の番号へ
- `on_result(ASRResult)` → `asr/worker.py` では `emit("result", ...)`
- partial のときは `_last_partial_for_flush` を更新し、final のときはクリア

### ストリーミングエンジン（VAD ループを持たない構成）

#### `qwen3-asr`（WebSocket）

- `raw_audio_queue` から読み出し、VAD でセッションを開始・終了
- チャンクを `audio_chunk` としてサーバーへ送信し、応答の `type` から `is_final` を判定
- `_publish_result` は原則 `is_streaming=True`（サーバーの応答仕様に従う）
- VAD の終了時は `qwen_finish_on_vad_end` などに従って `finish_session` を送り、最終結果を `_publish_response` で送出

#### `chirp3-asr`

- `_chirp3_stream_loop()` → `_chirp3_streaming_recognize_once()`。gRPC の `streaming_recognize` を使い、`result.is_final` をそのまま `is_final` に設定（`is_streaming=True`）

#### `external-asr-tcp`

- 送信ループ：`_run_segment_loop(forward_to_turn_queue=True)` で PCM を外部へ送信
- 受信ループ（`_external_asr_tcp_loop()`）：TCP で 1 行ごとの JSON を厳密に解釈する。`is_final` / `is_streaming` はペイロードに必須。
  区間の時刻は `_external_*` のロックで保護したヒントと照合

#### `nemotron-asr`（専用 venv への分離）

`nemo-toolkit` はアプリ本体の依存（`torch` / `transformers`）と競合するため、専用の venv に分離し、別プロセスで推論する。

- `ASRService._init_engine_nemotron` が `asr/nemotron_asr_server.py` を専用 venv の Python（設定 `asr_nemotron_python`、既定は
  `.venv_nemotron/Scripts/python.exe`、相対パスはリポジトリのルート基準で解決）で起動し、ローカルソケットで接続する。
- 専用 venv は `scripts/setup_nemotron_venv.ps1` で作成する（`nemo-toolkit[asr]` と CUDA 版 torch）。
- サブプロセスへ委譲するのは推論のみである。VAD と発話の区切りはアプリ側の `_nemotron_stream_loop` が担い、
  `qwen3-asr` と同様にストリームループの起動時に接続する。

### ターン検出

#### MaAI（`turn_backend == "maai"`）

ターンの推論は MaAI ワーカー（`maai/worker.py`）で行う。ASR 内ではターン用のスレッドを起動せず、`turn_audio_queue` にもフレームを積まない。
`vad_backend == "maai"` の場合、ASR 内の MaAI（`_maai_instance`）は VAD スコアの算出にのみ使い、その結果は
`maai_result` としてプロット用に送出する（[VAD](#vad_get_vad_score)）。

#### smart-turn（`turn_backend == "smart-turn"`）

- `turn_audio_queue` のフレームを `_turn_segment_loop` が改めて VAD にかけ、区間を `smart_turn_queue` へ積む（VAD ループを持たないエンジンの場合）
- `smart_turn_queue` の発話区間の音声から、ONNX と `WhisperFeatureExtractor` で確率を算出（末尾 8 秒分を使用）
- `on_turn_probability` で通知

### コールバックと IPC イベント

`ASRService` のコールバック：

| コールバック | 内容 |
|---|---|
| `on_result` | `ASRResult`（`text`・`is_final`・`is_streaming`・`utterance_index` など） |
| `on_turn_probability` | ターン確率と時刻 |
| `on_maai_result` | MaAI のプロット用の各種系列（dict 形式） |
| `on_inference_state` | 推論中・待機中の状態（UI と対話サーバーの待ち状態に連動） |
| `on_force_finalize_outcome` | 強制確定の結果 |

`asr/worker.py` が対話サーバーへ送出するイベント：

| イベント | 発生元 |
|----------|--------|
| `ready` / `stopped` / `fatal` / `log` / `error` | ライフサイクル・ログ |
| `result` | `on_result` |
| `turn` | `on_turn_probability`（高頻度キュー経由） |
| `maai_result` | `on_maai_result`（同上） |
| `inference_state` | 推論中かどうか |
| `force_finalize_status` | 標準入力の `force_finalize` を受けた時点の発話状態 |
| `force_finalize_outcome` | `force_finalize` の結果（`_emit_force_finalize_outcome`、発話中でない場合の `not_speaking` を含む） |

`turn` / `maai_result` は `high_freq_event_queue`（`maxsize=64`、`DROP_OLDEST`）に積み、別スレッドが `emit` する。

### 対話サーバーと Controller での扱い

主な経路は次のとおりである。

`asr/worker.py` →（`JsonLineEventServer`）→ `dialog/runtime.py` の `VoiceDialogRuntime.forward_asr_ingress_event` →
内部の `event_queue` → `dialog/worker.py` の `_handle_event`（`asr_result` / `turn` など）→ `DialogOrchestrator.on_*` など →
必要に応じて Duplex で Controller へ `asr_result` / `history_snapshot` などを送出

| イベント | 扱い |
|---|---|
| `result` | `forward_asr_ingress_event` で `asr_result` に正規化し、受信時刻を `final_arrival_ts` として付与して `DialogOrchestrator.on_asr_result` へ |
| `maai_result` | `VoiceDialogRuntime.handle_maai_result` が `DialogOrchestrator.on_maai_result` に渡すとともに、`from_orchestrator=True` を付けてプロット用に Controller へ送出 |
| `turn` | `VoiceDialogRuntime.handle_turn` で処理。沈黙後のターン取得は、対話サーバー内部の `on_tick` と `idle take turn` でも補完 |
| `force_finalize_status` / `force_finalize_outcome` | `VoiceDialogRuntime` のハンドラで処理。Duplex の `force_finalize` コマンドは対話サーバーが ASR の子プロセスへ中継 |

Controller 側では、`_on_dialog_orchestrator_event` が `gui/orchestrator_events.py` のハンドラ（`asr_result` は `_orch_ev_asr_result`）へ振り分け、
`_append_asr_result_event` で表示する。

| イベント | Controller での表示 |
|---|---|
| `ready` / `module_started` | 対話サーバーへの接続・モジュールの準備完了 |
| `asr_result` | 認識結果の行の更新（`is_final` / `is_streaming` などは対話サーバー側で正規化済み） |
| `turn` / `maai_result` | プロットとログ |

### 構成別の早見表

| 構成 | 音声の入口 | VAD | 推論キュー | 結果 |
|------|----------|-----|------------|------|
| sherpa-onnx + MaAI VAD + MaAI ターン | `raw_audio_queue` | `_vad_loop` 内の MaAI `vad[0]` | partial / final を `inference_queue` へ | `sherpa-onnx` の結果と `maai_result`（VAD）。ターンは MaAI ワーカー |
| Whisper + Silero VAD | `raw_audio_queue` | Silero（512 サンプル） | 同上 | `faster-whisper` の結果 |
| Qwen（WebSocket）+ MaAI ターン | `raw_audio_queue`（ストリームループ内で VAD） | ループ内 | なし（WebSocket で直接送信） | `qwen3-asr` の応答 |
| External TCP + smart-turn | 送信側の区間分割ループ + TCP 受信 | 送信側の `_run_segment_loop` + `_turn_segment_loop` | 外部 | JSON の `is_final` |
| 短い発話で final のキューに積まれない場合 | 同上 | 終了または `force_finalize` | スキップされる場合あり | `_flush_with_last_partial` で最終テキスト化 |

### ASR Compare Worker（`asr/compare_worker.py`）

同一の音声ストリームを 2 種類の ASR エンジンへ並列に入力し、認識結果を比較表示するためのデバッグ・評価用ワーカーである。
通常の音声対話の経路とは独立に有効化する（`asr_compare_mode_enabled` など）。

1. `--config-json` から `compare_top_engine` と `compare_bottom_engine` を取得する（必須）。いずれかが欠けていれば FATAL で終了する。
2. `ASRService` を 2 インスタンス生成し、`_services` リストで保持する。
3. 対話サーバー内の比較用 `JsonLineEventServer`（Supervisor が `--host` / `--port` で指定）へ 1 本の `JsonLineEventClient` で接続し、
   同じ接続で音声の受信と結果の送信を行う。標準入力でも音声（`audio_frame:` / `system_audio:` / `system_audio_frame:`）と
   `stop` / `force_finalize` を受け付ける。
4. 受信した PCM をバッファし、VAD バックエンドに応じた `target_chunk_size`（Silero では 512、それ以外はサンプルレートと
   フレームレートから算出）ごとに `AudioFrame` を作る。同じフレームを top 用・bottom 用のキュー（`top_queue` / `bottom_queue`、
   各サービスの `external_raw_audio_queue`）の両方へ投入する。
5. 各サービスの `on_result`・ターン確率・ログなどを、`compare_slot`（`top` / `bottom`）を付けて対話サーバーへ送信する。
   ログには `[top:<engine>]` / `[bottom:<engine>]` の接頭辞を付ける。
6. 標準入力の `force_finalize` を受けると、top・bottom の両方に `request_force_finalize_current_utterance` を発行し、
   `compare_slot` 付きの `force_finalize_status` を返す。

### 関連ソースと主要な関数

| ファイル | 内容・主要な関数 |
|---|---|
| `asr/worker.py` | `main`。再チャンク化、`raw_audio_queue`、IPC の送出、高頻度キュー |
| `asr/service.py` | `ASRService` 全体（`start`・`_asr_loop` など） |
| `asr/vad_turn.py` | VAD（`_get_vad_score`）、`_turn_segment_loop`、smart-turn |
| `asr/engines_*.py` | 各エンジンの初期化・推論・ストリームループ（`_qwen_stream_loop`・`_chirp3_stream_loop`・`_decode_with_faster_whisper`・`_decode_with_sherpa` など） |
| `dialog/runtime.py` | `forward_asr_ingress_event`、`handle_maai_result` |
| `dialog/worker.py` | `_handle_event` とイベントの振り分け表 |
| `ipc/orchestrator_supervisor.py` | `ensure_asr`・`stop_module`・`mark_asr_ready` など（ワーカーへの文字列コマンドは `supervisor.asr.send_command`） |
| `gui/controller.py` | `launch_asr`・`_launch_asr_worker`・`_start_module_via_orchestrator`・`stop_asr`・`_on_dialog_orchestrator_event`（`asr_result` の表示）・`_append_asr_result_event` |

---

## MaAI（MaAI Worker）

### 役割

- ユーザー音声のチャネルと、構成によってはシステム再生音声のチャネルを `maai` ライブラリに供給し、次のモデルを独立したプロセスで実行する。
  - ターン検出用の VAP（`turn_vap_enabled` が有効な場合）
  - 相槌（バックチャネル）用のモデル
  - nod_para（頷きパラメータ）用のモデル
- ASR ワーカー内の MaAI と役割が重なる場合があるが、本ワーカーは独立したプロセスであり、対話サーバーが公開する MaAI 用の
  `JsonLineEventServer` に 1 本だけ接続する。

### 処理の流れ

1. `--config-json` から、デバイス・フレームレート・言語・モデル種別・相槌 / 頷き / ターン検出のモードを読み取る。
2. `Maai` インスタンスと `QueueInput`（`maai.input.Base` の拡張）を用意する。`push(frame)` で float 列をエンジンへ渡す。
3. `JsonLineEventClient` で対話サーバーの MaAI 用ポートに接続し、JSON 行の `audio_frame` / `system_audio_frame` などを受信して `QueueInput.push` へ渡す。
4. MaAI のフレームごとの結果をイベントとして上流へ送る（`vad` / `p_now` / 各種確率ベクトルなど）。
5. システム音声のタイムライン（`source_sample_end` など）を累積し、WAV ファイル入力時の位相合わせに用いる。

### 入出力・イベント

| イベント | 内容 |
|---|---|
| `audio_frame` / `system_audio_frame`（受信） | ユーザー音声 / システム音声のフレーム |
| `maai_result`（送信） | VAP の結果（`vad` / `p_now` など） |
| `maai_bc_result`（送信） | 相槌モデルの結果 |
| `maai_nod_result`（送信） | 頷きモデル（nod_para）の結果 |

### 設定

| キー（`--config-json`） | 内容 |
|---|---|
| `mode` | `bc` / `nod_para` / `both` / `none`（既定は `both`） |
| `turn_vap_enabled` | ターン検出用 VAP の有効化 |
| `use_mimi_onnx` | Mimi encoder を ONNX Runtime で実行する |
| `use_local_mimi_onnx` | True のときだけ `maai_root` または明示パスのローカル ONNX を使う。False なら Hugging Face（`maai-kyoto/continuous-mimi-onnx`）から取得する |
| `inference_chunk_frames` | Mimi encoder で N フレームをまとめて 1 回の ONNX 呼び出しで推論する |
| `mimi_onnx_cpu_intra_threads` / `mimi_onnx_cpu_inter_threads` | Mimi の ONNX Runtime スレッド数（UI から設定。CPU EP のみ有効。cuda では 0 固定） |

`maai` は upstream main の固定コミットを使用する。ONNX で実行するのは Mimi encoder のみで、相槌・頷き・VAP の transformer は PyTorch で実行する。

### 対話サーバーとの関係

- nod_para の発火は、`p_nod` の閾値・連続フレーム数・クールダウンで判定する。発火時はバッファに蓄えたモデル予測を平均と多数決で集計し、モーションに渡す。
- 頷きの回数（1 回・2 回・3 回以上）と振り上げは、フレームごとに `utils.nod_para_frame_decision` で決めて多数決する。
  maai は回数の確率を `nod_repetitions`、モデル自身の判定を `nod_repetitions_pred` / `nod_swing_up_pred` で返す。
  「MaAIモデル設定」の頷き回数しきい値（t0〜t2）と振り上げしきい値が空欄のときはモデル自身の判定を使う（.pt に入っている
  学習時のしきい値。公式モデルは t0〜t2 が約 0.33、振り上げが 0.43）。数値を入れたときはモデルと同じ規則で決め直す
  （回数は「確率÷しきい値」が最大のもの、振り上げは確率がしきい値以上）。回数のしきい値は 3 つとも数値のときだけ使う。
- 対話サーバー側では、`on_maai_result` を割り込み判定（`user_side`）とターンロックの解除に、`on_maai_bc_result` / `on_maai_nod_result` を
  相槌の TTS 要求と頷きの発火・可視化イベントの生成に用いる。
- MaAI を二重に動かす構成（ASR ワーカー内と本ワーカー）では、設定とポート転送の整合に注意が必要である。
- 頷きモーションの生成と送出は [MMDAgent-EX 連携](#mmdagent-ex-連携ブリッジペーサー頷き) を参照。

---

## LLM

### 役割

- OpenAI 互換 API（LM Studio / OpenAI / Custom の base URL）へ `OpenAI` Python クライアントで接続し、チャット補完を実行する。
- ストリーミング応答では、チャンクを逐次 emit する。TTS 向けに応答を文単位のセグメントへ分割し、`tts_segment` として送る。
- 文の分割は `custom_delimiters` に含まれる区切り文字の位置で行う（外部のトークナイザーは使わない）。各セグメントには、
  TTS へ送る直前に絵文字・括弧書き・単独の「笑」の除去を適用できる（`llm/worker.py` 内の正規表現処理）。
- LLM の呼び出しは LLM ワーカー（`llm/worker.py` と `LLMService`）に一本化している。設定画面の LLM テストも対話サーバー経由である。

### 処理の流れ（`llm/worker.py`）

1. `JsonLineDuplexClient` で、対話サーバー内の `OrchestratorSupervisor` が起動する Duplex サーバー（`--host` / `--port`）に接続する。
   Controller は LLM ワーカーと直接通信しない。
2. `start` コマンドで messages・model・stream フラグなどを受け取る。要求ごとにスレッドを立てて処理する。
3. `LLMService` が接続先ごとのクライアントを生成・キャッシュする。`warmup` コマンドで接続とモデルを事前に温め、初回のレイテンシを抑える。
4. `LLMService.run_chat()` が `chat.completions.create(stream=...)` を実行する。使用できないパラメータで拒否された場合は修正して再送する（`_create_completion()`）。
5. 非ストリーミング時は `assistant_text` を一括で返す。ストリーミング時はチャンクごとに `chunk` を送り、TTS 連携が有効な場合は
   蓄積したテキストを区切り文字で分割して `tts_segment` を送る。完了時にメタデータ（`ttft_sec` / `total_sec` など）を付けた `done` を送る。
6. `cancel` コマンド（`request_id` を指定）で実行中の要求を取り消す。取り消した要求は `error`（`message: "cancelled"`）で終了する。

### `llm/service.py`

| 関数 | 内容 |
|---|---|
| `run_chat` | OpenAI SDK によるチャット補完の実行（ストリーミング / 非ストリーミング） |
| `get_or_create_client` | `backend` に応じて `base_url` と `api_key` を組み立て、接続先ごとにクライアントをキャッシュする |
| `warmup_client_once` | 本番の要求の前に、軽いリクエストで接続とモデルを温める。重複は `_warmup_inflight` で抑止 |
| `resolve_openai_api_key` | API キーを直接指定、または環境変数名で解決する（実装は `llm/common.py`） |

### 音声対話での呼び出し

- `DialogOrchestrator` が `trigger_llm` を返すと、コールバック経由で `VoiceDialogRuntime.llm_dispatch_thread` が
  `dispatch_llm_request` をスレッドで起動し、`supervisor.send_llm(...)` で LLM ワーカーへ `start` を送る。
  API リクエストの送出は対話サーバー側で行い、Controller は `trigger_llm` を受けて表示を更新するのみである。
- LLM ワーカーからの `chunk` は、音声対話では `llm_chunk`、LLM テストでは `llm_test_chat_chunk` として UI へ送る。
- ストリーミング TTS が有効な場合は、LLM ワーカーからの `tts_segment` などを対話サーバーが受け、TTS ワーカー・
  Audio Output ワーカーを Supervisor 経由で駆動する（[LLM からのストリーミング連携](#llm-からのストリーミング連携)）。

### バックエンドとモデルの管理

LLM のバックエンドは LM Studio・OpenAI・Custom（OpenAI 互換 API）を分離し、モデルの入力欄もバックエンドごとに管理する。
モデル一覧の取得などのローカル HTTP は、Controller が `send_to_llm_admin` を送り、対話サーバー内の
`OrchestratorSupervisor.handle_llm_admin_http_command` が `LlmAdminHttp`（`llm/admin_http.py`）で実行する。Controller は RPC の応答で UI を更新する。

| 処理 | 実装 |
|---|---|
| バックエンドの切り替え | `_on_llm_backend_changed()` |
| バックエンドごとのモデルの記憶 | `_llm_model_by_backend` |
| LM Studio のモデル一覧の取得 | `_fetch_lmstudio_models_worker()` → `lmstudio_fetch_models`（`/v1/models`） |
| OpenAI のモデル一覧の取得 | `_fetch_openai_models_worker()` → `openai_fetch_models`（`client.models.list()`） |

LM Studio では、現在のモデル以外の一括アンロードと、明示的なアンロードに対応する。

- 一覧の取得：`/api/v1/models`
- アンロード：`/api/v1/models/unload`（`instance_id` を優先し、互換のため `model` も試行）

関連する関数：`_on_llm_backend_changed()`・`_on_lmstudio_models_fetched()`・`_on_openai_models_fetched()`・`refresh_lmstudio_models()`・
`unload_lmstudio_non_current_models()`・`_unload_lmstudio_models_worker()`（`gui/controller.py`）、`_unload_lmstudio_model()`（`llm/admin_http.py`）

### LLM テスト

UI の入口は `send_llm_test_chat()` である。

1. 入力の検証（`context_len`・`max_tokens`・モデル。Custom ではベース URL と temperature も確認）
2. 履歴の調整（system と最新 N 件）
3. `start_llm_gui_test` のリクエストを作成
4. 対話サーバー経由で LLM ワーカーを起動（`_start_module_via_orchestrator("llm", ...)`）。TTS 連携を有効にした場合は再生経路も準備
5. リクエストを対話サーバーに送信。対話サーバーは `VoiceDialogRuntime.dispatch_llm_gui_test` を別スレッドで実行
6. 対話サーバーは UI へ `llm_test_chat_chunk` / `llm_test_finished` を送り、UI は `llm_test_chat_chunk_signal` で表示を更新する。
   TTS 連携が有効な場合は `tts_segment` ごとに合成して音声出力へ送出する

関連する関数：`send_llm_test_chat()`・`_trim_llm_messages()`・`_ensure_llm_history()`・`_parse_llm_max_tokens()`（`gui/controller.py`）

### 主要な関数

- `gui/controller.py`：`_on_llm_backend_changed`, `refresh_lmstudio_models`, `send_llm_test_chat`, `_on_dialog_orchestrator_event`（`llm_chunk` など）
- `dialog/runtime.py`（`VoiceDialogRuntime`）：`dispatch_llm_request`, `dispatch_llm_gui_test`, `handle_llm_worker_event`
- `ipc/orchestrator_supervisor.py`（`OrchestratorSupervisor`）：`ensure_llm`, `send_llm`
- `llm/worker.py`：`main`
- `llm/service.py`（`LLMService`）：`run_chat`, `get_or_create_client`, `warmup_client_once`

---

## TTS と音声出力

### 役割

| 区分 | 担当 |
|---|---|
| ブリッジ型エンジン（`VOICEVOX` / `TEMPLATE`） | 対話サーバー内の `TTSBridge` が HTTP で呼び出す（Controller は `send_to_tts_bridge` を送るのみ）。`tts/playback_service.py` の `BRIDGE_BACKED_ENGINES` |
| ワーカー型エンジン（`Qwen3-TTS-CPP` / `LFM2.5-Audio` / `VOICEVOX-Core` / `Kokoro-RealtimeTTS` / `Cartesia-Realtime` / `AivisSpeech` / `Supertonic-3` / `Irodori-v4` / `Tsukasa`） | `tts/worker.py` の別プロセスに分離し、GPU や実行環境の衝突を避ける。`tts/playback_service.py` の `WORKER_BACKED_ENGINES` |
| 合成と PCM の振り分け | 対話サーバー内の `TtsRouter`（`tts/routing.py`） |
| 再生 | `audio/output_worker.py` が PCM を実時間で送出する。出力先はローカルのスピーカー・TCP・MMDAgent-EX（SND）のいずれか |

LLM 設定画面のローカル HTTP（LM Studio の `/v1/models`・`/api/v1/models`、OpenAI のモデル一覧）は
[`ipc/orchestrator_supervisor.py`](../src/maai_sds/ipc/orchestrator_supervisor.py) の `OrchestratorSupervisor.handle_llm_admin_http_command` が
実行する（[バックエンドとモデルの管理](#バックエンドとモデルの管理)）。

### TTS ワーカー（`tts/worker.py`）

- `JsonLineDuplexClient` で、対話サーバー内の `OrchestratorSupervisor` が起動する Duplex サーバー（`--host` / `--port`）に接続する。
  Controller はワーカーと直接通信しない。
- `--engine` と `--config-json` に応じてエンジンを初期化する（`ENGINE_FACTORIES`）。起動時に `engine.load()` を実行して `ready` を送信する。
- 要求ごとに PCM（16 kHz・モノラル）を生成し、base64 の `audio` イベントとして返す。その後はコマンドキューで一括合成の `synthesize`、
  またはテキストの逐次入力のストリーミングのコマンド（`start_stream` / `push_text` / `end_stream`、取り消しは `cancel_stream`）を処理する。
- ストリーミングのコマンドは `TEXT_STREAM_ENGINES` に含まれるエンジンが対象である。`Kokoro-RealtimeTTS` はエンジン自身が対応し
  （`NATIVE_TEXT_STREAM_ENGINES`）、その他のエンジンは `TextStreamAdapterSession` で対応する。

| エンジン | 実装 |
|---|---|
| `Qwen3-TTS-CPP` | `Qwen3TTSCppEngine` |
| `VOICEVOX-Core` | `VoicevoxCoreEngine` |
| `Kokoro-RealtimeTTS` | `RealtimeTTSKokoroEngine` |
| `Cartesia-Realtime` | `CartesiaRealtimeEngine` |
| `AivisSpeech` | `AivisSpeechEngine` |
| `Supertonic-3` | `SupertonicEngine` |
| `Irodori-v4` | `make_irodori_engine` |
| `Tsukasa` | `TsukasaEngine` |
| `LFM2.5-Audio` | `Lfm2AudioEngine`（`tts/engines_audiocpp.py`） |

#### audio.cpp で動かすエンジン（`LFM2.5-Audio`・Irodori の audio.cpp 版）

どちらも `tts/engines_audiocpp.py` にあり、`tts/audiocpp_client.py` の `AudioCppServer` で audio.cpp のサーバーに接続する
（つながらなければモデル 1 つだけの設定で自分で立ち上げる）。`synthesize_stream` を持ち、届いた音から少しずつ返す
（`tts/playback_service.py` の `BINARY_STREAM_ENGINES`）。

| エンジン | クラス | 設定キー | 備考 |
|---|---|---|---|
| `LFM2.5-Audio` | `Lfm2AudioEngine` | `lfm2_audio_host` / `lfm2_audio_port`（既定 18082）/ `lfm2_audio_model_id` / `lfm2_audio_model_path`（フォルダー）/ `lfm2_audio_server_exe` | LFM2.5-Audio-1.5B-JP。声は 1 つで選べない（設定画面の話者欄は無効になる）。1 文 約 0.33 秒・最初の声まで 約 50 ms |
| `Irodori-v4`（`irodori_backend` = `audiocpp`） | `IrodoriAudioCppEngine`（`make_irodori_engine` が選ぶ） | `irodori_audiocpp_model`（`500m-v3` / `600m-v3-voicedesign`）/ `irodori_audiocpp_host` / `irodori_audiocpp_port`（既定 18083）。seed・参照 WAV・声質キャプションは他の Irodori と共通（`irodori_seed` / `irodori_reference_audio` / `tts_voice_irodori`） | 600M v3 VoiceDesign は声質キャプションで声を指定する。参照 WAV があればそちらを優先する |

ワーカー設定は `dialog/settings.py` の `build_tts_worker_config` が組み立てる。Docker では両方とも compose の `audiocpp` サービス
（`docker/audiocpp-server.json`）を共有するので、`controller_config.docker.example.json` ではホストを `audiocpp`、ポートを 8080 にしている。

#### `Kokoro-RealtimeTTS` のストリーミング

| メソッド | 動作 |
|---|---|
| `RealtimeTTSKokoroEngine.start_stream()` | セッションの生成 |
| `push_text()` | テキスト断片をキューへ投入 |
| `end_stream()` | 入力終了の通知 |
| `cancel_stream()` | 強制終了 |

低遅延化のための実装：

- ワーカー内で RealtimeTTS の `KokoroEngine` を使い回し、再初期化のコストを削減
- 投入されたテキストを TTS 内で文に分割せず、そのまま 1 回の合成に渡す（`RealtimeTTSKokoroSession`）

関連するクラス：`RealtimeTTSKokoroEngine`・`RealtimeTTSKokoroSession`（`tts/engines.py`）

### TTS ワーカーのライフサイクル

Controller の `_ensure_tts_worker()` と Supervisor の `OrchestratorSupervisor.ensure_tts()` が中核である。

1. ワーカーの再利用可否を判定（同一エンジン・同一設定・ready 済み。Supervisor 側では IPC の接続も確認）
2. 再利用できない場合は既存のワーカーを停止し、新しい `JsonLineDuplexServer` を起動
3. `tts/worker.py` を起動
4. ワーカーの `ready` を待機（既定 180 秒）。`fatal` を受けた場合や、ready 前にプロセスが終了した場合はエラー

関連する関数：`_ensure_tts_worker()`・`_teardown_tts_ipc()`（`gui/controller.py`）、`ensure_tts()`（`ipc/orchestrator_supervisor.py`）

### TTS Bridge（`tts/service.py`）

- HTTP クライアントの `TTSBridge` は、対話サーバー内の `OrchestratorSupervisor` が保持する（コマンド処理は `tts/bridge_handler.py` の `TtsBridgeHandler`）。
  `send_to_tts_bridge` コマンドで、合成・ストリーミング合成・VOICEVOX / AivisSpeech の子プロセスの起動と停止を行う。
- Controller は JSON コマンドの送受信のみを行い、メインプロセスから HTTP を直接呼び出さない。

### モードとテストの入口

エンジンごとのモードは `TTS_STREAM_MODE_BY_ENGINE`（`gui/controller.py`）で定義する。表にないエンジンは `batch_only` として扱い、
現在 `output_streaming` を割り当てたエンジンはない。

| モード | 動作 |
|---|---|
| `batch_only` | 一括生成後に送信・再生 |
| `output_streaming` | テキストは一括入力、音声を逐次受信 |
| `full_streaming` | テキストも分割して投入しながら、音声を逐次出力 |

TTS テストの入口は `gui/controller.py` の `send_mmdagent_tts_test()` である。エンジンごとの入力を組み立て、
`start_tts_gui_test` コマンドを対話サーバーに送る。対話サーバーは `VoiceDialogRuntime.dispatch_tts_gui_test`
を別スレッドで実行し、音声対話の非ストリーミング発話と同じ合成・再生経路を使う。完了は `tts_test_finished` で通知する。

### 合成から再生までの経路（`TtsRouter`）

`TtsRouter`（`tts/routing.py`）は、Supervisor を経由した TTS ワーカー・TTS Bridge への合成要求、L16BE から L16LE への変換、
音声出力ワーカーへの振り分けを担当する。

- 合成要求ごとの `started` / `audio` / `done` / `error` は、`TtsRouter.complete_from_event()` が `request_id` 単位で管理する。

  | イベント | 処理 |
  |---|---|
  | `audio` | base64 をデコードし、`on_chunk` コールバックへ渡す |
  | `done` | 完了イベントを立てる |
  | `error` | 状態にエラーを格納して終了 |

- ワーカーから届くチャンク（L16BE）を L16LE に変換
- 可聴判定（`audioop.max(...) > 120`）で最初に聞こえる音までの遅延を計測し、体感遅延として UI に表示（`dialog/runtime.py`）
- PCM は `tts_play_chunk` として `OrchestratorSupervisor.send_audio_output` から `audio/output_worker.py` へ送出。失敗時のみ MaAI への直接注入にフォールバック
- 音声出力ワーカーは、出力先に応じてローカル再生または MMDAgent-EX（`MmdAudioPacer`）へ送出
- MMDAgent-EX への送出では、発話の終端で `SNDBRKS` を送信（`audio/mmd_pacer.py`。WebSocket 接続ではセグメントの区切りでも送信）

Controller の `_on_tts_ipc_event()` は、`ready` / `log` / `fatal` / `bridge_*` のみを扱う。

関連する関数：`synthesize_l16be()`・`route_pcm_l16le()`（`tts/routing.py`）、`_synthesize_segment_pcm_l16le()`・`_route_pcm_l16le_with_audible()`（`dialog/runtime.py`）

### LLM からのストリーミング連携

音声対話と LLM テストは、同じ経路で LLM の出力を TTS に渡す。LLM から TTS への受け渡しは対話サーバー内の `VoiceDialogRuntime` が担当する。

1. LLM ワーカーが応答を受信し、蓄積したテキストを区切り文字で分割した `tts_segment` を対話サーバーに送信
2. `VoiceDialogRuntime._llm_play_tts_segment()` がセグメントを合成（`TtsRouter.synthesize_l16be()`）
3. `Cartesia-Realtime`・`Kokoro-RealtimeTTS`・`Qwen3-TTS-CPP`・`Irodori-v4`・`LFM2.5-Audio`（`BINARY_STREAM_ENGINES`）は合成しながらチャンクを受信して逐次送出
   （`Irodori-v4` は speech.cpp 版・audio.cpp 版のみ。PyTorch 版は 1 文をまとめて返す）。その他のエンジンは合成結果をまとめて送出
4. PCM は `tts_play_chunk` として音声出力ワーカーへ送られ、MMDAgent-EX へ逐次送出
5. セグメントの区切りで `tts_play_break` を送信

LLM のストリーミングと TTS の合成は `request_id` で対応付ける。

### 音声対話での典型的な経路

1. 非ストリーミング：対話サーバーが `request_nonstream_tts` / `request_opening_tts` / `request_scheduled_exit_tts` を発行する。
   合成・再生・`on_tts_done` までは対話サーバー（`dialog/worker.py`）内で完結する。Controller はログや履歴の表示のために
   観測用として受信するのみである（経路の二重化を避けるため）。
2. ストリーミング：対話サーバー内で LLM ワーカーのイベントを処理し、TTS ワーカーと Audio Output ワーカーを `OrchestratorSupervisor` 経由で駆動する。
   送出完了の `tts_mmd_send_complete` に含まれる `burst_compensation_ms` などは、`VoiceDialogRuntime` のイベントハンドラで処理する。

TTS 由来のシステム音声（ch2）の経路は、音声出力ワーカーの `MmdAudioPacer` への送出や、`inject_system_audio` の Duplex コマンドなど、
設定と出力モードに依存する。

### Audio Output Worker（`audio/output_worker.py`）

- `OrchestratorSupervisor` との Duplex 経由で、`tts_play_chunk` の PCM やセグメント制御（`tts_play_break` など）を受け取る
  （Controller は対話サーバーにコマンドを送るのみ）。
- `output_mode` によって出力先を切り替える。

| `output_mode` | 動作 |
|---|---|
| `mmd` | `remote_plugin_apply_settings` を初めて受け取った時点で `MmdRemoteBridge` を生成・設定し、MMDAgent-EX との接続後に `MmdAudioPacer` で SND を送出する。接続前に届いた再生操作は保持し、接続後に順に流す |
| `local` | `sounddevice` のコールバックモードで再生する（`_LocalCallbackAdapter`。デバイスのクロックに送出の間隔を委ねる。バーストなし） |
| `tcp` | ソケットと `SegmentedAudioPacer` で送出する（バーストなし） |

- 先頭のバースト（`mmd_burst_snd_frames`）は `mmd` モードの `MmdAudioPacer` でのみ適用する。`local` / `tcp` では UI のバースト設定を使わない。
- MMDAgent-EX への頷きのコマンドは、対話サーバーの `MmdRuntime` から `mmd_bind_head` / `mmd_send_text` で受け取り、`MmdRemoteBridge` から送る。

### 主要な関数

- `gui/controller.py`：`send_mmdagent_tts_test`, `_start_module_via_orchestrator` / `_send_to_worker_via_orchestrator`, `_on_dialog_orchestrator_event`（`_src: tts`）
- `dialog/runtime.py`（`VoiceDialogRuntime`）：`dispatch_tts_gui_test`, `_llm_play_tts_segment`
- `ipc/orchestrator_supervisor.py`（`OrchestratorSupervisor`）：`ensure_tts`, `send_tts`, `handle_tts_bridge_command`
- `tts/routing.py`（`TtsRouter`）：`synthesize_l16be`, `route_pcm_l16le`, `complete_from_event`
- `tts/worker.py`：`main`, 各エンジン

---

## MMDAgent-EX 連携（ブリッジ・ペーサー・頷き）

### 役割

MMDAgent-EX への音声（SND）・頭部動作・まばたきの送出を担う。口パクはアプリから送らず、MMDAgent-EX が受け取った SND 音声から自分で作る（TCP・WebSocket とも同じ）。通信路（`MmdRemoteBridge`）と音声のペーサー（`MmdAudioPacer`）は
Audio Output ワーカー内にあり、頷きの制御（`MmdRuntime`）は対話サーバー内にある。

| コンポーネント | 所在するプロセス | 役割 |
|---|---|---|
| `MmdRemoteBridge`（`mmd/remote_bridge.py`） | Audio Output ワーカー | MMDAgent-EX の Plugin_Remote との通信 |
| `MmdAudioPacer`（`audio/mmd_pacer.py`） | Audio Output ワーカー | TTS の PCM を SND フレームとして送出 |
| `SegmentedAudioPacer`（`audio/segmented_pacer.py`） | Audio Output ワーカー | MMD / TCP 出力に共通のペーサー |
| `MmdRuntime`（`mmd/runtime.py`） | 対話サーバー | 頷きモーションの送出 |
| `maai/nod_motion.py` | 対話サーバー | 頷きモーションの系列の生成 |

### `MmdRemoteBridge`（`mmd/remote_bridge.py`）

- Qt を使わずに MMDAgent-EX の Plugin_Remote と通信する。モードは TCP クライアント・TCP サーバー・WebSocket サーバーの 3 つ。
- 受信ループと送信キューをスレッドで管理する。`send_text` はロックを取って改行付きのテキストコマンドを送る。

プロトコル上の注意点：

- `ensure_snd_segment_preamble`：最初の SND の前に `__AV_START` / `__AV_SETMODEL,<別名>` を送る。Plugin_Remote は
  `__AV_SETMODEL` で割り当てたモデルにだけ音声を渡し、割り当てが無いと音声を捨てる（再生も口パクもしない）。別名は MDF と同じ名前の
  FST の最初の `MODEL_ADD|<別名>|` から読む（公式のサンプル集は `gene`、無ければ `0`）。WebSocket 以外の経路では、`__AV_SETMODEL` が
  アバター制御を作り直すため、接続ごとに 1 回だけ送る。
- 頷き・まばたきの結び付け（`set_head_binding` / `set_blink_binding`）：`MODEL_BINDBONE` / `MODEL_BINDFACE` を、接続時と
  MMDAgent-EX から `MODEL_EVENT_ADD|<別名>` が届いたときに送り直す（モデルの読み込み前に送った結び付けは効かない）。
- `mark_snd_segment_ended`：`SNDBRKS` 相当の区切りの後、次のセグメント用にプリアンブルを再送できる状態にする。

バインドとセキュリティ：

- WebSocket / TCP サーバーの待受アドレスは既定で `127.0.0.1` である（`ws_host` / `tcp_server_host` を明示した場合のみ LAN に公開する）。
  **認証は無いため、LAN に公開する場合は信頼できるネットワークに限ること。**

### `MmdAudioPacer`（`audio/mmd_pacer.py`）

- Audio Output ワーカー内（`output_mode=mmd`）で TTS の PCM バイト列を受け取り、640 バイト（16 kHz・16 bit で 20 ms）の SND フレームに分割する。
- 先頭の N フレーム（`mmd_burst_snd_frames`）をバースト送出し、その後はタイムスタンプに合わせて実時間で送出する。
- 発話の終端で `SNDBRKS` を送る（WebSocket 接続ではセグメントの区切りでも送る）。
- オプションで同じ音声を MaAI 用のイベント（base64 フレームの `tts_maai_pcm_frame`）として emit する。システムの発話を
  ターン検出に混ぜないための位相・注入制御と組み合わせて使う。
- 口形状は送らない。MMDAgent-EX（Plugin_Remote）は TCP・WebSocket のどちらで受け取った SND も同じ処理に通し、音声から口パクを生成する。

### `SegmentedAudioPacer`（`audio/segmented_pacer.py`）

- Audio Output ワーカー側のペーサー。セグメントの index・世代番号・完了種別を追跡し、ストリーミング TTS の文の区切りと MMD への送出を同期させる。
- 送出完了時に `tts_mmd_send_complete`（`burst_compensation_ms` を含む）を emit する。

### 頷き（`maai/nod_motion.py`・`mmd/runtime.py`）

- 頷きモーションの入力は、対話サーバーが MaAI の連続した nod 予測をバッファに集計して決定する（単一の経路）。
- `generate_natural_nod` が自然な頷きの角度系列（ラジアン、正が上向き）を生成する。
- 頷きとまばたきは公式 MMDAgent-EX の仕組みで動かす（`mmd/avatar_keys.py`）。頭のボーン（`nod_para_bone`。設定に値が無いときは `頭`、見本の設定 `controller_config.example.json` は `首`）と
  左目・右目を KeyValue `MaAI_HeadPitch` に `MODEL_BINDBONE` で結び付け、目は頭と逆向きに `nod_para_eye_scale` 倍だけ回して
  視線を保つ。まばたきのモーフ（`mmd_blink_morph`）は KeyValue `MaAI_Blink`（0〜1）に `MODEL_BINDFACE` で結び付ける。
- 対話サーバーの `MmdRuntime` は頷きの開始時に `mmd_bind_head` を送り、その後は別スレッドで `KEYVALUE_SET|MaAI_HeadPitch|<角度>` を
  `mmd_send_text` で Audio Output ワーカーへ送る。系列（`nod_para_fps`、既定 30）を時間で線形補間し、60 Hz で送る
  （MMDAgent-EX は結び付けたボーンを受け取った値そのものにするので、30 Hz のままだと段々に見える）。途中で止めたときは
  角度 0 を送って頭を正面へ戻す。
- 結び付けたボーンはモーションの動きを上書きするため、待機モーションの頭の揺れは止まる。

---

## 設定画面（Qt・Web）

### 役割

Qt アプリの起動（`main.py`）、画面の構築（`gui/ui.py`）、UI 状態と対話サーバーとの通信の管理（`gui/controller.py` の
`MMDAgentController`）を担う。Controller はワーカーと直接は通信せず、すべての操作を対話サーバー経由で行う。
画面の構成と表記の規約は [ui-guidelines.md](./ui-guidelines.md) を参照のこと。

### `main.py`

リポジトリ直下の `main.py` は互換用の薄いラッパーであり、実体は `src/maai_sds/main.py` である。

1. `--config` の指定がなく `controller_config.json` も存在しない場合は `controller_config.example.json` から作成する。続いて `QApplication` を生成する。
2. 対話サーバー（ローカル / Docker）の選択を `dialog_server_mode` で解決したうえで、`MMDAgentController` を生成して `show()` する。
3. SIGINT のハンドラはフラグを立て、終了処理を `QTimer.singleShot(0, ...)` で予約するだけにとどめる。あわせて 200 ms 間隔の
   `QTimer` でイベントループを定期的に起こし、`close` / `quit` はメインスレッドで実行する。スレッド安全性のため、
   シグナルハンドラ内で `close` / `quit` などの Qt API を直接呼ばない。

### `gui/ui.py`

- 役割：ウィジェットの生成・レイアウト・シグナル接続のみ。処理の大半は `MMDAgentController` に委譲
- 音声対話パネル：開始・停止ボタン、履歴ビュー、割り込みの設定、沈黙によるターン取得（秒）、スケジュール終了、MMD の先頭バースト、音声の出力先など
- 接続のパターン：`controller.some_edit.editingFinished.connect(controller._save_app_config)` で設定を JSON に永続化し、
  必要に応じて Controller 側で `_push_dialog_orchestrator_settings()` を実行

### `gui/controller.py`（`MMDAgentController`）

規模の大きいクラスであり、次の処理を同一プロセス内でまとめて扱う。一部の処理は mixin（`gui/config_io.py`・
`gui/orchestrator_events.py`・`gui/obs_mixin.py` など）に分割している。

1. UI 状態（各種 `QLineEdit` / `QComboBox`）
2. 対話サーバーとの Duplex 接続の管理（ワーカー向けの唯一の IPC）
3. 表示用のイベント処理（`_on_dialog_orchestrator_event`）
4. JSON Lines の IPC を持たない外部プロセスの寿命管理（MMDAgent-EX、Qwen3-ASR のコンテナ、外部 ASR の起動スクリプト、
   Docker の対話サーバーなど）。VOICEVOX / AivisSpeech は実行ファイルのパスを対話サーバーの TTS ブリッジへ渡し、起動は対話サーバー側で行う

#### 対話サーバーのプロセス管理

- アプリの起動時に対話サーバーを常駐起動し、終了時に停止する（プロセス管理は `ipc/process_service.py` の `DialogOrchestratorProcessService`）。
- `dialog_orchestrator_event_signal` と Duplex のハンドラでイベントを受信する。
- IPC の切断時は再接続を自動で試行する。
- 対話は常に対話サーバーを経由して動作する。経路を切り替える設定（旧 `voice_dialog_orchestrator_enabled`）はない。

#### 対話サーバーへの一元化

Controller は Python のワーカー（ASR・LLM・TTS・MaAI・AudioInput・AudioOutput）と直接 IPC を行わない。ワーカーの操作は、
すべて対話サーバーへの Duplex コマンドで行う。

| コマンド | 内容 |
|---|---|
| `start_module` / `stop_module` | ワーカーの起動・停止 |
| `send_to_llm` / `send_to_tts` / `send_to_audio_output` | ワーカーへのコマンドの中継 |
| `force_finalize` | ASR ワーカーへの強制確定 |
| `inject_system_audio` | AudioInput ワーカーへのシステム音声の注入（AEC 用） |
| `audio_input_send_command` | AudioInput ワーカーへの stdin コマンド |

このほか `send_to_tts_bridge`・`send_to_llm_admin`・`asr_send_command`・`inject_user_audio` などがある（`dialog/worker.py` の `DISPATCH`）。
MMDAgent-EX への音声（SND）は、`tts_play_chunk` などを対話サーバーに送る経路に集約している。

#### 音声対話の開始と停止

`_start_voice_dialog` の処理：

1. 二重起動を防止する（`_voice_dialog_starting` / `_voice_dialog_active`）。
2. `_ensure_voice_dialog_modules_ready` で対話サーバーの起動を確認する。
3. `_build_dialog_orchestrator_settings()` の結果に `worker_configs`（`_build_worker_configs(engine)`）・`llm_connection`・
   `tts_stream_inputs` を加えた settings を、`_send_dialog_orchestrator_command` で
   `{"type": "start_voice_dialog", "settings": settings}` として送信する。
4. 対話サーバー内の Supervisor が全ワーカーを起動し、`DialogOrchestrator` の状態機械が動作を開始する
   （[音声対話の開始と停止](#音声対話の開始と停止)）。

`_stop_voice_dialog()` は `stop_voice_dialog` を Duplex で送り、対話サーバーがワーカーの停止と状態のリセットを行う。
Controller はローカルのフラグと、OBS などの周辺機能を後処理する。

#### 対話サーバーからのイベント（`_on_dialog_orchestrator_event`）

分岐の本体は `gui/orchestrator_events.py` にある。主なイベントと Controller 側の処理は次のとおりである。

| `type` | Controller 側の主な処理 |
|--------|-------------------------|
| `ready` | 対話サーバーとの接続の確立、設定の送信、Remote プラグイン設定の再送 |
| `module_started` / `module_start_failed` | モジュールの準備完了の通知（`threading.Event` による待機の解除） |
| `module_stopped` | モジュールの停止の通知 |
| `history_snapshot` | 音声対話のメッセージ一覧の更新、MaAI プロットのターン表示との連携 |
| `trigger_llm` | UI 状態の更新のみ（LLM の呼び出しは対話サーバー内部で処理） |
| `llm_chunk` / `llm_done` | 音声対話の LLM 応答の表示 |
| `asr_result` | ASR の認識結果の表示 |
| `turn` / `maai_result` / `raw_audio` / `inference_state` | 画面への反映（プロット・ログ・状態表示） |
| `request_nonstream_tts` / `request_opening_tts` / `request_scheduled_exit_tts` | 観測・ログのみ（合成・再生は `dialog/worker.py` 内で完結） |
| `latency` | UI ラベルの更新シグナル |
| `_src: "tts"` | TTS ワーカーのイベントの中継 → `_on_tts_ipc_event` |
| `_src: "audio_output"` | AudioOutput ワーカーのイベントの中継 → `_on_audio_output_ipc_event` |
| `_src: "llm_admin"` | LLM 管理用 HTTP のイベントの中継 → `_on_llm_admin_ipc_event` |

LLM テストのイベント（`llm_test_chat_chunk`・`llm_test_tts_segment`・`llm_test_finished` など）は、`_src` による分岐の後で `type` ごとに処理する。
`tts_play_chunk` / `tts_play_break` は Controller が受け取るイベントではなく、対話サーバーが AudioOutput ワーカーへ中継するコマンドである。
MMD 出力時のペーシング（`MmdAudioPacer`）は AudioOutput ワーカー内で行う。

#### テスト画面のワーカー操作

各テスト画面（ASR・LLM・TTS・MaAI）は、いずれも対話サーバー経由でワーカーを管理する。

- `_start_module_via_orchestrator`：`start_module` コマンドの送信と `module_started` イベントの待機
- `_stop_module_via_orchestrator`：`stop_module` コマンドの送信
- `_send_to_worker_via_orchestrator`：`send_to_llm` などのコマンドの中継

#### 設定の永続化（`_save_app_config`）

- `_collect_app_config_payload()`（`gui/config_io.py`）がウィジェットの値を `payload` dict にまとめ、`config_store.save` で保存
- 保存後に `_push_dialog_orchestrator_settings()` を実行し、実行中の対話サーバーへ `update_settings` を送信
- 平坦なフィールド（176 キー）は `gui/app_config_spec.py` の宣言テーブルを唯一の定義元とし、キー名と既定値の二重管理を回避
  （[`gui/app_config_spec.py`](#guiapp_config_specpy平坦なフィールドの宣言テーブル)）。エンジン切り替え時のマイグレーションや
  `tts_template` の JSON など特殊なものは、明示的なコードとして記述

#### ワーカー設定の組み立て（`dialog/settings.py` への一本化）

`_build_asr_config_dict`・`_build_maai_worker_config`・`_build_tts_inputs` などの 10 個のビルダーは、`dialog/settings.py` の
`build_*(cfg)`（Controller 内では `vds` として import）へ委譲する。GUI と Web UI で同じ変換を共有するためである。
GUI 側は `_widget_config_snapshot()` でウィジェットの現在値を cfg dict にしてから渡す。

GUI 固有の事情は、引数や前処理として明示的に渡す。

| 事情 | 渡し方 |
|---|---|
| MMDAgent-EX が起動中か（`mmd` から `local` へのフォールバック） | `build_audio_output_worker_config_dict(..., assume_mmd_running=...)` |
| `.mdf` に未保存の、編集中の値を使う | スナップショットに `Plugin_Remote_*` のウィジェット値を重ね、`mdf_path` を外す |
| エンジン別の声・ポートの補正（ウィジェットを書き換える副作用を伴う） | `_build_tts_inputs` の前処理として残す |
| テンプレート JSON が不正なときにエラーを出す（`strict=False` では `{}`） | `parse_tts_template_text(strict=True)` |

### Web UI（`web/`）

- `web/server.py`：FastAPI によるブラウザと対話サーバーの間の中継。設定は `ConfigStore` で読み書きし、`start_voice_dialog` の
  ペイロードは Qt アプリと同じ `build_start_voice_dialog_payload`（`dialog/settings.py`）で組み立てる。
- `web/mmdagent.py`：`dialog/settings.py` の変換処理を利用する。
- `web/obs_link.py`：OBS 録画の参照カウント（Qt アプリの `gui/obs_mixin.py` に相当。[補助サービス](#補助サービス)）。
- ブラウザのマイク音声は、音声入力ワーカーの `browser` モード（16 kHz の PCM）で受け取る。

---

## 設定とユーティリティ

### 役割

アプリ設定の読み書き、設定からワーカー起動用の dict への変換、ワーカー間で共有する小規模なヘルパーをまとめる。

- アプリ全体の設定：`controller_config.json`（`config_store.py`）
- 永続化するキーの定義：平坦なフィールドは `gui/app_config_spec.py` の宣言テーブル、それ以外は `gui/config_io.py` の
  `_collect_app_config_payload`（`llm_model_lmstudio`・`llm_model_openai` など）。廃止したキーは読み込み対象外

### `config_store.py`

- `ConfigStore(path: Path)`
  - `load()`：JSON ファイルを `dict` として読み込み。ファイルが存在しない・破損している場合は `{}`
  - `save(data)`：`indent=2` の UTF-8 JSON で上書き。同じディレクトリの一時ファイルに書いてから `os.replace` で置き換えるアトミックな書き込み
    （置き換えに失敗した場合は直接上書き）。書き込みの失敗は無視
  - `save_raising(data)`：`save` と同じ書き込みで、失敗時に例外を送出
- 利用元：`gui/controller.py`（ウィジェットと `controller_config.json` の間の永続化）、`dialog/worker.py`（起動時の先行読み込み）、`web/server.py`

### `dialog/settings.py`

- 役割：`controller_config.json` の dict から `start_voice_dialog` のペイロードを組み立てる、GUI に依存しない純関数群
- 主な公開関数
  - `build_start_voice_dialog_payload(cfg, *, tts_engine=None) -> dict`：エントリーポイント。`worker_configs`・`llm_connection`・
    `tts_stream_inputs` などまで揃えた settings dict を返す
  - `build_dialog_orchestrator_settings(cfg)`：`Orchestrator.start(settings)` に渡す主要パラメーター（VAD・MaAI・nod_para・scheduled_exit・opening など）
  - `build_worker_configs(cfg, tts_engine)`：`asr`・`tts`・`llm`・`maai`・`audio_input`・`audio_output` の dict 一式
  - `build_audio_input_worker_config(cfg)`・`build_audio_output_worker_config_dict(cfg)`・`build_maai_worker_config(cfg)`・
    `build_asr_config_dict(cfg)`・`build_tts_worker_config(cfg, engine)`・`build_tts_inputs(...)` など
  - `active_llm_connection_settings(cfg)`・`resolve_llm_split_settings(cfg)`・`cfg_str/int/float/bool(...)` など
- 利用元：`web/server.py`（`build_start_voice_dialog_payload`）、`gui/controller.py`（共通の変換処理の委譲先）、`web/mmdagent.py`、
  `headless_main.py` など。Qt にも IPC クライアントにも非依存

### `gui/app_config_spec.py`（平坦なフィールドの宣言テーブル）

`controller_config.json` の平坦なフィールド 176 キーについて、
`(キー, ウィジェット属性, 種別, hasattr ガード, 保存時の代替値, 読み込み時の既定値, setter, 空文字を既定値に戻すか)`
を 1 か所で定義する。`gui/config_io.py` の `_read_app_config_field` / `_apply_app_config_field` がこれを参照し、cfg とウィジェットの間で値を変換する。

- テーブルは既存のコードから AST で機械的に抽出して生成したものであり、既定値の転記誤りが生じない。
- インデックス 4（5 番目）の要素「保存時の代替値」は、ウィジェットが存在しない場合にのみ使う値である。空欄用の値ではない
  （元のコードの `... if hasattr(self, "X") else "既定"` に相当する）。
- エンジン切り替え時のマイグレーション・`tts_template`・`CONFLICT_CHECKBOX_PAIRS`・`MAAI_PLOT_VISIBILITY_SPECS` は宣言化せず、明示的なコードとして残している。

### `tts/playback_service.py`

- 役割：TTS の合成・再生に関する小規模なヘルパー群。Qt にも IPC クライアントにも非依存
- 主な定義・関数
  - `WORKER_BACKED_ENGINES`・`BRIDGE_BACKED_ENGINES`・`BINARY_STREAM_ENGINES`：エンジンの分類の定義元（`tts/worker.py` も `BINARY_STREAM_ENGINES` をここから import する）
  - `is_worker_engine(engine)`・`is_bridge_engine(engine)`・`is_binary_stream_engine(engine)`：エンジンの分類（ワーカー経由・ブリッジ経由・逐次ストリーム）
  - `default_synth_timeout_sec(engine)`：合成タイムアウトの既定値
  - `gap_silence_pcm_l16le(gap_sec)`：セグメント間の無音 PCM
  - `split_pcm_into_frames(pcm)`：640 バイト単位のフレーム分割
  - `to_l16le(pcm, source_endian)`：ビッグエンディアンからリトルエンディアンへの変換
  - `SynthRequest`・`build_voicevox_bridge_command(...)`・`build_template_bridge_command(...)`：合成リクエストの dict の組み立て
- 利用元：`dialog/runtime.py`（`segment_gap_silence_sec` に基づく無音の挿入など）、`tts/routing.py`、`gui/controller.py`

### `llm/common.py`

`normalize_lmstudio_base_url`・`resolve_openai_api_key` の共通実装であり、`llm/admin_http.py`・`llm/service.py` などから参照する。
同じ処理を各モジュールに複製すると修正時に食い違いが生じるため、実装はこのモジュールに一本化している。
Docker 内で `127.0.0.1` / `localhost` を `host.docker.internal` に付け替える `normalize_docker_loopback_host` も提供する。

### `ipc/worker_runtime.py`

- `parse_json_object`：ワーカーが受け取る `--config-json` 文字列の解析。不正な場合は `ValueError` を送出し、呼び出し側のワーカーが
  stderr にメッセージを出して終了コード 2 で終了
- `safe_emit` / `to_jsonable`：NumPy 配列やネストした構造を JSON 直列化可能な形に変換。`client.send` の失敗を握りつぶし、
  ワーカー全体の停止を防止
- `connect_worker_client` / `finish_worker` / `start_parent_death_watchdog`：IPC 接続のリトライ、終了処理、親プロセスの終了検知

### `ipc/queue_policy.py`

- ワーカー内キューへの `put` の方針（`drop_oldest`・`drop_newest`・`block`）
- 音声入力の高頻度イベントによるキューの詰まりの回避（`asr/worker.py`・`asr/compare_worker.py` で使用）

### `utils.py`

- プロジェクト横断のヘルパー。型変換（`to_bool`・`json_config_bool`・`to_float_or_none`）と、MaAI の頷き（nod_para）の閾値まわりの補助関数
- 処理の流れの中心ではないが、Controller・サービス層の各所から個別に import

---

## 補助サービス

音声対話の主経路には含まれないが、アプリ全体で使用する補助的な処理である。

### `mmd/mdf_service.py`（`MdfService`）

- `.mdf` テキストのキー＝値の行（`Plugin_*` など）の解析と書き換え
- Controller が MMDAgent-EX の実行引数・プラグイン設定の読み込みと保存に使用

### `obs/remote.py`・`gui/obs_mixin.py`

- OBS WebSocket（v5）経由の録画の開始・停止（`obs/remote.py`）
- 録画の参照カウント（`gui/obs_mixin.py`。Web UI では `web/obs_link.py`）。録画を取得する理由は「手動」「音声対話」「MaAI テスト」の 3 つで、
  いずれかが残っている間は録画を継続し、最後の 1 つが解放されたときに停止。音声対話・MaAI テストとの連動は
  `obs_remote_link_voice`・`obs_remote_link_maai` で有効化

### `maai/plot_window.py`

- MaAI の時系列（`p_now`・VAD・相槌など）を別ウィンドウにプロット。表示するグラフが 9 個までは 1 列で縦長の窓、10 個以上は 2 列で横長の窓にする（列の数が変わったときだけ窓の形を合わせ、同じ列の数の間は手で変えた大きさを保つ。Web は引き出しの幅を変える）
- Controller が `history_snapshot` の `turn_blocked_until_user_speech` を受け取り、`update_turn_lock_state` でターンロックの表示を
  更新するなど、対話サーバーの状態と連動

### `maai/nod_motion_chart_widget.py`

- 頷き波形を表示するチャートのコンポーネント
