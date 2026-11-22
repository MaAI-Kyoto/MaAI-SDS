# 実行ガイド

MaAI-SDS を 4 通りの実行構成で起動し、MMDAgent-EX のアバターと対話できることを確かめるまでの手順です。
環境構築・API キー・外部アプリケーションの準備は [README](../README.md) を参照してください。

## 概要

対話サーバーと設定画面は、次の 4 通りの組み合わせで動作します。音声入出力とアバターは、いずれの構成でもホストで扱います。

| 対話サーバー | 設定画面 | 用途 | 手順 |
|---|---|---|---|
| ホスト（Python） | Qt アプリ | 標準の構成 | [ホスト × Qt アプリ](#ホスト--qt-アプリ) |
| ホスト（Python） | Web UI | ブラウザからの操作 | [ホスト × Web UI](#ホスト--web-ui) |
| Docker | Qt アプリ | エンジンの依存関係をコンテナに分離 | [Docker × Qt アプリ](#docker--qt-アプリ) |
| Docker | Web UI | 設定画面を含めてコンテナで実行 | [Docker × Web UI](#docker--web-ui) |

- ブラウザ内の MMD ビューアーは使用しません。アバターの表示・音声・口パク・頷きは、Windows ホストで別ウィンドウとして
  起動するネイティブの MMDAgent-EX で確認します。
- ホストと Docker は、設定画面のヘッダーにある「対話サーバー」から切り替え・起動・停止できます
  （[ホストと Docker の切り替え](#ホストと-docker-の切り替え)）。
- Docker の対話サーバーに同時に接続できる設定画面は 1 つです。

---

## 共通の準備

### MMDAgent-EX と MDF

MMDAgent-EX の本体とアバターの取得は [README](../README.md#3-mmdagent-ex-とアバター) を参照してください。

1. 「MMDAgent-EX起動設定」の「ファイル設定」で次を指定します（README の配置なら自動で入ります）。Docker の構成では
   `controller_config.docker.json` の `mdf_path` も確認します（[Docker の準備](#docker-の準備)）。
   - 実行ファイル：`C:\MMDAgent-EX\MMDAgent-EX.exe`（ソースからビルドした場合は `...\Release\MMDAgent-EX.exe`）
   - MDF：使用する `.mdf`（例：サンプル集に入れた MaAI-SDS 用の `C:\MMDAgent-EX\example\maai_sds\maai_sds.mdf`。
     TCP サーバー方式・ポート 39001）
2. MDF の `Plugin_Remote_*` で Remote の通信方式を 1 つだけ選び、保存します。`EnableServer` と `EnableClient` を
   同時に `true` にしないでください。通信方式は MDF の設定から自動で選択されます。

| 方式 | MDF の設定 | Python 側の動作 | 待ち受け側 | Docker でのポートの公開 |
|---|---|---|---|---|
| WebSocket | `EnableServer`・`EnableClient` とも `false` | WebSocket サーバー | 対話サーバー（コンテナ） | 必要（`MMDAGENT_REMOTE_WS_PORT`） |
| TCP サーバー | `EnableClient=true`、`EnableServer=false` | TCP サーバー | 対話サーバー（コンテナ） | 必要（`MMDAGENT_REMOTE_TCP_PORT`） |
| TCP クライアント | `EnableServer=true`、`EnableClient=false` | TCP クライアント | MMDAgent-EX | 不要 |

Docker の構成では、WebSocket 方式または TCP サーバー方式のポートを `.env` の値と一致させます。

#### WebSocket（対話サーバーがサーバー）

```text
Plugin_Remote_EnableServer=false
Plugin_Remote_EnableClient=false
Plugin_Remote_Websocket_Host=localhost
Plugin_Remote_Websocket_Port=8080
Plugin_Remote_Websocket_Directory=/chat
```

MMDAgent-EX は `ws://localhost:8080/chat` に接続します。Docker では `.env` の `MMDAGENT_REMOTE_WS_PORT=8080`（既定 8080）と
一致させます。ディレクトリは設定した値（`/chat` など）を指定します。`maai_sds.mdf` を WebSocket にする場合は、
`Plugin_Remote_EnableClient=false` にして上の `Plugin_Remote_Websocket_*` の 3 行を足します（`.env.example` の
`MMDAGENT_REMOTE_WS_PORT` は 9001 にしてあるので、ポートは 9001 にそろえます）。

#### TCP サーバー（対話サーバーがサーバー、MMDAgent-EX がクライアント）

MaAI-SDS 用の `maai_sds.mdf` の既定です。

```text
Plugin_Remote_EnableClient=true
Plugin_Remote_Hostname=localhost
Plugin_Remote_Port=39001
Plugin_Remote_RetryCount=1000
```

MMDAgent-EX は `localhost:39001` に接続します（対話サーバーより先に起動しても、つながるまで再試行します）。
Docker では `.env` の `MMDAGENT_REMOTE_TCP_PORT=39001` と一致させます。MMDAgent-EX はループバックに公開された
コンテナのポートへ接続します。ホストの対話サーバーは `127.0.0.1` で待ち受けます（LAN に公開する場合だけ MDF に
`Plugin_Remote_Bind_Host=0.0.0.0` を書きます。認証は無いので、信頼できるネットワークに限ってください）。

ポートは Windows の動的ポート（49152〜65535）の外にします。Hyper-V・WSL・Docker Desktop が動的ポートの一部を予約する
ことがあり、予約された範囲のポートは待ち受けられません（`WinError 10013`）。予約された範囲は
`netsh interface ipv4 show excludedportrange protocol=tcp` で確認できます。

#### TCP クライアント（対話サーバーがクライアント、MMDAgent-EX がサーバー）

```text
Plugin_Remote_EnableServer=true
Plugin_Remote_EnableClient=false
Plugin_Remote_Hostname=localhost
Plugin_Remote_ListenPort=39001
```

Docker では、この方式に compose でのポートの公開は不要です。コンテナは `localhost` を `host.docker.internal` に読み替え、
`host.docker.internal:39001`（`host.docker.internal:Plugin_Remote_ListenPort`）に接続します。選択したポートでの待ち受けが、
プライベートネットワークに対して Windows ファイアウォールで許可されていることを確認してください。

### 音声の設定

1. 「音声出力の設定」で「音声出力先」を「MMD（Remote）」にします。
2. 「音声入力の設定」で入力元を選びます。
   - Qt アプリ：入力モード `mic` と、使用する Windows の入力デバイスを選びます。
   - Web UI：「Web UIマイク入力元」で、mic モードの入力元を選びます。
     - 「ブラウザのマイク」：ブラウザの許可ダイアログでマイクの使用を許可します。
     - 「WebサーバーPCのローカルマイク」：「マイクデバイス」で Windows の入力デバイスを選びます。

### Docker の準備

Docker × Web UI と Docker × Qt アプリに共通の手順です。

1. Docker Desktop を起動し、Linux コンテナが利用可能になるまで待ちます。
2. 設定ファイルと `.env` を用意します（初回のみ）。

   ```powershell
   copy controller_config.docker.example.json controller_config.docker.json
   copy .env.example .env
   if (-not (Test-Path enabled_engines.json)) { Set-Content enabled_engines.json "[]" }
   ```

3. `.env` の `MMDAGENT_ASSETS_HOST` に、MDF と関連する MMDAgent-EX の素材を含む Windows のフォルダーを指定します
   （公式のサンプル集なら `C:/MMDAgent-EX/example`）。入力用の WAV がリポジトリの外にある場合は、`INPUT_WAV_HOST` も指定します。
4. WebSocket 方式または TCP サーバー方式を使う場合は、`.env` のポートを MDF の値と一致させます。
5. `controller_config.docker.json` を確認します。ファイルのパスはコンテナ内の形式のまま使います。
   - `mdf_path`：`/app/mmdagent_assets/<MMDAGENT_ASSETS_HOST からの相対パス>`（サンプル集なら
     `/app/mmdagent_assets/maai_sds/maai_sds.mdf`）
   - `audio_output_mode`：`mmd`
   - コンテナが読むファイルのパス：`/app/mmdagent_assets/...`・`/app/input_wav/...`・`/host_repo/...` のいずれか
   - 例外は `exec_path`：Qt アプリはホストで MMDAgent-EX を起動するため、`MMDAgent-EX.exe` の Windows のパスのまま

> [!NOTE]
> MDF は `/app/mmdagent_assets` に読み取り専用でマウントされます。Web UI では参照と再読み込みだけを行い、
> 編集は Windows 側のファイルに対して行います。

> [!WARNING]
> compose の `web` サービスと `orchestrator` サービスは同時に起動しないでください。どちらも MMDAgent-EX の
> Remote ブリッジ用に同じポートを公開します。

---

## ホスト × Qt アプリ

1. Qt アプリを起動します。

   ```powershell
   .venv\Scripts\python main.py
   ```

   `uv run python main.py` は使わないでください（理由は [README](../README.md)）。uv 経由で起動する場合は
   `uv run --no-sync python main.py` を使います。
2. [共通の準備](#共通の準備)に従って MMDAgent-EX・MDF・音声を設定します。
3. メイン画面の「MMDAgent-EX」欄で「MMDAgent-EXを起動」を押します。
4. 「音声対話開始」を押し、1 往復以上対話します。

**期待される結果**

- ヘッダーの表示が「対話サーバー：接続済み（ローカル）」になる
- [動作確認](#動作確認)の項目を満たす

停止は「音声対話終了」で音声対話を止め、「MMDAgent-EX」欄の「停止」で MMDAgent-EX を終了してから、Qt を閉じます。

## ホスト × Web UI

### 1. Web UI の起動

PowerShell で MaAI-SDS のリポジトリのルートに移動し、プロジェクトの Python を直接実行します。

```powershell
.venv\Scripts\python -m maai_sds.web.server
```

<http://127.0.0.1:8000/> を開きます。ホスト・ポートは `MAAI_SDS_WEB_HOST`・`MAAI_SDS_WEB_PORT` で変更できます。

```powershell
Set-Location C:\path\to\MaAI-SDS
$env:MAAI_SDS_WEB_HOST = "127.0.0.1"
$env:MAAI_SDS_WEB_PORT = "18080"
.\.venv\Scripts\python.exe -m maai_sds.web.server
```

この例ではブラウザで <http://127.0.0.1:18080/> を開きます。ヘッダーの表示が「対話サーバー：接続済み」になることを確認します。

### 2. MMDAgent-EX と音声の設定

[共通の準備](#共通の準備)に従って、「MMDAgent-EX起動設定」・「音声出力の設定」・「音声入力の設定」を行います。

### 3. MMDAgent-EX の起動

1. メイン画面の「MMDAgent-EX」欄で「起動」を押します。
2. ネイティブの MMDAgent-EX のウィンドウが開き、ボタンが「起動中」、表示が「プロセス：起動中」になることを確認します
   （PID は「詳細」に表示されます）。
3. 「プロセスコンソール」で、使用する方式に応じた次のメッセージを確認します。

**成功の目印**

| 対象 | メッセージ |
|---|---|
| 起動 | `[MMDAgent] 起動しました (PID=...)` |
| WebSocket | `Remote bridge: WebSocket server 開始 ...` → `WebSocket クライアント接続` |
| TCP クライアント | `Remote bridge: TCP client 開始 ...` → `TCP client 接続済み` |
| TCP サーバー | `Remote bridge: TCP server 待受開始 ...` → `TCP server クライアント接続` |
| 接続 | `[MMDAgent Remote] 接続 mode=websocket\|tcp_client\|tcp_server` |

**失敗の目印**

| 表示先 | メッセージ |
|---|---|
| 画面のエラー表示 | `MMDAgent-EX実行ファイルが見つかりません` / `MDFファイルが見つかりません` |
| 画面のエラー表示 | `Plugin_RemoteのWebSocketポートが不正です。` |
| コンソール | `Remote bridge: ... 待受失敗` / `WS server エラー` |
| コンソール | `[MMDAgent][WARN] Dialog worker への設定同期に失敗` |
| コンソール | `[NodMotion] bridge not connected` |

### 4. 音声対話

1. 「音声対話開始」を押します。
2. 入力元に応じて次を確認します。
   - ブラウザのマイク：`[マイク] ブラウザ取り込み開始` と `browser mic: 最初の音声フレームを受信` が出力される
   - ローカルマイク：ブラウザがマイクの使用許可を求めず、選択した Windows のデバイスへの発話が ASR に入力される
3. 1 往復以上対話し、[動作確認](#動作確認)の項目を確認します。

### 5. 停止

1. 「音声対話終了」で音声対話を止めます。
2. 「MMDAgent-EX」欄の「停止」で、この Web サーバーが起動した MMDAgent-EX が終了することを確認します。
3. コンソールに `[MMDAgent] MMDAgent-EXを停止しました。` が出力され、ボタンが「起動」に戻ることを確認します。
4. PowerShell で `Ctrl+C` を押し、Web サーバーを終了します。

> [!NOTE]
> Docker で動かす Web UI には起動ボタンを表示せず、代わりにホストで実行するコマンドを表示します
> （[Docker × Web UI](#docker--web-ui)）。Windows 以外で動かすネイティブの Web UI も、起動できない理由と、
> Windows ホスト側で起動する旨を表示します。コンテナなどから Windows の実行ファイルを起動する機能は対象外です。

## Docker × Web UI

対話サーバーと Web UI は Docker で動作し、MMDAgent-EX は Windows ホストで起動します。

### 1. 起動

[Docker の準備](#docker-の準備)を済ませてから、compose を起動します。

```powershell
# GPU 構成
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
# CPU 構成
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up
```

バックグラウンドで起動する場合は `up -d` とします。`--profile qwen3`・`--profile nemotron`・`--profile audiocpp` などの
追加サービスは [README](../README.md) を参照してください。<http://127.0.0.1:8000/> を開きます。

**期待される結果**

- メイン画面の「MMDAgent-EX」欄に、接続先と、ホストで実行するコマンド（「コピー」ボタン付き）が表示される
- 「詳細」を開くと、MDF 名・通信方式・MDF の設定が表示される
- 接続状態が「未接続」、ヘッダーが「MMDAgent-EX：未接続」と表示される

### 2. ホストの MMDAgent-EX と音声対話

1. Windows PowerShell で MMDAgent-EX のフォルダーに移動し、Web UI に表示されたコマンドを実行します。

   ```powershell
   Set-Location C:\MMDAgent-EX
   .\MMDAgent-EX.exe ".\example\maai_sds\maai_sds.mdf"
   ```

2. Web UI の接続状態が「未接続」から「接続済み（websocket|tcp_server|tcp_client）」に変わることを確認します。
3. 「音声対話開始」を押し、1 往復以上対話します。

**期待される結果**

- [動作確認](#動作確認)の項目を満たす
- 切断・再接続に Web UI の接続状態が追従する

### 3. 停止

1. Web UI の「音声対話終了」で音声対話を止め、MMDAgent-EX を終了します。
2. compose を実行している画面で `Ctrl+C` を押した後、起動時と同じ構成を指定して終了します。

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
# CPU 構成は docker-compose.gpu.yml を docker-compose.cpu.yml に置き換える
```

## Docker × Qt アプリ

Windows ホストで動かすのは Qt アプリ（PySide6）と MMDAgent-EX だけです。対話サーバー（orchestrator）と、
ASR・LLM・TTS・MaAI・音声入出力の各ワーカーは Docker で動作します。

### 1. 準備

1. [Docker の準備](#docker-の準備)を済ませます。
2. compose の `web` サービスが動いている場合は停止します。プロファイル付きの他のサービスを動かしたままにする
   必要がある場合は `compose down` を使わず、このサービスだけを停止します。

   ```powershell
   docker compose stop web
   ```

### 2. 起動

Windows PowerShell 5.1 から起動スクリプトを実行します。Python はリポジトリの `.venv` が自動で選ばれます。
PySide6 を導入した別の Python を使う場合は、明示的に指定します。

```powershell
$env:MAAI_SDS_QT_PYTHON = "C:\path\to\python.exe"
.\scripts\start_qt_docker.ps1
```

スクリプトは次の処理を行います。

| 処理 | 内容 |
|---|---|
| 構成の選択 | `.device_extra` を参照。`cpu` なら CPU 用、`cu128` または記録なしなら GPU 用の overlay |
| 対話サーバーの起動 | `orchestrator` サービスを起動し、ポート（既定 8765）が開くまで待機 |
| 環境変数 | `PYQTGRAPH_QT_LIB=PySide6`、`PYTHONPATH` への `src` の追加、Python 出力の UTF-8 化 |
| Qt の起動 | `controller_config.docker.json` を指定して起動 |

手動で実行する場合のコマンドは次のとおりです。

```powershell
# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile qt up -d orchestrator
# CPU
docker compose -f docker-compose.yml -f docker-compose.cpu.yml --profile qt up -d orchestrator
python main.py --remote-orchestrator 127.0.0.1:8765 --config controller_config.docker.json
```

ホスト側のポートを変える場合：

```powershell
.\scripts\start_qt_docker.ps1 -Port 18765 -Python "C:\path\to\python.exe"
```

構成の自動選択は `-Device gpu` / `-Device cpu` で上書きできます。

**期待される結果**

- Qt のコンソールに `[DockerBackend] remote orchestrator 接続完了: <ホスト>:<ポート>` と `[DialogOrchestrator] ready` が出力される
- コンテナが起動途中の場合は、接続できない旨の警告（`[DockerBackend][WARN] ...`）を表示し、1.5 秒ごとに再試行する

### 3. MMDAgent-EX の Remote ブリッジ

この構成では、TCP サーバー方式（コンテナが TCP サーバー）を推奨します。方式ごとの MDF の設定は
[MMDAgent-EX と MDF](#mmdagent-ex-と-mdf) を参照してください。

1. MDF に TCP サーバー方式の値（`Plugin_Remote_Port` は `39001` または `MMDAGENT_REMOTE_TCP_PORT` の値）を設定し、
   Qt で MDF を保存します。
2. 「MMDAgent-EXを起動」を押します。MMDAgent-EX が TCP クライアントとなり、ループバックに公開された
   コンテナのポートへ接続します。
3. Qt のコンソールで音声出力の Remote ブリッジが接続済みになったこと
   （`[AudioOutput] MMD remote bridge connected (mode=...)`）を確認します。
4. TTS のテストを実行し、音声とリップシンクを確認します。

### 4. 機能の確認

| # | 操作 | 期待される結果 |
|---|---|---|
| 1 | 「音声入力の設定」で入力モード `mic` と使用する Windows の入力デバイスを選び、ASR・MaAI・音声対話のいずれかを開始 | `[DockerMic]` がホストでの取り込み開始を報告（`host microphone started: device=..., rate=...Hz`。通常は 16 kHz、受け付けないデバイスのみデバイスの既定値）。コンテナが `browser mic: 最初の音声フレームを受信` を出力 |
| 2 | 発話する | ASR の認識結果、ユーザー／システムのプロット、VAD・ターンのイベント、MaAI の結果が Qt で更新される |
| 3 | AEC を有効にして TTS を再生 | TTS の再生音でユーザーのターンが誤検出されない。AEC はコンテナ内で動作し、TTS の音声をシステム側の参照信号として受け取る |
| 4 | 音声対話を開始 | LLM・TTS のストリーミング出力と MMDAgent-EX の動作・音声。割り込みと停止も動作する |
| 5 | モジュール状態の操作、「プロセスコンソール」、「MaAIプロット」、LLM の管理操作、OBS の操作を使う | ローカル構成の Qt と同じ動作（同じ対話サーバーのイベント・コマンド接続を使うため） |
| 6 | 入力モード `wav` で、`INPUT_WAV_HOST` 以下またはリポジトリ内のファイルを選ぶ | 再生・一時停止・終了、および範囲の末尾での音声対話の自動停止。compose のどのマウントにも含まれないファイルをファイル選択で選ぶと、パスの変換に関する警告を表示 |
| 7 | 音声出力先に「ローカルスピーカー」または「TCP送出」を選び、TTS・音声対話の操作を行う | Docker バックエンドでは「MMD（Remote）」のみ利用できる旨を表示 |
| 8 | TTS に `VOICEVOX` または `AivisSpeech` を選び、起動・テストを行う | コンテナから Windows のアプリケーションは起動できない旨を表示 |
| 9 | `VOICEVOX-Core`（またはコンテナで動作する他のエンジン）に切り替える | 正常に起動する |

### 5. 再接続の確認

1. 対話サーバーを動かしたまま Qt を閉じ、compose のサービスが動作し続けていることを確認します。

   ```powershell
   docker compose ps orchestrator
   ```

2. Qt だけを再度起動します。

   ```powershell
   python main.py --remote-orchestrator 127.0.0.1:8765 --config controller_config.docker.json
   ```

3. 次を確認します。

   - ready、音声対話の状態と履歴、モジュールの状態を受信する
   - 入力モード `mic` の音声入力ワーカーが動作中であれば、ホストでの取り込みが再開する

> [!NOTE]
> Qt を終了すると対話サーバーとの接続は切れますが、コンテナのプロセスとワーカーは終了しません。一方、Qt が
> 起動したホストの MMDAgent-EX は Qt の終了時に停止します。Qt を開き直したら「MMDAgent-EXを起動」を再度押してください。

### 6. Qt からの起動と停止

`scripts/start_qt_docker.ps1` から起動した Qt には、スクリプトが compose のファイルを渡します
（`MAAI_SDS_QT_COMPOSE_FILES`・`MAAI_SDS_QT_COMPOSE_DIR`）。この場合、モジュール状態の先頭行（対話サーバー）の
■ で `docker compose ... stop orchestrator`、▶ で `docker compose ... up -d orchestrator` を実行し、再接続します。

1. ■ を押します。音声対話とモジュールが先に停止し、続いてコンテナが停止すること
   （`docker compose ps orchestrator` で exited）を確認します。状態表示は「対話サーバー：停止（Docker）」になり、Qt は自動では再接続しません。
2. ▶ を押します。コンテナが再び起動し、行の表示が「稼働」に戻ることを確認します。
3. 対話サーバーが停止した状態で、モジュールの ▶ または「音声対話開始」を押します。先に対話サーバーが起動することを確認します。

これらの環境変数なしで Qt を手動で起動した場合、▶ / ■ は接続と切断だけを行い、コンテナは動作し続けます。

### 7. 停止

Qt を閉じ、Qt が起動していないホストの MMDAgent-EX があれば終了してから、Qt 用のバックエンドサービスだけを停止します。

```powershell
docker compose stop orchestrator
```

他のサービスに影響を与えずに、停止したサービスのコンテナを削除する場合：

```powershell
docker compose rm -f orchestrator
```

---

## ホストと Docker の切り替え

Qt アプリでは、対話サーバーの行の右端で「ローカル」/「Docker」を選択できます。もう一方を選ぶと確認の後、Qt がそのモードで
開き直します（元のウィンドウは先に閉じます）。

| 項目 | 内容 |
|---|---|
| 選択の保存先 | ユーザーの設定フォルダーの `launch.json`（`%APPDATA%\MaAI-SDS`。ネイティブの Web UI と共通。`MAAI_SDS_LAUNCH_PREFS` で変更可） |
| 選択が使われる場合 | 引数なしの `python main.py` |
| 優先されるもの | `--remote-orchestrator`、`--dialog-server local\|docker`、`scripts/start_qt_docker.ps1` |
| Docker モードの起動 | 接続の前に Qt 自身が `docker compose up -d orchestrator` を実行。compose のファイル・ポート（`MAAI_SDS_QT_ORCHESTRATOR_PORT` / `.env`）・`controller_config.docker.json` は起動スクリプトと同じ |
| Docker を選べない条件 | `docker` コマンド、`docker-compose.yml`、`controller_config.docker.json` のいずれかがない（グレー表示） |
| モードごとの設定ファイル | `MAAI_SDS_QT_LOCAL_CONFIG` / `MAAI_SDS_QT_DOCKER_CONFIG` で上書き |

確認手順：

1. Docker → ローカル：Qt が開き直し、状態表示が「対話サーバー：接続済み（ローカル）」になる
2. コンテナが停止した状態でローカル → Docker：Qt が開き直し、コンテナを起動して接続する
3. Qt を閉じて引数なしで `python main.py` を実行：最後に選んだモードで開く

ネイティブの Web UI（`python -m maai_sds.web.server`）にも同じ選択があります。Docker を選ぶと、Web サーバーが
同じポートで compose の orchestrator に接続する形で再起動し、ページは自動で再読み込みされます。このモードでは
ブラウザのマイクを使い、エンジン管理は無効になります。ファイル選択ダイアログには、バインドマウントされたフォルダーが
コンテナ内のパスで表示されます（MMDAgent-EX の実行ファイルはホストのパスで選びます）。

> [!NOTE]
> Docker の対話サーバーに同時に接続できる設定画面は 1 つです。Web UI を Docker に切り替える前に Qt を閉じてください。

---

## 動作確認

どの構成でも、音声対話を 1 往復以上行い、映像と音声で次を確認します。

- [ ] TTS の音声が PC の通常の再生経路ではなく MMDAgent-EX から再生される（SND）
- [ ] 発話中に口パクする（SND とリップシンク）
- [ ] 頷きの条件が成立したときにアバターが頷く（コンソールに `[NodMotion] start ...` が出力される）
- [ ] Remote の切断後に MMDAgent-EX を再接続すると、再び接続され（Web UI のコンソールでは `[MMDAgent Remote] 接続 ...`）、
      次の発話が再生される

構成ごとの確認項目は次のとおりです。

| 構成 | 確認項目 |
|---|---|
| ホスト × Web UI | [成功・失敗の目印](#3-mmdagent-ex-の起動)、[入力元ごとの確認](#4-音声対話) |
| Docker × Web UI | [起動直後の表示](#1-起動)、接続状態の追従 |
| Docker × Qt アプリ | [機能の確認](#4-機能の確認)、[再接続](#5-再接続の確認)、[Qt からの起動と停止](#6-qt-からの起動と停止) |
| 共通 | [ホストと Docker の切り替え](#ホストと-docker-の切り替え) |

---

## トラブルシューティング

| 症状 | 確認・対処 |
|---|---|
| 「未接続」のまま変わらない | 画面に表示された MDF 名・方式・接続先と、実際に起動した MDF を照合する。特に WebSocket の `Directory` と各 `Port` |
| MDF が見つからない（Docker） | `.env` の `MMDAGENT_ASSETS_HOST` が MDF を含むフォルダーか、`mdf_path` が `/app/mmdagent_assets/<そのフォルダーからの相対パス>` か |
| `port is already allocated` / `address already in use` | 使用中の 8080 / 39001 を解放するか、`.env` と MDF の両方を同じ空きポートに変更。compose の `web` と `orchestrator` を同時に起動していないかも確認 |
| TCP クライアント方式だけ接続できない | Windows Defender ファイアウォールで MMDAgent-EX の待ち受けが遮断されていないか。`host.docker.internal` はコンテナから見た Windows ホスト |
| MDF の保存が 403 になる（Docker の Web UI） | 仕様（読み取り専用）。Windows 側の MDF を編集し、Web UI の「MDFから読込」で再読み込み |
| Qt が対話サーバーにつながらない（Docker） | コンテナの起動途中は `[DockerBackend][WARN] ...` を表示して 1.5 秒ごとに再試行する。`docker compose ps orchestrator` でサービスの状態を確認 |
| Web UI で MMDAgent-EX を起動できない | [失敗の目印](#3-mmdagent-ex-の起動)を確認 |
