"""Nemotron 系 ASR エンジン（別 venv のサブプロセス版と ONNX 版）。

共通のストリーミングループ `_run_nemotron_style_stream_loop` も
ここに置く（使うのはこの 2 エンジンだけなので、モジュール間の依存が消える）。

`asr/service.ASRService` から**そのまま移した**もの（mixin）。
本文は 1 文字も変えていない。エンジンごとにファイルを分けたのは、
1 エンジンを読む/直すときに他のエンジンを読み飛ばさなくて済むようにするため。
"""

from __future__ import annotations

import math
import queue
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import engine_manager, paths as _paths
from .frame_types import AudioFrame, SegmentLoopHooks


class AsrNemotronEnginesMixin:
    """Nemotron 系 ASR エンジン（別 venv のサブプロセス版と ONNX 版）。 `ASRService` に mixin する。"""

    def _init_engine_nemotron(self) -> None:
        """専用 venv で nemotron_asr_server.py を起動し、ローカルソケットで接続する。

        アプリ env には nemo を入れない（transformers 等を壊さない）。モデル推論のみ
        サブプロセスに委譲し、VAD/発話区切りはアプリ側 `_nemotron_stream_loop` が担う。
        """
        import os
        import subprocess
        import sys

        self._faster_whisper_model = None
        self._sherpa_recognizer = None
        self._nemotron = None

        if self.nemotron_remote_host and self.nemotron_remote_port:
            self._nemotron_proc = None
            self._nemotron_log_fp = None
            self._nemotron_host = self.nemotron_remote_host
            self._nemotron_port = int(self.nemotron_remote_port)
            self._connect_nemotron_client(timeout_sec=240.0)
            print(
                f"[nemotron] ready: connected to remote {self._nemotron_host}:{self._nemotron_port}",
                file=sys.stderr,
                flush=True,
            )
            return

        py = self.nemotron_python
        # 相対パス（既定の .venv_nemotron/Scripts/python.exe 等）はリポジトリルート基準で
        # 解決する。これにより外部の絶対パス（旧 C:\NeMo）に依存しない。
        py = str(_paths.resolve_venv_python(py, ".venv_nemotron"))
        if not py or not Path(py).exists():
            raise RuntimeError(
                "Nemotron 用 Python（専用 venv）が見つかりません。"
                "エンジン管理で Nemotron ASR を適用するか、"
                f"ASR設定の『Nemotron Python』にパスを設定してください: {py!r}"
            )
        server_script = str(_paths.package_path("asr", "nemotron_asr_server.py"))

        # 空きポートを確保してサブプロセスへ渡す。
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        host = "127.0.0.1"
        self._nemotron_host = host
        self._nemotron_port = port

        cmd = [
            py,
            server_script,
            "--host", host,
            "--port", str(port),
            "--model", self.nemotron_model,
            "--chunk-size", str(self.nemotron_chunk_size_ms),
            "--window-sec", str(self.nemotron_window_sec),
            "--max-symbols", str(self.nemotron_max_symbols),
            "--lang", self.nemotron_lang,
            "--device", self.nemotron_device,
        ]
        # サブプロセスの全出力（NeMo ログ・例外・CUDA エラー含む）はログファイルに集約する。
        # IPC へ流すと Orchestrator 側の同期転送で worker send が詰まるため、ファイルに退避する。
        log_dir = _paths.logs_dir()
        try:
            log_dir.mkdir(exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
        log_path = log_dir / f"nemotron_server_{port}.log"
        try:
            self._nemotron_log_fp = open(log_path, "w", encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            self._nemotron_log_fp = None
        # 起動情報はサブプロセスの起動中ログと同じ stderr 経路に流す（[ASR][STDERR] として表示）。
        print(
            f"[nemotron] starting: model={self.nemotron_model} chunk_size={self.nemotron_chunk_size_ms}ms "
            f"lang={self.nemotron_lang} finalize={self.nemotron_finalize_mode} port={port}",
            file=sys.stderr,
            flush=True,
        )

        # 別 venv の python が app venv の設定を拾わないよう、PYTHON* / VIRTUAL_ENV を除いた環境で起動。
        child_env = {
            k: v
            for k, v in os.environ.items()
            if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"}
        }
        child_env["PYTHONUNBUFFERED"] = "1"

        # サブプロセスの起動中ログは stderr 経路へ流しつつ（[ASR][STDERR]）、ファイルにも残す。
        proc = subprocess.Popen(
            cmd,
            cwd=str(_paths.REPO_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        self._nemotron_proc = proc

        def _drain(stream) -> None:
            try:
                for raw in iter(stream.readline, ""):
                    line = (raw or "").rstrip()
                    if not line:
                        continue
                    print(line, file=sys.stderr, flush=True)
                    fp = self._nemotron_log_fp
                    if fp is not None:
                        try:
                            fp.write(line + "\n")
                            fp.flush()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_drain, args=(proc.stdout,), daemon=True, name="asr-nemotron-log").start()

        # **モデルロード完了まで待ってから返す。**
        # サーバはモデルロードを終えてから bind/listen するので、接続できた時点が真の準備完了
        # （nemotron_asr_server.py の「接続成功 = ready の合図」）。
        #
        # ここで待たないと、asr/worker が `emit("ready")` を先に出して GUI が
        # 「初期化完了」と表示するのに実際はまだ使えない、という状態になる。
        # `asr_worker_ready_timeout_sec` が nemotron-asr にだけ 300 秒を割いているのは
        # まさにこの待ち時間のためで、その契約に戻す。
        #
        # `_nemotron_stream_loop` は `self._nemotron is None` のときだけ接続するので、
        # ここで繋いでも二重接続にはならない（サーバは 1 接続ずつ受ける）。
        # 外側の ready 待ち（300 秒）より先に切れるようにして、
        # 「ready タイムアウト」ではなく原因の分かるエラーを出す。
        self._connect_nemotron_client(timeout_sec=240.0)
        print(
            f"[nemotron] ready: connected to 127.0.0.1:{port}",
            file=sys.stderr,
            flush=True,
        )

    def _nemotron_stream_loop(self) -> None:
        if self._nemotron is None:
            # 通常は `_init_engine_nemotron` が接続済み。ここは保険
            # （再起動などで接続が落ちていた場合のみ繋ぎ直す）。
            try:
                self._connect_nemotron_client(timeout_sec=300.0)
            except Exception as ex:  # noqa: BLE001
                self._error(f"Nemotron 初期化に失敗しました: {ex}")
                self._terminate_nemotron_proc()
                return
        if self._stop_event.is_set() or self._nemotron is None:
            return
        self._run_nemotron_style_stream_loop(engine="nemotron-asr", backend=self._nemotron)

    def _init_engine_nemotron_onnx(self) -> None:
        """sherpa-onnx の Nemotron 3.5 streaming（online transducer, 多言語）。

        言語は per-stream の set_option("language", ...) で指定する（PR k2-fsa/sherpa-onnx#3671）。
        nemotron-asr と同様、VAD 区切りまたは continuous でフレーム逐次 feed する。
        """
        engine_manager.require_module_installed("sherpa_onnx", "sherpa-onnx")
        try:
            import sherpa_onnx
        except Exception as ex:
            raise RuntimeError(
                f"sherpa-onnx をインポートできません: {ex}"
            ) from ex
        from .nemotron_onnx_stream import NemotronOnnxStreamingSession
        from .nemotron_chunk import finalize_tail_sec_for_chunk, read_encoder_chunk_size_ms

        chunk_ms = int(self.nemotron_chunk_size_ms)
        from .nemotron_chunk import resolve_nemotron_onnx_chunk_ms

        onnx_chunk_ms = resolve_nemotron_onnx_chunk_ms(chunk_ms)
        if onnx_chunk_ms != int(chunk_ms):
            self._log(
                f"nemotron-onnx: chunk {chunk_ms}ms (NeMo) -> ONNX {onnx_chunk_ms}ms model"
            )
        model_paths = self._resolve_nemotron_onnx_chunk_paths(chunk_ms)
        self._log(
            "nemotron-onnx: loading sherpa-onnx "
            f"{self._sherpa_onnx_version_str()} from {getattr(sherpa_onnx, '__file__', '?')} "
            f"chunk={chunk_ms}ms provider={self.nemotron_onnx_provider} "
            f"quant={self.nemotron_onnx_quant} encoder={model_paths['encoder']}"
        )
        required_paths = [
            ("tokens", model_paths["tokens"]),
            ("encoder", model_paths["encoder"]),
            ("decoder", model_paths["decoder"]),
            ("joiner", model_paths["joiner"]),
        ]
        missing = [name for name, path in required_paths if not path or not Path(path).exists()]
        if missing:
            raise RuntimeError(
                f"Nemotron onnx モデルファイルが不足しています: {', '.join(missing)}。"
                " エンジン管理で「Nemotron ONNX (ASR)」にチェックを入れて適用するか、"
                " scripts/setup_assets.ps1 -Only NemotronOnnxModels（Linux は setup_assets.sh --only NemotronOnnxModels）で取得してください。"
                " fp32 はローカル export 専用です。preprocessor.onnx は setup_assets が assets/ から配置します。"
            )
        encoder_meta_ms = read_encoder_chunk_size_ms(model_paths["encoder"])
        if encoder_meta_ms is not None and int(encoder_meta_ms) != chunk_ms:
            self._log(
                f"nemotron-onnx: warning encoder metadata chunk_size_ms={encoder_meta_ms} "
                f"!= requested {chunk_ms}ms"
            )
        if self._nemotron_onnx_encoder_has_prompt_index(model_paths["encoder"]):
            self._ensure_nemotron_onnx_git_sherpa()
        max_symbols = max(1, int(self.nemotron_max_symbols))
        if max_symbols > 10:
            self._log(
                "nemotron-onnx: nemotron_max_symbols="
                f"{max_symbols} (sherpa Git ビルドは patch_sherpa_nemo_max_symbols で "
                f"max_symbols_per_frame を合わせる必要があります。未パッチ時は 10 固定)"
            )
        self._nemotron_onnx_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=model_paths["tokens"],
            encoder=model_paths["encoder"],
            decoder=model_paths["decoder"],
            joiner=model_paths["joiner"],
            num_threads=self.nemotron_onnx_num_threads,
            sample_rate=self.sherpa_sample_rate,
            feature_dim=self.nemotron_onnx_feature_dim,
            decoding_method="greedy_search",
            provider=self.nemotron_onnx_provider,
            dither=0.0,
        )
        self._verify_nemotron_onnx_runtime()
        self._nemotron_onnx_session = NemotronOnnxStreamingSession(
            self._nemotron_onnx_recognizer,
            sample_rate=self.sherpa_sample_rate,
            apply_language=self._apply_nemotron_onnx_stream_language,
            finalize_tail_sec=finalize_tail_sec_for_chunk(
                onnx_chunk_ms,
                encoder_path=model_paths["encoder"],
                sample_rate=self.sherpa_sample_rate,
            ),
        )
        self._sherpa_recognizer = None
        self._faster_whisper_model = None

    def _nemotron_onnx_stream_loop(self) -> None:
        if self._nemotron_onnx_session is None:
            self._error("nemotron-onnx セッションが初期化されていません。")
            return
        self._run_nemotron_style_stream_loop(
            engine="nemotron-onnx",
            backend=self._nemotron_onnx_session,
        )

    def _decode_with_nemotron_onnx(
        self,
        audio_segment: np.ndarray,
        start_ts: float | None,
        end_ts: float | None,
        *,
        is_final: bool,
        is_streaming: bool,
    ) -> bool:
        if self._nemotron_onnx_recognizer is None:
            raise RuntimeError("nemotron-onnx recognizer が初期化されていません。")
        normalized = np.asarray(audio_segment, dtype=np.float32).flatten()
        if not normalized.flags["C_CONTIGUOUS"]:
            normalized = np.ascontiguousarray(normalized, dtype=np.float32)
        original_sample_count = int(len(normalized))
        start_v, end_v = self._resolve_segment_bounds(start_ts, end_ts, original_sample_count)
        # sherpa と同様、先頭に 0.5 秒無音を付与して cache-aware encoder の先頭を安定化させる。
        # （区間境界 start_v/end_v は付与前の original_sample_count から算出済み）
        lead = np.zeros(int(self.sherpa_sample_rate * 0.5), dtype=np.float32)
        normalized = np.concatenate([lead, normalized])

        rec = self._nemotron_onnx_recognizer
        stream = rec.create_stream()
        self._apply_nemotron_onnx_stream_language(stream)
        stream.accept_waveform(self.sherpa_sample_rate, normalized)
        # 末尾無音を流して encoder を flush し、最終トークンまで出させる。
        tail = np.zeros(int(self.sherpa_sample_rate * 0.3), dtype=np.float32)
        stream.accept_waveform(self.sherpa_sample_rate, tail)
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        text = str(rec.get_result(stream)).strip()
        if text:
            self._publish_result(
                text=text,
                end_ts=end_v,
                engine=self.asr_engine,
                is_final=is_final,
                is_streaming=is_streaming,
                segment_start_ts=start_v,
                segment_end_ts=end_v,
            )
            return True
        return False

    def _run_nemotron_style_stream_loop(self, *, engine: str, backend) -> None:
        """nemotron-asr / nemotron-onnx 共通: VAD 区切り or continuous で feed → partial/final。"""
        self._ensure_vad_backend_initialized()
        while True:
            try:
                self.raw_audio_queue.get_nowait()
            except queue.Empty:
                break

        frame_hz = max(1.0, float(self.sample_rate) / float(max(1, self.block_size)))
        activation_frames = max(1, int(math.ceil(self.vad_start_hold_sec * frame_hz)))
        deactivation_frames = max(1, int(math.ceil(self.vad_end_hold_sec * frame_hz)))
        pre_roll_frames = max(1, int(math.ceil(max(0.0, self.pre_roll_buffer_sec) * frame_hz)))
        pre_roll_buffer: deque[AudioFrame] = deque(maxlen=pre_roll_frames)
        continuous = self.nemotron_finalize_mode == "continuous"
        # sherpa と同様、新しいセグメント（fresh stream）の先頭に 0.5 秒無音を feed して
        # cache-aware encoder の先頭を安定化させる。
        lead_samples = int(self.sample_rate * 0.5)

        @dataclass
        class NemotronLoopState:
            speech_active: bool = False
            activation_count: int = 0
            deactivation_count: int = 0
            segment_start_ts: float | None = None
            segment_end_ts: float | None = None
            last_vad: float | None = None

        st = NemotronLoopState()

        # 重い backend 推論(prime/feed/finalize/reset)は専用ワーカースレッドで順序保証して実行する。
        # 本ループ(_on_frame)は silero VAD と発話区切り判定だけを実時間で回し、コマンドを積むだけ。
        # これにより発話中も silero VAD の可視化が nemotron 推論速度に律速されない。
        cmd_q: "queue.Queue[tuple]" = queue.Queue()
        # worker 専用の停止フラグ(#10)。global _stop_event だけで回すと、stop→backend close 後に
        # start() が _stop_event.clear() した場合、未 join のまま残った旧 worker のループ条件が
        # 永久 True になり、閉じた backend を叩き続けるゾンビになる。per-loop Event で確実に止める。
        worker_stop = threading.Event()
        # feed は支配的。backend が実時間より遅いと continuous/長発話で cmd_q が無制限に伸び、
        # OOM/遅延増大する(#8)。制御(prime/finalize/reset)は必ず通し、feed だけ上限超過時に落とす。
        feed_backlog_cap = max(64, int(round(frame_hz * 30.0)))

        def _enqueue_feed(item) -> None:
            if cmd_q.qsize() >= feed_backlog_cap:
                return
            cmd_q.put(item)

        # 直近に enqueue したが worker が未処理の finalize。stop が queue 済み finalize を追い越して
        # worker が未処理終了すると、その発話の最終結果が失われる(#10 round-3)。finally で救済する。
        # set と cmd_q.put、worker 側の clear を final_lock で原子化する。put→set 順だと worker が
        # put で起床し set 前に finalize 処理+publish を完了→set が公開済み発話を再アームし stop 時に
        # 二重 final を出す(#10 round-4)。set→put 順＋同一性 clear で公開済みの再アームを防ぐ。
        final_pending: "list" = [None]
        final_lock = threading.Lock()

        def _publish_from_worker(text, *, is_final, end_ts, seg_start_ts, last_holder):
            cleaned = (text or "").strip()
            if not cleaned:
                return
            if (not is_final) and cleaned == last_holder[0]:
                return
            last_holder[0] = cleaned
            self._publish_result(
                text=cleaned,
                end_ts=end_ts,
                engine=engine,
                is_final=is_final,
                is_streaming=True,
                segment_start_ts=seg_start_ts,
                segment_end_ts=end_ts,
            )

        def _decode_worker() -> None:
            last_holder = [""]
            while not self._stop_event.is_set() and not worker_stop.is_set():
                try:
                    cmd = cmd_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                op = cmd[0]
                try:
                    if op == "prime":
                        if lead_samples > 0:
                            backend.feed(np.zeros(lead_samples, dtype=np.float32))
                        last_holder[0] = ""
                    elif op == "feed":
                        _, audio, seg_start, end_ts = cmd
                        text = backend.feed(audio)
                        _publish_from_worker(
                            text, is_final=False, end_ts=end_ts, seg_start_ts=seg_start, last_holder=last_holder
                        )
                    elif op == "finalize":
                        _, seg_start, end_ts = cmd
                        with final_lock:
                            # 自分が処理する finalize だけを消費印にする（後から別の finalize が
                            # arm されていたらそれは消さない＝同一性 clear）。
                            if final_pending[0] == (seg_start, end_ts):
                                final_pending[0] = None
                        try:
                            final_text = backend.finalize()
                        except Exception as ex:  # noqa: BLE001
                            self._error(f"{engine} finalize に失敗しました: {ex}")
                            final_text = last_holder[0]
                        _publish_from_worker(
                            final_text, is_final=True, end_ts=end_ts, seg_start_ts=seg_start, last_holder=last_holder
                        )
                        try:
                            backend.reset()
                        except Exception:  # noqa: BLE001
                            pass
                        last_holder[0] = ""
                    elif op == "reset":
                        try:
                            backend.reset()
                        except Exception:  # noqa: BLE001
                            pass
                        last_holder[0] = ""
                except Exception as ex:  # noqa: BLE001
                    self._error(f"{engine} decode worker error: {ex}")

        worker = threading.Thread(target=_decode_worker, daemon=True, name=f"{engine}-decode")
        worker.start()

        def _on_frame(frame: AudioFrame) -> None:
            x = np.ascontiguousarray(frame.audio.astype(np.float32))
            pre_roll_buffer.append(frame)
            vad_raw = self._get_vad_score(x, frame)  # silero（実時間, 可視化 emit はこの中）
            if vad_raw is not None:
                st.last_vad = float(vad_raw)
            effective_vad = float(vad_raw) if vad_raw is not None else st.last_vad

            if continuous:
                if st.segment_start_ts is None:
                    st.segment_start_ts = float(frame.ts_start)
                    cmd_q.put(("prime",))
                st.segment_end_ts = float(frame.ts_end)
                _enqueue_feed(("feed", x, st.segment_start_ts, float(frame.ts_end)))
                return

            is_vad_on = bool(effective_vad is not None and effective_vad > self.vad_start_threshold)
            if not st.speech_active:
                if is_vad_on:
                    st.activation_count += 1
                    if st.activation_count >= activation_frames:
                        st.speech_active = True
                        st.activation_count = 0
                        st.deactivation_count = 0
                        seg_frames = list(pre_roll_buffer)
                        st.segment_start_ts = (
                            float(seg_frames[0].ts_start) if seg_frames else float(frame.ts_start)
                        )
                        st.segment_end_ts = float(frame.ts_end)
                        cmd_q.put(("prime",))  # 新セグメント(reset 後 fresh stream)の先頭を無音で安定化
                        if seg_frames:
                            pre_audio = np.ascontiguousarray(
                                np.concatenate([f.audio for f in seg_frames], axis=0).astype(np.float32)
                            )
                            cmd_q.put(("feed", pre_audio, st.segment_start_ts, float(frame.ts_end)))
                elif vad_raw is not None:
                    st.activation_count = 0
                return

            st.segment_end_ts = float(frame.ts_end)
            _enqueue_feed(("feed", x, st.segment_start_ts, float(frame.ts_end)))

            if vad_raw is not None:
                if vad_raw < self.vad_end_threshold:
                    st.deactivation_count += 1
                    if st.deactivation_count >= deactivation_frames:
                        end_ts = st.segment_end_ts or float(frame.ts_end)
                        with final_lock:
                            # set→put 順。put が worker を起こすので set を先に確定させる。
                            final_pending[0] = (st.segment_start_ts, end_ts)
                            cmd_q.put(("finalize", st.segment_start_ts, end_ts))
                        st.speech_active = False
                        st.activation_count = 0
                        st.deactivation_count = 0
                        st.segment_start_ts = None
                        st.segment_end_ts = None
                        pre_roll_buffer.clear()
                else:
                    st.deactivation_count = 0

        try:
            self._run_segment_loop(
                source_queue=self.raw_audio_queue,
                hooks=SegmentLoopHooks(
                    emit_to_inference=False,
                    emit_to_smart_turn_on_noninference=False,
                    forward_to_turn_queue=True,
                    process_only_frames=True,
                    on_frame=_on_frame,
                    # continuous モードでは speech_active が立たない(#7)。発話中扱いにして
                    # raw バックログ破棄(転写中の音声欠落)を抑止する。メモリは cmd_q 側で上限。
                    is_in_active_segment=lambda: st.speech_active
                    or (continuous and st.segment_start_ts is not None),
                ),
            )
        finally:
            worker_stop.set()
            worker.join(timeout=2.0)
            # ワーカーが時間内に終了した＝backend へ並行アクセスしない場合のみ best-effort で最終確定する。
            if not worker.is_alive():
                if (continuous and st.segment_start_ts is not None) or st.speech_active:
                    # 停止時に未確定（VAD end 前）のセグメントがある場合。
                    try:
                        final_text = backend.finalize()
                    except Exception:  # noqa: BLE001
                        final_text = ""
                    if final_text and final_text.strip():
                        end_ts = st.segment_end_ts or time.time()
                        self._publish_result(
                            text=final_text.strip(),
                            end_ts=end_ts,
                            engine=engine,
                            is_final=True,
                            is_streaming=True,
                            segment_start_ts=st.segment_start_ts,
                            segment_end_ts=end_ts,
                        )
                elif final_pending[0] is not None:
                    # VAD end で finalize は queue 済みだが worker が処理前に停止した場合の救済(#10 round-3)。
                    seg_start, end_ts = final_pending[0]
                    try:
                        final_text = backend.finalize()
                    except Exception:  # noqa: BLE001
                        final_text = ""
                    if final_text and final_text.strip():
                        self._publish_result(
                            text=final_text.strip(),
                            end_ts=end_ts,
                            engine=engine,
                            is_final=True,
                            is_streaming=True,
                            segment_start_ts=seg_start,
                            segment_end_ts=end_ts,
                        )
