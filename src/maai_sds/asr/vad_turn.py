"""VAD（どこが発話か）とターン判定（話者交替が起きたか）。

`asr/service.ASRService` から**そのまま移した**もの（mixin）。
どの ASR エンジンを使っていても共通の関心事なので、エンジン依存部分
（`asr_engines.py`）とは分けてある。

VAD バックエンドは silero / maai、ターンは maai / smart-turn を切り替えられる。
初期化が遅延なのは、使わないバックエンドのモデルを読まないため。
"""

from __future__ import annotations

import queue
import time
from pathlib import Path

import numpy as np
from maai import Maai, MaaiInput

from .frame_types import AudioFrame, SegmentLoopHooks


class VadTurnMixin:
    """VAD とターン判定。`ASRService` に mixin する。"""

    def _initialize_turn_backend_if_needed(self) -> None:
        if self.turn_backend != "smart-turn":
            return
        if not self.smart_turn_model_path.strip() or not Path(self.smart_turn_model_path.strip()).exists():
            raise RuntimeError("smart-turn モデルパスが不正です。")
        try:
            import onnxruntime as ort
        except Exception as ex:
            raise RuntimeError(f"smart-turn 依存 onnxruntime の読み込みに失敗しました: {ex}") from ex
        try:
            from transformers import WhisperFeatureExtractor
        except Exception:
            try:
                from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
            except Exception as ex:
                raise RuntimeError(f"smart-turn 依存 transformers の読み込みに失敗しました: {ex}") from ex

        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        # intra_op を指定しないと ORT は「コア数-1」本のスレッドを作る
        # （24 論理コアのマシンで実測 23 本）。smart-turn は発話中ずっと走るので、
        # 放置すると MaAI(torch 8 + ONNX 8) や TTS とコアを奪い合い、
        # MaAI が実時間（180ms チャンクを 108ms で処理）に追いつかなくなる。
        session_options.intra_op_num_threads = int(self.smart_turn_num_threads)
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self._smart_turn_session = ort.InferenceSession(self.smart_turn_model_path, sess_options=session_options)
        except Exception as ex:
            raise RuntimeError(f"smart-turn ONNXセッション初期化に失敗しました: {ex}") from ex
        self._smart_turn_feature_extractor = WhisperFeatureExtractor(chunk_length=8)

    def _ensure_vad_backend_initialized(self) -> None:
        with self._vad_init_lock:
            if self.vad_backend == "silero":
                if self._silero_model is not None:
                    return
            elif self.vad_backend == "maai":
                if self._maai_instance is not None and self._maai_mic_chunk is not None and self._maai_zero_chunk is not None:
                    return
            self._initialize_vad_backend()

    @staticmethod
    def _load_silero_vad():
        """silero-vad を読む。import の副作用で torch のスレッド数が変わるのを戻す。

        silero_vad/model.py はモジュール直下で torch.set_num_threads(1) を実行する。
        これはプロセス全体に効くので、これまでは silero を使わない設定でも
        （vad_turn を import しただけで）ASR プロセスの torch が 1 スレッドに
        固定されていた。turn_backend="maai" のときは同じプロセスでターン判定用の
        MaAI が torch で走るため、そちらが 1 スレッドに絞られてしまう。

        silero 本体は ONNX 側で intra/inter=1 を明示しているので、torch のスレッド数を
        戻しても silero の負荷は増えない。
        """
        import torch  # noqa: PLC0415

        before = torch.get_num_threads()
        import silero_vad  # noqa: PLC0415

        if torch.get_num_threads() != before:
            torch.set_num_threads(before)
        return silero_vad.load_silero_vad(onnx=True)

    def _initialize_vad_backend(self) -> None:
        if self.vad_backend == "silero":
            self._silero_model = self._load_silero_vad()
            return

        if self.vad_backend == "maai":
            self._maai_mic_chunk = MaaiInput.Chunk()
            self._maai_zero_chunk = MaaiInput.Chunk()
            self._maai_instance = Maai(
                mode=self.maai_vap_mode,
                lang=self.maai_vap_lang,
                frame_rate=float(self.frame_rate),
                audio_ch1=self._maai_mic_chunk,
                audio_ch2=self._maai_zero_chunk,
                device=self.maai_vap_device,
                model_type=self.maai_vap_model_type,
            )
            self._maai_instance.start()
            return

        raise RuntimeError(f"未対応のVADバックエンドです：{self.vad_backend}")

    def _vad_loop(self) -> None:
        self._run_segment_loop(
            source_queue=self.raw_audio_queue,
            hooks=SegmentLoopHooks(emit_to_inference=True),
        )

    def _turn_segment_loop(self) -> None:
        self._run_segment_loop(
            source_queue=self.turn_audio_queue,
            hooks=SegmentLoopHooks(emit_to_inference=False),
        )

    def _get_vad_score(self, audio_chunk: np.ndarray, frame: AudioFrame | None = None) -> float | None:
        if self.vad_backend == "silero":
            if self._silero_model is None:
                return None
            try:
                import torch
            except Exception as ex:
                self._error(f"silero-vad 実行に必要な torch の読み込みに失敗しました: {ex}")
                return None
            try:
                self._silero_input_buffer = (
                    np.concatenate([self._silero_input_buffer, np.asarray(audio_chunk, dtype=np.float32).reshape(-1)])
                    if self._silero_input_buffer.size > 0
                    else np.asarray(audio_chunk, dtype=np.float32).reshape(-1).copy()
                )
                chunks, self._silero_input_buffer = self._split_fixed_chunks(
                    self._silero_input_buffer, self._silero_model_chunk_size
                )
                if not chunks:
                    return None
                score: float | None = None
                for chunk in chunks:
                    score = float(self._silero_model(torch.from_numpy(chunk), self.sample_rate).item())
                if score is None:
                    return None
                self._latest_silero_vad_score = score
                # MaAI VAD バックエンド時は get_result 経由で _emit_maai_result されるが、
                # Silero のみのときは推論が無いためここからプロット用イベントを送る。
                if frame is not None:
                    self._emit_maai_result(
                        {"silero_vad_plot_only": True},
                        source_ts_start=float(frame.ts_start),
                        source_ts_end=float(frame.ts_end),
                        source_sample_start=int(frame.sample_start),
                        source_sample_end=int(frame.sample_end),
                    )
                return score
            except Exception as ex:
                self._error(f"silero-vad 推論に失敗しました: {ex}")
                return None

        if self.vad_backend == "maai":
            if self._maai_instance is None or self._maai_mic_chunk is None or self._maai_zero_chunk is None:
                return None
            try:
                self._vap_maai_vad_buffer = (
                    np.concatenate([self._vap_maai_vad_buffer, np.asarray(audio_chunk, dtype=np.float32).reshape(-1)])
                    if self._vap_maai_vad_buffer.size > 0
                    else np.asarray(audio_chunk, dtype=np.float32).reshape(-1).copy()
                )
                chunks, self._vap_maai_vad_buffer = self._split_fixed_chunks(
                    self._vap_maai_vad_buffer, self._vap_maai_chunk_size
                )
                if not chunks:
                    return None
                last_vad: float | None = None
                for chunk in chunks:
                    self._maai_mic_chunk.put_chunk(chunk)
                    self._maai_zero_chunk.put_chunk(self._get_system_audio_chunk(chunk.shape[0]))
                    # 同期取得: 現在フレームに対応する最新のMAAI結果を待って取得する（結果が無いチャンクもある）。
                    for result in self._take_maai_vad_results():
                        if frame is not None:
                            self._emit_maai_result(
                                result,
                                source_ts_start=float(frame.ts_start),
                                source_ts_end=float(frame.ts_end),
                                source_sample_start=int(frame.sample_start),
                                source_sample_end=int(frame.sample_end),
                            )
                        else:
                            self._emit_maai_result(result)
                        last_vad = float(result["vad"][0])
                return last_vad
            except Exception as ex:
                self._error(f"MAAI VAD 推論に失敗しました: {ex}")
                return None
        return None

    # MaAI の結果を待つ上限。1 チャンクの推論（CPU で数十 ms）より十分長く、超えても VAD が 1 フレーム遅れるだけ。
    MAAI_VAD_RESULT_TIMEOUT_SEC = 0.5

    def _take_maai_vad_results(self) -> list[dict]:
        """チャンクを 1 つ入れたあとの MaAI の結果を受け取る（0 個のこともある）。

        MaAI はチャンクごとに結果を 1 つ返すとは限らない。Mimi を ONNX で動かすモデル
        （normal-ver2 など）は最初のエンコーダー出力を捨てて結果を出さず、音声特徴が
        取れなかったフレームも結果を出さない。Maai.get_result() は結果が来るまで待ち続けるため、
        それを呼ぶと VAD のループが止まり、ASR が区間を 1 つも切り出さなくなる。
        待つ時間に上限を付け、遅れて届いた結果もここでまとめて受け取る。
        """
        result_queue = getattr(self._maai_instance, "result_dict_queue", None)
        if result_queue is None:
            return [self._maai_instance.get_result()]
        results: list[dict] = []
        try:
            results.append(result_queue.get(timeout=self.MAAI_VAD_RESULT_TIMEOUT_SEC))
        except queue.Empty:
            return results
        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                return results

    def _smart_turn_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.smart_turn_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            audio_segment = item["audio"]
            try:
                probability = self._predict_turn_probability(audio_segment)
                if probability is not None and self.on_turn_probability is not None:
                    self.on_turn_probability(probability, time.time())
            except Exception as ex:
                self._error(f"smart-turn 推論に失敗しました: {ex}")

    def _predict_turn_probability(self, audio_segment: np.ndarray) -> float | None:
        if self.turn_backend != "smart-turn":
            return None
        if self._smart_turn_session is None or self._smart_turn_feature_extractor is None:
            return None
        target_samples = 8 * self.sample_rate
        current_samples = len(audio_segment)
        if current_samples >= target_samples:
            windowed = audio_segment[-target_samples:].astype(np.float32)
        else:
            padding = np.zeros(target_samples - current_samples, dtype=np.float32)
            windowed = np.concatenate([padding, audio_segment.astype(np.float32)])
        features = self._smart_turn_feature_extractor(
            windowed,
            sampling_rate=self.sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=target_samples,
            truncation=True,
            do_normalize=True,
        )
        input_features = np.expand_dims(features.input_features.squeeze(0).astype(np.float32), axis=0)
        outputs = self._smart_turn_session.run(None, {"input_features": input_features})
        return float(outputs[0][0].item())
