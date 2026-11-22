"""ASR まわりで共有する小さなデータ型。

`asr/service.py` に定義されていたが、エンジン部分を `asr_engines.py` へ切り出す際に
循環 import になるためここへ移した（両方がここから読む）。
`asr/worker` / `asr/compare_worker` / `audio/input_worker` も利用する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class AudioFrame:
    audio: np.ndarray
    sample_start: int
    sample_end: int
    ts_start: float
    ts_end: float
    # このフレームを **送り出した側の壁時計時刻**（audio/input_worker の emit 時点）。
    # 端から端まで（マイク → テキスト）の遅延を測るために使う。
    # 音声時刻（ts_start/ts_end）はサンプルクロック由来で壁時計とずれるので、
    # 遅延の計測にはこちらを使う。運ばれてこない経路では None。
    origin_ts: float | None = None

@dataclass
class SegmentLoopHooks:
    emit_to_inference: bool
    emit_to_smart_turn_on_noninference: bool = True
    forward_to_turn_queue: bool = False
    process_only_frames: bool = False
    on_frame: Callable[[AudioFrame], None] | None = None
    on_segment_start: Callable[[AudioFrame], None] | None = None
    on_segment_end: Callable[[float], None] | None = None
    # process_only_frames=True（ストリーミングエンジン）では本ループの speech_active が常に False。
    # その間もバックログ破棄が効くとエンジン発話中の音声を on_frame 到達前に捨ててしまうため、
    # エンジン側の発話状態を返すコールバックを渡し、発話中はバックログを破棄しない。
    is_in_active_segment: Callable[[], bool] | None = None
