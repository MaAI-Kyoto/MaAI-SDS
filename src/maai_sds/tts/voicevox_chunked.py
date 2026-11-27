"""VOICEVOX-Core を文節（アクセント句）単位に分けて合成し、最初の声までの時間を縮める。

VOICEVOX-Core の合成時間は、ほぼ「固定分 約 90 ms ＋ 音声 1 秒あたり 約 70 ms」（M4 Max・CPU 10 スレッドの実測）。
1 文をまとめて合成すると、最初の声までに文の長さに比例した時間がかかる（23 文字で約 320 ms）。
そこで AudioQuery（抑揚と長さは文全体から決まったもの）を文節ごとに分け、先頭の文節から順に
波形にして返す（最初の文節だけなら約 110〜130 ms）。

分け方の要点:

- 抑揚（モーラごとの高さ・長さ）は文全体の AudioQuery から決まっているので、分けても変わらない。
  分けるのは波形を作る段階だけ。
- ただし波形を作るモデルは前後の文脈も見るので、文節だけを切り出して合成すると、つなぎ目で
  スペクトルが大きくずれる（実測で最大 約 24 dB、プツッと鳴る）。前後の文節の数モーラを
  「のりしろ」として一緒に合成し、そこを切り落として 10 ms でクロスフェードすると、
  つなぎ目のずれは文中のほかの場所と同じ程度（2〜4 dB）に収まる。
- のりしろを正確に切り落とすため、各音素の長さからサンプル数を計算する。VOICEVOX は音素ごとに
  `round(長さ × 93.75)` フレーム（1 フレーム 256 サンプル @ 24 kHz）にする。話速（speed_scale）と
  疑問文の語尾上げ（合成時にモーラを 1 つ足す）はこの計算を狂わせるので、どちらも先に
  AudioQuery へ反映させておき、合成には speed_scale=1・語尾上げなしで渡す。
- 計算した長さと合成結果の長さが合わないときは、残りをまとめて 1 回で合成する（安全側に倒す）。
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import Any, Callable, Sequence

import numpy as np

SAMPLE_RATE = 24000
HOP = 256
FRAME_RATE = SAMPLE_RATE / HOP  # 93.75
# 疑問文の語尾上げ（voicevox_core の apply_interrogative_upspeak と同じ値）
_UPSPEAK_LENGTH = 0.15
_UPSPEAK_PITCH_ADD = 0.3
_UPSPEAK_PITCH_MAX = 6.5

SynthFn = Callable[[Any], np.ndarray]
"""AudioQuery を受け取り、24 kHz・int16 の波形（1 次元）を返す関数。語尾上げなしで合成すること。"""


def _frames(seconds: float) -> int:
    # VOICEVOX は長さを f32 で持つので、いったん f32 に丸めてから 93.75 倍する（0.08 のような値で
    # ちょうど .5 になるときに結果が変わる）。丸めは Rust の round（0.5 は 0 から遠い方へ）に合わせる。
    # Python の round は偶数丸めなので使わない。
    return int(math.floor(float(np.float32(seconds)) * FRAME_RATE + 0.5))


def _pause_len(phrase: Any) -> float | None:
    pm = getattr(phrase, "pause_mora", None)
    if pm is None:
        return None
    return float(pm["vowel_length"] if isinstance(pm, dict) else pm.vowel_length)


def _speed_length(seconds: float, speed: float) -> float:
    """話速を掛けた音素の長さを、VOICEVOX が出すのと同じフレーム数ちょうどの秒数で返す。

    VOICEVOX は音素ごとに「長さ（f32）を 93.75 倍して丸めたフレーム数」を話速で割り、もう一度丸める
    （実測で 48 件中 47 件一致。単純に長さを話速で割ってから丸めると 13 件しか合わない）。
    フレーム数 k に対して k / 93.75 を返すので、speed_scale=1 で合成すればちょうど k フレームになる。
    """
    base = _frames(seconds)
    k = int(math.floor(base / float(np.float32(speed)) + 0.5))
    return k / FRAME_RATE


def _scale_mora(mora: Any, speed: float) -> None:
    if isinstance(mora, dict):
        if mora.get("consonant_length") is not None:
            mora["consonant_length"] = _speed_length(mora["consonant_length"], speed)
        mora["vowel_length"] = _speed_length(mora["vowel_length"], speed)
        return
    if mora.consonant_length is not None:
        mora.consonant_length = _speed_length(mora.consonant_length, speed)
    mora.vowel_length = _speed_length(mora.vowel_length, speed)


def _mora_frames(mora: Any) -> int:
    if isinstance(mora, dict):
        c, v = mora.get("consonant_length"), mora["vowel_length"]
    else:
        c, v = mora.consonant_length, mora.vowel_length
    return (_frames(c) if c is not None else 0) + _frames(v)


def phrase_frames(phrase: Any) -> int:
    """文節 1 つ（ポーズ込み）のフレーム数。"""
    n = sum(_mora_frames(m) for m in phrase.moras)
    pause = _pause_len(phrase)
    return n + (_frames(pause) if pause is not None else 0)


def _upspeak_mora(last: Any) -> Any:
    pitch = min(float(last.pitch) + _UPSPEAK_PITCH_ADD, _UPSPEAK_PITCH_MAX)
    return dataclasses.replace(
        last,
        text=_vowel_kana(str(last.vowel)),
        consonant=None,
        consonant_length=None,
        vowel_length=_UPSPEAK_LENGTH,
        pitch=pitch,
    )


def _vowel_kana(vowel: str) -> str:
    return {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ", "N": "ン", "A": "ア", "I": "イ", "U": "ウ", "E": "エ", "O": "オ"}.get(vowel, "ー")


def prepare_query(query: Any, *, interrogative_upspeak: bool = True) -> Any:
    """話速と疑問文の語尾上げを AudioQuery に焼き込んだ複製を返す（speed_scale=1・語尾上げ済み）。"""
    q = copy.deepcopy(query)
    if interrogative_upspeak:
        for i, phrase in enumerate(q.accent_phrases):
            if not getattr(phrase, "is_interrogative", False) or not phrase.moras:
                continue
            last = phrase.moras[-1]
            if float(last.pitch) <= 0.0:
                continue
            q.accent_phrases[i] = dataclasses.replace(phrase, moras=[*phrase.moras, _upspeak_mora(last)])
    for i, phrase in enumerate(q.accent_phrases):
        if getattr(phrase, "is_interrogative", False):
            q.accent_phrases[i] = dataclasses.replace(phrase, is_interrogative=False)
    speed = float(q.speed_scale or 1.0)
    if speed > 0 and speed != 1.0:
        for phrase in q.accent_phrases:
            for mora in phrase.moras:
                _scale_mora(mora, speed)
            if phrase.pause_mora is not None:
                _scale_mora(phrase.pause_mora, speed)
        q.pre_phoneme_length = _speed_length(q.pre_phoneme_length, speed)
        q.post_phoneme_length = _speed_length(q.post_phoneme_length, speed)
        q.speed_scale = 1.0
    return q


def plan_chunks(
    phrase_frame_counts: Sequence[int],
    *,
    first_min_sec: float = 0.3,
    next_min_sec: float = 0.8,
    growth: float = 3.0,
    min_total_sec: float = 1.0,
) -> list[tuple[int, int]]:
    """文節の並びを [start, end) のかたまりに分ける。

    最初のかたまりは first_min_sec 以上になるまで文節を足す（短いほど最初の声が早い）。
    2 つ目以降は「前のかたまりの growth 倍」か next_min_sec の大きい方まで足す。合成時間は
    固定分 ＋ 長さに比例する分なので、前のかたまりを再生している間に次を作り終えられるよう、
    かたまりを倍々で大きくして、つなぎ目と固定分を減らす。全体が min_total_sec 未満なら分けない。
    """
    n = len(phrase_frame_counts)
    secs = [c / FRAME_RATE for c in phrase_frame_counts]
    if n <= 1 or sum(secs) < min_total_sec:
        return [(0, n)] if n else []
    chunks: list[tuple[int, int]] = []
    start = 0
    target = first_min_sec
    while start < n:
        end, dur = start, 0.0
        while end < n and (end == start or dur < target):
            dur += secs[end]
            end += 1
        chunks.append((start, end))
        start = end
        target = max(next_min_sec, dur * growth)
    return chunks


class CostModel:
    """合成 1 回にかかる時間を「固定分 a ＋ 音声 1 秒あたり b」とみなし、実測で倍率を合わせる。

    初期値は M4 Max・CPU 10 スレッド・ほかに負荷がないときの実測。MaAI や別のアプリが動いていると
    2〜3 倍遅くなるので、合成のたびに実測との比（scale）を指数移動平均で追う。エンジンが持ち続ける。
    """

    def __init__(self, fixed_sec: float = 0.09, per_audio_sec: float = 0.07, scale: float = 1.0) -> None:
        self.fixed_sec = float(fixed_sec)
        self.per_audio_sec = float(per_audio_sec)
        self.scale = float(scale)

    def predict(self, audio_sec: float) -> float:
        return self.scale * (self.fixed_sec + self.per_audio_sec * max(0.0, float(audio_sec)))

    def observe(self, audio_sec: float, elapsed_sec: float) -> None:
        base = self.fixed_sec + self.per_audio_sec * max(0.0, float(audio_sec))
        if base <= 0 or elapsed_sec <= 0:
            return
        ratio = min(8.0, max(0.25, float(elapsed_sec) / base))
        self.scale = 0.6 * self.scale + 0.4 * ratio


def plan_chunks_adaptive(
    phrase_frame_counts: Sequence[int],
    left_ctx_frames: Sequence[int],
    right_ctx_frames: Sequence[int],
    cost: CostModel,
    *,
    margin: float = 1.3,
    first_min_sec: float = 0.25,
    min_total_sec: float = 1.0,
) -> list[tuple[int, int]]:
    """再生が途切れない範囲で、最初の声がいちばん早くなる分け方を選ぶ。

    かたまりは順に 1 つずつ合成し、最初のかたまりができた時点で再生を始めるとみなす。
    i 番目のかたまりは「i-1 番目までの再生が終わる時刻」までに作り終えなければならない
    （予測時間は margin 倍して余裕を見る）。最初のかたまりを短い方から試し、2 つ目以降は締め切りに
    間に合う範囲でできるだけ大きく取る（かたまりが少ないほど固定分が減る）。どの分け方でも間に合わない
    （マシンが遅すぎる）ときは分けない。

    left_ctx_frames[a] は a 番目から始まるかたまりの前に付くのりしろ、right_ctx_frames[b] は b 番目の前で
    終わるかたまりの後ろに付くのりしろのフレーム数（端では 0）。
    """
    n = len(phrase_frame_counts)
    secs = [c / FRAME_RATE for c in phrase_frame_counts]
    if n <= 1 or sum(secs) < min_total_sec:
        return [(0, n)] if n else []

    def synth_time(a: int, b: int) -> float:
        audio = sum(secs[a:b]) + (left_ctx_frames[a] if a > 0 else 0) / FRAME_RATE
        audio += (right_ctx_frames[b] if b < n else 0) / FRAME_RATE
        return cost.predict(audio) * margin

    for j in range(1, n):
        if sum(secs[:j]) < first_min_sec and j < n - 1:
            continue
        chunks = [(0, j)]
        done = synth_time(0, j)  # 最初のかたまりができた時刻 = 再生開始
        play_end = done + sum(secs[:j])
        a, ok = j, True
        while a < n:
            # 締め切り（いまの再生が終わる時刻）までに作り終わる最大の b を探す
            best = None
            for b in range(a + 1, n + 1):
                if done + synth_time(a, b) <= play_end:
                    best = b
                else:
                    break
            if best is None:
                ok = False
                break
            done += synth_time(a, best)
            play_end += sum(secs[a:best])
            chunks.append((a, best))
            a = best
        if ok:
            return chunks
    return [(0, n)]


def _left_context(phrase: Any, n_moras: int) -> Any:
    moras = list(phrase.moras)[-n_moras:]
    return dataclasses.replace(phrase, moras=moras, accent=max(1, min(int(phrase.accent), len(moras))), is_interrogative=False)


def _right_context(phrase: Any, n_moras: int) -> Any:
    moras = list(phrase.moras)[:n_moras]
    keep_pause = len(moras) == len(phrase.moras)
    return dataclasses.replace(
        phrase,
        moras=moras,
        accent=max(1, min(int(phrase.accent), len(moras))),
        pause_mora=phrase.pause_mora if keep_pause else None,
        is_interrogative=False,
    )


class ChunkedSynthesisError(RuntimeError):
    pass


def synthesize_chunked(
    query: Any,
    synth: SynthFn,
    on_audio: Callable[[np.ndarray], None],
    *,
    context_moras: int = 3,
    crossfade_samples: int = 240,
    plan: Callable[[Sequence[int]], list[tuple[int, int]]] | None = None,
    cost: CostModel | None = None,
) -> int:
    """`prepare_query` 済みの AudioQuery を文節単位で合成し、24 kHz・int16 の波形を順に on_audio へ渡す。

    返り値は渡したサンプル数の合計。synth は語尾上げなしで 1 つの AudioQuery を合成する関数。
    cost を渡すと、その速さで再生が途切れない分け方（plan_chunks_adaptive）を選び、合成のたびに
    実測で cost を更新する。plan を渡すとその分け方を使う。どちらも無ければ cost の初期値で選ぶ。
    """
    phrases = list(query.accent_phrases)
    n = len(phrases)
    counts = [phrase_frames(p) for p in phrases]
    if plan is not None:
        chunks = plan(counts)
    else:
        cost = cost or CostModel()
        lctx = [0] + [
            phrase_frames(_left_context(phrases[a - 1], context_moras)) if context_moras > 0 else 0
            for a in range(1, n)
        ]
        rctx = [
            phrase_frames(_right_context(phrases[b], context_moras)) if (context_moras > 0 and b < n) else 0
            for b in range(n)
        ] + [0]
        chunks = plan_chunks_adaptive(counts, lctx, rctx, cost)
    pre_frames = _frames(query.pre_phoneme_length)
    total = 0
    carry: np.ndarray | None = None  # 前のかたまりの末尾（クロスフェード用、2*ov サンプル）
    ov = max(0, int(crossfade_samples))

    def _emit(x: np.ndarray) -> None:
        nonlocal total
        if x.size:
            on_audio(np.clip(np.rint(x), -32768, 32767).astype(np.int16))
            total += int(x.size)

    for ci, (a, b) in enumerate(chunks):
        first, last = a == 0, b == n
        sub = copy.copy(query)
        parts: list[Any] = []
        left_frames = 0
        if not first and context_moras > 0:
            lc = _left_context(phrases[a - 1], context_moras)
            parts.append(lc)
            left_frames = phrase_frames(lc)
        parts.extend(phrases[a:b])
        right_frames = 0
        if not last and context_moras > 0:
            rc = _right_context(phrases[b], context_moras)
            parts.append(rc)
            right_frames = phrase_frames(rc)
        sub.accent_phrases = parts
        sub.pre_phoneme_length = query.pre_phoneme_length if first else 0.0
        sub.post_phoneme_length = query.post_phoneme_length if last else 0.0
        t_start = time.perf_counter()
        audio = np.asarray(synth(sub), dtype=np.float32).reshape(-1)
        if cost is not None:
            cost.observe(audio.size / SAMPLE_RATE, time.perf_counter() - t_start)

        head = (pre_frames if first else 0) + left_frames
        body = sum(counts[a:b])
        tail = (_frames(query.post_phoneme_length) if last else 0) + right_frames
        expected = (head + body + tail) * HOP
        if audio.size != expected:
            if first:
                # まだ何も渡していない。呼び出し側が 1 文まとめての合成に切り替える。
                raise ChunkedSynthesisError(
                    f"chunk {ci} length {audio.size} != expected {expected} (phrases {a}:{b})"
                )
            # 途中で長さが合わなくなった。音を途切れさせないよう、残りを 1 回でまとめて合成して続ける
            # （つなぎ目はのりしろなしになる）。
            rest = copy.copy(query)
            rest.accent_phrases = phrases[a:]
            rest.pre_phoneme_length = 0.0
            if carry is not None and carry.size:
                _emit(carry[: carry.size // 2])
            _emit(np.asarray(synth(rest), dtype=np.float32).reshape(-1))
            return total
        start = head * HOP
        end = (head + body) * HOP if not last else audio.size
        # のりしろが短くクロスフェード分に足りないときは、つなぎ目をそのままつなぐ
        ov_in = ov if (carry is not None and left_frames * HOP >= ov) else 0
        if carry is not None:
            if ov_in and carry.size == 2 * ov_in:
                seg = audio[start - ov_in : start + ov_in]
                w = np.linspace(0.0, 1.0, seg.size, dtype=np.float32)
                _emit(carry * (1.0 - w) + seg * w)
                start += ov_in
            else:
                _emit(carry[: carry.size // 2] if carry.size else carry)
        if last:
            _emit(audio[start:end])
            carry = None
        else:
            ov_out = ov if right_frames * HOP >= ov else 0
            _emit(audio[start : end - ov_out])
            carry = audio[end - ov_out : end + ov_out] if ov_out else np.zeros(0, dtype=np.float32)
    if carry is not None and carry.size:
        _emit(carry[: carry.size // 2])
    return total
