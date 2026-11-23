"""MMDAgent-EX の標準出力のうち、プロセスコンソールに出さない行。

まばたきと頷きは KEYVALUE_SET で毎フレーム（約 30 回/秒）値を送り、MMDAgent-EX はその 1 通ごとに
「Remote: Sent: KEYVALUE_SET|...」と「MMDAgent: Captured: KEYVALUE_SET|...」の 2 行を出す。
実測では MMDAgent の出力の 9 割以上がこの 2 種類で、接続・読み込みのメッセージやエラーが埋もれる。
送っている値は「頷きモーション波形」の画面で見られるので、コンソールには出さない。
"""

from __future__ import annotations

import re

_PER_FRAME = re.compile(r"(?:Remote: Sent|MMDAgent: Captured): KEYVALUE_SET\|")


def is_per_frame_line(line: str) -> bool:
    """毎フレーム出る KEYVALUE_SET の行なら True（コンソールに出さない）。"""
    return bool(_PER_FRAME.search(line))
