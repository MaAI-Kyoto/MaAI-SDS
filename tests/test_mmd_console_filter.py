"""MMDAgent-EX の出力のうち、毎フレームの KEYVALUE_SET だけをコンソールから外す。"""

from __future__ import annotations

import unittest

from maai_sds.mmd.console_filter import is_per_frame_line


class PerFrameLineTest(unittest.TestCase):
    def test_per_frame_keyvalue_lines_are_dropped(self) -> None:
        for line in (
            "[2026/11/23 20:18:17.948] [229236348] Remote: Sent: KEYVALUE_SET|MaAI_Blink|0.0974",
            "[2026/11/23 20:18:17.951] [229236348] MMDAgent: Captured: KEYVALUE_SET|MaAI_Blink|0.0974",
            "[2026/11/22 22:10:01.001] [1] Remote: Sent: KEYVALUE_SET|MaAI_HeadPitch|-0.12",
        ):
            self.assertTrue(is_per_frame_line(line), line)

    def test_other_lines_are_kept(self) -> None:
        for line in (
            "Remote: Sent: MODEL_BINDBONE|MaAI_HeadPitch",
            "MMDAgent: Captured: MODEL_BINDFACE|MaAI_Blink",
            "Remote: Status: SNDBRKS",
            "MMDAgent: Status: loading user mdf file = C:\\MMDAgent-EX\\example\\maai_sds\\maai_sds.mdf",
            "MMDAgent: Sent: KEYVALUE_EVENT_SET|MaAI_Blink",  # 毎フレームではない通知は残す
        ):
            self.assertFalse(is_per_frame_line(line), line)


if __name__ == "__main__":
    unittest.main()
