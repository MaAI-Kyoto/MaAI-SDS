"""頷き・まばたきを公式 MMDAgent-EX の仕組み（MODEL_BINDBONE / MODEL_BINDFACE + KEYVALUE_SET）で送る。"""

from __future__ import annotations

import threading
import time
import unittest

from maai_sds.mmd import avatar_keys as keys
from maai_sds.mmd.blink import AutoBlinker
from maai_sds.mmd.remote_bridge import MmdRemoteBridge
from maai_sds.mmd.runtime import MmdRuntime


class AvatarKeysTest(unittest.TestCase):
    def test_head_binding_points_up_for_positive_pitch(self) -> None:
        head, left, right = keys.head_bind_commands("gene", "頭", eye_scale=0.5)
        # 数値 -1 で X 軸 +57.3°（下向き）、+1 で -57.3°（上向き）
        self.assertEqual(head, "MODEL_BINDBONE|MaAI_HeadPitch|-1|1|gene|頭|0,0,0|57.2958,0,0|0,0,0|-57.2958,0,0\n")
        # 目は頭と逆向きに eye_scale 倍
        self.assertEqual(left, "MODEL_BINDBONE|MaAI_HeadPitch|-1|1|gene|左目|0,0,0|-28.6479,0,0|0,0,0|28.6479,0,0\n")
        self.assertIn("|右目|", right)

    def test_no_eye_binding_when_eye_scale_is_zero(self) -> None:
        self.assertEqual(len(keys.head_bind_commands("0", "首", 0.0)), 1)

    def test_values(self) -> None:
        self.assertEqual(keys.head_pitch_command(0.05), "KEYVALUE_SET|MaAI_HeadPitch|0.0500\n")
        self.assertEqual(keys.head_pitch_command(5.0), "KEYVALUE_SET|MaAI_HeadPitch|1.0000\n")
        self.assertEqual(keys.blink_command(1.2), "KEYVALUE_SET|MaAI_Blink|1\n")
        self.assertEqual(keys.blink_command(0.25), "KEYVALUE_SET|MaAI_Blink|0.25\n")
        self.assertEqual(keys.blink_bind_command("gene", "まばたき"), "MODEL_BINDFACE|MaAI_Blink|0|1|gene|まばたき|0|1\n")


class _Bridge(MmdRemoteBridge):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []
        self.connected = True

    def send_text(self, text: str) -> bool:  # type: ignore[override]
        self.sent.append(text)
        return True

    def is_connected(self) -> bool:  # type: ignore[override]
        return self.connected


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class BridgeBindingTest(unittest.TestCase):
    def test_bindings_are_sent_once_and_again_when_the_model_loads(self) -> None:
        b = _Bridge()
        b._model_alias = "gene"
        b.set_head_binding("頭", 0.0)
        b.set_blink_binding("まばたき")
        self.assertEqual(len(b.sent), 1 + 2)  # 頭（1 回目）＋ 頭とまばたき（2 回目）
        b.sent.clear()
        b.set_head_binding("頭", 0.0)  # 変わらなければ送らない
        self.assertEqual(b.sent, [])
        # 別のモデルの読み込みでは送らない。自分のモデルなら送り直す（行が途中で切れて届いても）
        b._on_inbound("MODEL_EVENT_ADD|other\nMODEL_EVENT_A")
        b._on_inbound(b"DD|gene\n")
        self.assertTrue(_wait_for(lambda: len(b.sent) == 2))
        self.assertTrue(b.sent[0].startswith("MODEL_BINDBONE|MaAI_HeadPitch|-1|1|gene|頭|"))
        self.assertTrue(b.sent[1].startswith("MODEL_BINDFACE|MaAI_Blink|0|1|gene|まばたき|"))

    def test_bindings_follow_the_model_alias(self) -> None:
        b = _Bridge()
        b.set_blink_binding("まばたき")
        b.sent.clear()
        b.configure({"model_alias": "gene", "ws_port": 0})
        self.assertTrue(_wait_for(lambda: any("|gene|まばたき|" in s for s in b.sent)))

    def test_no_bindings_before_anything_is_requested(self) -> None:
        b = _Bridge()
        b._on_inbound("MODEL_EVENT_ADD|0\n")
        time.sleep(0.05)
        self.assertEqual(b.sent, [])


class NodAndBlinkSendTest(unittest.TestCase):
    def test_nod_binds_the_head_then_streams_keyvalues(self) -> None:
        sent: list[dict] = []
        rt = MmdRuntime(emit=lambda _e: None, send_audio_output=lambda p: sent.append(p) or True, log_fn=lambda _m: None)
        rt.on_remote_status_event({"connected": True})
        rt.start_nod_motion({"range_rad": 0.1, "count": 1, "fps": 30, "velocity": 2.0, "bone": "首", "eye_scale": 0.0})
        self.assertTrue(_wait_for(lambda: rt._nod_thread is not None and not rt._nod_thread.is_alive(), 5.0))
        self.assertEqual(sent[0], {"type": "mmd_bind_head", "bone": "首", "eye_scale": 0.0})
        texts = [p["text"] for p in sent[1:]]
        self.assertTrue(texts and all(t.startswith("KEYVALUE_SET|MaAI_HeadPitch|") for t in texts))
        self.assertEqual(texts[-1], "KEYVALUE_SET|MaAI_HeadPitch|0.0000\n")

    def test_blinker_sends_blink_keyvalues(self) -> None:
        sent: list[str] = []
        done = threading.Event()
        blinker = AutoBlinker(lambda t: sent.append(t) or True, enable=False)
        self.assertTrue(blinker._blink_once("まばたき"))
        done.set()
        self.assertTrue(sent and all(t.startswith("KEYVALUE_SET|MaAI_Blink|") for t in sent))
        self.assertEqual(sent[-1], "KEYVALUE_SET|MaAI_Blink|0\n")


if __name__ == "__main__":
    unittest.main()
