"""Chunk capture must not leak nod_para's private VAP heads into turn plots."""
import unittest

from maai_sds.maai.frame_capture import project_captured_frames


class TestPublicChunkOutputs(unittest.TestCase):
    def test_nod_internal_heads_are_removed_without_changing_nod_values(self):
        public = {"p_nod": 0.8, "nod_range": 0.05, "nod_speed": 0.1}
        captured = [
            {**public, "p_nod": 0.2, "p_now": [0.1, 0.9],
             "p_future": [0.3, 0.7], "vad": [0.4, 0.6], "p_bc": 0.99},
            {**public, "p_now": [0.9, 0.1], "p_future": [0.7, 0.3],
             "vad": [0.6, 0.4], "p_bc": 0.01},
        ]
        frames = project_captured_frames(captured, public)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames, [{**public, "p_nod": 0.2}, public])
        self.assertIn("p_now", captured[0])  # Input capture remains intact.

    def test_turn_public_heads_and_per_frame_values_are_preserved(self):
        frames = [
            {"p_now": [0.2, 0.8], "p_future": [0.1, 0.9], "vad": [0.3, 0.4]},
            {"p_now": [0.6, 0.4], "p_future": [0.9, 0.1], "vad": [0.7, 0.2]},
        ]
        self.assertEqual(project_captured_frames(frames, frames[-1]), frames)

    def test_single_frame_fallback_keeps_public_contract(self):
        public = {"p_nod": 0.8, "nod_range": 0.05}
        self.assertEqual(project_captured_frames(None, public), [public])


if __name__ == "__main__":
    unittest.main()
