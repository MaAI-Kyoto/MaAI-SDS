import unittest

from maai_sds.audio.convert import Pcm16BeLeadingSilenceGate, strip_leading_silence_pcm16be


class TestLeadingSilenceStreaming(unittest.TestCase):
    def test_odd_byte_boundaries_match_complete_pcm(self):
        pcm = b"\x00\x00" * 100 + b"\x20\x00" * 80 + b"\x00\x00" * 40
        expected = strip_leading_silence_pcm16be(pcm)
        for size in (1, 3, 37, 128, 257):
            with self.subTest(size=size):
                gate = Pcm16BeLeadingSilenceGate()
                chunks = [gate.feed(pcm[i:i + size]) for i in range(0, len(pcm), size)]
                self.assertTrue(all(len(chunk) % 2 == 0 for chunk in chunks))
                self.assertEqual(b"".join(chunks) + gate.flush(), expected)

    def test_negative_full_scale_sample_is_audible(self):
        pcm = b"\x00\x00" * 5 + b"\x80\x00" + b"\x00\x00" * 5
        self.assertEqual(strip_leading_silence_pcm16be(pcm), pcm[10:])
        gate = Pcm16BeLeadingSilenceGate()
        self.assertEqual(gate.feed(pcm), pcm[10:])


if __name__ == "__main__":
    unittest.main()
