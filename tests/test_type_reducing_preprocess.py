# -*- coding: utf-8 -*-

import unittest

from src.encoder.pipe_encoder import PipeEncoderBase


class TypeReducingPreprocessTest(unittest.TestCase):
    def test_reducing_tee_removes_equal_size_marker(self):
        encoder = PipeEncoderBase.__new__(PipeEncoderBase)
        encoder._is_reducing_size_by_encoded = lambda _value: True
        entities = {
            "TYPE": {"BODY": "等径三通", "MANU": ["SMLS"]},
            "SIZE": [{"type": "DN", "value": "50"}, {"type": "DN", "value": "40"}],
        }

        result = encoder._preprocess_tee_reducing(entities, "等径三通 DN50x40")

        self.assertEqual(result["TYPE"]["BODY"], "异径三通")

    def test_reducing_lateral_tee_removes_same_size_marker(self):
        self.assertEqual(
            PipeEncoderBase._insert_reducing_before_body("同径斜三通"),
            "异径斜三通",
        )

    def test_reducing_coupling_removes_equal_size_marker(self):
        self.assertEqual(
            PipeEncoderBase._insert_reducing_before_body("等径双口管箍"),
            "异径双口管箍",
        )


if __name__ == "__main__":
    unittest.main()
