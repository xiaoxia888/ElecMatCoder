from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from apps.trainer.confidence.train_encoding_confidence import FeatureHasher, Record
from src.confidence.encoding_correctness import (
    EncodingCorrectnessPredictor,
    extract_difficulty,
    extract_field_codes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "encoding-confidence"
    / "20260731-no-old-confidence"
    / "encoding_confidence_model.npz"
)


class EncodingCorrectnessRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = EncodingCorrectnessPredictor(MODEL_PATH)

    def test_runtime_features_match_training_features(self) -> None:
        field_codes = (
            ("TYPE", "90ELW"),
            ("SIZE", "250"),
            ("THICKNESS", "S40"),
            ("PRESSURE", ""),
            ("MATERIAL", "304L"),
            ("STANDARD", "GBT12459"),
        )
        record = Record(
            source="管件",
            description="90度长半径弯头 DN250 SCH40 S30403 GB/T12459",
            predicted_code="90ELW250S40304LGBT12459",
            human_code="90ELW250S40304LGBT12459",
            project="测试项目",
            category="管件",
            current_confidence=None,
            difficulty=1,
            field_codes=field_codes,
            is_correct=1,
        )
        training_hasher = FeatureHasher(
            dimension=self.predictor.dimension,
            max_description_chars=self.predictor.max_description_chars,
            include_difficulty=True,
            include_old_confidence=False,
        )
        expected = np.sort(training_hasher.transform(record))
        actual = np.sort(
            self.predictor.feature_indices(
                description=record.description,
                predicted_code=record.predicted_code,
                category=record.category,
                field_codes=record.field_codes,
                difficulty=record.difficulty,
            )
        )
        np.testing.assert_array_equal(actual, expected)

    def test_predict_result_is_deterministic_and_bounded(self) -> None:
        result = {
            "original_text": "直管 PIPE A106 Gr.B SMLS SCH40 DN50",
            "final_code": "P50S40A106",
            "material_category": "直管",
            "routing": {"final_level": 2},
            "fields": {
                "TYPE": {"stage2_output": {"code": "P"}},
                "SIZE": {"stage2_output": {"code": "50"}},
                "THICKNESS": {"stage2_output": {"code": "S40"}},
                "PRESSURE": {"stage2_output": {"code": ""}},
                "MATERIAL": {"stage2_output": {"code": "A106"}},
                "STANDARD": {"stage2_output": {"code": ""}},
            },
        }
        first, difficulty = self.predictor.predict_result(result)
        second, _ = self.predictor.predict_result(result)
        self.assertEqual(difficulty, 2)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)

    def test_platform_payload_extractors(self) -> None:
        result = {
            "routing": {"final_level": None},
            "second_pass": {"final_level": 1},
            "fields": {
                "TYPE": {"stage2_output": {"code": "T"}},
                "SIZE": {"stage2_output": {"code": "25"}},
            },
        }
        self.assertEqual(extract_difficulty(result), 1)
        self.assertEqual(
            extract_field_codes(result),
            (
                ("TYPE", "T"),
                ("SIZE", "25"),
                ("THICKNESS", ""),
                ("PRESSURE", ""),
                ("MATERIAL", ""),
                ("STANDARD", ""),
            ),
        )


if __name__ == "__main__":
    unittest.main()
