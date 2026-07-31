from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from apps.trainer.confidence.train_encoding_confidence import (
    FeatureHasher,
    Record,
    apply_platt,
    fit_platt_scaler,
    normalize_code,
    partition_records,
    split_projects,
)


def make_record(
    project: str,
    description: str,
    predicted: str,
    human: str,
    difficulty: int = 0,
) -> Record:
    return Record(
        source="管件",
        description=description,
        predicted_code=predicted,
        human_code=human,
        project=project,
        category="管件",
        current_confidence=0.68,
        difficulty=difficulty,
        field_codes=(("TYPE", "T"), ("SIZE", "20")),
        is_correct=int(normalize_code(predicted) == normalize_code(human)),
    )


class ConfidenceTrainingTests(unittest.TestCase):
    def test_code_normalization_is_conservative(self) -> None:
        self.assertEqual(normalize_code(" rt 20x15 "), "RT20X15")
        self.assertNotEqual(normalize_code("RT20-15"), "RT20X15")

    def test_project_split_has_no_project_overlap(self) -> None:
        records = [
            make_record(
                f"项目{project}",
                f"描述{project}-{row}",
                "T20",
                "T20" if row % 4 else "RT20",
            )
            for project in range(12)
            for row in range(8)
        ]
        split = split_projects(records, 20260731, 0.2, 0.2)
        self.assertFalse(split["train"] & split["calibration"])
        self.assertFalse(split["train"] & split["test"])
        self.assertFalse(split["calibration"] & split["test"])
        self.assertEqual(
            split["train"] | split["calibration"] | split["test"],
            {record.project for record in records},
        )

    def test_description_overlap_can_be_purged(self) -> None:
        records = [
            make_record("训练项目", "重复描述", "T20", "T20"),
            make_record("校准项目", "校准描述", "T20", "T20"),
            make_record("测试项目", "重复描述", "T20", "RT20"),
        ]
        project_split = {
            "train": {"训练项目"},
            "calibration": {"校准项目"},
            "test": {"测试项目"},
        }
        partitions, removed = partition_records(
            records,
            project_split,
            purge_description_overlap=True,
        )
        self.assertEqual(len(partitions["train"]), 0)
        self.assertEqual(removed["train_overlap_removed"], 1)

    def test_project_and_difficulty_are_not_features(self) -> None:
        hasher = FeatureHasher(dimension=1 << 12)
        first = make_record("项目A", "三通 DN20", "T20", "T20", difficulty=0)
        second = make_record("项目B", "三通 DN20", "T20", "T20", difficulty=2)
        self.assertTrue(
            np.array_equal(
                np.sort(hasher.transform(first)),
                np.sort(hasher.transform(second)),
            )
        )

    def test_old_confidence_can_be_excluded_from_features(self) -> None:
        low_confidence = make_record("项目A", "三通 DN20", "T20", "T20")
        high_confidence = replace(low_confidence, current_confidence=0.98)

        default_hasher = FeatureHasher(dimension=1 << 12)
        self.assertFalse(
            np.array_equal(
                np.sort(default_hasher.transform(low_confidence)),
                np.sort(default_hasher.transform(high_confidence)),
            )
        )

        independent_hasher = FeatureHasher(
            dimension=1 << 12,
            include_old_confidence=False,
        )
        self.assertTrue(
            np.array_equal(
                np.sort(independent_hasher.transform(low_confidence)),
                np.sort(independent_hasher.transform(high_confidence)),
            )
        )

    def test_platt_scaler_preserves_probability_order(self) -> None:
        logits = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
        labels = np.asarray([0, 0, 0, 1, 1])
        slope, intercept = fit_platt_scaler(logits, labels)
        probabilities = apply_platt(logits, slope, intercept)
        self.assertGreater(slope, 0)
        self.assertTrue(np.all(np.diff(probabilities) > 0))


if __name__ == "__main__":
    unittest.main()
