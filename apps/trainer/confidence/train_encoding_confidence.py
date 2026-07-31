#!/usr/bin/env python3
"""Train a per-row probability model for final material-code correctness.

The model predicts whether the pipeline's final code agrees with the
human-confirmed project code. Project name is reserved for group splitting.
Difficulty and the legacy confidence can be independently enabled as runtime
features for controlled ablation experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DESCRIPTION_COLUMN = "材料描述(多行)"
PREDICTED_CODE_COLUMN = "excel2_原始总编码"
HUMAN_CODE_COLUMN = "本项目材料代码"
PROJECT_COLUMN = "项目简称"
CATEGORY_COLUMN = "分类"
CURRENT_CONFIDENCE_COLUMN = "excel2_总置信度"
DIFFICULTY_COLUMN = "excel2_分流最终难度（0=困难，1=中等，2=简单）"

FIELD_CODE_COLUMNS = {
    "TYPE": "excel2_TYPE_原始编码",
    "SIZE": "excel2_SIZE_原始编码",
    "THICKNESS": "excel2_THICKNESS_原始编码",
    "PRESSURE": "excel2_PRESSURE_原始编码",
    "MATERIAL": "excel2_MATERIAL_原始编码",
    "STANDARD": "excel2_STANDARD_原始编码",
}

TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|"
    r"[A-Z]+(?:[./_-]?[A-Z0-9]+)*|"
    r"\d+(?:\.\d+)?",
    re.IGNORECASE,
)
SPACE_PATTERN = re.compile(r"\s+")
CODE_SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class Record:
    source: str
    description: str
    predicted_code: str
    human_code: str
    project: str
    category: str
    current_confidence: float | None
    difficulty: int | None
    field_codes: tuple[tuple[str, str], ...]
    is_correct: int

    @property
    def description_key(self) -> str:
        return normalize_text(self.description)


def text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", text(value)).upper()
    return SPACE_PATTERN.sub(" ", normalized).strip()


def normalize_code(value: Any) -> str:
    """Conservative code comparison: normalize case, Unicode and whitespace."""
    normalized = unicodedata.normalize("NFKC", text(value)).upper()
    return CODE_SPACE_PATTERN.sub("", normalized)


def parse_optional_float(value: Any) -> float | None:
    raw = text(value).rstrip("%")
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if parsed > 1.5:
        parsed /= 100.0
    return float(max(0.0, min(1.0, parsed)))


def parse_optional_int(value: Any) -> int | None:
    raw = text(value)
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def load_records(source_specs: Sequence[tuple[str, Path]]) -> list[Record]:
    requested_columns = {
        DESCRIPTION_COLUMN,
        PREDICTED_CODE_COLUMN,
        HUMAN_CODE_COLUMN,
        PROJECT_COLUMN,
        CATEGORY_COLUMN,
        CURRENT_CONFIDENCE_COLUMN,
        DIFFICULTY_COLUMN,
        *FIELD_CODE_COLUMNS.values(),
    }
    records: list[Record] = []

    for source, path in source_specs:
        frame = pd.read_excel(
            path,
            usecols=lambda column: column in requested_columns,
            dtype=object,
        )
        missing = {
            DESCRIPTION_COLUMN,
            PREDICTED_CODE_COLUMN,
            HUMAN_CODE_COLUMN,
            PROJECT_COLUMN,
            CATEGORY_COLUMN,
        } - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 缺少必要列: {sorted(missing)}")

        for row in frame.to_dict(orient="records"):
            description = text(row.get(DESCRIPTION_COLUMN))
            predicted_code = text(row.get(PREDICTED_CODE_COLUMN))
            human_code = text(row.get(HUMAN_CODE_COLUMN))
            project = text(row.get(PROJECT_COLUMN))
            category = text(row.get(CATEGORY_COLUMN)) or source
            if not all((description, predicted_code, human_code, project)):
                continue

            field_codes = tuple(
                (field_name, text(row.get(column)))
                for field_name, column in FIELD_CODE_COLUMNS.items()
            )
            records.append(
                Record(
                    source=source,
                    description=description,
                    predicted_code=predicted_code,
                    human_code=human_code,
                    project=project,
                    category=category,
                    current_confidence=parse_optional_float(
                        row.get(CURRENT_CONFIDENCE_COLUMN)
                    ),
                    difficulty=parse_optional_int(row.get(DIFFICULTY_COLUMN)),
                    field_codes=field_codes,
                    is_correct=int(
                        normalize_code(predicted_code) == normalize_code(human_code)
                    ),
                )
            )
    return records


def _stable_tie_breaker(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_projects(
    records: Sequence[Record],
    seed: int,
    calibration_ratio: float,
    test_ratio: float,
) -> dict[str, set[str]]:
    """Greedily balance rows, errors and categories while keeping projects whole."""
    if calibration_ratio <= 0 or test_ratio <= 0:
        raise ValueError("calibration_ratio 和 test_ratio 必须大于 0")
    if calibration_ratio + test_ratio >= 0.8:
        raise ValueError("校准集与测试集比例之和必须小于 0.8")

    category_names = sorted({record.category for record in records})
    project_stats: dict[str, dict[str, Any]] = {}
    for record in records:
        stats = project_stats.setdefault(
            record.project,
            {
                "rows": 0,
                "errors": 0,
                "categories": Counter(),
                "difficulties": Counter(),
            },
        )
        stats["rows"] += 1
        stats["errors"] += 1 - record.is_correct
        stats["categories"][record.category] += 1
        stats["difficulties"][
            str(record.difficulty if record.difficulty is not None else "缺失")
        ] += 1

    split_names = ("train", "calibration", "test")
    difficulty_names = sorted(
        {
            str(record.difficulty if record.difficulty is not None else "缺失")
            for record in records
        }
    )
    fractions = {
        "train": 1.0 - calibration_ratio - test_ratio,
        "calibration": calibration_ratio,
        "test": test_ratio,
    }
    totals = {
        "rows": len(records),
        "errors": sum(1 - record.is_correct for record in records),
        "categories": Counter(record.category for record in records),
        "difficulties": Counter(
            str(record.difficulty if record.difficulty is not None else "缺失")
            for record in records
        ),
    }
    targets = {
        split: {
            "rows": totals["rows"] * fractions[split],
            "errors": max(1.0, totals["errors"] * fractions[split]),
            "categories": {
                category: max(
                    1.0, totals["categories"][category] * fractions[split]
                )
                for category in category_names
            },
            "difficulties": {
                difficulty: max(
                    1.0, totals["difficulties"][difficulty] * fractions[split]
                )
                for difficulty in difficulty_names
            },
        }
        for split in split_names
    }
    assigned = {
        split: {
            "rows": 0,
            "errors": 0,
            "categories": Counter(),
            "difficulties": Counter(),
            "projects": set(),
        }
        for split in split_names
    }

    projects = sorted(
        project_stats,
        key=lambda project: (
            -project_stats[project]["rows"],
            _stable_tie_breaker(seed, project),
        ),
    )
    for position, project in enumerate(projects):
        stats = project_stats[project]
        remaining = len(projects) - position
        empty_splits = [split for split in split_names if not assigned[split]["projects"]]
        candidates = empty_splits if remaining <= len(empty_splits) else split_names

        def score(split: str) -> float:
            target = targets[split]
            current = assigned[split]
            row_ratio = (current["rows"] + stats["rows"]) / target["rows"]
            error_ratio = (current["errors"] + stats["errors"]) / target["errors"]
            category_penalty = 0.0
            for category in category_names:
                value = (
                    current["categories"][category]
                    + stats["categories"][category]
                )
                category_penalty += (
                    value / target["categories"][category]
                ) ** 2
            difficulty_penalty = 0.0
            for difficulty in difficulty_names:
                value = (
                    current["difficulties"][difficulty]
                    + stats["difficulties"][difficulty]
                )
                difficulty_penalty += (
                    value / target["difficulties"][difficulty]
                ) ** 2
            overfill_penalty = max(0.0, row_ratio - 1.08) ** 2 * 12.0
            return (
                row_ratio**2
                + 1.5 * error_ratio**2
                + 0.25 * category_penalty
                + 0.35 * difficulty_penalty
                + overfill_penalty
            )

        selected = min(candidates, key=lambda split: (score(split), split))
        assigned[selected]["rows"] += stats["rows"]
        assigned[selected]["errors"] += stats["errors"]
        assigned[selected]["categories"].update(stats["categories"])
        assigned[selected]["difficulties"].update(stats["difficulties"])
        assigned[selected]["projects"].add(project)

    return {
        split: set(assigned[split]["projects"])
        for split in split_names
    }


def partition_records(
    records: Sequence[Record],
    split_projects_map: dict[str, set[str]],
    purge_description_overlap: bool,
) -> tuple[dict[str, list[Record]], dict[str, int]]:
    partitions = {
        split: [
            record
            for record in records
            if record.project in split_projects_map[split]
        ]
        for split in ("train", "calibration", "test")
    }
    removed = {"train_overlap_removed": 0, "calibration_overlap_removed": 0}
    if not purge_description_overlap:
        return partitions, removed

    test_descriptions = {record.description_key for record in partitions["test"]}
    calibration_descriptions = {
        record.description_key for record in partitions["calibration"]
    }
    original_calibration_size = len(partitions["calibration"])
    partitions["calibration"] = [
        record
        for record in partitions["calibration"]
        if record.description_key not in test_descriptions
    ]
    removed["calibration_overlap_removed"] = (
        original_calibration_size - len(partitions["calibration"])
    )

    protected_descriptions = test_descriptions | calibration_descriptions
    original_train_size = len(partitions["train"])
    partitions["train"] = [
        record
        for record in partitions["train"]
        if record.description_key not in protected_descriptions
    ]
    removed["train_overlap_removed"] = original_train_size - len(partitions["train"])
    return partitions, removed


class FeatureHasher:
    def __init__(
        self,
        dimension: int = 1 << 19,
        max_description_chars: int = 800,
        include_difficulty: bool = False,
        include_old_confidence: bool = True,
    ):
        if dimension <= 0 or dimension & (dimension - 1):
            raise ValueError("dimension 必须为 2 的幂")
        self.dimension = dimension
        self.mask = dimension - 1
        self.max_description_chars = max_description_chars
        self.include_difficulty = include_difficulty
        self.include_old_confidence = include_old_confidence

    def _index(self, feature: str) -> int:
        return zlib.crc32(feature.encode("utf-8")) & self.mask

    @staticmethod
    def _bucket(value: int, boundaries: Sequence[int]) -> str:
        for boundary in boundaries:
            if value <= boundary:
                return str(boundary)
        return f">{boundaries[-1]}"

    @staticmethod
    def _bounded_text(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        half = limit // 2
        return f"{value[:half]} … {value[-half:]}"

    def _add_token_features(
        self,
        features: set[int],
        namespace: str,
        value: str,
        char_ngram_range: tuple[int, int],
    ) -> None:
        normalized = normalize_text(value)
        if not normalized:
            features.add(self._index(f"{namespace}:<EMPTY>"))
            return
        tokens = TOKEN_PATTERN.findall(normalized)
        for token in tokens[:160]:
            features.add(self._index(f"{namespace}:TOK:{token}"))
            if len(token) >= 4:
                features.add(self._index(f"{namespace}:PRE:{token[:4]}"))
                features.add(self._index(f"{namespace}:SUF:{token[-4:]}"))
            low, high = char_ngram_range
            if len(token) <= 40:
                for width in range(low, high + 1):
                    for start in range(max(0, len(token) - width + 1)):
                        features.add(
                            self._index(
                                f"{namespace}:C{width}:{token[start:start + width]}"
                            )
                        )
        for left, right in zip(tokens[:80], tokens[1:81]):
            features.add(self._index(f"{namespace}:PAIR:{left}|{right}"))

    def transform(self, record: Record) -> np.ndarray:
        features: set[int] = {self._index("BIAS")}
        description = self._bounded_text(
            normalize_text(record.description),
            self.max_description_chars,
        )
        predicted_code = normalize_code(record.predicted_code)

        features.add(self._index(f"CATEGORY:{record.category}"))
        self._add_token_features(features, "DESC", description, (2, 4))
        self._add_token_features(features, "CODE", predicted_code, (2, 5))

        features.add(
            self._index(
                "DESC_LEN:"
                + self._bucket(len(description), (20, 40, 80, 120, 200, 400, 800))
            )
        )
        features.add(
            self._index(
                "CODE_LEN:"
                + self._bucket(len(predicted_code), (5, 10, 20, 30, 50, 80, 120))
            )
        )
        features.add(
            self._index(
                "DIGITS:"
                + self._bucket(
                    sum(character.isdigit() for character in description),
                    (2, 5, 10, 20, 40, 80),
                )
            )
        )
        for marker in ("/", "\\", ";", ",", "X", "×", "+", "-", "(", ")"):
            count = description.count(marker)
            if count:
                features.add(
                    self._index(
                        f"MARK:{marker}:{self._bucket(count, (1, 2, 4, 8, 16))}"
                    )
                )

        missing_fields: list[str] = []
        for field_name, field_code in record.field_codes:
            normalized_field_code = normalize_code(field_code)
            if not normalized_field_code:
                missing_fields.append(field_name)
                features.add(self._index(f"FIELD:{field_name}:<EMPTY>"))
                continue
            features.add(self._index(f"FIELD:{field_name}:VALUE:{normalized_field_code}"))
            self._add_token_features(
                features,
                f"FIELD_{field_name}",
                normalized_field_code,
                (2, 4),
            )
            relation = (
                "IN_FINAL"
                if normalized_field_code in predicted_code
                else "NOT_IN_FINAL"
            )
            features.add(self._index(f"FIELD:{field_name}:{relation}"))
        features.add(
            self._index(
                "MISSING_FIELDS:" + ",".join(missing_fields)
                if missing_fields
                else "MISSING_FIELDS:<NONE>"
            )
        )

        if self.include_old_confidence:
            if record.current_confidence is None:
                features.add(self._index("OLD_CONF:<MISSING>"))
            else:
                bucket = int(record.current_confidence * 100) // 2 * 2
                features.add(self._index(f"OLD_CONF:{bucket:02d}"))
        if self.include_difficulty:
            difficulty = (
                str(record.difficulty)
                if record.difficulty is not None
                else "<MISSING>"
            )
            features.add(self._index(f"DIFFICULTY:{difficulty}"))

        code_fragments = {
            token
            for token in TOKEN_PATTERN.findall(predicted_code)
            if len(token) >= 2
        }
        overlap_count = sum(fragment in description for fragment in code_fragments)
        features.add(
            self._index(
                "CODE_DESC_OVERLAP:"
                + self._bucket(overlap_count, (0, 1, 2, 4, 8, 16, 32))
            )
        )
        return np.fromiter(features, dtype=np.int32)


class FTRLBinaryClassifier:
    def __init__(
        self,
        dimension: int,
        alpha: float = 0.08,
        beta: float = 1.0,
        l1: float = 0.0,
        l2: float = 1.0,
    ):
        self.dimension = dimension
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z = np.zeros(dimension, dtype=np.float32)
        self.n = np.zeros(dimension, dtype=np.float32)

    def _weights_for(self, indices: np.ndarray) -> np.ndarray:
        z_values = self.z[indices]
        n_values = self.n[indices]
        weights = np.zeros_like(z_values)
        mask = np.abs(z_values) > self.l1
        if np.any(mask):
            weights[mask] = -(
                z_values[mask] - np.sign(z_values[mask]) * self.l1
            ) / (
                (self.beta + np.sqrt(n_values[mask])) / self.alpha
                + self.l2
            )
        return weights

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-35.0, min(35.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def predict_indices(self, indices: np.ndarray) -> float:
        return self._sigmoid(float(self._weights_for(indices).sum()))

    def update(self, indices: np.ndarray, label: int) -> float:
        weights = self._weights_for(indices)
        probability = self._sigmoid(float(weights.sum()))
        gradient = probability - float(label)
        old_n = self.n[indices].copy()
        new_n = old_n + gradient * gradient
        sigma = (np.sqrt(new_n) - np.sqrt(old_n)) / self.alpha
        self.z[indices] += gradient - sigma * weights
        self.n[indices] = new_n
        return probability

    def final_weights(self) -> np.ndarray:
        indices = np.arange(self.dimension, dtype=np.int32)
        return self._weights_for(indices).astype(np.float32)


def train_model(
    records: Sequence[Record],
    hasher: FeatureHasher,
    epochs: int,
    seed: int,
) -> FTRLBinaryClassifier:
    model = FTRLBinaryClassifier(hasher.dimension)
    generator = np.random.default_rng(seed)
    order = np.arange(len(records), dtype=np.int32)
    for epoch in range(epochs):
        generator.shuffle(order)
        running_loss = 0.0
        for index in order:
            record = records[int(index)]
            probability = model.update(hasher.transform(record), record.is_correct)
            probability = max(1e-7, min(1.0 - 1e-7, probability))
            running_loss += -(
                record.is_correct * math.log(probability)
                + (1 - record.is_correct) * math.log(1.0 - probability)
            )
        print(
            f"[训练] epoch={epoch + 1}/{epochs} "
            f"logloss={running_loss / max(1, len(records)):.6f}"
        )
    return model


def predict_raw(
    records: Sequence[Record],
    hasher: FeatureHasher,
    weights: np.ndarray,
) -> np.ndarray:
    logits = np.empty(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        logits[index] = float(weights[hasher.transform(record)].sum())
    return logits


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_platt_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
    ridge: float = 1e-3,
) -> tuple[float, float]:
    """Fit sigmoid(a * logit + b) with a positive slope."""
    if len(np.unique(labels)) < 2:
        return 1.0, 0.0
    logit_mean = float(np.mean(logits))
    logit_scale = max(1e-6, float(np.std(logits)))
    standardized_logits = np.clip(
        (logits - logit_mean) / logit_scale,
        -12.0,
        12.0,
    )
    positive_rate = float(np.clip(labels.mean(), 1e-5, 1.0 - 1e-5))
    parameters = np.array(
        [1.0, math.log(positive_rate / (1.0 - positive_rate))],
        dtype=np.float64,
    )
    first_moment = np.zeros(2, dtype=np.float64)
    second_moment = np.zeros(2, dtype=np.float64)
    learning_rate = 0.03
    beta1 = 0.9
    beta2 = 0.999
    for step_index in range(1, 2001):
        linear = parameters[0] * standardized_logits + parameters[1]
        probabilities = sigmoid_array(linear)
        residual = probabilities - labels
        gradient = np.array(
            [
                float(np.mean(residual * standardized_logits))
                + ridge * parameters[0],
                float(np.mean(residual)),
            ],
            dtype=np.float64,
        )
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        corrected_first = first_moment / (1.0 - beta1**step_index)
        corrected_second = second_moment / (1.0 - beta2**step_index)
        update = (
            learning_rate
            * corrected_first
            / (np.sqrt(corrected_second) + 1e-8)
        )
        parameters -= update
        parameters[0] = float(np.clip(parameters[0], 0.01, 20.0))
        parameters[1] = float(np.clip(parameters[1], -20.0, 20.0))
        if float(np.max(np.abs(update))) < 1e-8:
            break
    slope = float(parameters[0] / logit_scale)
    intercept = float(
        parameters[1] - parameters[0] * logit_mean / logit_scale
    )
    return slope, intercept


def apply_platt(logits: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return sigmoid_array(slope * logits + intercept)


def fit_grouped_platt_scalers(
    records: Sequence[Record],
    logits: np.ndarray,
    labels: np.ndarray,
    use_difficulty: bool,
    minimum_rows: int = 500,
    minimum_class_rows: int = 20,
) -> dict[str, tuple[float, float]]:
    scalers = {"global": fit_platt_scaler(logits, labels)}
    if not use_difficulty:
        return scalers
    for difficulty in sorted(
        {
            record.difficulty
            for record in records
            if record.difficulty is not None
        }
    ):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.difficulty == difficulty
            ],
            dtype=np.int32,
        )
        if len(indices) < minimum_rows:
            continue
        subset_labels = labels[indices]
        positives = int(subset_labels.sum())
        negatives = len(subset_labels) - positives
        if min(positives, negatives) < minimum_class_rows:
            continue
        scalers[f"difficulty:{difficulty}"] = fit_platt_scaler(
            logits[indices],
            subset_labels,
        )
    return scalers


def apply_grouped_platt_scalers(
    records: Sequence[Record],
    logits: np.ndarray,
    scalers: dict[str, tuple[float, float]],
) -> np.ndarray:
    global_slope, global_intercept = scalers["global"]
    probabilities = apply_platt(logits, global_slope, global_intercept)
    for difficulty in sorted(
        {
            record.difficulty
            for record in records
            if record.difficulty is not None
        }
    ):
        key = f"difficulty:{difficulty}"
        if key not in scalers:
            continue
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record.difficulty == difficulty
            ],
            dtype=np.int32,
        )
        slope, intercept = scalers[key]
        probabilities[indices] = apply_platt(
            logits[indices],
            slope,
            intercept,
        )
    return probabilities


def wilson_lower_bound(successes: int, total: int, z_value: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    probability = successes / total
    denominator = 1.0 + z_value * z_value / total
    centre = probability + z_value * z_value / (2.0 * total)
    margin = z_value * math.sqrt(
        probability * (1.0 - probability) / total
        + z_value * z_value / (4.0 * total * total)
    )
    return (centre - margin) / denominator


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = probabilities >= lower
        mask &= probabilities <= upper if index == bin_count - 1 else probabilities < upper
        if not np.any(mask):
            continue
        count = int(mask.sum())
        correct = int(labels[mask].sum())
        rows.append(
            {
                "lower": round(float(lower), 4),
                "upper": round(float(upper), 4),
                "rows": count,
                "mean_confidence": round(float(probabilities[mask].mean()), 6),
                "actual_accuracy": round(correct / count, 6),
                "wilson_lower_95": round(
                    wilson_lower_bound(correct, count),
                    6,
                ),
            }
        )
    return rows


def calibration_quantiles(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int = 10,
) -> list[dict[str, Any]]:
    order = np.argsort(probabilities)
    rows: list[dict[str, Any]] = []
    for indices in np.array_split(order, bin_count):
        if len(indices) == 0:
            continue
        count = len(indices)
        correct = int(labels[indices].sum())
        rows.append(
            {
                "min_confidence": round(float(probabilities[indices].min()), 6),
                "max_confidence": round(float(probabilities[indices].max()), 6),
                "rows": count,
                "mean_confidence": round(
                    float(probabilities[indices].mean()),
                    6,
                ),
                "actual_accuracy": round(correct / count, 6),
                "wilson_lower_95": round(
                    wilson_lower_bound(correct, count),
                    6,
                ),
            }
        )
    return rows


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((clipped - labels) ** 2))
    logloss = float(
        -np.mean(
            labels * np.log(clipped)
            + (1.0 - labels) * np.log(1.0 - clipped)
        )
    )
    bins = calibration_bins(labels, clipped)
    ece = sum(
        row["rows"]
        / len(labels)
        * abs(row["mean_confidence"] - row["actual_accuracy"])
        for row in bins
    )
    error_labels = 1 - labels
    error_scores = 1.0 - clipped
    error_auc = roc_auc(error_labels, error_scores)
    return {
        "rows": int(len(labels)),
        "correct": int(labels.sum()),
        "errors": int((1 - labels).sum()),
        "accuracy": round(float(labels.mean()), 6),
        "mean_confidence": round(float(clipped.mean()), 6),
        "brier_score": round(brier, 6),
        "log_loss": round(logloss, 6),
        "ece_10_bins": round(float(ece), 6),
        "error_detection_roc_auc": (
            round(float(error_auc), 6) if error_auc is not None else None
        ),
        "calibration_bins": bins,
        "calibration_equal_count_bins": calibration_quantiles(labels, clipped),
    }


def threshold_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in (0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.98, 0.99):
        mask = probabilities >= threshold
        count = int(mask.sum())
        correct = int(labels[mask].sum()) if count else 0
        rows.append(
            {
                "threshold": threshold,
                "rows": count,
                "coverage": round(count / len(labels), 6),
                "accuracy": round(correct / count, 6) if count else None,
                "wrong_rows": count - correct,
                "wilson_lower_95": (
                    round(wilson_lower_bound(correct, count), 6)
                    if count
                    else None
                ),
            }
        )
    return rows


def slice_metrics(
    records: Sequence[Record],
    probabilities: np.ndarray,
    attribute: str,
) -> list[dict[str, Any]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        value = getattr(record, attribute)
        grouped_indices[str(value if value is not None else "缺失")].append(index)
    result: list[dict[str, Any]] = []
    for value, indices in sorted(grouped_indices.items()):
        index_array = np.asarray(indices, dtype=np.int32)
        labels = np.asarray(
            [records[index].is_correct for index in indices],
            dtype=np.int8,
        )
        subset_probabilities = probabilities[index_array]
        result.append(
            {
                "value": value,
                **{
                    key: metric_value
                    for key, metric_value in metrics(
                        labels,
                        subset_probabilities,
                    ).items()
                    if key not in {"calibration_bins", "calibration_equal_count_bins"}
                },
            }
        )
    return result


def old_confidence_baseline(records: Sequence[Record]) -> dict[str, Any]:
    selected = [
        record for record in records if record.current_confidence is not None
    ]
    if not selected:
        return {"rows": 0}
    labels = np.asarray(
        [record.is_correct for record in selected],
        dtype=np.int8,
    )
    probabilities = np.asarray(
        [record.current_confidence for record in selected],
        dtype=np.float64,
    )
    return metrics(labels, probabilities)


def risk_controlled_thresholds(
    calibration_labels: np.ndarray,
    calibration_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    minimum_rows: int = 200,
) -> list[dict[str, Any]]:
    candidates = np.unique(
        np.quantile(
            calibration_probabilities,
            np.linspace(0.0, 0.999, 1000),
        )
    )
    rows: list[dict[str, Any]] = []
    for target_accuracy in (0.95, 0.97, 0.98, 0.99, 0.995):
        selected_threshold: float | None = None
        calibration_snapshot: dict[str, Any] | None = None
        for threshold in candidates:
            mask = calibration_probabilities >= threshold
            count = int(mask.sum())
            if count < minimum_rows:
                continue
            correct = int(calibration_labels[mask].sum())
            lower_bound = wilson_lower_bound(correct, count)
            if lower_bound >= target_accuracy:
                selected_threshold = float(threshold)
                calibration_snapshot = {
                    "rows": count,
                    "accuracy": correct / count,
                    "wilson_lower_95": lower_bound,
                }
                break
        if selected_threshold is None or calibration_snapshot is None:
            rows.append(
                {
                    "target_accuracy": target_accuracy,
                    "status": "校准集无法在最低样本量约束下达到目标",
                }
            )
            continue

        test_mask = test_probabilities >= selected_threshold
        test_count = int(test_mask.sum())
        test_correct = int(test_labels[test_mask].sum()) if test_count else 0
        rows.append(
            {
                "target_accuracy": target_accuracy,
                "status": "可评估",
                "threshold": round(selected_threshold, 6),
                "calibration_rows": calibration_snapshot["rows"],
                "calibration_accuracy": round(
                    calibration_snapshot["accuracy"],
                    6,
                ),
                "calibration_wilson_lower_95": round(
                    calibration_snapshot["wilson_lower_95"],
                    6,
                ),
                "test_rows": test_count,
                "test_coverage": round(test_count / len(test_labels), 6),
                "test_accuracy": (
                    round(test_correct / test_count, 6)
                    if test_count
                    else None
                ),
                "test_wrong_rows": test_count - test_correct,
                "test_wilson_lower_95": (
                    round(
                        wilson_lower_bound(test_correct, test_count),
                        6,
                    )
                    if test_count
                    else None
                ),
            }
        )
    return rows


def duplicate_audit(records: Sequence[Record]) -> dict[str, Any]:
    truths_by_description: dict[str, set[str]] = defaultdict(set)
    projects_by_description: dict[str, set[str]] = defaultdict(set)
    for record in records:
        truths_by_description[record.description_key].add(
            normalize_code(record.human_code)
        )
        projects_by_description[record.description_key].add(record.project)
    return {
        "unique_descriptions": len(truths_by_description),
        "descriptions_seen_in_multiple_projects": sum(
            len(projects) > 1 for projects in projects_by_description.values()
        ),
        "descriptions_with_conflicting_human_codes": sum(
            len(truths) > 1 for truths in truths_by_description.values()
        ),
    }


def records_summary(records: Sequence[Record]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "projects": len({record.project for record in records}),
        "correct": sum(record.is_correct for record in records),
        "errors": sum(1 - record.is_correct for record in records),
        "accuracy": round(
            sum(record.is_correct for record in records) / max(1, len(records)),
            6,
        ),
        "category_counts": dict(Counter(record.category for record in records)),
        "difficulty_counts": dict(
            Counter(
                str(record.difficulty if record.difficulty is not None else "缺失")
                for record in records
            )
        ),
    }


def save_predictions(
    path: Path,
    records: Sequence[Record],
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "项目名称": [record.project for record in records],
            "分类": [record.category for record in records],
            "原始描述": [record.description for record in records],
            "模型最终编码": [record.predicted_code for record in records],
            "人工确认编码": [record.human_code for record in records],
            "是否正确": [record.is_correct for record in records],
            "预测正确概率": probabilities,
            "原难度_仅评估": [record.difficulty for record in records],
            "原置信度_仅对照": [
                record.current_confidence for record in records
            ],
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def parse_source_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--source 必须使用 分类=/path/file.xlsx")
    category, raw_path = raw.split("=", 1)
    category = category.strip()
    path = Path(raw_path).expanduser()
    if not category or not raw_path.strip():
        raise argparse.ArgumentTypeError("--source 的分类和路径都不能为空")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"文件不存在: {path}")
    return category, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练按项目隔离验证的单条编码正确概率模型"
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source_spec,
        required=True,
        help="可重复指定，格式：分类=/path/file.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hash-bits", type=int, default=19)
    parser.add_argument(
        "--include-difficulty-feature",
        action="store_true",
        help="将人工确认前生成的难度等级作为输入特征；默认仅用于评估切片",
    )
    parser.add_argument(
        "--exclude-old-confidence-feature",
        action="store_true",
        help="不把旧版总置信度作为输入特征，用于训练独立的新置信度模型",
    )
    parser.add_argument(
        "--no-purge-description-overlap",
        action="store_true",
        help="不清除跨集合完全相同的描述；默认清除以获得更严格评估",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[数据] 读取 Excel")
    records = load_records(args.source)
    if not records:
        raise RuntimeError("没有可用于训练的完整样本")
    print(
        f"[数据] rows={len(records)} projects="
        f"{len({record.project for record in records})}"
    )

    project_splits = split_projects(
        records,
        seed=args.seed,
        calibration_ratio=args.calibration_ratio,
        test_ratio=args.test_ratio,
    )
    partitions, removed_overlap = partition_records(
        records,
        project_splits,
        purge_description_overlap=not args.no_purge_description_overlap,
    )
    for split, split_records in partitions.items():
        if not split_records:
            raise RuntimeError(f"{split} 划分后为空")
        print(f"[划分] {split}: {records_summary(split_records)}")
    print(f"[划分] 跨集合重复描述清理: {removed_overlap}")

    hasher = FeatureHasher(
        dimension=1 << args.hash_bits,
        include_difficulty=args.include_difficulty_feature,
        include_old_confidence=not args.exclude_old_confidence_feature,
    )
    model = train_model(
        partitions["train"],
        hasher,
        epochs=args.epochs,
        seed=args.seed,
    )
    weights = model.final_weights()

    calibration_logits = predict_raw(
        partitions["calibration"],
        hasher,
        weights,
    )
    calibration_labels = np.asarray(
        [record.is_correct for record in partitions["calibration"]],
        dtype=np.float64,
    )
    platt_scalers = fit_grouped_platt_scalers(
        partitions["calibration"],
        calibration_logits,
        calibration_labels,
        use_difficulty=args.include_difficulty_feature,
    )
    calibration_probabilities = apply_grouped_platt_scalers(
        partitions["calibration"],
        calibration_logits,
        platt_scalers,
    )

    test_logits = predict_raw(partitions["test"], hasher, weights)
    test_probabilities = apply_grouped_platt_scalers(
        partitions["test"],
        test_logits,
        platt_scalers,
    )
    test_labels = np.asarray(
        [record.is_correct for record in partitions["test"]],
        dtype=np.int8,
    )

    report = {
        "purpose": "预测单条模型最终编码与人工确认编码一致的概率",
        "label_definition": (
            "模型最终编码与人工确认编码经 Unicode、大小写、空白归一后完全一致"
        ),
        "excluded_features": [
            "本项目材料代码",
            "项目简称",
            "全部 C1-* 字段",
        ]
        + (
            ["excel2_总置信度"]
            if args.exclude_old_confidence_feature
            else []
        )
        + (
            []
            if args.include_difficulty_feature
            else ["excel2_分流最终难度"]
        ),
        "parameters": {
            "seed": args.seed,
            "calibration_ratio": args.calibration_ratio,
            "test_ratio": args.test_ratio,
            "epochs": args.epochs,
            "hash_dimension": hasher.dimension,
            "include_difficulty_feature": args.include_difficulty_feature,
            "include_old_confidence_feature": hasher.include_old_confidence,
            "purge_description_overlap": not args.no_purge_description_overlap,
        },
        "source_files": [
            {"category": category, "path": str(path)}
            for category, path in args.source
        ],
        "all_data": {
            **records_summary(records),
            **duplicate_audit(records),
        },
        "split": {
            split: {
                **records_summary(split_records),
                "project_names": sorted(project_splits[split]),
            }
            for split, split_records in partitions.items()
        },
        "overlap_purge": removed_overlap,
        "calibration": {
            "method": (
                "Platt scaling on held-out projects; difficulty-specific "
                "scalers when difficulty is enabled and sufficiently sampled"
            ),
            "scalers": {
                key: {"slope": value[0], "intercept": value[1]}
                for key, value in platt_scalers.items()
            },
            "metrics": metrics(
                calibration_labels.astype(np.int8),
                calibration_probabilities,
            ),
        },
        "test": {
            "metrics": metrics(test_labels, test_probabilities),
            "thresholds": threshold_table(test_labels, test_probabilities),
            "risk_controlled_thresholds_selected_on_calibration": (
                risk_controlled_thresholds(
                    calibration_labels.astype(np.int8),
                    calibration_probabilities,
                    test_labels,
                    test_probabilities,
                )
            ),
            "old_confidence_baseline_on_available_rows": old_confidence_baseline(
                partitions["test"]
            ),
            "by_category": slice_metrics(
                partitions["test"],
                test_probabilities,
                "category",
            ),
            "by_difficulty_diagnostic_only": slice_metrics(
                partitions["test"],
                test_probabilities,
                "difficulty",
            ),
            "by_project": slice_metrics(
                partitions["test"],
                test_probabilities,
                "project",
            ),
        },
    }

    model_path = output_dir / "encoding_confidence_model.npz"
    np.savez_compressed(
        model_path,
        weights=weights,
        hash_dimension=np.asarray([hasher.dimension], dtype=np.int64),
        max_description_chars=np.asarray(
            [hasher.max_description_chars],
            dtype=np.int64,
        ),
        include_difficulty=np.asarray(
            [int(hasher.include_difficulty)],
            dtype=np.int8,
        ),
        include_old_confidence=np.asarray(
            [int(hasher.include_old_confidence)],
            dtype=np.int8,
        ),
        platt_keys=np.asarray(list(platt_scalers), dtype=np.str_),
        platt_slopes=np.asarray(
            [platt_scalers[key][0] for key in platt_scalers],
            dtype=np.float64,
        ),
        platt_intercepts=np.asarray(
            [platt_scalers[key][1] for key in platt_scalers],
            dtype=np.float64,
        ),
    )
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions_path = output_dir / "test_predictions.csv"
    save_predictions(
        predictions_path,
        partitions["test"],
        test_probabilities,
    )
    print(f"[完成] 模型: {model_path}")
    print(f"[完成] 报告: {report_path}")
    print(f"[完成] 测试明细: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
