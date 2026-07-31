"""Runtime predictor for final encoding correctness probability."""

from __future__ import annotations

import math
import re
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FIELD_NAMES = ("TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD")
TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|"
    r"[A-Z]+(?:[./_-]?[A-Z0-9]+)*|"
    r"\d+(?:\.\d+)?",
    re.IGNORECASE,
)
SPACE_PATTERN = re.compile(r"\s+")
CODE_SPACE_PATTERN = re.compile(r"\s+")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value)).upper()
    return SPACE_PATTERN.sub(" ", normalized).strip()


def normalize_code(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value)).upper()
    return CODE_SPACE_PATTERN.sub("", normalized)


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def extract_difficulty(result: Mapping[str, Any]) -> int | None:
    """Read the final routing difficulty used by the training data."""
    candidates = (
        (result.get("routing") or {}).get("final_level"),
        (result.get("second_pass") or {}).get("final_level"),
        (result.get("difficulty_split") or {}).get("level"),
        (result.get("difficulty_split") or {}).get("difficulty"),
    )
    for candidate in candidates:
        parsed = _safe_int(candidate)
        if parsed is not None:
            return parsed
    return None


def extract_field_codes(result: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    fields = result.get("fields")
    fields = fields if isinstance(fields, Mapping) else {}
    extracted: list[tuple[str, str]] = []
    for field_name in FIELD_NAMES:
        field_payload = fields.get(field_name)
        field_payload = field_payload if isinstance(field_payload, Mapping) else {}
        stage2_output = field_payload.get("stage2_output")
        stage2_output = stage2_output if isinstance(stage2_output, Mapping) else {}
        extracted.append((field_name, _text(stage2_output.get("code"))))
    return tuple(extracted)


class EncodingCorrectnessPredictor:
    """CPU-only hashed linear model with Platt probability calibration."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        with np.load(self.model_path, allow_pickle=False) as data:
            self.weights = np.asarray(data["weights"], dtype=np.float32)
            self.dimension = int(data["hash_dimension"][0])
            self.max_description_chars = int(data["max_description_chars"][0])
            self.include_difficulty = bool(data["include_difficulty"][0])
            self.include_old_confidence = bool(data["include_old_confidence"][0])
            keys = [str(value) for value in data["platt_keys"].tolist()]
            slopes = np.asarray(data["platt_slopes"], dtype=np.float64)
            intercepts = np.asarray(data["platt_intercepts"], dtype=np.float64)

        if self.dimension <= 0 or self.dimension & (self.dimension - 1):
            raise ValueError("置信度模型 hash_dimension 必须为 2 的幂")
        if self.weights.shape != (self.dimension,):
            raise ValueError(
                f"置信度模型权重维度不匹配: {self.weights.shape} != {(self.dimension,)}"
            )
        if self.include_old_confidence:
            raise ValueError("平台仅接入不依赖旧置信度的模型")
        if not (len(keys) == len(slopes) == len(intercepts)):
            raise ValueError("置信度模型校准参数长度不一致")
        self.mask = self.dimension - 1
        self.platt_scalers = {
            key: (float(slope), float(intercept))
            for key, slope, intercept in zip(keys, slopes, intercepts)
        }
        if "global" not in self.platt_scalers:
            raise ValueError("置信度模型缺少 global 校准参数")

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

    def feature_indices(
        self,
        *,
        description: str,
        predicted_code: str,
        category: str,
        field_codes: Sequence[tuple[str, str]],
        difficulty: int | None,
    ) -> np.ndarray:
        """Build exactly the same feature set used by the training script."""
        features: set[int] = {self._index("BIAS")}
        description = self._bounded_text(
            normalize_text(description),
            self.max_description_chars,
        )
        predicted_code = normalize_code(predicted_code)

        features.add(self._index(f"CATEGORY:{category}"))
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
        for field_name, field_code in field_codes:
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
            relation = "IN_FINAL" if normalized_field_code in predicted_code else "NOT_IN_FINAL"
            features.add(self._index(f"FIELD:{field_name}:{relation}"))
        features.add(
            self._index(
                "MISSING_FIELDS:" + ",".join(missing_fields)
                if missing_fields
                else "MISSING_FIELDS:<NONE>"
            )
        )

        if self.include_difficulty:
            difficulty_value = str(difficulty) if difficulty is not None else "<MISSING>"
            features.add(self._index(f"DIFFICULTY:{difficulty_value}"))

        code_fragments = {
            token for token in TOKEN_PATTERN.findall(predicted_code) if len(token) >= 2
        }
        overlap_count = sum(fragment in description for fragment in code_fragments)
        features.add(
            self._index(
                "CODE_DESC_OVERLAP:"
                + self._bucket(overlap_count, (0, 1, 2, 4, 8, 16, 32))
            )
        )
        return np.fromiter(features, dtype=np.int32)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
        exp_value = math.exp(max(value, -60.0))
        return exp_value / (1.0 + exp_value)

    def predict(
        self,
        *,
        description: str,
        predicted_code: str,
        category: str,
        field_codes: Sequence[tuple[str, str]],
        difficulty: int | None,
    ) -> float:
        indices = self.feature_indices(
            description=description,
            predicted_code=predicted_code,
            category=category,
            field_codes=field_codes,
            difficulty=difficulty,
        )
        raw_logit = float(self.weights[indices].sum(dtype=np.float64))
        scaler_key = f"difficulty:{difficulty}"
        slope, intercept = self.platt_scalers.get(
            scaler_key,
            self.platt_scalers["global"],
        )
        return self._sigmoid(slope * raw_logit + intercept)

    def predict_result(self, result: Mapping[str, Any]) -> tuple[float, int | None]:
        difficulty = extract_difficulty(result)
        score = self.predict(
            description=_text(result.get("original_text")),
            predicted_code=_text(result.get("final_code")),
            category=_text(
                result.get("material_category")
                or result.get("imported_category")
                or result.get("model_material_category")
            ),
            field_codes=extract_field_codes(result),
            difficulty=difficulty,
        )
        return score, difficulty
