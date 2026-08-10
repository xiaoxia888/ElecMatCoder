# -*- coding: utf-8 -*-
"""Utilities for deriving field confidence from generated-token log probabilities."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _clamp_probability(value: float) -> float:
    return max(1e-12, min(1.0, float(value)))


def normalize_token_logprobs(
    records: Any,
    *,
    raw_text: str = "",
) -> List[Dict[str, Any]]:
    """Normalize service-specific token records to token/logprob/start/end."""
    if not isinstance(records, list):
        return []

    normalized: List[Dict[str, Any]] = []
    cursor = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        token = item.get("text", item.get("token", ""))
        token = str(token or "")
        logprob = item.get("logprob")
        if logprob is None:
            probability = item.get("probability", item.get("prob"))
            if probability is not None:
                try:
                    logprob = math.log(_clamp_probability(float(probability)))
                except (TypeError, ValueError):
                    logprob = None
        try:
            logprob = float(logprob)
        except (TypeError, ValueError):
            continue

        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            start = cursor
            end = start + len(token)

        if end < start:
            continue
        if raw_text and token and 0 <= start <= end <= len(raw_text):
            token = raw_text[start:end]
        normalized.append(
            {
                "token": token,
                "logprob": logprob,
                "start": start,
                "end": end,
            }
        )
        cursor = max(cursor, end)
    return normalized


def score_token_logprobs(
    records: Any,
    *,
    raw_text: str = "",
    spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the length-normalized probability of selected semantic tokens."""
    tokens = normalize_token_logprobs(records, raw_text=raw_text)
    selected: List[Dict[str, Any]] = []
    for item in tokens:
        token_text = str(item.get("token") or "")
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in token_text):
            continue
        if spans and not any(item["end"] > start and item["start"] < end for start, end in spans):
            continue
        selected.append(item)

    if not selected:
        return None

    logprobs = [float(item["logprob"]) for item in selected]
    sum_logprob = sum(logprobs)
    mean_logprob = sum(logprobs) / len(logprobs)
    probability = _clamp_probability(math.exp(mean_logprob))
    return {
        "confidence": probability,
        "token_count": len(selected),
        "sum_logprob": sum_logprob,
        "joint_probability": _clamp_probability(math.exp(sum_logprob)),
        "mean_logprob": mean_logprob,
        "mean_probability": probability,
        "min_probability": min(_clamp_probability(math.exp(lp)) for lp in logprobs),
    }


def _combine_token_scores(
    components: Sequence[Tuple[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Combine sequential model decisions with one length-normalized score."""
    valid = [
        (str(name), score)
        for name, score in components
        if isinstance(score, dict) and int(score.get("token_count") or 0) > 0
    ]
    if not valid:
        return None

    token_count = sum(int(score["token_count"]) for _, score in valid)
    sum_logprob = sum(float(score["sum_logprob"]) for _, score in valid)
    mean_logprob = sum_logprob / token_count
    probability = _clamp_probability(math.exp(mean_logprob))
    return {
        "confidence": probability,
        "token_count": token_count,
        "sum_logprob": sum_logprob,
        "joint_probability": _clamp_probability(math.exp(sum_logprob)),
        "mean_logprob": mean_logprob,
        "mean_probability": probability,
        "min_probability": min(float(score["min_probability"]) for _, score in valid),
        "component_fields": [name for name, _ in valid],
        "component_confidences": {
            name: float(score["confidence"])
            for name, score in valid
        },
    }


def _find_json_value_span(raw: str, key: str) -> Optional[Tuple[int, int]]:
    key_pattern = re.compile(rf'"{re.escape(str(key))}"\s*:', re.IGNORECASE)
    match = key_pattern.search(raw)
    if not match:
        return None

    start = match.end()
    while start < len(raw) and raw[start].isspace():
        start += 1
    if start >= len(raw):
        return None

    opener = raw[start]
    if opener == '"':
        escaped = False
        pos = start + 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == '"' and not escaped:
                return start, pos + 1
            escaped = (ch == '\\' and not escaped)
            if ch != '\\':
                escaped = False
            pos += 1
        return None

    if opener in "[{":
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escaped = False
        for pos in range(start, len(raw)):
            ch = raw[pos]
            if in_string:
                if ch == '"' and not escaped:
                    in_string = False
                escaped = (ch == '\\' and not escaped)
                if ch != '\\':
                    escaped = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return start, pos + 1
        return None

    end = start
    while end < len(raw) and raw[end] not in ",}\n\r":
        end += 1
    return start, end


def _iter_scalar_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_scalar_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_scalar_values(child)
    elif value not in (None, "", False):
        yield value


def _find_scalar_spans(raw: str, field_span: Tuple[int, int], value: Any) -> List[Tuple[int, int]]:
    field_start, field_end = field_span
    segment = raw[field_start:field_end]
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for scalar in _iter_scalar_values(value):
        encoded = json.dumps(scalar, ensure_ascii=False)
        pos = segment.find(encoded, cursor)
        if pos < 0:
            pos = segment.find(encoded)
        if pos < 0:
            continue
        scalar_start = field_start + pos
        scalar_end = scalar_start + len(encoded)
        if isinstance(scalar, str) and len(encoded) >= 2:
            scalar_start += 1
            scalar_end -= 1
        if scalar_end > scalar_start:
            spans.append((scalar_start, scalar_end))
        cursor = pos + len(encoded)
    return spans


def build_field_token_confidences(
    raw_text: str,
    structured: Dict[str, Any],
    token_logprobs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Score only JSON scalar values belonging to each top-level output field."""
    if not isinstance(structured, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for field, value in structured.items():
        if str(field).startswith("_") or value in (None, "", [], {}):
            continue
        field_span = _find_json_value_span(raw_text, str(field))
        if field_span is None:
            continue
        scalar_spans = _find_scalar_spans(raw_text, field_span, value)
        scored = score_token_logprobs(
            token_logprobs,
            raw_text=raw_text,
            spans=scalar_spans or [field_span],
        )
        if scored is not None:
            result[str(field)] = scored

    # TYPE is not independent of CATEGORY: score their semantic tokens together
    # so the category decision participates without introducing length bias.
    category_score = result.get("CATEGORY")
    type_score = result.get("TYPE")
    if category_score is not None and type_score is not None:
        combined = _combine_token_scores((
            ("CATEGORY", category_score),
            ("TYPE", type_score),
        ))
        if combined is not None:
            result["TYPE"] = combined
    return result


__all__ = [
    "build_field_token_confidences",
    "normalize_token_logprobs",
    "score_token_logprobs",
]
