# -*- coding: utf-8 -*-
"""Detect special tokens that indicate inherently difficult descriptions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import DifficultyFeature, GlueHit


class SpecialTokenDetector:
    """Detect configured special complexity tokens in raw description text."""

    CONFIG_PATH = Path(__file__).parent / "config" / "special_difficulty_tokens.yaml"

    def __init__(self) -> None:
        self.groups = self._load_groups()

    def _load_groups(self) -> list[dict[str, Any]]:
        if not self.CONFIG_PATH.exists():
            return []
        raw = yaml.safe_load(self.CONFIG_PATH.read_text(encoding="utf-8")) or {}
        groups = raw.get("groups") or {}
        ordered: list[dict[str, Any]] = []
        for group_name, payload in groups.items():
            if not isinstance(payload, dict):
                continue
            reason = str(payload.get("reason") or "").strip()
            tokens = payload.get("tokens") or []
            normalized_tokens: list[dict[str, str]] = []
            for item in tokens:
                if isinstance(item, dict):
                    token = str(item.get("token") or "").strip()
                else:
                    token = str(item or "").strip()
                if not token:
                    continue
                normalized_tokens.append(
                    {
                        "token": token,
                        "match_token": token.upper(),
                    }
                )
            if normalized_tokens:
                ordered.append(
                    {
                        "name": str(group_name),
                        "reason": reason,
                        "tokens": normalized_tokens,
                    }
                )
        return ordered

    def analyze(self, text: str) -> DifficultyFeature:
        raw_text = str(text or "")
        if not raw_text or not self.groups:
            return DifficultyFeature(name="special_token", matched=False)

        upper_text = raw_text.upper()
        hits: list[GlueHit] = []
        reasons: list[str] = []
        seen_spans: set[tuple[int, int, str]] = set()

        for group in self.groups:
            group_name = group["name"]
            group_reason = group["reason"]
            matched_group = False
            for token_info in group["tokens"]:
                token = token_info["token"]
                match_token = token_info["match_token"]
                start = 0
                while True:
                    idx = upper_text.find(match_token, start)
                    if idx < 0:
                        break
                    end = idx + len(match_token)
                    key = (idx, end, token)
                    if key not in seen_spans:
                        seen_spans.add(key)
                        hits.append(
                            GlueHit(
                                tag="special_token",
                                code_group=group_name,
                                code=token,
                                token=raw_text[idx:end],
                                start=idx,
                                end=end,
                                note=f"{group_reason}: {token}",
                            )
                        )
                        matched_group = True
                    start = idx + len(match_token)
            if matched_group and group_reason and group_reason not in reasons:
                reasons.append(group_reason)

        return DifficultyFeature(
            name="special_token",
            matched=bool(hits),
            reason=" | ".join(reasons),
            hits=hits,
        )
