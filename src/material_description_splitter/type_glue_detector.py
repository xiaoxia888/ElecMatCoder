# -*- coding: utf-8 -*-
"""Detector for glued type-related codes."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import yaml

from .models import DifficultyFeature, GlueHit


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class TypeGlueDetector:
    """Detect whether type-related short codes are glued with other letters/digits."""

    TAG_NAME = "type_glue"

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.code_groups = self._load_code_groups(config_path)
        self.all_codes = {code for codes in self.code_groups.values() for code in codes}

    def analyze(self, text: str) -> DifficultyFeature:
        hits: list[GlueHit] = []
        for token_match in TOKEN_RE.finditer(text):
            token = token_match.group(0)
            normalized = token.upper()
            token_hits = self._find_token_hits(
                text=text,
                token=token,
                normalized=normalized,
                token_start=token_match.start(),
            )
            hits.extend(token_hits)

        reason = ""
        if hits:
            reason = "种类编码与其他字母或数字发生粘连"

        return DifficultyFeature(
            name=self.TAG_NAME,
            matched=bool(hits),
            reason=reason,
            hits=hits,
        )

    def _find_token_hits(self, text: str, token: str, normalized: str, token_start: int) -> list[GlueHit]:
        if normalized in self.all_codes:
            return []

        hits: list[GlueHit] = []

        for group_name, codes in self.code_groups.items():
            best = self._match_one_group(
                text=text,
                token=token,
                normalized=normalized,
                token_start=token_start,
                group_name=group_name,
                codes=codes,
            )
            if best is not None:
                hits.append(best)

        return hits

    def _match_one_group(
        self,
        text: str,
        token: str,
        normalized: str,
        token_start: int,
        group_name: str,
        codes: Iterable[str],
    ) -> GlueHit | None:
        for code in sorted(codes, key=len, reverse=True):
            if normalized == code:
                continue

            idx = normalized.find(code)
            if idx < 0:
                continue

            prefix = normalized[:idx]
            suffix = normalized[idx + len(code) :]
            if not prefix and not suffix:
                continue

            if self._is_unit_context(normalized=normalized, code=code, idx=idx):
                continue

            if self._is_material_grade_context(
                text=text,
                token=token,
                normalized=normalized,
                token_start=token_start,
                code=code,
                idx=idx,
            ):
                continue

            if self._covered_by_longer_code(normalized=normalized, code=code, idx=idx):
                continue

            if not self._looks_glued(code=code, prefix=prefix, suffix=suffix):
                continue

            return GlueHit(
                tag=self.TAG_NAME,
                code_group=group_name,
                code=code,
                token=token,
                start=token_start + idx,
                end=token_start + idx + len(code),
                note=self._build_note(code=code, token=token),
            )

        return None

    @staticmethod
    def _is_unit_context(normalized: str, code: str, idx: int) -> bool:
        """
        Suppress obvious unit patterns such as:
        - 0mm
        - 2.5mm (tokenized as 5mm)
        We only special-case single-letter M here, because it is the main
        false-positive source in thickness units.
        """
        if code != "M":
            return False

        if idx + 1 < len(normalized) and normalized[idx : idx + 2] == "MM":
            return True
        if idx > 0 and normalized[idx - 1 : idx + 1] == "MM":
            return True
        return False

    @staticmethod
    def _is_material_grade_context(
        text: str,
        token: str,
        normalized: str,
        token_start: int,
        code: str,
        idx: int,
    ) -> bool:
        """
        Suppress short-code false positives inside material/grade expressions, e.g.:
        - ASTM A350GR.LF2
        - A350 LF2
        - LF3

        This filter is intentionally narrow:
        - only for short codes (<=2)
        - only when token looks like a material grade tail: letters + digits
        - only when nearby context contains obvious material-grade anchors
        """
        if len(code) > 2:
            return False

        suffix = normalized[idx + len(code) :]
        if not suffix or not suffix.isdigit():
            return False

        left_context = text[max(0, token_start - 20) : token_start].upper()
        token_upper = token.upper()

        material_anchor_patterns = (
            r"\bASTM\b",
            r"\bASME\b",
            r"\bA\d{2,4}\b",
            r"\bGR\.?\b",
            r"\bGRADE\b",
        )
        if any(re.search(pattern, left_context) for pattern in material_anchor_patterns):
            return True

        if re.match(r"^(LF|F|WP|TP)\d+[A-Z0-9]*$", token_upper):
            return True

        return False

    @staticmethod
    def _looks_glued(code: str, prefix: str, suffix: str) -> bool:
        """
        False-positive control:
        - long codes (>=3): any extra alnum chars count as glue
        - short codes (2 chars): only keep if the extra fragment is digit-containing
          or short uppercase tail/head (e.g. SOL, SWT, RF20)
        - single-char codes: only keep if glued with digits, otherwise false-positive risk is too high
        """
        if not prefix and not suffix:
            return False

        remainder = prefix + suffix
        code_len = len(code)

        if code_len >= 3:
            return True

        if code_len == 1:
            return any(ch.isdigit() for ch in remainder)

        # code_len == 2
        if any(ch.isdigit() for ch in remainder):
            return True

        if prefix and suffix:
            return False

        side = prefix or suffix
        return side.isalpha() and side.isupper() and len(side) <= 2

    @staticmethod
    def _build_note(code: str, token: str) -> str:
        return f"编码 {code} 与其他字符粘连，原始片段为 {token}"

    def _covered_by_longer_code(self, normalized: str, code: str, idx: int) -> bool:
        """
        If the current match is fully contained in a longer legal code that also
        appears in the same token, prefer the longer legal code and suppress the
        shorter one. This avoids cases like:
        - MFM20  -> do not report MF/FM/M
        - SAWLX  -> do not report SAW
        - FLRJ20 -> do not report RJ
        """
        end = idx + len(code)
        for other in self.all_codes:
            if len(other) <= len(code):
                continue

            other_idx = normalized.find(other)
            if other_idx < 0:
                continue

            other_end = other_idx + len(other)
            if other_idx <= idx and end <= other_end:
                return True

        return False

    @staticmethod
    def _load_code_groups(config_path: str | Path | None) -> dict[str, list[str]]:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config" / "type_glue_codes.yaml"
        else:
            config_path = Path(config_path)

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        code_groups: dict[str, list[str]] = {}
        for group_name, codes in data.items():
            if not isinstance(codes, list):
                raise ValueError(f"{group_name} must be a list in {config_path}")
            code_groups[group_name] = [str(code).upper() for code in codes if str(code).strip()]
        return code_groups
