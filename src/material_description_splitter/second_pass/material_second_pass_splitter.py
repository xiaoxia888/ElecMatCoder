# -*- coding: utf-8 -*-
"""Second-pass material auto-pass splitter."""

from __future__ import annotations

from .material_surface_matcher import MaterialSurfaceMatcher
from .models import MaterialSecondPassResult


class MaterialSecondPassSplitter:
    def __init__(self, matcher: MaterialSurfaceMatcher | None = None) -> None:
        self.matcher = matcher or MaterialSurfaceMatcher()

    @staticmethod
    def _filter_conflict_codes(text: str, conflict_map: dict[str, list], *, base_code: str) -> list[str]:
        if str(base_code or "").strip().upper() == "20":
            return []
        filtered: list[str] = []
        for code, hits in conflict_map.items():
            if code == "20":
                continue
            filtered.append(code)
        return filtered

    def analyze(self, text: str, material_code: str) -> MaterialSecondPassResult:
        clean_text = str(text or "").strip()
        clean_code = str(material_code or "").strip().upper()
        if not clean_text:
            return MaterialSecondPassResult(
                text=clean_text,
                material_code=clean_code,
                passed=False,
                reason="描述为空",
            )
        if not clean_code:
            return MaterialSecondPassResult(
                text=clean_text,
                material_code=clean_code,
                passed=False,
                reason="材质编码为空",
            )

        base_code, suffix_code = self.matcher.split_material_code(clean_code)
        if not base_code or not self.matcher.is_supported_code(clean_code):
            return MaterialSecondPassResult(
                text=clean_text,
                material_code=clean_code,
                passed=False,
                reason="不在二次分流常见材质白名单中",
                base_code=base_code,
                suffix_code=suffix_code,
            )

        base_hits = self.matcher.match_base_surfaces(clean_text, base_code)
        if not base_hits:
            return MaterialSecondPassResult(
                text=clean_text,
                material_code=clean_code,
                passed=False,
                reason="未命中主材强锚点表达",
                base_code=base_code,
                suffix_code=suffix_code,
            )

        any_suffix_hits = self.matcher.find_any_suffix_hits(clean_text)
        suffix_hits = []
        if suffix_code:
            suffix_hits = self.matcher.match_suffix_surfaces(clean_text, suffix_code)
            if not suffix_hits:
                combined_hits = self.matcher.match_combined_code(clean_text, clean_code)
                if combined_hits:
                    suffix_hits = combined_hits
            if not suffix_hits:
                return MaterialSecondPassResult(
                    text=clean_text,
                    material_code=clean_code,
                    passed=False,
                    reason=f"未命中{suffix_code}后缀表达",
                    base_code=base_code,
                    suffix_code=suffix_code,
                    base_hits=base_hits,
                )
            other_suffixes = [code for code in any_suffix_hits.keys() if code != suffix_code]
            if other_suffixes:
                return MaterialSecondPassResult(
                    text=clean_text,
                    material_code=clean_code,
                    passed=False,
                    reason=f"文本命中其他后缀表达: {'/'.join(other_suffixes)}",
                    base_code=base_code,
                    suffix_code=suffix_code,
                    base_hits=base_hits,
                    suffix_hits=suffix_hits,
                )
        else:
            if any_suffix_hits:
                found_suffixes = sorted(any_suffix_hits.keys())
                return MaterialSecondPassResult(
                    text=clean_text,
                    material_code=clean_code,
                    passed=False,
                    reason=f"文本命中后缀表达，但编码缺少后缀: {'/'.join(found_suffixes)}",
                    base_code=base_code,
                    suffix_code=suffix_code,
                    base_hits=base_hits,
                )

        excluded = {clean_code, base_code}
        if "/" in clean_code:
            excluded.update(part for part in clean_code.split("/") if part)
        conflict_map = self.matcher.find_conflict_hits(clean_text, excluded_codes=excluded)
        conflict_codes = self._filter_conflict_codes(clean_text, conflict_map, base_code=base_code)
        if conflict_codes:
            return MaterialSecondPassResult(
                text=clean_text,
                material_code=clean_code,
                passed=False,
                reason="命中其他常见材质强锚点，存在冲突",
                base_code=base_code,
                suffix_code=suffix_code,
                base_hits=base_hits,
                suffix_hits=suffix_hits,
                conflict_codes=conflict_codes,
            )

        return MaterialSecondPassResult(
            text=clean_text,
            material_code=clean_code,
            passed=True,
            reason="命中主材强锚点表达" + (f"，并命中{suffix_code}后缀表达" if suffix_code else ""),
            base_code=base_code,
            suffix_code=suffix_code,
            base_hits=base_hits,
            suffix_hits=suffix_hits,
        )
