# -*- coding: utf-8 -*-
"""Second-pass standard auto-pass splitter."""

from __future__ import annotations

from .models import StandardCodeCheck, StandardSecondPassResult
from .standard_surface_matcher import ParsedStandardCode, StandardSurfaceMatcher


class StandardSecondPassSplitter:
    def __init__(self, matcher: StandardSurfaceMatcher | None = None) -> None:
        self.matcher = matcher or StandardSurfaceMatcher()

    def analyze(self, text: str, standard_code: str) -> StandardSecondPassResult:
        clean_text = str(text or "").strip()
        raw_code = str(standard_code or "").strip()
        if not clean_text:
            return StandardSecondPassResult(
                text=clean_text,
                standard_code=raw_code,
                passed=False,
                reason="描述为空",
            )
        if not raw_code:
            return StandardSecondPassResult(
                text=clean_text,
                standard_code=raw_code,
                passed=False,
                reason="规范编码为空",
            )

        codes = self.matcher.split_codes(raw_code)
        if not codes:
            return StandardSecondPassResult(
                text=clean_text,
                standard_code=raw_code,
                passed=False,
                reason="规范编码为空",
            )

        checks: list[StandardCodeCheck] = []
        for code in codes:
            checks.append(self._analyze_single_code(clean_text, code, check_suspicious=False))

        global_consumed_suffix_hits = [
            hit
            for item in checks
            for hit in item.suffix_hits
        ]
        for item in checks:
            if not item.passed:
                continue
            if not item.base_hits:
                continue
            suspicious_suffix_hits = self.matcher.find_suspicious_suffix_hits(
                clean_text,
                ParsedStandardCode(
                    raw_code=item.raw_code,
                    family=item.family,
                    core=item.core,
                    suffix=item.suffix,
                ),
                item.base_hits[0],
                consumed_hits=global_consumed_suffix_hits,
            )
            item.suspicious_suffix_hits = suspicious_suffix_hits
            if suspicious_suffix_hits:
                item.passed = False
                item.reason = "存在疑似未编码后缀"

        failed = [item for item in checks if not item.passed]
        if failed:
            first = failed[0]
            return StandardSecondPassResult(
                text=clean_text,
                standard_code=raw_code,
                passed=False,
                reason=f"{first.raw_code}: {first.reason}",
                checks=checks,
                unmatched_standard_candidates=[],
                has_unmatched_standard_risk=False,
            )

        return StandardSecondPassResult(
            text=clean_text,
            standard_code=raw_code,
            passed=True,
            reason="命中规范强锚点表达",
            checks=checks,
            unmatched_standard_candidates=[],
            has_unmatched_standard_risk=False,
        )

    def analyze_items(self, text: str, standard_items: object) -> StandardSecondPassResult:
        clean_text = str(text or "").strip()
        items = self._normalize_standard_items(standard_items)
        if not clean_text:
            return StandardSecondPassResult(
                text=clean_text,
                standard_code="",
                passed=False,
                reason="描述为空",
            )
        if not items:
            return StandardSecondPassResult(
                text=clean_text,
                standard_code="",
                passed=False,
                reason="规范编码为空",
            )

        checks: list[StandardCodeCheck] = []
        for item in items:
            code = item["code"]
            category = item["category"]
            if not category:
                checks.append(
                    StandardCodeCheck(
                        raw_code=code,
                        category="",
                        passed=False,
                        reason="规范无法确定类型",
                    )
                )
                continue
            check = self._analyze_single_code(clean_text, code, check_suspicious=False)
            check.category = category
            checks.append(check)

        global_consumed_suffix_hits = [
            hit
            for item in checks
            if item.passed
            for hit in item.suffix_hits
        ]
        for item in checks:
            if not item.passed or not item.base_hits:
                continue
            suspicious_suffix_hits = self.matcher.find_suspicious_suffix_hits(
                clean_text,
                ParsedStandardCode(
                    raw_code=item.raw_code,
                    family=item.family,
                    core=item.core,
                    suffix=item.suffix,
                ),
                item.base_hits[0],
                consumed_hits=global_consumed_suffix_hits,
            )
            item.suspicious_suffix_hits = suspicious_suffix_hits
            if suspicious_suffix_hits:
                item.passed = False
                item.reason = "存在疑似未编码后缀"

        failed = [item for item in checks if not item.passed]
        joined_codes = "｜".join(item["code"] for item in items if item.get("code"))
        if failed:
            first = failed[0]
            return StandardSecondPassResult(
                text=clean_text,
                standard_code=joined_codes,
                passed=False,
                reason=f"{first.raw_code}: {first.reason}",
                checks=checks,
                unmatched_standard_candidates=[],
                has_unmatched_standard_risk=False,
            )

        return StandardSecondPassResult(
            text=clean_text,
            standard_code=joined_codes,
            passed=True,
            reason="命中规范强锚点表达",
            checks=checks,
            unmatched_standard_candidates=[],
            has_unmatched_standard_risk=False,
        )

    def _analyze_single_code(self, text: str, raw_code: str, check_suspicious: bool = True) -> StandardCodeCheck:
        parsed = self.matcher.parse_code(raw_code)
        if parsed is None:
            return StandardCodeCheck(raw_code=raw_code, passed=False, reason="无法解析规范编码")

        base_hits = self.matcher.find_base_hits(text, parsed)
        if not base_hits:
            return StandardCodeCheck(
                raw_code=raw_code,
                family=parsed.family,
                core=parsed.core,
                suffix=parsed.suffix,
                passed=False,
                reason="未命中规范主体表达",
            )

        family_requires_prefix = self.matcher.family_requires_prefix(parsed.family)
        family_allows_bare_core = self.matcher.family_allows_bare_core(parsed.family)
        valid_base_hits = []
        all_expected_prefix_hits = []
        all_conflicting_prefix_hits = []
        all_suffix_hits = []
        matched_prefix_base_exists = False

        for base_hit in sorted(base_hits, key=lambda hit: (-(hit.end - hit.start), hit.start)):
            expected_prefix_hits, conflicting_prefix_hits = self.matcher.find_prefix_hits(text, parsed, base_hit)
            all_expected_prefix_hits.extend(expected_prefix_hits)
            all_conflicting_prefix_hits.extend(conflicting_prefix_hits)
            if conflicting_prefix_hits and not expected_prefix_hits:
                continue
            if expected_prefix_hits:
                matched_prefix_base_exists = True
            elif family_requires_prefix:
                continue
            elif not family_allows_bare_core:
                continue

            valid_base_hits.append(base_hit)
            all_suffix_hits.extend(self.matcher.find_suffix_hits(text, parsed, base_hit))

        if not valid_base_hits:
            if all_conflicting_prefix_hits:
                return StandardCodeCheck(
                    raw_code=raw_code,
                    family=parsed.family,
                    core=parsed.core,
                    suffix=parsed.suffix,
                    passed=False,
                    reason="前缀不合规",
                    prefix_status="conflict",
                    base_hits=[sorted(base_hits, key=lambda hit: (-(hit.end - hit.start), hit.start))[0]],
                    prefix_hits=self._dedupe_hit_list(all_conflicting_prefix_hits),
                )
            return StandardCodeCheck(
                raw_code=raw_code,
                family=parsed.family,
                core=parsed.core,
                suffix=parsed.suffix,
                passed=False,
                reason="缺少规范前缀",
                prefix_status="missing",
                base_hits=[sorted(base_hits, key=lambda hit: (-(hit.end - hit.start), hit.start))[0]],
            )

        prefix_status = "matched" if matched_prefix_base_exists else "missing_allowed"
        suffix_hits = self._dedupe_hit_list(all_suffix_hits)
        if parsed.suffix:
            if not self.matcher.suffix_supported(parsed.suffix):
                return StandardCodeCheck(
                    raw_code=raw_code,
                    family=parsed.family,
                    core=parsed.core,
                    suffix=parsed.suffix,
                    passed=False,
                    reason="不支持的规范后缀",
                    prefix_status=prefix_status,
                    base_hits=self._dedupe_hit_list(valid_base_hits),
                    prefix_hits=self._dedupe_hit_list(all_expected_prefix_hits),
                )
            if not self.matcher.suffix_satisfied(parsed.suffix, suffix_hits):
                return StandardCodeCheck(
                    raw_code=raw_code,
                    family=parsed.family,
                    core=parsed.core,
                    suffix=parsed.suffix,
                    passed=False,
                    reason="未命中规范后缀表达",
                    prefix_status=prefix_status,
                    base_hits=self._dedupe_hit_list(valid_base_hits),
                    prefix_hits=self._dedupe_hit_list(all_expected_prefix_hits),
                    suffix_hits=suffix_hits,
                )

        suspicious_suffix_hits: list = []
        if check_suspicious:
            primary_base_hit = sorted(valid_base_hits, key=lambda hit: (hit.start, -(hit.end - hit.start)))[0]
            suspicious_suffix_hits = self.matcher.find_suspicious_suffix_hits(
                text, parsed, primary_base_hit, consumed_hits=suffix_hits
            )
            if not parsed.suffix and suspicious_suffix_hits:
                return StandardCodeCheck(
                    raw_code=raw_code,
                    family=parsed.family,
                    core=parsed.core,
                    suffix=parsed.suffix,
                    passed=False,
                    reason="存在疑似未编码后缀",
                    prefix_status=prefix_status,
                    base_hits=self._dedupe_hit_list(valid_base_hits),
                    prefix_hits=self._dedupe_hit_list(all_expected_prefix_hits),
                    suspicious_suffix_hits=suspicious_suffix_hits,
                )

        return StandardCodeCheck(
            raw_code=raw_code,
            family=parsed.family,
            core=parsed.core,
            suffix=parsed.suffix,
            passed=True,
            reason="命中规范强锚点表达",
            prefix_status=prefix_status,
            base_hits=self._dedupe_hit_list(valid_base_hits),
            prefix_hits=self._dedupe_hit_list(all_expected_prefix_hits),
            suffix_hits=suffix_hits,
            suspicious_suffix_hits=suspicious_suffix_hits,
        )

    @staticmethod
    def _normalize_standard_items(standard_items: object) -> list[dict[str, str]]:
        if not isinstance(standard_items, list):
            return []
        result: list[dict[str, str]] = []
        for item in standard_items:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            category = str(item.get("category") or "").strip()
            result.append({"code": code, "category": category})
        return result

    @staticmethod
    def _dedupe_hit_list(hits):
        dedup = {}
        for hit in hits:
            dedup[(hit.field, hit.alias, hit.start, hit.end)] = hit
        items = list(dedup.values())
        items.sort(key=lambda item: (item.start, -(item.end - item.start), item.alias))
        return items
