# -*- coding: utf-8 -*-
"""Second-pass type auto-pass splitter."""

from __future__ import annotations

from .models import TypeSecondPassResult
from .type_surface_matcher import TypeSurfaceMatcher


class TypeSecondPassSplitter:
    def __init__(self, matcher: TypeSurfaceMatcher | None = None) -> None:
        self.matcher = matcher or TypeSurfaceMatcher()

    def analyze(self, text: str, type_code: str) -> TypeSecondPassResult:
        clean_text = str(text or "").strip()
        clean_code = str(type_code or "").strip().upper()
        if not clean_text:
            return TypeSecondPassResult(
                text=clean_text,
                type_code=clean_code,
                passed=False,
                reason="描述为空",
            )
        if not clean_code:
            return TypeSecondPassResult(
                text=clean_text,
                type_code=clean_code,
                passed=False,
                reason="种类编码为空",
            )
        if not self.matcher.is_supported_code(clean_code):
            return TypeSecondPassResult(
                text=clean_text,
                type_code=clean_code,
                passed=False,
                reason="不在二次分流常见种类白名单中",
            )

        direct_hits = self.matcher.match_direct(clean_text, clean_code)
        body_hits = self.matcher.match_body(clean_text, clean_code)
        manu_hits = self.matcher.match_manu(clean_text, clean_code)
        conn_hits = self.matcher.match_conn(clean_text, clean_code)
        seal_hits = self.matcher.match_seal(clean_text, clean_code)
        angle_hits = self.matcher.match_angle(clean_text, clean_code)
        radius_hits = self.matcher.match_radius(clean_text, clean_code)

        # 父码不能在文本已经强支持更具体子码时自动通过。
        for blocking_code in self.matcher.get_more_specific_codes(clean_code):
            if self._matches_code(clean_text, blocking_code):
                return TypeSecondPassResult(
                    text=clean_text,
                    type_code=clean_code,
                    passed=False,
                    reason=f"文本命中更具体子类证据: {blocking_code}",
                    direct_hits=direct_hits,
                    body_hits=body_hits,
                    manu_hits=manu_hits,
                    conn_hits=conn_hits,
                    seal_hits=seal_hits,
                    angle_hits=angle_hits,
                    radius_hits=radius_hits,
                    blocking_code=blocking_code,
                )

        matched_path = self._match_mode(clean_text, clean_code)
        if matched_path:
            return TypeSecondPassResult(
                text=clean_text,
                type_code=clean_code,
                passed=True,
                reason="命中种类强锚点表达",
                matched_path=matched_path,
                direct_hits=direct_hits,
                body_hits=body_hits,
                manu_hits=manu_hits,
                conn_hits=conn_hits,
                seal_hits=seal_hits,
                angle_hits=angle_hits,
                radius_hits=radius_hits,
            )

        return TypeSecondPassResult(
            text=clean_text,
            type_code=clean_code,
            passed=False,
            reason="未命中种类强锚点表达",
            direct_hits=direct_hits,
            body_hits=body_hits,
            manu_hits=manu_hits,
            conn_hits=conn_hits,
            seal_hits=seal_hits,
            angle_hits=angle_hits,
            radius_hits=radius_hits,
        )

    def _matches_code(self, text: str, type_code: str) -> bool:
        return bool(self._match_mode(text, type_code))

    def _match_mode(self, text: str, type_code: str) -> str:
        direct_hits = self.matcher.match_direct(text, type_code)
        body_hits = self.matcher.match_body(text, type_code)
        manu_hits = self.matcher.match_manu(text, type_code)
        conn_hits = self.matcher.match_conn(text, type_code)
        seal_hits = self.matcher.match_seal(text, type_code)
        angle_hits = self.matcher.match_angle(text, type_code)
        radius_hits = self.matcher.match_radius(text, type_code)

        if direct_hits:
            return "direct"
        has_manu_rule = bool(self.matcher.manu_patterns.get(type_code))
        has_conn_rule = bool(self.matcher.conn_patterns.get(type_code))
        has_seal_rule = bool(self.matcher.seal_patterns.get(type_code))
        has_angle_rule = bool(self.matcher.angle_patterns.get(type_code))
        has_radius_rule = bool(self.matcher.radius_patterns.get(type_code))
        if not body_hits:
            return ""
        if has_angle_rule and not angle_hits:
            return ""
        if has_radius_rule and not radius_hits:
            return ""
        if has_manu_rule and not manu_hits:
            return ""
        if has_conn_rule and not conn_hits:
            return ""
        if has_seal_rule and not seal_hits:
            return ""
        if has_angle_rule and has_radius_rule and has_manu_rule and has_conn_rule and has_seal_rule:
            return "body_plus_angle_radius_manu_conn_seal"
        if has_angle_rule and has_radius_rule and has_manu_rule and has_conn_rule:
            return "body_plus_angle_radius_manu_conn"
        if has_angle_rule and has_radius_rule and has_manu_rule and has_seal_rule:
            return "body_plus_angle_radius_manu_seal"
        if has_angle_rule and has_radius_rule and has_conn_rule and has_seal_rule:
            return "body_plus_angle_radius_conn_seal"
        if has_angle_rule and has_radius_rule and has_manu_rule:
            return "body_plus_angle_radius_manu"
        if has_angle_rule and has_radius_rule and has_conn_rule:
            return "body_plus_angle_radius_conn"
        if has_angle_rule and has_radius_rule and has_seal_rule:
            return "body_plus_angle_radius_seal"
        if has_angle_rule and has_radius_rule:
            return "body_plus_angle_radius"
        if has_manu_rule and has_conn_rule and has_seal_rule:
            return "body_plus_manu_conn_seal"
        if has_manu_rule and has_conn_rule:
            return "body_plus_manu_conn"
        if has_manu_rule and has_seal_rule:
            return "body_plus_manu_seal"
        if has_conn_rule and has_seal_rule:
            return "body_plus_conn_seal"
        if has_manu_rule:
            return "body_plus_manu"
        if has_conn_rule:
            return "body_plus_conn"
        if has_seal_rule:
            return "body_plus_seal"
        return "body_only"
