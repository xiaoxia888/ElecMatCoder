# -*- coding: utf-8 -*-
"""Project-level low-frequency type/material detector."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .models import DifficultyFeature, GlueHit


@dataclass
class ProjectFrequencyConfig:
    # 项目最小样本量：低于该值不启用项目维度低频判断，避免小项目天然波动导致误判。
    min_project_size: int = 100
    # 种类编码低频的绝对次数阈值：项目内出现次数小于等于该值时，才可能继续判低频。
    type_low_count_threshold: int = 8
    # 种类编码低频的占比阈值：项目内占比需低于该值，避免中等频次种类被误标。
    type_low_ratio_threshold: float = 0.015
    # 种类编码相对项目头部种类的阈值：当前频次 / 项目前三频次均值 <= 该值，说明明显落在长尾。
    type_top3_rel_threshold: float = 0.10
    # 材质编码低频的绝对次数阈值：材质天然更分散，因此阈值比种类更低。
    material_low_count_threshold: int = 3
    # 材质编码低频的占比阈值：用于限制项目内极低占比材质，避免正常长尾被大量误报。
    material_low_ratio_threshold: float = 0.008


class ProjectFrequencyDetector:
    """Batch-only detector based on project-level type/material frequency."""

    def __init__(self, config: ProjectFrequencyConfig | None = None) -> None:
        self.config = config or ProjectFrequencyConfig()

    @staticmethod
    def _clean(value: Any) -> str:
        text = str(value or "").strip()
        if text.lower() == "nan":
            return ""
        return text

    @staticmethod
    def _top3_mean(counter: Counter[str]) -> float:
        values = sorted((v for v in counter.values() if v > 0), reverse=True)
        if not values:
            return 0.0
        top = values[:3]
        return float(sum(top)) / float(len(top))

    def analyze_rows(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        project_key: str = "project",
        type_key: str = "type_code",
        material_key: str = "material_code",
    ) -> list[DifficultyFeature]:
        row_list = list(rows)
        features = [DifficultyFeature(name="project_frequency", matched=False) for _ in row_list]
        grouped_indices: dict[str, list[int]] = defaultdict(list)

        for idx, row in enumerate(row_list):
            project = self._clean(row.get(project_key, ""))
            if project:
                grouped_indices[project].append(idx)

        for project, indices in grouped_indices.items():
            project_size = len(indices)
            if project_size < self.config.min_project_size:
                continue

            type_counter = Counter(
                self._clean(row_list[idx].get(type_key, ""))
                for idx in indices
                if self._clean(row_list[idx].get(type_key, ""))
            )
            material_counter = Counter(
                self._clean(row_list[idx].get(material_key, ""))
                for idx in indices
                if self._clean(row_list[idx].get(material_key, ""))
            )
            type_top3_mean = self._top3_mean(type_counter)

            for idx in indices:
                hits: list[GlueHit] = []
                reasons: list[str] = []

                type_code = self._clean(row_list[idx].get(type_key, ""))
                if type_code:
                    type_count = int(type_counter.get(type_code, 0))
                    type_ratio = (float(type_count) / float(project_size)) if project_size else 0.0
                    type_top_rel = (float(type_count) / float(type_top3_mean)) if type_top3_mean > 0 else 0.0
                    type_low = (
                        type_count == 1
                        or (
                            type_count <= self.config.type_low_count_threshold
                            and type_ratio <= self.config.type_low_ratio_threshold
                            and type_top_rel <= self.config.type_top3_rel_threshold
                        )
                    )
                    if type_low:
                        reasons.append("项目内种类编码低频")
                        hits.append(
                            GlueHit(
                                tag="project_type_frequency",
                                code_group="type_code",
                                code=type_code,
                                token=type_code,
                                start=-1,
                                end=-1,
                                note=(
                                    f"项目内种类编码低频: {type_code} "
                                    f"({type_count}/{project_size}, {type_ratio * 100:.2f}%, "
                                    f"相对Top3均值 {type_top_rel * 100:.2f}%)"
                                ),
                            )
                        )

                material_code = self._clean(row_list[idx].get(material_key, ""))
                if material_code:
                    material_count = int(material_counter.get(material_code, 0))
                    material_ratio = (float(material_count) / float(project_size)) if project_size else 0.0
                    material_low = (
                        material_count == 1
                        or (
                            material_count <= self.config.material_low_count_threshold
                            and material_ratio <= self.config.material_low_ratio_threshold
                        )
                    )
                    if material_low:
                        reasons.append("项目内材质编码低频")
                        hits.append(
                            GlueHit(
                                tag="project_material_frequency",
                                code_group="material_code",
                                code=material_code,
                                token=material_code,
                                start=-1,
                                end=-1,
                                note=(
                                    f"项目内材质编码低频: {material_code} "
                                    f"({material_count}/{project_size}, {material_ratio * 100:.2f}%)"
                                ),
                            )
                        )

                if hits:
                    features[idx] = DifficultyFeature(
                        name="project_frequency",
                        matched=True,
                        reason="、".join(dict.fromkeys(reasons)),
                        hits=hits,
                    )

        return features
