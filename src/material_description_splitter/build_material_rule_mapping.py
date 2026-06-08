from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoder.processors.build_material_rule_analysis_excel import DEFAULT_INPUT, build_analysis


OUT_PATH = Path(__file__).resolve().parent / "config" / "material_value_mapping.yaml"


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({value.strip() for value in values if value and value.strip()})


def build_material_rule_mapping(out_path: Path = OUT_PATH) -> Path:
    entries, _ = build_analysis(DEFAULT_INPUT)

    strong_map: dict[str, list[str]] = defaultdict(list)
    weak_map: dict[str, list[str]] = defaultdict(list)
    fallback_map: dict[str, list[str]] = defaultdict(list)
    strong_count: Counter[str] = Counter()
    weak_count: Counter[str] = Counter()
    fallback_count: Counter[str] = Counter()
    excluded_values: list[str] = []
    low_sample_values: list[str] = []

    for entry in entries:
        aliases = _sorted_unique([entry.raw, *entry.rule_candidates])
        if entry.category == "可规则":
            strong_map[entry.dominant_target].extend(aliases)
            strong_count[entry.dominant_target] += entry.in_desc_count
        elif entry.category in {"弱规则", "主体可规则_后缀另处理"}:
            weak_map[entry.dominant_target].extend(aliases)
            weak_count[entry.dominant_target] += entry.in_desc_count
        elif entry.category == "兜底规则":
            fallback_map[entry.dominant_target].extend(aliases)
            fallback_count[entry.dominant_target] += entry.in_desc_count
        elif entry.category == "不建议规则":
            excluded_values.append(entry.raw)
        elif entry.category == "样本过少":
            low_sample_values.append(entry.raw)

    def _sorted_targets(mapping: dict[str, list[str]], counts: Counter[str]) -> list[str]:
        return sorted(mapping, key=lambda target: (-counts[target], target))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        out.write("meta:\n")
        out.write("  source: material_rule_analysis.xlsx\n")
        out.write("  note: 按最新材质分析结果生成，三层规则仅用于候选提取，不用于最终拍板\n")
        out.write("  strong_rule: 允许粘连匹配，适合长牌号和高置信材质主体\n")
        out.write("  weak_rule: 不允许粘连，必须边界完整，适合短值和弱材质字面\n")
        out.write("  fallback_rule: 仅当 strong/weak 都未命中时才尝试，主要放中文/俗称\n")
        out.write("  suffix_policy: 命中 CE/ZN/GR.I/GR.II/GR.III 等后缀时，规则不拍板，交给大模型\n\n")

        out.write("strong_value_mapping:\n")
        for target in _sorted_targets(strong_map, strong_count):
            out.write(f"  {_quote(target)}:\n")
            for alias in _sorted_unique(strong_map[target]):
                out.write(f"    - {_quote(alias)}\n")
        out.write("\n")

        out.write("weak_value_mapping:\n")
        for target in _sorted_targets(weak_map, weak_count):
            out.write(f"  {_quote(target)}:\n")
            for alias in _sorted_unique(weak_map[target]):
                out.write(f"    - {_quote(alias)}\n")
        out.write("\n")

        out.write("fallback_value_mapping:\n")
        for target in _sorted_targets(fallback_map, fallback_count):
            out.write(f"  {_quote(target)}:\n")
            for alias in _sorted_unique(fallback_map[target]):
                out.write(f"    - {_quote(alias)}\n")
        out.write("\n")

        out.write("model_only_suffix:\n")
        out.write("  CE:\n")
        out.write('    - "CE"\n')
        out.write('    - "ANTI-H2S"\n')
        out.write('    - "NACE"\n')
        out.write("  ZN:\n")
        out.write('    - "ZN"\n')
        out.write('    - "GALV"\n')
        out.write('    - "GALVANIZED"\n')
        out.write("\n")

        out.write("model_only_grade_suffix:\n")
        out.write("  I:\n")
        out.write('    - "I"\n')
        out.write('    - "Gr.I"\n')
        out.write('    - "GR.I"\n')
        out.write("  II:\n")
        out.write('    - "II"\n')
        out.write('    - "Gr.II"\n')
        out.write('    - "GR.II"\n')
        out.write("  III:\n")
        out.write('    - "III"\n')
        out.write('    - "Gr.III"\n')
        out.write('    - "GR.III"\n')
        out.write("\n")

        out.write("excluded_values:\n")
        for value in sorted(set(excluded_values)):
            out.write(f"  - {_quote(value)}\n")
        out.write("\n")

        out.write("low_sample_values:\n")
        for value in sorted(set(low_sample_values)):
            out.write(f"  - {_quote(value)}\n")

    return out_path


if __name__ == "__main__":
    path = build_material_rule_mapping()
    print(path)
