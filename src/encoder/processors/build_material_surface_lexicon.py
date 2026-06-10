from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CSV_PATH = Path(
    "/Users/guoxi/Desktop/workspace/NJNCC/python_code/review_platform/materials_export.csv"
)
OUT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "material_surface_lexicon.yaml"
)

MIN_COUNT = 3
STRONG_RATIO = 0.98

NUMERIC_LIKE_RE = re.compile(r"^\d+(?:[#.-]\d+)?[#A-Za-z]*$")
PURE_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_material_surface_lexicon(
    csv_path: Path = CSV_PATH,
    out_path: Path = OUT_PATH,
    min_count: int = MIN_COUNT,
    strong_ratio: float = STRONG_RATIO,
) -> dict[str, object]:
    rows = 0
    raw_counter: Counter[str] = Counter()
    final_counter: Counter[str] = Counter()
    raw_to_final: dict[str, Counter[str]] = defaultdict(Counter)
    raw_in_desc_counter: Counter[str] = Counter()
    raw_in_desc_to_final: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            desc = (row.get("材料描述(多行)") or row.get("材料描述") or "").strip()
            raw = (row.get("原始材质") or "").strip()
            std = (row.get("标准化材质") or "").strip()
            fix = (row.get("修正材质") or "").strip()
            final = fix or std

            if raw:
                raw_counter[raw] += 1
            if final:
                final_counter[final] += 1
            if raw and final:
                raw_to_final[raw][final] += 1

            if raw and final and raw in desc:
                raw_in_desc_counter[raw] += 1
                raw_in_desc_to_final[raw][final] += 1
                if len(examples[raw]) < 2:
                    examples[raw].append(desc)

    recommended: dict[str, dict[str, object]] = {}
    weak_numeric: dict[str, dict[str, object]] = {}
    ambiguous: dict[str, dict[str, object]] = {}

    for raw, total in raw_in_desc_counter.items():
        mapping = raw_in_desc_to_final[raw]
        dominant_target, dominant_count = mapping.most_common(1)[0]
        ratio = dominant_count / total if total else 0.0
        entry = {
            "target": dominant_target,
            "count": total,
            "consistency": round(ratio, 4),
            "all_targets": dict(mapping),
            "examples": examples.get(raw, []),
        }

        if len(mapping) > 1 or ratio < strong_ratio:
            ambiguous[raw] = entry
        else:
            if (
                PURE_NUMERIC_RE.fullmatch(raw)
                or raw in {"20#", "20", "304", "316", "316L", "304L"}
                or NUMERIC_LIKE_RE.fullmatch(raw)
            ):
                weak_numeric[raw] = entry
            else:
                recommended[raw] = entry

    recommended_items = sorted(
        recommended.items(), key=lambda item: (-int(item[1]["count"]), item[0])
    )
    weak_items = sorted(
        weak_numeric.items(), key=lambda item: (-int(item[1]["count"]), item[0])
    )
    ambiguous_items = sorted(
        ambiguous.items(), key=lambda item: (-int(item[1]["count"]), item[0])
    )
    final_top = final_counter.most_common(120)

    with out_path.open("w", encoding="utf-8") as out:
        out.write("meta:\n")
        out.write(f"  total_rows: {rows}\n")
        out.write(f"  raw_unique: {len(raw_counter)}\n")
        out.write(f"  final_unique: {len(final_counter)}\n")
        out.write(f"  raw_in_description_unique: {len(raw_in_desc_counter)}\n")
        out.write(f"  min_count: {min_count}\n")
        out.write(f"  strong_ratio: {strong_ratio}\n\n")

        out.write("top_final_materials:\n")
        for value, count in final_top:
            out.write(f"  - value: {_quote(value)}\n")
            out.write(f"    count: {count}\n")
        out.write("\n")

        out.write("recommended_literal_map:\n")
        for raw, entry in recommended_items:
            if int(entry["count"]) < min_count:
                continue
            out.write(f"  {_quote(raw)}:\n")
            out.write(f"    target: {_quote(str(entry['target']))}\n")
            out.write(f"    count: {entry['count']}\n")
            out.write(f"    consistency: {entry['consistency']}\n")
        out.write("\n")

        out.write("weak_numeric_or_short_literal_map:\n")
        for raw, entry in weak_items:
            if int(entry["count"]) < min_count:
                continue
            out.write(f"  {_quote(raw)}:\n")
            out.write(f"    target: {_quote(str(entry['target']))}\n")
            out.write(f"    count: {entry['count']}\n")
            out.write(f"    consistency: {entry['consistency']}\n")
        out.write("\n")

        out.write("ambiguous_literal_map:\n")
        for raw, entry in ambiguous_items:
            if int(entry["count"]) < min_count:
                continue
            out.write(f"  {_quote(raw)}:\n")
            out.write(f"    count: {entry['count']}\n")
            out.write(f"    dominant_target: {_quote(str(entry['target']))}\n")
            out.write(f"    consistency: {entry['consistency']}\n")
            out.write("    all_targets:\n")
            for target, count in sorted(
                entry["all_targets"].items(),
                key=lambda item: (-int(item[1]), item[0]),
            ):
                out.write(f"      {_quote(target)}: {count}\n")
            if entry["examples"]:
                out.write("    examples:\n")
                for example in entry["examples"]:
                    out.write(f"      - {_quote(example[:220])}\n")

    return {
        "rows": rows,
        "raw_unique": len(raw_counter),
        "final_unique": len(final_counter),
        "raw_in_description_unique": len(raw_in_desc_counter),
        "recommended_count": len(recommended_items),
        "weak_count": len(weak_items),
        "ambiguous_count": len(ambiguous_items),
    }


if __name__ == "__main__":
    summary = build_material_surface_lexicon()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
