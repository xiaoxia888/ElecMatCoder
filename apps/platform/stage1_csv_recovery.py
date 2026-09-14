"""从平台编码结果 CSV 恢复一阶段训练数据。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from apps.platform.batch_export import _format_field_value


_CONNECTION_TOKENS = {
    "BE",
    "BW",
    "FE",
    "FNPT",
    "MNPT",
    "NPT",
    "PBE",
    "PE",
    "SW",
    "THD",
}
_SEAL_TOKENS = {
    "FF",
    "FLFF",
    "FLM",
    "FLMF",
    "FLRJ",
    "FLRF",
    "FM",
    "LM",
    "M",
    "MF",
    "RF",
    "RJ",
    "SERRATED",
    "TG",
}
_MANUFACTURE_TOKENS = {
    "EFW", "ERW", "FORGED", "SAW", "SEAMLESS", "SMLS", "WELD", "WELDED"
}
_RADIUS_TOKENS = {"LR", "SR"}
_FLANGE_STYLE_TOKENS = {"FIXED_FLANGED", "FLANGED", "LAP_JOINT_FLANGED", "固定法兰"}
_SPECIAL_REQUIREMENTS = (
    "GALVANIZED",
    "GALV",
    "PWHT",
    "NACE",
    "HIC",
    "CE",
    "ZN",
)

_RESULT_COLUMNS = (
    "TYPE_原始结果",
    "SIZE_原始结果",
    "THICKNESS_原始结果",
    "PRESSURE_原始结果",
    "MATERIAL_原始结果",
    "STANDARD_原始结果",
)


@dataclass(frozen=True)
class RecoveryReferences:
    type_by_signature: Mapping[str, dict[str, Any]]
    material_by_signature: Mapping[str, list[dict[str, Any]]]
    standard_by_signature: Mapping[str, list[dict[str, Any]]]
    structure_by_signature: Mapping[tuple[str, str, str], dict[str, Any]]

    @classmethod
    def empty(cls) -> "RecoveryReferences":
        return cls({}, {}, {}, {})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_values(candidates: Mapping[Any, set[str]]) -> dict[Any, Any]:
    unique: dict[Any, Any] = {}
    for signature, values in candidates.items():
        if len(values) == 1:
            unique[signature] = json.loads(next(iter(values)))
    return unique


def load_references(project_root: Optional[Path] = None) -> RecoveryReferences:
    """从仓库现有训练集建立“导出显示文本 -> 原结构”唯一映射。"""
    root = project_root or Path(__file__).resolve().parents[2]
    dataset_root = root / "apps/trainer/qwen3_fte/output/按8类拆分数据集"
    type_files = sorted((dataset_root / "种类").glob("*_train.json"))
    type_files += sorted((dataset_root / "种类").glob("*_val.json"))
    material_root = dataset_root / "材质规范/结构化原始牌号"
    material_files = sorted(material_root.glob("*_train.json"))
    material_files += sorted(material_root.glob("*_val.json"))
    structure_root = dataset_root / "尺寸壁厚磅级/V2已划分"
    structure_files = [
        structure_root / "尺寸壁厚磅级V2_train.json",
        structure_root / "尺寸壁厚磅级V2_val.json",
    ]

    type_candidates: defaultdict[str, set[str]] = defaultdict(set)
    material_candidates: defaultdict[str, set[str]] = defaultdict(set)
    standard_candidates: defaultdict[str, set[str]] = defaultdict(set)
    structure_candidates: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)

    for path in type_files:
        for row in json.loads(path.read_text(encoding="utf-8")):
            value = (row.get("output") or {}).get("TYPE")
            if isinstance(value, dict):
                type_candidates[_format_field_value("TYPE", value)].add(_canonical(value))

    for path in material_files:
        for row in json.loads(path.read_text(encoding="utf-8")):
            output = row.get("output") or {}
            material = output.get("MATERIAL") or []
            standard = output.get("STANDARD") or []
            material_candidates[_format_field_value("MATERIAL", material)].add(
                _canonical(material)
            )
            standard_candidates[_format_field_value("STANDARD", standard)].add(
                _canonical(standard)
            )

    for path in structure_files:
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            output = row.get("output") or {}
            signature = tuple(
                _format_field_value(field, output)
                for field in ("SIZE", "THICKNESS", "PRESSURE")
            )
            value = {
                "ITEMS": output.get("ITEMS") or [],
                "LENGTH": output.get("LENGTH") or "",
                "PRESSURE": output.get("PRESSURE") or "",
            }
            structure_candidates[signature].add(_canonical(value))

    return RecoveryReferences(
        type_by_signature=_unique_values(type_candidates),
        material_by_signature=_unique_values(material_candidates),
        standard_by_signature=_unique_values(standard_candidates),
        structure_by_signature=_unique_values(structure_candidates),
    )


def parse_type_field(text: str, *, category: str) -> dict[str, Any]:
    tokens = [part.strip() for part in text.split(";") if part.strip()]
    if category.strip() == "法兰":
        body = tokens[0] if tokens else ""
        conn = [token for token in tokens[1:] if token.upper() in _CONNECTION_TOKENS]
        seal = [token for token in tokens[1:] if token.upper() not in _CONNECTION_TOKENS]
        return {"BODY": body, "CONN": conn, "SEAL": seal}

    flange_style = ""
    if len(tokens) >= 2 and tokens[0].upper() in _FLANGE_STYLE_TOKENS:
        flange_style = tokens[0]
        body = tokens[1]
        remaining_tokens = tokens[2:]
    else:
        body = tokens[0] if tokens else ""
        remaining_tokens = tokens[1:]
    angle = ""
    radius = ""
    manufacture: list[str] = []
    connections: list[str] = []
    for token in remaining_tokens:
        upper = token.upper()
        if re.fullmatch(r"\d+(?:\.\d+)?", token) and not angle:
            angle = token
        elif upper in _RADIUS_TOKENS or re.fullmatch(r"\d+(?:\.\d+)?D", upper):
            radius = token
        elif upper in _CONNECTION_TOKENS:
            connections.append(token)
        elif upper in _MANUFACTURE_TOKENS:
            manufacture.append(token)
        elif upper in _FLANGE_STYLE_TOKENS and not flange_style:
            flange_style = token
        else:
            connections.append(token)
    return {
        "BODY": body,
        "GEOMETRY": {"ANGLE": angle, "RADIUS": radius},
        "FLANGE_STYLE": flange_style,
        "MANU": manufacture,
        "CONN": connections,
    }


def parse_material_field(text: str) -> list[dict[str, Any]]:
    materials: list[dict[str, Any]] = []
    for raw_part in text.split(" ; "):
        part = raw_part.strip()
        if not part:
            continue
        role, separator, value = part.partition(":")
        if separator:
            role = role.strip() or "BODY"
            value = value.strip()
        else:
            role, value = "BODY", role.strip()
        special: list[str] = []
        upper = value.upper()
        for suffix in _SPECIAL_REQUIREMENTS:
            if upper.endswith(suffix) and len(value) > len(suffix):
                value = value[: -len(suffix)].rstrip()
                special.insert(0, suffix)
                upper = value.upper()
        materials.append({"PART": role, "VALUE": value, "SPECIAL_REQ": special})
    return materials


def parse_standard_field(text: str) -> list[dict[str, str]]:
    standards: list[dict[str, str]] = []
    for raw_part in text.split(" ; "):
        body = re.sub(r"（[^（）]*）\s*$", "", raw_part.strip()).strip()
        if body:
            standards.append({"BODY": body})
    return standards


def _standard_signature(text: str) -> str:
    parts = [
        re.sub(r"（[^（）]*）\s*$", "", part.strip()).strip()
        for part in text.split(" ; ")
    ]
    return " ; ".join(part for part in parts if part)


def _parse_atom(text: str, *, field: str) -> dict[str, str]:
    value = text.strip()
    upper = value.upper()
    if field == "THICKNESS":
        if upper.endswith("MM"):
            return {"type": "MM", "value": value[:-2].strip()}
        return {"type": "SCHEDULE", "value": value}
    for prefix in ("INCH", "DN", "OD", "NPS"):
        if upper.startswith(prefix):
            return {"type": prefix, "value": value[len(prefix) :].strip()}
    return {"type": "DN", "value": value}


def _parse_structural_segments(text: str, *, field: str) -> tuple[list[dict[str, Any]], str]:
    items: list[dict[str, Any]] = []
    length = ""
    for raw_segment in text.split(" ; "):
        segment = raw_segment.strip()
        if not segment:
            continue
        label, separator, content = segment.partition(":")
        if not separator:
            label, content = "SINGLE", label
        label = label.strip().upper()
        content = content.strip()
        if label == "LENGTH":
            length = content
            continue
        parts = label.split()
        role = parts[-1] if parts else "SINGLE"
        scope = " ".join(parts[:-1]) if len(parts) > 1 else "BODY"
        atoms = [
            _parse_atom(value, field=field)
            for value in content.split(" | ")
            if value.strip()
        ]
        items.append({"SCOPE": scope, "ROLE": role, field: atoms})
    return items, length


def parse_structural_fields(
    *, size_text: str, thickness_text: str, pressure_text: str
) -> dict[str, Any]:
    size_items, length = _parse_structural_segments(size_text, field="SIZE")
    thickness_items, _ = _parse_structural_segments(thickness_text, field="THICKNESS")
    combined: list[dict[str, Any]] = []
    positions: dict[tuple[str, str, int], int] = {}
    for source in (size_items, thickness_items):
        occurrences: Counter[tuple[str, str]] = Counter()
        for item in source:
            pair = (item["SCOPE"], item["ROLE"])
            key = (*pair, occurrences[pair])
            occurrences[pair] += 1
            if key not in positions:
                positions[key] = len(combined)
                combined.append(
                    {
                        "SCOPE": item["SCOPE"],
                        "ROLE": item["ROLE"],
                        "SIZE": [],
                        "THICKNESS": [],
                    }
                )
            target = combined[positions[key]]
            if "SIZE" in item:
                target["SIZE"] = item["SIZE"]
            if "THICKNESS" in item:
                target["THICKNESS"] = item["THICKNESS"]
    return {"ITEMS": combined, "LENGTH": length, "PRESSURE": pressure_text.strip()}


def recover_row(
    row: Mapping[str, str],
    *,
    references: Optional[RecoveryReferences] = None,
    preprocess: Optional[Callable[[str], str]] = None,
) -> tuple[Optional[dict[str, Any]], dict[str, str]]:
    """恢复单条训练数据；没有任何编码结果的行返回 ``None``。"""
    if not any(str(row.get(column, "") or "").strip() for column in _RESULT_COLUMNS):
        return None, {}

    references = references or RecoveryReferences.empty()
    original = str(row.get("原始描述", "") or "").strip()
    model_input = preprocess(original) if preprocess else original
    category = str(row.get("分类", "") or "").strip()

    type_text = str(row.get("TYPE_原始结果", "") or "").strip()
    type_value = references.type_by_signature.get(type_text)
    type_method = "reference" if type_value is not None else "parsed"
    if type_value is None:
        type_value = parse_type_field(type_text, category=category)

    material_text = str(row.get("MATERIAL_原始结果", "") or "").strip()
    material_value = references.material_by_signature.get(material_text)
    material_method = "reference" if material_value is not None else "parsed"
    if material_value is None:
        material_value = parse_material_field(material_text)

    standard_text = str(row.get("STANDARD_原始结果", "") or "").strip()
    normalized_standard = _standard_signature(standard_text)
    standard_value = references.standard_by_signature.get(normalized_standard)
    standard_method = "reference" if standard_value is not None else "parsed"
    if standard_value is None:
        standard_value = parse_standard_field(standard_text)

    structural_signature = tuple(
        str(row.get(f"{field}_原始结果", "") or "").strip()
        for field in ("SIZE", "THICKNESS", "PRESSURE")
    )
    structural_value = references.structure_by_signature.get(structural_signature)
    structural_method = "reference" if structural_value is not None else "parsed"
    if structural_value is None:
        structural_value = parse_structural_fields(
            size_text=structural_signature[0],
            thickness_text=structural_signature[1],
            pressure_text=structural_signature[2],
        )

    output = {
        "TYPE": copy.deepcopy(type_value),
        "MATERIAL": copy.deepcopy(material_value),
        "STANDARD": copy.deepcopy(standard_value),
        "ITEMS": copy.deepcopy(structural_value.get("ITEMS") or []),
        "LENGTH": str(structural_value.get("LENGTH") or ""),
        "PRESSURE": str(structural_value.get("PRESSURE") or ""),
    }
    return (
        {"original_input": original, "input": model_input, "output": output},
        {
            "TYPE": type_method,
            "MATERIAL": material_method,
            "STANDARD": standard_method,
            "STRUCTURE": structural_method,
        },
    )


def recover_csv(
    source_path: Path,
    dataset_path: Path,
    report_path: Path,
    *,
    references: Optional[RecoveryReferences] = None,
    preprocess: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    source_path = Path(source_path)
    dataset_path = Path(dataset_path)
    report_path = Path(report_path)
    references = references or load_references()
    if preprocess is None:
        from src.tokenizer_utils.preprocessor import TextPreprocessor

        preprocess = TextPreprocessor().process

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    recovered_rows = 0
    skipped: list[dict[str, str]] = []
    parsed_rows: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()

    with source_path.open(encoding="utf-8-sig", newline="") as source, dataset_path.open(
        "w", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        missing_columns = [column for column in _RESULT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"CSV 缺少必要列: {', '.join(missing_columns)}")
        target.write("[\n")
        first = True
        for row in reader:
            source_rows += 1
            recovered, methods = recover_row(
                row,
                references=references,
                preprocess=preprocess,
            )
            if recovered is None:
                skipped.append(
                    {
                        "序号": str(row.get("序号", "") or ""),
                        "原始描述": str(row.get("原始描述", "") or ""),
                        "原因": "CSV 中没有任何字段编码结果",
                    }
                )
                continue
            if not first:
                target.write(",\n")
            first = False
            target.write("  " + json.dumps(recovered, ensure_ascii=False))
            recovered_rows += 1
            for field, method in methods.items():
                method_counts[f"{field}.{method}"] += 1
            heuristic_fields = [field for field, method in methods.items() if method == "parsed"]
            if heuristic_fields:
                parsed_rows.append(
                    {
                        "序号": str(row.get("序号", "") or ""),
                        "字段": heuristic_fields,
                    }
                )
        target.write("\n]\n")

    summary: dict[str, Any] = {
        "source_csv": str(source_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "source_rows": source_rows,
        "recovered_rows": recovered_rows,
        "skipped_rows": len(skipped),
        "method_counts": dict(sorted(method_counts.items())),
        "parsed_rows": parsed_rows,
        "skipped": skipped,
        "important_note": (
            "原 CSV 优先导出 stage2_input；恢复标签表示二阶段实际使用/修正后的字段，"
            "不保证等同于已删除任务中的 stage1_raw。"
        ),
    }
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="平台导出的 encoding_results.csv")
    parser.add_argument("--output", required=True, type=Path, help="恢复后的一阶段训练 JSON")
    parser.add_argument("--report", required=True, type=Path, help="恢复统计与待复核报告 JSON")
    args = parser.parse_args()
    summary = recover_csv(args.csv, args.output, args.report)
    print(json.dumps({key: summary[key] for key in ("source_rows", "recovered_rows", "skipped_rows", "method_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
