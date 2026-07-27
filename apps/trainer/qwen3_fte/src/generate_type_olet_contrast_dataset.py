#!/usr/bin/env python3
"""Generate a standalone OLET-family contrast dataset for type extraction."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit_type_olet_annotations import DEFAULT_DATASET, SCOPE_RE, audit_row


QWEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = QWEN_ROOT / "output" / "按8类拆分数据集" / "种类"
DATASET_NAME = "管件_OLET支管台专项对比增强"
SOURCE_LABEL = "数据增强-OLET支管台专项对比"


@dataclass(frozen=True)
class Signal:
    text: str
    body: str
    conn: tuple[str, ...] = ()
    manu: tuple[str, ...] = ()
    angle: str = ""
    pattern: str = ""


SUPPORT_SIGNALS = (
    Signal("OLET", "支管台", pattern="裸OLET"),
    Signal("Olet", "支管台", pattern="裸OLET-大小写"),
    Signal("支管台", "支管台", pattern="裸中文主体词"),
    Signal("支管座", "支管台", pattern="裸中文同义词"),
    Signal("管接台", "支管台", pattern="裸中文同义词"),
    Signal("SMLSOlet", "支管台", manu=("SMLS",), pattern="裸OLET-前缀粘连"),
    Signal("OletSMLS", "支管台", manu=("SMLS",), pattern="裸OLET-后缀粘连"),
    Signal("OLET SMLS", "支管台", manu=("SMLS",), pattern="裸OLET-工艺干扰"),
)

BUTT_SIGNALS = (
    Signal("OLET BW", "对焊支管台", pattern="裸OLET-BW"),
    Signal("BW OLET", "对焊支管台", pattern="裸OLET-BW前置"),
    Signal("BWOlet", "对焊支管台", pattern="裸OLET-BW前缀粘连"),
    Signal("OletBW", "对焊支管台", pattern="裸OLET-BW后缀粘连"),
    Signal("OLET B.W.", "对焊支管台", pattern="裸OLET-BW点号"),
    Signal("OLET BE", "对焊支管台", pattern="裸OLET-BE"),
    Signal("BE OLET", "对焊支管台", pattern="裸OLET-BE前置"),
    Signal("OLET BUTT WELD", "对焊支管台", pattern="裸OLET-BUTT-WELD"),
    Signal("WELDOLET", "对焊支管台", pattern="显式WELDOLET"),
    Signal("WELD OLET", "对焊支管台", pattern="显式WELD-OLET"),
    Signal("WOL-90", "对焊支管台", pattern="显式WOL"),
    Signal("对焊支管台", "对焊支管台", pattern="显式中文产品词"),
    Signal("对焊支管座", "对焊支管台", pattern="显式中文同义词"),
    Signal("支管台;BW", "对焊支管台", pattern="中文裸主体-BW"),
    Signal("OletSMLS;BW", "对焊支管台", manu=("SMLS",), pattern="BW与SMLS对比"),
    Signal("SMLSWELDOLET", "对焊支管台", manu=("SMLS",), pattern="显式产品词-前缀粘连"),
)

SOCKET_SIGNALS = (
    Signal("OLET SW", "承插焊支管台", conn=("SW",), pattern="裸OLET-SW"),
    Signal("SW OLET", "承插焊支管台", conn=("SW",), pattern="裸OLET-SW前置"),
    Signal("SWOlet", "承插焊支管台", conn=("SW",), pattern="裸OLET-SW前缀粘连"),
    Signal("OletSW", "承插焊支管台", conn=("SW",), pattern="裸OLET-SW后缀粘连"),
    Signal("OLET S.W.", "承插焊支管台", conn=("SW",), pattern="裸OLET-SW点号"),
    Signal("OLET SOCKET WELD", "承插焊支管台", conn=("SW",), pattern="裸OLET-SOCKET-WELD"),
    Signal("SOCKOLET", "承插焊支管台", conn=("SW",), pattern="显式SOCKOLET"),
    Signal("SOCKET OLET", "承插焊支管台", conn=("SW",), pattern="显式SOCKET-OLET"),
    Signal("SOCKET WELD OLET", "承插焊支管台", conn=("SW",), pattern="显式SOCKET-WELD-OLET"),
    Signal("SOL-90", "承插焊支管台", conn=("SW",), pattern="显式SOL"),
    Signal("承插焊支管台", "承插焊支管台", conn=("SW",), pattern="显式中文产品词"),
    Signal("承插焊支管座", "承插焊支管台", conn=("SW",), pattern="显式中文同义词"),
    Signal("支管台;SW", "承插焊支管台", conn=("SW",), pattern="中文裸主体-SW"),
    Signal(
        "SockoletSMLS",
        "承插焊支管台",
        conn=("SW",),
        manu=("SMLS",),
        pattern="显式产品词-后缀粘连",
    ),
    Signal(
        "SMLSSockolet",
        "承插焊支管台",
        conn=("SW",),
        manu=("SMLS",),
        pattern="显式产品词-前缀粘连",
    ),
    Signal(
        "SMLS OLET SW",
        "承插焊支管台",
        conn=("SW",),
        manu=("SMLS",),
        pattern="SW与SMLS对比",
    ),
)

THREAD_SIGNALS = (
    Signal("OLET FNPT", "螺纹支管台", conn=("FNPT",), pattern="裸OLET-FNPT"),
    Signal("OLET MNPT", "螺纹支管台", conn=("MNPT",), pattern="裸OLET-MNPT"),
    Signal("OLET NPT", "螺纹支管台", conn=("NPT",), pattern="裸OLET-NPT"),
    Signal("OLET NPTF", "螺纹支管台", conn=("NPTF",), pattern="裸OLET-NPTF"),
    Signal("OLET THD", "螺纹支管台", conn=("THD",), pattern="裸OLET-THD"),
    Signal("OLET SCRD", "螺纹支管台", conn=("SCRD",), pattern="裸OLET-SCRD"),
    Signal("FNPTOlet", "螺纹支管台", conn=("FNPT",), pattern="裸OLET-螺纹前缀粘连"),
    Signal("OletFNPT", "螺纹支管台", conn=("FNPT",), pattern="裸OLET-螺纹后缀粘连"),
    Signal("THREDOLET", "螺纹支管台", pattern="显式THREDOLET"),
    Signal("THREAD OLET", "螺纹支管台", pattern="显式THREAD-OLET"),
    Signal("THREADED OUTLET", "螺纹支管台", pattern="显式THREADED-OUTLET"),
    Signal("TOL-90", "螺纹支管台", pattern="显式TOL"),
    Signal("螺纹支管台", "螺纹支管台", pattern="显式中文产品词"),
    Signal("螺纹支管座", "螺纹支管台", pattern="显式中文同义词"),
    Signal("支管台;FNPT", "螺纹支管台", conn=("FNPT",), pattern="中文裸主体-FNPT"),
    Signal(
        "SMLSTHREDOLET",
        "螺纹支管台",
        manu=("SMLS",),
        pattern="显式产品词-前缀粘连",
    ),
)

LATERAL_SIGNALS = (
    Signal("LATROLET", "斜支管台", pattern="显式LATROLET"),
    Signal("LATROLET BW", "斜支管台", pattern="LATROLET优先于BW"),
    Signal("LATROLET BE", "斜支管台", pattern="LATROLET优先于BE"),
    Signal("45 LATROLET", "斜支管台", angle="45", pattern="LATROLET显式角度"),
    Signal("45° LATROLET BW", "斜支管台", angle="45", pattern="LATROLET角度与BW"),
    Signal(
        "SMLSLATROLET",
        "斜支管台",
        manu=("SMLS",),
        pattern="LATROLET前缀粘连",
    ),
    Signal("斜支管台", "斜支管台", pattern="显式中文产品词"),
    Signal("斜支管座", "斜支管台", pattern="显式中文同义词"),
    Signal("45°斜支管台", "斜支管台", angle="45", pattern="中文显式角度"),
    Signal("嵌入式支管台", "斜支管台", pattern="嵌入式同义词"),
    Signal("45°嵌入式支管台", "斜支管台", angle="45", pattern="嵌入式显式角度"),
    Signal("45°对焊斜支管台 BW", "斜支管台", angle="45", pattern="斜支管台优先于对焊"),
)

CONTEXTS = (
    "{signal}, DN100x25, CL3000, ASTM A105, MSS SP-97",
    "NAME:{signal};DN80x20;SCH40xSCH80;NB/T 47008 20;GB/T 19326",
    '8"x1" ASTM A182 F304 {signal} STDxSCH80 MSS SP-97',
    "支管件 | {signal} | DN150x40 | CL6000 | S30408 | GB/T 19326",
)

SIZE_PAIRS = (("100", "25"), ("80", "20"), ("150", "40"), ("200", "50"))

OLET_TEE_CONTRAST_SPECS = (
    ("10", "1", "SCH20", "SCH80", "A105"),
    ("8", "1", "STD", "SCH80", "A105"),
    ("12", "2", "SCH40", "SCH80", "A105"),
    ("6", "1", "SCH80", "SCH160", "A182 F304"),
    ("4", "0.5", "SCH40S", "SCH80S", "A182 F316L"),
    ("24", "6", "XS", "SCH80", "A234 WPB"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 OLET 支管台种类专项对比增强训练集")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def type_output(signal: Signal) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": signal.body,
            "GEOMETRY": {
                "ANGLE": signal.angle,
                "RADIUS": "",
            },
            "FLANGE_STYLE": "",
            "MANU": list(signal.manu),
            "CONN": list(signal.conn),
        },
    }


def make_row(description: str, signal: Signal) -> dict[str, Any]:
    return {
        "input": description,
        "output": type_output(signal),
        "来源": SOURCE_LABEL,
        "增强模式": signal.pattern,
    }


def build_signal_rows(
    signals: tuple[Signal, ...],
    repeats: int,
    context_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_index, signal in enumerate(signals):
        for repeat in range(repeats):
            context = CONTEXTS[(signal_index + repeat + context_offset) % len(CONTEXTS)]
            rows.append(make_row(context.format(signal=signal.text), signal))
    return rows


def build_hard_negatives() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    products = (
        ("90 Deg. Elbow BW", "90", "", ()),
        ("90°弯头 SW", "90", "", ("SW",)),
        ("45 Deg. Elbow B.W.", "45", "", ()),
        ("45°弯头 S.W.", "45", "", ("SW",)),
        ("90 ELBOW LR BW", "90", "LR", ()),
        ("90 ELBOW SR SW", "90", "SR", ("SW",)),
        ("45 ELBOW LR BW", "45", "LR", ()),
        ("弯头 90° 长半径 BW", "90", "LR", ()),
        ("弯头 90° 短半径 SW", "90", "SR", ("SW",)),
        ("ELBOW BW", "", "", ()),
        ("承插焊弯头 SW", "", "", ("SW",)),
        ("对焊弯头 BW", "", "", ()),
    )
    for index, (product, angle, radius, conn) in enumerate(products):
        main_size, branch_size = SIZE_PAIRS[index % len(SIZE_PAIRS)]
        description = (
            f'{product}, {main_size} NB, SCH40, ASTM A105, '
            f'MSS SP-75, REF SIZE {main_size}"x{branch_size}"'
        )
        signal = Signal(
            product,
            "弯头",
            conn=conn,
            angle=angle,
            pattern="弯头硬负例",
        )
        output = type_output(signal)
        output["TYPE"]["GEOMETRY"]["RADIUS"] = radius
        rows.append(
            {
                "input": description,
                "output": output,
                "来源": SOURCE_LABEL,
                "增强模式": signal.pattern,
            }
        )
    return rows


def build_olet_tee_hard_contrasts() -> list[dict[str, Any]]:
    """Keep dimensions identical so the product headword is the only BODY signal."""
    rows: list[dict[str, Any]] = []
    for main_size, branch_size, main_sch, branch_sch, material in OLET_TEE_CONTRAST_SPECS:
        size = f'{main_size}"*{branch_size}"'
        schedules = f"{main_sch}*{branch_sch}"
        olet_signal = Signal(
            "OLET BW",
            "对焊支管台",
            pattern="OLET与异径三通同尺寸强对比-OLET",
        )
        tee_signal = Signal(
            "REDUCING TEE",
            "异径三通",
            pattern="OLET与异径三通同尺寸强对比-异径三通",
        )
        olet_descriptions = (
            f"Olet {size} {schedules} Olet,{size},{schedules} BW,{material},MSS SP-97",
            f"OLET,{size},{schedules},BW,{material},MSS SP-97",
            f"BW OLET {size} {schedules} {material} MSS SP-97",
            f"支管台 {size} {schedules} BW {material} MSS SP-97",
        )
        tee_descriptions = (
            f"Reducing Tee {size} {schedules} BW,{material},ASME B16.9",
            f"RT,{size},{schedules},BW,{material},ASME B16.9",
            f"异径三通 {size} {schedules} BW {material} ASME B16.9",
            f"TEE REDUCING {size} {schedules} BW {material} ASME B16.9",
        )
        rows.extend(make_row(description, olet_signal) for description in olet_descriptions)
        rows.extend(make_row(description, tee_signal) for description in tee_descriptions)
    return rows


def build_regression_rows() -> list[dict[str, Any]]:
    cases = (
        (
            'Olet 10"*1" SCH20*SCH80 Olet,10"*1",SCH20*SCH80 BW,A105,MSS SP-97',
            Signal("OLET BW", "对焊支管台", pattern="线上误判原句回归-BW"),
        ),
        (
            'Olet 8"*1" STD Olet,8"*1",STD*STD,A105,MSS SP-97',
            Signal("OLET", "支管台", pattern="线上误判同句式-无端部"),
        ),
        (
            'Olet 8"*1" STD Olet,8"*1",STD*STD BW,A105,MSS SP-97',
            Signal("OLET BW", "对焊支管台", pattern="线上误判回归-BW"),
        ),
        (
            'Olet 8"*1" STD Olet,8"*1",STD*STD SOCKET WELD,A105,MSS SP-97',
            Signal(
                "OLET SOCKET WELD",
                "承插焊支管台",
                conn=("SW",),
                pattern="线上误判同句式-SOCKET-WELD",
            ),
        ),
        (
            'Olet 8"*1" STD Olet,8"*1",STD*STD FNPT,A105,MSS SP-97',
            Signal(
                "OLET FNPT",
                "螺纹支管台",
                conn=("FNPT",),
                pattern="线上误判同句式-FNPT",
            ),
        ),
    )
    return [make_row(description, signal) for description, signal in cases]


def normalize_description(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def load_existing_inputs(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalize_description(row["input"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("input"), str)
    }


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    for row in rows:
        key = normalize_description(row["input"])
        signature = json.dumps(row["output"], ensure_ascii=False, sort_keys=True)
        previous = seen.get(key)
        if previous is not None:
            if previous != signature:
                raise ValueError(f"同一描述存在冲突标签: {row['input']}")
            duplicate_count += 1
            continue
        seen[key] = signature
        result.append(row)
    return result, duplicate_count


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows):
        output = row.get("output")
        type_value = output.get("TYPE") if isinstance(output, dict) else None
        if not isinstance(row.get("input"), str) or not row["input"].strip():
            errors.append(f"row={index}: input 为空")
        elif not isinstance(type_value, dict) or output.get("CATEGORY") != "管件":
            errors.append(f"row={index}: 输出结构不完整")
        elif not str(type_value.get("BODY") or "").strip():
            errors.append(f"row={index}: TYPE.BODY 为空")
        elif SCOPE_RE.search(row["input"]) and type_value["BODY"] != "弯头":
            audit = audit_row(row, index)
            if audit["status"] != "correct":
                errors.append(
                    f"row={index}: OLET审计未通过: {audit['status']} "
                    f"{audit['current_body']} -> {audit['expected_body']} | {row['input']}"
                )
    return errors


def main() -> int:
    args = parse_args()
    source_dataset = args.source_dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(build_signal_rows(SUPPORT_SIGNALS, repeats=4, context_offset=0))
    rows.extend(build_signal_rows(BUTT_SIGNALS, repeats=3, context_offset=1))
    rows.extend(build_signal_rows(SOCKET_SIGNALS, repeats=3, context_offset=2))
    rows.extend(build_signal_rows(THREAD_SIGNALS, repeats=3, context_offset=3))
    rows.extend(build_signal_rows(LATERAL_SIGNALS, repeats=3, context_offset=0))
    rows.extend(build_regression_rows())
    rows.extend(build_hard_negatives())
    rows.extend(build_olet_tee_hard_contrasts())

    rows, duplicate_count = deduplicate_rows(rows)
    errors = validate_rows(rows)
    if errors:
        raise ValueError("\n".join(errors[:30]))

    existing_inputs = load_existing_inputs(source_dataset)
    incremental_rows = [
        row
        for row in rows
        if normalize_description(row["input"]) not in existing_inputs
    ]
    overlap_count = len(rows) - len(incremental_rows)
    random.Random(args.seed).shuffle(rows)
    random.Random(args.seed).shuffle(incremental_rows)

    output_path = output_dir / f"{DATASET_NAME}.json"
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    incremental_output_path = output_dir / f"{DATASET_NAME}_仅新增.json"
    incremental_output_path.write_text(
        json.dumps(incremental_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "source_dataset": str(source_dataset),
        "output": str(output_path),
        "incremental_output": str(incremental_output_path),
        "seed": args.seed,
        "rows": len(rows),
        "incremental_rows": len(incremental_rows),
        "duplicates_removed": duplicate_count,
        "existing_input_overlaps": overlap_count,
        "validation_errors": 0,
        "category_counts": dict(Counter(row["output"]["CATEGORY"] for row in rows)),
        "body_counts": dict(Counter(row["output"]["TYPE"]["BODY"] for row in rows)),
        "connection_counts": dict(
            Counter(
                "|".join(row["output"]["TYPE"]["CONN"]) or "[]"
                for row in rows
            )
        ),
        "pattern_counts": dict(Counter(row["增强模式"] for row in rows)),
        "incremental_body_counts": dict(
            Counter(row["output"]["TYPE"]["BODY"] for row in incremental_rows)
        ),
        "incremental_pattern_counts": dict(
            Counter(row["增强模式"] for row in incremental_rows)
        ),
    }
    report_path = output_dir / f"{DATASET_NAME}_报告.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
