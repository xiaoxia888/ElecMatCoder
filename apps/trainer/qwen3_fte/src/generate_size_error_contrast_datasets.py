#!/usr/bin/env python3
"""Generate standalone size/thickness/length contrast datasets from reviewed error patterns."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


NPS_ROWS = [
    ("0.5", "15", "21.3", "2.77"),
    ("0.75", "20", "26.7", "2.87"),
    ("1", "25", "33.4", "3.38"),
    ("1.5", "40", "48.3", "3.68"),
    ("2", "50", "60.3", "3.91"),
    ("2.5", "65", "73", "5.16"),
    ("3", "80", "88.9", "5.49"),
    ("4", "100", "114.3", "6.02"),
    ("6", "150", "168.3", "7.11"),
    ("8", "200", "219.1", "8.18"),
    ("10", "250", "273.1", "9.27"),
    ("12", "300", "323.9", "9.53"),
    ("16", "400", "406.4", "12.7"),
    ("20", "500", "508", "15.09"),
    ("24", "600", "610", "17.48"),
]


def size_item(item_type: str, value: str) -> dict[str, str]:
    return {"type": item_type, "value": value}


def thickness_item(item_type: str, value: str) -> dict[str, str]:
    return {"type": item_type, "value": value}


def make_row(
    text: str,
    *,
    sizes: list[dict[str, str]] | None = None,
    length: str = "",
    thicknesses: list[dict[str, str]] | None = None,
    pressure: str = "",
    augmentation_type: str,
    group_id: str,
    prototype: str,
) -> dict[str, Any]:
    return {
        "input": text,
        "output": {
            "SIZE_ITEMS": sizes or [],
            "LENGTH": length,
            "THICKNESS_ITEMS": thicknesses or [],
            "PRESSURE": pressure,
        },
        "来源": "数据增强",
        "数据增强标识": True,
        "增强类型": augmentation_type,
        "增强组ID": group_id,
        "原型来源": prototype,
    }


def build_numeric_semantic_contrast() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    standards = ["ASME B36.10M", "ASME B36.19M", "SH/T 3405"]
    for index, (_, dn, od, wall) in enumerate(NPS_ROWS):
        standard = standards[index % len(standards)]
        group = f"NUM-{index:03d}"
        rows.extend(
            [
                make_row(
                    f"PIPE, SMLS, {standard}, SCH40 {dn}",
                    sizes=[size_item("DN", dn)],
                    thicknesses=[thickness_item("SCHEDULE", "SCH40")],
                    augmentation_type="数字语义最小对比",
                    group_id=group,
                    prototype="用户反馈：管道尾部裸数字应识别为DN",
                ),
                make_row(
                    f"管道安装 {dn}mm, PIPE, SMLS, {standard}, SCH40",
                    sizes=[size_item("DN", dn)],
                    thicknesses=[thickness_item("SCHEDULE", "SCH40")],
                    augmentation_type="数字语义最小对比",
                    group_id=group,
                    prototype="用户反馈：安装描述中的公称mm口径不是长度",
                ),
                make_row(
                    f"PIPE Φ{od}×{wall} {standard}",
                    sizes=[size_item("OD", od)],
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="数字语义最小对比",
                    group_id=group,
                    prototype="用户反馈：明确外径与壁厚必须分离",
                ),
                make_row(
                    f"PIPE DN{dn}, THK={wall}mm, {standard}",
                    sizes=[size_item("DN", dn)],
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="数字语义最小对比",
                    group_id=group,
                    prototype="用户反馈：THK数字是壁厚",
                ),
                make_row(
                    f"PIPE DN{dn}, L={od}mm, {standard}",
                    sizes=[size_item("DN", dn)],
                    length=f"{od}MM",
                    augmentation_type="数字语义最小对比",
                    group_id=group,
                    prototype="用户反馈：L锚定数字是长度",
                ),
            ]
        )
    return rows


def build_bare_and_inch_variants() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (inch, dn, od, wall) in enumerate(NPS_ROWS):
        group = f"INCH-{index:03d}"
        schedule = "SCH10S" if index % 2 else "SCH40"
        rows.extend(
            [
                make_row(
                    f"碳素无缝钢管 {inch}PIPE 20 GB/T8163 SMLS BE {schedule} HG/T20553(Ia)",
                    sizes=[size_item("INCH", inch)],
                    thicknesses=[thickness_item("SCHEDULE", schedule)],
                    augmentation_type="裸尺寸与英制格式专项",
                    group_id=group,
                    prototype="用户反馈：数字与PIPE粘连表示寸径",
                ),
                make_row(
                    f"CS PIPE {inch} {schedule} ASTM A106 Gr.B, ASME B36.10M DN{dn}",
                    sizes=[size_item("INCH", inch), size_item("DN", dn)],
                    thicknesses=[thickness_item("SCHEDULE", schedule)],
                    augmentation_type="裸尺寸与英制格式专项",
                    group_id=group,
                    prototype="用户反馈：裸寸径与明确DN同时保留",
                ),
                make_row(
                    f"GB/T 8163;SH/T 3405 Pipe {od} {schedule} BE 20",
                    sizes=[size_item("OD", od)],
                    thicknesses=[thickness_item("SCHEDULE", schedule)],
                    augmentation_type="裸尺寸与英制格式专项",
                    group_id=group,
                    prototype="用户反馈：Pipe后的管系外径不是英寸",
                ),
                make_row(
                    f"PIPE, SMLS, ASME B36.19M {schedule}, A312 TP304, BE {dn}",
                    sizes=[size_item("DN", dn)],
                    thicknesses=[thickness_item("SCHEDULE", schedule)],
                    augmentation_type="裸尺寸与英制格式专项",
                    group_id=group,
                    prototype="用户反馈：管道尾部常见DN裸数字不是英寸",
                ),
            ]
        )

    fraction_rows = [
        ('PIPE,A106 GR.B,SMLS,BE,SCH80,ASME B36.10M 1/2"', "1/2"),
        ('PIPE,A106 GR.B,SMLS,BE,SCH80,ASME B36.10M 3/4"', "3/4"),
        ('PIPE,A106 GR.B,SMLS,BE,SCH80,ASME B36.10M 11/2"', "1-1/2"),
        ('PIPE,A106 GR.B,SMLS,BE,SCH80,ASME B36.10M 1-1/2"', "1-1/2"),
        ('PIPE,A106 GR.B,SMLS,BE,SCH80,ASME B36.10M 2 1/2"', "2-1/2"),
        ("PIPE,A106 GR.B,SMLS,BE,SCH80,ASME B36.10M 2 in", "2"),
    ]
    for index, (text, value) in enumerate(fraction_rows):
        rows.append(
            make_row(
                text,
                sizes=[size_item("INCH", value)],
                thicknesses=[thickness_item("SCHEDULE", "SCH80")],
                augmentation_type="裸尺寸与英制格式专项",
                group_id=f"FRACTION-{index:03d}",
                prototype="用户反馈：英制分数及粘连变体",
            )
        )
    return rows


def build_od_wall_combinations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (inch, dn, od, wall) in enumerate(NPS_ROWS):
        group = f"ODWT-{index:03d}"
        rows.extend(
            [
                make_row(
                    f"无缝钢管 D{od}×{wall} GB/T 9948-2013 20# SMLS",
                    sizes=[size_item("OD", od)],
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="外径与壁厚组合专项",
                    group_id=group,
                    prototype="用户反馈：D外径乘壁厚",
                ),
                make_row(
                    f"Pipe OD:{od} WT:{wall}mm ASTM A312 TP316L",
                    sizes=[size_item("OD", od)],
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="外径与壁厚组合专项",
                    group_id=group,
                    prototype="用户反馈：OD/WT强锚定",
                ),
                make_row(
                    f"不锈无缝钢管 Φ{od}×{wall} S30408 GB/T14976",
                    sizes=[size_item("OD", od)],
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="外径与壁厚组合专项",
                    group_id=group,
                    prototype="用户反馈：Φ外径乘壁厚",
                ),
                make_row(
                    f"不锈钢管 {inch}\" SCH80S {od}x{wall} SMLS ASTM A312 TP316L",
                    sizes=[size_item("INCH", inch), size_item("OD", od)],
                    thicknesses=[
                        thickness_item("SCHEDULE", "SCH80S"),
                        thickness_item("MM", wall),
                    ],
                    augmentation_type="外径与壁厚组合专项",
                    group_id=group,
                    prototype="用户反馈：SCH与MM壁厚同时保留",
                ),
            ]
        )

    for index in range(len(NPS_ROWS) - 1):
        inch1, _, od1, wall1 = NPS_ROWS[index + 1]
        inch2, _, od2, wall2 = NPS_ROWS[index]
        rows.append(
            make_row(
                f"Red.Tee {od1}x{wall1}-{od2}x{wall2} BW 20 {inch1}X{inch2}\"",
                sizes=[
                    size_item("OD", od1),
                    size_item("OD", od2),
                    size_item("INCH", inch1),
                    size_item("INCH", inch2),
                ],
                thicknesses=[thickness_item("MM", wall1), thickness_item("MM", wall2)],
                augmentation_type="外径与壁厚组合专项",
                group_id=f"RED-ODWT-{index:03d}",
                prototype="用户反馈：多组外径、壁厚和英寸顺序",
            )
        )
    return rows


def build_length_precision() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lengths = [
        "144.012", "148.5", "185.488", "210.707", "291.487", "304.421",
        "328.061", "330.998", "347.012", "361.546", "499.354", "503.056",
        "601.012", "680.886", "681.506", "722.693", "745.55", "839.682",
        "889.947", "934.009", "938.727", "2020.5",
    ]
    for index, length in enumerate(lengths):
        _, dn, od, wall = NPS_ROWS[index % len(NPS_ROWS)]
        group = f"LEN-{index:03d}"
        rows.extend(
            [
                make_row(
                    f"成品内衬管件 PIPE Lined L={length}mm CL150 RF {od}x{wall}(3.0) S30408/PP DN{dn}",
                    sizes=[size_item("OD", od), size_item("DN", dn)],
                    length=f"{length}MM",
                    thicknesses=[thickness_item("MM", wall), thickness_item("MM", "3")],
                    pressure="CL150",
                    augmentation_type="长度小数精度专项",
                    group_id=group,
                    prototype="用户反馈：L=小数长度不得丢失小数点",
                ),
                make_row(
                    f"法兰管, PTFE lined GB/T8163-20, RF, CL150, DN{dn}, S-40 {length}mm",
                    sizes=[size_item("DN", dn)],
                    length=f"{length}MM",
                    thicknesses=[thickness_item("SCHEDULE", "SCH40")],
                    pressure="CL150",
                    augmentation_type="长度小数精度专项",
                    group_id=group,
                    prototype="用户反馈：成品法兰管尾部mm是长度",
                ),
                make_row(
                    f"Pipe, X2CrNi19-11, SMLS, BE, {od}x{wall}, EN 10216-5, Length={length} mm",
                    sizes=[size_item("OD", od)],
                    length=f"{length}MM",
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="长度小数精度专项",
                    group_id=group,
                    prototype="用户反馈：Length明确锚定长度",
                ),
            ]
        )

    unit_variants = [
        ("1.2m", "1200MM"),
        ("0.75m", "750MM"),
        ("32.5cm", "325MM"),
        ("80cm", "800MM"),
        ("2500毫米", "2500MM"),
    ]
    for index, (raw, normalized) in enumerate(unit_variants):
        rows.append(
            make_row(
                f"成品管段 DN50 长度={raw} HG/T20538",
                sizes=[size_item("DN", "50")],
                length=normalized,
                augmentation_type="长度小数精度专项",
                group_id=f"LEN-UNIT-{index:03d}",
                prototype="用户反馈：长度统一转换为MM",
            )
        )
    return rows


def build_length_size_boundary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (_, dn, od, wall) in enumerate(NPS_ROWS):
        length = str(300 + index * 137)
        group = f"BOUNDARY-{index:03d}"
        rows.extend(
            [
                make_row(
                    f"不锈钢管道安装 {dn}mm, PIPE, SMLS, A312 TP304, SCH40S",
                    sizes=[size_item("DN", dn)],
                    thicknesses=[thickness_item("SCHEDULE", "SCH40S")],
                    augmentation_type="长度与尺寸边界对比",
                    group_id=group,
                    prototype="用户反馈：安装mm公称口径不是长度",
                ),
                make_row(
                    f"成品管段 DN{dn} 长度={length}mm PIPE Lined",
                    sizes=[size_item("DN", dn)],
                    length=f"{length}MM",
                    augmentation_type="长度与尺寸边界对比",
                    group_id=group,
                    prototype="用户反馈：明确长度锚定",
                ),
                make_row(
                    f"PIPE Φ{od}×{wall}mm DN{dn} L={length}mm",
                    sizes=[size_item("OD", od), size_item("DN", dn)],
                    length=f"{length}MM",
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="长度与尺寸边界对比",
                    group_id=group,
                    prototype="用户反馈：外径、壁厚、DN和长度同时出现",
                ),
            ]
        )
    return rows


def build_explicit_multi_evidence() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (inch, dn, od, wall) in enumerate(NPS_ROWS):
        group = f"MULTI-{index:03d}"
        rows.extend(
            [
                make_row(
                    f"不锈钢管道 {inch}\"(φ{od}) THK={wall}mm SS304 ASME B36.19M",
                    sizes=[size_item("INCH", inch), size_item("OD", od)],
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="显式多证据尺寸专项",
                    group_id=group,
                    prototype="用户反馈：明确英寸与外径证据同时保留",
                ),
                make_row(
                    f"Pipe, P235GH, SMLS, BE, {od}x{wall}, EN 10216-2;{inch}\" DN{dn}",
                    sizes=[size_item("OD", od), size_item("INCH", inch), size_item("DN", dn)],
                    thicknesses=[thickness_item("MM", wall)],
                    augmentation_type="显式多证据尺寸专项",
                    group_id=group,
                    prototype="用户反馈：OD、INCH、DN均有明确证据时按原文顺序保留",
                ),
                make_row(
                    f"PIPE DN{dn}, Φ{od}×{wall}, {inch}\", SMLS, SCH40",
                    sizes=[size_item("DN", dn), size_item("OD", od), size_item("INCH", inch)],
                    thicknesses=[thickness_item("MM", wall), thickness_item("SCHEDULE", "SCH40")],
                    augmentation_type="显式多证据尺寸专项",
                    group_id=group,
                    prototype="用户反馈：多种明确口径证据顺序不得重排",
                ),
            ]
        )
    return rows


BUILDERS = {
    "数字语义最小对比专项增强_带来源.json": build_numeric_semantic_contrast,
    "裸尺寸与英制格式专项增强_带来源.json": build_bare_and_inch_variants,
    "外径与壁厚组合专项增强_带来源.json": build_od_wall_combinations,
    "长度小数精度专项增强_带来源.json": build_length_precision,
    "长度与尺寸边界对比专项增强_带来源.json": build_length_size_boundary,
    "显式多证据尺寸专项增强_带来源.json": build_explicit_multi_evidence,
}


def validate_datasets(datasets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_inputs: set[str] = set()
    stats: dict[str, Any] = {}
    for filename, rows in datasets.items():
        if not rows:
            raise ValueError(f"专项集为空: {filename}")
        type_counts: Counter[str] = Counter()
        length_rows = 0
        for row in rows:
            text = row["input"]
            if text in all_inputs:
                raise ValueError(f"跨文件重复样本: {text}")
            all_inputs.add(text)
            if row.get("来源") != "数据增强" or row.get("数据增强标识") is not True:
                raise ValueError(f"缺少数据增强标识: {text}")
            output = row["output"]
            if set(output) != {"SIZE_ITEMS", "LENGTH", "THICKNESS_ITEMS", "PRESSURE"}:
                raise ValueError(f"输出结构不完整: {text}")
            if any(item == {"type": "DN", "value": "57"} for item in output["SIZE_ITEMS"]):
                raise ValueError("禁止对DN57脏冲突做数据增强")
            if output["LENGTH"] and not re.fullmatch(r"\d+(?:\.\d+)?MM", output["LENGTH"]):
                raise ValueError(f"长度未统一带MM: {text}")
            length_rows += bool(output["LENGTH"])
            type_counts.update(item["type"] for item in output["SIZE_ITEMS"])
            type_counts.update(item["type"] for item in output["THICKNESS_ITEMS"])
        stats[filename] = {
            "样本数": len(rows),
            "长度样本数": length_rows,
            "标签类型统计": dict(type_counts),
        }
    return {"总样本数": len(all_inputs), "文件统计": stats}


def generate(output_dir: Path) -> dict[str, Any]:
    datasets = {filename: builder() for filename, builder in BUILDERS.items()}
    report = validate_datasets(datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in datasets.items():
        (output_dir / filename).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report.update(
        {
            "输出目录": str(output_dir),
            "生成原则": [
                "全部样本为独立数据增强文件，未合并回原训练集。",
                "尺寸、壁厚按原文出现顺序标注。",
                "LENGTH统一保留MM单位且不丢失小数点。",
                "不生成DN57+Φ57类矛盾DN扩增样本。",
                "不在一阶段样本中学习OD到DN的换算。",
            ],
        }
    )
    (output_dir / "尺寸壁厚长度专项增强_生成报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
