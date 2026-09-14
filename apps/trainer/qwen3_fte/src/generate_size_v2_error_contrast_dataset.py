#!/usr/bin/env python3
"""Generate standalone V2 contrast data for recurrent structural extraction errors."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = (
    ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2已划分"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2专项增强对比"
)
DEFAULT_ACTUAL_ERRORS = Path("/Users/guoxi/Downloads/测试结果.xlsx")

SCOPES = {"BODY", "INNER", "OUTER", "LINING", "UNKNOWN"}
ROLES = {"SINGLE", "MAIN", "BRANCH", "END_A", "END_B", "UNKNOWN"}
SIZE_TYPES = {"DN", "OD", "INCH"}
THICKNESS_TYPES = {"MM", "SCHEDULE"}

REDUCING_VALUES = [
    ("32", "20", "3", "2.5"),
    ("40", "25", "3.5", "3"),
    ("50", "25", "4", "3"),
    ("50", "40", "4.5", "4"),
    ("65", "25", "5", "3"),
    ("65", "50", "5.5", "4.5"),
    ("80", "40", "5.5", "4"),
    ("80", "50", "6", "4.5"),
    ("100", "50", "6.5", "4"),
    ("100", "80", "7", "5"),
    ("125", "80", "7.5", "5.5"),
    ("150", "50", "8", "4"),
    ("150", "100", "8.5", "6"),
    ("200", "100", "9", "6"),
    ("200", "150", "10", "7"),
    ("250", "150", "12", "8"),
    ("300", "200", "14", "9"),
    ("350", "250", "16", "10"),
    ("400", "300", "18", "12"),
    ("500", "300", "20", "14"),
    ("600", "400", "22", "16"),
]

PIPE_VALUES = [
    ("15", "21.3", "2.77", "1/2"),
    ("20", "26.9", "2.87", "3/4"),
    ("25", "33.7", "3.38", "1"),
    ("32", "42.4", "3.56", "1.25"),
    ("40", "48.3", "3.68", "1.5"),
    ("50", "60.3", "3.91", "2"),
    ("65", "76.1", "5.16", "2.5"),
    ("80", "88.9", "5.49", "3"),
    ("100", "114.3", "6.02", "4"),
    ("125", "139.7", "6.55", "5"),
    ("150", "168.3", "7.11", "6"),
    ("200", "219.1", "8.18", "8"),
    ("250", "273.1", "9.27", "10"),
    ("300", "323.9", "9.53", "12"),
    ("400", "406.4", "12.7", "16"),
]


@dataclass(frozen=True)
class Sample:
    input: str
    output: dict[str, Any]
    category: str
    group: str
    source: str = "synthetic_contrast"

    def training_row(self) -> dict[str, Any]:
        return {"input": self.input, "output": self.output}


def field(item_type: str, value: str) -> dict[str, str]:
    return {"type": item_type, "value": str(value)}


def item(
    role: str,
    *,
    sizes: Iterable[tuple[str, str]] = (),
    thicknesses: Iterable[tuple[str, str]] = (),
    scope: str = "BODY",
) -> dict[str, Any]:
    return {
        "SCOPE": scope,
        "ROLE": role,
        "SIZE": [field(kind, value) for kind, value in sizes],
        "THICKNESS": [field(kind, value) for kind, value in thicknesses],
    }


def output(
    items: Iterable[dict[str, Any]] = (),
    *,
    length: str = "",
    pressure: str = "",
) -> dict[str, Any]:
    return {"ITEMS": list(items), "LENGTH": length, "PRESSURE": pressure}


def add(
    rows: list[Sample],
    category: str,
    group: str,
    text: str,
    result: dict[str, Any],
    *,
    source: str = "synthetic_contrast",
) -> None:
    rows.append(Sample(text, result, category, group, source))


def build_reducing_and_topology_samples(rows: list[Sample]) -> None:
    category = "异径第二端与拓扑归属"
    materials = ["20", "06Cr19Ni10", "S30408", "A234 WPB"]
    for index, (main, branch, wall_a, wall_b) in enumerate(REDUCING_VALUES):
        group = f"topology-{index:03d}"
        material = materials[index % len(materials)]
        reducer = output(
            [
                item("END_A", sizes=[("DN", main)], thicknesses=[("MM", wall_a)]),
                item("END_B", sizes=[("DN", branch)], thicknesses=[("MM", wall_b)]),
            ]
        )
        tee = output(
            [
                item("MAIN", sizes=[("DN", main)], thicknesses=[("MM", wall_a)]),
                item("BRANCH", sizes=[("DN", branch)], thicknesses=[("MM", wall_b)]),
            ]
        )
        olet = output(
            [
                item("MAIN", sizes=[("DN", main)], thicknesses=[("MM", wall_a)]),
                item("BRANCH", sizes=[("DN", branch)], thicknesses=[("MM", wall_b)]),
            ]
        )
        add(
            rows,
            category,
            group,
            f"TR, BE, GB/T12459II, GB/T13401, {wall_a}X{wall_b}mm, {material}, SMLS, DN{main}X{branch}",
            tee,
        )
        add(
            rows,
            category,
            group,
            f"TR, BE, GB/T12459II, GB/T13401, {wall_a}X{wall_b}mm, {material}, SMLS, GB/T14976 DN{main}X{branch}",
            tee,
        )
        add(
            rows,
            category,
            group,
            f"偏心异径管;SMLS;BW;GB/T12459(II);φ{main}Xφ{branch};{wall_a}mmX{wall_b}mm",
            output(
                [
                    item("END_A", sizes=[("OD", main)], thicknesses=[("MM", wall_a)]),
                    item("END_B", sizes=[("OD", branch)], thicknesses=[("MM", wall_b)]),
                ]
            ),
        )
        add(
            rows,
            category,
            group,
            f"CON REDUCER {material} SMLS BW ASME B16.9 DN{main}xDN{branch} THK {wall_a}x{wall_b}mm",
            reducer,
        )
        add(
            rows,
            category,
            group,
            f"REDUCING TEE, {material}, SMLS, BW, SH/T3408, {wall_a}X{wall_b}mm DN{main}x{branch}",
            tee,
        )
        add(
            rows,
            category,
            group,
            f"TEE RED;ASME B16.9;BE;{material};DN{main}XDN{branch};THK={wall_a}X{wall_b}MM",
            tee,
        )
        add(
            rows,
            category,
            group,
            f"异径三通;SMLS;BW;GB/T13401;GB/T12459 Series II;{wall_a}mmX{wall_b}mm DN{main}X{branch}",
            tee,
        )
        add(
            rows,
            category,
            group,
            f"WELDOLET BW GB/T19326(I) {wall_a}X{wall_b}mm DN{main}X{branch} {material}",
            olet,
        )
        add(
            rows,
            category,
            group,
            f"对焊支管台;BW;NB/T47008;GB/T19326;主管DN{main} 支管DN{branch};{wall_a}x{wall_b}mm",
            olet,
        )

    ocr_values = [
        ("200", "100", "9", "6"),
        ("250", "150", "12", "8"),
        ("300", "150", "14", "8"),
        ("350", "100", "16", "6"),
        ("400", "150", "18", "8"),
        ("500", "100", "20", "6"),
        ("600", "150", "22", "8"),
        ("700", "100", "24", "6"),
    ]
    for index, (end_a, end_b, wall_a, wall_b) in enumerate(ocr_values):
        group = f"ocr-second-end-{index:03d}"
        expected = output(
            [
                item("END_A", sizes=[("DN", end_a)], thicknesses=[("MM", wall_a)]),
                item("END_B", sizes=[("DN", end_b)], thicknesses=[("MM", wall_b)]),
            ]
        )
        suffix = end_b[1:]
        add(
            rows,
            category,
            group,
            f"CON.REDUCER SMLS BW GB/T12459 DN{end_a}xl{suffix} {wall_a}X{wall_b}mm",
            expected,
        )
        add(
            rows,
            category,
            group,
            f"偏心大小头 GB/T13401 DN{end_a}xI{suffix}, THK={wall_a}/{wall_b}mm",
            expected,
        )



def build_numeric_boundary_samples(rows: list[Sample]) -> None:
    category = "尺寸数字边界与非目标数字"
    for index, (dn, od, wall, inch) in enumerate(PIPE_VALUES):
        group = f"boundary-{index:03d}"
        single_od = output(
            [item("SINGLE", sizes=[("OD", od)], thicknesses=[("MM", wall)])]
        )
        add(
            rows,
            category,
            group,
            f"90°弯头 R=1.5DN:φ{od}X{wall},20#,GB/T8163 SH/T3408",
            single_od,
        )
        add(
            rows,
            category,
            group,
            f"45 Deg Elbow R=1.5D, A234 WPB, SMLS, BW, {od}x{wall}, ASME B16.9",
            single_od,
        )
        add(
            rows,
            category,
            group,
            f"Pipe, P235GH(1.0345), SMLS, BE, {od}x{wall}, EN 10216-2 {300 + index * 100}mm",
            output(
                [item("SINGLE", sizes=[("OD", od)], thicknesses=[("MM", wall)])],
                length=f"{300 + index * 100}MM",
            ),
        )
        add(
            rows,
            category,
            group,
            f"PIPE {inch} in, A106 Gr.B, SMLS, BE, SCH40, ASME B36.10M",
            output(
                [item("SINGLE", sizes=[("INCH", inch)], thicknesses=[("SCHEDULE", "SCH40")])]
            ),
        )
        add(
            rows,
            category,
            group,
            f"PIPE {inch}\", OD {od} x {wall}mm, 20#, GB/T8163",
            output(
                [
                    item(
                        "SINGLE",
                        sizes=[("INCH", inch), ("OD", od)],
                        thicknesses=[("MM", wall)],
                    )
                ]
            ),
        )
        add(
            rows,
            category,
            group,
            f"弯头 DN{dn}, R=1.5DN, {od}x{wall}, 20#, 100%RT, 设计温度150℃",
            output(
                [
                    item(
                        "SINGLE",
                        sizes=[("DN", dn), ("OD", od)],
                        thicknesses=[("MM", wall)],
                    )
                ]
            ),
        )

    category = "长度与英制尺寸边界"
    for index, (length, inch) in enumerate(
        [("50", "1/2"), ("80", "3/4"), ("100", "1"), ("120", "1.25"),
         ("150", "1.5"), ("200", "2"), ("250", "2.5"), ("300", "3")]
    ):
        group = f"length-inch-{index:03d}"
        expected = output(
            [item("SINGLE", sizes=[("INCH", inch)], thicknesses=[("SCHEDULE", "XS")])],
            length=f"{length}MM",
        )
        add(
            rows,
            category,
            group,
            f"NIPPLE, ASTM A106 Gr.B, SMLS, PE, XS, {length}MM;{inch} in",
            expected,
        )
        add(
            rows,
            category,
            group,
            f"单丝头 {inch}\" SCH XS L={length}mm 20# NB/T47008",
            expected,
        )


def build_multiple_thickness_samples(rows: list[Sample]) -> None:
    category = "同位置多壁厚与双端壁厚"
    for index, (dn, od, wall, _) in enumerate(PIPE_VALUES):
        other = str(round(float(wall) - 0.5, 2)).rstrip("0").rstrip(".")
        group = f"multi-wall-{index:03d}"
        single = output(
            [
                item(
                    "SINGLE",
                    sizes=[("OD", od), ("DN", dn)],
                    thicknesses=[("MM", wall), ("MM", other)],
                )
            ]
        )
        add(
            rows,
            category,
            group,
            f"45 Deg Elbow, P235GH, SMLS, TYPE5, BW, {od}x{wall}/{other}, DIN2605-1 DN{dn}",
            single,
        )
        add(
            rows,
            category,
            group,
            f"弯头 TYPE5 DN{dn} φ{od}×{wall}/{other}mm BW EN10253",
            single,
        )
        add(
            rows,
            category,
            group,
            f"PIPE DN{dn} OD{od} THK={wall}mm SCH40 SMLS",
            output(
                [
                    item(
                        "SINGLE",
                        sizes=[("DN", dn), ("OD", od)],
                        thicknesses=[("MM", wall), ("SCHEDULE", "SCH40")],
                    )
                ]
            ),
        )
        add(
            rows,
            category,
            group,
            f"PIPE DN{dn} THK={wall}/{wall}mm SMLS GB/T8163",
            output(
                [item("SINGLE", sizes=[("DN", dn)], thicknesses=[("MM", wall)])]
            ),
        )

    for index, (end_a, end_b, wall_a, wall_b) in enumerate(REDUCING_VALUES):
        group = f"paired-wall-{index:03d}"
        add(
            rows,
            category,
            group,
            f"REDUCER DN{end_a}X{end_b} {wall_a}X{wall_b}mm SMLS BW",
            output(
                [
                    item("END_A", sizes=[("DN", end_a)], thicknesses=[("MM", wall_a)]),
                    item("END_B", sizes=[("DN", end_b)], thicknesses=[("MM", wall_b)]),
                ]
            ),
        )
        add(
            rows,
            category,
            group,
            f"RED.TEE DN{end_a}X{end_b} SCH40XSCH80 THK={wall_a}X{wall_b}mm",
            output(
                [
                    item(
                        "MAIN",
                        sizes=[("DN", end_a)],
                        thicknesses=[("SCHEDULE", "SCH40"), ("MM", wall_a)],
                    ),
                    item(
                        "BRANCH",
                        sizes=[("DN", end_b)],
                        thicknesses=[("SCHEDULE", "SCH80"), ("MM", wall_b)],
                    ),
                ]
            ),
        )


def build_schedule_contrast_samples(rows: list[Sample]) -> None:
    category = "壁厚等级与相似字符负例"
    materials = ["S22053", "S30408", "S31603", "A105", "20"]
    for index, (dn, od, wall, _) in enumerate(PIPE_VALUES):
        group = f"schedule-{index:03d}"
        material = materials[index % len(materials)]
        base_size = [("DN", dn)]
        add(
            rows,
            category,
            group,
            f"弯头 DN{dn} {material} SH/T3408 GB/T12459 SMLS BW",
            output([item("SINGLE", sizes=base_size)]),
        )
        add(
            rows,
            category,
            group,
            f"弯头 DN{dn} {material} S3408 GB/T12459 SMLS BW",
            output([item("SINGLE", sizes=base_size)]),
        )
        add(
            rows,
            category,
            group,
            f"弯头 DN{dn} {material} SH/T3408 SCH40 SMLS BW",
            output(
                [item("SINGLE", sizes=base_size, thicknesses=[("SCHEDULE", "SCH40")])]
            ),
        )
        add(
            rows,
            category,
            group,
            f"PIPE DN{dn}, S-40, {material}, ASME B36.10M",
            output(
                [item("SINGLE", sizes=base_size, thicknesses=[("SCHEDULE", "SCH40")])]
            ),
        )
        add(
            rows,
            category,
            group,
            f"PIPE DN{dn}, MFR STD, {material}, ASME B36.10M",
            output([item("SINGLE", sizes=base_size)]),
        )
        add(
            rows,
            category,
            group,
            f"PIPE DN{dn}, XS, {material}, OD{od}X{wall}mm",
            output(
                [
                    item(
                        "SINGLE",
                        sizes=[("DN", dn), ("OD", od)],
                        thicknesses=[("SCHEDULE", "XS"), ("MM", wall)],
                    )
                ]
            ),
        )

    for index, (main, branch, _, _) in enumerate(REDUCING_VALUES[:12]):
        group = f"olet-trailing-mm-{index:03d}"
        thickness = str((index % 5) + 3)
        add(
            rows,
            category,
            group,
            f"对焊管接台 BW GB/T19326 DN{main}X{branch}-{thickness} {20 + index % 2}",
            output(
                [
                    item("MAIN", sizes=[("DN", main)]),
                    item("BRANCH", sizes=[("DN", branch)], thicknesses=[("MM", thickness)]),
                ]
            ),
        )


def build_pressure_contrast_samples(rows: list[Sample]) -> None:
    category = "压力等级语义正负对比"
    for index, (dn, _, _, _) in enumerate(PIPE_VALUES):
        group = f"pressure-{index:03d}"
        pressure_number = "150" if index % 2 == 0 else "300"
        pressure = f"CL{pressure_number}"
        base = [item("SINGLE", sizes=[("DN", dn)])]
        add(rows, category, group, f"BLIND FLANGE DN{dn} RF Class {pressure_number} ASME B16.5", output(base, pressure=pressure))
        add(rows, category, group, f"BLIND FLANGE DN{dn} RF {pressure_number} Lbs ASME B16.5", output(base, pressure=pressure))
        add(rows, category, group, f"BLIND FLANGE DN{dn} RF {pressure_number}# ASME B16.5", output(base, pressure=pressure))
        add(rows, category, group, f"法兰盖;RF;CL{pressure_number};HG/T20592;DN{dn}", output(base, pressure=pressure))
        add(rows, category, group, f"法兰盖;RFPN16;HG/T20592;DN{dn}", output(base, pressure="PN16"))
        add(rows, category, group, f"法兰 DN{dn}, A350 Gr.LF2 CL1, RF, ASME B16.5", output(base))
        add(rows, category, group, f"管件 DN{dn}, A234 WP11 CL2, SMLS, BW, ASME B16.9", output(base))
        add(rows, category, group, f"管件 DN{dn}, GB/T12459 Series II Class I, SMLS, BW", output(base))


def build_long_noise_samples(rows: list[Sample]) -> None:
    category = "长描述关键字段抗干扰"
    coating = ["50", "80", "100", "120"]
    for index, (main, branch, wall_a, wall_b) in enumerate(REDUCING_VALUES):
        group = f"long-{index:03d}"
        coat = coating[index % len(coating)]
        pressure = "PN16" if index % 2 == 0 else "CL150"
        expected = output(
            [
                item("MAIN", sizes=[("DN", main)], thicknesses=[("MM", wall_a)]),
                item("BRANCH", sizes=[("DN", branch)], thicknesses=[("MM", wall_b)]),
            ],
            pressure=pressure,
        )
        text = (
            f"1.名称:异径三通 2.材质:20# SMLS BW 3.规格:DN{main}X{branch} "
            f"THK={wall_a}X{wall_b}mm {pressure} 4.执行标准:GB/T12459-2017、GB/T13401-2017 "
            f"5.焊接方法:氩电联焊 6.除锈防腐:Sa2.5级，环氧富锌底漆1道，"
            f"涂层最小干膜厚度1×{coat}μm，面漆2道，设计温度150℃，100%RT，"
            "安装运输及吊装费用综合考虑"
        )
        add(rows, category, group, text, expected)
        add(
            rows,
            category,
            group,
            f"设计温度150℃；检测比例100%；制造标准GB/T13401；{pressure}；"
            f"REDUCING TEE SMLS BW DN{main}X{branch} {wall_a}X{wall_b}mm；"
            f"外表面涂层{coat}μm；材质20；项目说明及安装方式综合考虑",
            expected,
        )
        add(
            rows,
            category,
            group,
            f"管件安装，材质S30408，标准GB/T12459，防腐层{coat}μm，试验压力仅作技术要求；"
            f"产品参数在末尾：异径三通 DN{main}X{branch} THK {wall_a}X{wall_b}mm {pressure}",
            expected,
        )


def build_samples() -> list[Sample]:
    rows: list[Sample] = []
    build_reducing_and_topology_samples(rows)
    build_numeric_boundary_samples(rows)
    build_multiple_thickness_samples(rows)
    build_schedule_contrast_samples(rows)
    build_pressure_contrast_samples(rows)
    build_long_noise_samples(rows)
    return rows


def _number(value: str) -> str:
    number = str(value).strip()
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def _actual_sample(
    row_number: int,
    text: str,
    category: str,
    result: dict[str, Any],
) -> Sample:
    return Sample(
        input=text,
        output=result,
        category=f"真实项目错误-{category}",
        group=f"actual-row-{row_number}",
        source="actual_project_error",
    )


def parse_actual_project_error(
    row_number: int,
    text: str,
    correct_size: Any,
    correct_thickness: Any,
    correct_pressure: Any,
    size_ok: Any,
    thickness_ok: Any,
    pressure_ok: Any,
) -> Sample | None:
    if all(value not in (0, "0", False) for value in (size_ok, thickness_ok, pressure_ok)):
        return None

    # Material grade 20# beside R=1.5DN must not become a DN20 size.
    match = re.search(
        r"R\s*=\s*1\.5DN\s*20#.*?[ØΦφ]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)",
        text,
        re.I | re.S,
    )
    if match:
        od = _number(match.group(1))
        reviewed_walls = re.findall(r"(\d+(?:\.\d+)?)MM", str(correct_thickness or ""), re.I)
        if not reviewed_walls:
            return None
        wall = _number(reviewed_walls[0])
        return _actual_sample(
            row_number,
            text,
            "材质数字与尺寸边界",
            output([item("SINGLE", sizes=[("OD", od)], thicknesses=[("MM", wall)])]),
        )

    # Reducing tee: the two sizes and two thicknesses map to MAIN/BRANCH.
    match = re.search(
        r"^TR\s*,\s*BE.*?(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm"
        r".*?DN\s*(\d+)\s*[x×]\s*(\d+)",
        text,
        re.I | re.S,
    )
    if match:
        wall_a, wall_b, main, branch = map(_number, match.groups())
        return _actual_sample(
            row_number,
            text,
            "异径三通双端漏提",
            output(
                [
                    item("MAIN", sizes=[("DN", main)], thicknesses=[("MM", wall_a)]),
                    item("BRANCH", sizes=[("DN", branch)], thicknesses=[("MM", wall_b)]),
                ]
            ),
        )

    # TYPE5 elbows explicitly retain both distinct wall thicknesses at one position.
    match = re.search(
        r"TYPE5.*?BW\s*,\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)"
        r"\s*/\s*(\d+(?:\.\d+)?).*?DN\s*(\d+)",
        text,
        re.I | re.S,
    )
    if match:
        od, wall_a, wall_b, dn = map(_number, match.groups())
        return _actual_sample(
            row_number,
            text,
            "同位置多壁厚",
            output(
                [
                    item(
                        "SINGLE",
                        sizes=[("OD", od), ("DN", dn)],
                        thicknesses=[("MM", wall_a), ("MM", wall_b)],
                    )
                ]
            ),
        )

    # S3408 is a compact standard fragment, not a Schedule thickness.
    match = None
    if "弯头" in text:
        match = re.search(
            r"DN\s*(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s+1\.5D\s+S3408\b",
            text,
            re.I | re.S,
        )
    if match:
        dn, wall = map(_number, match.groups())
        return _actual_sample(
            row_number,
            text,
            "标准字符误识别为壁厚",
            output([item("SINGLE", sizes=[("DN", dn)], thicknesses=[("MM", wall)])]),
        )

    # A trailing '-3' in reviewed OLET descriptions is an explicit branch wall thickness.
    if "支管座" in text and re.search(r"[x×]\s*\d+\s*-\s*\d+", text, re.I):
        size_match = re.search(r"(?:DN)?\s*(\d+)\s*[x×]\s*(\d+)\s*$", text, re.I)
        wall_match = re.search(r"[x×]\s*\d+\s*-\s*(\d+)", text, re.I)
        if size_match and wall_match:
            main, branch = map(_number, size_match.groups())
            wall = _number(wall_match.group(1))
            return _actual_sample(
                row_number,
                text,
                "支管座尾部毫米壁厚",
                output(
                    [
                        item("MAIN", sizes=[("DN", main)]),
                        item("BRANCH", sizes=[("DN", branch)], thicknesses=[("MM", wall)]),
                    ]
                ),
            )

    # OCR l/I/O substitutions in an explicit branch fitting size are reviewed corrections.
    if "管接台" in text:
        ocr_match = re.search(r"DN\s*(\d+)\s*[x×]\s*[lI](\d+)", text, re.I)
        if ocr_match:
            main, branch_suffix = ocr_match.groups()
            return _actual_sample(
                row_number,
                text,
                "OCR第二端尺寸",
                output(
                    [
                        item("MAIN", sizes=[("DN", _number(main))]),
                        item(
                            "BRANCH",
                            sizes=[("DN", f"1{branch_suffix}")],
                            thicknesses=[("SCHEDULE", "STD")],
                        ),
                    ]
                ),
            )
        ocr_match = re.search(r"DN\s*(\d+)[oO]\s*[x×]\s*(\d+)", text, re.I)
        if ocr_match:
            main_prefix, branch = ocr_match.groups()
            return _actual_sample(
                row_number,
                text,
                "OCR主管尺寸",
                output(
                    [
                        item("MAIN", sizes=[("DN", f"{main_prefix}0")]),
                        item(
                            "BRANCH",
                            sizes=[("DN", _number(branch))],
                            thicknesses=[("SCHEDULE", "STD")],
                        ),
                    ]
                ),
            )

    # Explicit length and inch size are kept separate.
    if text.upper().startswith("NIPPLE"):
        length_match = re.search(r"(?:^|[;,])\s*(\d+(?:\.\d+)?)\s*mm\s*[;,]", text, re.I)
        inch_match = re.search(r"(\d+(?:\.\d+)?)\s*[\"”'']", text)
        schedule_match = re.search(r"\bSCH\s*-?\s*(\d+S?)\b", text, re.I)
        if length_match and inch_match and schedule_match:
            return _actual_sample(
                row_number,
                text,
                "长度与英制尺寸边界",
                output(
                    [
                        item(
                            "SINGLE",
                            sizes=[("INCH", _number(inch_match.group(1)))],
                            thicknesses=[("SCHEDULE", f"SCH{schedule_match.group(1).upper()}")],
                        )
                    ],
                    length=f"{_number(length_match.group(1))}MM",
                ),
            )

    if "单丝头" in text:
        match = re.search(r"DN\s*(\d+).*?\bS\s*(\d+)\b.*?\bL\s*(\d+(?:\.\d+)?)", text, re.I)
        if match:
            dn, schedule, length = map(_number, match.groups())
            return _actual_sample(
                row_number,
                text,
                "长度与壁厚等级边界",
                output(
                    [
                        item(
                            "SINGLE",
                            sizes=[("DN", dn)],
                            thicknesses=[("SCHEDULE", f"SCH{schedule}")],
                        )
                    ],
                    length=f"{length}MM",
                ),
            )

    # Long reviewed descriptions: keep OD evidence as OD and normalize only Class syntax.
    if "压力等级" in text and re.search(r"Class\s*(150|300)\b", text, re.I):
        pressure_match = re.search(r"Class\s*(150|300)\b", text, re.I)
        spec_match = re.search(r"规格\s*[:：]\s*D\s*([\d.×xX]+)", text, re.I)
        if pressure_match and spec_match:
            values = [_number(value) for value in re.split(r"[×xX]", spec_match.group(1)) if value]
            pressure = f"CL{pressure_match.group(1)}"
            if "三通" in text and len(values) in (3, 4):
                main = values[0]
                branch = values[1] if len(values) == 3 else values[2]
                wall = values[-1]
                return _actual_sample(
                    row_number,
                    text,
                    "长描述三通",
                    output(
                        [
                            item("MAIN", sizes=[("OD", main)], thicknesses=[("MM", wall)]),
                            item("BRANCH", sizes=[("OD", branch)]),
                        ],
                        pressure=pressure,
                    ),
                )
            if "异径管" in text and len(values) == 3:
                return _actual_sample(
                    row_number,
                    text,
                    "长描述异径管",
                    output(
                        [
                            item("END_A", sizes=[("OD", values[0])], thicknesses=[("MM", values[2])]),
                            item("END_B", sizes=[("OD", values[1])]),
                        ],
                        pressure=pressure,
                    ),
                )
            if len(values) == 2:
                return _actual_sample(
                    row_number,
                    text,
                    "长描述单规格管件",
                    output(
                        [
                            item(
                                "SINGLE",
                                sizes=[("OD", values[0])],
                                thicknesses=[("MM", values[1])],
                            )
                        ],
                        pressure=pressure,
                    ),
                )

    match = re.search(
        r"三通\s+SMLS\s+RFPN(\d+).*?THK\s*=\s*(\d+(?:\.\d+)?)mm"
        r"\s*[x×]\s*(\d+(?:\.\d+)?)mm\s+DN(\d+)\s*[x×]\s*(\d+)",
        text,
        re.I | re.S,
    )
    if match:
        pressure, wall_a, wall_b, main, branch = map(_number, match.groups())
        return _actual_sample(
            row_number,
            text,
            "RFPN粘连压力",
            output(
                [
                    item("MAIN", sizes=[("DN", main)], thicknesses=[("MM", wall_a)]),
                    item("BRANCH", sizes=[("DN", branch)], thicknesses=[("MM", wall_b)]),
                ],
                pressure=f"PN{pressure}",
            ),
        )

    match = re.search(
        r"法兰管.*?CL\s*(150|300).*?DN(\d+).*?S-?(\d+S?)\s+(\d+(?:\.\d+)?)mm",
        text,
        re.I | re.S,
    )
    if match:
        pressure, dn, schedule, length = map(_number, match.groups())
        return _actual_sample(
            row_number,
            text,
            "法兰管长度与压力",
            output(
                [
                    item(
                        "SINGLE",
                        sizes=[("DN", dn)],
                        thicknesses=[("SCHEDULE", f"SCH{schedule.upper()}")],
                    )
                ],
                length=f"{length}MM",
                pressure=f"CL{pressure}",
            ),
        )
    return None


def build_actual_project_error_samples(path: Path) -> tuple[list[Sample], dict[str, int]]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("读取真实项目错误Excel需要安装openpyxl") from exc

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook.active
    samples: list[Sample] = []
    stats = Counter()
    seen: set[str] = set()
    for row_number, row in enumerate(worksheet.iter_rows(min_row=2, values_only=True), start=2):
        text = str(row[0] or "").strip()
        if not text or text in seen:
            stats["空描述或重复描述"] += 1
            continue
        seen.add(text)
        sample = parse_actual_project_error(
            row_number,
            text,
            row[1],
            row[2],
            row[3],
            row[22],
            row[23],
            row[24],
        )
        if sample is None:
            stats["未纳入_标签或归属不够确定"] += 1
            continue
        samples.append(sample)
        stats[f"已纳入_{sample.category.removeprefix('真实项目错误-')}"] += 1
    return samples, dict(stats)


def fingerprint(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"×": "x", "＊": "*", "；": ";", "，": ","}))
    return re.sub(r"\s+", "", text)


def load_existing_inputs(paths: Iterable[Path]) -> tuple[set[str], int]:
    fingerprints: set[str] = set()
    count = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"现有数据集必须是数组: {path}")
        for row in data:
            text = row.get("input") if isinstance(row, dict) else None
            if isinstance(text, str):
                count += 1
                fingerprints.add(fingerprint(text))
    return fingerprints, count


def validate_output(result: dict[str, Any], text: str) -> None:
    if set(result) != {"ITEMS", "LENGTH", "PRESSURE"}:
        raise ValueError(f"输出顶层字段错误: {text}")
    if not isinstance(result["ITEMS"], list):
        raise ValueError(f"ITEMS不是数组: {text}")
    positions: set[tuple[str, str]] = set()
    for current in result["ITEMS"]:
        if set(current) != {"SCOPE", "ROLE", "SIZE", "THICKNESS"}:
            raise ValueError(f"ITEM字段错误: {text}")
        position = (current["SCOPE"], current["ROLE"])
        if current["SCOPE"] not in SCOPES or current["ROLE"] not in ROLES:
            raise ValueError(f"SCOPE/ROLE非法: {text}")
        if position in positions:
            raise ValueError(f"重复SCOPE+ROLE: {text}")
        positions.add(position)
        for key, allowed in (("SIZE", SIZE_TYPES), ("THICKNESS", THICKNESS_TYPES)):
            if not isinstance(current[key], list):
                raise ValueError(f"{key}不是数组: {text}")
            seen: set[tuple[str, str]] = set()
            for value in current[key]:
                if set(value) != {"type", "value"} or value["type"] not in allowed:
                    raise ValueError(f"{key}字段非法: {text}")
                pair = (value["type"], value["value"])
                if pair in seen:
                    raise ValueError(f"同位置重复值: {text}: {pair}")
                seen.add(pair)
    if result["LENGTH"] and not re.fullmatch(r"\d+(?:\.\d+)?MM", result["LENGTH"]):
        raise ValueError(f"LENGTH格式错误: {text}")
    if result["PRESSURE"] and not re.fullmatch(r"(?:PN|CL)\d+(?:\.\d+)?", result["PRESSURE"]):
        raise ValueError(f"PRESSURE格式错误: {text}")


def filter_and_validate_samples(
    samples: Iterable[Sample], existing: set[str]
) -> tuple[list[Sample], Counter[str]]:
    accepted: list[Sample] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for sample in samples:
        validate_output(sample.output, sample.input)
        key = fingerprint(sample.input)
        if key in existing:
            skipped["与现有V2描述重叠"] += 1
            continue
        if key in seen:
            skipped["专项集内部重复"] += 1
            continue
        if "dn" not in sample.input.casefold():
            has_dn_label = any(
                value["type"] == "DN"
                for current in sample.output["ITEMS"]
                for value in current["SIZE"]
            )
            if has_dn_label:
                raise ValueError(f"未明示DN却标注DN，触发OD转DN策略冲突: {sample.input}")
        seen.add(key)
        accepted.append(sample)
    return accepted, skipped


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    train = [sample for sample in samples if sample.source == "synthetic_contrast"]
    val = [sample for sample in samples if sample.source == "actual_project_error"]
    unknown_sources = {sample.source for sample in samples} - {
        "synthetic_contrast",
        "actual_project_error",
    }
    if unknown_sources:
        raise ValueError(f"存在未知样本来源: {sorted(unknown_sources)}")

    train.sort(key=lambda row: (row.category, row.group, row.input))
    val.sort(key=lambda row: (row.category, row.group, row.input))
    return train, val


def dataset_stats(samples: list[Sample]) -> dict[str, Any]:
    categories = Counter(sample.category for sample in samples)
    roles = Counter()
    scopes = Counter()
    size_types = Counter()
    thickness_types = Counter()
    pressure_rows = 0
    length_rows = 0
    for sample in samples:
        pressure_rows += bool(sample.output["PRESSURE"])
        length_rows += bool(sample.output["LENGTH"])
        for current in sample.output["ITEMS"]:
            roles[current["ROLE"]] += 1
            scopes[current["SCOPE"]] += 1
            size_types.update(value["type"] for value in current["SIZE"])
            thickness_types.update(value["type"] for value in current["THICKNESS"])
    return {
        "样本数": len(samples),
        "来源": dict(sorted(Counter(sample.source for sample in samples).items())),
        "对比组数": len({(sample.category, sample.group) for sample in samples}),
        "类别": dict(sorted(categories.items())),
        "ROLE": dict(sorted(roles.items())),
        "SCOPE": dict(sorted(scopes.items())),
        "尺寸类型": dict(sorted(size_types.items())),
        "壁厚类型": dict(sorted(thickness_types.items())),
        "含长度样本": length_rows,
        "含压力样本": pressure_rows,
    }


def find_existing_risk_rows(paths: Iterable[Path]) -> dict[str, int]:
    counters = Counter()
    ocr_pattern = re.compile(r"DN\s*\d+\s*[x×]\s*[lI]\d+", re.I)
    compact_pair = re.compile(r"DN\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", re.I)
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8")):
            text = row.get("input", "")
            if ocr_pattern.search(text):
                counters["疑似OCR第二端表达"] += 1
            pair_match = compact_pair.search(text)
            if pair_match and pair_match.group(1) != pair_match.group(2):
                items = row.get("output", {}).get("ITEMS") or []
                roles = {item.get("ROLE") for item in items if isinstance(item, dict)}
                if not ({"END_A", "END_B"} <= roles or {"MAIN", "BRANCH"} <= roles):
                    counters["紧凑双尺寸但未形成双位置标签"] += 1
    return dict(counters)


def generate(
    train_path: Path,
    val_path: Path,
    actual_errors_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    existing_paths = [train_path, val_path]
    existing, existing_count = load_existing_inputs(existing_paths)
    synthetic = build_samples()
    actual, actual_stats = build_actual_project_error_samples(actual_errors_path)
    # Keep a real project error in validation when a generated description happens to be identical.
    generated = actual + synthetic
    accepted, skipped = filter_and_validate_samples(generated, existing)
    train, val = split_samples(accepted)
    if not train or not val:
        raise ValueError("专项训练集或真实项目验证集为空")

    train_fingerprints = {fingerprint(sample.input) for sample in train}
    val_fingerprints = {fingerprint(sample.input) for sample in val}
    if train_fingerprints & val_fingerprints:
        raise ValueError("专项训练集与验证集存在描述重叠")
    train_groups = {(sample.category, sample.group) for sample in train}
    val_groups = {(sample.category, sample.group) for sample in val}
    if train_groups & val_groups:
        raise ValueError("同一对比组被拆到训练集和验证集")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "尺寸壁厚磅级V2专项增强_train.json"
    val_output = output_dir / "尺寸壁厚磅级V2专项增强_val.json"
    report_output = output_dir / "尺寸壁厚磅级V2专项增强_报告.json"
    train_output.write_text(
        json.dumps([sample.training_row() for sample in train], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    val_output.write_text(
        json.dumps([sample.training_row() for sample in val], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in accepted:
        if len(examples[sample.category]) < 2:
            examples[sample.category].append(sample.training_row())

    report = {
        "用途": "独立V2专项增强对比集：合成增强进入训练集，真实项目错误进入验证集",
        "现有V2样本数": existing_count,
        "候选生成数": len(generated),
        "合成增强候选数": len(synthetic),
        "真实项目错误候选数": len(actual),
        "真实项目错误解析统计": actual_stats,
        "实际生成数": len(accepted),
        "跳过统计": dict(skipped),
        "训练集": dataset_stats(train),
        "验证集": dataset_stats(val),
        "校验": {
            "与现有V2描述重叠": 0,
            "训练验证描述重叠": 0,
            "训练验证对比组重叠": 0,
            "训练集仅含合成增强": all(sample.source == "synthetic_contrast" for sample in train),
            "验证集仅含真实项目错误": all(sample.source == "actual_project_error" for sample in val),
            "样本输出仅含input_output": True,
        },
        "暂不处理": [
            "OD或INCH到DN的换算标签策略",
            "压力一阶段答案与最终C/CL编码格式差异",
            "二阶段编码与平台后处理错误",
            "设计压力6Bar等非等级压力表达（标签口径待确认）",
        ],
        "现有主数据潜在冲突提示_未自动修改": find_existing_risk_rows(existing_paths),
        "各类别代表样本": dict(examples),
    }
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_DATA_DIR / "尺寸壁厚磅级V2_train.json",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_DATA_DIR / "尺寸壁厚磅级V2_val.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--actual-errors", type=Path, default=DEFAULT_ACTUAL_ERRORS)
    args = parser.parse_args()
    report = generate(args.train, args.val, args.actual_errors, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
