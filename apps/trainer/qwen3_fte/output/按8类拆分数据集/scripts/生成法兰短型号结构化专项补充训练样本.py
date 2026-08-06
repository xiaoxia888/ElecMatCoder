#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "output"
    / "按8类拆分数据集"
    / "尺寸壁厚磅级"
    / "法兰短型号结构化专项补充.json"
)


CL_SUFFIXES = {"150", "300", "400", "600", "900", "1500", "2500", "3000"}
PN_SUFFIXES = {"10", "16", "20", "25", "40", "50", "63", "100", "110", "160"}

THICKNESS_BY_SIZE = {
    "15": "4",
    "20": "4",
    "25": "4.5",
    "40": "4",
    "50": "4.5",
    "80": "5.5",
    "100": "6",
    "250": "8",
    "500": "10",
}


@dataclass(frozen=True)
class Combo:
    size: str
    suffix: str


@dataclass(frozen=True)
class Template:
    text: str
    has_thickness: bool = False


COMBOS = (
    Combo("15", "150"),
    Combo("20", "16"),
    Combo("25", "20"),
    Combo("40", "150"),
    Combo("50", "300"),
    Combo("80", "150"),
    Combo("80", "50"),
    Combo("100", "600"),
    Combo("100", "110"),
    Combo("250", "150"),
    Combo("500", "150"),
)


CHINESE_TEMPLATES = (
    Template("突面钢制管法兰盖:{size}-{suffix} RF,16Mn,NB/T47008 HG/T20615-2009 {size}-{suffix} RF"),
    Template("突面钢制管法兰:{size}-{suffix} RF,16Mn,NB/T47008 HG/T20615-2009 {size}-{suffix} RF"),
    Template("突面对焊钢制管法兰:WN {size}-{suffix} RF {thk}mm,16Mn,NB/T47008 HG/T20615-2009 WN {size}-{suffix} RF {thk}mm", True),
    Template("突面承插焊钢制管法兰:SW {size}-{suffix} RF {thk}mm,16Mn,NB/T47008 HG/T20615-2009 SW {size}-{suffix} RF {thk}mm", True),
    Template("突面平焊钢制管法兰:SO {size}-{suffix} RF,20,HG/T20615-2009 SO {size}-{suffix} RF"),
    Template("盲法兰 BF {size}-{suffix} RF,16Mn,HG/T20615-2009 BF {size}-{suffix} RF"),
    Template("法兰盖 DN{size} {suffix} RF 16Mn HG/T20615-2009"),
)


ENGLISH_TEMPLATES = (
    Template("WELD NECK FLANGE WN {size}-{suffix} RF {thk}MM A105 HG/T20615-2009", True),
    Template("SOCKET WELD FLANGE SW {size}-{suffix} RF A105 HG/T20615-2009"),
    Template("SLIP ON FLANGE SO {size}-{suffix} RF A105 HG/T20615-2009"),
    Template("BLIND FLANGE {size}-{suffix} RF A105 HG/T20615-2009"),
    Template("FLANGE WN {size}-{suffix} RF {thk}mm 16Mn NB/T47008 HG/T20615-2009", True),
    Template("FLANGE SW {size}-{suffix} RF S30408 HG/T20615-2009"),
)


STICKY_TEMPLATES = (
    Template("WN{size}-{suffix}RF{thk}mm16MnNBT47008HGT20615-2009", True),
    Template("SW{size}-{suffix}RF16MnNBT47008HGT20615-2009"),
    Template("BF{size}-{suffix}RF20HGT20615-2009"),
    Template("FLANGEWN{size}-{suffix}RFA105HGT20615-2009"),
    Template("法兰盖{size}-{suffix}RF16MnNBT47008HGT20615-2009"),
)


MANUAL_SEEDS = (
    {
        "input": "突面钢制管法兰盖:80-150 RF,16Mn,NB/T47008 HG/T20615-2009 80-150 RF",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "80"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "CL150",
        },
    },
    {
        "input": "突面钢制管法兰:80-150 RF,16Mn,NB/T47008 HG/T20615-2009 80-150 RF",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "80"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "CL150",
        },
    },
    {
        "input": "突面钢制管法兰盖:WN 40-150 RF,16Mn,NB/T47008 HG/T20615-2009 40-150 RF",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "40"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "CL150",
        },
    },
    {
        "input": "突面钢制管法兰盖:80-16 RF,16Mn,NB/T47008 HG/T20615-2009 80-16 RF",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "80"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "PN16",
        },
    },
    {
        "input": "突面钢制管法兰:80-50 RF,16Mn,NB/T47008 HG/T20615-2009 80-50 RF",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "80"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "PN50",
        },
    },
    {
        "input": "WELD NECK FLANGE WN 40-150 RF A105 HG/T20615-2009",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "40"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "CL150",
        },
    },
    {
        "input": "SOCKET WELD FLANGE SW 25-20 RF A105 HG/T20615-2009",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "25"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "PN20",
        },
    },
    {
        "input": "BLIND FLANGE 100-600 RF A105 HG/T20615-2009",
        "output": {
            "SIZE_ITEMS": [{"type": "DN", "value": "100"}],
            "LENGTH": "",
            "THICKNESS_ITEMS": [],
            "PRESSURE": "CL600",
        },
    },
)


def build_pressure(suffix: str) -> str:
    if suffix in CL_SUFFIXES:
        return f"CL{suffix}"
    if suffix in PN_SUFFIXES:
        return f"PN{suffix}"
    raise ValueError(f"未识别的法兰短型号后缀: {suffix}")


def build_output(combo: Combo, has_thickness: bool) -> dict:
    output = {
        "SIZE_ITEMS": [{"type": "DN", "value": combo.size}],
        "LENGTH": "",
        "THICKNESS_ITEMS": [],
        "PRESSURE": build_pressure(combo.suffix),
    }
    if has_thickness:
        output["THICKNESS_ITEMS"] = [
            {"type": "MM", "value": THICKNESS_BY_SIZE[combo.size]}
        ]
    return output


def render_template(template: Template, combo: Combo) -> dict:
    thk = THICKNESS_BY_SIZE[combo.size]
    return {
        "input": template.text.format(size=combo.size, suffix=combo.suffix, thk=thk),
        "output": build_output(combo, template.has_thickness),
    }


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for record in records:
        key = record["input"].strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成法兰短型号结构化专项补充训练样本")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出 JSON 路径")
    args = parser.parse_args()

    records: list[dict] = []
    for combo in COMBOS:
        for template in CHINESE_TEMPLATES:
            records.append(render_template(template, combo))
        for template in ENGLISH_TEMPLATES:
            records.append(render_template(template, combo))
        for template in STICKY_TEMPLATES:
            records.append(render_template(template, combo))

    records.extend(MANUAL_SEEDS)
    records = dedupe(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with_thickness = sum(1 for x in records if x["output"]["THICKNESS_ITEMS"])
    cl_count = sum(1 for x in records if str(x["output"]["PRESSURE"]).startswith("CL"))
    pn_count = sum(1 for x in records if str(x["output"]["PRESSURE"]).startswith("PN"))

    print(f"已生成: {output_path}")
    print(f"样本数: {len(records)}")
    print(f"CL 体系样本数: {cl_count}")
    print(f"PN 体系样本数: {pn_count}")
    print(f"带壁厚样本数: {with_thickness}")


if __name__ == "__main__":
    main()
