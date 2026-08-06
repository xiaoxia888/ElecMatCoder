#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_OUTPUT = REPO_ROOT / "apps" / "trainer" / "qwen3_fte" / "output" / "按8类拆分数据集" / "材质规范_标准后缀专项补充.json"


@dataclass(frozen=True)
class Template:
    text: str
    material: str
    extra_standards: tuple[str, ...] = ()


B_SURFACES = (
    "SH/T 3406 (B)",
    "SH/T 3406(B)",
    "SH/T3406(B)",
    "SH/T3406B",
    "SH/T 3406B",
    "SH/T 3406 Serial B",
    "SH/T3406 Serial B",
    "SH/T 3406 Serial-B",
)

PLAIN_SURFACES = (
    "SH/T 3406",
    "SH/T3406",
    "SHT3406",
)


POSITIVE_TEMPLATES = (
    Template('FLANGE WN, CL150, RF, S30408 NB/T 47010, {std},11.91 mm', "304", ("NBT47010",)),
    Template('FLANGE WN, CL150, RF, S30408 NB/T 47010, {std},10.31 mm', "304", ("NBT47010",)),
    Template('法兰盖, CL150, RF, 20 NB/T 47008, {std}', "20", ("NBT47008",)),
    Template('对焊法兰,PN20,RF,WN,T=12mm,NB/T47010,{std},06Cr18Ni11Ti DN700', "321", ("NBT47010",)),
    Template('对焊法兰,PN20,RF,WN,T=12mm,NB/T47010,{std},022Cr17Ni12Mo2 DN700', "316L", ("NBT47010",)),
    Template('对焊法兰CL150 DN700 STD RF {std} 20#', "20"),
    Template('对焊法兰CL150 DN800×10 RF {std} 20#', "20"),
    Template('对焊法兰,PN20,RF,WN,T=14mm,NB/T47010,{std},06Cr18Ni11Ti DN800', "321", ("NBT47010",)),
    Template('对焊法兰,PN50,RF,WN,T=14mm,NB/T47010,{std},06Cr18Ni11Ti DN800', "321", ("NBT47010",)),
    Template('对焊法兰,PN20,RF,WN,T=14mm,NB/T47010,{std},06Cr18Ni11Ti DN1000', "321", ("NBT47010",)),
    Template('(1)带颈对焊法兰 DN900 WELDNECK FLANGE, CL150, STD3, A105(C≤0.3%), RF, {std}, 氩电联焊', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 24MM DN1300', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 20MM DN800', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 20MM DN1000', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 18MM DN850', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 18MM DN900', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 16MM DN750', "A105"),
    Template('WELDING NECK FLANGE CL150(PN20) RF A105 {std} - 12MM DN1100', "A105"),
    Template('WELDING NECK FLANGE CL150(PN20) RF A105 {std} - 10MM DN750', "A105"),
    Template('WELDING NECK FLANGE CL600(PN110) RF A105 {std} - 30MM DN700', "A105"),
)


PLAIN_TEMPLATES = (
    Template('FLANGE WN, CL150, RF, S30408 NB/T 47010, {std},11.91 mm', "304", ("NBT47010",)),
    Template('FLANGE WN, CL150, RF, S30408 NB/T 47010, {std},10.31 mm', "304", ("NBT47010",)),
    Template('法兰盖, CL150, RF, 20 NB/T 47008, {std}', "20", ("NBT47008",)),
    Template('对焊法兰,PN20,RF,WN,T=12mm,NB/T47010,{std},06Cr18Ni11Ti DN700', "321", ("NBT47010",)),
    Template('对焊法兰,PN20,RF,WN,T=12mm,NB/T47010,{std},022Cr17Ni12Mo2 DN700', "316L", ("NBT47010",)),
    Template('对焊法兰CL150 DN700 STD RF {std} 20#', "20"),
    Template('对焊法兰CL150 DN800×10 RF {std} 20#', "20"),
    Template('(1)带颈对焊法兰 DN900 WELDNECK FLANGE, CL150, STD3, A105(C≤0.3%), RF, {std}, 氩电联焊', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 24MM DN1300', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 20MM DN800', "A105"),
    Template('WELDING NECK FLANGE CL300(PN50) RF A105 {std} - 18MM DN850', "A105"),
    Template('WELDING NECK FLANGE CL150(PN20) RF A105 {std} - 12MM DN1100', "A105"),
    Template('WELDING NECK FLANGE CL150(PN20) RF A105 {std} - 10MM DN750', "A105"),
    Template('WELDING NECK FLANGE CL600(PN110) RF A105 {std} - 30MM DN700', "A105"),
    Template('BLIND FLANGE CL300(PN50) RF A105 {std} - DN750', "A105"),
    Template('BLIND FLANGE CL300(PN50) RF A105 {std} - DN1200', "A105"),
    Template('大口径法兰 DN1300-14mm CL300(PN5.0) RF 20II {std} DN1300', "20"),
    Template('WELDING NECK FLANGE CL300 RF A105 {std} DN900', "A105"),
    Template('BLIND FLANGE CL150 RF 20 {std} DN700', "20"),
    Template('FLANGE WN RF PN20 20 {std} DN800', "20"),
)


MANUAL_RECORDS = (
    {
        "input": "对焊法兰,PN20,RF,WN,T=12mm,NB/T47010,SH/T3406-B06Cr18Ni11Ti DN700",
        "output": {
            "MATERIAL": [{"ROLE": "MAIN", "VALUE": "321"}],
            "STANDARD": [{"BODY": "SHT3406B"}, {"BODY": "NBT47010"}],
        },
    },
    {
        "input": "对焊法兰,PN20,RF,WN,T=12mm,NB/T47010,SH/T3406-B022Cr17Ni12Mo2 DN700",
        "output": {
            "MATERIAL": [{"ROLE": "MAIN", "VALUE": "316L"}],
            "STANDARD": [{"BODY": "SHT3406B"}, {"BODY": "NBT47010"}],
        },
    },
    {
        "input": "大口径法兰 DN1300-14mm CL300(PN5.0) RF 20II SH/T 3406 Serial-B DN1300",
        "output": {
            "MATERIAL": [{"ROLE": "MAIN", "VALUE": "20"}],
            "STANDARD": [{"BODY": "SHT3406B"}],
        },
    },
    {
        "input": "FLANGE WN, CL150, RF, S30408 NB/T 47010, SH/T 3406 (B),11.91 mm",
        "output": {
            "MATERIAL": [{"ROLE": "MAIN", "VALUE": "304"}],
            "STANDARD": [{"BODY": "SHT3406B"}, {"BODY": "NBT47010"}],
        },
    },
    {
        "input": "FLANGE WN, CL150, RF, S30408 NB/T 47010, SH/T 3406,11.91 mm",
        "output": {
            "MATERIAL": [{"ROLE": "MAIN", "VALUE": "304"}],
            "STANDARD": [{"BODY": "SHT3406"}, {"BODY": "NBT47010"}],
        },
    },
)


def build_output(material: str, standards: list[str]) -> dict:
    return {
        "MATERIAL": [{"ROLE": "MAIN", "VALUE": material}],
        "STANDARD": [{"BODY": body} for body in standards],
    }


def expand_templates(
    templates: tuple[Template, ...],
    surfaces: tuple[str, ...],
    std_body: str,
) -> list[dict]:
    records: list[dict] = []
    for template in templates:
        for surface in surfaces:
            standards = [std_body, *template.extra_standards]
            records.append(
                {
                    "input": template.text.format(std=surface),
                    "output": build_output(template.material, standards),
                }
            )
    return records


def dedupe_records(records: list[dict]) -> list[dict]:
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
    parser = argparse.ArgumentParser(description="生成 SH/T 3406 B 后缀专项补充训练样本")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    records: list[dict] = []
    records.extend(expand_templates(POSITIVE_TEMPLATES, B_SURFACES, "SHT3406B"))
    records.extend(expand_templates(PLAIN_TEMPLATES, PLAIN_SURFACES, "SHT3406"))
    records.extend(MANUAL_RECORDS)
    records = dedupe_records(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已生成: {output_path}")
    print(f"样本数: {len(records)}")
    positive_count = len(POSITIVE_TEMPLATES) * len(B_SURFACES)
    plain_count = len(PLAIN_TEMPLATES) * len(PLAIN_SURFACES)
    print(f"B 后缀样本: {positive_count}")
    print(f"无后缀对照样本: {plain_count}")
    print(f"人工补充样本: {len(MANUAL_RECORDS)}")


if __name__ == "__main__":
    main()
