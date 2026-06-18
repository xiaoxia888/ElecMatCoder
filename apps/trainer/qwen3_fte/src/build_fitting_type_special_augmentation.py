#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/fitting_body_project_sampling"
INPUT_DATASET = BASE / "管件BODY训练草稿.json"
AUG_PATH = BASE / "管件BODY专项增强草稿_带标识.json"
MERGED_META_PATH = BASE / "管件BODY训练草稿_专项增强合并_带标识.json"
MERGED_TRAIN_PATH = BASE / "管件BODY训练草稿_专项增强合并.json"
SUMMARY_PATH = BASE / "管件BODY专项增强统计.json"


def make_output(
    *,
    body: str,
    angle: str = "",
    radius: str = "",
    manu: list[str] | None = None,
    conn: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "TYPE": {
            "BODY": body,
            "GEOMETRY": {"ANGLE": angle, "RADIUS": radius},
            "MANU": manu or [],
            "CONN": conn or [],
        }
    }


def rec(input_text: str, output: dict[str, Any], reason: str, family: str) -> dict[str, Any]:
    return {
        "input": input_text,
        "output": output,
        "_source": "fitting_type_augmentation",
        "_aug_tag": "AUG_FITTING_TYPE",
        "_reason": reason,
        "_family": family,
    }


def coupling_pattern_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    materials = [("20", "NB/T47008"), ("S30408", "NB/T47010"), ("S31603", "NB/T47010")]
    equal_sizes = ["DN15xDN15", "DN20xDN20", "DN25xDN25", "DN40xDN40"]
    reducing_sizes = ["DN25xDN15", "DN40xDN25", "DN50xDN20", "DN80xDN40"]
    conn_variants = ["SW", "FNPT", "MNPT", "FTE"]

    plain_equal_bodies = [
        "同心等径管箍",
        "偏心等径管箍",
    ]
    plain_generic_bodies = [
        "同心管箍",
        "偏心管箍",
    ]
    plain_reducing_bodies = [
        "同心异径管箍",
        "偏心异径管箍",
    ]
    double_equal_bodies = [
        "同心双口管箍",
        "偏心双口管箍",
    ]
    double_reducing_bodies = [
        "同心异径双口管箍",
        "偏心异径双口管箍",
    ]

    for body in plain_generic_bodies:
        for idx, size in enumerate(equal_sizes):
            mat, std = materials[idx % len(materials)]
            conn = conn_variants[idx % len(conn_variants)]
            desc = f"{body} {size} CL3000 {conn} SH/T 3410 {mat} {std}"
            rows.append(rec(desc, make_output(body=body, conn=[conn]), "COUPLING_PATTERN", "COUPLING_GENERIC"))

    for body in plain_equal_bodies:
        for idx, size in enumerate(equal_sizes):
            mat, std = materials[idx % len(materials)]
            conn = conn_variants[(idx + 1) % len(conn_variants)]
            desc = f"{body} {size} CL3000 {conn} SH/T 3410 {mat} {std}"
            rows.append(rec(desc, make_output(body=body, conn=[conn]), "COUPLING_PATTERN", "COUPLING_EQUAL"))

    for body in plain_reducing_bodies:
        for idx, size in enumerate(reducing_sizes):
            mat, std = materials[idx % len(materials)]
            conn = conn_variants[(idx + 2) % len(conn_variants)]
            desc = f"{body} {size} CL3000 {conn} SH/T 3410 {mat} {std}"
            rows.append(rec(desc, make_output(body=body, conn=[conn]), "COUPLING_PATTERN", "COUPLING_REDUCING"))

    for body in double_equal_bodies:
        for idx, size in enumerate(equal_sizes):
            mat, std = materials[idx % len(materials)]
            conn = conn_variants[idx % len(conn_variants)]
            base = "双口管箍(同心)" if body == "同心双口管箍" else "双口管箍(偏心)"
            desc = f"{base} {size} CL3000 {conn} SH/T 3410 {mat} {std}"
            rows.append(rec(desc, make_output(body=body, conn=[conn]), "COUPLING_PATTERN", "DOUBLE_EQUAL"))

    for body in double_reducing_bodies:
        for idx, size in enumerate(reducing_sizes):
            mat, std = materials[idx % len(materials)]
            conn = conn_variants[(idx + 1) % len(conn_variants)]
            desc = f"{body} {size} CL3000 {conn} SH/T 3410 {mat} {std}"
            rows.append(rec(desc, make_output(body=body, conn=[conn]), "COUPLING_PATTERN", "DOUBLE_REDUCING"))

    return rows


def manu_boundary_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        # 有缝不等于 WELDED
        ("有缝弯头\\45° DN850×12 R=1.5D SH/T 3408 Q245R GB/T 713 100%RT", make_output(body="弯头", angle="45", radius="1.5D")),
        ("有缝弯头\\90° DN600×10 R=1.5D SH/T 3408 Q245R GB/T 713 100%RT", make_output(body="弯头", angle="90", radius="1.5D")),
        ("有缝弯头 45° DN300 SCH10S R=1.5D SH/T3408 022Cr17Ni12Mo2 GB/T 4237", make_output(body="弯头", angle="45", radius="1.5D")),
        ("有缝弯头 90° DN500×8 R=LR SH/T3408 06Cr19Ni10 GB/T 4237", make_output(body="弯头", angle="90", radius="LR")),
        ("有缝异径三通 DN700×DN400×8/8 SH/T3408 Q235B", make_output(body="异径三通")),
        ("有缝等径三通 DN350×DN350 SCH10S/SCH10S SH/T3408 022Cr17Ni12Mo2 GB/T 4237", make_output(body="等径三通")),
        ("有缝同心异径管 DN500×DN300 SCH20 SH/T3408 Q245R", make_output(body="同心异径管")),
        ("有缝偏心异径管 DN500×DN300 SCH20 SH/T3408 Q245R", make_output(body="偏心异径管")),
        # 锻制不等于 WELDED
        ("锻制三通\\DN15×DN15 CL6000 SW ASME B16.11 20# NB/T 47008", make_output(body="等径三通", conn=["SW"])),
        ("锻制三通\\DN25×DN15 CL3000 SW ASME B16.11 20# NB/T 47008", make_output(body="异径三通", conn=["SW"])),
        ("锻制弯头\\90° DN40 CL3000 SW GB/T 14383 20# NB/T 47008", make_output(body="弯头", angle="90", conn=["SW"])),
        ("锻制弯头\\45° DN25 CL3000 FNPT GB/T 14383 20# NB/T 47008", make_output(body="弯头", angle="45", conn=["FNPT"])),
        ("锻制弯头\\90° DN50 CL3000 MNPT GB/T 14383 20# NB/T 47008", make_output(body="弯头", angle="90", conn=["MNPT"])),
        ("锻制等径三通 DN20 CL3000 FNPT GB/T14383(I) 20 NB/T47008", make_output(body="等径三通", conn=["FNPT"])),
        ("锻制异径三通 DN25×DN15 CL3000 MNPT GB/T14383(I) 20 NB/T47008", make_output(body="异径三通", conn=["MNPT"])),
        ("锻制同心异径管接头 DN40X25 CL3000 SW SH/T3410 S30403 NB/T47010", make_output(body="同心异径管接头", conn=["SW"])),
        ("锻制偏心异径管接头 DN40X25 CL3000 FNPT SH/T3410 S30403 NB/T47010", make_output(body="偏心异径管接头", conn=["FNPT"])),
    ]
    for desc, output in cases:
        rows.append(rec(desc, output, "MANU_BOUNDARY", "MANU_BOUNDARY"))
    return rows


def forged_not_welded_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        # 专门补强：锻制承插/螺纹管件不要漂到 WELDED
        ("锻制三通\\DN40×DN40 SCH160/SCH160 SW SH/T 3410 20# NB/T 47008", make_output(body="三通", conn=["SW"])),
        ("锻制三通\\DN50×DN50 SCH80/SCH80 SW SH/T 3410 20# NB/T 47008", make_output(body="三通", conn=["SW"])),
        ("锻制三通\\DN25×DN25 SCH40S/SCH40S SW SH/T 3410 S30408 NB/T 47010", make_output(body="三通", conn=["SW"])),
        ("锻制三通\\DN20×DN20 SCH80S/SCH80S SW SH/T 3410 S31603 NB/T 47010", make_output(body="三通", conn=["SW"])),
        ("锻制三通\\DN40×DN25 SCH160/SCH160 SW SH/T 3410 20# NB/T 47008", make_output(body="三通", conn=["SW"])),
        ("锻制三通\\DN50×DN25 SCH80/SCH40 SW SH/T 3410 20# NB/T 47008", make_output(body="三通", conn=["SW"])),
        ("锻制三通\\DN40×DN20 SCH80S/SCH40S SW SH/T 3410 S30408 NB/T 47010", make_output(body="三通", conn=["SW"])),
        ("锻制三通\\DN25×DN15 SCH40S/SCH40S SW SH/T 3410 S31603 NB/T 47010", make_output(body="三通", conn=["SW"])),
        ("锻制等径三通 DN40 CL3000 SW SH/T3410 20 NB/T47008", make_output(body="三通", conn=["SW"])),
        ("锻制等径三通 DN25 CL6000 SW SH/T3410 S30408 NB/T47010", make_output(body="三通", conn=["SW"])),
        ("锻制异径三通 DN40×DN20 CL3000 SW SH/T3410 20 NB/T47008", make_output(body="三通", conn=["SW"])),
        ("锻制异径三通 DN50×DN25 CL6000 SW SH/T3410 S30403 NB/T47010", make_output(body="三通", conn=["SW"])),
        ("锻制等径三通 DN40 CL3000 FNPT SH/T3410 20 NB/T47008", make_output(body="三通", conn=["FNPT"])),
        ("锻制异径三通 DN40×DN20 CL3000 MNPT SH/T3410 20 NB/T47008", make_output(body="三通", conn=["MNPT"])),
        ("锻制等径三通 DN25 CL3000 FTE SH/T3410 S30408 NB/T47010", make_output(body="三通", conn=["FTE"])),
        ("锻制异径三通 DN40×DN25 CL3000 FNPT SH/T3410 S31603 NB/T47010", make_output(body="三通", conn=["FNPT"])),
        # 混合中文/英文表达，避免模型只记住少数模板
        ("FORGED STRAIGHT TEE SW SH/T3410 NB/T47008 20 DN40x40 SCH160xSCH160", make_output(body="三通", conn=["SW"])),
        ("FORGED REDUCING TEE SW SH/T3410 NB/T47008 20 DN40x25 SCH160xSCH160", make_output(body="三通", conn=["SW"])),
        ("STRAIGHT TEE FORGED SW SH/T3410 NB/T47010 S30408 DN25x25 SCH40SxSCH40S", make_output(body="三通", conn=["SW"])),
        ("REDUCING TEE FORGED FNPT SH/T3410 NB/T47010 S31603 DN40x20 CL3000", make_output(body="三通", conn=["FNPT"])),
        # 弯头一并补强，避免“锻制+SW/FNPT”泛化到 WELDED
        ("锻制弯头\\90° DN40 SCH160 SW SH/T 3410 20# NB/T 47008", make_output(body="弯头", angle="90", conn=["SW"])),
        ("锻制弯头\\45° DN25 SCH40S FNPT SH/T 3410 S30408 NB/T 47010", make_output(body="弯头", angle="45", conn=["FNPT"])),
        ("90° ELBOW FORGED SW SH/T3410 NB/T47008 20 DN50 SCH160", make_output(body="弯头", angle="90", conn=["SW"])),
        ("45° ELBOW FORGED MNPT SH/T3410 NB/T47010 S31603 DN20 CL3000", make_output(body="弯头", angle="45", conn=["MNPT"])),
    ]
    for desc, output in cases:
        rows.append(rec(desc, output, "FORGED_NOT_WELDED", "FORGED_NOT_WELDED"))
    return rows


def strip_meta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"input": r["input"], "output": r["output"]} for r in rows]


def dedupe(rows: list[dict[str, Any]], existing_inputs: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set(existing_inputs)
    for r in rows:
        inp = r["input"].strip()
        if inp in seen:
            continue
        seen.add(inp)
        out.append(r)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    body = Counter()
    conn = Counter()
    reason = Counter()
    family = Counter()
    for r in rows:
        t = r["output"]["TYPE"]
        body[t["BODY"]] += 1
        for c in t.get("CONN", []):
            conn[c] += 1
        reason[r.get("_reason", "")] += 1
        family[r.get("_family", "")] += 1
    return {
        "count": len(rows),
        "body": dict(body),
        "conn": dict(conn),
        "reason": dict(reason),
        "family": dict(family),
    }


def main() -> None:
    base = json.loads(INPUT_DATASET.read_text(encoding="utf-8"))
    existing_inputs = {item["input"].strip() for item in base}
    aug = coupling_pattern_records() + manu_boundary_records() + forged_not_welded_records()
    aug = dedupe(aug, existing_inputs)
    merged_meta = base + aug
    merged_train = strip_meta(base) + strip_meta(aug)

    AUG_PATH.write_text(json.dumps(aug, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MERGED_META_PATH.write_text(json.dumps(merged_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MERGED_TRAIN_PATH.write_text(json.dumps(merged_train, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "base_count": len(base),
        "aug_summary": summarize(aug),
        "merged_count": len(merged_train),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
