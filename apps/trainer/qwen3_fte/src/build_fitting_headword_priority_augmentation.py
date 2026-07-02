#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/管件_主词优先专项增强.json"
)


def make_type(
    *,
    body: str,
    angle: str = "",
    radius: str = "",
    conn: list[str] | None = None,
    manu: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": body,
            "GEOMETRY": {"ANGLE": angle, "RADIUS": radius},
            "FLANGE_STYLE": "",
            "SEAL": [],
            "CONN": conn or [],
            "MANU": manu or [],
        },
    }


def rec(
    input_text: str,
    *,
    body: str,
    angle: str = "",
    radius: str = "",
    conn: list[str] | None = None,
    manu: list[str] | None = None,
    family: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "input": input_text,
        "output": make_type(
            body=body,
            angle=angle,
            radius=radius,
            conn=conn,
            manu=manu,
        ),
        "_source": "fitting_headword_priority_augmentation",
        "_family": family,
        "_reason": reason,
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row["input"].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_cap_records() -> list[dict[str, Any]]:
    return [
        rec("Cap,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="HEADWORD_CORE"),
        rec("CAP,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="HEADWORD_CASE"),
        rec("End Cap,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="HEADWORD_ALIAS"),
        rec("Pipe Cap,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="HEADWORD_ALIAS"),
        rec("Cap,Oval,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="RARE_MODIFIER"),
        rec("CAP,OVAL,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="RARE_MODIFIER"),
        rec("Oval Cap,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="RARE_MODIFIER"),
        rec("Elliptical Cap,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="RARE_MODIFIER"),
        rec("Cap Oval BW SMLS A234 WPB ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="ORDER_VARIANT"),
        rec("CAP OVAL BW SMLS A234 WPB ASME B16.9 STD DN600", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="ORDER_VARIANT"),
        rec("不锈钢管帽 SMLS BW A403 WP304L ASME B16.9 SCH40S DN150", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="CN_EN_MIXED"),
        rec("管帽 Oval SMLS BW A234 WPB ASME B16.9 STD DN300", body="管帽", conn=["BW"], manu=["SMLS"], family="CAP", reason="CN_EN_MIXED"),
    ]


def build_elbow_records() -> list[dict[str, Any]]:
    return [
        rec("90 Elbow,LR,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="弯头", angle="90", radius="LR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="HEADWORD_CORE"),
        rec("90 ELBOW LR SMLS BW A234 WPB ASME B16.9 STD DN600", body="弯头", angle="90", radius="LR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="HEADWORD_CASE"),
        rec("45 Elbow,LR,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", body="弯头", angle="45", radius="LR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="ANGLE_VARIANT"),
        rec("90deg Elbow,SR,SMLS,BW,A234 WPB,ASME B16.9 STD DN300", body="弯头", angle="90", radius="SR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="ANGLE_VARIANT"),
        rec("Long Radius Elbow 90 SMLS BW A234 WPB ASME B16.9 DN200", body="弯头", angle="90", radius="LR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="RADIUS_ALIAS"),
        rec("Short Radius Elbow 90 SMLS BW A234 WPB ASME B16.9 DN200", body="弯头", angle="90", radius="SR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="RADIUS_ALIAS"),
        rec("90EL LR SMLS BW A234 WPB ASME B16.9 DN200", body="弯头", angle="90", radius="LR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="ABBR_VARIANT"),
        rec("45EL LR SMLS BW A234 WPB ASME B16.9 DN200", body="弯头", angle="45", radius="LR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="ABBR_VARIANT"),
        rec("弯头 90度 LR SMLS BW A234 WPB ASME B16.9 DN200", body="弯头", angle="90", radius="LR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="CN_EN_MIXED"),
        rec("弯头 45度 SR SMLS BW A234 WPB ASME B16.9 DN150", body="弯头", angle="45", radius="SR", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="CN_EN_MIXED"),
        rec("Elbow 90 BW WELDED A234 WPB ASME B16.9 SCH40 DN250", body="弯头", angle="90", conn=["BW"], manu=["WELDED"], family="ELBOW", reason="MANU_VARIANT"),
        rec("Elbow 45 BW SMLS A420 WPL6 ASME B16.9 SCH80 DN250", body="弯头", angle="45", conn=["BW"], manu=["SMLS"], family="ELBOW", reason="MATERIAL_VARIANT"),
    ]


def build_tee_records() -> list[dict[str, Any]]:
    return [
        rec("Equal Tee,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x600", body="等径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="HEADWORD_CORE"),
        rec("Equal Tee BW SMLS A234 WPB ASME B16.9 STD DN600x600", body="等径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="ORDER_VARIANT"),
        rec("Eq Tee,SMLS,BW,A234 WPB,ASME B16.9 STD DN400x400", body="等径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="ABBR_VARIANT"),
        rec("Reducing Tee,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="异径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="HEADWORD_CORE"),
        rec("Red Tee,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="异径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="ABBR_VARIANT"),
        rec("TEE RED,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="异径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="ORDER_VARIANT"),
        rec("TEE,RED,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="异径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="ORDER_VARIANT"),
        rec("Tee Equal SMLS BW A234 WPB ASME B16.9 DN300x300", body="等径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="ORDER_VARIANT"),
        rec("Tee,SMLS,BW,A234 WPB,ASME B16.9 STD DN500x500", body="三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="GENERIC_TEE"),
        rec("三通 SMLS BW A234 WPB ASME B16.9 STD DN500x500", body="三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="CN_EN_MIXED"),
        rec("等径三通 SMLS BW A234 WPB ASME B16.9 STD DN250x250", body="等径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="CN_EN_MIXED"),
        rec("异径三通 SMLS BW A234 WPB ASME B16.9 STD DN250x150", body="异径三通", conn=["BW"], manu=["SMLS"], family="TEE", reason="CN_EN_MIXED"),
    ]


def build_reducer_records() -> list[dict[str, Any]]:
    return [
        rec("Concentric Reducer,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="同心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="HEADWORD_CORE"),
        rec("Conc Reducer,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="同心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="ABBR_VARIANT"),
        rec("Reducer Concentric,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="同心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="ORDER_VARIANT"),
        rec("Con Reducer,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="同心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="ABBR_VARIANT"),
        rec("Eccentric Reducer,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="偏心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="HEADWORD_CORE"),
        rec("Ecc Reducer,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="偏心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="ABBR_VARIANT"),
        rec("Reducer Eccentric,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="偏心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="ORDER_VARIANT"),
        rec("Reducer ECC.,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", body="偏心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="ABBR_VARIANT"),
        rec("异径管 同心 SMLS BW A234 WPB ASME B16.9 STD DN400x250", body="同心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="CN_EN_MIXED"),
        rec("异径管 偏心 SMLS BW A234 WPB ASME B16.9 STD DN400x250", body="偏心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="CN_EN_MIXED"),
        rec("同心异径管 SMLS BW A234 WPB ASME B16.9 STD DN300x150", body="同心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="CN_EN_MIXED"),
        rec("偏心异径管 SMLS BW A234 WPB ASME B16.9 STD DN300x150", body="偏心异径管", conn=["BW"], manu=["SMLS"], family="REDUCER", reason="CN_EN_MIXED"),
    ]


def build_adversarial_records() -> list[dict[str, Any]]:
    cases = [
        ("Cap,Oval,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", "管帽", "CAP"),
        ("Ecc Reducer,Oval,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", "偏心异径管", "REDUCER"),
        ("Con Reducer,Oval,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", "同心异径管", "REDUCER"),
        ("Equal Tee,Oval,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x600", "等径三通", "TEE"),
        ("Reducing Tee,Oval,SMLS,BW,A234 WPB,ASME B16.9 STD DN600x400", "异径三通", "TEE"),
        ("90 Elbow,Oval,SMLS,BW,A234 WPB,ASME B16.9 STD DN600", "弯头", "ELBOW"),
        ("End Cap,SMLS,BW,A420 WPL6,ASME B16.9 SCH80 DN500", "管帽", "CAP"),
        ("Ecc Reducer,SMLS,BW,A420 WPL6,ASME B16.9 SCH80 DN500x300", "偏心异径管", "REDUCER"),
        ("Con Reducer,SMLS,BW,A420 WPL6,ASME B16.9 SCH80 DN500x300", "同心异径管", "REDUCER"),
        ("Tee,SMLS,BW,A420 WPL6,ASME B16.9 SCH80 DN500x500", "三通", "TEE"),
        ("Equal Tee,SMLS,BW,A420 WPL6,ASME B16.9 SCH80 DN500x500", "等径三通", "TEE"),
        ("Reducing Tee,SMLS,BW,A420 WPL6,ASME B16.9 SCH80 DN500x300", "异径三通", "TEE"),
    ]
    rows: list[dict[str, Any]] = []
    for text, body, family in cases:
        conn = ["BW"]
        manu = ["SMLS"]
        angle = "90" if body == "弯头" else ""
        rows.append(
            rec(
                text,
                body=body,
                angle=angle,
                conn=conn,
                manu=manu,
                family=f"{family}_ADVERSARIAL",
                reason="HEADWORD_PRIORITY",
            )
        )
    return rows


def build_olet_records() -> list[dict[str, Any]]:
    return [
        rec('Olet 4"*0.75" STD*XS Olet,4"*0.75",STD*XS CL3000,SW,A105,MSS SP-97', body="支管台", family="OLET", reason="GENERIC_OLET"),
        rec('Olet 8"*1" STD Olet,8"*1",STD*STD BW,A105,MSS SP-97', body="支管台", family="OLET", reason="GENERIC_OLET"),
        rec('Olet 10"*0.5" STD Olet,10"*0.5",STD*STD BW,A182 F304/304L,MSS SP-97', body="支管台", family="OLET", reason="GENERIC_OLET"),
        rec('Olet 6"*2" STD Olet,6"*2",STD*STD BW,A105,MSS SP-97', body="支管台", family="OLET", reason="GENERIC_OLET"),
        rec('Olet 12"*4" STD Olet,12"*4",STD*STD FNPT,A105,MSS SP-97', body="支管台", family="OLET", reason="GENERIC_OLET"),
        rec('Generic Olet BW MSS SP-97 A105 8"*1" STD*STD', body="支管台", family="OLET", reason="GENERIC_OLET"),
        rec('OLET MSS SP-97 A105 BW STD x STD 8"*1"', body="支管台", family="OLET", reason="GENERIC_OLET"),
        rec('Weldolet 8"*1" STD Weldolet,8"*1",STD*STD BW,A105,MSS SP-97', body="对焊支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('WELDOLET MSS SP-97 A105 BW STD x STD 8"*1"', body="对焊支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('Weld Olet MSS SP-97 A105 BW STD x STD 8"*1"', body="对焊支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('Welding Outlet A105 BW MSS SP-97 8"*1" STD x STD', body="对焊支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('Sockolet 4"*0.75" STD*XS Sockolet,4"*0.75",STD*XS CL3000,SW,A105,MSS SP-97', body="承插焊支管台", conn=["SW"], family="OLET", reason="SUBTYPE_OLET"),
        rec('SOCKOLET MSS SP-97 A105 SW CL3000 4"*0.75"', body="承插焊支管台", conn=["SW"], family="OLET", reason="SUBTYPE_OLET"),
        rec('Socket Olet MSS SP-97 A105 SW CL3000 4"*0.75"', body="承插焊支管台", conn=["SW"], family="OLET", reason="SUBTYPE_OLET"),
        rec('Thredolet 4"*0.75" STD*XS Thredolet,4"*0.75",STD*XS CL3000,FNPT,A105,MSS SP-97', body="螺纹支管台", conn=["FNPT"], family="OLET", reason="SUBTYPE_OLET"),
        rec('THREDOLET MSS SP-97 A105 FNPT CL3000 4"*0.75"', body="螺纹支管台", conn=["FNPT"], family="OLET", reason="SUBTYPE_OLET"),
        rec('Thread Olet MSS SP-97 A105 FNPT CL3000 4"*0.75"', body="螺纹支管台", conn=["FNPT"], family="OLET", reason="SUBTYPE_OLET"),
        rec('Threadolet MSS SP-97 A105 MNPT CL3000 6"*1"', body="螺纹支管台", conn=["MNPT"], family="OLET", reason="SUBTYPE_OLET"),
        rec('Latrolet 8"*1" STD Latrolet,8"*1",STD*STD BW,A105,MSS SP-97', body="斜支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('LATROLET MSS SP-97 A105 BW STD x STD 8"*1"', body="斜支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('Sweepolet 10"*2" STD Sweepolet,10"*2",STD*STD BW,A105,MSS SP-97', body="支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('SWEEPOLET MSS SP-97 A105 BW STD x STD 10"*2"', body="支管台", family="OLET", reason="SUBTYPE_OLET"),
        rec('Olet 8"*1" STD Olet,8"*1",STD*STD BW,A105,MSS SP-97', body="支管台", family="OLET_ADVERSARIAL", reason="HEADWORD_PRIORITY"),
        rec('Elbow 8" STD Elbow,90 DEG,LR,A105,ASME B16.9', body="弯头", angle="90", radius="LR", conn=["BW"], family="OLET_ADVERSARIAL", reason="HEADWORD_PRIORITY"),
        rec('Cap 8" STD Cap,SMLS,BW,A234 WPB,ASME B16.9', body="管帽", conn=["BW"], manu=["SMLS"], family="OLET_ADVERSARIAL", reason="HEADWORD_PRIORITY"),
    ]


def build_systematic_cap_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aliases = ["Cap", "CAP", "End Cap", "Pipe Cap", "Oval Cap", "Elliptical Cap"]
    materials = ["A234 WPB", "A420 WPL6", "A403 WP304L", "A403 WP316L"]
    standards = ["ASME B16.9", "GB/T 12459"]
    sizes = [("DN50", "STD"), ("DN150", "SCH40"), ("DN300", "XS")]
    for alias in aliases:
        for material in materials:
            for standard in standards:
                for size, thk in sizes:
                    text = f"{alias},SMLS,BW,{material},{standard} {thk} {size}"
                    rows.append(rec(text, body="管帽", conn=["BW"], manu=["SMLS"], family="CAP_SYSTEMATIC", reason="HEADWORD_PRIORITY"))
    return rows


def build_systematic_elbow_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    heads = [
        ("90 Elbow", "90", "LR"),
        ("45 Elbow", "45", "LR"),
        ("90EL", "90", "LR"),
        ("45EL", "45", "LR"),
        ("Long Radius Elbow 90", "90", "LR"),
        ("Short Radius Elbow 90", "90", "SR"),
    ]
    materials = ["A234 WPB", "A420 WPL6", "A403 WP304L"]
    sizes = [("DN50", "STD"), ("DN150", "SCH40"), ("DN300", "XS")]
    for head, angle, radius in heads:
        for material in materials:
            for size, thk in sizes:
                text = f"{head},SMLS,BW,{material},ASME B16.9 {thk} {size}"
                rows.append(rec(text, body="弯头", angle=angle, radius=radius, conn=["BW"], manu=["SMLS"], family="ELBOW_SYSTEMATIC", reason="HEADWORD_PRIORITY"))
    return rows


def build_systematic_tee_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = [
        ("Equal Tee", "等径三通", "DN100x100"),
        ("Eq Tee", "等径三通", "DN200x200"),
        ("Tee Equal", "等径三通", "DN300x300"),
        ("Reducing Tee", "异径三通", "DN100x50"),
        ("Red Tee", "异径三通", "DN200x100"),
        ("TEE RED", "异径三通", "DN300x150"),
        ("TEE,RED", "异径三通", "DN400x200"),
        ("Tee", "三通", "DN150x150"),
    ]
    materials = ["A234 WPB", "A403 WP304L", "A403 WP316L"]
    thks = ["STD", "SCH40", "XS"]
    for head, body, size in configs:
        for material in materials:
            for thk in thks:
                text = f"{head},SMLS,BW,{material},ASME B16.9 {thk} {size}"
                rows.append(rec(text, body=body, conn=["BW"], manu=["SMLS"], family="TEE_SYSTEMATIC", reason="HEADWORD_PRIORITY"))
    return rows


def build_systematic_reducer_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = [
        ("Concentric Reducer", "同心异径管", "DN100x50"),
        ("Conc Reducer", "同心异径管", "DN200x100"),
        ("Con Reducer", "同心异径管", "DN300x150"),
        ("Eccentric Reducer", "偏心异径管", "DN100x50"),
        ("Ecc Reducer", "偏心异径管", "DN200x100"),
        ("Reducer ECC.", "偏心异径管", "DN300x150"),
    ]
    materials = ["A234 WPB", "A403 WP304L", "A403 WP316L"]
    thks = ["STD", "SCH40", "XS"]
    for head, body, size in configs:
        for material in materials:
            for thk in thks:
                text = f"{head},SMLS,BW,{material},ASME B16.9 {thk} {size}"
                rows.append(rec(text, body=body, conn=["BW"], manu=["SMLS"], family="REDUCER_SYSTEMATIC", reason="HEADWORD_PRIORITY"))
    return rows


def build_systematic_olet_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = [
        ("Olet", "支管台", [], "CL3000"),
        ("Generic Olet", "支管台", [], "CL3000"),
        ("Weldolet", "对焊支管台", [], "CL3000"),
        ("Weld Olet", "对焊支管台", [], "CL3000"),
        ("Sockolet", "承插焊支管台", ["SW"], "CL3000"),
        ("Socket Olet", "承插焊支管台", ["SW"], "CL3000"),
        ("Thredolet", "螺纹支管台", ["FNPT"], "CL3000"),
        ("Thread Olet", "螺纹支管台", ["FNPT"], "CL3000"),
        ("Latrolet", "斜支管台", [], "CL3000"),
        ("Sweepolet", "支管台", [], "CL3000"),
    ]
    materials = ["A105", "A182 F304/304L", "A182 F316/316L"]
    specs = [
        ('4"*0.75"', "STD*XS"),
        ('6"*1"', "STD*STD"),
        ('8"*1"', "STD*STD"),
        ('10"*0.5"', "XS*STD"),
    ]
    for head, body, conn, pressure in configs:
        for material in materials:
            for size, thk in specs:
                if body == "承插焊支管台":
                    text = f"{head},{size},{thk},{pressure},SW,{material},MSS SP-97"
                elif body == "螺纹支管台":
                    text = f"{head},{size},{thk},{pressure},FNPT,{material},MSS SP-97"
                else:
                    text = f"{head},{size},{thk},BW,{material},MSS SP-97"
                rows.append(rec(text, body=body, conn=conn, family="OLET_SYSTEMATIC", reason="HEADWORD_PRIORITY"))
    return rows


def main() -> None:
    rows = []
    rows.extend(build_cap_records())
    rows.extend(build_elbow_records())
    rows.extend(build_tee_records())
    rows.extend(build_reducer_records())
    rows.extend(build_adversarial_records())
    rows.extend(build_olet_records())
    rows.extend(build_systematic_cap_records())
    rows.extend(build_systematic_elbow_records())
    rows.extend(build_systematic_tee_records())
    rows.extend(build_systematic_reducer_records())
    rows.extend(build_systematic_olet_records())
    rows = dedupe_rows(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary: dict[str, int] = {}
    for row in rows:
        body = row["output"]["TYPE"]["BODY"]
        summary[body] = summary.get(body, 0) + 1
    print(f"已生成: {OUTPUT_PATH}")
    print(f"总条数: {len(rows)}")
    for body, count in sorted(summary.items(), key=lambda x: (-x[1], x[0])):
        print(f"{body}: {count}")


if __name__ == "__main__":
    main()
