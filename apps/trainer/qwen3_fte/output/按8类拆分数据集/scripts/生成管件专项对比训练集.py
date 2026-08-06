#!/usr/bin/env python3
"""Generate a review-only contrastive dataset for fitting type extraction."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
TYPE_DIR = REPO_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类"
DEFAULT_OUTPUT_DIR = TYPE_DIR / "专项增强_20260806_V31"
BASE_DATASETS = {
    "train": TYPE_DIR / "管件_train.json",
    "val": TYPE_DIR / "管件_val.json",
}
RULE_VERSION = "管件种类数据集标注规范V3.1-20260806"

DNS = (20, 25, 40, 50, 80, 100)
MATERIALS = ("20", "S30408", "A234 WPB", "A403 WP304L", "Q245R", "022Cr17Ni12Mo2")
PAIRS = ("50x25", "80x40", "100x50", "150x80", "250x150", "400x200")

FAMILY_NAMES = {
    "MANU": "MANU边界与WELDED过度提取对比",
    "CODE": "受控产品代号语义对比",
    "BODY": "BODY简称与近邻产品对比",
    "GEOMETRY": "角度尺寸与弯曲半径边界对比",
    "CONN": "CONN词边界与设计选型优先对比",
    "NAMED_CONN": "产品名称内嵌连接工艺对比",
    "OLET": "支管台结构与连接双维度对比",
    "JACKET": "夹套产品与施工语境对比",
}

ALLOWED_MANU = {
    "SMLS", "WELDED", "EFW", "ERW", "HFW", "SAW", "SAWL", "SAWH",
    "DASW", "DSAW", "DSAWL", "DSAWH",
}
ALLOWED_CONN = {"SW", "THD", "SCRD", "NPT", "NPTF", "FNPT", "MNPT", "FTE", "MTE", "GRV"}
THREAD_VALUES = ALLOWED_CONN - {"SW", "GRV"}
V3_NEW_BODIES = {"马鞍座接头", "SlipRing"}
DEPRECATED_BODIES = {"对焊支管台", "承插焊支管台", "螺纹支管台", "螺纹管帽"}
LONG_EL_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:W|S)?(?:30|45|60|90)EL(?![A-Za-z0-9])",
    re.I,
)
SHORT_ES_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:S(?:30|45|60|90)ES|(?:30|45|60|90)ESS|(?:30|45|60|90)ES)(?![A-Za-z0-9])",
    re.I,
)

# Keep this normalization aligned with the type train/val splitter.
STD_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:GB|HG|SH|NB|SY|JB|JIS|DIN|EN|ASTM|ASME|API|MSS)"
    r"\s*(?:/\s*T)?\s*[A-Z]*\s*\d+(?:\.\d+)*(?:-\d+)?(?:\([A-Z0-9IVX]+\))?",
    re.IGNORECASE,
)
DN_PATTERN = re.compile(r"\bDN\s*\d+(?:\s*[X×*]\s*(?:DN\s*)?\d+)*", re.IGNORECASE)
NPS_PATTERN = re.compile(r"\bNPS\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*(?:NPS\s*)?\d+(?:\.\d+)?)*", re.IGNORECASE)
INCH_PATTERN = re.compile(r'(?<![\d.])(?:\d+\s+)?\d+(?:\.\d+)?(?:/\d+)?\s*["″”]')
OD_PATTERN = re.compile(r"[Φφ]\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*\d+(?:\.\d+)?)+(?:\s*MM)?", re.IGNORECASE)
SCHEDULE_PATTERN = re.compile(
    r"\b(?:SCH(?:EDULE)?\s*[-.]?\s*|S-)(?:\d+(?:\.\d+)?S?|STD|XS|XXS)"
    r"(?:\s*[X×*]\s*(?:(?:SCH(?:EDULE)?\s*[-.]?\s*|S-)?(?:\d+(?:\.\d+)?S?|STD|XS|XXS)))*",
    re.IGNORECASE,
)
PRESSURE_PATTERN = re.compile(r"\b(?:PN\s*\d+(?:\.\d+)?|CL(?:ASS)?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:LB|MPA))\b", re.IGNORECASE)
ANGLE_TEXT_PATTERN = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:°|度|DEG(?:REE)?S?\b)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![A-Z])\d+(?:\.\d+)?(?![A-Z])", re.IGNORECASE)


def type_output(
    body: str,
    *,
    angle: str = "",
    radius: str = "",
    flange_style: str = "",
    manu: tuple[str, ...] = (),
    conn: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": body,
            "GEOMETRY": {"ANGLE": angle, "RADIUS": radius},
            "FLANGE_STYLE": flange_style,
            "MANU": list(manu),
            "CONN": list(conn),
        },
    }


def text_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def output_key(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def surface_skeleton(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).upper()
    for pattern, replacement in (
        (STD_PATTERN, " STD "),
        (DN_PATTERN, " SIZE "),
        (NPS_PATTERN, " SIZE "),
        (INCH_PATTERN, " SIZE "),
        (OD_PATTERN, " SIZE "),
        (SCHEDULE_PATTERN, " THK "),
        (PRESSURE_PATTERN, " PRESS "),
        (ANGLE_TEXT_PATTERN, " ANGLE "),
        (NUMBER_PATTERN, " N "),
    ):
        value = pattern.sub(replacement, value)
    value = re.sub(r"[,;:|/()\[\]{}]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"数据集顶层必须是数组: {path}")
    return rows


class DatasetBuilder:
    def __init__(self, base_inputs: set[str]) -> None:
        self.records: list[dict[str, Any]] = []
        self.by_input: dict[str, dict[str, Any]] = {}
        self.base_inputs = base_inputs
        self.skipped_base_duplicates = 0

    def add(
        self,
        text: str,
        output: dict[str, Any],
        *,
        family: str,
        group_id: str,
        contrast_role: str,
        prototype: str,
    ) -> None:
        normalized = text_key(text)
        if not normalized:
            raise ValueError("输入不能为空")
        if normalized in self.base_inputs:
            self.skipped_base_duplicates += 1
            return
        existing = self.by_input.get(normalized)
        if existing is not None:
            if output_key(existing["output"]) != output_key(output):
                raise ValueError(f"专项集同描述标签冲突: {text}")
            return
        record = {
            "input": text,
            "output": output,
            "来源": "基于真实模型错误的数据增强",
            "数据增强标识": True,
            "增强类型": FAMILY_NAMES[family],
            "增强组ID": group_id,
            "对比角色": contrast_role,
            "原型来源": prototype,
            "标注规则版本": RULE_VERSION,
            "建议用途": "仅并入训练集，禁止并入验证集",
        }
        self.records.append(record)
        self.by_input[normalized] = record


def add_manu_contrasts(builder: DatasetBuilder) -> None:
    for i, (dn, material) in enumerate(zip(DNS, MATERIALS)):
        group = f"MANU-{i:03d}"
        rows = (
            (f"锻制管帽;SW;NB/T47008 {material};GB/T14383;CL3000 DN{dn}", type_output("管帽", conn=("SW",)), "负例-锻制和SW不产生WELDED"),
            (f"90°锻制弯头 DN{dn} CL3000 SW SH/T3410 {material}", type_output("弯头", angle="90", conn=("SW",)), "负例-承插焊连接不产生WELDED"),
            (f"WELD OLET Forged BW {material} MSS SP-97 DN{dn * 4}x{dn}", type_output("支管台"), "负例-WELDOLET产品名不产生WELDED"),
            (f"90 DEG ELBOW LR BW {material} ASME B16.9 DN{dn}", type_output("弯头", angle="90", radius="LR"), "负例-BW不产生WELDED"),
            (f"45度弯头 SFR 1.0MPa HG/T3731 FRP/PVC DN{dn}", type_output("弯头", angle="45"), "负例-SFR和FRP不是MANU"),
            (f"合金钢管件45度弯头 DN{dn} AF12G GB/T13401", type_output("弯头", angle="45"), "负例-AF12G是材质"),
            (f"WELDED 90 DEG ELBOW LR {material} ASME B16.9 DN{dn}", type_output("弯头", angle="90", radius="LR", manu=("WELDED",)), "正例-原文明示WELDED"),
            (f"SAW 90 DEG ELBOW LR {material} ASME B16.9 DN{dn}", type_output("弯头", angle="90", radius="LR", manu=("SAW",)), "正例-具体制造工艺SAW"),
            (f"弯头 DN{dn} 连接方式:焊接 焊接方法:氩电联焊 GB/T12459 {material}", type_output("弯头"), "负例-施工焊接不产生WELDED"),
        )
        for text, output, role in rows:
            builder.add(text, output, family="MANU", group_id=group, contrast_role=role, prototype="用户反馈：MANU乱提取及WELDED过度提取")


def add_code_contrasts(builder: DatasetBuilder) -> None:
    for i, (dn, material) in enumerate(zip(DNS, MATERIALS)):
        group = f"CODE-{i:03d}"
        rows = (
            (f"S90E CL3000 {material} GB/T14383 II DN{dn}", type_output("弯头", angle="90", conn=("SW",)), "正例-S90E承插焊代号"),
            (f"90ES {material} GB/T12459 DN{dn}", type_output("弯头", angle="90", radius="SR"), "正例-90ES短半径代号"),
            (f"S90ES CL3000 {material} GB/T14383 II DN{dn}", type_output("弯头", angle="90", radius="SR", conn=("SW",)), "正例-S90ES短半径承插焊代号"),
            (f"90ESS CL3000 {material} GB/T14383 II DN{dn}", type_output("弯头", angle="90", radius="SR", conn=("SW",)), "正例-90ESS短半径承插焊代号"),
            (f"W90EL {material} GB/T13401 BW GB/T12459 DN{dn}", type_output("弯头", angle="90", radius="LR", manu=("WELDED",)), "正例-W90EL长半径焊接产品代号"),
            (f"WTS {material} GB/T13401 BW GB/T12459 DN{dn}x{dn}", type_output("等径三通", manu=("WELDED",)), "正例-WTS焊接等径三通代号"),
            (f"STS CL3000 {material} GB/T14383 II DN{dn}x{dn}", type_output("等径三通", conn=("SW",)), "正例-STS承插焊等径三通代号"),
            (f"弯头 DN{dn}II-4 90EL-20 GB/T12459", type_output("弯头", angle="90", radius="LR"), "对照例-EL表示LR但牌号20不产生20D"),
        )
        for text, output, role in rows:
            builder.add(text, output, family="CODE", group_id=group, contrast_role=role, prototype="用户确认：S90E/90ES/S90ES/W90EL/WTS/STS固定语义")


def add_body_contrasts(builder: DatasetBuilder) -> None:
    for i, (pair, material) in enumerate(zip(PAIRS, MATERIALS)):
        large, small = pair.split("x")
        group = f"BODY-{i:03d}"
        rows = (
            (f"短管支管台;FNPT;NB/T47008 {material};GB/T19326;CL3000 DN{pair}", type_output("短管支管台", conn=("FNPT",)), "正例-短管支管台不是短节"),
            (f"同管 DN{pair} S40/S80 PE/PE SH/T3419 {material}", type_output("同心异径管"), "正例-同管简称"),
            (f"同头 DN{pair} S40/S80 SH/T3408 {material}", type_output("同心异径管"), "正例-同头简称"),
            (f"偏异管 DN{pair} S40/S80 BE/PE SH/T3419 {material}", type_output("偏心异径管"), "正例-偏异管简称"),
            (f"Swage Nipple Ecc DN{pair} SCH40xSCH80 {material} MSS SP-95 SMLS", type_output("偏心异径短节", manu=("SMLS",)), "正例-Ecc异径短节"),
            (f"Nipple Swage Conc DN{pair} SCH40xSCH80 {material} MSS SP-95 SMLS", type_output("同心异径短节", manu=("SMLS",)), "正例-Conc异径短节"),
            (f"双口管箍(同心) DN{pair} CL3000 SW SH/T3410 {material}", type_output("同心双口管箍", conn=("SW",)), "正例-同心双口管箍"),
            (f"RED. CROSS {material} SMLS BE ASME B16.9 {large}x{small}", type_output("异径四通", manu=("SMLS",)), "正例-异径四通不是三通"),
            (f"六角头管塞 DN{small} CL3000 {material} GB/T14383", type_output("六角头管塞"), "正例-六角头管塞"),
            (f"加强管接头 DN{large}x{small}-SCH80-BW {material} NB/T47008", type_output("加强管接头"), "正例-加强管接头不得降级"),
            (f"加强管嘴 SW 平底型 DN{large}x{small}-SCH80 {material} BTA008-1", type_output("加强管嘴", conn=("SW",)), "正例-加强管嘴不得降级"),
            (f"马鞍座接头 SR 1.0MPa HG/T3731 DN{pair} FRP/PVC", type_output("马鞍座接头"), "正例-马鞍座接头不得降级"),
            (f"GB/T14976;HG/T20538 SlipRing CL150 DN{large}-DN{small} RF S30408/PP", type_output("SlipRing"), "正例-SlipRing保留独立主体"),
            (f"管接头 DN{pair} {material} NB/T47008", type_output("管接头"), "对照例-普通管接头不得补加强"),
        )
        for text, output, role in rows:
            builder.add(text, output, family="BODY", group_id=group, contrast_role=role, prototype="用户反馈：BODY简称及相邻产品降级错误")


def add_geometry_contrasts(builder: DatasetBuilder) -> None:
    radii = ("76", "102", "127", "257", "410.3", "586.2")
    inches = ("2", "4", "6", "8", "10", "12")
    for i, (dn, material, radius, inch) in enumerate(zip(DNS, MATERIALS, radii, inches)):
        group = f"GEOMETRY-{i:03d}"
        rows = (
            (f"Elbow {inch} STD Elbow,{inch}\",STD BW,{material},ASME B16.9", type_output("弯头"), "负例-英制尺寸不是角度"),
            (f"Elbow {inch} 4.5mm Elbow,{inch}\",4.5mm BW,{material},ASME B16.9", type_output("弯头"), "负例-尺寸和壁厚都不是角度"),
            (f"90 DEG Elbow,{inch}\",BW,{material},ASME B16.9", type_output("弯头", angle="90"), "正例-明确90度且保留英制尺寸边界"),
            (f"外管弯头 GB/T12459 90° OD{dn}x4.0 R={radius}mm {material}", type_output("弯头", angle="90", radius=f"{radius}MM"), "正例-绝对弯曲半径带MM"),
            (f"外管弯头 GB/T12459 90° OD{radius}mm x4.0 {material} DN{dn}", type_output("弯头", angle="90"), "负例-外径不是弯曲半径"),
            (f"GB/T713;SH/T3408;GB/T12459-B Elbow45 {dn}x12.7 BW {material}", type_output("弯头", angle="45"), "负例-尺寸不进入RADIUS"),
            (f"弯头 DN{dn}II-4 90EL-20 GB/T12459", type_output("弯头", angle="90", radius="LR"), "对照例-EL表示LR但牌号20不产生20D"),
            (f"30EL BW {material} ASME B16.9 DN{dn}", type_output("弯头", angle="30", radius="LR"), "正例-30EL同时表示角度和长半径"),
            (f"90°弯头 R=1.5D {material} GB/T12459 DN{dn}", type_output("弯头", angle="90", radius="1.5D"), "正例-相对弯曲半径"),
        )
        for text, output, role in rows:
            builder.add(text, output, family="GEOMETRY", group_id=group, contrast_role=role, prototype="用户反馈：英寸误作角度、普通数字误作半径、绝对半径丢单位")


def add_conn_contrasts(builder: DatasetBuilder) -> None:
    for i, (dn, material, pair) in enumerate(zip(DNS, MATERIALS, PAIRS)):
        group = f"CONN-{i:03d}"
        rows = (
            (f"弯头 90° DN{dn} SH/T3408 {material}", type_output("弯头", angle="90"), "负例-SH标准前缀不是CONN"),
            (f"CON. SWAGE; BLE/PSE; SMLS; {material}; MSS SP-95 DN{pair}", type_output("同心异径短节", manu=("SMLS",)), "负例-SWAGE和组合端部不产生SW"),
            (f"偏心异径短节 DN{pair} TBE SH/T3419 {material}", type_output("偏心异径短节"), "负例-TBE当前由正则处理"),
            (f"短型单头螺纹短节 DN{dn} SCH80 {material} BTA008-8", type_output("单头螺纹短节"), "负例-BTA不是CONN"),
            (f"管帽 NB/T47008;SH/T3410 CAP DN{dn} CL3000 THF {material}", type_output("管帽"), "负例-THF不是THD"),
            (f"等径三通 FNPT {material} GB/T14383 DN{dn} 连接方式:国标锻钢制承插焊", type_output("等径三通", conn=("FNPT",)), "冲突例-设计FNPT优先于施工SW"),
            (f"CONC REDUCER DI GALV GRV CL150 MFR'S STD DN{pair}", type_output("同心异径管", conn=("GRV",)), "正例-GRV沟槽连接"),
            (f"螺纹管帽 DN{dn} SCH80 {material} SH/T3410", type_output("管帽", conn=("THD",)), "正例-泛螺纹拆入CONN且不猜NPT"),
            (f"NPT管帽 DN{dn} SCH80 {material} ASME B16.11", type_output("管帽", conn=("NPT",)), "正例-明示NPT保留具体制式"),
        )
        for text, output, role in rows:
            builder.add(text, output, family="CONN", group_id=group, contrast_role=role, prototype="用户反馈：SH/SWAGE/BTA/THF截断误提取、GRV漏标及FNPT冲突")


def add_named_conn_contrasts(builder: DatasetBuilder) -> None:
    """Cover connection terms embedded directly in Chinese/English product names."""
    ods = (22, 27, 34, 48, 60, 76)
    for i, (dn, material, pair, od) in enumerate(zip(DNS, MATERIALS, PAIRS, ods)):
        group = f"NAMED-CONN-{i:03d}"
        rows = (
            (
                f"锻制90°承插弯头 R=1.5D:φ{od}X4,{material},SH/T3410 DN{dn}",
                type_output("弯头", angle="90", radius="1.5D", conn=("SW",)),
                "正例-承插写入弯头产品名时提取SW",
            ),
            (
                f"90度弯头，管端连接：承插焊，{material} DN{dn}",
                type_output("弯头", angle="90", conn=("SW",)),
                "正例-明示承插焊连接",
            ),
            (
                f"锻制承插等径三通:φ{od}X4,{material},GB/T14383 DN{dn}",
                type_output("等径三通", conn=("SW",)),
                "正例-承插写入等径三通产品名时提取SW",
            ),
            (
                f"SOCKET WELD REDUCING TEE {material} ASME B16.11 DN{pair}",
                type_output("异径三通", conn=("SW",)),
                "正例-英文SOCKET WELD提取SW",
            ),
            (
                f"90 DEG BUTT WELD ELBOW {material} ASME B16.9 DN{dn}",
                type_output("弯头", angle="90"),
                "对照例-对焊不误标SW",
            ),
            (
                f"等径三通 {material} GB/T12459 DN{dn} 现场焊接安装",
                type_output("等径三通"),
                "对照例-施工焊接不推导SW",
            ),
            (
                f"等径三通 FNPT {material} GB/T14383 DN{dn} 工艺执行：承插焊",
                type_output("等径三通", conn=("FNPT",)),
                "冲突例-设计FNPT优先于工艺承插焊",
            ),
            (
                f"SOCKET WELD CAP CL3000 {material} ASME B16.11 DN{dn}",
                type_output("管帽", conn=("SW",)),
                "正例-英文承插焊管帽",
            ),
        )
        for text, output, role in rows:
            builder.add(
                text,
                output,
                family="NAMED_CONN",
                group_id=group,
                contrast_role=role,
                prototype="管件核对0805.xlsx：承插弯头和承插等径三通漏标SW",
            )


def add_olet_contrasts(builder: DatasetBuilder) -> None:
    for i, (pair, material) in enumerate(zip(PAIRS, MATERIALS)):
        group = f"OLET-{i:03d}"
        rows = (
            (f"OLET {material} MSS SP-97 DN{pair}", type_output("支管台"), "对照例-普通支管台"),
            (f"WELDOLET Forged BW {material} MSS SP-97 DN{pair}", type_output("支管台"), "正例-对焊端不拼入BODY且CONN留空"),
            (f"SOCKOLET Forged SW {material} MSS SP-97 DN{pair}", type_output("支管台", conn=("SW",)), "正例-承插焊拆入CONN"),
            (f"THREDOLET FNPT {material} MSS SP-97 DN{pair}", type_output("支管台", conn=("FNPT",)), "正例-螺纹制式拆入CONN"),
            (f"45° WELDOLET Forged BW {material} GB/T19326 DN{pair}", type_output("斜支管台", angle="45"), "正例-45度对焊斜支管台"),
            (f"45° SOCKOLET Forged SW {material} GB/T19326 DN{pair}", type_output("斜支管台", angle="45", conn=("SW",)), "正例-45度承插焊斜支管台"),
            (f"NIPOLET FNPT {material} MSS SP-97 DN{pair}", type_output("短管支管台", conn=("FNPT",)), "正例-短管支管台保留结构身份"),
            (f"SWEEPOLET BW {material} MSS SP-97 DN{pair}", type_output("扫掠式支管台"), "正例-扫掠式支管台"),
            (f"LIGHT WEIGHT BW OLET {material} MFRS STD DN{pair}", type_output("轻型支管台"), "正例-轻型支管台"),
            (f"ELBOLET BW {material} MSS SP-97 DN{pair}", type_output("弯头支管台"), "正例-弯头支管台"),
        )
        for text, output, role in rows:
            builder.add(text, output, family="OLET", group_id=group, contrast_role=role, prototype="用户确认：支管台结构优先，连接和角度分字段")


def add_jacket_contrasts(builder: DatasetBuilder) -> None:
    for i, (pair, material) in enumerate(zip(PAIRS[:5], MATERIALS[:5])):
        group = f"JACKET-{i:03d}"
        rows = (
            (f"CON REDUCER {material} SMLS BW ASME B16.9 JACKET DN{pair}", type_output("夹套同心异径管", manu=("SMLS",)), "正例-JACKET直接修饰异径管"),
            (f"CON REDUCER {material} SMLS BW ASME B16.9 DN{pair}", type_output("同心异径管", manu=("SMLS",)), "对照例-无夹套结构"),
            (f"JACKET 90 DEG ELBOW {material} SMLS BW ASME B16.9 DN{pair.split('x')[0]}", type_output("夹套弯头", angle="90", manu=("SMLS",)), "正例-JACKET直接修饰弯头"),
            (f"90 DEG ELBOW {material} SMLS BW ASME B16.9 DN{pair.split('x')[0]}", type_output("弯头", angle="90", manu=("SMLS",)), "对照例-普通弯头"),
            (f"夹套管件安装 异径三通 DN{pair} GB/T12459 {material} SMLS BW", type_output("异径三通", manu=("SMLS",)), "负例-施工项目名称不代表当前产品夹套"),
            (f"夹套异径三通 DN{pair} GB/T12459 {material} SMLS BW", type_output("夹套异径三通", manu=("SMLS",)), "正例-明确夹套产品名称"),
        )
        for text, output, role in rows:
            builder.add(text, output, family="JACKET", group_id=group, contrast_role=role, prototype="用户反馈：夹套结构漏识别与夹套安装误判")


def validate_records(records: list[dict[str, Any]], base_bodies: set[str]) -> dict[str, Any]:
    seen: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_bodies: set[str] = set()

    for index, record in enumerate(records):
        normalized = text_key(record["input"])
        if normalized in seen:
            raise ValueError(f"专项集输入重复: index={index}")
        seen.add(normalized)
        groups[record["增强组ID"]].append(record)

        output = record["output"]
        if set(output) != {"CATEGORY", "TYPE"} or output["CATEGORY"] != "管件":
            raise ValueError(f"输出骨架错误: index={index}")
        type_data = output["TYPE"]
        if set(type_data) != {"BODY", "GEOMETRY", "FLANGE_STYLE", "MANU", "CONN"}:
            raise ValueError(f"TYPE骨架错误: index={index}")
        if not set(type_data["MANU"]).issubset(ALLOWED_MANU):
            raise ValueError(f"非法MANU: index={index}, {type_data['MANU']}")
        if not set(type_data["CONN"]).issubset(ALLOWED_CONN):
            raise ValueError(f"非法CONN: index={index}, {type_data['CONN']}")
        if type_data["BODY"] in DEPRECATED_BODIES:
            raise ValueError(f"专项集仍使用V3.1历史BODY: index={index}, {type_data['BODY']}")
        if type_data["BODY"] not in base_bodies and type_data["BODY"] not in V3_NEW_BODIES:
            unknown_bodies.add(type_data["BODY"])

        text = record["input"]
        if re.search(r"WELD\s*OLET|WELDOLET", text, re.I):
            if type_data["BODY"] not in {"支管台", "斜支管台"} or "WELDED" in type_data["MANU"]:
                raise ValueError(f"WELDOLET字段拆分错误: index={index}")
        if re.search(r"SOCKOLET", text, re.I):
            if type_data["BODY"] not in {"支管台", "斜支管台"} or "SW" not in type_data["CONN"]:
                raise ValueError(f"SOCKOLET字段拆分错误: index={index}")
        if re.search(r"THREDOLET", text, re.I):
            if type_data["BODY"] != "支管台" or not set(type_data["CONN"]) & THREAD_VALUES:
                raise ValueError(f"THREDOLET字段拆分错误: index={index}")
        if re.search(r"螺纹管帽", text) and type_data["BODY"] != "管帽":
            raise ValueError(f"螺纹管帽BODY未拆分: index={index}")
        if LONG_EL_CODE_PATTERN.search(text) and not type_data["GEOMETRY"]["RADIUS"]:
            raise ValueError(f"EL产品代号未标注长半径: index={index}")
        if SHORT_ES_CODE_PATTERN.search(text) and not type_data["GEOMETRY"]["RADIUS"]:
            raise ValueError(f"ES产品代号未标注短半径: index={index}")

    if unknown_bodies:
        raise ValueError(f"专项集出现未受控BODY: {sorted(unknown_bodies)}")

    weak_groups = []
    for group_id, group_records in groups.items():
        signatures = {output_key(record["output"]) for record in group_records}
        if len(signatures) < 2:
            weak_groups.append(group_id)
    if weak_groups:
        raise ValueError(f"以下增强组不构成输出对比: {weak_groups}")

    return {
        "唯一输入数": len(seen),
        "增强组数": len(groups),
        "所有增强组均含不同输出": True,
        "非法MANU数": 0,
        "非法CONN数": 0,
            "未受控BODY数": 0,
            "V3.1历史BODY数": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = {name: load_rows(path) for name, path in BASE_DATASETS.items()}
    base_inputs = {text_key(row["input"]) for rows in base_rows.values() for row in rows}
    base_bodies = {
        row["output"]["TYPE"]["BODY"]
        for rows in base_rows.values()
        for row in rows
    }
    builder = DatasetBuilder(base_inputs)

    add_manu_contrasts(builder)
    add_code_contrasts(builder)
    add_body_contrasts(builder)
    add_geometry_contrasts(builder)
    add_conn_contrasts(builder)
    add_olet_contrasts(builder)
    add_jacket_contrasts(builder)
    # Keep previously reviewed rows stable; new families are appended at the end.
    add_named_conn_contrasts(builder)

    validation = validate_records(builder.records, base_bodies)
    special_skeletons = {surface_skeleton(record["input"]) for record in builder.records}
    train_skeletons = {surface_skeleton(record["input"]) for record in base_rows["train"]}
    val_skeletons = {surface_skeleton(record["input"]) for record in base_rows["val"]}
    train_overlap = special_skeletons & train_skeletons
    val_overlap = special_skeletons & val_skeletons
    if train_overlap or val_overlap:
        raise ValueError(
            f"专项集骨架与基础集重叠: train={len(train_overlap)}, val={len(val_overlap)}"
        )
    validation.update(
        {
            "专项描述骨架数": len(special_skeletons),
            "与训练集描述骨架重叠数": 0,
            "与验证集描述骨架重叠数": 0,
        }
    )
    family_counts = Counter(record["增强类型"] for record in builder.records)
    body_counts = Counter(record["output"]["TYPE"]["BODY"] for record in builder.records)
    role_counts = Counter(record["对比角色"].split("-", 1)[0] for record in builder.records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "管件专项对比训练集_审核版_带来源.json"
    report_path = args.output_dir / "管件专项对比训练集_生成报告.json"
    with data_path.open("w", encoding="utf-8") as file:
        json.dump(builder.records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    report = {
        "规则版本": RULE_VERSION,
        "说明": "本文件为训练集候选审核版，尚未并入管件_train.json；禁止并入验证集。",
        "基础数据": {name: {"路径": str(BASE_DATASETS[name]), "条数": len(rows)} for name, rows in base_rows.items()},
        "输出文件": str(data_path),
        "统计": {
            "专项样本数": len(builder.records),
            "占当前训练集比例": round(len(builder.records) / len(base_rows["train"]), 6),
            "跳过基础集完全重复数": builder.skipped_base_duplicates,
            "按增强类型": dict(family_counts),
            "按对比角色前缀": dict(role_counts),
            "按BODY": dict(body_counts.most_common()),
        },
        "质量检查": validation,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(json.dumps(report["统计"], ensure_ascii=False))
    print(data_path)
    print(report_path)


if __name__ == "__main__":
    main()
