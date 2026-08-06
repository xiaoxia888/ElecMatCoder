#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_BASE = (
    REPO_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/法兰.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/专项增强_20260804"
)

DNS = (15, 20, 25, 40, 50, 80, 100, 125, 150, 200, 250, 300)
PAIRS = (
    "50x25",
    "80x50",
    "100x80",
    "125x80",
    "150x100",
    "200x150",
    "250x200",
    "300x250",
    "350x300",
    "400x300",
    "450x400",
    "500x400",
)
MATERIALS = ("20", "A105", "A182 F304L", "06Cr19Ni10", "S31603")
PRESSURES = ("PN16", "PN25", "CL150", "CL300", "CL600")

FAMILY_NAMES = {
    "THREAD": "螺纹制式最小对比",
    "JACKET": "JWN/JSO夹套主体与粘连对比",
    "RING": "RJ/RTJ/FRJ/MRJ环连接面对比",
    "FACE": "M/FM/MF/MFM与EP边界对比",
    "ALIAS": "密封面组合代码归一对比",
    "BODY": "低频法兰主体对比",
    "REAL": "真实项目错误样本",
}

# 同一增强组的所有表达必须位于同一个数据分区。
VAL_GROUPS = {
    "THREAD": {10, 11},
    "JACKET": {10, 11},
    "RING": {8, 9},
    "FACE": {8, 9},
    "ALIAS": {7},
    "BODY": {8, 9},
}


def type_output(body: str, *, conn: tuple[str, ...] = (), seal: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "CATEGORY": "法兰",
        "TYPE": {
            "BODY": body,
            "CONN": list(conn),
            "SEAL": list(seal),
        },
    }


def text_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def output_key(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DatasetBuilder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.by_input: dict[str, dict[str, Any]] = {}

    def add(
        self,
        text: str,
        output: dict[str, Any],
        *,
        family: str,
        group_id: str,
        source: str,
        augmented: bool,
        prototype: str,
    ) -> None:
        normalized = text_key(text)
        if not normalized:
            raise ValueError("输入不能为空")
        existing = self.by_input.get(normalized)
        if existing is not None:
            if output_key(existing["output"]) != output_key(output):
                raise ValueError(f"专项集同描述标签冲突: {text}")
            return
        record = {
            "input": text,
            "output": output,
            "来源": source,
            "数据增强标识": augmented,
            "增强类型": FAMILY_NAMES[family],
            "增强组ID": group_id,
            "原型来源": prototype,
        }
        self.records.append(record)
        self.by_input[normalized] = record


def add_real_project_records(builder: DatasetBuilder) -> None:
    def add(text: str, output: dict[str, Any], group_id: str, prototype: str) -> None:
        builder.add(
            text,
            output,
            family="REAL",
            group_id=group_id,
            source="真实项目数据-用户反馈-20260804",
            augmented=False,
            prototype=prototype,
        )

    for dn in (40, 50):
        add(
            f"法兰DN{dn}FLANGEHG/T20592(A)20PN16SCRDRFGALV.",
            type_output("螺纹法兰", conn=("SCRD",), seal=("RF",)),
            "REAL-THREAD-SCRDRF",
            "真实项目：全粘连SCRD+RF",
        )

    for dn in (50, 125):
        add(
            f"法兰DN{dn}FLANGEHG/T2061506Cr19Ni10CL.1500WNRJSch80",
            type_output("带颈对焊法兰", seal=("RJ",)),
            "REAL-WN-RJ",
            "真实项目：全粘连WN+RJ，不是JWN",
        )

    add(
        "法兰DN400*DN300FLANGEHG/T20592(A)06Cr19Ni10PN63JWNRFSch40S",
        type_output("带颈对焊夹套法兰", seal=("RF",)),
        "REAL-JWN-RF",
        "真实项目：全粘连JWN+RF",
    )
    add(
        "凹凸面夹套法兰 DN250*DN200 FLANGE HG/T20592(A) 06Cr19Ni10 PN25 JWN MF DN250 Sch 20 DN200 Sch 40S",
        type_output("带颈对焊夹套法兰", seal=("MF",)),
        "REAL-JWN-MF",
        "真实项目：JWN是主体，MF是密封面",
    )

    female_specs = (
        ("CL900", "A105", 20),
        ("CL900", "A105", 25),
        ("CL900", "A105", 40),
        ("CL900", "A105", 50),
        ("CL900", "A182 F321", 100),
        ("CL600", "NB/T47008 20", 150),
        ("CL600", "NB/T47008 20", 200),
    )
    for pressure, material, dn in female_specs:
        standard = "ASME B16.48" if "A105" in material or "A182" in material else "SH/T3425"
        add(
            f"8字盲板;{pressure};Female RJ;{material};{standard};- DN{dn}",
            type_output("8字盲板", seal=("FRJ",)),
            "REAL-FEMALE-RJ",
            "真实项目：Female RJ必须保留阴式语义",
        )
    add(
        "SPECTACLE BLANK CL1500(PN260) Female RJ(RTJ) A105 ASME B16.48 - DN25",
        type_output("8字盲板", seal=("FRJ",)),
        "REAL-FEMALE-RJ-RTJ",
        "真实项目：Female RJ(RTJ)以Female RJ为准",
    )

    for size in ("40x80", "50x80", "80x125"):
        add(
            f"法兰,150LB,RF JSO{size},HG/T20615-2009,20",
            type_output("带颈平焊夹套法兰", seal=("RF",)),
            "REAL-JSO-RF",
            "真实项目：JSO表示带颈平焊夹套法兰",
        )

    for size in ("200x150", "50x25", "150x100", "100x80", "300x250", "80x50"):
        add(
            f"REDUCING FLANGE SO A182GR.F304L CL.150 RF ASME B16.5 JACKET DWG No.003 DN{size}",
            type_output("异径带颈平焊夹套法兰", seal=("RF",)),
            "REAL-REDUCING-SO-JACKET",
            "真实项目：REDUCING+SO+JACKET组合主体",
        )

    for size in ("150X80", "350X300"):
        add(
            f"WELDING NECK FLANGE CL300(PN50) RF A105 SH/T 3426 - Jacketed SCH40XSCH40 DN{size}",
            type_output("带颈对焊夹套法兰", seal=("RF",)),
            "REAL-WN-JACKETED",
            "真实项目：WELDING NECK+Jacketed",
        )

    for dn in (50, 80, 150, 200, 250, 300, 400, 450, 500):
        add(
            f"FLANGE FITTING WITH ADAPTER STEEL REINFORCED POLYETHYLENE CL.150 CJ/T 124 DN{dn}",
            type_output("带适配器法兰"),
            "REAL-ADAPTER-FLANGE",
            "真实项目：FLANGE FITTING WITH ADAPTER",
        )

    blind_spacers = (
        "盲板,垫环;CL150;RF;NB/T 47008 20;SH/T 3425;- DN600",
        "盲板，垫环; SH/T 3425 CL150 RF20 DN400",
        "盲板,垫环;SH/T 3425 CL300 RF 20 DN600",
        "盲板,垫环;CL300;RF;NB/T 47008 20;SH/T 3425;- DN300",
    )
    for text in blind_spacers:
        add(
            text,
            type_output("盲板垫环组件", seal=("RF",)),
            "REAL-BLIND-SPACER",
            "真实项目：盲板与垫环是组合主体",
        )

    add(
        "SCREWED FLANGE CL150(PN20) RF NB/T 47008 20 Galv. SH/T 3406 - DN50",
        type_output("螺纹法兰", conn=("SCRD",), seal=("RF",)),
        "REAL-SCREWED",
        "真实项目：SCREWED一阶段保留为SCRD",
    )

    ep_records = (
        ("法兰盖\\PN25 DN25 M HG/T20592 20#", ("M",)),
        ("不锈钢衬里法兰盖,M,END,PN160,材料：S31603-TUBE-EP,HG/T 20592-2009,DN25", ("M",)),
        ("法兰盖10(B)-160 M HG/T 20592-2009,S31603-TUBE-EP", ("M",)),
        ("法兰盖15(B)-160 HG/T 20592-2009,S31603-TUBE-EP", ()),
    )
    for text, seal in ep_records:
        add(
            text,
            type_output("盲板法兰", seal=seal),
            "REAL-M-EP",
            "真实项目：独立M是密封面，材料牌号中EP不是SEAL",
        )

    add(
        "8字盲板 A105,ASME B16.48,FWRF,150LB,DN65",
        type_output("8字盲板", seal=("FWRF",)),
        "REAL-FWRF",
        "真实项目：一阶段保留原文FWRF",
    )


def add_augmented_records(builder: DatasetBuilder) -> None:
    for i in range(12):
        dn = DNS[i]
        material = MATERIALS[i % len(MATERIALS)]
        pressure = PRESSURES[i % len(PRESSURES)]
        group = f"THREAD-{i:03d}"
        prototype = "螺纹法兰制式与粘连最小对比"
        rows = (
            (f"法兰 DN{dn} FLANGE HG/T20592(A) {material} {pressure} SCRD RF", ("SCRD",), ("RF",)),
            (f"法兰DN{dn}FLANGEHG/T20592(A){material}{pressure}SCRDRF", ("SCRD",), ("RF",)),
            (f"SCREWED FLANGE {pressure} RF {material} ASME B16.5 DN{dn}", ("SCRD",), ("RF",)),
            (f"THREADED FLANGE {pressure} RF {material} ASME B16.5 DN{dn}", ("THD",), ("RF",)),
            (f"NPT FLANGE {pressure} RF {material} ASME B16.5 DN{dn}", ("NPT",), ("RF",)),
            (f"FEMALE NPT FLANGE {pressure} RF {material} ASME B16.5 DN{dn}", ("FNPT",), ("RF",)),
        )
        for text, conn, seal in rows:
            builder.add(text, type_output("螺纹法兰", conn=conn, seal=seal), family="THREAD", group_id=group, source="数据增强", augmented=True, prototype=prototype)

    for i in range(12):
        pair = PAIRS[i]
        material = MATERIALS[i % len(MATERIALS)]
        pressure = PRESSURES[i % len(PRESSURES)]
        group = f"JACKET-{i:03d}"
        prototype = "JWN/JSO与WN/SO的夹套主体最小对比"
        rows = (
            (f"FLANGE DN{pair} {material} {pressure} JWN RF", "带颈对焊夹套法兰", ("RF",)),
            (f"FLANGEDN{pair}{material}{pressure}JWNRF", "带颈对焊夹套法兰", ("RF",)),
            (f"JACKETED FLANGE DN{pair} JWN RJ {material} {pressure}", "带颈对焊夹套法兰", ("RJ",)),
            (f"FLANGEDN{pair}{material}{pressure}JWNRJ", "带颈对焊夹套法兰", ("RJ",)),
            (f"凹凸面夹套法兰 DN{pair} JWN MF {material} {pressure}", "带颈对焊夹套法兰", ("MF",)),
            (f"夹套法兰 DN{pair} JWN MFM {material} {pressure}", "带颈对焊夹套法兰", ("MFM",)),
            (f"WELD NECK FLANGE DN{pair.split("x")[0]} WN RF {material} {pressure}", "带颈对焊法兰", ("RF",)),
            (f"FLANGEDN{pair.split("x")[0]}{material}{pressure}WNRJ", "带颈对焊法兰", ("RJ",)),
            (f"FLANGEDN{pair.split("x")[0]}{material}{pressure}RJWN", "带颈对焊法兰", ("RJ",)),
            (f"FLANGEDN{pair.split("x")[0]}{material}{pressure}RTJWN", "带颈对焊法兰", ("RTJ",)),
            (f"FLANGE DN{pair} JSO RF {material} {pressure}", "带颈平焊夹套法兰", ("RF",)),
            (f"FLANGEDN{pair}{material}{pressure}JSORF", "带颈平焊夹套法兰", ("RF",)),
            (f"SLIP ON FLANGE DN{pair.split("x")[0]} SO RF {material} {pressure}", "带颈平焊法兰", ("RF",)),
            (f"FLANGEDN{pair.split("x")[0]}{material}{pressure}SORF", "带颈平焊法兰", ("RF",)),
        )
        for text, body, seal in rows:
            builder.add(text, type_output(body, seal=seal), family="JACKET", group_id=group, source="数据增强", augmented=True, prototype=prototype)

    for i in range(10):
        dn = DNS[i]
        material = MATERIALS[i % len(MATERIALS)]
        pressure = PRESSURES[i % len(PRESSURES)]
        group = f"RING-{i:03d}"
        prototype = "RJ/RTJ/Female RJ/Male RJ最小对比"
        rows = (
            (f"SPECTACLE BLANK {pressure} Female RJ {material} ASME B16.48 DN{dn}", ("FRJ",)),
            (f"SPECTACLEBLANK{pressure}FemaleRJ(RTJ){material}ASMEB16.48DN{dn}", ("FRJ",)),
            (f"SPECTACLE BLANK {pressure} RJ {material} ASME B16.48 DN{dn}", ("RJ",)),
            (f"SPECTACLE BLANK {pressure} RTJ {material} ASME B16.48 DN{dn}", ("RTJ",)),
            (f"SPECTACLE BLANK {pressure} Male RJ {material} ASME B16.48 DN{dn}", ("MRJ",)),
            (f"SPECTACLE BLANK {pressure} RF {material} ASME B16.48 DN{dn}", ("RF",)),
        )
        for text, seal in rows:
            builder.add(text, type_output("8字盲板", seal=seal), family="RING", group_id=group, source="数据增强", augmented=True, prototype=prototype)

    for i in range(10):
        dn = DNS[i]
        pressure = PRESSURES[i % len(PRESSURES)]
        group = f"FACE-{i:03d}"
        prototype = "M/FM/MF/MFM与材料后缀EP最小对比"
        rows = (
            (f"法兰盖 DN{dn} {pressure} M HG/T20592 材料:S31603-TUBE-EP", ("M",)),
            (f"法兰盖 DN{dn} {pressure} HG/T20592 材料:S31603-TUBE-EP", ()),
            (f"法兰盖 DN{dn} {pressure} FM HG/T20592 材料:S31603-TUBE-EP", ("FM",)),
            (f"凹凸面法兰 DN{dn} {pressure} MF HG/T20592 20", ("MF",)),
            (f"法兰 DN{dn} {pressure} MFM HG/T20592 20", ("MFM",)),
            (f"法兰 DN{dn} {pressure} LM HG/T20592 20", ("LM",)),
            (f"法兰 DN{dn} {pressure} LF HG/T20592 20", ("LF",)),
        )
        for text, seal in rows:
            builder.add(text, type_output("盲板法兰" if "法兰盖" in text else "法兰", seal=seal), family="FACE", group_id=group, source="数据增强", augmented=True, prototype=prototype)

    for i in range(8):
        dn = DNS[i]
        material = MATERIALS[i % len(MATERIALS)]
        group = f"ALIAS-{i:03d}"
        prototype = "FLRF/FLRJ/FWRF/LMFE组合代码边界对比"
        rows = (
            (f"WN FLANGE FLRF {material} ASME B16.5 DN{dn}", ()),
            (f"WN FLANGE FLRJ {material} ASME B16.5 DN{dn}", ("RJ",)),
            (f"SPECTACLE BLANK FWRF {material} ASME B16.48 DN{dn}", ("FWRF",)),
            (f"WN FLANGE LMFE {material} HG/T20592 DN{dn}", ("LM",)),
            (f"WN FLANGE SERRATED FINISH {material} ASME B16.5 DN{dn}", ("SERRATED",)),
        )
        for text, seal in rows:
            body = "8字盲板" if "SPECTACLE" in text else "带颈对焊法兰"
            builder.add(text, type_output(body, seal=seal), family="ALIAS", group_id=group, source="数据增强", augmented=True, prototype=prototype)

    for i in range(10):
        dn = DNS[i]
        pair = PAIRS[i]
        material = MATERIALS[i % len(MATERIALS)]
        pressure = PRESSURES[i % len(PRESSURES)]
        group = f"BODY-{i:03d}"
        prototype = "盲板垫环、适配器和夹套主体最小对比"
        rows = (
            (f"盲板,垫环;{pressure};RF;{material};SH/T3425;DN{dn}", "盲板垫环组件", ("RF",)),
            (f"盲板法兰;{pressure};RF;{material};ASME B16.5;DN{dn}", "盲板法兰", ("RF",)),
            (f"FLANGE FITTING WITH ADAPTER {material} {pressure} CJ/T124 DN{dn}", "带适配器法兰", ()),
            (f"FLANGE FITTING {material} {pressure} CJ/T124 DN{dn}", "法兰", ()),
            (f"REDUCING FLANGE SO {material} {pressure} RF ASME B16.5 JACKET DN{pair}", "异径带颈平焊夹套法兰", ("RF",)),
            (f"REDUCING FLANGE SO {material} {pressure} RF ASME B16.5 DN{pair}", "异径带颈平焊法兰", ("RF",)),
            (f"WELDING NECK FLANGE {material} {pressure} RF JACKETED DN{pair}", "带颈对焊夹套法兰", ("RF",)),
            (f"WELDING NECK FLANGE {material} {pressure} RF DN{dn}", "带颈对焊法兰", ("RF",)),
        )
        for text, body, seal in rows:
            builder.add(text, type_output(body, seal=seal), family="BODY", group_id=group, source="数据增强", augmented=True, prototype=prototype)


def family_from_group(group_id: str) -> str:
    return group_id.split("-", 1)[0]


def split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for record in records:
        group_id = record["增强组ID"]
        family = family_from_group(group_id)
        if family == "REAL":
            target = train
        else:
            group_no = int(group_id.rsplit("-", 1)[1])
            target = val if group_no in VAL_GROUPS[family] else train
        copied = dict(record)
        copied["数据划分"] = "val" if target is val else "train"
        target.append(copied)
    return train, val


def load_base(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("基础法兰数据集必须是JSON数组")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = str(row.get("input") or "")
        if text.strip():
            result[text_key(text)] = row
    return result


def build_report(
    all_records: list[dict[str, Any]],
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    base: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_inputs = {text_key(row["input"]) for row in train}
    val_inputs = {text_key(row["input"]) for row in val}
    train_groups = {row["增强组ID"] for row in train}
    val_groups = {row["增强组ID"] for row in val}
    overlap_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for row in all_records:
        base_row = base.get(text_key(row["input"]))
        if base_row is None:
            continue
        detail = {
            "input": row["input"],
            "专项集标签": row["output"],
            "基础集标签": base_row.get("output"),
            "增强组ID": row["增强组ID"],
        }
        overlap_rows.append(detail)
        if output_key(row["output"]) != output_key(base_row.get("output", {})):
            conflict_rows.append(detail)

    def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        return dict(Counter(row[field] for row in rows).most_common())

    body_counts = Counter(row["output"]["TYPE"]["BODY"] for row in all_records)
    conn_counts = Counter(value for row in all_records for value in row["output"]["TYPE"]["CONN"])
    seal_counts = Counter(value for row in all_records for value in row["output"]["TYPE"]["SEAL"])
    digest = hashlib.sha256(
        json.dumps(all_records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "生成参数": {
            "划分方式": "按增强组ID隔离，真实项目样本全部进入训练集",
            "数据集SHA256": digest,
        },
        "数量统计": {
            "全部": len(all_records),
            "训练集": len(train),
            "验证集": len(val),
            "真实项目": sum(not row["数据增强标识"] for row in all_records),
            "合成增强": sum(row["数据增强标识"] for row in all_records),
            "训练验证输入重叠": len(train_inputs & val_inputs),
            "训练验证增强组重叠": len(train_groups & val_groups),
            "与基础法兰集输入重叠": len(overlap_rows),
            "与基础法兰集标签冲突": len(conflict_rows),
        },
        "专题分布": distribution(all_records, "增强类型"),
        "BODY分布": dict(body_counts.most_common()),
        "CONN分布": dict(conn_counts.most_common()),
        "SEAL分布": dict(seal_counts.most_common()),
        "与基础集标签冲突明细": conflict_rows,
        "与基集重复明细": overlap_rows,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成独立的法兰种类专项对比训练集")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    builder = DatasetBuilder()
    add_real_project_records(builder)
    add_augmented_records(builder)
    train, val = split_records(builder.records)
    report = build_report(builder.records, train, val, load_base(args.base))

    write_json(args.output_dir / "法兰专项对比_train.json", train)
    write_json(args.output_dir / "法兰专项对比_val.json", val)
    write_json(args.output_dir / "法兰专项对比_全部_带来源.json", builder.records)
    write_json(args.output_dir / "法兰专项对比_生成报告.json", report)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "all": len(builder.records),
        "train": len(train),
        "val": len(val),
        "base_conflicts": report["数量统计"]["与基础法兰集标签冲突"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
