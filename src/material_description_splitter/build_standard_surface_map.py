# -*- coding: utf-8 -*-
"""Build standard surface-locator config for standard glue detection."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


CODE_RE = re.compile(r"^([A-Z]+)(\d+(?:\.\d+)?)([A-Z0-9.]*)$")


def build_ab_patterns(core: str) -> list[str]:
    # 语料里 AB 家族几乎都来自 ASME Bxx.xx / Bx.xx.x 这种表面写法。
    if "." in core:
        dotted = core
    elif core == "1201":
        dotted = "1.20.1"
    elif len(core) == 4:
        dotted = f"{core[:2]}.{core[2:]}"
    elif len(core) == 3:
        dotted = f"{core[:2]}.{core[2]}"
    else:
        dotted = core
    return [
        fr"ASME\s*-?\s*B\s*{re.escape(dotted)}(?:M)?",
        fr"ASMEB\s*{re.escape(dotted)}(?:M)?",
        fr"B\s*{re.escape(dotted)}(?:M)?",
    ]


def build_anm_patterns(core: str) -> list[str]:
    dotted = core
    return [
        fr"ASME\s*NM\.?\s*{re.escape(dotted)}",
        fr"NM\.?\s*{re.escape(dotted)}",
        fr"ANM\s*{re.escape(dotted)}",
    ]


def build_asmc_patterns(core: str) -> list[str]:
    dotted = core
    return [
        fr"AS\s*MC\s*{re.escape(dotted)}",
        fr"ASMC\s*{re.escape(dotted)}",
        fr"ASME\s*C\.?\s*{re.escape(dotted)}",
    ]


FAMILY_SURFACE_BUILDERS: dict[str, Any] = {
    "GBT": lambda core: [fr"GB/T\s*{re.escape(core)}", fr"GBT\s*{re.escape(core)}"],
    "HGT": lambda core: [fr"HG/T\s*{re.escape(core)}", fr"HGT\s*{re.escape(core)}"],
    "NBT": lambda core: [fr"NB/T\s*{re.escape(core)}", fr"NBT\s*{re.escape(core)}"],
    "SHT": lambda core: [fr"SH/T\s*{re.escape(core)}", fr"SHT\s*{re.escape(core)}"],
    "SYT": lambda core: [fr"SY/T\s*{re.escape(core)}", fr"SYT\s*{re.escape(core)}"],
    "CJT": lambda core: [fr"CJ/T\s*{re.escape(core)}", fr"CJT\s*{re.escape(core)}"],
    "DLT": lambda core: [fr"DL/T\s*{re.escape(core)}", fr"DLT\s*{re.escape(core)}"],
    "DIN": lambda core: [fr"DIN\s*{re.escape(core)}"],
    "EN": lambda core: [fr"EN\s*{re.escape(core)}"],
    "ENI": lambda core: [fr"EN\s*ISO\s*{re.escape(core)}", fr"ENI\s*{re.escape(core)}"],
    "API": lambda core: [fr"API\s*{re.escape(core)}"],
    "ASTM": lambda core: [fr"ASTM\s*[A-Z]?\s*{re.escape(core)}", fr"ASTM\s*{re.escape(core)}"],
    "AB": build_ab_patterns,
    "ANM": build_anm_patterns,
    "ASMC": build_asmc_patterns,
    "GB": lambda core: [fr"GB\s*/?\s*T?\s*{re.escape(core)}", fr"GB\s*{re.escape(core)}"],
    "NB": lambda core: [fr"NB\s*/?\s*T?\s*{re.escape(core)}", fr"NB\s*{re.escape(core)}"],
    "SH": lambda core: [fr"SH\s*/?\s*T?\s*{re.escape(core)}", fr"SH\s*{re.escape(core)}"],
    "HG": lambda core: [fr"HG\s*/?\s*T?\s*{re.escape(core)}", fr"HG\s*{re.escape(core)}"],
    "TB": lambda core: [fr"TB\s*{re.escape(core)}"],
    "MC": lambda core: [fr"MC\s*{re.escape(core)}"],
    "MS": lambda core: [fr"MS\s*{re.escape(core)}"],
    "GD": lambda core: [fr"GD\s*{re.escape(core)}"],
}


def load_standard_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("standard_groups", {}) or {}


def split_code(code: str) -> tuple[str, str, str] | None:
    m = CODE_RE.match(code)
    if not m:
        return None
    family, core, suffix = m.groups()
    return family, core, suffix


def build_patterns(family: str, core: str) -> list[str]:
    builder = FAMILY_SURFACE_BUILDERS.get(family)
    if builder is None:
        return [fr"{re.escape(family)}\s*{re.escape(core)}"]
    patterns = builder(core)
    dedup: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if pattern and pattern not in seen:
            seen.add(pattern)
            dedup.append(pattern)
    return dedup


def build_surface_map(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    locator_map: dict[str, dict[str, Any]] = {}

    for _family, items in groups.items():
        for item in items:
            code = str(item.get("code", "")).strip()
            parsed = split_code(code)
            if not parsed:
                continue
            parsed_family, core, _suffix = parsed
            locator_code = f"{parsed_family}{core}"
            locator_map.setdefault(
                locator_code,
                {
                    "family": parsed_family,
                    "core": core,
                    "patterns": build_patterns(parsed_family, core),
                },
            )

    ordered = dict(sorted(locator_map.items(), key=lambda kv: kv[0]))
    return {
        "meta": {
            "total_locator_codes": len(ordered),
            "ignore_suffix": True,
            "note": "规范粘连定位只匹配主体标准，不考虑后缀等级。",
        },
        "locator_codes": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard surface locator config")
    parser.add_argument(
        "--input",
        default="src/material_description_splitter/config/standard_code_map.yaml",
        help="Input standard code summary yaml",
    )
    parser.add_argument(
        "--output",
        default="src/material_description_splitter/config/standard_surface_map.yaml",
        help="Output standard surface locator yaml",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = load_standard_groups(input_path)
    result = build_surface_map(groups)
    output_path.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"input: {input_path}")
    print(f"locator codes: {result['meta']['total_locator_codes']}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
