from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "根据材料描述匹配材料代码.py"
SPEC = importlib.util.spec_from_file_location("material_description_code_matching", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_description_key_ignores_whitespace_case_and_equivalent_punctuation() -> None:
    left = "Flange\tASTM A105；RF，DN50（B）—SCH40×25"
    right = "flangeastma105;rf,dn50(B)-sch40x25"

    assert MODULE.normalize_description_key(left) == MODULE.normalize_description_key(right)


def test_description_key_does_not_remove_non_equivalent_punctuation() -> None:
    assert MODULE.normalize_description_key("DN50/25") != MODULE.normalize_description_key("DN50-25")


def test_matching_keeps_original_output_text() -> None:
    source = pd.DataFrame({"材料描述": ["Flange ASTM A105，RF DN50"]})
    lookup = pd.DataFrame(
        {
            "原始描述": ["flangeastma105,rf\tdn50"],
            "原始总编码": ["F50A105"],
            "分流难度": ["简单"],
            "分流原因": ["原始 Reason"],
        }
    )

    result = MODULE.match_codes(
        source,
        lookup,
        source_desc_col="材料描述",
        source_desc_fallback_col=None,
        lookup_desc_col="原始描述",
        lookup_code_col="原始总编码",
        lookup_second_pass_col="分流难度",
        lookup_reason_col="分流原因",
    )

    assert result.loc[0, "匹配材料代码"] == "F50A105"
    assert result.loc[0, "分流原因"] == "原始 Reason"
    assert result.loc[0, "excel2_原始描述"] == "flangeastma105,rf dn50"


def test_normalized_key_collision_is_reported_as_code_conflict() -> None:
    source = pd.DataFrame({"材料描述": ["A B，C"]})
    lookup = pd.DataFrame(
        {
            "原始描述": ["ab,c", "A B, C"],
            "原始总编码": ["CODE1", "CODE2"],
        }
    )

    result = MODULE.match_codes(
        source,
        lookup,
        source_desc_col="材料描述",
        source_desc_fallback_col=None,
        lookup_desc_col="原始描述",
        lookup_code_col="原始总编码",
        lookup_second_pass_col="分流难度",
        lookup_reason_col="分流原因",
    )

    assert result.loc[0, "匹配命中行数"] == 2
    assert result.loc[0, "编码冲突标记"] == "是"
    assert result.loc[0, "候选材料代码"] == "CODE1 | CODE2"
