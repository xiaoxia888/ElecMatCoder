import copy
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "按任务ID重算分流.py"
SPEC = importlib.util.spec_from_file_location("job_routing_export", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _sample_job():
    description = "3/4'' SOCKET WELD FLANGE ASTM A105 150# RF (3.2 - 6.3 μm Ra)"
    return {
        "items": [
            {
                "index": 0,
                "text": description,
                "project_name": "测试项目",
                "category": "法兰",
            }
        ],
        "results": {
            "0": {
                "original_text": description,
                "processed_text": description,
                "success": True,
                "need_review": False,
                "confidence": 0.88,
                "final_code": "FS20C150A105AB165",
                "material_category": "法兰",
                "routing": {"final_level": 2, "reason_text": "二次分流全部通过"},
                "difficulty_split": {"level": 1, "reason_text": ""},
                "second_pass": {"final_level": 2, "failed_checks": []},
                "fields": {
                    "TYPE": {
                        "stage1_raw": {"value": {"BODY": "承插焊法兰", "SEAL": ["RF"]}},
                        "stage2_input": {"value": {"BODY": "承插焊法兰", "SEAL": ["RF"]}},
                        "stage2_output": {"code": "FS"},
                    },
                    "SIZE": {
                        "stage1_raw": {"value": {"ordered_items": [{"type": "INCH", "value": "3/4"}]}},
                        "stage2_input": {"value": {"ordered_items": [{"type": "INCH", "value": "3/4"}]}},
                        "stage2_output": {"code": "20"},
                    },
                    "PRESSURE": {
                        "stage1_raw": {"value": "C150"},
                        "stage2_input": {"value": "C150"},
                        "stage2_output": {"code": "C150"},
                    },
                    "MATERIAL": {
                        "stage1_raw": {"value": [{"PART": "BODY", "VALUE": "ASTM A105"}]},
                        "stage2_input": {"value": [{"PART": "BODY", "VALUE": "ASTM A105"}]},
                        "stage2_output": {"code": "A105"},
                    },
                    "STANDARD": {
                        "stage1_raw": {"value": [{"BODY": "AB165", "CATEGORY": "制造"}]},
                        "stage2_input": {"value": [{"BODY": "AB165", "CATEGORY": "制造"}]},
                        "stage2_output": {"code": "AB165"},
                    },
                },
            }
        },
    }


def test_recompute_does_not_mutate_stored_job_and_adds_only_two_columns():
    job = _sample_job()
    original = copy.deepcopy(job)

    dataframe = MODULE.build_export_dataframe(job)

    assert job == original
    assert dataframe.columns.tolist()[-2:] == ["新的分流结果", "新的原因"]
    assert dataframe.loc[0, "分流最终难度（0=困难，1=中等，2=简单）"] == 2
    assert dataframe.loc[0, "新的分流结果"] == 0
    assert "括号" in dataframe.loc[0, "新的原因"]


def test_export_has_platform_sheet_and_column_order(tmp_path):
    output_path = tmp_path / "result.xlsx"

    MODULE.export_job(_sample_job(), output_path)

    workbook = pd.ExcelFile(output_path)
    assert workbook.sheet_names == ["编码结果"]
    dataframe = pd.read_excel(output_path)
    assert dataframe.columns.tolist() == MODULE.EXPORT_HEADERS
    assert dataframe.loc[0, "序号"] == 1
    assert dataframe.loc[0, "项目名称"] == "测试项目"
    assert dataframe.loc[0, "模型置信分"] == "88.00%"
