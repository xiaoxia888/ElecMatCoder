# -*- coding: utf-8 -*-
"""通过平台批次任务 ID 读取已有编码结果，仅按当前规则重算分流并导出 Excel。

该脚本不调用模型，不重新编码，也不修改平台中保存的任务结果。

示例：
python scripts/按任务ID重算分流.py \
    --job-id 0123456789abcdef \
    --server-url http://127.0.0.1:8000 \
    --output /Users/guoxi/Downloads/重算分流.xlsx
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.material_description_splitter.routing_pipeline import apply_project_frequency, attach_routing


DIFFICULTY_HEADER = "分流最终难度（0=困难，1=中等，2=简单）"
NEW_DIFFICULTY_HEADER = "新的分流结果"
NEW_REASON_HEADER = "新的原因"
SHEET_NAME = "编码结果"

EXPORT_FIELDS = ("TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD")
EXPORT_HEADERS = [
    "序号",
    "项目名称",
    "分类",
    "原始描述",
    "原始总编码",
    "是否需审核",
    "模型置信分",
    DIFFICULTY_HEADER,
    "分流原因",
    *[column for field in EXPORT_FIELDS for column in (f"{field}_原始结果", f"{field}_原始编码")],
    NEW_DIFFICULTY_HEADER,
    NEW_REASON_HEADER,
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _scalar(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _value_to_text(value: Any, joiner: str = "；") -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value or "—"
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        text = joiner.join(
            item_text
            for item in value
            if (item_text := _value_to_text(item, joiner)) and item_text != "—"
        )
        return text or "—"
    if isinstance(value, dict):
        ordered_items = value.get("ordered_items")
        if isinstance(ordered_items, list):
            ordered_parts: list[str] = []
            for item in ordered_items:
                if not isinstance(item, dict):
                    ordered_parts.append(_value_to_text(item, joiner))
                    continue
                label = " ".join(filter(None, (_scalar(item.get("role")), _scalar(item.get("type")))))
                item_value = _scalar(item.get("value"))
                ordered_parts.append(": ".join(filter(None, (label, item_value))))
            ordered_text = " | ".join(filter(None, ordered_parts))
            if ordered_text:
                return ordered_text

        parts: list[str] = []
        for key, item in value.items():
            if key in {"ordered_items", "thickness_mm_context"}:
                continue
            item_text = _value_to_text(item, joiner)
            if item_text and item_text != "—":
                parts.append(f"{key}: {item_text}")
        return "；".join(parts) or "—"
    return "—"


def _format_type(value: Any) -> str:
    if not isinstance(value, dict):
        return _scalar(value)
    parts: list[str] = []

    def add(values: list[Any]) -> None:
        for item in values:
            text = _scalar(item)
            if text and text not in parts:
                parts.append(text)

    add([value.get("FLANGE_STYLE")])
    geometry = _as_dict(value.get("GEOMETRY"))
    add(
        [
            value.get("BODY"),
            geometry.get("ANGLE") or value.get("ANGLE"),
            geometry.get("RADIUS") or value.get("RADIUS"),
        ]
    )
    add(_as_list(value.get("SEAL")))
    add([*_as_list(value.get("CONN")), *_as_list(value.get("ENDS"))])
    add(_as_list(value.get("MANU")))
    return ";".join(parts)


def _format_material_item(item: Any) -> str:
    if not isinstance(item, dict):
        return _scalar(item)
    part = _scalar(item.get("PART") or item.get("ROLE"))
    value = _scalar(item.get("VALUE"))
    special = "".join(_scalar(entry) for entry in _as_list(item.get("SPECIAL_REQ")) if _scalar(entry))
    body = f"{value}{special}"
    if not part or part in {"BODY", "MAIN"}:
        return body
    return f"{part}:{body}" if body else part


def _format_standard_item(item: Any) -> str:
    if not isinstance(item, dict):
        return _scalar(item)
    main = "".join(
        filter(
            None,
            (
                _scalar(item.get("BODY")),
                _scalar(item.get("GRADE")),
                _scalar(item.get("APPENDIX")),
                _scalar(item.get("METHOD")),
            ),
        )
    )
    category = _scalar(item.get("CATEGORY"))
    return f"{main}（{category}）" if category else main


def _format_field_value(field_type: str, value: Any) -> str:
    if value is None:
        return "—"
    if field_type == "TYPE":
        return _format_type(value) or "—"
    if field_type == "MATERIAL":
        items = value if isinstance(value, list) else [value]
        return " ; ".join(filter(None, (_format_material_item(item) for item in items))) or "—"
    if field_type == "STANDARD":
        items = value if isinstance(value, list) else [value]
        return " ; ".join(filter(None, (_format_standard_item(item) for item in items))) or "—"
    if field_type == "PRESSURE":
        if not isinstance(value, dict):
            return _scalar(value) or "—"
        parts: list[str] = []
        for item in _as_list(value.get("items")):
            if isinstance(item, dict):
                text = ": ".join(filter(None, (_scalar(item.get("type")), _scalar(item.get("value")))))
            else:
                text = _scalar(item)
            if text:
                parts.append(text)
        return " ; ".join(parts) or "—"
    if field_type in {"SIZE", "THICKNESS"}:
        if not isinstance(value, dict):
            return _scalar(value) or "—"
        parts: list[str] = []
        for item in _as_list(value.get("ordered_items")):
            if not isinstance(item, dict):
                continue
            label = " ".join(filter(None, (_scalar(item.get("role")), _scalar(item.get("type")))))
            item_value = _scalar(item.get("value"))
            if item_value:
                parts.append(f"{label}: {item_value}".strip())
        return " | ".join(parts) or "—"
    return _value_to_text(value)


def _difficulty_level(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    candidates = (
        _as_dict(result.get("routing")).get("final_level"),
        _as_dict(result.get("second_pass")).get("final_level"),
        _as_dict(result.get("difficulty_split")).get("level"),
        _as_dict(result.get("difficulty_split")).get("difficulty"),
    )
    for value in candidates:
        if value is None or value == "":
            continue
        try:
            level = int(value)
        except (TypeError, ValueError):
            continue
        if level in {0, 1, 2}:
            return level
    return None


def _route_reason(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    routing_reason = _clean(_as_dict(result.get("routing")).get("reason_text"))
    if routing_reason:
        return routing_reason

    level = _difficulty_level(result)
    difficulty_split = _as_dict(result.get("difficulty_split"))
    stage1_reason = _clean(difficulty_split.get("reason_text"))
    if not stage1_reason:
        stage1_reason = " | ".join(filter(None, (_clean(item) for item in _as_list(difficulty_split.get("reasons")))))
    errors = _as_list(result.get("errors"))
    stage1_error = _clean(errors[0]) if errors else ""
    if level == 0:
        return stage1_reason or stage1_error or "未提供一阶段分流原因"

    second_pass = _as_dict(result.get("second_pass"))
    reason_parts: list[str] = []
    for check in _as_list(second_pass.get("failed_checks")):
        if not isinstance(check, dict):
            continue
        field = _clean(check.get("field"))
        reason = _clean(check.get("reason"))
        if reason:
            reason_parts.append(f"{field}: {reason}" if field else reason)
    if reason_parts:
        return " | ".join(reason_parts)
    if level == 1 or second_pass.get("final_level") is not None:
        return "未提供二次分流原因"
    if level == 2:
        return "无需二次分流原因"
    return stage1_reason or stage1_error or "未提供原因说明"


def _format_percent(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _category(item: dict[str, Any], result: dict[str, Any]) -> str:
    return _clean(item.get("category")) or _clean(result.get("imported_category")) or _clean(result.get("material_category"))


def _field_export_values(result: dict[str, Any], field_type: str) -> tuple[str, str]:
    field = _as_dict(_as_dict(result.get("fields")).get(field_type))
    if not field:
        return "", ""
    stage2_input = _as_dict(field.get("stage2_input"))
    stage1_raw = _as_dict(field.get("stage1_raw"))
    value = stage2_input.get("value") if "value" in stage2_input else stage1_raw.get("value")
    display = _format_field_value(field_type, value)
    code = _clean(_as_dict(field.get("stage2_output")).get("code")) or "—"
    return ("" if display == "—" else display, "" if code == "—" else code)


def _build_job_url(server_url: str, job_id: str) -> str:
    base = server_url.rstrip("/")
    suffix = f"/pipe/encode/batch/jobs/{quote(job_id, safe='')}" if base.endswith("/api") else f"/api/pipe/encode/batch/jobs/{quote(job_id, safe='')}"
    return f"{base}{suffix}"


def fetch_job(server_url: str, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    url = _build_job_url(server_url, job_id)
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise RuntimeError(f"读取批次任务失败（HTTP {exc.response.status_code}）: {detail}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"无法连接平台服务 {server_url}: {exc}") from exc

    payload = response.json()
    job = payload.get("job") if isinstance(payload, dict) else None
    if not isinstance(job, dict):
        raise RuntimeError("批次任务接口未返回有效的 job 数据")
    return job


def recompute_routing(job: dict[str, Any]) -> list[dict[str, Any]]:
    """使用当前分流规则重算，返回与 items 顺序一致的结果副本。"""
    items = job.get("items") if isinstance(job.get("items"), list) else []
    stored_results = job.get("results") if isinstance(job.get("results"), dict) else {}
    recalculated: list[dict[str, Any]] = []

    for order_index, _item in enumerate(items):
        stored = stored_results.get(str(order_index), stored_results.get(order_index))
        result = copy.deepcopy(stored) if isinstance(stored, dict) else {}
        if result.get("success") and not result.get("skipped_encoding"):
            result = attach_routing(result)
        recalculated.append(result)

    project_names = [_clean(_as_dict(item).get("project_name")) for item in items]
    return apply_project_frequency(recalculated, project_names)


def build_export_dataframe(job: dict[str, Any]) -> pd.DataFrame:
    items = job.get("items") if isinstance(job.get("items"), list) else []
    stored_results = job.get("results") if isinstance(job.get("results"), dict) else {}
    recalculated = recompute_routing(job)
    rows: list[dict[str, Any]] = []

    for order_index, raw_item in enumerate(items):
        item = _as_dict(raw_item)
        stored = stored_results.get(str(order_index), stored_results.get(order_index))
        result = stored if isinstance(stored, dict) else {}
        new_result = recalculated[order_index] if order_index < len(recalculated) else {}
        recognized = bool(result.get("success"))
        row: dict[str, Any] = {
            "序号": int(item.get("index", order_index)) + 1,
            "项目名称": _clean(item.get("project_name")),
            "分类": _category(item, result),
            "原始描述": _clean(item.get("text")) or _clean(result.get("original_text")),
            "原始总编码": _clean(result.get("final_code")) if recognized else "",
            "是否需审核": ("是" if result.get("need_review") else "否") if recognized else "",
            "模型置信分": _format_percent(result.get("confidence")) if recognized else "",
            DIFFICULTY_HEADER: _difficulty_level(result) if recognized else "",
            "分流原因": _route_reason(result) if recognized else "",
        }
        for field_type in EXPORT_FIELDS:
            raw_value, code = _field_export_values(result, field_type) if recognized else ("", "")
            row[f"{field_type}_原始结果"] = raw_value
            row[f"{field_type}_原始编码"] = code
        row[NEW_DIFFICULTY_HEADER] = _difficulty_level(new_result) if recognized else ""
        row[NEW_REASON_HEADER] = _route_reason(new_result) if recognized else ""
        rows.append(row)

    return pd.DataFrame(rows, columns=EXPORT_HEADERS)


def export_job(job: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = build_export_dataframe(job)
    dataframe.to_excel(output_path, index=False, sheet_name=SHEET_NAME)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通过批次任务 ID 读取已有编码结果，仅重算分流并导出平台格式 Excel。"
    )
    parser.add_argument("--job-id", required=True, help="平台批次任务 ID")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000", help="平台后端地址")
    parser.add_argument("--output", help="输出 Excel 路径，默认为当前目录下的 编码结果_<job_id>_重算分流.xlsx")
    parser.add_argument("--timeout", type=float, default=120.0, help="读取批次任务的超时秒数")
    args = parser.parse_args()

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path.cwd() / f"编码结果_{args.job_id}_重算分流.xlsx"
    )
    job = fetch_job(args.server_url, args.job_id, timeout=args.timeout)
    export_job(job, output_path)
    print(f"已生成: {output_path}")
    print(f"任务条数: {len(job.get('items') or [])}；未调用模型，未修改原任务。")


if __name__ == "__main__":
    main()
