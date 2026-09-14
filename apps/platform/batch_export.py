"""大批量编码任务的低内存导出工具。"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple


ResultRow = Tuple[int, int, Dict[str, Any]]
EXPORT_FIELDS = ("TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD")
DIFFICULTY_HEADER = "分流最终难度（0=困难，2=简单）"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _difficulty_level(result: Dict[str, Any]) -> Optional[int]:
    candidates = (
        (result.get("routing") or {}).get("final_level"),
        (result.get("second_pass") or {}).get("final_level"),
        (result.get("difficulty_split") or {}).get("level"),
        (result.get("difficulty_split") or {}).get("difficulty"),
    )
    for raw in candidates:
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return 2 if value == 2 else 0 if value in {0, 1} else None
    return None


def _route_reason(result: Dict[str, Any]) -> str:
    for container in ("routing", "difficulty_split", "second_pass"):
        payload = result.get(container) or {}
        reason = _text(payload.get("reason_text"))
        if reason:
            return reason
        reasons = payload.get("reasons")
        if isinstance(reasons, list):
            joined = " | ".join(_text(item) for item in reasons if _text(item))
            if joined:
                return joined
    errors = result.get("errors") or []
    return _text(errors[0]) if errors else ""


def _format_atom(item: Any) -> str:
    if not isinstance(item, dict):
        return _text(item)
    kind = (_text(item.get("type")) or _text(item.get("TYPE"))).upper()
    value = _text(item.get("value")) or _text(item.get("VALUE"))
    if not value:
        return ""
    if kind == "SCHEDULE":
        return value
    if kind == "MM":
        return f"{value}MM"
    if kind == "INCH":
        return f"INCH{value}"
    return f"{kind}{value}"


def _format_structural(field_type: str, value: Dict[str, Any]) -> str:
    if field_type == "PRESSURE":
        return _text(value.get("PRESSURE"))
    lines = []
    for item in value.get("ITEMS") or []:
        if not isinstance(item, dict):
            continue
        content = " | ".join(
            part for part in (_format_atom(atom) for atom in item.get(field_type) or []) if part
        )
        if not content:
            continue
        scope = _text(item.get("SCOPE")).upper()
        role = _text(item.get("ROLE")).upper()
        label = " ".join(part for part in ((scope if scope != "BODY" else ""), role) if part)
        lines.append(f"{label}: {content}" if label else content)
    if field_type == "SIZE" and _text(value.get("LENGTH")):
        lines.append(f"LENGTH: {_text(value.get('LENGTH'))}")
    return " ; ".join(lines)


def _format_field_value(field_type: str, value: Any) -> str:
    if value is None:
        return ""
    if field_type == "TYPE" and isinstance(value, dict):
        geometry = value.get("GEOMETRY") if isinstance(value.get("GEOMETRY"), dict) else {}
        values = [value.get("FLANGE_STYLE"), value.get("BODY"), geometry.get("ANGLE") or value.get("ANGLE"), geometry.get("RADIUS") or value.get("RADIUS")]
        for key in ("SEAL", "CONN", "ENDS", "MANU"):
            extra = value.get(key)
            values.extend(extra if isinstance(extra, list) else [])
        unique = []
        for item in values:
            text = _text(item)
            if text and text not in unique:
                unique.append(text)
        return ";".join(unique)
    if field_type == "MATERIAL":
        values = value if isinstance(value, list) else [value]
        parts = []
        for item in values:
            if not isinstance(item, dict):
                parts.append(_text(item))
                continue
            role = _text(item.get("PART")) or _text(item.get("ROLE"))
            body = _text(item.get("VALUE")) + "".join(map(_text, item.get("SPECIAL_REQ") or []))
            parts.append(body if not role or role in {"BODY", "MAIN"} else f"{role}:{body}")
        return " ; ".join(part for part in parts if part)
    if field_type == "STANDARD":
        values = value if isinstance(value, list) else [value]
        parts = []
        for item in values:
            if not isinstance(item, dict):
                parts.append(_text(item))
                continue
            body = "".join(_text(item.get(key)) for key in ("BODY", "GRADE", "APPENDIX", "METHOD"))
            category = _text(item.get("CATEGORY"))
            parts.append(f"{body}（{category}）" if category else body)
        return " ; ".join(part for part in parts if part)
    if field_type in {"SIZE", "THICKNESS", "PRESSURE"} and isinstance(value, dict):
        structural = _format_structural(field_type, value)
        if structural:
            return structural
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return _text(value)


def export_headers() -> list[str]:
    return [
        "序号", "项目名称", "分类", "原始描述", "原始总编码", "是否需审核",
        "模型置信分", DIFFICULTY_HEADER, "分流原因",
        *(name for field in EXPORT_FIELDS for name in (f"{field}_原始结果", f"{field}_原始编码")),
    ]


def build_export_record(item: Dict[str, Any], result: Optional[Dict[str, Any]], order_index: int) -> Dict[str, Any]:
    recognized = bool(result and result.get("success"))
    result = result or {}
    confidence = result.get("confidence")
    record: Dict[str, Any] = {
        "序号": int(item.get("index", order_index)) + 1,
        "项目名称": _text(item.get("project_name")),
        "分类": _text(item.get("category")) or _text(result.get("imported_category")) or _text(result.get("material_category")),
        "原始描述": _text(item.get("text")),
        "原始总编码": _text(result.get("final_code")) if recognized else "",
        "是否需审核": ("是" if result.get("need_review") else "否") if recognized else "",
        "模型置信分": f"{float(confidence) * 100:.2f}%" if recognized and confidence is not None else "",
        DIFFICULTY_HEADER: _difficulty_level(result) if recognized else "",
        "分流原因": _route_reason(result) if recognized else "",
    }
    fields = result.get("fields") or {}
    for field_type in EXPORT_FIELDS:
        field = fields.get(field_type) or {}
        value = (field.get("stage2_input") or {}).get("value")
        if value is None:
            value = (field.get("stage1_raw") or {}).get("value")
        record[f"{field_type}_原始结果"] = _format_field_value(field_type, value) if recognized else ""
        record[f"{field_type}_原始编码"] = _text((field.get("stage2_output") or {}).get("code")) if recognized else ""
    return record


def iter_item_results(items: Sequence[Dict[str, Any]], result_rows: Iterable[ResultRow]) -> Iterator[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
    rows = iter(result_rows)
    current = next(rows, None)
    for order_index, item in enumerate(items):
        while current is not None and current[0] < order_index:
            current = next(rows, None)
        result = current[2] if current is not None and current[0] == order_index else None
        if current is not None and current[0] == order_index:
            current = next(rows, None)
        yield order_index, item, result


def build_stage1_row(item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    structural: Optional[Dict[str, Any]] = None
    for field_type, field in (result.get("fields") or {}).items():
        value = (field.get("stage1_raw") or {}).get("value")
        if field_type in {"SIZE", "THICKNESS", "PRESSURE"} and isinstance(value, dict) and isinstance(value.get("ITEMS"), list):
            structural = structural or value
            continue
        output[field_type] = value
    if structural is not None:
        output["ITEMS"] = structural.get("ITEMS") or []
        output["LENGTH"] = structural.get("LENGTH") or ""
        output["PRESSURE"] = structural.get("PRESSURE") or ""
    original = _text(result.get("original_text")) or _text(item.get("text"))
    return {
        "original_input": original,
        "input": _text(result.get("processed_text")) or original,
        "output": output,
    }


def iter_stage1_json(items: Sequence[Dict[str, Any]], result_rows: Iterable[ResultRow]) -> Iterator[bytes]:
    yield "[\n".encode("utf-8")
    first = True
    for _, item, result in iter_item_results(items, result_rows):
        if result is None:
            continue
        if not first:
            yield ",\n".encode("utf-8")
        first = False
        yield ("  " + json.dumps(build_stage1_row(item, result), ensure_ascii=False)).encode("utf-8")
    yield "\n]\n".encode("utf-8")


def iter_csv(items: Sequence[Dict[str, Any]], result_rows: Iterable[ResultRow]) -> Iterator[bytes]:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=export_headers(), lineterminator="\n")
    writer.writeheader()
    yield ("\ufeff" + buffer.getvalue()).encode("utf-8")
    for order_index, item, result in iter_item_results(items, result_rows):
        buffer.seek(0)
        buffer.truncate(0)
        writer.writerow(build_export_record(item, result, order_index))
        yield buffer.getvalue().encode("utf-8")


def write_xlsx(path: Path, items: Sequence[Dict[str, Any]], result_rows: Iterable[ResultRow]) -> None:
    from openpyxl import Workbook

    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("编码结果")
    headers = export_headers()
    sheet.append(headers)
    for order_index, item, result in iter_item_results(items, result_rows):
        record = build_export_record(item, result, order_index)
        sheet.append([record.get(header, "") for header in headers])
    workbook.save(path)
