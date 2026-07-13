#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
独立导出脚本：
1. 调用氚云 LoadBizObjects 分页接口拉取数据
2. 按配置提取主表 / 子表字段
3. 导出为“字段做列”的 CSV 或 XLSX 文件

输出格式示例：
第一行：字段编码/路径
第二行：字段中文名
第三行开始：字段值

例如：
A1 = F0000001
A2 = 通用材料代码
A3 = 90ELS32C300020GBT14383IINBT47008
B1 = F0000002
B2 = 材料代码
...

不依赖当前项目任何代码。
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SCHEMA_CODE = "D148357d3274485dac145f38eddea861e740eaf"
DEFAULT_BATCH_SIZE = 2000 # 单次请求最大条数
DEFAULT_API_BATCH_SIZE = 500 # 单次请求上限，氚云分页接口单词最多返回500条
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3


def print_progress(message: str) -> None:
    print(message, flush=True)


_LAST_DYNAMIC_LENGTH = 0


def print_dynamic_progress(message: str) -> None:
    global _LAST_DYNAMIC_LENGTH
    padded = message
    if _LAST_DYNAMIC_LENGTH > len(message):
        padded = message + (" " * (_LAST_DYNAMIC_LENGTH - len(message)))
    sys.stdout.write("\r" + padded)
    sys.stdout.flush()
    _LAST_DYNAMIC_LENGTH = len(message)


def finish_dynamic_progress() -> None:
    global _LAST_DYNAMIC_LENGTH
    if _LAST_DYNAMIC_LENGTH > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
        _LAST_DYNAMIC_LENGTH = 0


def call_h3_api(
    base_url: str,
    engine_code: str,
    engine_secret: str,
    payload: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=base_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "EngineCode": engine_code,
            "EngineSecret": engine_secret,
        },
        method="POST",
    )
    last_error: Exception | None = None
    total_attempts = max(int(retries), 1)
    action_name = str(payload.get("ActionName") or "")

    for attempt in range(1, total_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
            result = json.loads(text)
            if not result.get("Successful", False):
                raise RuntimeError(result.get("ErrorMessage") or "氚云接口调用失败")
            return result
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt >= total_attempts:
                break
            wait_seconds = min(2 * attempt, 8)
            finish_dynamic_progress()
            print_progress(
                f"[重试] {action_name or 'H3 API'} 第 {attempt}/{total_attempts} 次失败：{exc}，{wait_seconds}s 后重试"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"{action_name or 'H3 API'} 调用失败，已重试 {total_attempts} 次，最后错误：{last_error}"
    ) from last_error


def build_filter_payload(
    from_row_num: int,
    to_row_num: int,
    matcher: dict[str, Any] | None = None,
    sort_by_collection: list[dict[str, str]] | None = None,
    require_count: bool = True,
    return_items: list[str] | None = None,
    page_size_limit: int = DEFAULT_API_BATCH_SIZE,
) -> str:
    if to_row_num - from_row_num > page_size_limit:
        raise ValueError(f"ToRowNum - FromRowNum 不能大于 {page_size_limit}")

    payload = {
        "FromRowNum": from_row_num,
        "ToRowNum": to_row_num,
        "Matcher": matcher or {"Type": "And", "Matchers": [{"Type": "And", "Matchers": []}]},
        "SortByCollection": sort_by_collection
        or [{"ItemName": "CreatedTime", "Direction": "Ascending"}],
        "RequireCount": require_count,
        "ReturnItems": return_items or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def load_all_biz_objects(
    base_url: str,
    engine_code: str,
    engine_secret: str,
    schema_code: str,
    matcher: dict[str, Any] | None = None,
    sort_by_collection: list[dict[str, str]] | None = None,
    return_items: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    all_rows: list[dict[str, Any]] = []
    from_row = 0
    total_count: int | None = None
    page_index = 1

    while True:
        logical_page_rows: list[dict[str, Any]] = []
        logical_page_start = from_row
        logical_page_end = from_row + batch_size

        while len(logical_page_rows) < batch_size:
            request_size = min(DEFAULT_API_BATCH_SIZE, batch_size - len(logical_page_rows))
            to_row = from_row + request_size
            payload = {
                "ActionName": "LoadBizObjects",
                "SchemaCode": schema_code,
                "Filter": build_filter_payload(
                    from_row_num=from_row,
                    to_row_num=to_row,
                    matcher=matcher,
                    sort_by_collection=sort_by_collection,
                    require_count=True,
                    return_items=return_items,
                    page_size_limit=DEFAULT_API_BATCH_SIZE,
                ),
            }
            response = call_h3_api(
                base_url=base_url,
                engine_code=engine_code,
                engine_secret=engine_secret,
                payload=payload,
                timeout=timeout,
                retries=retries,
            )
            return_data = response.get("ReturnData") or {}
            current_rows = return_data.get("BizObjectArray") or []
            if total_count is None:
                total_count_raw = return_data.get("TotalCount")
                if total_count_raw is not None:
                    try:
                        total_count = int(total_count_raw)
                    except Exception:
                        total_count = None
                if total_count is not None:
                    finish_dynamic_progress()
                    print_progress(f"[总量] 接口返回总条数：{total_count}")
            if not current_rows:
                finish_dynamic_progress()
                print_progress("[分页完成] 当前页无数据，停止继续请求")
                return all_rows

            logical_page_rows.extend(current_rows)
            all_rows.extend(current_rows)
            from_row += len(current_rows)

            if total_count is not None:
                percent = min(len(all_rows) / total_count * 100, 100.0) if total_count > 0 else 100.0
                print_dynamic_progress(
                    f"[抓取进度] 第 {page_index} 页 {logical_page_start}-{logical_page_end}，已拉取 {len(all_rows)}/{total_count} 条，完成 {percent:.2f}%"
                )
            else:
                print_dynamic_progress(
                    f"[抓取进度] 第 {page_index} 页 {logical_page_start}-{logical_page_end}，已拉取 {len(all_rows)} 条"
                )

            if len(current_rows) < request_size:
                finish_dynamic_progress()
                print_progress(
                    f"[分页完成] 当前子请求返回 {len(current_rows)} 条，小于请求大小 {request_size}，停止继续请求"
                )
                return all_rows
            if total_count is not None and len(all_rows) >= total_count:
                finish_dynamic_progress()
                print_progress(f"[分页完成] 已达到接口总条数 {total_count}，停止继续请求")
                return all_rows

        page_index += 1

    finish_dynamic_progress()
    return all_rows


def extract_value(record: dict[str, Any], field_path: str, child_joiner: str = "\n") -> str:
    if "." not in field_path:
        value = record.get(field_path)
        return "" if value is None else str(value)

    child_schema, child_field = field_path.split(".", 1)
    child_rows = record.get(child_schema) or []
    if not isinstance(child_rows, list):
        return ""

    values: list[str] = []
    for row in child_rows:
        if not isinstance(row, dict):
            continue
        value = row.get(child_field)
        if value is None:
            continue
        text = str(value).strip()
        if text == "":
            continue
        values.append(text)
    return child_joiner.join(values)


def _get_child_schema_code_counts(
    field_config: dict[str, str],
    main_schema_code: str,
) -> dict[str, int]:
    child_schema_code_counts: dict[str, int] = {}
    for field_path in field_config.values():
        if "." not in field_path:
            continue
        schema_code, _ = field_path.split(".", 1)
        if schema_code == main_schema_code:
            continue
        child_schema_code_counts[schema_code] = child_schema_code_counts.get(schema_code, 0) + 1
    return child_schema_code_counts


def flatten_records_for_export(
    records: list[dict[str, Any]],
    field_config: dict[str, str],
    main_schema_code: str,
    child_joiner: str = "\n",
) -> list[dict[str, str]]:
    child_schema_code_counts = _get_child_schema_code_counts(field_config, main_schema_code)
    child_schema_code = ""
    if child_schema_code_counts:
        child_schema_code = max(
            child_schema_code_counts,
            key=lambda schema_code: child_schema_code_counts[schema_code],
        )
    if len(child_schema_code_counts) > 1:
        finish_dynamic_progress()
        print_progress(
            "[多子表] 检测到多个子表Schema，已自动选择字段数最多的子表按行展开："
            f"{child_schema_code}。其余子表字段将按 {repr(child_joiner)} 合并到单元格。"
        )

    rows: list[dict[str, str]] = []
    warned_missing_child_schema = False

    for record_index, record in enumerate(records, start=1):
        if record_index % 500 == 0:
            print_dynamic_progress(f"[展开数据] 已处理主表 {record_index}/{len(records)} 条")

        if not child_schema_code:
            row: dict[str, str] = {}
            for label, field_path in field_config.items():
                row[label] = extract_value(record, field_path, child_joiner=child_joiner)
            rows.append(row)
            continue

        child_rows = record.get(child_schema_code) or []
        if (
            child_schema_code
            and not warned_missing_child_schema
            and child_schema_code not in record
        ):
            warned_missing_child_schema = True
            finish_dynamic_progress()
            print_progress(
                f"[警告] 主表返回结果里未找到子表 SchemaCode={child_schema_code}。当前记录可用键示例："
                f"{', '.join(sorted(record.keys())[:20])}"
            )
        static_values: dict[str, str] = {}
        primary_child_fields: list[tuple[str, str]] = []
        for label, field_path in field_config.items():
            if "." not in field_path:
                static_values[label] = extract_value(record, field_path, child_joiner=child_joiner)
                continue

            current_schema, child_field = field_path.split(".", 1)
            if current_schema == main_schema_code:
                static_values[label] = extract_value(record, child_field, child_joiner=child_joiner)
                continue
            if current_schema != child_schema_code:
                static_values[label] = extract_value(record, field_path, child_joiner=child_joiner)
                continue
            primary_child_fields.append((label, child_field))

        if not isinstance(child_rows, list) or len(child_rows) == 0:
            row = dict(static_values)
            for label, _ in primary_child_fields:
                row[label] = ""
            rows.append(row)
            continue

        for child_row in child_rows:
            row = dict(static_values)
            for label, child_field in primary_child_fields:
                value = child_row.get(child_field) if isinstance(child_row, dict) else None
                row[label] = "" if value is None else str(value)
            rows.append(row)

    finish_dynamic_progress()
    return rows


def build_export_matrix(
    flattened_rows: list[dict[str, str]],
    field_config: dict[str, str],
) -> list[list[str]]:
    matrix: list[list[str]] = []
    matrix.append(list(field_config.values()))
    matrix.append(list(field_config.keys()))

    total_rows = len(flattened_rows)
    for index, row_dict in enumerate(flattened_rows, start=1):
        if index % 1000 == 0:
            print_dynamic_progress(f"[组装矩阵] 已写入数据行 {index}/{total_rows}")
        row = [row_dict.get(label, "") for label in field_config.keys()]
        matrix.append(row)
    finish_dynamic_progress()
    return matrix


def export_to_csv(matrix: list[list[str]], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print_progress(f"[写文件] 开始写入 CSV：{output_path}")
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)
    print_progress(f"[写文件] CSV 写入完成：{output_path}")


def export_to_xlsx(matrix: list[list[str]], output_file: str) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment
    except ImportError as exc:
        raise RuntimeError("导出 xlsx 需要先安装 openpyxl：pip install openpyxl") from exc

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print_progress(f"[写文件] 开始写入 XLSX：{output_path}")

    wb = Workbook()
    ws = wb.active
    ws.title = "materials"
    default_row_height = 15
    ws.sheet_format.defaultRowHeight = default_row_height

    for row_index, row in enumerate(matrix, start=1):
        if row_index % 1000 == 0:
            print_dynamic_progress(f"[写文件] XLSX 已写入 {row_index} 行")
        ws.row_dimensions[row_index].height = default_row_height
        for col_index, value in enumerate(row, start=1):
            cell = ws.cell(row=row_index, column=col_index, value=value)
            cell.alignment = Alignment(wrap_text=False, vertical="top")

    wb.save(output_path)
    finish_dynamic_progress()
    print_progress(f"[写文件] XLSX 写入完成：{output_path}")


def export_material_columns(
    base_url: str,
    engine_code: str,
    engine_secret: str,
    output_file: str,
    field_config: dict[str, str],
    schema_code: str = DEFAULT_SCHEMA_CODE,
    matcher: dict[str, Any] | None = None,
    sort_by_collection: list[dict[str, str]] | None = None,
    return_items: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    child_joiner: str = "\n",
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    print_progress("[开始] 开始导出氚云表单数据")
    records = load_all_biz_objects(
        base_url=base_url,
        engine_code=engine_code,
        engine_secret=engine_secret,
        schema_code=schema_code,
        matcher=matcher,
        sort_by_collection=sort_by_collection,
        return_items=return_items,
        batch_size=batch_size,
        timeout=timeout,
        retries=retries,
    )

    print_progress(f"[数据抓取完成] 共获取 {len(records)} 条主表记录，开始按子表展开")
    flattened_rows = flatten_records_for_export(
        records,
        field_config,
        main_schema_code=schema_code,
        child_joiner=child_joiner,
    )
    print_progress(f"[展开完成] 共得到 {len(flattened_rows)} 条导出数据行，开始组装导出矩阵")
    matrix = build_export_matrix(flattened_rows, field_config)
    suffix = Path(output_file).suffix.lower()
    if suffix == ".xlsx":
        export_to_xlsx(matrix, output_file)
    else:
        export_to_csv(matrix, output_file)

    print_progress("[完成] 导出结束")
    return {
        "record_count": len(records),
        "export_row_count": len(flattened_rows),
        "column_count": len(field_config),
        "output_file": str(Path(output_file).resolve()),
    }


if __name__ == "__main__":
    FILE_NAME = "materials_export0701.csv"
    FIELD_CONFIG = {
        "主表id":"ObjectId",
        "Status":"Status",
        "子表id":"D148357F08a29fd65d324a4bb3fc0f676517003e.ObjectId",
        # "定额材料代码子表Id":"D148357F8c8b39581c3c415c89a0acda63cedd90.ObjectId",
        # "定额材料代码（工序）子表id":"D1483572dbd20125eba43088c381a36493f59cf.ObjectId",
        # "材料描述":"D148357F17c2e0548b94497f873300934ea06164.F0000003",
        # "材料描述(多行)":"D148357F17c2e0548b94497f873300934ea06164.F0000056",
        # "项目名称": "D148357F17c2e0548b94497f873300934ea06164.F0000049",
        # "分类":"D148357F17c2e0548b94497f873300934ea06164.F0000050",
        # "国标美标标记":"D148357F17c2e0548b94497f873300934ea06164.F0000051",
        # "编码":"D148357F17c2e0548b94497f873300934ea06164.F0000004",
        # "修正编码":"D148357F17c2e0548b94497f873300934ea06164.F0000008",

        # "原始种类":"D148357F17c2e0548b94497f873300934ea06164.F0000009",
        # "标准化种类":"D148357F17c2e0548b94497f873300934ea06164.F0000010",
        # "修正种类":"D148357F17c2e0548b94497f873300934ea06164.F0000011",
        
        # "压力等级代码":"F0000005",
        # "材质代码":"F0000006",
        # "标准号代码":"F0000007",
        
        # "原始尺寸":"D148357F17c2e0548b94497f873300934ea06164.F0000012",
        # "标准化尺寸":"D148357F17c2e0548b94497f873300934ea06164.F0000013",
        # "修正尺寸":"D148357F17c2e0548b94497f873300934ea06164.F0000014",

        # "原始壁厚":"D148357F17c2e0548b94497f873300934ea06164.F0000015",
        # "标准化壁厚":"D148357F17c2e0548b94497f873300934ea06164.F0000016",
        # "修正壁厚":"D148357F17c2e0548b94497f873300934ea06164.F0000017",

        # "原始磅级":"D148357F17c2e0548b94497f873300934ea06164.F0000018",
        # "标准化磅级":"D148357F17c2e0548b94497f873300934ea06164.F0000019",
        # "修正磅级":"D148357F17c2e0548b94497f873300934ea06164.F0000020",

        # "原始材质":"D148357F17c2e0548b94497f873300934ea06164.F0000021",
        # "标准化材质":"D148357F17c2e0548b94497f873300934ea06164.F0000022",
        # "修正材质":"D148357F17c2e0548b94497f873300934ea06164.F0000023",

        # "原始规范":"D148357F17c2e0548b94497f873300934ea06164.F0000024",
        # "标准化规范":"D148357F17c2e0548b94497f873300934ea06164.F0000025",
        # "修正规范":"D148357F17c2e0548b94497f873300934ea06164.F0000026",

        # "备注":"D148357F17c2e0548b94497f873300934ea06164.F0000058",

        # "二次核对":"D148357F17c2e0548b94497f873300934ea06164.F0000069"
        "材料描述":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000010",
        "材料描述(多行)":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000055",
        "项目简称":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000028",
        "分类":"F0000027",
        "通用材料代码":"F0000001",
        "通用材料代码（子表）":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000024",
        "本项目材料代码":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000053",
        # "定额材料代码": "D148357F8c8b39581c3c415c89a0acda63cedd90.F0000056",
        "是否再次运行":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000120",
        "是否一致":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000121",
        "C1-名称简写":"F0000002",
        "C1-规格简写":"F0000003",
        "C1-壁厚简写":"F0000004",
        "C1-压力等级简写":"F0000005",
        "C1-材质简写":"F0000006",
        "C1-标准简写":"F0000007",
        "壁厚转换标记":"F0000081",
        "材质分类":"F0000082",
        "B1写入时间":"F0000077",
        "设计院":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000022",
        "规格":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000029",
        "阶段":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000054",
        "B1写入时间":"D148357F08a29fd65d324a4bb3fc0f676517003e.F0000079"
    }

    MATCHER = {"Type": "And", "Matchers": [{"Type": "And", "Matchers": []}]}
    SORT_BY = [{"ItemName": "CreatedTime", "Direction": "Ascending"}]

    result = export_material_columns(
        base_url="https://www.h3yun.com/OpenApi/Invoke",
        engine_code="ety58sf4upb95mibri9qatvi5",
        engine_secret="u3toDgywMDwgbYgnrHNmjzH0g0fzn9mWAj0PY659taS7sxeVPoor5g==",
        output_file=FILE_NAME,
        field_config=FIELD_CONFIG,
        schema_code=DEFAULT_SCHEMA_CODE,
        matcher=MATCHER,
        sort_by_collection=SORT_BY,
        return_items=[],
        batch_size=DEFAULT_BATCH_SIZE,
        child_joiner="\n",
        timeout=DEFAULT_TIMEOUT,
        retries=DEFAULT_RETRIES,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
