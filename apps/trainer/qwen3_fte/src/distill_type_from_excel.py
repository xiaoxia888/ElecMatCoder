from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer_utils.llm.services.deepseek_service import DeepSeekService


# 固定输入配置
INPUT_EXCEL_PATH = PROJECT_ROOT / "apps/trainer/qwen3_fte/data" / "parser_train_main_inputs.xlsx"
INPUT_SHEET_NAME = "Sheet1"
INPUT_TEXT_COLUMN = "材料描述"

# 固定输出配置
OUTPUT_DIR = PROJECT_ROOT / "apps" / "trainer" / "qwen3_fte" / "output"
OUTPUT_JSONL_PATH = OUTPUT_DIR / "type_distill_results.jsonl"
OUTPUT_EXCEL_PATH = OUTPUT_DIR / "type_distill_results.xlsx"
MAX_CONCURRENCY = 5

# 固定提示词路径
PROMPT_PATH = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "prompt"
    / "type_extraction_distill_prompt.txt"
)


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def read_excel_rows() -> pd.DataFrame:
    df = pd.read_excel(INPUT_EXCEL_PATH, sheet_name=INPUT_SHEET_NAME)
    if INPUT_TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"列名不存在: {INPUT_TEXT_COLUMN}. 实际列名: {list(df.columns)}"
        )

    df = df.copy()
    df["_source_text"] = df[INPUT_TEXT_COLUMN].fillna("").astype(str).str.strip()
    df = df[df["_source_text"] != ""].reset_index(drop=True)
    return df


def try_parse_json(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def build_training_record(text: str, parsed: Any | None) -> dict[str, Any]:
    output = parsed if isinstance(parsed, dict) else {}
    return {
        "input": text,
        "output": output,
    }


async def call_deepseek(service: DeepSeekService, prompt: str, text: str) -> str:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    return await service.chat(messages=messages, temperature=0.0, max_tokens=1200)


async def process_row(
    idx: int,
    row: dict[str, Any],
    prompt: str,
    service: DeepSeekService,
    semaphore: asyncio.Semaphore,
    total: int,
) -> tuple[int, dict[str, Any]]:
    text = row["_source_text"]
    print(f"[queued {idx + 1}/{total}] {text[:80]}", flush=True)

    record = {
        "input": text,
        "output": {},
        "type_prompt_path": str(PROMPT_PATH),
        "type_raw_response": "",
        "type_error": "",
    }

    async with semaphore:
        try:
            raw = await call_deepseek(service, prompt, text)
            parsed, error = try_parse_json(raw)
            record["type_raw_response"] = raw
            training_record = build_training_record(text, parsed)
            record["output"] = training_record["output"]
            record["type_error"] = error or ""
        except Exception as exc:  # noqa: BLE001
            record["type_raw_response"] = ""
            record["output"] = {}
            record["type_error"] = str(exc)

    status = "OK" if record["output"] else "ERR"
    detail = record["type_error"][:120] if record["type_error"] else ""
    suffix = f" | {detail}" if detail else ""
    print(f"[done {idx + 1}/{total}] {status}{suffix}", flush=True)

    return idx, record


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt()
    df = read_excel_rows()
    service = DeepSeekService()

    results: list[dict[str, Any]] = []

    total = len(df)
    print(f"读取 {INPUT_EXCEL_PATH}")
    print(f"Sheet: {INPUT_SHEET_NAME}")
    print(f"列名: {INPUT_TEXT_COLUMN}")
    print(f"待处理条数: {total}")
    print(f"并发数: {MAX_CONCURRENCY}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        process_row(
            idx=idx,
            row=row.to_dict(),
            prompt=prompt,
            service=service,
            semaphore=semaphore,
            total=total,
        )
        for idx, row in df.iterrows()
    ]

    indexed_results: list[tuple[int, dict[str, Any]]] = []
    completed = 0
    success_count = 0

    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as f:
        for future in asyncio.as_completed(tasks):
            idx, record = await future
            indexed_results.append((idx, record))
            jsonl_record = {
                "input": record["input"],
                "output": record["output"],
            }
            f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
            f.flush()

            completed += 1
            if record["output"]:
                success_count += 1

            if completed % 10 == 0 or completed == total:
                print(
                    f"进度: {completed}/{total}, 成功 {success_count}, 失败 {completed - success_count}",
                    flush=True,
                )

    indexed_results.sort(key=lambda item: item[0])
    results = [record for _, record in indexed_results]

    excel_rows = []
    for item in results:
        excel_rows.append(
            {
                "input": item["input"],
                "output": json.dumps(item["output"], ensure_ascii=False, indent=2),
                "type_raw_response": item["type_raw_response"],
                "type_error": item["type_error"],
                "type_prompt_path": item["type_prompt_path"],
            }
        )

    out_df = pd.DataFrame(excel_rows)
    out_df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    success_count = int((out_df["output"] != "{}").sum())
    fail_count = total - success_count

    print(f"完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"JSONL: {OUTPUT_JSONL_PATH}")
    print(f"Excel: {OUTPUT_EXCEL_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
