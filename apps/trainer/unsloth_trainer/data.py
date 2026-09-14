from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def build_messages(sample: dict[str, Any], data_config: dict[str, Any]) -> list[dict[str, str]]:
    columns = data_config["columns"]
    data_format = data_config["format"]

    if data_format == "chatml":
        messages = sample.get(columns["messages"])
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"ChatML 样本缺少非空 {columns['messages']} 数组")
        normalized = []
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("每条 ChatML message 必须包含 role 和 content")
            normalized.append({"role": str(message["role"]), "content": _as_text(message["content"])})
        return normalized

    if data_format != "alpaca":
        raise ValueError("text 格式不需要构造 messages")

    instruction_key = columns["instruction"]
    output_key = columns["output"]
    if instruction_key not in sample or output_key not in sample:
        raise ValueError(f"Alpaca 样本必须包含 {instruction_key} 和 {output_key}")

    instruction = _as_text(sample[instruction_key]).strip()
    input_text = _as_text(sample.get(columns["input"])).strip()
    if input_text:
        instruction = f"{instruction}{data_config['input_separator']}{input_text}"

    messages: list[dict[str, str]] = []
    system_prompt = _as_text(data_config.get("system_prompt")).strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": _as_text(sample[output_key])},
        ]
    )
    return messages


def format_batch(batch: dict[str, list[Any]], tokenizer: Any, data_config: dict[str, Any]) -> dict[str, list[str]]:
    if data_config["format"] == "text":
        text_key = data_config["columns"]["text"]
        if text_key not in batch:
            raise ValueError(f"text 数据集缺少 {text_key} 列")
        return {"text": [_as_text(value) for value in batch[text_key]]}

    keys = list(batch)
    size = len(batch[keys[0]]) if keys else 0
    texts: list[str] = []
    for index in range(size):
        sample = {key: values[index] for key, values in batch.items()}
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        enable_thinking = data_config.get("enable_thinking")
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            text = tokenizer.apply_chat_template(build_messages(sample, data_config), **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            kwargs.pop("enable_thinking", None)
            text = tokenizer.apply_chat_template(build_messages(sample, data_config), **kwargs)
        texts.append(text)
    return {"text": texts}


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(4096).lstrip("\ufeff \t\r\n")
        handle.seek(0)
        if prefix.startswith("["):
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"JSON 文件顶层必须是数组: {path}")
            yield from payload
            return

        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行无效: {path}") from exc


def validate_data_file(path: Path, data_config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    try:
        sample = next(iter(iter_json_records(path)))
    except StopIteration as exc:
        raise ValueError(f"数据文件为空: {path}") from exc
    if not isinstance(sample, dict):
        raise ValueError(f"数据样本必须是对象: {path}")

    if data_config["format"] == "text":
        text_key = data_config["columns"]["text"]
        if text_key not in sample:
            raise ValueError(f"text 数据集缺少 {text_key} 列: {path}")
    else:
        build_messages(sample, data_config)
    return sample


def load_and_format_datasets(
    train_path: Path,
    validation_path: Path | None,
    tokenizer: Any,
    data_config: dict[str, Any],
) -> tuple[Any, Any | None]:
    from datasets import load_dataset

    files = {"train": str(train_path)}
    if validation_path is not None:
        files["validation"] = str(validation_path)
    datasets = load_dataset("json", data_files=files)

    formatted = {}
    for split_name, dataset in datasets.items():
        formatted[split_name] = dataset.map(
            format_batch,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "data_config": data_config},
            remove_columns=dataset.column_names,
            num_proc=int(data_config.get("num_proc", 1)),
            desc=f"格式化 {split_name} 数据",
        )
    return formatted["train"], formatted.get("validation")
