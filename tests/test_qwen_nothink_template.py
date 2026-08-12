from __future__ import annotations

from apps.hf_lazy_service.server import _build_chat_prompt as build_hf_prompt
from apps.mlx_service.server import _build_chat_prompt as build_mlx_prompt
from src.llm_ner.structured_llamafactory_adapter import _build_chat_prompt


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered-prompt"


def assert_nothink_call(tokenizer: RecordingTokenizer) -> None:
    assert len(tokenizer.calls) == 1
    messages, kwargs = tokenizer.calls[0]
    assert messages == [
        {"role": "system", "content": "system-prompt"},
        {"role": "user", "content": "material-text"},
    ]
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_structured_adapter_uses_tokenizer_nothink_template() -> None:
    tokenizer = RecordingTokenizer()
    assert _build_chat_prompt(tokenizer, "material-text", "system-prompt") == "rendered-prompt"
    assert_nothink_call(tokenizer)


def test_hf_lazy_service_uses_tokenizer_nothink_template() -> None:
    tokenizer = RecordingTokenizer()
    assert build_hf_prompt(tokenizer, "system-prompt", "material-text") == "rendered-prompt"
    assert_nothink_call(tokenizer)


def test_mlx_service_uses_tokenizer_nothink_template() -> None:
    tokenizer = RecordingTokenizer()
    assert build_mlx_prompt(tokenizer, "system-prompt", "material-text") == "rendered-prompt"
    assert_nothink_call(tokenizer)
