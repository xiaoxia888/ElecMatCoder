from __future__ import annotations

import httpx

from apps.vllm_service.compare_vllm_mlx import (
    diagnosis,
    post_predict,
    summarize_phase,
)


def test_post_predict_reads_gateway_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(
            200,
            json={
                "instruction": "same",
                "prompt": "prompt",
                "raw_response": '{"CATEGORY":"管件"}',
                "parsed_json": {"CATEGORY": "管件"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = post_predict(
            client,
            base_url="http://service",
            body={"model": "type", "text": "OLET"},
        )

    assert result["ok"] is True
    assert result["parsed_json"] == {"CATEGORY": "管件"}


def test_phase_summary_uses_canonical_json() -> None:
    vllm = [
        {
            "ok": True,
            "instruction": "same",
            "prompt": [],
            "parsed_json": {"TYPE": {"BODY": "对焊支管台"}, "CATEGORY": "管件"},
        }
    ]
    mlx = [
        {
            "ok": True,
            "instruction": "same",
            "prompt": "raw",
            "parsed_json": {"CATEGORY": "管件", "TYPE": {"BODY": "对焊支管台"}},
        }
    ]

    summary = summarize_phase(vllm, mlx)

    assert summary["instructions_equal"] is True
    assert summary["all_paired_responses_equal"] is True


def test_diagnosis_identifies_template_or_weight_difference() -> None:
    configured = {
        "instructions_equal": True,
        "all_paired_responses_equal": False,
        "vllm": {"stable_across_repeats": True},
        "mlx": {"stable_across_repeats": True},
    }
    forced = {
        "instructions_equal": True,
        "all_paired_responses_equal": False,
        "paired_successes": 1,
        "vllm": {"successes": 1, "stable_across_repeats": True},
        "mlx": {"successes": 1, "stable_across_repeats": True},
    }

    messages = diagnosis(configured, forced)

    assert any("chat template" in message for message in messages)


def test_diagnosis_does_not_call_failed_requests_unstable() -> None:
    configured = {
        "instructions_equal": False,
        "all_paired_responses_equal": False,
        "vllm": {"successes": 1, "stable_across_repeats": True},
        "mlx": {"successes": 1, "stable_across_repeats": True},
    }
    forced = {
        "instructions_equal": False,
        "all_paired_responses_equal": False,
        "paired_successes": 0,
        "vllm": {"successes": 0, "stable_across_repeats": False},
        "mlx": {"successes": 1, "stable_across_repeats": True},
    }

    messages = diagnosis(configured, forced)

    assert any("没有成功配对" in message for message in messages)
    assert not any("vLLM 在 temperature=0 下重复输出不稳定" in message for message in messages)
