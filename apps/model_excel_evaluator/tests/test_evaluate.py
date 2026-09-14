from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from apps.model_excel_evaluator.evaluate import (
    ManagedVLLMService,
    TaskEncoder,
    clean_code_output,
    draw_analysis,
    normalize_code,
    model_supports_task,
    normalize_stage_backend,
    normalize_task,
    parse_json_output,
    parse_named_overrides,
    prepare_runtime,
    resolve_adapter,
    resolve_full_stages,
    merge_full_stage_records,
    run_stage1_jobs,
    run_vllm_stage1,
    select_task_model_config,
    stage_model_signature,
)


class EvaluateHelpersTest(unittest.TestCase):
    def test_task_aliases(self):
        self.assertEqual(normalize_task("种类"), "type")
        self.assertEqual(normalize_task("尺寸"), "size")
        self.assertEqual(normalize_task("磅级"), "pressure")
        self.assertEqual(normalize_task("完整编码"), "full")
        self.assertEqual(normalize_task("完整"), "full")

    def test_full_stages_require_exact_pipeline(self):
        self.assertEqual(resolve_full_stages({}), ("type", "material", "size"))
        with self.assertRaises(ValueError):
            resolve_full_stages({"stages": ["type", "size"]})

    def test_managed_vllm_backend_aliases(self):
        self.assertEqual(normalize_stage_backend("vllm"), "managed_vllm")
        self.assertEqual(normalize_stage_backend("vllm_service"), "managed_vllm")
        self.assertEqual(normalize_stage_backend("transformers"), "local_transformers")

    def test_task_specific_managed_routes(self):
        model = {
            "routes": {
                "type": {"served_model": "type"},
                "size": {"served_model": "size-thick-pressure"},
            }
        }
        self.assertEqual(select_task_model_config(model, "type", "m")["served_model"], "type")
        self.assertEqual(
            select_task_model_config(model, "pressure", "m")["served_model"],
            "size-thick-pressure",
        )
        with self.assertRaises(ValueError):
            select_task_model_config(model, "material", "m")

        self.assertTrue(model_supports_task(model, "pressure"))
        self.assertFalse(model_supports_task(model, "material"))

    def test_parse_json_from_fence(self):
        value = parse_json_output('```json\n{"TYPE":{"BODY":"弯头"}}\n```')
        self.assertEqual(value["TYPE"]["BODY"], "弯头")

    def test_clean_code(self):
        self.assertEqual(clean_code_output("```text\nBF90\n```"), "BF90")

    def test_named_overrides(self):
        self.assertEqual(parse_named_overrides(["m=/tmp/a"], "--adapter"), {"m": "/tmp/a"})
        with self.assertRaises(ValueError):
            parse_named_overrides(["broken"], "--adapter")

    def test_code_normalization(self):
        config = {"ignore_case": True, "remove_whitespace": True}
        self.assertEqual(normalize_code(" ab 12 ", config), "AB12")

    def test_type_coder_value_uses_training_shape(self):
        value = TaskEncoder._type_coder_value(
            {"BODY": "弯头", "GEOMETRY": {"ANGLE": "90", "RADIUS": "LR"}}
        )
        self.assertEqual(value, '{"BODY":"弯头","ANGLE":"90","RADIUS":"LR"}')

    def test_related_tasks_share_adapter(self):
        model = {"adapters": {"size": "/size", "material": "/material"}}
        self.assertEqual(resolve_adapter(model, "pressure"), "/size")
        self.assertEqual(resolve_adapter(model, "standard"), "/material")

    def test_full_stage_outputs_merge_to_pipe_encoder_entities(self):
        records = {
            "type": {
                "raw": "type raw",
                "parsed": {"CATEGORY": "管件", "TYPE": {"BODY": "弯头"}},
                "stage1_seconds": 0.1,
            },
            "material": {
                "raw": "material raw",
                "parsed": {
                    "MATERIAL": [{"VALUE": "20#", "SPECIAL_REQ": []}],
                    "STANDARD": [{"BODY": "GB/T 8163"}],
                },
                "stage1_seconds": 0.2,
            },
            "size": {
                "raw": "size raw",
                "parsed": {
                    "ITEMS": [
                        {
                            "SCOPE": "BODY",
                            "ROLE": "SINGLE",
                            "SIZE": [{"type": "DN", "value": "100"}],
                            "THICKNESS": [{"type": "SCHEDULE", "value": "40"}],
                        }
                    ],
                    "LENGTH": "",
                    "PRESSURE": "PN16",
                },
                "stage1_seconds": 0.3,
            },
        }
        merged = merge_full_stage_records(records)
        self.assertEqual(merged["material_category"], "管件")
        self.assertEqual(merged["parsed"]["TYPE"]["BODY"], "弯头")
        self.assertEqual(merged["parsed"]["MATERIAL"][0]["VALUE"], "20#")
        self.assertEqual(merged["parsed"]["STRUCTURAL"]["_schema_version"], "v2")
        self.assertNotIn("SIZE", merged["parsed"])
        self.assertAlmostEqual(merged["stage1_seconds"], 0.6)

    def test_full_merge_keeps_running_when_one_stage_is_invalid(self):
        records = {
            "type": {"raw": "bad", "parsed": None, "stage1_seconds": 0.1},
            "material": {"raw": "{}", "parsed": {}, "stage1_seconds": 0.1},
            "size": {"raw": "{}", "parsed": {}, "stage1_seconds": 0.1},
        }
        merged = merge_full_stage_records(records)
        self.assertIn("type 一阶段 JSON 解析失败", merged["errors"])

    def test_full_runtime_allows_a_different_base_for_each_stage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "full.yaml"
            config_path.write_text(
                """
groups:
  group1:
    stages:
      type: {base_model_path: /models/type, adapter_path: /lora/type}
      material: {base_model_path: /models/material, adapter_path: /lora/material}
      size: {base_model_path: /models/size, adapter_path: /lora/size}
tasks:
  type: {prompt_path: /prompts/type, field: TYPE}
  material: {prompt_path: /prompts/material, field: MATERIAL}
  size: {prompt_path: /prompts/size, field: SIZE}
  full: {stages: [type, material, size], field: FULL}
""".strip(),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                config=config_path,
                task="full",
                input=root / "input.xlsx",
                output=None,
                output_dir=root / "output",
                prompt=None,
                models=None,
                groups=["group1"],
                base_model=[],
                adapter=[],
            )
            runtime = prepare_runtime(args)
            self.assertEqual(runtime["models"][0]["name"], "group1")
            stages = runtime["models"][0]["stages"]
            self.assertEqual(stages["type"]["base_model_path"], Path("/models/type"))
            self.assertEqual(stages["material"]["base_model_path"], Path("/models/material"))
            self.assertEqual(stages["size"]["base_model_path"], Path("/models/size"))

    def test_vllm_stage1_preserves_input_order_under_concurrency(self):
        model = {
            "name": "type-model",
            "backend": "managed_vllm",
            "served_model": "type",
            "service_config": Path("/tmp/service.yaml"),
            "max_new_tokens": 128,
        }
        service = SimpleNamespace(service_url="http://127.0.0.1:8200", process=None)

        def fake_predict(**kwargs):
            value = kwargs["text"]
            return {"raw": value, "parsed": {"value": value}, "stage1_seconds": 0.01}

        settings = {
            "max_new_tokens": 256,
            "_managed_vllm": {"concurrency": 4, "progress_every": 100},
        }
        descriptions = [f"row-{index}" for index in range(20)]
        with patch(
            "apps.model_excel_evaluator.evaluate._post_vllm_predict",
            side_effect=fake_predict,
        ):
            records = run_vllm_stage1(model, descriptions, settings, service)
        self.assertEqual([record["raw"] for record in records], descriptions)
        self.assertEqual(
            stage_model_signature(model),
            ("managed_vllm", "/tmp/service.yaml", "type"),
        )

    def test_managed_service_starts_and_stops_launcher(self):
        class FakeProcess:
            returncode = None

            def __init__(self):
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        deployment = SimpleNamespace(
            gateway=SimpleNamespace(port=8200, startup_timeout_seconds=10),
        )
        process = FakeProcess()
        with patch("apps.vllm_service.config.load_config", return_value=deployment), patch(
            "apps.model_excel_evaluator.evaluate.subprocess.Popen",
            return_value=process,
        ):
            service = ManagedVLLMService(
                Path("/tmp/service.yaml"),
                {"startup_timeout": 5, "health_interval": 0.5, "shutdown_cooldown": 0},
            )
            with patch.object(service, "_health", side_effect=[None, {"ok": True}]):
                service.start()
            service.stop()
        self.assertTrue(process.terminated)

    def test_stage_jobs_share_one_service_for_routes_with_the_same_config(self):
        service_a = Path("/tmp/service-a.yaml")
        service_b = Path("/tmp/service-b.yaml")
        models = [
            {
                "name": "type",
                "backend": "managed_vllm",
                "service_config": service_a,
                "served_model": "type",
            },
            {
                "name": "size",
                "backend": "managed_vllm",
                "service_config": service_a,
                "served_model": "size",
            },
            {
                "name": "material",
                "backend": "managed_vllm",
                "service_config": service_b,
                "served_model": "material",
            },
        ]
        jobs = {
            stage_model_signature(model): (model, "prompt")
            for model in models
        }
        lifecycle = []

        class FakeService:
            def __init__(self, service_config, settings):
                self.service_config = service_config
                self.load_seconds = 1.0

            def __enter__(self):
                lifecycle.append(("start", self.service_config))
                return self

            def __exit__(self, exc_type, exc, traceback):
                lifecycle.append(("stop", self.service_config))

        def fake_run_stage1(model, descriptions, instruction, settings, managed_service=None):
            self.assertIsNotNone(managed_service)
            return [
                {"raw": model["served_model"], "parsed": {}, "stage1_seconds": 0.01}
                for _ in descriptions
            ], 0.0

        with patch(
            "apps.model_excel_evaluator.evaluate.ManagedVLLMService",
            FakeService,
        ), patch(
            "apps.model_excel_evaluator.evaluate.run_stage1",
            side_effect=fake_run_stage1,
        ):
            records, loads = run_stage1_jobs(jobs, ["row"], {"_managed_vllm": {}})

        self.assertEqual(
            lifecycle,
            [
                ("start", service_a),
                ("stop", service_a),
                ("start", service_b),
                ("stop", service_b),
            ],
        )
        self.assertEqual(records[stage_model_signature(models[0])][0]["raw"], "type")
        self.assertEqual(records[stage_model_signature(models[1])][0]["raw"], "size")
        self.assertEqual(loads[stage_model_signature(models[0])], 1.0)
        self.assertEqual(loads[stage_model_signature(models[1])], 0.0)

    def test_analysis_is_written_as_three_images(self):
        summary = pd.DataFrame([
            {"模型": "m1", "准确率": 0.8, "平均耗时秒": 0.1, "P95耗时秒": 0.2},
            {"模型": "m2", "准确率": 0.9, "平均耗时秒": 0.2, "P95耗时秒": 0.3},
        ])
        pairwise = pd.DataFrame([
            {"模型A": "m1", "模型B": "m2", "预测不一致率": 0.25},
        ])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "accuracy": root / "accuracy.png",
                "latency": root / "latency.png",
                "disagreement": root / "disagreement.png",
            }
            draw_analysis(summary, pairwise, paths)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
