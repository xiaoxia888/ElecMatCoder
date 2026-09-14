import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from apps.trainer.unsloth_trainer.config import _detect_project_root, load_config
from apps.trainer.unsloth_trainer.data import build_messages, format_batch, validate_data_file
from apps.trainer.unsloth_trainer.train import (
    _assert_no_meta_parameters,
    _moving_average,
    _prepare_run_dir,
    _resolve_warmup_steps,
    _save_training_curves,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is False
        return "|".join(f"{item['role']}:{item['content']}" for item in messages)


def _write_config(tmp_path: Path, train_file: Path) -> Path:
    config = {
        "model": {"name_or_path": "Qwen/Qwen3.5-4B"},
        "data": {"train_file": str(train_file)},
        "training": {"load_best_model_at_end": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


class ConfigAndDataTest(unittest.TestCase):
    def test_meta_parameter_preflight_rejects_unloaded_weights(self) -> None:
        class Parameter:
            def __init__(self, is_meta: bool):
                self.is_meta = is_meta

        class Model:
            def named_parameters(self):
                return [("loaded", Parameter(False)), ("lm_head.weight", Parameter(True))]

        with self.assertRaisesRegex(RuntimeError, "lm_head.weight"):
            _assert_no_meta_parameters(Model())

    def test_training_curve_export_creates_separate_artifacts(self) -> None:
        history = [
            {"step": 20, "loss": 0.683, "learning_rate": 2e-6},
            {"step": 40, "loss": 0.08, "learning_rate": 4e-6},
            {"step": 60, "loss": 0.009, "learning_rate": 6e-6},
            {"step": 1000, "loss": 0.0015, "learning_rate": 2e-5},
            {"step": 1000, "eval_loss": 0.00197},
            {"step": 2000, "loss": 0.0009, "learning_rate": 1.5e-5},
            {"step": 2000, "eval_loss": 0.00112},
            {"step": 3000, "loss": 0.0006, "learning_rate": 8e-6},
            {"step": 3000, "eval_loss": 0.00094},
        ]

        with TemporaryDirectory() as directory:
            artifacts = _save_training_curves(history, Path(directory))

            self.assertEqual(
                set(artifacts), {"train_loss", "eval_loss", "overview", "learning_rate"}
            )
            for artifact_path in artifacts.values():
                path = Path(artifact_path)
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_moving_average_keeps_steps_and_smooths_values(self) -> None:
        points = [(20, 0.6), (40, 0.3), (60, 0.0)]

        smoothed = _moving_average(points, window=2)

        self.assertEqual([step for step, _ in smoothed], [20, 40, 60])
        for actual, expected in zip(
            [value for _, value in smoothed], [0.6, 0.45, 0.15]
        ):
            self.assertAlmostEqual(actual, expected)

    def test_auto_resume_creates_resumes_and_rejects_completed_run(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = {
                "training": {
                    "output_dir": str(tmp_path / "outputs"),
                    "append_timestamp": True,
                    "resume_from_checkpoint": "auto",
                },
                "save": {"adapter_subdir": "final_adapter"},
            }

            run_dir, checkpoint = _prepare_run_dir(config, tmp_path / "config.yaml")
            self.assertIsNone(checkpoint)
            self.assertTrue(run_dir.name.startswith("run_"))

            for step in (200, 400):
                checkpoint_dir = run_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir()
                (checkpoint_dir / "trainer_state.json").write_text("{}", encoding="utf-8")

            resumed_dir, checkpoint = _prepare_run_dir(config, tmp_path / "config.yaml")
            self.assertEqual(resumed_dir, run_dir)
            self.assertEqual(checkpoint, str(run_dir / "checkpoint-400"))

            adapter_dir = run_dir / "final_adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "训练已经完成"):
                _prepare_run_dir(config, tmp_path / "config.yaml")

    def test_project_root_supports_repository_and_standalone_layouts(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            standalone = tmp_path / "unsloth_trainer"
            standalone.mkdir()
            self.assertEqual(_detect_project_root(standalone), tmp_path)

            repository_package = tmp_path / "repo" / "apps" / "trainer" / "unsloth_trainer"
            repository_package.mkdir(parents=True)
            self.assertEqual(_detect_project_root(repository_package), tmp_path / "repo")

    def test_load_config_applies_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train_file = tmp_path / "train.json"
            train_file.write_text("[]", encoding="utf-8")
            config, _ = load_config(_write_config(tmp_path, train_file))

        self.assertTrue(config["model"]["load_in_16bit"])
        self.assertFalse(config["model"]["offload_embedding"])
        self.assertEqual(config["lora"]["use_gradient_checkpointing"], "unsloth")
        self.assertEqual(config["training"]["gradient_accumulation_steps"], 16)
        self.assertEqual(config["training"]["resume_from_checkpoint"], "auto")

    def test_single_finetuning_method_maps_low_level_switches(self) -> None:
        expected = {
            "lora": {
                "load_in_4bit": False,
                "load_in_16bit": True,
                "full_finetuning": False,
                "lora_enabled": True,
            },
            "qlora": {
                "load_in_4bit": True,
                "load_in_16bit": False,
                "full_finetuning": False,
                "lora_enabled": True,
            },
            "full": {
                "load_in_4bit": False,
                "load_in_16bit": True,
                "full_finetuning": True,
                "lora_enabled": False,
            },
        }

        for method, switches in expected.items():
            with self.subTest(method=method), TemporaryDirectory() as directory:
                tmp_path = Path(directory)
                train_file = tmp_path / "train.json"
                train_file.write_text("[]", encoding="utf-8")
                config_path = _write_config(tmp_path, train_file)
                task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                task_config["finetuning_method"] = method
                config_path.write_text(
                    yaml.safe_dump(task_config, allow_unicode=True), encoding="utf-8"
                )

                config, _ = load_config(config_path)

                self.assertEqual(config["finetuning_method"], method)
                self.assertEqual(config["model"]["load_in_4bit"], switches["load_in_4bit"])
                self.assertEqual(config["model"]["load_in_16bit"], switches["load_in_16bit"])
                self.assertEqual(
                    config["model"]["full_finetuning"], switches["full_finetuning"]
                )
                self.assertEqual(config["lora"]["enabled"], switches["lora_enabled"])
                self.assertTrue(config["training"]["bf16"])
                self.assertFalse(config["training"]["fp16"])

    def test_rejects_unknown_finetuning_method(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train_file = tmp_path / "train.json"
            train_file.write_text("[]", encoding="utf-8")
            config_path = _write_config(tmp_path, train_file)
            task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            task_config["finetuning_method"] = "lore"
            config_path.write_text(
                yaml.safe_dump(task_config, allow_unicode=True), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "lora、qlora 或 full"):
                load_config(config_path)

    def test_alpaca_sample_is_converted_to_chat_messages(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train_file = tmp_path / "train.json"
            train_file.write_text("[]", encoding="utf-8")
            config, _ = load_config(_write_config(tmp_path, train_file))
        sample = {"instruction": "抽取字段", "input": "DN50", "output": {"SIZE": "DN50"}}

        messages = build_messages(sample, config["data"])
        result = format_batch(
            {key: [value] for key, value in sample.items()}, FakeTokenizer(), config["data"]
        )

        self.assertEqual(messages[0], {"role": "user", "content": "抽取字段\nDN50"})
        self.assertEqual(messages[1]["content"], '{"SIZE":"DN50"}')
        self.assertEqual(result, {"text": ['user:抽取字段\nDN50|assistant:{"SIZE":"DN50"}']})

    def test_task_config_overrides_shared_config(self) -> None:
        task_path = (
            Path(__file__).parents[1]
            / "train_config"
            / "种类"
            / "qwen35_9b_rtx5090.yaml"
        )
        config, loaded_path = load_config(task_path)

        self.assertEqual(loaded_path, task_path.resolve())
        self.assertEqual(config["model"]["name_or_path"], "/home/waas/base-models/Qwen3.5-9B")
        self.assertEqual(config["model"]["max_seq_length"], 1024)
        self.assertEqual(config["training"]["learning_rate"], 0.00002)
        self.assertEqual(config["training"]["gradient_accumulation_steps"], 16)
        self.assertEqual(config["lora"]["target_modules"][0], "q_proj")
        self.assertTrue(config["runtime"]["compile_disable"])

    def test_four_task_configs_contain_their_own_training_parameters(self) -> None:
        package_dir = Path(__file__).parents[1]
        expected = {
            "种类": {"epochs": 2, "max_seq_length": 1024},
            "尺寸": {"epochs": 2, "max_seq_length": 1024},
            "材质规范": {"epochs": 1, "max_seq_length": 1024},
            "编码": {"epochs": 3, "max_seq_length": 256},
        }

        for task, values in expected.items():
            with self.subTest(task=task):
                config_path = package_dir / "train_config" / task / "train.yaml"
                source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                config, loaded_path = load_config(config_path)

                self.assertEqual(loaded_path, config_path.resolve())
                self.assertEqual(
                    config["finetuning_method"], source["finetuning_method"]
                )
                self.assertEqual(config["model"]["name_or_path"], source["base_model"])
                self.assertEqual(
                    config["model"]["max_seq_length"], values["max_seq_length"]
                )
                self.assertEqual(
                    config["training"]["num_train_epochs"], values["epochs"]
                )
                self.assertEqual(config["training"]["warmup_fraction"], 0.03)
                self.assertEqual(config["training"]["warmup_steps"], "auto")
                self.assertFalse(config["model"]["trust_remote_code"])
                self.assertEqual(config["data"]["num_proc"], 4)
                self.assertEqual(config["training"]["output_dir"], source["output_dir"])
                self.assertEqual(config["data"]["train_file"], source["train_dataset"])
                self.assertEqual(
                    config["data"]["validation_file"], source["validation_dataset"]
                )
                self.assertEqual(
                    config["training"]["num_train_epochs"],
                    source["training"]["num_train_epochs"],
                )

                if task == "材质规范":
                    self.assertEqual(config["training"]["eval_strategy"], "steps")
                    self.assertEqual(config["training"]["eval_steps"], 1000)
                    self.assertEqual(config["training"]["save_strategy"], "steps")
                    self.assertEqual(config["training"]["save_steps"], 1000)
                    self.assertEqual(config["training"]["save_total_limit"], 3)
                    self.assertTrue(config["training"]["load_best_model_at_end"])

    def test_auto_warmup_uses_effective_batch_and_never_passes_a_ratio(self) -> None:
        config = {
            "training": {
                "max_steps": -1,
                "num_train_epochs": 1.0,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "warmup_fraction": 0.03,
                "warmup_steps": "auto",
            }
        }

        self.assertEqual(_resolve_warmup_steps(config, 69719), 131)

    def test_validate_jsonl_chatml_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            sample = {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "y"},
                ]
            }
            path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
            data_config = {"format": "chatml", "columns": {"messages": "messages"}}
            self.assertEqual(validate_data_file(path, data_config), sample)

    def test_rejects_conflicting_precision(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train_file = tmp_path / "train.json"
            train_file.write_text("[]", encoding="utf-8")
            config_path = _write_config(tmp_path, train_file)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["training"].update({"bf16": True, "fp16": True})
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "不能同时启用"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
