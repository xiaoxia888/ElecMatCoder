from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.vllm_service.config import DEFAULT_CONFIG_PATH, build_engine_command, load_config
from apps.vllm_service.launch import _engine_env, _format_command


CONFIG = """
profile: test
gateway:
  port: 8200
engines:
  base4b:
    port: 8302
    model_path: /models/base4b
    served_model_name: base4b
    cuda_visible_devices: "1"
    max_loras: 2
    max_cpu_loras: 3
    max_lora_rank: 16
    lora_modules:
      size: /models/lora/size
      material: /models/lora/material
models:
  size:
    engine: base4b
    upstream_model: size
    instruction: extract size
  material:
    engine: base4b
    upstream_model: material
    instruction: extract material
"""

PROFILE = """
engines:
  base4b:
    cuda_visible_devices: "1"
    dtype: float16
    max_model_len: 1024
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 1
    max_num_seqs: 16
"""


class ConfigTest(unittest.TestCase):
    def _load(
        self,
        content: str = CONFIG,
        profile_content: str = PROFILE,
        profile: str | None = None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.yaml"
            path.write_text(content, encoding="utf-8")
            profile_dir = path.parent / "profiles"
            profile_dir.mkdir()
            (profile_dir / "test.yaml").write_text(profile_content, encoding="utf-8")
            (profile_dir / "override.yaml").write_text(
                profile_content.replace('cuda_visible_devices: "1"', 'cuda_visible_devices: "0"'),
                encoding="utf-8",
            )
            return load_config(path, profile=profile)

    def test_loads_multi_lora_routes(self):
        config = self._load()
        self.assertEqual(config.profile_name, "test")
        self.assertEqual(set(config.models), {"size", "material"})
        self.assertEqual(config.models["size"].engine, "base4b")
        self.assertEqual(config.engines["base4b"].max_loras, 2)

    def test_builds_vllm_multi_lora_command(self):
        engine = self._load().engines["base4b"]
        command = build_engine_command(engine)
        self.assertIn("--enable-lora", command)
        self.assertIn("--max-loras", command)
        self.assertIn("size=/models/lora/size", command)
        self.assertIn("material=/models/lora/material", command)
        self.assertNotIn("--swap-space", command)

    def test_rejects_unknown_upstream_model(self):
        bad = CONFIG.replace("upstream_model: size", "upstream_model: missing")
        with self.assertRaisesRegex(ValueError, "未在 engine"):
            self._load(bad)

    def test_engine_environment_is_opt_in(self):
        config = self._load()
        engine = config.engines["base4b"]
        self.assertEqual(engine.environment, {})
        self.assertNotIn("VLLM_USE_FLASHINFER_SAMPLER", _format_command(engine))

        configured = self._load(
            profile_content=PROFILE.replace(
                '    max_num_seqs: 16',
                '    max_num_seqs: 16\n    environment:\n'
                '      VLLM_USE_FLASHINFER_SAMPLER: "0"',
            ),
        ).engines["base4b"]
        self.assertEqual(_engine_env(configured)["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=0", _format_command(configured))

    def test_cli_profile_overrides_service_profile(self):
        config = self._load(profile="override")
        self.assertEqual(config.profile_name, "override")
        self.assertEqual(config.engines["base4b"].cuda_visible_devices, "0")
        self.assertEqual(config.models["size"].instruction, "extract size")

    def test_requires_profile(self):
        with self.assertRaisesRegex(ValueError, "必须配置顶层profile"):
            self._load(CONFIG.replace("profile: test\n", ""))

    def test_profile_cannot_override_business_configuration(self):
        with self.assertRaisesRegex(ValueError, "不允许配置这些顶层字段"):
            self._load(profile_content=PROFILE + "models: {}\n")

        with self.assertRaisesRegex(ValueError, "不允许覆盖engine"):
            self._load(
                profile_content=PROFILE.replace(
                    '    cuda_visible_devices: "1"',
                    '    model_path: /wrong/model\n    cuda_visible_devices: "1"',
                )
            )

    def test_profile_cannot_reference_unknown_engine(self):
        with self.assertRaisesRegex(ValueError, "未知engine"):
            self._load(profile_content=PROFILE + "  missing:\n    dtype: float16\n")

    def test_profile_requires_every_engine_and_hardware_field(self):
        with self.assertRaisesRegex(ValueError, "缺少必填字段"):
            self._load(profile_content=PROFILE.replace("    max_num_seqs: 16\n", ""))

    def test_builtin_profiles_isolate_blackwell_workaround(self):
        config_5090 = load_config(DEFAULT_CONFIG_PATH, profile="dual-5090")
        config_4090 = load_config(DEFAULT_CONFIG_PATH, profile="dual-4090")
        for engine in config_5090.engines.values():
            self.assertEqual(engine.environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        for engine in config_4090.engines.values():
            self.assertNotIn("VLLM_USE_FLASHINFER_SAMPLER", engine.environment)

    def test_rejects_excessive_shared_gpu_memory_ratio(self):
        shared_profile = PROFILE.replace('cuda_visible_devices: "1"', 'cuda_visible_devices: "0"')
        shared_config = CONFIG.replace(
            "models:\n",
            "  second4b:\n"
            "    port: 8303\n"
            "    model_path: /models/second4b\n"
            "    served_model_name: second4b\n"
            "models:\n",
        )
        shared_profile += (
            "  second4b:\n"
            '    cuda_visible_devices: "0"\n'
            "    dtype: float16\n"
            "    max_model_len: 1024\n"
            "    gpu_memory_utilization: 0.2\n"
            "    tensor_parallel_size: 1\n"
            "    max_num_seqs: 8\n"
        )
        with self.assertRaisesRegex(ValueError, "总和不能超过1"):
            self._load(content=shared_config, profile_content=shared_profile)


if __name__ == "__main__":
    unittest.main()
