from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.vllm_service.config import build_engine_command, load_config


CONFIG = """
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


class ConfigTest(unittest.TestCase):
    def _load(self, content: str = CONFIG):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.yaml"
            path.write_text(content, encoding="utf-8")
            return load_config(path)

    def test_loads_multi_lora_routes(self):
        config = self._load()
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


if __name__ == "__main__":
    unittest.main()
