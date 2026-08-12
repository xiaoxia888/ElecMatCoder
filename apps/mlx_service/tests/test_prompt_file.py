from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from apps.mlx_service.server import load_registry


class PromptFileTest(unittest.TestCase):
    def _write_config(self, directory: str, prompt_file: str) -> Path:
        path = Path(directory) / "service.yaml"
        path.write_text(
            "models:\n"
            "  size:\n"
            "    model_path: /models/size\n"
            f"    prompt_file: {prompt_file}\n",
            encoding="utf-8",
        )
        return path

    def test_loads_relative_prompt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_dir = Path(directory) / "prompts"
            prompt_dir.mkdir()
            prompt = prompt_dir / "size.txt"
            prompt.write_text("size prompt\n", encoding="utf-8")
            config = self._write_config(directory, "prompts/size.txt")

            spec = load_registry(config)["size"]

            self.assertEqual(spec.instruction, "size prompt")
            self.assertEqual(spec.prompt_file, str(prompt.resolve()))

    def test_requires_prompt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._write_config(directory, '\"\"')
            with self.assertRaisesRegex(ValueError, "缺少 prompt_file"):
                load_registry(config)

    def test_rejects_missing_prompt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._write_config(directory, "missing.txt")
            with self.assertRaisesRegex(FileNotFoundError, "提示词文件不存在"):
                load_registry(config)


if __name__ == "__main__":
    unittest.main()
