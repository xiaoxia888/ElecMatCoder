from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apps.vllm_service.benchmark import read_texts
from apps.vllm_service.gateway import parse_json_output


class HelperTest(unittest.TestCase):
    def test_parse_json_output_removes_thinking(self):
        parsed = parse_json_output('<think>ignored</think>\n```json\n{"ok": true}\n```')
        self.assertEqual(parsed, {"ok": True})

    def test_read_json_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            path.write_text(
                json.dumps([{"材料描述": "PIPE DN50"}, {"材料描述": "TEE DN50"}]),
                encoding="utf-8",
            )
            self.assertEqual(
                read_texts(path, "材料描述", [], 0),
                ["PIPE DN50", "TEE DN50"],
            )


if __name__ == "__main__":
    unittest.main()

