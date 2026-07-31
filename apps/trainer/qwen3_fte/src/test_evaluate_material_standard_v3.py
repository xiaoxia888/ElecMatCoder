from __future__ import annotations

import unittest

from apps.trainer.qwen3_fte.src.evaluate_llamafactory_model import (
    resolve_instruction,
    validate_material_standard_v3,
)


class EvaluateMaterialStandardV3Test(unittest.TestCase):
    def test_material_standard_task_uses_v3_prompt(self) -> None:
        instruction, path = resolve_instruction("material_standard")

        self.assertIsNotNone(path)
        self.assertEqual(path.name, "material_standard_extraction_sft_instruction_v3.txt")
        self.assertIn('"VALUE"', instruction)
        self.assertNotIn('"MATERIAL_RELATION"', instruction)

    def test_accepts_v3_output(self) -> None:
        output = {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "VALUE": "ASTM A312 TP316L",
                    "SPECIAL_REQ": ["NACE"],
                }
            ],
            "STANDARD": [{"BODY": "AB3619"}],
        }

        self.assertEqual(validate_material_standard_v3(output), [])

    def test_rejects_v2_output(self) -> None:
        output = {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "STANDARD": "ASTM A312",
                    "GRADE": "TP316L",
                    "CLASS": "",
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [{"BODY": "AB3619"}],
            "MATERIAL_RELATION": "SINGLE",
        }

        errors = validate_material_standard_v3(output)

        self.assertTrue(any("MATERIAL_RELATION" in error for error in errors))
        self.assertTrue(any("VALUE" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
