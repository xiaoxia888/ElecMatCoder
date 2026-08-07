import json
import math
import unittest

from src.confidence.model_token_confidence import build_field_token_confidences
from src.encoder.pipe_encoder import EncodedFieldResult, PipeEncoderBase, PipeEncodingResult
from src.llm_ner.stage1_orchestrator import Stage1FieldOrchestrator


class ModelTokenConfidenceTest(unittest.TestCase):
    def test_field_score_uses_values_not_json_keys(self):
        structured = {
            "TYPE": {"BODY": "盲板法兰", "CONN": [], "SEAL": ["RF"]},
            "MATERIAL": [{"PART": "BODY", "VALUE": "ASTM A105"}],
        }
        raw = json.dumps(structured, ensure_ascii=False)
        value_characters = set("盲板法兰RF")
        records = []
        for index, character in enumerate(raw):
            probability = 0.8 if character in value_characters else 0.99
            records.append(
                {
                    "token": character,
                    "logprob": math.log(probability),
                    "start": index,
                    "end": index + 1,
                }
            )

        scores = build_field_token_confidences(raw, structured, records)

        self.assertAlmostEqual(scores["TYPE"]["confidence"], 0.8, places=6)
        self.assertEqual(scores["TYPE"]["token_count"], 6)

    def test_deterministic_stage2_does_not_raise_stage1_score(self):
        encoder = PipeEncoderBase.__new__(PipeEncoderBase)
        field = EncodedFieldResult(similarity=1.0)

        encoder._set_field_confidence_triplet(field, 0.7, 1.0)

        self.assertEqual(field.stage1_confidence, 0.7)
        self.assertEqual(field.stage2_confidence, 1.0)
        self.assertAlmostEqual(field.field_confidence, 0.7, places=8)

    def test_two_model_stages_use_serial_probability(self):
        encoder = PipeEncoderBase.__new__(PipeEncoderBase)
        field = EncodedFieldResult(similarity=1.0)

        encoder._set_field_confidence_triplet(field, 0.8, 0.75)

        self.assertAlmostEqual(field.field_confidence, 0.6, places=8)

    def test_rule_structural_extraction_has_full_stage_confidence(self):
        score = Stage1FieldOrchestrator._build_structural_field_confidence_v2(
            text="DN50",
            field="SIZE",
            field_value={"DN": "50"},
            source="rule_extraction",
        )

        self.assertEqual(score["confidence"], 1.0)
        self.assertEqual(score["reason"], "deterministic_rule_extraction")

    def test_model_without_logprobs_is_unavailable_not_full_confidence(self):
        score = Stage1FieldOrchestrator._build_structural_field_confidence_v2(
            text="DN50",
            field="SIZE",
            field_value={"DN": "50"},
            source="finetuned_structural_model",
        )

        self.assertIsNone(score["confidence"])

    def test_result_uses_weakest_available_field(self):
        encoder = PipeEncoderBase.__new__(PipeEncoderBase)
        result = PipeEncodingResult(
            original_text="x",
            fields={
                "TYPE": EncodedFieldResult(field_confidence=0.82),
                "SIZE": EncodedFieldResult(field_confidence=1.0),
                "STANDARD": EncodedFieldResult(field_confidence=None),
            },
        )

        self.assertEqual(encoder._compute_result_confidence(result), 0.82)


if __name__ == "__main__":
    unittest.main()
