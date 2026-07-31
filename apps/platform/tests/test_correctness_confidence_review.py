import unittest
from unittest.mock import patch

from apps.platform import server


class _Predictor:
    def __init__(self, score: float):
        self.score = score

    def predict_result(self, _result):
        return self.score, 1


class CorrectnessConfidenceReviewTest(unittest.TestCase):
    def _attach(self, score: float, need_review: bool = False):
        config = {
            "enabled": True,
            "mode": "enforce",
            "review_threshold": 0.90,
            "model_version": "test",
            "on_error": "review",
        }
        result = {
            "original_text": "PIPE DN50",
            "final_code": "P50",
            "success": True,
            "need_review": need_review,
            "fields": {},
        }
        with patch.object(server, "_get_encoding_confidence_config", return_value=config):
            with patch.object(
                server,
                "_get_encoding_correctness_predictor",
                return_value=_Predictor(score),
            ):
                return server._attach_correctness_confidence(result)

    def test_score_below_90_percent_enters_review(self):
        result = self._attach(0.8999)

        self.assertTrue(result["need_review"])
        self.assertTrue(result["correctness_confidence"]["review_triggered"])

    def test_score_at_90_percent_does_not_trigger_review(self):
        result = self._attach(0.90)

        self.assertFalse(result["need_review"])
        self.assertFalse(result["correctness_confidence"]["review_triggered"])

    def test_other_review_rules_are_not_cleared(self):
        result = self._attach(0.99, need_review=True)

        self.assertTrue(result["need_review"])
        self.assertFalse(result["correctness_confidence"]["review_triggered"])


if __name__ == "__main__":
    unittest.main()
