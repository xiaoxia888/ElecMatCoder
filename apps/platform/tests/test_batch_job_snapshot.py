import unittest

from apps.platform.server import _batch_job_public


class BatchJobSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.job = {
            "job_id": "job-1",
            "status": "running",
            "total": 2,
            "processed": 1,
            "success_count": 1,
            "review_count": 0,
            "items_meta": [
                {"index": 0, "text": "PIPE DN20"},
                {"index": 1, "text": "PIPE DN25"},
            ],
        }

    def test_summary_does_not_include_items(self) -> None:
        snapshot = _batch_job_public(self.job)

        self.assertNotIn("items", snapshot)
        self.assertNotIn("results", snapshot)
        self.assertEqual(snapshot["processed"], 1)

    def test_detail_can_include_items_and_results(self) -> None:
        results = {"0": {"final_code": "P20"}}
        snapshot = _batch_job_public(
            self.job,
            include_items=True,
            results=results,
        )

        self.assertEqual(snapshot["items"], self.job["items_meta"])
        self.assertIsNot(snapshot["items"], self.job["items_meta"])
        self.assertEqual(snapshot["results"], results)


if __name__ == "__main__":
    unittest.main()
