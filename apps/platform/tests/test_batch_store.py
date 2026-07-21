import tempfile
import unittest
from pathlib import Path

from apps.platform.batch_store import BatchJobStore


class BatchJobStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = BatchJobStore(Path(self.temp_dir.name) / "batch_jobs.db")
        self.store.create_job(
            {
                "job_id": "job-1",
                "status": "running",
                "total": 2,
                "processed": 0,
                "success_count": 0,
                "review_count": 0,
                "threshold": 0.8,
                "max_concurrent": 2,
                "error": "",
                "items_meta": [],
                "created_at": 10.0,
                "started_at": 11.0,
                "finished_at": None,
                "duration_seconds": 0.0,
                "updated_at": 11.0,
            }
        )

    def tearDown(self) -> None:
        self.store._conn.close()
        self.temp_dir.cleanup()

    def test_save_result_and_progress_are_committed_together(self) -> None:
        self.store.save_result_and_advance_job(
            "job-1",
            0,
            10,
            {"success": True, "need_review": False, "final_code": "P100"},
            success_delta=1,
            review_delta=0,
            updated_at=12.0,
            duration_seconds=1.0,
        )
        self.store.save_result_and_advance_job(
            "job-1",
            1,
            11,
            {"success": False, "need_review": True, "final_code": ""},
            success_delta=0,
            review_delta=1,
            updated_at=13.0,
            duration_seconds=2.0,
        )

        job = self.store.get_job("job-1")
        self.assertIsNotNone(job)
        self.assertEqual(job["processed"], 2)
        self.assertEqual(job["success_count"], 1)
        self.assertEqual(job["review_count"], 1)
        self.assertEqual(job["duration_seconds"], 2.0)

        results = self.store.get_results("job-1")
        self.assertEqual(results["0"]["final_code"], "P100")
        self.assertTrue(results["1"]["need_review"])

    def test_finalize_only_overwrites_changed_results(self) -> None:
        original_first = {"success": True, "need_review": False, "final_code": "P100"}
        original_second = {"success": False, "need_review": True, "final_code": ""}
        self.store.save_result_and_advance_job(
            "job-1",
            0,
            10,
            original_first,
            success_delta=1,
            review_delta=0,
            updated_at=12.0,
            duration_seconds=1.0,
        )
        self.store.save_result_and_advance_job(
            "job-1",
            1,
            11,
            original_second,
            success_delta=0,
            review_delta=1,
            updated_at=13.0,
            duration_seconds=2.0,
        )

        finalized_second = {"success": True, "need_review": False, "final_code": "P200"}
        self.store.finalize_job(
            "job-1",
            [(1, 11, finalized_second)],
            processed=2,
            success_count=2,
            review_count=0,
            finished_at=20.0,
            duration_seconds=9.0,
        )

        job = self.store.get_job("job-1")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "finished")
        self.assertEqual(job["processed"], 2)
        self.assertEqual(job["success_count"], 2)
        self.assertEqual(job["review_count"], 0)
        self.assertEqual(job["finished_at"], 20.0)
        self.assertEqual(job["duration_seconds"], 9.0)

        results = self.store.get_results("job-1")
        self.assertEqual(results["0"], original_first)
        self.assertEqual(results["1"], finalized_second)


if __name__ == "__main__":
    unittest.main()
