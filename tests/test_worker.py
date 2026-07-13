from __future__ import annotations

import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from slippi_ai_review import worker


class AdvantageReviewWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.uploads = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def create_review(self, *, status: str = "queued", target: bool = True, created: str = "2026-01-01T00:00:00Z") -> dict:
        review_id = str(uuid.uuid4())
        review_dir = self.uploads / review_id
        review_dir.mkdir()
        (review_dir / "replay.slp").write_bytes(b"replay")
        player = {"playerIndex": 0, "port": 1, "character": {"name": "Fox"}}
        payload = {
            "schemaVersion": 2,
            "reviewId": review_id,
            "status": status,
            "message": "test",
            "createdAt": created,
            "updatedAt": created,
            "players": [player, {"playerIndex": 1, "port": 2, "character": {"name": "Sheik"}}],
            "targetPlayer": player if target else None,
            "urls": {
                "review": f"/reviews/{review_id}/",
                "status": f"/api/reviews/{review_id}",
                "artifacts": f"/review-artifacts/{review_id}/",
                "report": f"/reviews/{review_id}/report",
                "artifactReport": None,
            },
        }
        (review_dir / "review.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def args(self) -> SimpleNamespace:
        return SimpleNamespace(
            upload_dir=self.uploads,
            preflight_samples=1,
            refinement_samples=1,
            max_batch_lanes=1,
            render_workers=1,
            force=False,
        )

    def test_next_job_skips_unselected_review(self) -> None:
        self.create_review(status="awaiting_target", target=False, created="2025-01-01T00:00:00Z")
        queued = self.create_review(created="2026-01-01T00:00:00Z")
        self.assertEqual(worker.next_job(self.uploads)["reviewId"], queued["reviewId"])

    def test_pipeline_command_uses_review_quality_preset(self) -> None:
        review = self.create_review()
        review["settings"] = {"qualityPreset": "quick"}
        args = self.args()
        args.preflight_samples = None
        args.refinement_samples = None
        command = worker.pipeline_command(args, review)
        self.assertEqual(command[command.index("--preflight-samples") + 1], "16")
        self.assertEqual(command[command.index("--refinement-samples") + 1], "32")

    def test_process_job_publishes_complete_report(self) -> None:
        review = self.create_review()
        job_dir = self.uploads / review["reviewId"]
        script = (
            "import json; from pathlib import Path; "
            f"p=Path({str(job_dir)!r})/'artifacts'; p.mkdir(); "
            "(p/'advantage_review.html').write_text('report'); "
            "print(json.dumps({'event':'progress','stage':'artifacts','message':'Built artifacts.'}))"
        )
        with patch.object(worker, "pipeline_command", return_value=[sys.executable, "-c", script]):
            self.assertTrue(worker.process_job(self.args(), review))
        result = json.loads((job_dir / "review.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["progress"]["stage"], "complete")
        self.assertEqual(result["urls"]["artifactReport"], f"/review-artifacts/{review['reviewId']}/advantage_review.html")

    def test_process_job_records_failure(self) -> None:
        review = self.create_review()
        with patch.object(worker, "pipeline_command", return_value=[sys.executable, "-c", "raise SystemExit(3)"]):
            self.assertFalse(worker.process_job(self.args(), review))
        result = json.loads((self.uploads / review["reviewId"] / "review.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "failed")
        self.assertTrue((self.uploads / review["reviewId"] / "logs" / "worker_error.log").is_file())


if __name__ == "__main__":
    unittest.main()
