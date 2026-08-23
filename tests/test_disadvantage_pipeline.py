from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slippi_ai_review.disadvantage_pipeline import disadvantage_inventory, publish_disadvantage_artifacts


class DisadvantagePipelineTests(unittest.TestCase):
    def test_inventory_keeps_only_disadvantage_segments_and_frames(self) -> None:
        inventory = {
            "replay": "game.slp",
            "analyzedPort": 1,
            "counts": {"advantage": 1, "neutral": 1, "disadvantage": 1},
            "segments": [
                {"id": "a", "phase": "advantage", "injectionFrame": 10},
                {"id": "d", "phase": "disadvantage", "injectionFrame": 20},
            ],
            "frames": [
                {"frame": 10, "point_evidence": {"sweep_id": "a"}},
                {"frame": 20, "point_evidence": {"sweep_id": "d"}},
            ],
        }
        filtered = disadvantage_inventory(inventory)
        self.assertEqual([segment["id"] for segment in filtered["segments"]], ["d"])
        self.assertEqual([frame["frame"] for frame in filtered["frames"]], [20])
        self.assertEqual(filtered["counts"], {"advantage": 0, "neutral": 0, "disadvantage": 1})

    def test_publish_keeps_disadvantage_trace_family_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            (source / "traces").mkdir(parents=True)
            (source / "viewer").mkdir()
            (source / "disadvantage_review.html").write_text("defense", encoding="utf-8")
            (source / "disadvantage_defense.json").write_text("{}", encoding="utf-8")
            (source / "traces" / "disadvantage_001_agent.msltrace.json").write_text("{}", encoding="utf-8")
            (source / "traces" / "001_agent.msltrace.json").write_text("{}", encoding="utf-8")
            (source / "viewer" / "compare.html").write_text("viewer", encoding="utf-8")
            publish_disadvantage_artifacts(source, destination)
            self.assertTrue((destination / "disadvantage_review.html").is_file())
            self.assertTrue((destination / "traces" / "disadvantage_001_agent.msltrace.json").is_file())
            self.assertFalse((destination / "traces" / "001_agent.msltrace.json").exists())

    def test_publish_accepts_placeholder_without_viewer_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "disadvantage_review.html").write_text("empty", encoding="utf-8")
            publish_disadvantage_artifacts(source, destination)
            self.assertEqual((destination / "disadvantage_review.html").read_text(encoding="utf-8"), "empty")


if __name__ == "__main__":
    unittest.main()
