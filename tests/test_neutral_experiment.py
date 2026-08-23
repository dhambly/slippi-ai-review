from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slippi_ai_review.neutral_experiment import boundary, build_points, main, summarize_test


class NeutralExperimentTests(unittest.TestCase):
    def test_build_points_uses_ten_frame_lookbacks(self) -> None:
        events = [{"eventId": "avoid_1", "kind": "avoid", "openingFrame": 100, "openingMove": "nair"}]
        tests, points = build_points(events, first_frame=-123, step_frames=10, max_lookback_frames=30)
        self.assertEqual([item["lookbackFrames"] for item in tests], [10, 20, 30])
        self.assertEqual([item["frame"] for item in points], [70, 80, 90])

    def test_summary_uses_fixed_opening_deadline(self) -> None:
        test = {
            "eventId": "avoid_1",
            "kind": "avoid",
            "openingFrame": 100,
            "openingMove": "nair",
            "lookbackFrames": 20,
            "injectionFrame": 80,
            "postOpeningFrames": 10,
        }
        rows = [
            {"neutralContact": {"firstDamageDealtFrame": 95, "firstDamageTakenFrame": None}},
            {"neutralContact": {"firstDamageDealtFrame": None, "firstDamageTakenFrame": 105}},
            {"neutralContact": {"firstDamageDealtFrame": None, "firstDamageTakenFrame": 111}},
        ]
        result = summarize_test(test, rows)
        self.assertAlmostEqual(result["avoidHitRate"], 2 / 3, places=3)
        self.assertAlmostEqual(result["landHitRate"], 1 / 3, places=3)
        self.assertAlmostEqual(result["winFirstHitRate"], 1 / 3, places=3)

    def test_boundary_reports_first_and_sustained_crossing(self) -> None:
        points = [
            {"lookbackFrames": 10, "avoidHitRate": 0.5},
            {"lookbackFrames": 20, "avoidHitRate": 0.8},
            {"lookbackFrames": 30, "avoidHitRate": 0.7},
            {"lookbackFrames": 40, "avoidHitRate": 0.9},
        ]
        result = boundary(points, "avoidHitRate", 0.75)
        self.assertEqual(result["firstPassingLookbackFrames"], 20)
        self.assertEqual(result["furthestPassingLookbackFrames"], 40)
        self.assertIsNone(result["contiguousPassingLookbackFrames"])
        self.assertEqual(result["sustainedPassingLookbackFrames"], 40)

    def test_main_writes_empty_experiment_without_starting_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = root / "game.slp"
            replay.write_bytes(b"slp")
            out = root / "out"
            raw = {
                "firstFrame": -123,
                "settings": {"players": [{"port": 1, "playerIndex": 0}]},
                "conversions": [],
            }
            argv = [
                "neutral_experiment",
                "--replay", str(replay),
                "--analyzed-port", "1",
                "--kind", "avoid",
                "--max-events-per-kind", "0",
                "--out", str(out),
            ]
            with patch("slippi_ai_review.neutral_experiment.extract_raw_conversions", return_value=raw), patch.object(
                sys,
                "argv",
                argv,
            ):
                self.assertEqual(main(), 0)
            payload = json.loads((out / "neutral_experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["events"], [])
            self.assertFalse((out / "simulation").exists())


if __name__ == "__main__":
    unittest.main()
