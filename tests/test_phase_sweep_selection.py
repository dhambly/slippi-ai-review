from __future__ import annotations

import unittest

from slippi_ai_review.phase_sweep_selection import build_queue


class PhaseSweepSelectionTests(unittest.TestCase):
    def test_selects_best_mean_option_per_phase_segment(self) -> None:
        inventory = {"replay": "game.slp", "analyzedPort": 1, "counts": {"neutral": 1}, "segments": [{"id": "neutral_1", "phase": "neutral", "label": "Neutral", "injectionFrame": 100}]}
        rows = [
            {"laneId": 0, "baseFrame": 100, "takeoverFrame": 100, "comboOptionSignature": "ROLL", "score": 1, "controllerStream": {"path": "x"}},
            {"laneId": 1, "baseFrame": 100, "takeoverFrame": 100, "comboOptionSignature": "ROLL", "score": 3, "controllerStream": {"path": "x"}},
            {"laneId": 2, "baseFrame": 100, "takeoverFrame": 100, "comboOptionSignature": "NAIR", "score": 5, "controllerStream": {"path": "x"}},
        ]
        queue = build_queue(inventory, rows)
        target = queue["targets"][0]
        self.assertEqual(target["option"]["optionSignature"], "NAIR")
        self.assertEqual(target["option"]["sweepSamples"], 3)
        self.assertAlmostEqual(target["option"]["optionShare"], 1 / 3, places=4)
        self.assertEqual(target["representative_lane"]["laneId"], 2)
        self.assertEqual(target["phase"], "neutral")


if __name__ == "__main__":
    unittest.main()
