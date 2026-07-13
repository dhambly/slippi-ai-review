"""Regression tests for robust combo-extension render selection."""

from __future__ import annotations

import unittest

from slippi_ai_review.pipeline import (
    PREFLIGHT_MIN_OPTION_SAMPLES,
    REFINEMENT_MIN_OPTION_SAMPLES,
)
from slippi_ai_review.selection import lane_outperformed, meaningful_option


class PipelineSelectionThresholdTests(unittest.TestCase):
    def test_preflight_is_less_strict_than_refinement(self) -> None:
        self.assertEqual(PREFLIGHT_MIN_OPTION_SAMPLES, 2)
        self.assertEqual(REFINEMENT_MIN_OPTION_SAMPLES, 8)
        self.assertLess(PREFLIGHT_MIN_OPTION_SAMPLES, REFINEMENT_MIN_OPTION_SAMPLES)


BASELINE = {
    "original_followup_damage": 0.0,
    "original_followup_kill": False,
}


class RenderSelectionTest(unittest.TestCase):
    def test_authoritative_simulator_verdict_wins(self) -> None:
        row = {
            "outperformedReplay": False,
            "comboReversed": False,
            "followupDamage": 40.0,
            "defenderStocksLost": 1,
            "resolution": {"reason": "combo_timeout"},
        }
        self.assertFalse(lane_outperformed(row, BASELINE))

    def test_same_replay_stock_is_not_an_improvement(self) -> None:
        row = {
            "outperformedReplay": True,
            "comboReversed": False,
            "followupDamage": 40.0,
            "defenderStocksLost": 1,
            "resolution": {"reason": "combo_stock_lost"},
        }
        baseline = {
            "original_followup_damage": 10.0,
            "original_followup_kill": True,
        }
        self.assertFalse(lane_outperformed(row, baseline))

    def test_legacy_timeout_is_never_an_improvement(self) -> None:
        row = {
            "comboReversed": False,
            "followupDamage": 40.0,
            "defenderStocksLost": 0,
            "resolution": {"reason": "combo_timeout"},
        }
        self.assertFalse(lane_outperformed(row, BASELINE))

    def test_legacy_resolved_damage_gain_can_improve(self) -> None:
        row = {
            "comboReversed": False,
            "followupDamage": 10.0,
            "defenderStocksLost": 0,
            "resolution": {"reason": "combo_escape"},
        }
        self.assertTrue(lane_outperformed(row, BASELINE))

    def test_no_commitment_is_not_a_renderable_option(self) -> None:
        self.assertFalse(meaningful_option({"comboOptionSignature": "NO_COMMITMENT"}))
        self.assertFalse(meaningful_option({"optionSignature": "WAIT"}))
        self.assertTrue(meaningful_option({"comboOptionSignature": "BAIR"}))


if __name__ == "__main__":
    unittest.main()
