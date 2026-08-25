from __future__ import annotations

import unittest

from slippi_ai_review.phase_sweep import build_inventory, limit_segments, segment_presentation
from slippi_ai_review.phase_sweep_report import interactive_url


class PhaseSweepTests(unittest.TestCase):
    def test_interactive_url_separates_trace_switch_from_model_control(self) -> None:
        url = interactive_url(
            {
                "replay_trace": "../traces/replay.json",
                "agent_trace": "../traces/agent.json",
                "switch_frame": 60,
                "model_control_frame": 81,
                "frame_count": 100,
            },
            {"game": {"players": []}},
        )

        self.assertIn("switch=60", url)
        self.assertIn("takeover=81", url)

    def test_inventory_splits_advantage_by_owner_and_skips_transitions(self) -> None:
        timeline = {
            "firstFrame": -123,
            "settings": {"players": [{"port": 1, "playerIndex": 0}, {"port": 2, "playerIndex": 1}]},
            "segments": [
                {"phase": "pregame", "startFrame": -123, "endFrame": -1},
                {"phase": "neutral", "startFrame": 10, "endFrame": 80, "label": "Neutral"},
                {"phase": "advantage", "owner": 0, "startFrame": 81, "endFrame": 140, "openingFrame": 88},
                {"phase": "advantage", "owner": 1, "startFrame": 141, "endFrame": 200, "openingFrame": 150},
                {"phase": "stock_transition", "startFrame": 201, "endFrame": 260},
            ],
        }
        payload = build_inventory(timeline, analyzed_port=1)
        self.assertEqual(payload["counts"], {"advantage": 1, "neutral": 1, "disadvantage": 1})
        self.assertEqual([item["phase"] for item in payload["segments"]], ["neutral", "advantage", "disadvantage"])
        self.assertEqual(payload["segments"][0]["injectionFrame"], 22)
        self.assertEqual(payload["segments"][1]["injectionFrame"], 88)

    def test_presentation_uses_the_opening_move_not_the_raw_timeline_label(self) -> None:
        segment = {"phase": "advantage", "owner": 0, "startFrame": 100}
        opportunities = [{
            "owner": 0, "startFrame": 90, "endFrame": 180, "openingType": "counter-attack",
            "moves": 5, "damage": 55, "moveSequence": [{"moveShortName": "down-b"}],
        }]
        self.assertEqual(segment_presentation(segment, opportunities), ("Down B counter attack · 5 hits · +55%", "Counter Attack"))

    def test_nightly_limit_prioritizes_neutral_and_preserves_timeline_order(self) -> None:
        segments = [
            {
                "id": f"segment-{index}",
                "timelineIndex": index,
                "phase": ("neutral", "advantage", "disadvantage")[index % 3],
                "startFrame": index * 100,
                "endFrame": index * 100 + 60,
                "injectionFrame": index * 100 + 12,
                "originalOutcome": {
                    "phase": "disadvantage" if index % 3 != 1 else "advantage",
                    "damage": index,
                    "killed": False,
                },
            }
            for index in range(18)
        ]

        selected = limit_segments(segments, 8)

        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(item["phase"] == "neutral" for item in selected), 4)
        self.assertEqual(
            [item["timelineIndex"] for item in selected],
            sorted(item["timelineIndex"] for item in selected),
        )


if __name__ == "__main__":
    unittest.main()
