from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slippi_ai_review.neutral_report import build_page, interactive_url


class NeutralReportTests(unittest.TestCase):
    def test_interactive_url_separates_trace_switch_from_model_control(self) -> None:
        url = interactive_url({
            "replay_trace": "../traces/replay.json",
            "agent_trace": "../traces/agent.json",
            "switch_frame": 60,
            "model_control_frame": 81,
            "frame_count": 100,
        })

        self.assertIn("switch=60", url)
        self.assertIn("takeover=81", url)

    def test_report_contains_comparison_and_all_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.json"
            queue_path.write_text(
                json.dumps({
                    "replay": "game.slp",
                    "display_name": "Uploaded Game",
                    "controlled_port": 1,
                    "targets": [{
                        "base_frame": 70,
                        "option": {"optionSignature": "ROLL_FORWARD", "avoidRate": 1, "successfulPolicyShare": .8, "emergency": True},
                        "representative_lane": {"neutralContact": {"firstDamageDealtFrame": None}},
                        "neutral_loss": {
                            "openingFrame": 100,
                            "openingMove": "jab",
                            "injectionFrame": 70,
                            "lookbackFrames": 30,
                            "deadlineFrame": 130,
                            "avoidRate": 1,
                            "avoidWilsonLower": .92,
                            "samples": 96,
                        },
                    }],
                }),
                encoding="utf-8",
            )
            route = {
                "replay_trace": "../traces/replay.json",
                "agent_trace": "../traces/agent.json",
                "switch_frame": 60,
                "frame_count": 100,
                "opening_display_name": "Jab",
                "option": {"optionSignature": "ROLL_FORWARD", "avoidRate": 1, "successfulPolicyShare": .8},
                "alternative_routes": [{
                    "replay_trace": "../traces/replay2.json",
                    "agent_trace": "../traces/agent2.json",
                    "switch_frame": 60,
                    "frame_count": 100,
                    "option": {"optionSignature": "SPOTDODGE", "avoidRate": .9, "successfulPolicyShare": .2},
                }],
            }
            page = build_page({"queue_json": str(queue_path), "results": [{"target_index": 1, "interactive": route}]})
            self.assertIn("Neutral losses", page)
            self.assertIn("Options before Jab", page)
            self.assertIn("Roll Forward", page)
            self.assertIn("Spotdodge", page)
            self.assertIn("30f lead", page)
            self.assertIn("emergency escape", page)
            self.assertIn("advantage_review.html", page)
            self.assertIn("Uploaded Game", page)
            self.assertIn('href="/">Dashboard</a>', page)


if __name__ == "__main__":
    unittest.main()
