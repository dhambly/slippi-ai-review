from __future__ import annotations

import unittest
from pathlib import Path

from slippi_ai_review.neutral_selection import (
    build_neutral_queue,
    causal_option_name,
    nearest_reliable_boundary,
    option_statistics,
    representative_lane,
    wilson_lower,
)


class NeutralSelectionTests(unittest.TestCase):
    def test_wilson_requires_more_than_raw_threshold(self) -> None:
        self.assertLess(wilson_lower(6, 8), 0.75)
        self.assertGreater(wilson_lower(48, 48), 0.75)

    def test_nearest_reliable_boundary_uses_first_confident_point(self) -> None:
        points = [
            {"lookbackFrames": 10, "samples": 48, "avoidHitRate": 0.8, "baselineReproduced": True},
            {"lookbackFrames": 20, "samples": 48, "avoidHitRate": 1.0, "baselineReproduced": True},
            {"lookbackFrames": 30, "samples": 48, "avoidHitRate": 1.0, "baselineReproduced": True},
        ]
        selected = nearest_reliable_boundary(points, threshold=0.75)
        self.assertEqual(selected["lookbackFrames"], 20)

    def test_option_selection_prefers_frequent_successful_action(self) -> None:
        rows = []
        for lane_id in range(20):
            rows.append(
                {
                    "laneId": lane_id,
                    "comboOptionSignature": "ROLL_FORWARD" if lane_id < 16 else "SPOTDODGE",
                    "optionSignature": "ROLL_FORWARD>SHIELD" if lane_id < 16 else "SPOTDODGE>SHIELD",
                    "neutralContact": {"firstDamageTakenFrame": None},
                    "resolution": {"step": 40},
                }
            )
        options = option_statistics(rows, deadline=100, threshold=0.75, min_samples=2)
        self.assertEqual(options[0]["optionSignature"], "ROLL_FORWARD")
        self.assertTrue(options[0]["qualified"])
        self.assertEqual(representative_lane(options[0]["successfulRows"])["laneId"], 0)

    def test_low_share_option_is_kept_when_its_confidence_bound_passes(self) -> None:
        rows = []
        for lane_id in range(100):
            signature = "NAIR" if lane_id < 91 else "AIRDODGE"
            rows.append({
                "laneId": lane_id,
                "comboOptionSignature": signature,
                "optionSignature": signature,
                "neutralContact": {"firstDamageTakenFrame": None},
            })
        options = option_statistics(rows, deadline=100, threshold=0.75, min_samples=8, min_share=0.10)
        airdodge = next(option for option in options if option["optionSignature"] == "AIRDODGE")
        self.assertLess(airdodge["optionShare"], 0.10)
        self.assertEqual(airdodge["optionShare"], 0.09)
        self.assertTrue(airdodge["qualified"])

    def test_future_attack_does_not_name_neutral_avoidance(self) -> None:
        row = {
            "takeoverFrame": 100,
            "comboOptionSignature": "BAIR",
            "option": {
                "initialActionName": "JUMPING_FORWARD",
                "actionSegments": [
                    {"actionName": "JUMPING_ARIAL_FORWARD", "startStep": 2},
                    {"actionName": "LANDING", "startStep": 20},
                    {"actionName": "BAIR", "startStep": 40},
                ],
            },
        }
        self.assertEqual(causal_option_name(row, decision_frame=110), "AIR_DRIFT")
        self.assertEqual(causal_option_name(row, decision_frame=145), "BAIR")

    def test_queue_keeps_every_qualified_option_as_a_route(self) -> None:
        rows = []
        for lane_id in range(40):
            signature = "ROLL_FORWARD" if lane_id < 24 else "SPOTDODGE"
            rows.append(
                {
                    "laneId": lane_id,
                    "baseFrame": 70,
                    "takeoverFrame": 70,
                    "comboOptionSignature": signature,
                    "optionSignature": signature,
                    "option": {
                        "initialActionName": "STANDING",
                        "actionSegments": [{"actionName": signature, "startStep": 0}],
                    },
                    "neutralContact": {"firstDamageTakenFrame": None},
                    "resolution": {"step": 20},
                    "controllerStream": {
                        "path": "stream.npz",
                        "takeoverFrame": 70,
                        "frames": 61,
                        "endFrame": 130,
                    },
                }
            )
        experiment = {
            "replay": "game.slp",
            "analyzedPort": 1,
            "maxLookbackFrames": 30,
            "postOpeningFrames": 20,
            "events": [{"kind": "avoid"}],
        }
        selected = [{
            "eventId": "avoid_1",
            "kind": "avoid",
            "openingFrame": 100,
            "openingMove": "jab",
            "boundaryPoint": {
                "injectionFrame": 70,
                "deadlineFrame": 120,
                "lookbackFrames": 30,
                "baselineReproduced": True,
            },
        }]
        queue = build_neutral_queue(
            experiment=experiment,
            selected_events=selected,
            refinement_rows=rows,
            refinement_dir=Path("refinement"),
            threshold=0.75,
            min_option_samples=4,
        )
        self.assertEqual(queue["targets"][0]["option"]["optionSignature"], "ROLL_FORWARD")
        self.assertEqual(
            [route["option"]["optionSignature"] for route in queue["targets"][0]["alternative_routes"]],
            ["SPOTDODGE"],
        )

    def test_queue_exposes_best_route_for_each_grid_lead_time(self) -> None:
        def row(frame: int, lane_id: int, action: str) -> dict:
            return {
                "laneId": lane_id, "baseFrame": frame, "takeoverFrame": frame,
                "comboOptionSignature": action, "optionSignature": action,
                "option": {"initialActionName": "STANDING", "actionSegments": [{"actionName": action, "startStep": 0}]},
                "neutralContact": {"firstDamageTakenFrame": None}, "resolution": {"step": 40, "reason": "stable_neutral"},
                "controllerStream": {"path": "stream.npz", "takeoverFrame": frame, "frames": 121, "endFrame": frame + 120},
            }
        selected = [{
            "eventId": "avoid_1", "kind": "avoid", "openingFrame": 100, "openingMove": "jab",
            "boundaryPoint": {"injectionFrame": 70, "deadlineFrame": 120, "lookbackFrames": 30, "baselineReproduced": True},
        }]
        rows = [row(70, index, "ROLL_FORWARD") for index in range(16)]
        grid_rows = [
            *(row(90, index, "AIRDODGE") for index in range(16)),
            *(row(80, index + 16, "KNEE_BEND") for index in range(16)),
            *(row(70, index + 32, "DASHING") for index in range(16)),
        ]
        for item in grid_rows[:16]:
            item["neutralContact"] = {"firstDamageDealtFrame": 110, "firstDamageTakenFrame": None}
        queue = build_neutral_queue(
            experiment={"replay": "game.slp", "analyzedPort": 1, "maxLookbackFrames": 30, "postOpeningFrames": 20, "events": [{"kind": "avoid"}]},
            selected_events=selected, refinement_rows=rows, route_grid_rows=grid_rows,
            refinement_dir=Path("refinement"), threshold=.75, min_option_samples=4,
        )
        target = queue["targets"][0]
        self.assertEqual(target["neutral_loss"]["availableLookbackFrames"], [10, 20, 30])
        self.assertEqual(target["option"]["optionSignature"], "AIRDODGE")
        self.assertTrue(target["option"]["emergency"])
        self.assertEqual(target["alternative_routes"][0]["route_kind"], "win")
        self.assertEqual(target["alternative_routes"][0]["option"]["neutralWinRate"], 1.0)
        self.assertEqual(target["alternative_routes"][2]["option"]["optionSignature"], "GROUND_MOVEMENT")


if __name__ == "__main__":
    unittest.main()
