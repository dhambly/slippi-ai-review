"""Focused coverage tests for comprehensive advantage candidate selection."""

from __future__ import annotations

import unittest

from slippi_ai_review.candidates import build_candidate_payload


SETTINGS = {
    "stageId": 2,
    "players": [
        {"playerIndex": 0, "port": 1, "displayName": "bes", "characterId": 2},
        {"playerIndex": 1, "port": 2, "displayName": "opponent", "characterId": 19},
    ],
}


def snapshot(*, x: float = 0.0, y: float = 0.0, action: int = 14) -> dict:
    return {
        "positionX": x,
        "positionY": y,
        "actionStateId": action,
        "actionStateName": f"state {action}",
    }


def move(
    frame: int,
    name: str = "nair",
    damage: float = 10.0,
    *,
    action_start: int | None = None,
    defender_x: float = 0.0,
    defender_y: float = 0.0,
) -> dict:
    return {
        "frame": frame,
        "playerIndex": 0,
        "moveId": 13,
        "moveName": name,
        "moveShortName": name,
        "hitCount": 1,
        "damage": damage,
        "actionStartFrame": frame if action_start is None else action_start,
        "attacker": snapshot(x=0.0, y=0.0, action=65),
        "attackerBefore": snapshot(x=0.0, y=0.0, action=65),
        "defender": snapshot(x=defender_x, y=defender_y, action=88),
        "defenderBefore": snapshot(x=defender_x, y=defender_y, action=88),
    }


def conversion(
    index: int,
    start: int,
    moves: list[dict],
    *,
    attacker: int = 0,
    defender: int = 1,
    did_kill: bool = False,
    opening_type: str = "neutral-win",
    end: int | None = None,
    defender_runs: list[dict] | None = None,
) -> dict:
    total_damage = sum(float(item["damage"]) for item in moves)
    return {
        "conversionIndex": index,
        "startFrame": start,
        "endFrame": end if end is not None else moves[-1]["frame"] + 30,
        "lastHitBy": attacker,
        "playerIndex": defender,
        "openingType": opening_type,
        "startPercent": 0.0,
        "endPercent": total_damage,
        "didKill": did_kill,
        "moves": moves,
        "attackerStart": snapshot(),
        "attackerEnd": snapshot(),
        "defenderStart": moves[0]["defenderBefore"],
        "defenderEnd": moves[-1]["defender"],
        "attackerActionRuns": [],
        "defenderActionRuns": defender_runs or [],
    }


def presentation_segment(item: dict) -> dict:
    moves = item["moves"]
    return {
        "phase": "advantage",
        "owner": item["lastHitBy"],
        "defender": item["playerIndex"],
        "startFrame": item["startFrame"],
        "endFrame": item["endFrame"],
        "conversion": {
            "damage": item["endPercent"] - item["startPercent"],
            "didKill": item["didKill"],
            "moveSequence": moves,
        },
    }


def fixture() -> tuple[dict, dict]:
    minor = conversion(0, 100, [move(100, damage=4.5)], end=140)
    multi_hit_kill = conversion(
        1,
        200,
        [move(200, "uair"), move(240, "usmash", 18.0, action_start=230)],
        did_kill=True,
        end=280,
    )
    one_hit_kill = conversion(2, 300, [move(300, "usmash", 20.0)], did_kill=True, end=330)
    opponent = conversion(3, 400, [move(400)], attacker=1, defender=0, end=430)
    edgeguard = conversion(
        4,
        9794,
        [
            move(9794, "nair", 12.0, defender_x=-64.0, defender_y=-4.0),
            move(9985, "dtilt", 30.0, action_start=9980, defender_x=-62.0, defender_y=0.0),
        ],
        end=10272,
        defender_runs=[
            {"startFrame": 9900, "endFrame": 9930, "actionStateId": 252, "actionStateName": "ledge catch"},
            {"startFrame": 9931, "endFrame": 9960, "actionStateId": 253, "actionStateName": "ledge hang"},
        ],
    )
    raw = {
        "replay": "synthetic.slp",
        "settings": SETTINGS,
        "conversions": [minor, multi_hit_kill, one_hit_kill, opponent, edgeguard],
    }
    # Deliberately omit the minor conversion and merge only the edgeguard opener
    # into presentation data. Raw conversions must remain the coverage universe.
    timeline = {
        "replay": "synthetic.slp",
        "settings": SETTINGS,
        "segments": [presentation_segment(multi_hit_kill), presentation_segment(edgeguard)],
    }
    return timeline, raw


class CandidateCoverageTests(unittest.TestCase):
    def payload(self, *, max_candidates: int = 0) -> dict:
        timeline, raw = fixture()
        return build_candidate_payload(
            timeline=timeline,
            raw_events=raw,
            analyzed_port=1,
            min_damage=1.0,
            dedupe_frames=3,
            max_candidates=max_candidates,
        )

    def test_raw_slippi_is_complete_advantage_universe(self) -> None:
        payload = self.payload()
        coverage = payload["coverage"]
        self.assertTrue(coverage["raw_slippi_conversions_available"])
        self.assertEqual(4, coverage["total_advantage_sequences"])
        self.assertEqual(3, coverage["eligible_sequences"])
        self.assertEqual(3, coverage["covered_eligible_sequences"])
        self.assertEqual([], coverage["uncovered_sequences"])
        self.assertEqual({"one_hit_kill": 1}, coverage["excluded_sequence_reasons"])

    def test_minor_and_multi_hit_kill_sequences_are_recovered(self) -> None:
        payload = self.payload()
        coverage = payload["coverage"]
        missed = {item["start_frame"]: item["reason"] for item in coverage["previously_missed_sequences"]}
        self.assertEqual("minor_conversion_filtered_from_phase_timeline", missed[100])
        self.assertEqual("multi_hit_kill_route_excluded", missed[200])
        candidate_frames = {item["frame"] for item in payload["frames"]}
        self.assertIn(100, candidate_frames)
        self.assertIn(200, candidate_frames)
        self.assertIn(230, candidate_frames)

    def test_f9794_edgeguard_reopening_gets_multiple_branch_points(self) -> None:
        payload = self.payload()
        edgeguard = [item for item in payload["frames"] if item["segment_start_frame"] == 9794]
        frames = {item["frame"] for item in edgeguard}
        self.assertTrue({9794, 9980, 9985}.issubset(frames))
        self.assertTrue(all("edgeguard_recovery" in item["categories"] for item in edgeguard))
        self.assertTrue(
            all("Landing onstage does not end" in item["resolution_requirements"]["edgeguard"] for item in edgeguard)
        )

    def test_terminal_kill_hit_is_not_a_post_hit_branch(self) -> None:
        payload = self.payload()
        exclusions = payload["coverage"]["excluded_injection_points"]
        self.assertTrue(
            any(item["frame"] == 240 and item["reason"] == "terminal_kill_hit_already_connected" for item in exclusions)
        )

    def test_candidate_cap_preserves_one_branch_per_sequence_when_large_enough(self) -> None:
        payload = self.payload(max_candidates=3)
        self.assertEqual(3, len(payload["frames"]))
        self.assertEqual(3, payload["coverage"]["covered_eligible_sequences"])


if __name__ == "__main__":
    unittest.main()
