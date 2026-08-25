from __future__ import annotations

from pathlib import Path

from slippi_ai_review.nightly import discover_replays, target_player
from slippi_ai_review.nightly_report import Evidence, _phillip_action, build_html, build_patterns


def test_discover_replays_uses_newest_normal_folder_and_date(tmp_path: Path) -> None:
    old = tmp_path / "2026-07"
    current = tmp_path / "2026-08"
    mainline = tmp_path / "2026-08-Mainline"
    for folder in (old, current, mainline):
        folder.mkdir()
    (old / "Game_20260731T230000.slp").write_bytes(b"old")
    (current / "Game_20260823T230000.slp").write_bytes(b"yesterday")
    latest = current / "Game_20260824T230000.slp"
    latest.write_bytes(b"today")
    (mainline / "Game_20260825T230000.slp").write_bytes(b"ignore")

    folder, date, paths = discover_replays(tmp_path, None)

    assert folder == current
    assert date == "2026-08-24"
    assert paths == [latest]


def test_target_player_requires_one_alias_match() -> None:
    details = {
        "players": [
            {"playerIndex": 0, "name": "bes", "connectCode": "MOOB#964"},
            {"playerIndex": 1, "name": "opponent", "connectCode": "TEST#123"},
        ]
    }
    assert target_player(details, {"bes", "m#0085"}) == details["players"][0]
    assert target_player(details, {"missing"}) is None


def evidence(review_id: str, phase: str = "neutral") -> Evidence:
    return Evidence(
        review_id=review_id,
        target_index=1,
        phase=phase,
        frame=120,
        elapsed="0:02.00",
        game_name=f"{review_id}.slp",
        matchup="Fox vs Marth",
        stage="Battlefield",
        replay_action="nair",
        opponent_action="dtilt",
        phillip_action="dash movement",
        original_result="got opened by dtilt for 12%",
        phillip_result="chose dash movement, dealt 8% and took 0%",
        improvement=16,
        option_samples=4,
        sweep_samples=12,
        option_share=0.33,
        reversal_rate=0,
        self_death_rate=0,
        source_url=f"/review-artifacts/{review_id}/neutral_review.html#slide-1",
        viewer_url=None,
    )


def test_patterns_require_more_than_one_game_for_practice_priority() -> None:
    key = "neutral|Fox vs Marth|aerial commitment"
    recurring, watchlist = build_patterns([
        (key, evidence("00000000-0000-4000-8000-000000000001")),
        (key, evidence("00000000-0000-4000-8000-000000000002")),
    ])
    assert len(recurring) == 1
    assert recurring[0].title == "Getting opened during aerial commitments"
    assert not watchlist


def test_movement_pattern_has_human_title() -> None:
    recurring, _watchlist = build_patterns([
        ("neutral|Fox vs Marth|movement", evidence("00000000-0000-4000-8000-000000000001")),
        ("neutral|Fox vs Marth|movement", evidence("00000000-0000-4000-8000-000000000002")),
    ])
    assert recurring[0].title == "Getting opened while moving"


def test_phillip_action_skips_forced_missed_tech_state() -> None:
    target = {
        "representative_lane": {
            "comboOptionSignature": "TECH_MISS_UP",
            "option": {"actionSegments": [
                {"actionName": "TECH_MISS_UP"},
                {"actionName": "GROUND_ROLL_BACKWARD_UP"},
            ]},
        }
    }
    assert _phillip_action(target, "Fox") == "roll backward after missed tech"


def test_report_contains_practice_controls_and_plain_language() -> None:
    pattern, _watchlist = build_patterns([
        ("neutral|Fox vs Marth|aerial commitment", evidence("00000000-0000-4000-8000-000000000001")),
        ("neutral|Fox vs Marth|aerial commitment", evidence("00000000-0000-4000-8000-000000000002")),
    ])
    page = build_html({
        "session": {"date": "2026-08-24", "stats": {"analyzedGames": 2}, "games": []},
        "evidenceCount": 2,
        "recurringPatterns": [pattern[0].__dict__ | {"evidence": [item.__dict__ for item in pattern[0].evidence]}],
    })
    assert "Getting opened during aerial commitments" in page
    assert "Practice this in TMCE" in page
    assert "Repeated patterns only" in page
