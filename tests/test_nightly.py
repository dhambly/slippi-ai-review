from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from slippi_ai_review.nightly import _claim_active_run, _existing_review, _pid_running, discover_replays, target_player
from slippi_ai_review.nightly_report import (
    Evidence,
    _option_rows,
    _evidence_card,
    _phillip_action,
    apply_curation,
    build_html,
    build_patterns,
)


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


def test_active_nightly_lock_blocks_a_second_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("slippi_ai_review.nightly._pid_running", lambda pid: pid == 12345)
    lock_dir = tmp_path / ".nightly.lock"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text(
        json.dumps({"runId": "existing", "pid": 12345}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="already active"):
        _claim_active_run(tmp_path, "new")


def test_fresh_ownerless_lock_is_not_stolen(tmp_path: Path) -> None:
    (tmp_path / ".nightly.lock").mkdir()

    with pytest.raises(SystemExit, match="currently acquiring"):
        _claim_active_run(tmp_path, "new")


def test_stale_ownerless_lock_is_reclaimed_and_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_dir = tmp_path / ".nightly.lock"
    lock_dir.mkdir()
    callbacks = []
    modified = lock_dir.stat().st_mtime
    monkeypatch.setattr("slippi_ai_review.nightly.time.time", lambda: modified + 31)
    monkeypatch.setattr("slippi_ai_review.nightly.atexit.register", callbacks.append)

    _claim_active_run(tmp_path, "00000000-0000-4000-8000-000000000001")

    assert (lock_dir / "owner.json").is_file()
    assert (tmp_path / "active_nightly.json").is_file()
    callbacks[0]()
    assert not lock_dir.exists()
    assert not (tmp_path / "active_nightly.json").exists()


def test_stale_lock_with_non_object_json_is_reclaimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_dir = tmp_path / ".nightly.lock"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text("[]", encoding="utf-8")
    callbacks = []
    modified = lock_dir.stat().st_mtime
    monkeypatch.setattr("slippi_ai_review.nightly.time.time", lambda: modified + 31)
    monkeypatch.setattr("slippi_ai_review.nightly.atexit.register", callbacks.append)

    _claim_active_run(tmp_path, "00000000-0000-4000-8000-000000000001")

    assert callbacks
    callbacks[0]()


@pytest.mark.skipif(os.name != "nt", reason="Windows process API behavior")
def test_windows_pid_probe_treats_access_denied_as_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    class Function:
        argtypes = None
        restype = None

        def __init__(self, result: int) -> None:
            self.result = result

        def __call__(self, *_args) -> int:
            return self.result

    class Kernel32:
        OpenProcess = Function(0)
        GetExitCodeProcess = Function(0)
        CloseHandle = Function(1)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: Kernel32())
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)

    assert _pid_running(12345)


def _nightly_candidate(tmp_path: Path, *, status: str, port: int = 1) -> tuple[dict, str, Path]:
    review_id = "00000000-0000-4000-8000-000000000001"
    review_dir = tmp_path / review_id
    review_dir.mkdir()
    (review_dir / "replay.slp").write_bytes(b"replay")
    payload = {
        "reviewId": review_id,
        "status": status,
        "replay": {"sha256": "digest"},
        "settings": {
            "analysisMode": "nightly-sweep",
            "phaseSweepSamples": 4,
            "phaseSweepSegments": 8,
            "simulationBackend": "decomp",
        },
        "targetPlayer": {"port": port},
    }
    (review_dir / "review.json").write_text(json.dumps(payload), encoding="utf-8")
    state = {"replays": {"digest": {"reviewId": review_id}}}
    return state, review_id, review_dir


def _find_existing(tmp_path: Path, state: dict, *, port: int = 1, samples: int = 4):
    return _existing_review(
        tmp_path,
        state,
        "digest",
        {"port": port},
        samples=samples,
        segments_per_game=8,
        simulation_backend="decomp",
    )


def test_failed_nightly_review_resumes_in_place(tmp_path: Path) -> None:
    state, review_id, review_dir = _nightly_candidate(tmp_path, status="failed")

    assert _find_existing(tmp_path, state) == (review_id, review_dir, "resume")


def test_complete_nightly_review_requires_all_phase_artifacts(tmp_path: Path) -> None:
    state, review_id, review_dir = _nightly_candidate(tmp_path, status="complete")
    artifacts = review_dir / "artifacts"
    artifacts.mkdir()
    for phase in ("advantage", "neutral", "disadvantage"):
        (artifacts / f"{phase}_review.html").write_text(phase, encoding="utf-8")
    pipeline = review_dir / "pipeline"
    pipeline.mkdir()
    (pipeline / "phase_sweep_queue.json").write_text(json.dumps({"controlled_port": 1}), encoding="utf-8")

    assert _find_existing(tmp_path, state) == (review_id, review_dir, "complete")


def test_nightly_review_reuse_rejects_wrong_target_or_budget(tmp_path: Path) -> None:
    state, _review_id, _review_dir = _nightly_candidate(tmp_path, status="failed")

    assert _find_existing(tmp_path, state, port=2) is None
    assert _find_existing(tmp_path, state, samples=8) is None


def test_nightly_review_reuse_rejects_stale_hash_state(tmp_path: Path) -> None:
    state, _review_id, review_dir = _nightly_candidate(tmp_path, status="failed")
    payload = json.loads((review_dir / "review.json").read_text(encoding="utf-8"))
    payload["replay"]["sha256"] = "different"
    (review_dir / "review.json").write_text(json.dumps(payload), encoding="utf-8")

    assert _find_existing(tmp_path, state) is None


def test_processing_review_resumes_when_pid_was_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state, review_id, review_dir = _nightly_candidate(tmp_path, status="processing")
    payload = json.loads((review_dir / "review.json").read_text(encoding="utf-8"))
    payload["worker"] = {"pipelinePid": "12345"}
    (review_dir / "review.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("slippi_ai_review.nightly._pid_matches_review_pipeline", lambda _pid, _path: False)

    assert _find_existing(tmp_path, state) == (review_id, review_dir, "resume")


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


def test_disadvantage_card_names_the_opening_move_without_repeating_after() -> None:
    item = evidence("00000000-0000-4000-8000-000000000001", phase="disadvantage").__dict__
    item["original_result"] = "lost the stock"

    card = _evidence_card(item)

    assert "Replay:</b> lost the stock after dtilt." in card
    assert "after defense after hit" not in card


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
        ("neutral|Fox vs Marth|movement against dtilt", evidence("00000000-0000-4000-8000-000000000001")),
        ("neutral|Fox vs Marth|movement against dtilt", evidence("00000000-0000-4000-8000-000000000002")),
    ])
    assert recurring[0].title == "Getting opened by dtilt while moving"


def test_no_action_pattern_is_not_labeled_as_movement() -> None:
    recurring, _watchlist = build_patterns([
        ("neutral|Fox vs Marth|no action against dtilt", evidence("00000000-0000-4000-8000-000000000001")),
        ("neutral|Fox vs Marth|no action against dtilt", evidence("00000000-0000-4000-8000-000000000002")),
    ])
    assert recurring[0].title == "Getting opened by dtilt while standing still"


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


def test_phillip_action_reports_followup_after_forced_opener() -> None:
    target = {
        "representative_lane": {
            "comboOptionSignature": "THROW_UP",
            "option": {"actionSegments": [
                {"actionName": "GRAB_WAIT"},
                {"actionName": "THROW_UP"},
                {"actionName": "KNEE_BEND"},
                {"actionName": "BAIR"},
            ]},
        }
    }
    assert _phillip_action(target, "Fox", "up throw") == "bair"


def test_phillip_action_preserves_pummels_before_throw() -> None:
    target = {
        "representative_lane": {
            "comboOptionSignature": "THROW_UP",
            "option": {"actionSegments": [
                {"actionName": "GRAB_PUMMEL"},
                {"actionName": "THROW_UP"},
                {"actionName": "KNEE_BEND"},
            ]},
        }
    }
    assert _phillip_action(target, "Fox", "up throw") == "extra pummels before up throw"


def test_option_rows_match_combo_family_not_full_action_sequence() -> None:
    rows = [{
        "baseFrame": 120,
        "comboOptionSignature": "NEUTRAL_TECH",
        "optionSignature": "NEUTRAL_TECH>FALLING>DEAD_RIGHT",
    }]
    assert _option_rows(rows, 120, "NEUTRAL_TECH") == rows


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
    assert "qualified comparisons repeated across games" in page


def test_report_uses_singular_more_example_label() -> None:
    pattern, _watchlist = build_patterns([
        ("neutral|Fox vs Marth|aerial commitment", evidence(f"00000000-0000-4000-8000-00000000000{index}"))
        for index in range(1, 5)
    ])
    page = build_html({
        "session": {"date": "2026-08-24", "stats": {"analyzedGames": 4}, "games": []},
        "evidenceCount": 4,
        "recurringPatterns": [pattern[0].__dict__ | {"evidence": [item.__dict__ for item in pattern[0].evidence]}],
    })
    assert "1 more example" in page


def test_curation_reorders_overrides_and_excludes_without_changing_evidence() -> None:
    payload = {
        "recurringPatterns": [
            {"key": "keep-later", "title": "Raw", "score": 20, "evidence": [{"frame": 1}]},
            {"key": "exclude", "title": "Remove", "score": 30, "evidence": []},
            {"key": "keep-first", "title": "Raw first", "score": 10, "evidence": []},
        ]
    }
    curated = apply_curation(payload, {
        "excludeKeys": ["exclude"],
        "overrides": {
            "keep-first": {"title": "Human title", "order": 1},
            "keep-later": {"order": 2},
        },
        "reviewedBy": ["audit"],
    })
    assert [pattern["key"] for pattern in curated["recurringPatterns"]] == ["keep-first", "keep-later"]
    assert curated["recurringPatterns"][0]["title"] == "Human title"
    assert curated["recurringPatterns"][1]["evidence"] == [{"frame": 1}]
    assert curated["curation"]["reviewedBy"] == ["audit"]
