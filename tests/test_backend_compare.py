from __future__ import annotations

import json
from pathlib import Path

from slippi_ai_review.backend_compare import compare_runs, render_markdown


def write_rows(run: Path, rows: list[dict[str, object]]) -> None:
    run.mkdir(parents=True)
    (run / "lanes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def lane(option: str, score: float, *, state_x: float = 1.0) -> dict[str, object]:
    return {
        "baseFrame": 100,
        "offset": 0,
        "takeoverFrame": 105,
        "startState": {"frame": 105, "x": [state_x, 2.0]},
        "comboOptionSignature": option,
        "score": score,
        "damageDealt": 10.0,
        "damageTaken": 2.0,
        "outperformedReplay": score > 0,
        "resolution": {"reason": "timeout"},
    }


def test_compare_runs_separates_injection_parity_from_outcomes(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    decomp = tmp_path / "decomp"
    write_rows(legacy, [lane("BAIR", 20.0), lane("SHIELD", 10.0)])
    write_rows(decomp, [lane("BAIR", -20.0), lane("SHIELD", -10.0)])

    report = compare_runs(legacy, decomp)

    assert report["passed"] is True
    assert report["hardParityRate"] == 1.0
    assert report["optionParityRate"] == 1.0
    assert report["points"][0]["outcomes"]["legacy"]["meanScore"] == 15.0
    assert report["points"][0]["outcomes"]["decomp"]["meanScore"] == -15.0
    assert "Outcome differences are diagnostic only" in render_markdown(report)


def test_compare_runs_fails_when_branch_state_differs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    decomp = tmp_path / "decomp"
    write_rows(legacy, [lane("BAIR", 1.0)])
    write_rows(decomp, [lane("BAIR", 1.0, state_x=9.0)])

    report = compare_runs(legacy, decomp)

    assert report["passed"] is False
    assert report["hardParityRate"] == 0.0
