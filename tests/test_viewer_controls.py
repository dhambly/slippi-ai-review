from pathlib import Path

from slippi_ai_review.paths import PACKAGE_DIR


def viewer_source(filename: str) -> str:
    return (Path(PACKAGE_DIR) / "web" / "viewer" / filename).read_text(encoding="utf-8")


def test_comparison_viewer_has_move_navigation_controls() -> None:
    source = viewer_source("compare.html")
    assert 'id="previous-move"' in source
    assert 'id="next-move"' in source
    assert "function jumpToMove(direction)" in source
    assert "collectMoveFrames();" in source


def test_comparison_viewer_has_selectable_takeover_marker_without_color_legend() -> None:
    source = viewer_source("compare.html")
    assert 'id="takeover-line"' in source
    assert "function setupTakeoverMarker()" in source
    assert "timeline-key" not in source


def test_comparison_viewer_marks_other_player_takeover() -> None:
    source = viewer_source("compare.html")
    assert 'id="secondary-takeover-line"' in source
    assert "Other player Phillip takeover" in source
    assert 'timeline.style.setProperty("--defender-switch-pct"' in source


def test_space_toggles_playback_after_selecting_a_timeline_marker() -> None:
    source = viewer_source("compare.html")
    assert 'if (event.key === " ")' in source
    assert "setPlaying(!playing);" in source


def test_global_shortcuts_are_bridged_from_embedded_viewers() -> None:
    comparison = viewer_source("compare.html")
    frame = viewer_source("frame.html")
    assert "function runShortcut(rawKey)" in comparison
    assert 'event.data?.type === "msl-viewer-shortcut"' in comparison
    assert 'type: "msl-viewer-shortcut"' in frame
    assert 'key === ","' in comparison
    assert 'key === "." || key === "e"' in comparison
    assert 'percentage = key === "0" ? 100 : Number(key) * 10' in comparison
    assert "event.preventDefault();" in frame
