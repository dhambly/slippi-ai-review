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


def test_space_toggle_is_bridged_from_embedded_viewers() -> None:
    comparison = viewer_source("compare.html")
    frame = viewer_source("frame.html")
    assert 'event.data?.type === "msl-viewer-toggle-playback"' in comparison
    assert 'type: "msl-viewer-toggle-playback"' in frame
    assert "event.preventDefault();" in frame
