from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from slippi_ai_review.report import interactive_url, is_legacy_slp_version, practice_actions_html


def test_slp_replay_warning_uses_3_16_0_cutoff() -> None:
    assert is_legacy_slp_version("1.9.0")
    assert is_legacy_slp_version("2.0.1")
    assert is_legacy_slp_version("3.15.9")
    assert not is_legacy_slp_version("3.16.0")
    assert not is_legacy_slp_version("3.19.1")
    assert not is_legacy_slp_version(None)
    assert not is_legacy_slp_version("unknown")


def test_practice_warning_only_renders_for_legacy_slp() -> None:
    warning = "Older SLP files may not replay correctly"
    assert warning in practice_actions_html(legacy_slp=True)
    assert warning not in practice_actions_html(legacy_slp=False)


def test_practice_actions_include_exact_frame_random_defense() -> None:
    controls = practice_actions_html(legacy_slp=False)
    assert 'data-scenario-mode="variations"' in controls
    assert "Random defense at current frame" in controls
    assert 'data-variation-source="replay"' in controls
    assert 'data-variation-source="rollout"' in controls
    assert "exact frame currently shown in the clip" in controls


def test_interactive_viewer_separates_trace_switch_from_model_takeover() -> None:
    url = interactive_url({
        "replay_trace": "replay.json",
        "agent_trace": "agent.json",
        "switch_frame": 60,
        "model_control_frame": 81,
        "frame_count": 180,
    })
    params = parse_qs(urlparse(url).query)

    assert params["switch"] == ["60"]
    assert params["takeover"] == ["81"]
