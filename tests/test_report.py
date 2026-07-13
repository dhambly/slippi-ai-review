from __future__ import annotations

from slippi_ai_review.report import is_legacy_slp_version, practice_actions_html


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
