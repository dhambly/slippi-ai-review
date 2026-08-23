from __future__ import annotations

from slippi_ai_review.render_target import trace_alignment_frames


def test_trace_switch_precedes_delayed_model_control() -> None:
    target = {"takeover_frame": 237, "branch_start_frame": 216}
    lane = {"takeoverFrame": 216, "modelControlFrame": 237}
    stream = {"takeoverFrame": 216, "modelControlFrame": 237}

    assert trace_alignment_frames(target, lane, stream) == (216, 237)
