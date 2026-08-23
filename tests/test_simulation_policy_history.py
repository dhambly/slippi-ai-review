from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np

from slippi_ai_review import simulation


def test_normal_cli_defaults_to_dolphin_takeover_semantics() -> None:
    with mock.patch(
        "sys.argv",
        ["simulation.py", "--replay", "game.slp", "--analyzed-port", "1"],
    ):
        args = simulation.parse_args()

    assert args.replay_bridge_policy_delay is False


def test_policy_history_uses_parser_compatible_stream() -> None:
    buffers = SimpleNamespace(
        policy_controller_t=np.arange(12).reshape(4, 3),
        prev_input_t=np.full((4, 3), -1),
    )
    records = np.asarray([1, 3])
    expected = object()

    with mock.patch.object(
        simulation,
        "_controllers_from_policy_input",
        return_value=expected,
    ) as decode_policy, mock.patch.object(simulation, "_controllers_from_input") as decode_physical:
        actual = simulation._replay_policy_controllers(buffers, records, num_players=2)

    assert actual is expected
    np.testing.assert_array_equal(decode_policy.call_args.args[0], buffers.policy_controller_t[records])
    decode_physical.assert_not_called()


def test_old_buffer_fallback_uses_previous_input_not_next_input() -> None:
    buffers = SimpleNamespace(
        policy_controller_t=None,
        prev_input_t=np.arange(12).reshape(4, 3),
        input_t=np.full((4, 3), 99),
    )
    records = np.asarray([0, 2])
    expected = object()

    with mock.patch.object(
        simulation,
        "_controllers_from_input",
        return_value=expected,
    ) as decode_physical:
        actual = simulation._replay_policy_controllers(buffers, records, num_players=2)

    assert actual is expected
    np.testing.assert_array_equal(decode_physical.call_args.args[0], buffers.prev_input_t[records])


def test_policy_delay_bridge_is_available_for_diagnostics() -> None:
    args = SimpleNamespace(
        history_mode="teacher-forced",
        analyzed_mode="model",
        replay_bridge_policy_delay=True,
    )
    agent = SimpleNamespace(delay=21)

    assert simulation._replay_bridge_frames(args, agent) == 21


def test_normal_takeover_does_not_postpone_physical_control() -> None:
    args = SimpleNamespace(
        history_mode="teacher-forced",
        analyzed_mode="model",
        replay_bridge_policy_delay=False,
    )
    agent = SimpleNamespace(delay=21)

    assert simulation._replay_bridge_frames(args, agent) == 0


def test_policy_target_records_look_ahead_by_model_delay() -> None:
    records = np.asarray([10, 98, 99])

    actual = simulation._policy_target_records(records, delay=21, num_records=100)

    np.testing.assert_array_equal(actual, np.asarray([31, 99, 99]))


def test_teacher_forcing_stops_one_delay_window_before_takeover() -> None:
    # Outputs generated for f277 and later belong to Phillip even though the
    # observations that generated them occurred 21 frames earlier.
    actual = simulation._policy_force_mask(
        target_frames=np.asarray([276, 277, 298]),
        analyzed_boundary_frames=np.asarray([277, 277, 277]),
        defender_boundary_frames=np.asarray([398, 398, 398]),
        analyzed_idx=0,
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray([True, False, False, True, True, True]),
    )


def test_policy_force_mask_respects_analyzed_port_ordering() -> None:
    actual = simulation._policy_force_mask(
        target_frames=np.asarray([276, 277]),
        analyzed_boundary_frames=np.asarray([277, 277]),
        defender_boundary_frames=np.asarray([276, 277]),
        analyzed_idx=1,
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray([False, False, True, False]),
    )


def test_observed_followup_switches_after_recorded_opener_and_live_hit() -> None:
    actual = simulation._observed_followup_takeover_mask(
        hit_event=np.asarray([True, True, False]),
        observed_hit_counts=np.asarray([2, 1, 2]),
        takeover_steps=np.asarray([121, 121, 121]),
        step=48,
        resolved_mask=np.asarray([False, False, False]),
    )

    np.testing.assert_array_equal(actual, np.asarray([True, False, False]))


def test_observed_followup_does_not_move_an_earlier_takeover() -> None:
    actual = simulation._observed_followup_takeover_mask(
        hit_event=np.asarray([True]),
        observed_hit_counts=np.asarray([2]),
        takeover_steps=np.asarray([40]),
        step=48,
        resolved_mask=np.asarray([False]),
    )

    np.testing.assert_array_equal(actual, np.asarray([False]))
